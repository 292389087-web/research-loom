import json
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from loom.api import create_app
from loom.config import Settings
from loom.experiments import StudySpec, create_study, get_study, start_study, trainer
from loom.batch_experiments import preview, pause_batch, resume_batch
from loom.lab_worker import LabWorker
from loom.store import dump, IdempotencyConflict
from tests.test_workspace import setup_study, drive


def batch(tmp_path, script=None, **overrides):
    store, old, _ = setup_study(tmp_path, script=script)
    spec = StudySpec.model_validate({**old['spec'], 'objective':None,'batch_mode':'grid',
                                    'failure_policy':'continue',**overrides})
    return store, create_study(store,spec), spec


def test_matrix_cartesian_explicit_and_bounds(tmp_path):
    store, _, spec = batch(tmp_path)
    data={**spec.model_dump(),'parameters':{'temperature':[1,2,3,4],'migration':[.1,.2,.3,.4],'fixed':[6]}}
    adapter={'parameters':data['parameters']}
    p=preview(StudySpec.model_validate(data),adapter)
    assert p['configurations']==16 and p['executions']==32
    assert p['fixed_parameters']=={'fixed':6}
    assert len({dump(r) for r in p['matrix']})==16
    explicit=StudySpec.model_validate({**data,'batch_mode':'explicit','configurations':[p['matrix'][3],p['matrix'][0]]})
    assert preview(explicit,adapter)['matrix']==[p['matrix'][3],p['matrix'][0]]
    for changes in [
        {'parameters':{'x':list(range(11)),'y':list(range(10))}},
        {'parameters':{'x':[1,1]}},
        {'batch_mode':'explicit','configurations':[{'x':1},{'x':1}]},
        {'controller':'api_bounded','controller_max_calls':1,'allow_remote':True},
    ]:
        with pytest.raises(ValueError): StudySpec.model_validate({**spec.model_dump(),**changes})
    with pytest.raises(ValueError,match='登记'):
        preview(StudySpec.model_validate({**spec.model_dump(),'parameters':{'x':[99]}}),trainer(store,'fixture'))


def test_batch_real_processes_independent_results_and_zero_model_calls(tmp_path):
    store, b, _=batch(tmp_path)
    start_study(store,b['id']);result=drive(store,b['id'])
    assert result['status']=='completed'
    assert [t['params'] for t in result['trials']]==b['manifest']['batch']['matrix']
    attempts=[a for t in result['trials'] for a in t['repeats']]
    assert len(attempts)==4 and len({a['id'] for a in attempts})==4
    assert all(a['status']=='completed' for a in attempts)
    for a in attempts:
        folder=store.root/'experiments'/a['id']
        assert (folder/'config.json').is_file() and (folder/'result-manifest.json').is_file()
    assert not store.query('SELECT * FROM model_attempts')


FAIL_SCRIPT="""from loom.training import TrainingContext
c=TrainingContext.from_cli()
if c.config['parameters']['x']==1: raise RuntimeError('injected training failure')
c.finish({'loss':0,'constraint':0})
"""


@pytest.mark.parametrize('policy',['continue','pause'])
def test_failure_policy_retains_failed_attempt_and_continues(tmp_path,policy):
    store,b,_=batch(tmp_path,FAIL_SCRIPT,failure_policy=policy,seeds=[11])
    start_study(store,b['id']);result=drive(store,b['id'])
    if policy=='pause':
        assert result['status']=='paused' and len(result['trials'])==1
        assert result['trials'][0]['repeats'][0]['status']=='failed'
        resume_batch(store,b['id']);result=drive(store,b['id'])
    assert result['status']=='completed_with_failures'
    assert [t['status'] for t in result['trials']]==['failed','completed']
    assert len(store.query('SELECT id FROM trial_attempts'))==2


def test_pause_before_launch_recovery_and_budget(tmp_path):
    store,b,_=batch(tmp_path,seeds=[11])
    start_study(store,b['id']);worker=LabWorker(store);worker.tick()  # persist first trial only
    pause_batch(store,b['id']);worker.tick()
    assert get_study(store,b['id'])['status']=='paused'
    assert not worker.children
    resume_batch(store,b['id'])
    # Fresh dispatcher picks up queued trial without generating another identity.
    result=drive(store,b['id'])
    assert result['status']=='completed' and len(result['trials'])==2
    with pytest.raises(IdempotencyConflict):resume_batch(store,b['id'])
    store,b,_=batch(tmp_path/'expired',seeds=[11]);start_study(store,b['id']);pause_batch(store,b['id']);LabWorker(store).tick()
    store.execute('UPDATE studies SET deadline=? WHERE id=?',(time.time()-1,b['id']))
    with pytest.raises(ValueError,match='期限'):resume_batch(store,b['id'])
    LabWorker(store).tick();assert get_study(store,b['id'])['status']=='budget_exhausted'


def test_confirmation_api_and_04_handoff(tmp_path):
    store,b,spec=batch(tmp_path)
    app=create_app(Settings(data_dir=store.root,use_dense=False,allowed_hosts=('testserver',)),start_worker=False)
    with TestClient(app) as client:
        p=client.post('/api/batches/preview',json=spec.model_dump());assert p.status_code==200
        req={'spec':spec.model_dump(),'confirmation_sha256':p.json()['confirmation_sha256']}
        bad={**req,'spec':{**req['spec'],'seeds':[123]}}
        assert client.post('/api/batches',json=bad).status_code==422
        created=client.post('/api/batches',json=req);assert created.status_code==201,created.text
        sid=created.json()['id'];assert created.json()['status']=='draft'
        assert client.post('/api/studies/'+sid+'/start').status_code==200
        result=drive(store,sid);assert result['status']=='completed'
        advice=client.post('/api/experiment-advice',json={'study_id':sid})
        assert advice.status_code==422 and '不生成调参建议' in advice.text
        ds=client.post('/api/data/from-studies',json={'name':'batch results','study_ids':[sid]}).json()
        assert len({r['label'] for r in ds['rows']})==2
        assert all(r['parameters'] for r in ds['rows'])
        assert client.get('/api/studies/'+sid+'/export?format=json').json()['manifest']['batch']['matrix']


def test_killed_runner_recovered_without_duplicate_execution(tmp_path):
    script="""from loom.training import TrainingContext
import time
c=TrainingContext.from_cli()
if c.config['parameters']['x']==1: time.sleep(30)
c.finish({'loss':0,'constraint':0})
"""
    store,b,_=batch(tmp_path,script,seeds=[11])
    start_study(store,b['id']);worker=LabWorker(store)
    try:
        deadline=time.time()+15
        while time.time()<deadline:
            worker.tick()
            active=store.query("SELECT * FROM trial_attempts WHERE status='running'")
            if active: break
            time.sleep(.05)
        assert active
        first=active[0]['id'];child=worker.children[first]
        child.kill();child.wait(timeout=5)
        recovered=drive(store,b['id'])  # a new dispatcher reconciles lost process
        assert recovered['status']=='completed_with_failures'
        assert recovered['trials'][0]['repeats'][0]['id']==first
        assert 'outcome unknown' in recovered['trials'][0]['repeats'][0]['error']
        assert recovered['trials'][0]['repeats'][0]['execution_state']=='unknown'
        assert recovered['trials'][0]['repeats'][0]['artifact_state']=='not_checked'
        assert len(store.query('SELECT id FROM trial_attempts'))==2
    finally:
        for child in worker.children.values():
            if child.poll() is None:child.kill()
            child.wait(timeout=5)


def test_pause_waits_for_current_seed_then_resume(tmp_path):
    script="""from loom.training import TrainingContext
import time
c=TrainingContext.from_cli();time.sleep(.8);c.finish({'loss':0,'constraint':0})
"""
    store,b,_=batch(tmp_path,script,seeds=[11,23],parameters={'x':[1]})
    start_study(store,b['id']);worker=LabWorker(store)
    try:
        deadline=time.time()+15
        while time.time()<deadline:
            worker.tick()
            if store.query("SELECT id FROM trial_attempts WHERE status='running'"):break
            time.sleep(.05)
        pause_batch(store,b['id'])
        while time.time()<deadline:
            worker.tick();s=get_study(store,b['id'])
            if s['status']=='paused':break
            time.sleep(.05)
        assert s['status']=='paused'
        assert [r['status'] for r in s['trials'][0]['repeats']]==['completed','queued']
        resume_batch(store,b['id']);assert drive(store,b['id'])['status']=='completed'
    finally:
        for child in worker.children.values():
            if child.poll() is None:child.kill()
            child.wait(timeout=5)
