"""Fault injection with synthetic fixtures, never paper performance claims."""
import copy
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient
from loom.api import create_app
from loom.config import Settings
from loom.experiments import StudySpec, create_study, start_study, get_study, trainer
from loom.batch_experiments import preview, submit_batch
from loom.artifact_finalization import revalidate
from loom.preflight import check_preflight
from loom.store import IdempotencyConflict
from loom import research_data as rd
from tests.test_workspace import setup_study, drive
from tests.test_system import system


def update_adapter(store, **changes):
    path=store.root/'trainers'/'fixture.json'
    data=json.loads(path.read_text('utf8'));data.update(changes)
    path.write_text(json.dumps(data),'utf8')
    return data


def spec_for(study, **changes):
    return StudySpec.model_validate({**study['spec'],'objective':None,'batch_mode':'grid',
        'failure_policy':'continue',**changes})


def test_concurrent_submissions_and_replays(tmp_path):
    store,old,root=setup_study(tmp_path)
    spec=spec_for(old); confirmation=preview(spec,trainer(store,'fixture'))['confirmation_sha256']
    before=len(store.query('SELECT id FROM studies'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids=list(pool.map(lambda _:submit_batch(store,spec,confirmation,'shared-request')['id'],range(8)))
    assert len(set(ids))==1
    assert len(store.query('SELECT id FROM studies'))==before+1
    assert len(store.query('SELECT * FROM batch_submissions'))==1
    other=spec.model_copy(update={'name':'new name'})
    other_hash=preview(other,trainer(store,'fixture'))['confirmation_sha256']
    with pytest.raises(IdempotencyConflict):submit_batch(store,other,other_hash,'shared-request')
    assert submit_batch(store,spec,confirmation,'new-request')['id']!=ids[0]
    (root/'train.py').unlink()
    assert submit_batch(store,spec,confirmation,'shared-request')['id']==ids[0]
    with pytest.raises(ValueError):submit_batch(store,spec,'wrong','shared-request')


def test_batch_api_replay_and_conflict(tmp_path):
    store,old,_=setup_study(tmp_path);spec=spec_for(old)
    app=create_app(Settings(data_dir=store.root,use_dense=False,allowed_hosts=('testserver',)),start_worker=False)
    with TestClient(app) as c:
        p=c.post('/api/batches/preview',json=spec.model_dump()).json()
        req={'spec':spec.model_dump(),'confirmation_sha256':p['confirmation_sha256']}
        first=c.post('/api/batches',json=req,headers={'Idempotency-Key':'api-key'})
        second=c.post('/api/batches',json=req,headers={'Idempotency-Key':'api-key'})
        assert first.status_code==second.status_code==201
        assert first.json()['id']==second.json()['id']
        changed=spec.model_copy(update={'name':'changed'})
        p2=c.post('/api/batches/preview',json=changed.model_dump()).json()
        assert c.post('/api/batches',json={'spec':changed.model_dump(),'confirmation_sha256':p2['confirmation_sha256']},headers={'Idempotency-Key':'api-key'}).status_code==409


def test_preflight_dependencies_assets_and_no_entry_execution(tmp_path):
    store,old,root=setup_study(tmp_path,script="raise AssertionError('entrypoint must not execute')")
    assert check_preflight(trainer(store,'fixture'))['passed']
    update_adapter(store,required_modules=['nonexistent_loom_fixture_dependency'])
    check=check_preflight(trainer(store,'fixture'))
    assert not check['passed'] and 'nonexistent_loom_fixture_dependency' in str(check['errors'])
    frozen=create_study(store,spec_for(old))
    with pytest.raises(ValueError):start_study(store,frozen['id'])
    assert not store.query('SELECT * FROM trial_attempts')
    update_adapter(store,required_modules=[],required_assets={'weights':str(root/'missing.bin')})
    assert not check_preflight(trainer(store,'fixture'))['passed']
    with pytest.raises(ValueError):create_study(store,spec_for(old))


def test_preflight_timeout_and_changed_asset(tmp_path):
    store,old,root=setup_study(tmp_path)
    (root/'slow_import.py').write_text('import time; time.sleep(10)','utf8')
    adapter=update_adapter(store,required_modules=['slow_import'])
    assert not check_preflight(adapter,timeout=.2)['passed']
    asset=root/'weights.bin';asset.write_bytes(b'original')
    update_adapter(store,required_modules=[],required_assets={'weights':str(asset)})
    frozen=create_study(store,spec_for(old));asset.write_bytes(b'changed')
    with pytest.raises(ValueError):start_study(store,frozen['id'])
    assert not store.query('SELECT * FROM trial_attempts')


def test_source_changed_after_start_fails_at_execution(tmp_path):
    store,old,root=setup_study(tmp_path)
    study=create_study(store,spec_for(old,seeds=[11],parameters={'x':[1]}))
    start_study(store,study['id'])
    (root/'train.py').write_text("raise AssertionError('reject hash first')",'utf8')
    final=drive(store,study['id'])
    assert final['status']=='failed'
    assert final['trials'][0]['repeats'][0]['execution_state']=='failed'


ARTIFACT_SCRIPT="""from loom.training import TrainingContext
c=TrainingContext.from_cli()
(c.output/'model.bin').write_bytes(b'x'*128)
(c.output/'executed-once.txt').write_text('one execution')
c.finish({'loss':1,'constraint':0})
"""


def failed_artifact(tmp_path):
    store,old,_=setup_study(tmp_path,ARTIFACT_SCRIPT)
    update_adapter(store,artifact_files=['model.bin'],artifact_max_bytes=64)
    study=create_study(store,spec_for(old,seeds=[11],parameters={'x':[1]}))
    start_study(store,study['id']);final=drive(store,study['id'])
    attempt=final['trials'][0]['repeats'][0]
    assert attempt['status']=='failed' and attempt['execution_state']=='completed'
    assert attempt['artifact_state']=='failed' and attempt['execution_receipt']
    return store,final,attempt


def test_revalidation_does_not_retrain_and_keeps_audit(tmp_path):
    store,study,a=failed_artifact(tmp_path)
    folder=store.root/'experiments'/a['id']
    stamp=(folder/'executed-once.txt').stat().st_mtime_ns
    before=(folder/'model.bin').read_bytes()
    result=revalidate(store,a['id'],'Increase limit after verifying checkpoint size',256)
    assert result['passed'] and not result['training_reexecuted']
    final=get_study(store,study['id']);assert final['status']=='completed'
    assert final['trials'][0]['repeats'][0]['artifact_state']=='accepted'
    assert (folder/'executed-once.txt').stat().st_mtime_ns==stamp
    assert (folder/'model.bin').read_bytes()==before
    reviews=store.query('SELECT payload FROM artifact_reviews WHERE attempt_id=?',(a['id'],))
    assert json.loads(reviews[0]['payload'])['old_error']==a['error']
    assert len(store.query('SELECT id FROM trial_attempts'))==1
    with pytest.raises(ValueError):revalidate(store,a['id'],'Repeated acceptance must be rejected',256)


@pytest.mark.parametrize('name',['model.bin','metrics.json','config.json'])
def test_revalidation_rejects_tampering(tmp_path,name):
    store,study,a=failed_artifact(tmp_path)
    (store.root/'experiments'/a['id']/name).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed after training'):
        revalidate(store,a['id'],'Testing immutable execution receipt integrity',256)
    assert get_study(store,study['id'])['trials'][0]['repeats'][0]['status']=='failed'
    assert json.loads(store.query('SELECT payload FROM artifact_reviews')[0]['payload'])['passed'] is False


DAILY_SCRIPT="""from loom.training import TrainingContext, atomic_json
c=TrainingContext.from_cli();x=c.config['parameters']['x'];seed=c.config['seed']
atomic_json(c.output/'daily.json',[{'day':d,'energy':x*10+d+seed/10,'reward':x*d} for d in (1,2,3)])
c.finish({'loss':x*30+6+3*seed/10,'constraint':0})
"""


@pytest.fixture
def daily_dataset(tmp_path):
    store,old,_=setup_study(tmp_path,DAILY_SCRIPT)
    update_adapter(store,evaluation_phase='training',artifact_files=['daily.json'],
        timeseries_artifact='daily.json',timeseries_metadata={'energy':{'unit':'kWh'},'reward':{'unit':'reward','role':'reward'}},
        metric_metadata={'loss':{'unit':'kWh'}},parameter_groups={'x':'reward'})
    study=create_study(store,spec_for(old));start_study(store,study['id'])
    final=drive(store,study['id']);assert final['status']=='completed'
    ds=rd.from_studies(store,rd.StudySelection(name='daily fixture',study_ids=[study['id']]))
    return store,study,ds


def test_daily_filters_lineage_and_independent_arithmetic(daily_dataset):
    store,study,ds=daily_dataset
    report=rd.analyze_data(store,ds,rd.DataPlan(operation='trajectory',metrics=['energy'],
        filters=[{'parameter':'x','values':[1]}],day_start=2,day_end=3))
    assert len(report['rows'])==2
    row=report['rows'][0]
    assert row['day']==2 and row['mean']==pytest.approx(13.7)
    assert row['std']==pytest.approx(math.sqrt(.72)) and row['n']==2
    assert row['phase']=='training' and row['parameters']=={'x':1}
    assert report['dataset_snapshot']==ds['snapshot'] and len(report['lineage'])==2
    assert all(r['artifact_sha256'] for r in report['source_rows'])
    assert ',day,' in rd.export_csv(report) and '<polyline' in rd.chart_svg(report,'energy')
    assert len(rd.analyze_data(store,ds,rd.DataPlan(operation='trajectory',metrics=['energy']))['rows'])==6
    summary=rd.analyze_data(store,ds,rd.DataPlan(metrics=['loss'],filters=[{'parameter':'x','values':[2]}]))
    assert len(summary['rows'])==1 and summary['rows'][0]['mean']==pytest.approx(71.1)


@pytest.mark.parametrize('problem',['unknown_parameter','empty_filter','baseline_filtered','missing_day','missing_seed','reward_definition','cumulative_as_daily'])
def test_daily_semantic_boundaries(daily_dataset,problem):
    store,study,original=daily_dataset;ds=copy.deepcopy(original)
    args={'operation':'trajectory','metrics':['energy']}
    if problem=='unknown_parameter':args['filters']=[{'parameter':'unknown','values':[1]}]
    if problem=='empty_filter':args['filters']=[{'parameter':'x','values':[999]}]
    if problem=='baseline_filtered':
        args.update(filters=[{'parameter':'x','values':[1]}],baseline=next(r['group'] for r in ds['series'] if r['parameters']['x']==2))
    if problem=='missing_day':ds['series']=[r for r in ds['series'] if r['day']!=2]
    if problem=='missing_seed':ds['series']=[r for r in ds['series'] if not(r['day']==2 and r['seed']==23)]
    if problem=='reward_definition':args['metrics']=['reward']
    if problem=='cumulative_as_daily':args['metrics']=['loss']
    with pytest.raises(ValueError):rd.analyze_data(store,ds,rd.DataPlan(**args))


def test_daily_tampering_cannot_rewrite_snapshot(daily_dataset):
    store,study,ds=daily_dataset
    (store.root/'experiments'/ds['series'][0]['source']/'daily.json').write_text('[]','utf8')
    with pytest.raises(ValueError,match='哈希'):
        rd.from_studies(store,rd.StudySelection(name='changed data',study_ids=[study['id']]))
    # Rewriting the on-disk result manifest must not override the DB receipt.
    path=store.root/'experiments'/ds['series'][0]['source']/'result-manifest.json'
    manifest=json.loads(path.read_text('utf8'))
    manifest['artifacts']['daily.json']=hashlib.sha256(b'[]').hexdigest()
    path.write_text(json.dumps(manifest),'utf8')
    with pytest.raises(ValueError,match='凭据'):
        rd.from_studies(store,rd.StudySelection(name='forged manifest',study_ids=[study['id']]))
    assert rd.dataset(store,ds['id'])==ds


def test_plan_rejects_ambiguous_filter_and_range():
    for args in [dict(filters=[{'parameter':'x','values':[1]},{'parameter':'x','values':[2]}]),
                 dict(operation='trajectory',day_start=3,day_end=2),dict(day_start=1),
                 dict(filters=[{'parameter':'x','values':[float('nan')]}])]:
        with pytest.raises(ValueError):rd.DataPlan(**args)


def test_natural_language_daily_filter_contract(daily_dataset,system):
    _,_,data=daily_dataset
    client,store,worker,model=system
    ds=rd.save_dataset(store,'Fixture for planner',data)
    model.responses=[{'operation':'trajectory','metrics':['energy'],
        'filters':[{'parameter':'x','values':[1]}],'day_start':2,'day_end':3}]
    response=client.post('/api/data/questions',json={'dataset_id':ds['id'],
        'question':'Show days two to three for x equal to one.'})
    assert response.status_code==202,response.text
    rid=response.json()['run_id'];worker.execute(rid);run=store.run(rid)
    assert run['status']=='completed',run['error']
    assert len(run['result']['rows'])==2 and model.calls==1
    assert all(r['parameters']['x']==1 for r in run['result']['rows'])
    assert store.query('SELECT * FROM model_attempts WHERE run_id=?',(rid,))


def test_revalidation_api_and_legacy_rejection(tmp_path):
    store,study,a=failed_artifact(tmp_path)
    app=create_app(Settings(data_dir=store.root,use_dense=False,allowed_hosts=('testserver',)),start_worker=False)
    with TestClient(app) as client:
        url='/api/attempts/'+a['id']+'/revalidate'
        assert client.post(url,json={'note':'short','artifact_max_bytes':256}).status_code==422
        assert client.post(url,json={'note':'Valid explanation for acceptance','artifact_max_bytes':2**32}).status_code==422
        original=a['execution_receipt']
        store.execute('UPDATE trial_attempts SET execution_receipt=NULL WHERE id=?',(a['id'],))
        assert client.post(url,json={'note':'Legacy results lack completion receipt','artifact_max_bytes':256}).status_code==422
        store.execute('UPDATE trial_attempts SET execution_receipt=? WHERE id=?',(original,a['id']))
        response=client.post(url,json={'note':'Verified checkpoint was rejected solely for size','artifact_max_bytes':256})
        assert response.status_code==200,response.text
        assert response.json()['training_reexecuted'] is False
