import json
import pytest
from loom.experiments import StudySpec,create_study,get_study,summarize
from loom.experiment_assistant import AdviceRequest,prepare_advice,Decision,decide
from loom.store import dump
from tests.test_workspace import setup_study


def baseline(tmp_path):
    store,old,root=setup_study(tmp_path)
    path=store.root/'trainers/fixture.json';a=json.loads(path.read_text())
    a['parameters']={'reward':[1,2],'epochs':[2,4],'days':[10,26]}
    a['parameter_groups']={'reward':'reward','epochs':'hyperparameter','days':'protocol'}
    path.write_text(json.dumps(a))
    s=StudySpec.model_validate({**old['spec'],'review_mode':True,'max_trials':1,'parameters':{'reward':[1],'epochs':[2],'days':[10]},'evaluation_metrics':['loss','constraint']})
    b=create_study(store,s)
    store.execute("UPDATE studies SET status='completed' WHERE id=?",(b['id'],))
    store.execute("INSERT INTO trials VALUES (?,?,?,?,?,?,?,?)",('trial_base',b['id'],0,dump({'reward':1,'epochs':2,'days':10}),'completed',dump({'objective_mean':1}),None,'2026-09-19',))
    return store,get_study(store,b['id'])


def child_spec(b,scope='reward',**changes):
    return StudySpec.model_validate({**b['spec'],'parent_study_id':b['id'],'adjustment_scope':scope,**changes})


def test_change_scope_and_protocol_are_enforced(tmp_path):
    store,b=baseline(tmp_path)
    child=create_study(store,child_spec(b,parameters={'reward':[2],'epochs':[2],'days':[10]}))
    assert child['status']=='draft' and not child['trials']
    for params in ({'reward':[2],'epochs':[4],'days':[10]},{'reward':[2],'epochs':[2],'days':[26]}):
        with pytest.raises(ValueError,match='参数组'):
            create_study(store,child_spec(b,parameters=params))
    with pytest.raises(ValueError,match='评价协议'):
        create_study(store,child_spec(b,seeds=[999]))
    with pytest.raises(ValueError,match='参数组'):
        create_study(store,child_spec(b,'repeat',parameters={'reward':[2],'epochs':[2],'days':[10]}))


def test_advice_and_human_review_share_same_scope(tmp_path):
    store,b=baseline(tmp_path)
    req=prepare_advice(store,AdviceRequest(study_id=b['id'],adjustment_scope='reward'))
    assert req['allowed_parameters']==['reward']
    rid=store.create_run('experiment_advice',req)
    store.execute("UPDATE runs SET status='completed',result=? WHERE id=?",(dump({'proposed_parameters':{'reward':2,'epochs':4,'days':10}}),rid))
    with pytest.raises(ValueError,match='参数组'):
        decide(store,rid,Decision(decision='accept',note='试验'))
    assert not store.query('SELECT * FROM experiment_decisions')


def test_lower_and_upper_constraints_and_multi_metric_stats():
    spec={'objective':'energy','constraints':{},'metric_rules':[{'metric':'temperature','operator':'le','value':80},{'metric':'completed','operator':'ge','value':100}]}
    result=summarize(spec,[{'seed':1,'metrics':{'energy':10,'temperature':75,'completed':100}}, {'seed':2,'metrics':{'energy':12,'temperature':79,'completed':99}}])
    assert not result['feasible'] and result['metrics_mean']['energy']==11
    assert result['metrics_std']['energy']>0


def test_optional_objective_real_execution(tmp_path):
    from tests.test_workspace import drive
    from loom.experiments import start_study
    store,old,_=setup_study(tmp_path)
    spec=StudySpec.model_validate({**old['spec'],'review_mode':True,'max_trials':1,'objective':None,
                                  'evaluation_metrics':['loss','constraint'],'parameters':{'x':[1]}})
    s=create_study(store,spec);start_study(store,s['id']);result=drive(store,s['id'])
    assert result['status']=='completed'
    assert result['trials'][0]['metrics']['objective_mean'] is None
    assert result['trials'][0]['metrics']['repeat_count']==2
