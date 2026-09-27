import json
import pytest
from loom.store import Store
from loom import research_data as rd
from loom.routing import ModelRouter
from loom.config import Settings
from tests.test_system import system


def test_platform_rounds_snapshot_reward_definition_and_permissions(tmp_path):
    from tests.test_experiment_design import baseline, child_spec
    from loom.experiments import create_study
    from loom.store import dump, now
    store, b = baseline(tmp_path)
    child = create_study(store, child_spec(b, parameters={'reward':[2],'epochs':[2],'days':[10]}))
    store.execute('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?)',
                  ('trial_child',child['id'],0,dump({'reward':2,'epochs':2,'days':10}),'completed','{}',None,now()))
    for study, tid, energy, reward in [(b,'trial_base',100,10),(child,'trial_child',95,20)]:
        manifest=study['manifest']
        manifest['adapter']['metric_metadata']={'energy':{'unit':'kWh'},'mean_reward':{'unit':'reward','role':'reward'}}
        store.execute('UPDATE studies SET manifest=? WHERE id=?',(dump(manifest),study['id']))
        store.execute('INSERT INTO trial_attempts(id,trial_id,seed,status,metrics) VALUES (?,?,?,?,?)',
                      (tid+'_ok',tid,11,'completed',dump({'energy':energy,'mean_reward':reward})))
    store.execute('INSERT INTO trial_attempts(id,trial_id,seed,status,error) VALUES (?,?,?,?,?)',
                  ('failed_seed','trial_child',23,'failed','injected crash'))
    ds=rd.from_studies(store,rd.StudySelection(name='rounds',study_ids=[b['id'],child['id']]))
    result=rd.analyze_data(store,ds,rd.DataPlan(metrics=['energy','mean_reward'],baseline=b['id']+':trial_base'))
    energy=next(r for r in result['rows'] if r['group'].startswith(child['id']) and r['metric']=='energy')
    reward=next(r for r in result['rows'] if r['group'].startswith(child['id']) and r['metric']=='mean_reward')
    assert energy['delta']==-5 and energy['n']==1
    assert reward['delta'] is None and len(result['excluded'])==1
    router=ModelRouter(store,Settings(data_dir=tmp_path),object())
    assert not router.permitted({'dataset_id':ds['id'],'allow_remote':True},False)
    store.execute("UPDATE trial_attempts SET metrics=? WHERE id=?",(dump({'energy':999}),'trial_child_ok'))
    assert rd.dataset(store,ds['id'])==ds


def test_demo_snapshot_analysis_chart_and_audit(tmp_path):
    store=Store(tmp_path); ds=rd.demo_dataset(store)
    result=rd.analyze_data(store,ds,rd.DataPlan(metrics=['grid_energy','max_temperature'],baseline='基准'))
    row=next(r for r in result['rows'] if r['group']=='奖励调整' and r['metric']=='grid_energy')
    assert row['mean']==95 and row['std']==1 and row['n']==3 and row['delta']==-5
    assert result['dataset_kind']=='synthetic_demo'
    assert 'synthetic' in rd.export_csv(result) or result['dataset_snapshot'] in rd.export_csv(result)
    assert '<svg' in rd.chart_svg(result,'grid_energy')
    audit=rd.audit_data(store,rd.DataAudit(dataset_id=ds['id'],operation='reduction_percent',
        operands=[rd.Operand(group='基准',metric='grid_energy'),rd.Operand(group='奖励调整',metric='grid_energy')],
        reported='5',statement='电网能耗降低5%'))
    assert audit['passed'] and audit['expected']=='5.00'
    assert rd.report(store,audit['id'])['dataset_snapshot']==ds['snapshot']


@pytest.mark.parametrize('bad',[
    'g,1,m,NaN,kWh,p,evaluation',
    'g,1,m,1,kWh,p,unknown',
    'g,1,m,1,kWh,p,evaluation\ng,1,m,2,kWh,p,evaluation',
    'g,1,m,1,kWh,p,evaluation\nh,1,m,2,Wh,p,evaluation',
    'g,1,m,,kWh,p,evaluation',
])
def test_csv_rejects_ambiguous_or_invalid_data(tmp_path,bad):
    with pytest.raises(ValueError):
        rd.import_csv(Store(tmp_path),rd.CSVImport(name='test',source_note='人工实验数据说明',csv_text='group,seed,metric,value,unit,protocol,phase\n'+bad))


def test_incompatible_protocol_no_delta_no_ranking(tmp_path):
    store=Store(tmp_path)
    ds=rd.import_csv(store,rd.CSVImport(name='test',source_note='人工实验数据说明',csv_text='group,seed,metric,value,unit,protocol,phase\na,1,m,1,kWh,p1,evaluation\nb,1,m,2,kWh,p2,evaluation'))
    report=rd.analyze_data(store,ds,rd.DataPlan(baseline='a'))
    other=next(r for r in report['rows'] if r['group']=='b')
    assert other['delta'] is None and '不可直接比较' in other['comparison']
    with pytest.raises(ValueError,match='口径排名'):
        rd.analyze_data(store,ds,rd.DataPlan(operation='rank',metrics=['m'],sort_metric='m',direction='minimize'))
    with pytest.raises(ValueError,match='单位、评价协议'):
        rd.audit_data(store,rd.DataAudit(dataset_id=ds['id'],operation='reduction_percent',operands=[rd.Operand(group=g,metric='m') for g in ('a','b')],reported='1',statement='降幅1%'))


def test_data_api_natural_plan_uses_ledger_and_clarifies(system):
    client,store,worker,model=system
    ds=client.post('/api/data/demo').json()
    req={'dataset_id':ds['id'],'question':'哪个配置最好？'}
    rid=client.post('/api/data/questions',json=req).json()['run_id'];worker.execute(rid)
    assert store.run(rid)['result']['status']=='needs_clarification'
    assert model.calls==0
    model.responses=[{'operation':'compare','metrics':['grid_energy']}]
    rid=client.post('/api/data/questions',json={**req,'question':'比较电网能耗的均值与标准差','metrics':['grid_energy'],'baseline':'基准'}).json()['run_id']
    worker.execute(rid);run=store.run(rid)
    assert run['status']=='completed',run['error']
    assert len(run['result']['rows'])==3 and model.calls==1
    assert store.query('SELECT * FROM model_attempts WHERE run_id=?',(rid,))
    result_id=run['result']['id']
    assert client.get(f'/api/data/reports/{result_id}/chart?metric=grid_energy').status_code==200
    assert client.get(f'/api/data/reports/{result_id}/export?format=csv').status_code==200


def test_dataset_permission_is_not_overridden_by_task(tmp_path):
    store=Store(tmp_path);ds=rd.demo_dataset(store)
    router=ModelRouter(store,Settings(data_dir=tmp_path),object())
    assert not router.permitted({'dataset_id':ds['id'],'allow_remote':True},False)


def test_audit_binds_immutable_draft_and_rejects_wrong_statement(system):
    client,store,_,_=system;ds=rd.demo_dataset(store)
    response=client.post('/api/drafts',json={'title':'结果','text':'在相同实验协议下，实验结果显示：电网能耗降低5%。本句仅用于核查测试。'})
    assert response.status_code==201, response.text
    draft=response.json()
    req={'dataset_id':ds['id'],'operation':'reduction_percent','operands':[{'group':g,'metric':'grid_energy'} for g in ('基准','奖励调整')],
        'reported':'6','statement':'电网能耗降低5%。','draft_id':draft['id']}
    r=client.post('/api/data/audits',json=req)
    assert r.status_code==200 and not r.json()['passed']
    assert r.json()['draft']['draft_sha']==draft['sha']
    assert client.post('/api/data/audits',json={**req,'statement':'不在初稿中的话'}).status_code==422


def test_csv_export_and_chart_escape_untrusted_labels(tmp_path):
    store=Store(tmp_path)
    ds=rd.import_csv(store,rd.CSVImport(name='test',source_note='手工导入的测试数据',csv_text='group,seed,metric,value,unit,protocol,phase\n=evil<script>,1,m,1,kWh,p,evaluation'))
    result=rd.analyze_data(store,ds,rd.DataPlan())
    assert "'=evil" in rd.export_csv(result)
    assert '<script>' not in rd.chart_svg(result,'m')
