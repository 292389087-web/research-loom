import json
import pytest
from loom.experiment_assistant import AdviceRequest, Decision, prepare_advice, decide, notebook
from loom.experiments import StudySpec, create_study, start_study, get_study
from loom.store import Store, dump, IdempotencyConflict
from loom.routing import ModelRouter
from loom.config import Settings
from tests.test_workspace import setup_study, drive
from tests.test_system import system


def completed_round(tmp_path):
    store, old, _ = setup_study(tmp_path)
    spec = StudySpec.model_validate({**old['spec'], 'parameters': {'x': [1]}, 'max_trials': 1, 'review_mode': True})
    study = create_study(store, spec)
    start_study(store, study['id'])
    study = drive(store, study['id'])
    return store, study


def saved_advice(store, study):
    request = prepare_advice(store, AdviceRequest(study_id=study['id']))
    rid = store.create_run('experiment_advice', request)
    store.execute("UPDATE runs SET status='completed',result=? WHERE id=?", (dump({'proposed_parameters': {'x': 2}}), rid))
    return rid


def test_one_round_waits_and_review_creates_only_draft(tmp_path):
    store, study = completed_round(tmp_path)
    assert len(study['trials']) == 1
    assert study['status'] == 'completed'
    rid = saved_advice(store, study)
    decision = Decision(decision='accept', note='验证 x=2 的效果')
    record = decide(store, rid, decision)
    child = get_study(store, record['child_id'])
    assert child['status'] == 'draft' and not child['trials']
    assert child['spec']['parent_study_id'] == study['id']
    assert child['spec']['parameters'] == {'x': [2]}
    assert decide(store, rid, decision)['child_id'] == child['id']
    with pytest.raises(IdempotencyConflict):
        decide(store, rid, Decision(decision='reject', note='不能覆盖'))
    book = notebook(store, child['id'])
    assert book['rounds'][-1]['parameter_changes']['x'] == {'before': 1, 'after': 2}
    assert len(book['rounds']) == 2
    assert book['rounds'][0]['comparable_protocol']
    assert book['rounds'][0]['source_hashes'] == study['manifest']['source_hashes']


def test_fixed_trainer_advice_does_not_mark_all_versions_as_unfixed(tmp_path):
    store, study = completed_round(tmp_path)
    spec = {**study['spec'], 'trainer_id':'adaptive-reward-fixture'}
    store.execute('UPDATE studies SET spec=? WHERE id=?', (dump(spec), study['id']))
    request = prepare_advice(store, AdviceRequest(study_id=study['id']))
    limits = ' '.join(request['known_limitations'])
    assert '已有带源码哈希的修复记录' in limits
    assert '尚有待修复的问题' not in limits
    assert '不是所有数据集的固定事实' in limits


def test_rejection_and_out_of_bounds_never_start_training(tmp_path):
    store, study = completed_round(tmp_path)
    rid = saved_advice(store, study)
    with pytest.raises(ValueError, match='边界'):
        decide(store, rid, Decision(decision='modify', note='非法', parameters={'x': 100}))
    before = len(store.query('SELECT id FROM studies'))
    assert decide(store, rid, Decision(decision='reject', note='证据不足'))['child_id'] is None
    assert len(store.query('SELECT id FROM studies')) == before


def test_round_rejects_auto_search_and_privacy_checks_ancestors(tmp_path):
    store, study = completed_round(tmp_path)
    with pytest.raises(ValueError, match='一组'):
        StudySpec.model_validate({**study['spec'], 'parameters': {'x': [1,2]}})
    rid = saved_advice(store, study)
    child_id = decide(store, rid, Decision(decision='accept', note='check'))['child_id']
    child = get_study(store, child_id)
    store.execute('UPDATE studies SET spec=? WHERE id=?', (dump({**child['spec'], 'allow_remote': True}), child_id))
    router = ModelRouter(store, Settings(data_dir=store.root), object())
    assert not router.permitted({'study_id': child_id, 'allow_remote': True}, False)


def test_advice_uses_leased_model_ledger_and_does_not_execute(system, tmp_path):
    client, store, worker, model = system
    _, study = completed_round(tmp_path / 'source')
    (store.root / 'trainers').mkdir(exist_ok=True)
    (store.root / 'trainers' / 'fixture.json').write_text(dump(study['manifest']['adapter']), encoding='utf8')
    local = create_study(store, StudySpec.model_validate(study['spec']))
    start_study(store, local['id'])
    local = drive(store, local['id'])
    model.responses = [{'summary':'单参数验证', 'proposal': {
        'observation':'已有基线', 'evidence_trial_ids':[local['trials'][0]['id']],
        'hypothesis':'x=2 可能降低损失', 'changes':{'x':2}, 'expected_effect':'loss 下降',
        'verification':'使用相同 seeds 比较', 'risks':['单轮不足以证明泛化']}, 'uncertainties':['工程替身，不是论文结论']}]
    response = client.post('/api/experiment-advice', json={'study_id':local['id']})
    assert response.status_code == 202
    rid = response.json()['run_id']
    worker.execute(rid)
    run = store.run(rid)
    assert run['status'] == 'completed', run['error']
    assert run['result']['proposed_parameters'] == {'x':2}
    assert store.query('SELECT id FROM model_attempts WHERE run_id=?', (rid,))
    assert not store.query('SELECT * FROM experiment_decisions')
    result = client.post(f'/api/experiment-advice/{rid}/decision', json={'decision':'modify','parameters':{'x':1},'note':'先复跑确认'} )
    assert result.status_code == 200, result.text
    child = get_study(store, result.json()['child_id'])
    assert child['status'] == 'draft'
    assert client.get(f"/api/studies/{child['id']}/notebook").status_code == 200
