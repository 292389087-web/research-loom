import json

import pytest

from loom.evaluation import evaluate, load_jsonl


def test_semantic_metrics_include_missing_predictions_and_human_review():
    gold = [
        {"id": "q1", "gold_evidence_ids": ["e1", "e2"], "claims": [
            {"claim_id": "c1", "label": "supported", "evidence_ids": ["e1"]},
            {"claim_id": "c2", "label": "insufficient", "evidence_ids": []},
        ]},
        {"id": "q2", "gold_evidence_ids": ["e3"], "claims": [
            {"claim_id": "c3", "label": "unsupported", "evidence_ids": ["e3"]},
        ]},
    ]
    predictions = [{"id": "q1", "retrieved_evidence_ids": ["e2", "e1"], "claims": [
        {"claim_id": "c1", "label": "supported", "evidence_ids": ["e1"], "human_verdicts": ["accepted", "accepted"]},
        {"claim_id": "c2", "label": "supported", "evidence_ids": ["invented"], "human_verdicts": ["needs_revision", "needs_revision"]},
    ]}]
    report = evaluate(gold, predictions)
    assert report["counts"]["missing_task_predictions"] == 1
    assert report["retrieval"]["recall_at_5"] == 0.5
    assert report["claims"]["unsupported_claim_rate"] == 0.5
    assert report["citations"]["precision"] == 0.5
    assert report["human_review"]["acceptance_rate"] == 0.5
    assert report["human_review"]["cohen_kappa"] == 1.0


def test_jsonl_and_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps({"id": "same"}) + "\n" + json.dumps({"id": "same"}), encoding="utf-8")
    rows = load_jsonl(path)
    with pytest.raises(ValueError, match="重复 id"):
        evaluate(rows, [])


@pytest.mark.parametrize('field,value', [('gold_evidence_ids', 'e1'), ('gold_evidence_ids', [{}]), ('claims', None), ('claims', ['bad'])])
def test_bad_gold_shapes_are_rejected(field, value):
    with pytest.raises(ValueError):
        evaluate([{'id': 'q', field: value}], [])


def test_empty_claims_are_unmeasured_and_extra_tasks_block_comparison():
    report = evaluate([{'id': 'q'}], [{'id': 'other'}])
    assert report['claims']['macro_f1'] is None
    assert report['retrieval']['mrr'] is None
    assert not report['coverage']['eligible_for_comparison']


def test_duplicate_retrieval_does_not_inflate_ndcg_and_missing_claim_counts():
    report = evaluate([{'id':'q', 'gold_evidence_ids':['e'], 'claims':[{'claim_id':'c', 'label':'supported'}]}],
                      [{'id':'q', 'retrieved_evidence_ids':['e','e','e']}])
    assert report['retrieval']['ndcg_at_5'] == 1
    assert report['claims']['label_accuracy'] == 0


def test_example_cli_marks_report_and_rejects_input_overwrite(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    gold = root/'evals/semantic/gold.example.jsonl'
    pred = root/'evals/semantic/predictions.example.jsonl'
    base = [sys.executable, '-X', 'utf8', str(root/'scripts/evaluate_semantic_quality.py'), '--gold', str(gold), '--predictions', str(pred), '--example']
    output = tmp_path/'example.json'
    run = subprocess.run(base+['--output',str(output)], capture_output=True, encoding='utf-8')
    assert run.returncode == 0, run.stderr
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['report_kind'] == 'example'
    assert len(report['evaluator_sha256']) == 64
    assert '格式示例' in output.with_suffix('.md').read_text(encoding='utf-8')
    assert subprocess.run(base+['--output',str(gold)], capture_output=True).returncode != 0
