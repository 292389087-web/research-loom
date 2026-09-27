"""冻结人工标注上的确定性语义质量评测，不调用 LLM 充当裁判。"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


LABELS = ("supported", "unsupported", "insufficient")
VERDICTS = ("accepted", "needs_revision")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} 不是有效 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} 必须是 JSON 对象")
        rows.append(value)
    return rows


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity_map(rows: list[dict], source: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{source} 必须由 JSON 对象组成")
        identity = row.get("id")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"{source} 中每条记录都需要非空 id")
        if identity in result:
            raise ValueError(f"{source} 中存在重复 id: {identity}")
        result[identity] = row
    return result


def _ids(value, source):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"{source} 必须是非空字符串数组")
    return list(dict.fromkeys(value))


def _claims(row: dict, source: str) -> dict[str, dict]:
    result = {}
    claims = row.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError(f"{source} 的 claims 必须是数组")
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError(f"{source} 的 claim 必须是对象")
        identity = claim.get("claim_id")
        label = claim.get("label")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"{source} 的 claim 需要非空 claim_id")
        if identity in result:
            raise ValueError(f"{source} 中存在重复 claim_id: {identity}")
        if label not in LABELS:
            raise ValueError(f"{source}/{identity} 的 label 必须属于 {LABELS}")
        evidence_ids = claim.get("evidence_ids", [])
        _ids(evidence_ids, f"{source}/{identity}/evidence_ids")
        verdicts = claim.get("human_verdicts", [])
        if not isinstance(verdicts, list) or len(verdicts) > 2 or any(v not in VERDICTS for v in verdicts):
            raise ValueError(f"{source}/{identity} 的 human_verdicts 只能包含 {VERDICTS}")
        result[identity] = claim
    return result


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _divide(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _ndcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1 / math.log2(rank + 2) for rank, item in enumerate(retrieved[:k]) if item in relevant)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left = Counter(item[0] for item in pairs)
    right = Counter(item[1] for item in pairs)
    expected = sum((left[label] / len(pairs)) * (right[label] / len(pairs)) for label in VERDICTS)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def evaluate(gold_rows: list[dict], prediction_rows: list[dict], retrieval_ks=(1, 5, 10)) -> dict:
    """计算检索、主张标签、引用和人工复核指标。

    gold 与 prediction 必须在评测前冻结。缺失预测按错误处理，额外预测主张按无依据主张处理。
    """
    gold = _identity_map(gold_rows, "gold")
    predictions = _identity_map(prediction_rows, "predictions")
    if not gold:
        raise ValueError("gold 不能为空")
    if not retrieval_ks or any(type(k) is not int or k <= 0 for k in retrieval_ks) or len(set(retrieval_ks)) != len(retrieval_ks):
        raise ValueError("retrieval_ks 必须是互不重复的正整数")
    for task_id, row in predictions.items():
        _claims(row, f"predictions/{task_id}")
        _ids(row.get("retrieved_evidence_ids", []), f"predictions/{task_id}/retrieved_evidence_ids")

    retrieval = {k: [] for k in retrieval_ks}
    reciprocal_ranks = []
    ndcg = {k: [] for k in retrieval_ks}
    confusion = Counter()
    citation_tp = citation_fp = citation_fn = 0
    supported_predictions = unsupported_supported_predictions = 0
    insufficient_predictions = insufficient_true = insufficient_tp = 0
    review_values = []
    review_pairs = []
    missing_tasks = []
    extra_tasks = sorted(set(predictions) - set(gold))
    gold_claim_count = predicted_claim_count = 0

    for task_id, gold_row in gold.items():
        predicted_row = predictions.get(task_id)
        if predicted_row is None:
            missing_tasks.append(task_id)
            predicted_row = {"retrieved_evidence_ids": [], "claims": []}
        relevant = set(_ids(gold_row.get("gold_evidence_ids", []), f"gold/{task_id}/gold_evidence_ids"))
        retrieved = _ids(predicted_row.get("retrieved_evidence_ids", []), f"predictions/{task_id}/retrieved_evidence_ids")
        if relevant:
            for k in retrieval_ks:
                retrieval[k].append(len(relevant.intersection(retrieved[:k])) / len(relevant))
                ndcg[k].append(_ndcg(retrieved, relevant, k))
            first = next((rank for rank, item in enumerate(retrieved, 1) if item in relevant), None)
            reciprocal_ranks.append(1 / first if first else 0.0)

        gold_claims = _claims(gold_row, f"gold/{task_id}")
        predicted_claims = _claims(predicted_row, f"predictions/{task_id}")
        gold_claim_count += len(gold_claims)
        predicted_claim_count += len(predicted_claims)
        for claim_id in sorted(set(gold_claims) | set(predicted_claims)):
            expected = gold_claims.get(claim_id)
            actual = predicted_claims.get(claim_id)
            expected_label = expected["label"] if expected else "__extra__"
            actual_label = actual["label"] if actual else "__missing__"
            confusion[(expected_label, actual_label)] += 1

            expected_evidence = set(expected.get("evidence_ids", [])) if expected else set()
            actual_evidence = set(actual.get("evidence_ids", [])) if actual else set()
            citation_tp += len(expected_evidence & actual_evidence)
            citation_fp += len(actual_evidence - expected_evidence)
            citation_fn += len(expected_evidence - actual_evidence)

            if actual_label == "supported":
                supported_predictions += 1
                if expected_label != "supported":
                    unsupported_supported_predictions += 1
            if expected_label == "insufficient":
                insufficient_true += 1
            if actual_label == "insufficient":
                insufficient_predictions += 1
                if expected_label == "insufficient":
                    insufficient_tp += 1

            verdicts = actual.get("human_verdicts", []) if actual else []
            review_values.extend(verdicts)
            if len(verdicts) == 2:
                review_pairs.append((verdicts[0], verdicts[1]))

    macro_f1 = []
    per_label = {}
    for label in LABELS:
        tp = confusion[(label, label)]
        fp = sum(count for (expected, actual), count in confusion.items() if actual == label and expected != label)
        fn = sum(count for (expected, actual), count in confusion.items() if expected == label and actual != label)
        precision = _divide(tp, tp + fp)
        recall = _divide(tp, tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(count for (expected, _), count in confusion.items() if expected == label)}
        macro_f1.append(f1)

    matched = sum(confusion[(label, label)] for label in LABELS)
    total_gold_labels = sum(count for (expected, _), count in confusion.items() if expected in LABELS)
    accepted = sum(value == "accepted" for value in review_values)
    return {
        "schema": "research-loom-semantic-evaluation-v1",
        "counts": {
            "gold_tasks": len(gold),
            "predicted_tasks": len(predictions),
            "missing_task_predictions": len(missing_tasks),
            "extra_task_predictions": len(extra_tasks),
            "gold_claims": gold_claim_count,
            "predicted_claims": predicted_claim_count,
            "human_verdicts": len(review_values),
        },
        "retrieval": {
            **{f"recall_at_{k}": _mean(retrieval[k]) for k in retrieval_ks},
            **{f"ndcg_at_{k}": _mean(ndcg[k]) for k in retrieval_ks},
            "mrr": _mean(reciprocal_ranks),
            "evaluated_tasks": len(reciprocal_ranks),
        },
        "claims": {
            "label_accuracy": _divide(matched, total_gold_labels),
            "macro_f1": _mean(macro_f1) if gold_claim_count else None,
            "macro_f1_labels": list(LABELS),
            "per_label": per_label,
            "unsupported_claim_rate": _divide(unsupported_supported_predictions, supported_predictions),
            "abstention_precision": _divide(insufficient_tp, insufficient_predictions),
            "abstention_recall": _divide(insufficient_tp, insufficient_true),
        },
        "citations": {
            "precision": _divide(citation_tp, citation_tp + citation_fp),
            "recall": _divide(citation_tp, citation_tp + citation_fn),
            "true_positive_links": citation_tp,
            "false_positive_links": citation_fp,
            "false_negative_links": citation_fn,
        },
        "human_review": {
            "acceptance_rate": _divide(accepted, len(review_values)),
            "two_reviewer_items": len(review_pairs),
            "cohen_kappa": _cohen_kappa(review_pairs),
        },
        "coverage": {
            "eligible_for_comparison": not extra_tasks,
            "missing_task_ids": missing_tasks,
            "extra_task_ids": extra_tasks,
            "note": "缺失预测计为错误；额外任务不在冻结集分母中，并使本报告不可用于正式版本比较。human_review 为人工投票口径。",
        },
    }


def render_markdown(report: dict) -> str:
    def value(item):
        return "未评测" if item is None else f"{item:.4f}" if isinstance(item, float) else str(item)

    retrieval = report["retrieval"]
    claims = report["claims"]
    citations = report["citations"]
    review = report["human_review"]
    rows = [
        ("MRR", retrieval["mrr"]),
        ("Recall@5", retrieval.get("recall_at_5")),
        ("nDCG@5", retrieval.get("ndcg_at_5")),
        ("Claim Macro-F1", claims["macro_f1"]),
        ("Unsupported Claim Rate", claims["unsupported_claim_rate"]),
        ("Citation Precision", citations["precision"]),
        ("Citation Recall", citations["recall"]),
        ("Human Acceptance Rate", review["acceptance_rate"]),
        ("Cohen's Kappa", review["cohen_kappa"]),
    ]
    body = "\n".join(f"| {name} | {value(metric)} |" for name, metric in rows)
    return (
        "# Research Loom 语义质量评测\n\n"
        "> " + ("格式示例，不能作为质量结论。" if report.get("report_kind") == "example" else "确定性计分报告；冻结、标注与生成来源由提供者声明，脚本不自动证明。") + " 未提供的人工标注显示为“未评测”。\n\n"
        "| 指标 | 结果 |\n|---|---:|\n"
        + body
        + "\n\n## 覆盖范围\n\n```json\n"
        + json.dumps(report["counts"], ensure_ascii=False, indent=2)
        + "\n```\n"
    )
