"""初稿数值核查：公式显式确认，Decimal 计算，不以语言模型心算作为证据。"""

from decimal import Decimal, InvalidOperation
import hashlib
from typing import Literal
from pydantic import Field
from .schemas import Strict
from .store import uid, dump, now
from .workspace import draft_record


class Quantity(Strict):
    name: str = Field(min_length=1, max_length=100)
    value: str = Field(min_length=1, max_length=80)
    unit: str = Field(max_length=80)
    source: str = Field(min_length=1, max_length=500)


class AuditRequest(Strict):
    draft_id: str
    operation: Literal["sum", "weighted_sum"] = "sum"
    parts: list[Quantity] = Field(min_length=1, max_length=100)
    total: Quantity
    weights: list[str] = Field(default_factory=list, max_length=100)
    tolerance: str = Field(default="0.001", max_length=80)


def numeric(value):
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("数值格式错误") from exc
    if (
        not result.is_finite()
        or abs(result) > Decimal("1e50")
        or result.as_tuple().exponent < -50
    ):
        raise ValueError("数值必须有限且精度/幅度在支持范围内")
    return result


def audit(store, req):
    draft = draft_record(store, req.draft_id)
    if any(q.unit != req.total.unit for q in req.parts):
        raise ValueError("单位不一致；请显式换算后核查")
    if req.operation == "weighted_sum" and len(req.parts) != len(req.weights):
        raise ValueError("权重数须等于子项数")
    if req.operation == "sum" and req.weights:
        raise ValueError("普通求和不能携带权重")
    weights = (
        [numeric(w) for w in req.weights]
        if req.operation == "weighted_sum"
        else [Decimal(1)] * len(req.parts)
    )
    expected = sum(
        (numeric(q.value) * w for q, w in zip(req.parts, weights)), Decimal(0)
    )
    total = numeric(req.total.value)
    tolerance = numeric(req.tolerance)
    if tolerance < 0:
        raise ValueError("容差不能为负")
    identity = uid("audit")
    snapshot = req.model_dump()
    result = {
        "id": identity,
        "draft_id": draft["id"],
        "draft_sha": draft["sha"],
        "operation": req.operation,
        "expected": str(expected),
        "reported": str(total),
        "difference": str(total - expected),
        "passed": abs(total - expected) <= tolerance,
        "inputs": snapshot,
        "source_status": "user_supplied_not_independently_verified",
        "note": "只核查已确认数值与公式的一致性；不证明原始实验真实或初稿完整无误。",
    }
    store.execute(
        "INSERT INTO audits VALUES (?,?,?,?,?)",
        (identity, req.draft_id, dump(snapshot), dump(result), now()),
    )
    return result
