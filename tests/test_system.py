import json
import fitz
import pytest
from fastapi.testclient import TestClient
from loom.api import create_app
from loom.config import Settings
from loom.schemas import RunRequest
from loom.verification import verify_findings


class FakeModel:
    """仅用于契约/故障测试；生产没有模拟模型开关。"""
    model = None
    def __init__(self):
        self.responses = []
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value) if not isinstance(value, str) else value, {"input_tokens": 10, "output_tokens": 10, "seconds": .01}


@pytest.fixture
def system(tmp_path):
    model = FakeModel()
    app = create_app(Settings(data_dir=tmp_path, use_dense=False,allowed_hosts=("testserver",)), model=model, start_worker=False)
    with TestClient(app) as client:
        yield client, app.state.store, app.state.worker, model


def sample_pdf(text="The framework uses a shared encoder for sample efficiency. Evaluation uses reward and generalization tests."):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 80, 520, 300), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def upload(system, text=None):
    client, store, worker, model = system
    response = client.post("/api/papers", files={"file": ("study.pdf", sample_pdf(text) if text else sample_pdf(), "application/pdf")})
    assert response.status_code == 202
    body = response.json()
    worker.execute(body["run_id"])
    assert store.run(body["run_id"])["status"] == "completed"
    return body["paper"]["id"]


def outputs(store, paper_id, mode="understand"):
    evidence = store.get_evidence(paper_ids=[paper_id])[0]
    route = {"queries": ["shared encoder reward evaluation"], "visual_targets": [], "rationale": "检查方法和实验"}
    finding = {"findings": [{"category": "method", "statement": "作者使用共享编码器。", "citations": [{"evidence_id": evidence["id"], "quote": "The framework uses a shared encoder"}], "basis": "author_reported"}]}
    evaluation = {"findings": [{**finding["findings"][0], "category": "metric"}]}
    review = {"assessments": [{"finding_id": "f1", "status": "supported", "reason": "引文支持"}, {"finding_id": "f2", "status": "supported", "reason": "引文支持"}]}
    if mode == "plan":
        return [route, finding, evaluation]
    return [route, finding, evaluation, review] if mode == "understand" else [route, finding, review]


def test_upload_idempotent_scoped_and_persistent(system):
    client, store, worker, _ = system
    pid = upload(system)
    # PyMuPDF embeds generated document IDs, so exercise identical bytes explicitly.
    data = sample_pdf("A second paper studies model calibration and uncertainty.")
    first = client.post("/api/papers", files={"file": ("first.pdf", data, "application/pdf")}).json()
    second = client.post("/api/papers", files={"file": ("other.pdf", data, "application/pdf")}).json()
    assert second["duplicate"] and first["paper"]["id"] == second["paper"]["id"]
    worker.execute(first["run_id"])
    ev = store.get_evidence(paper_ids=[pid])
    assert ev and all(item["paper_id"] == pid for item in ev)
    assert client.get(f"/api/papers/{pid}/pages/1/image").headers["content-type"] == "image/png"
    from loom.store import Store
    reopened = Store(store.root)
    assert reopened.paper(pid)["status"] == "ready"


def test_run_sources_export_reviews_and_checkpoint(system):
    client, store, worker, model = system
    pid = upload(system)
    model.responses = outputs(store, pid)
    response = client.post("/api/runs", json={"paper_ids": [pid], "mode": "understand"})
    run_id = response.json()["run_id"]
    worker.execute(run_id)
    run = store.run(run_id)
    assert run["status"] == "completed", run["error"]
    assert run["result"]["quality"]["quotes_matched"] == 2
    assert len(run["checkpoint"]["论文理解"]["findings"]) == 1
    assert run["result"]["scope"]["external_search"] is False
    assert run["result"]["findings"][0]["semantic_status"] == "not_independently_verified"
    assert client.get(f"/api/runs/{run_id}/export").status_code == 200
    assert client.post(f"/api/runs/{run_id}/reviews", json={"item_id": "not-mine", "verdict": "accepted"}).status_code == 422
    assert client.post(f"/api/runs/{run_id}/reviews", json={"item_id": "f1", "verdict": "accepted", "note": "已核对原文"}).status_code == 200
    assert len(client.get(f"/api/runs/{run_id}").json()["reviews"]) == 1
    calls = model.calls
    worker.execute(run_id)
    assert model.calls == calls  # 完成阶段不会重复调用模型。
    response = client.get(f"/api/runs/{run_id}/events")
    assert "event: progress" in response.text and "event: done" in response.text
    last = store.events(run_id)[-1]["seq"]
    response = client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID":str(last)})
    assert "event: progress" not in response.text


def test_schema_failure_is_not_success_and_retry_resumes(system):
    client, store, worker, model = system
    pid = upload(system)
    model.responses = [outputs(store, pid)[0], "not json", "still not json"]
    run_id = client.post("/api/runs", json={"paper_ids": [pid]}).json()["run_id"]
    worker.execute(run_id)
    assert store.run(run_id)["status"] == "failed"
    assert store.run(run_id)["result"] is None
    assert "研究路径" in store.run(run_id)["checkpoint"]
    assert client.post(f"/api/runs/{run_id}/retry").status_code == 200
    model.responses = outputs(store, pid)[1:]
    worker.execute(run_id)
    assert store.run(run_id)["status"] == "completed"
    assert model.calls == 6


def test_cancellation_and_bounds(system):
    client, store, worker, model = system
    pid = upload(system)
    run_id = client.post("/api/runs", json={"paper_ids": [pid]}).json()["run_id"]
    client.post(f"/api/runs/{run_id}/cancel")
    worker.execute(run_id)
    assert store.run(run_id)["status"] == "cancelled" and model.calls == 0
    assert client.post("/api/runs", json={"paper_ids": [pid], "mode": "figure", "page": 2}).status_code == 422
    assert client.post("/api/runs", json={"paper_ids": [pid], "bbox": [0,0,2,1]}).status_code == 422
    assert client.post("/api/runs", json={"paper_ids": ["nonexistent"]}).status_code == 404
    assert client.post("/api/runs", json={"paper_ids": [pid]}, headers={"origin":"https://unrelated.example"}).status_code == 403


def test_missing_and_invented_quotes_never_pass():
    ev = {"id":"e1", "paper_id":"p1", "page":1, "kind":"text", "content":"The paper evaluates sample efficiency."}
    rows = verify_findings([
        {"statement":"a", "citations":[{"evidence_id":"wrong", "quote":"The paper evaluates"}]},
        {"statement":"b", "citations":[{"evidence_id":"e1", "quote":"The paper beats all baselines"}]},
        {"statement":"c", "citations":[]},
    ], [ev])
    assert all(r["source_status"] == "needs_review" for r in rows)
    visual = {**ev, "kind":"visual"}
    row = verify_findings([{"statement":"a", "citations":[{"evidence_id":"e1", "quote":"The paper evaluates"}]}], [visual])[0]
    assert row["source_status"] == "visual_interpretation"


def test_invalid_upload_and_queue_limit(system):
    client, store, worker, model = system
    assert client.post("/api/papers", files={"file": ("bad.pdf",b"not a pdf","application/pdf")}).status_code == 422
    assert client.post("/api/papers", files={"file": ("bad.txt",b"hello","text/plain")}).status_code == 415
    pid = upload(system)
    worker.settings.queue_limit = 1
    assert client.post("/api/runs", json={"paper_ids":[pid]}).status_code == 202
    assert client.post("/api/runs", json={"paper_ids":[pid]}).status_code == 429


def test_plan_has_falsification_and_never_claims_novelty(system):
    client, store, worker, model = system
    pid = upload(system)
    ev = store.get_evidence(paper_ids=[pid])[0]
    model.responses = outputs(store,pid,"plan") + [{"directions":[{
        "title":"检验共享编码器迁移", "motivation":"检验可迁移性", "change":"改变测试分布", "assumptions":["有训练数据"],
        "evidence_ids":[ev["id"]], "experiment":{"relation":{"candidate":"共享编码器","reference":"原方法","metric":"测试回报","metric_direction":"higher","expectation":"better","condition":"固定资源和跨任务测试"}, "hypothesis":"共享编码器能改善泛化", "setup":"固定训练资源，改变测试任务", "baselines":["原方法"], "metrics":["回报"], "ablations":["独立编码器"], "falsification":"跨分布收益消失", "feasibility":"先做小规模验证"}, "novelty_queries":["shared encoder generalization"], "risks":["收益来自额外参数"]
    }], "open_questions":["可用时间？"]}, {"assessments":[{"direction_id":"d1","recommendation":"explore_after_checks","concerns":[],"next_check":"先完成小规模验证"}]}, {"assessments":[{"finding_id":"f1","status":"supported","reason":"引文支持"},{"finding_id":"f2","status":"supported","reason":"引文支持"}]}]
    run_id = client.post("/api/runs",json={"paper_ids":[pid],"mode":"plan"}).json()["run_id"]
    worker.execute(run_id)
    run = store.run(run_id)
    assert run["status"] == "completed", run["error"]
    d = run["result"]["directions"][0]
    assert d["novelty_status"] == "not_verified" and not d["missing_experiment_fields"]
    assert d["model_review"]["recommendation"] == "explore_after_checks"
    assert d["experiment"]["relation_status"] == "compiled"


def test_line_wrapped_quotes_and_absence_claims():
    evidence = {"id":"e1", "paper_id":"p1", "page":1, "kind":"text", "content":"The shared repre-\nsentations improve prediction quality."}
    findings = [{"statement":"作者未明确说明泛化实验。", "basis":"author_reported", "citations":[{"evidence_id":"e1", "quote":"The shared representations improve prediction quality."}]}]
    result = verify_findings(findings,[evidence])[0]
    assert result["source_status"] == "quote_matched"
    assert result["basis"] == "model_inference" and result["review_reasons"]


def test_different_prompt_version_does_not_mix_checkpoints(system):
    client, store, worker, model = system
    pid = upload(system)
    run_id = client.post("/api/runs", json={"paper_ids":[pid]}).json()["run_id"]
    store.update_run(run_id, checkpoint={"prompt_version":"old-version", "研究路径": {"queries":["old"]}})
    worker.execute(run_id)
    run = store.run(run_id)
    assert run["status"] == "failed" and "旧版" in run["error"]
    assert model.calls == 0


def test_hypothesis_and_falsification_have_consistent_direction():
    from loom.verification import compile_relation
    relation = {"candidate":"A", "reference":"B", "metric":"RMSE", "metric_direction":"lower", "expectation":"not_worse", "condition":"相同负载"}
    hypothesis, falsification = compile_relation(relation)
    assert "不劣于" in hypothesis and "劣于B且超出" in falsification
    relation["expectation"] = "better"
    assert "低于B" in compile_relation(relation)[0]
    relation["metric_direction"] = "higher"
    assert "高于B" in compile_relation(relation)[0]
    relation["expectation"] = "exploratory"
    assert "没有可否定" in compile_relation(relation)[1]
