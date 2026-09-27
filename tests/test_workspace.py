import json
import subprocess
import sys
import time
from pathlib import Path
import httpx
import pytest
from loom.config import Settings, ROOT
from loom.store import Store, IdempotencyConflict
from loom.routing import ModelRouter, RemoteModel
from loom.gpu import gpu_slot, gpu_busy, ResourceBusy
from loom.workspace import Bibliography, Draft, create_draft, save_bibliography
from loom.experiments import (
    StudySpec,
    create_study,
    get_study,
    start_study,
    AnalysisSpec,
    analyze,
    next_parameters,
)
from loom.lab_worker import LabWorker
from loom.audit import AuditRequest, audit
from loom.execution import StageExecutor
from tests.test_system import system, upload


def configured(tmp_path):
    return Settings(
        data_dir=tmp_path,
        use_dense=False,
        remote_url="https://example.invalid/v1",
        remote_model="test-model",
        remote_key="test-secret",
    )


def test_router_privacy_complexity_busy_images(tmp_path):
    store = Store(tmp_path)
    settings = configured(tmp_path)
    router = ModelRouter(store, settings, object())
    assert router.choose({}, "research", "", busy=False)["backend"] == "local"
    assert (
        router.choose({"allow_remote": True}, "研究方案", "", busy=False)["backend"]
        == "api"
    )
    assert (
        router.choose({"allow_remote": True}, "simple", "", busy=True)["backend"]
        == "api"
    )
    with pytest.raises(ResourceBusy):
        router.choose({}, "研究方案", "", busy=True)
    with pytest.raises(ValueError):
        router.choose({"model_policy": "api"}, "simple", "", busy=False)
    assert (
        router.choose(
            {"allow_remote": True, "allow_remote_images": True},
            "visual",
            "",
            image=True,
            busy=False,
        )["backend"]
        == "local"
    )
    p = store.add_paper("a" * 64, "a.pdf", "title", tmp_path / "a.pdf", 1)
    request = {"allow_remote": True, "paper_ids": [p["id"]]}
    assert router.choose(request, "研究方案", "", busy=False)["backend"] == "local"
    save_bibliography(store, p["id"], Bibliography(allow_remote=True))
    assert router.choose(request, "研究方案", "", busy=False)["backend"] == "api"
    parent = store.create_run("research", {"allow_remote": False})
    request["parent_run_id"] = parent
    assert router.choose(request, "研究方案", "", busy=False)["backend"] == "local"


def test_remote_contract_no_secret_in_usage_and_no_redirect(tmp_path):
    seen = []

    def respond(req):
        seen.append(req)
        return httpx.Response(
            200,
            json={
                "id": "remote-1",
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            },
        )

    model = RemoteModel(configured(tmp_path), httpx.MockTransport(respond))
    text, usage = model.generate("rules", "question", max_tokens=30)
    assert (
        json.loads(text)["ok"]
        and usage["input_tokens"] == 12
        and usage["cost_status"] == "not_priced"
    )
    assert seen[0].url.path == "/v1/chat/completions"
    assert "test-secret" not in json.dumps(usage)
    model.transport = httpx.MockTransport(
        lambda req: httpx.Response(302, headers={"Location": "https://evil.invalid"})
    )
    with pytest.raises(ValueError, match="302"):
        model.generate("rules", "question")


def test_router_ledger_budget_and_revocation(tmp_path):
    settings = configured(tmp_path)
    settings.remote_call_limit = 1
    store = Store(tmp_path)
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"operation":"summary"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    router = ModelRouter(
        store, settings, object(), RemoteModel(settings, httpx.MockTransport(respond))
    )
    rid = store.create_run("analysis", {"allow_remote": True, "model_policy": "api"})
    _, epoch = store.claim("tester", 30, rid)
    cp = {"calls": 0, "usage": []}
    executor = StageExecutor(store, settings, router)
    with store.leased(rid, "tester", epoch):
        args = (
            rid,
            cp,
            "first",
            "system",
            "instruction",
            AnalysisSpec,
            "test",
            lambda *x: None,
            lambda: None,
        )
        assert executor.generate(*args, max_tokens=100)["operation"] == "summary"
        assert executor.generate(*args, max_tokens=100)["operation"] == "summary"
        with pytest.raises(ValueError, match="预算"):
            executor.generate(
                rid,
                cp,
                "second",
                "s",
                "q",
                AnalysisSpec,
                "test",
                lambda *x: None,
                lambda: None,
                max_tokens=100,
            )
    assert len(seen) == 1
    rows = store.query("SELECT input,usage FROM model_attempts WHERE run_id=?", (rid,))
    assert json.loads(rows[0]["input"])["routing"]["backend"] == "api"
    assert "test-secret" not in json.dumps(rows)
    study_store, study, _ = setup_study(tmp_path / "study")
    protected = ModelRouter(study_store, configured(tmp_path), object())
    assert not protected.permitted(
        {"study_id": study["id"], "allow_remote": True}, False
    )


def test_gpu_wait_does_not_spend_model_budget(tmp_path):
    store = Store(tmp_path)
    settings = Settings(data_dir=tmp_path, use_dense=False)
    router = ModelRouter(store, settings, object())
    rid = store.create_run("research", {})
    _, epoch = store.claim("tester", 30, rid)
    cp = {"calls": 0, "usage": []}
    with store.leased(rid, "tester", epoch):
        # 冻结本地路由后 GPU 被训练占用，模拟检查与调用之间的竞争。
        router.prepare(rid, cp, "first", "q", None, 100)
        with gpu_slot(store.root):
            with pytest.raises(ResourceBusy):
                StageExecutor(store, settings, router).generate(
                    rid,
                    cp,
                    "first",
                    "s",
                    "q",
                    AnalysisSpec,
                    "test",
                    lambda *x: None,
                    lambda: None,
                    max_tokens=100,
                )
        assert cp["calls"] == 0
    assert (
        store.query("SELECT status FROM model_attempts")[0]["status"] == "not_executed"
    )


def test_metadata_revision_draft_and_citation_anchors(system):
    client, store, worker, model = system
    pid = upload(system)
    meta = client.get("/api/library").json()[0]["bibliography"]
    meta.update(year=2024, venue="TestConf", allow_remote=True)
    assert client.put("/api/library/" + pid, json=meta).status_code == 200
    assert client.put("/api/library/" + pid, json=meta).status_code == 409
    bad = {**meta, "revision": 1, "grade_score": 90}
    assert client.put("/api/library/" + pid, json=bad).status_code == 422
    draft = client.post(
        "/api/drafts",
        json={
            "title": "Methods",
            "text": "We use a shared encoder to improve sample efficiency.",
        },
    ).json()
    ev = store.get_evidence(paper_ids=[pid])[0]
    model.responses = [
        {
            "suggestions": [
                {
                    "paragraph_id": "p1",
                    "evidence_id": ev["id"],
                    "quote": "The framework uses a shared encoder",
                    "relation": "supports",
                    "reason": "Related method.",
                },
                {
                    "paragraph_id": "p1",
                    "evidence_id": "invented",
                    "quote": "Fabricated evidence",
                    "relation": "supports",
                    "reason": "bad",
                },
            ]
        }
    ]
    req = {"draft_id": draft["id"], "paper_ids": [pid]}
    r = client.post(
        "/api/citations", json=req, headers={"Idempotency-Key": "citation-1"}
    )
    assert r.status_code == 202
    rid = r.json()["run_id"]
    assert (
        client.post(
            "/api/citations", json=req, headers={"Idempotency-Key": "citation-1"}
        ).json()["run_id"]
        == rid
    )
    worker.execute(rid)
    result = store.run(rid)
    assert result["status"] == "completed", result["error"]
    item = result["result"]["suggestions"][0]
    assert (
        len(result["result"]["suggestions"]) == 1
        and item["eligible"]
        and item["anchor"]["draft_sha"] == draft["sha"]
        and item["semantic_status"] == "needs_human_review"
    )
    revised = client.post(
        "/api/drafts",
        json={
            "title": "Methods",
            "text": "An entirely different draft that requires new citations.",
            "parent_id": draft["id"],
        },
    ).json()
    assert (
        revised["sha"] != draft["sha"] and item["anchor"]["draft_id"] != revised["id"]
    )


def setup_study(tmp_path, script=None, **changes):
    store = Store(tmp_path / "data")
    root = tmp_path / "source"
    root.mkdir(parents=True, exist_ok=True)
    code = (
        script
        or "import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--output');a=p.parse_args()\nc=json.loads(Path(a.config).read_text(encoding='utf-8'))\nPath(a.output,'metrics.json').write_text(json.dumps({'loss':(c['parameters']['x']-2)**2+c['seed']*.001,'constraint':0}))\n"
    )
    (root / "train.py").write_text(code, encoding="utf-8")
    adapter = {
        "id": "fixture",
        "name": "Engineering test fixture, not paper results",
        "root": str(root),
        "entrypoint": "train.py",
        "source_files": ["train.py"],
        "parameters": {"x": [1, 2]},
        "parameter_groups": {"x": "hyperparameter"},
        "metrics": ["loss", "constraint"],
        "fixture": True,
    }
    (store.root / "trainers").mkdir()
    (store.root / "trainers" / "fixture.json").write_text(
        json.dumps(adapter), encoding="utf-8"
    )
    spec = StudySpec(
        name="Test fixture",
        trainer_id="fixture",
        parameters={"x": [1, 2]},
        objective="loss",
        constraints={"constraint": 1},
        seeds=[11, 23],
        max_trials=2,
        budget_seconds=120,
        trial_timeout=20,
        protocol="fixed validation fixture; no paper claims",
        **changes
    )
    study = create_study(store, spec)
    return store, study, root


def drive(store, sid, timeout=40):
    worker = LabWorker(store)
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            worker.tick()
            study = get_study(store, sid)
            if study["status"] != "running":
                return study
            time.sleep(0.15)
        raise AssertionError("Study stalled: " + json.dumps(study))
    finally:
        for child in worker.children.values():
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


def test_real_subprocess_search_repeats_and_chatbi(tmp_path):
    store, study, _ = setup_study(tmp_path)
    sid = study["id"]
    start_study(store, sid)
    with pytest.raises(IdempotencyConflict):
        start_study(store, sid)
    final = drive(store, sid)
    assert final["status"] == "completed"
    assert len(final["trials"]) == 2
    assert all(t["metrics"]["repeat_count"] == 2 for t in final["trials"])
    result = analyze(store, sid, AnalysisSpec())
    assert result["rows"][0]["params"]["x"] == 2 and len(result["lineage"]) == 2
    attempt = final["trials"][0]["repeats"][0]
    assert (
        store.root / "experiments" / attempt["id"] / "result-manifest.json"
    ).is_file()
    assert not gpu_busy(store.root)


@pytest.mark.parametrize("failure", ["changed", "nan", "timeout"])
def test_faults_not_success_or_repeated_samples(tmp_path, failure):
    script = "import time; time.sleep(30)" if failure == "timeout" else None
    if failure == "nan":
        script = "import sys\nfrom pathlib import Path\nPath(sys.argv[-1],'metrics.json').write_text('{\"loss\":NaN}')"
    store, study, root = setup_study(tmp_path, script)
    if failure == "changed":
        (root / "train.py").write_text("raise AssertionError()", encoding="utf-8")
        with pytest.raises(ValueError):
            start_study(store, study["id"])
        assert get_study(store, study["id"])["status"] == "draft"
        assert not store.query("SELECT * FROM trial_attempts")
        return
    if failure == "timeout":
        spec = StudySpec.model_validate(
            {**study["spec"], "trial_timeout": 1, "max_trials": 1, "seeds": [11]}
        )
        study = create_study(store, spec)
    start_study(store, study["id"])
    final = drive(store, study["id"])
    assert final["status"] == "failed"
    assert all(
        t["status"] == "failed" and t["metrics"] is None for t in final["trials"]
    )
    assert not analyze(store, study["id"], AnalysisSpec())["rows"]


def test_gpu_lock_process_death_and_duplicate_attempt(tmp_path):
    store, study, _ = setup_study(tmp_path, "import time; time.sleep(30)")
    start_study(store, study["id"])
    worker = LabWorker(store)
    worker.tick()
    worker.tick()
    deadline = time.time() + 10
    while time.time() < deadline and not gpu_busy(store.root):
        time.sleep(0.1)
    assert gpu_busy(store.root)
    attempt = get_study(store, study["id"])["trials"][0]["repeats"][0]
    extra = subprocess.run(
        [
            sys.executable,
            "-m",
            "loom.trial_runner",
            "--data-dir",
            str(store.root),
            "--attempt",
            attempt["id"],
        ],
        cwd=ROOT,
        timeout=10,
        capture_output=True,
    )
    assert extra.returncode == 0
    assert (
        get_study(store, study["id"])["trials"][0]["repeats"][0]["pid"]
        == attempt["pid"]
    )
    child = worker.children[attempt["id"]]
    child.terminate()
    child.wait(timeout=5)
    assert not gpu_busy(store.root)
    worker.tick()
    final = get_study(store, study["id"])
    assert final["trials"][0]["repeats"][0]["status"] == "failed"
    # 下一 seed 可能已启动；测试结束清理自己创建的进程。
    store.execute("UPDATE studies SET stop=1 WHERE id=?", (study["id"],))
    for process in worker.children.values():
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


def test_dispatcher_restart_keeps_running_training_identity(tmp_path):
    import psutil
    from loom.training import TrainingContext

    store, study, _ = setup_study(tmp_path, "import time; time.sleep(30)")
    start_study(store, study["id"])
    dispatcher = subprocess.Popen(
        [sys.executable, "-m", "loom.lab_worker", "--data-dir", str(store.root)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    training_pid = None
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            rows = store.query("SELECT * FROM trial_attempts WHERE status='running'")
            if rows and gpu_busy(store.root):
                break
            time.sleep(0.1)
        assert rows and gpu_busy(store.root)
        identity = rows[0]["id"]
        training_pid = rows[0]["pid"]
        dispatcher.terminate()
        dispatcher.wait(timeout=5)
        assert psutil.pid_exists(training_pid) and gpu_busy(store.root)
        replacement = LabWorker(store)
        replacement.tick()
        row = store.query("SELECT * FROM trial_attempts WHERE id=?", (identity,))[0]
        assert (
            row["status"] == "running"
            and row["pid"] == training_pid
            and row["launches"] == 1
        )
        store.execute("UPDATE studies SET stop=1 WHERE id=?", (study["id"],))
        deadline = time.time() + 5
        while time.time() < deadline and gpu_busy(store.root):
            time.sleep(0.1)
        assert not gpu_busy(store.root)
        replacement.tick()
        assert get_study(store, study["id"])["status"] == "cancelled"
    finally:
        if dispatcher.poll() is None:
            dispatcher.terminate()
            dispatcher.wait(timeout=5)
        if training_pid:
            try:
                psutil.Process(training_pid).terminate()
            except psutil.NoSuchProcess:
                pass


def test_training_sdk_atomic_metrics_and_bounded_progress(tmp_path):
    from loom.training import TrainingContext

    config = tmp_path / "config.json"
    config.write_text('{"seed":11}', encoding="utf-8")
    ctx = TrainingContext(config, tmp_path / "out")
    ctx.report(10, {"loss": 0.2})
    ctx.finish({"loss": 0.1})
    assert json.loads((ctx.output / "metrics.json").read_text()) == {"loss": 0.1}

    assert (
        json.loads((ctx.output / "progress.jsonl").read_text())["axis"]
        == "environment_step"
    )
    assert not list(ctx.output.glob("*.tmp"))
    with pytest.raises(ValueError):
        ctx.finish({"loss": float("nan")})
    assert json.loads((ctx.output / "metrics.json").read_text()) == {"loss": 0.1}

    import numpy as np

    ctx.finish({"loss": np.float64(0.3)})
    assert json.loads((ctx.output / "metrics.json").read_text()) == {"loss": 0.3}


def test_api_controller_requires_authorised_outbound_summary(tmp_path):
    base = {
        "name": "Bounded controller fixture",
        "trainer_id": "fixture",
        "parameters": {"x": [1, 2]},
        "objective": "loss",
        "constraints": {"constraint": 1},
        "seeds": [11],
        "max_trials": 2,
        "budget_seconds": 120,
        "trial_timeout": 20,
        "protocol": "fixed protocol; only choose from frozen candidates",
        "controller": "api_bounded",
        "controller_max_calls": 1,
        "initial_parameters": {"x": 1},
    }
    with pytest.raises(ValueError, match="明确允许"):
        StudySpec.model_validate(base)
    assert StudySpec.model_validate({**base, "allow_remote": True}).controller == "api_bounded"


def test_api_controller_only_offers_one_parameter_neighbours():
    baseline = {"reward": 1.0, "gamma": 0.99, "H": 100.0}
    study = {
        "spec": {
            "controller": "api_bounded",
            "controller_max_calls": 0,
            "parameters": {"reward": [0.5, 1.0, 2.0], "gamma": [0.95, 0.99], "H": [50.0, 100.0]},
            "direction": "minimize", "sampler": "tpe", "search_seed": 1,
        },
        "trials": [{
            "status": "completed", "params": baseline,
            "metrics": {"feasible": True, "objective_mean": 3.0},
        }],
        "events": [],
    }
    proposal = next_parameters(study)
    assert proposal is not None
    assert sum(proposal[key] != baseline[key] for key in baseline) == 1


def test_external_watchdog_stops_native_gil_block(tmp_path):
    script = "import ctypes,os\nif os.name=='nt': ctypes.PyDLL('kernel32').Sleep(30000)\nelse: ctypes.PyDLL(None).sleep(30)\n"
    store, study, _ = setup_study(tmp_path, script)
    study = create_study(
        store,
        StudySpec.model_validate(
            {**study["spec"], "trial_timeout": 2, "max_trials": 1, "seeds": [11]}
        ),
    )
    start_study(store, study["id"])
    started = time.monotonic()
    final = drive(store, study["id"], timeout=12)
    assert final["status"] == "failed" and time.monotonic() - started < 10
    assert not gpu_busy(store.root)


def test_decimal_audit_units_and_provenance(tmp_path):
    store = Store(tmp_path)
    draft = create_draft(
        store, Draft(title="Draft", text="A draft with measurements to check.")
    )
    q = lambda n, v, u="kWh": {"name": n, "value": v, "unit": u, "source": "table 1"}
    req = AuditRequest(
        draft_id=draft["id"],
        parts=[q("a", "0.1"), q("b", "0.2")],
        total=q("total", "0.3"),
        tolerance="0",
    )
    result = audit(store, req)
    assert (
        result["passed"]
        and result["difference"] == "0.0"
        and result["draft_sha"] == draft["sha"]
    )
    req.total.unit = "Wh"
    with pytest.raises(ValueError):
        audit(store, req)


def test_analysis_natural_language_uses_fixed_operations(system, tmp_path):
    client, store, worker, model = system
    other, study, _ = setup_study(tmp_path / "fixture")
    # 当前 API 的 Store 中创建同一适配器计划，测试不运行训练。
    (store.root / "trainers").mkdir(exist_ok=True)
    (store.root / "trainers" / "fixture.json").write_text(
        (other.root / "trainers" / "fixture.json").read_text(), encoding="utf-8"
    )
    study = create_study(store, StudySpec.model_validate(study["spec"]))
    model.responses = [
        {"operation": "summary", "parameter": None, "feasible_only": True, "limit": 20}
    ]
    r = client.post(
        "/api/analyses",
        json={"study_id": study["id"], "question": "How many trials completed?"},
    )
    assert r.status_code == 202
    worker.execute(r.json()["run_id"])
    run = store.run(r.json()["run_id"])
    assert run["status"] == "completed", run["error"]
    assert run["result"]["rows"][0]["count"] == 0
