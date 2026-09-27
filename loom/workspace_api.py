"""文献、引用、实验和 Data Agent 的独立接口。"""

import csv
import io
import json
import time
from pydantic import Field
from typing import Literal
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response
from .config import ROOT
from .gpu import gpu_busy, ResourceBusy
from .workspace import (
    Bibliography,
    Draft,
    CitationRequest,
    bibliography,
    save_bibliography,
    create_draft,
    draft_record,
)
from .experiments import (
    StudySpec,
    AnalysisSpec,
    trainers,
    create_study,
    get_study,
    start_study,
    analyze,
    event,
)
from .platform import AnalysisRequest
from .audit import AuditRequest, audit
from .experiment_assistant import AdviceRequest, Decision, Note, notebook, prepare_advice, decide
from . import research_data as rd
from .schemas import Strict
from .batch_experiments import preview as preview_batch, pause_batch, resume_batch
from .experiments import trainer


class BatchSubmission(Strict):
    spec: StudySpec
    confirmation_sha256: str


class ArtifactReview(Strict):
    note: str = Field(min_length=10,max_length=2000)
    artifact_max_bytes: int | None = Field(default=None,ge=1,le=2*1024**3)


def install_routes(app, store, settings, worker, model):
    @app.post('/api/attempts/{aid}/revalidate')
    def revalidate_artifacts(aid: str, req: ArtifactReview):
        from .artifact_finalization import revalidate
        return revalidate(store,aid,req.note,req.artifact_max_bytes)

    @app.post('/api/batches/preview')
    def batch_preview(req: StudySpec):
        return preview_batch(req, trainer(store,req.trainer_id))

    @app.post('/api/batches', status_code=201)
    def batch_create(req: BatchSubmission, request: Request):
        from .batch_experiments import submit_batch
        return submit_batch(store,req.spec,req.confirmation_sha256,request.headers.get('Idempotency-Key'))

    @app.get('/api/trainers/{identity}/preflight')
    def trainer_preflight(identity: str):
        if settings.demo_mode:
            raise ValueError('CPU 演示模式不导入训练依赖，请在正常模式执行预检')
        from .preflight import check_preflight
        return check_preflight(trainer(store,identity))

    @app.post('/api/studies/{sid}/pause')
    def batch_pause(sid: str):
        pause_batch(store,sid)
        return get_study(store,sid)

    @app.post('/api/studies/{sid}/resume')
    def batch_resume(sid: str):
        if settings.demo_mode:
            raise ValueError('演示模式不能启动真实训练')
        resume_batch(store,sid)
        return get_study(store,sid)

    @app.get('/api/data/datasets')
    def data_sets():
        return store.query('SELECT id,name,created FROM research_datasets ORDER BY created DESC LIMIT 100')

    @app.post('/api/data/import', status_code=201)
    def data_import(req: rd.CSVImport):
        return rd.import_csv(store,req)

    @app.post('/api/data/from-studies', status_code=201)
    def data_studies(req: rd.StudySelection):
        return rd.from_studies(store,req)

    @app.post('/api/data/demo', status_code=201)
    def data_demo():
        return rd.demo_dataset(store)

    @app.get('/api/data/datasets/{identity}')
    def data_set(identity: str):
        return rd.dataset(store,identity)

    @app.post('/api/data/datasets/{identity}/analyze')
    def data_analyze(identity: str, req: rd.DataPlan):
        return rd.analyze_data(store,rd.dataset(store,identity),req)

    @app.post('/api/data/questions', status_code=202)
    def data_question(req: rd.DataQuestion, request: Request):
        rd.dataset(store,req.dataset_id)
        return enqueue('research_data',req,request)

    @app.post('/api/data/audits')
    def data_audit(req: rd.DataAudit):
        return rd.audit_data(store,req)

    @app.get('/api/data/reports/{identity}')
    def data_report(identity: str):
        return rd.report(store,identity)

    @app.get('/api/data/reports/{identity}/export')
    def data_export(identity: str, format: Literal['csv','json']='csv'):
        result=rd.report(store,identity)
        body=rd.export_csv(result) if format=='csv' else json.dumps(result,ensure_ascii=False,indent=2)
        return Response(body,media_type='text/csv' if format=='csv' else 'application/json',
                        headers={'Content-Disposition':f'attachment; filename="{identity}.{format}"'})

    @app.get('/api/data/reports/{identity}/chart')
    def data_chart(identity: str, metric: str):
        return Response(rd.chart_svg(rd.report(store,identity),metric),media_type='image/svg+xml')

    @app.get("/api/lab/status")
    def lab_status():
        rows = store.query("SELECT * FROM service_heartbeats WHERE name='lab'")
        heartbeat = rows[0] if rows else None
        return {
            "dispatcher_recent": bool(
                heartbeat and time.time() - heartbeat["updated"] < 10
            ),
            "heartbeat": heartbeat,
            "gpu_busy": gpu_busy(store.root),
            "studies": store.query(
                "SELECT status,COUNT(*) n FROM studies GROUP BY status"
            ),
            "note": "最近心跳只证明调度器近期响应，不证明某一训练仍有进展。",
        }

    @app.post("/api/audits")
    def numerical_audit(req: AuditRequest):
        return audit(store, req)

    @app.get("/api/audits/{identity}")
    def saved_audit(identity: str):
        rows = store.query("SELECT result FROM audits WHERE id=?", (identity,))
        if not rows:
            raise KeyError("核查不存在")
        return json.loads(rows[0]["result"])

    @app.get("/workspace")
    def workspace():
        return FileResponse(ROOT / "web" / "workspace.html")

    @app.get("/api/models")
    def models():
        return {
            'demo_mode':settings.demo_mode,
            "policy": "privacy > explicit selection > GPU availability > task complexity",
            "remote_configured": bool(
                settings.remote_url and settings.remote_model and settings.remote_key
            ),
            "remote_model": settings.remote_model or None,
            "remote_vision": settings.remote_vision,
            "gpu_busy": gpu_busy(store.root),
            "local_model": "Qwen3-VL-4B-Instruct",
            "remote_calls_per_run": settings.remote_call_limit,
            "remote_output_token_budget": settings.remote_output_budget,
            "cost_status": "not_priced",
            "note": "GPU 状态仅覆盖使用本平台资源锁的进程。API 默认关闭，许可需要同时覆盖任务和资料。",
        }

    @app.get("/api/library")
    def library():
        return [
            {**p, "bibliography": bibliography(store, p["id"])} for p in store.papers()
        ]

    @app.put("/api/library/{pid}")
    def update_metadata(pid: str, req: Bibliography):
        return save_bibliography(store, pid, req)

    @app.get("/api/library/{pid}/history")
    def metadata_history(pid: str):
        store.paper(pid)
        return [
            {**r, "data": json.loads(r["data"])}
            for r in store.query(
                "SELECT * FROM bibliography_history WHERE paper_id=? ORDER BY revision",
                (pid,),
            )
        ]

    @app.post("/api/drafts", status_code=201)
    def new_draft(req: Draft):
        return create_draft(store, req)

    @app.get("/api/drafts")
    def drafts():
        return store.query(
            "SELECT id,title,sha,parent_id,created FROM drafts ORDER BY created DESC LIMIT 100"
        )

    @app.get("/api/drafts/{did}")
    def draft(did: str):
        return draft_record(store, did)

    def enqueue(kind, req, request):
        key = request.headers.get("idempotency-key")
        if key is not None and (not key.strip() or len(key) > 128):
            raise HTTPException(422, "Invalid Idempotency-Key")
        if req.model_policy == "api" and hasattr(model, "choose"):
            model.choose(req.model_dump(), "preflight", "", busy=False)
        return {"run_id": worker.enqueue(kind, req.model_dump(), key=key)}

    @app.post("/api/citations", status_code=202)
    def citations(req: CitationRequest, request: Request):
        draft_record(store, req.draft_id)
        for pid in req.paper_ids:
            if store.paper(pid)["status"] != "ready":
                raise HTTPException(409, "请等待文献解析完成")
        return enqueue("citations", req, request)

    @app.get("/api/trainers")
    def adapters():
        return trainers(store)

    @app.post("/api/studies", status_code=201)
    def new_study(req: StudySpec):
        if req.batch_mode:
            raise ValueError('批次请使用矩阵预览和确认接口')
        req = StudySpec.model_validate({**req.model_dump(), "review_mode": True})
        return create_study(store, req)

    @app.get("/api/studies/{sid}/notebook")
    def experiment_notebook(sid: str):
        return notebook(store, sid)

    @app.post("/api/studies/{sid}/notes", status_code=201)
    def experiment_note(sid: str, req: Note):
        get_study(store, sid)
        event(store, sid, "researcher_note", {"text": req.text})
        return {"ok": True}

    @app.post("/api/experiment-advice", status_code=202)
    def experiment_advice(req: AdviceRequest, request: Request):
        payload = prepare_advice(store, req)
        if req.model_policy == "api" and hasattr(model, "choose"):
            model.choose(payload, "实验建议", "", busy=False)
        key = request.headers.get("idempotency-key")
        if key is not None and (not key.strip() or len(key) > 128):
            raise HTTPException(422, "Invalid Idempotency-Key")
        return {"run_id": worker.enqueue("experiment_advice", payload, key=key)}

    @app.get("/api/studies/{sid}/advice")
    def experiment_advice_history(sid: str):
        get_study(store, sid)
        rows = store.query("SELECT id,status,result,error,created FROM runs WHERE kind='experiment_advice' AND json_extract(request,'$.study_id')=? ORDER BY created", (sid,))
        for row in rows:
            row["result"] = json.loads(row["result"]) if row["result"] else None
            decisions = store.query("SELECT * FROM experiment_decisions WHERE run_id=?", (row["id"],))
            row["decision"] = decisions[0] if decisions else None
        return rows

    @app.post("/api/experiment-advice/{rid}/decision")
    def experiment_decision(rid: str, req: Decision):
        return decide(store, rid, req)

    @app.get("/api/studies")
    def studies():
        return [
            {**r, "spec": json.loads(r["spec"])}
            for r in store.query(
                "SELECT id,spec,status,created,deadline FROM studies ORDER BY created DESC LIMIT 100"
            )
        ]

    @app.get("/api/studies/{sid}")
    def study(sid: str):
        return get_study(store, sid)

    @app.post("/api/studies/{sid}/start")
    def start(sid: str):
        if settings.demo_mode:
            raise ValueError('演示模式不能启动真实训练')
        existing = get_study(store, sid)
        if not existing["spec"].get("review_mode") and not existing['spec'].get('batch_mode'):
            raise ValueError("旧版自动搜索计划只供历史查看，请新建一轮一组参数的实验助手计划")
        start_study(store, sid)
        return get_study(store, sid)

    @app.post("/api/studies/{sid}/cancel")
    def cancel(sid: str):
        current = get_study(store, sid)
        store.execute("UPDATE studies SET stop=1 WHERE id=?", (sid,))
        if current["status"] in ('draft','paused'):
            store.execute("UPDATE studies SET status='cancelled' WHERE id=?", (sid,))
            store.execute("UPDATE trial_attempts SET status='cancelled',error='cancelled while paused' WHERE status='queued' AND trial_id IN (SELECT id FROM trials WHERE study_id=?)",(sid,))
            store.execute("UPDATE trials SET status='cancelled' WHERE study_id=? AND status IN ('queued','running')",(sid,))
        event(store, sid, "cancel_requested", {})
        return {"ok": True}

    @app.get("/api/studies/{sid}/export")
    def export(sid: str, format: Literal["json", "csv"] = "json"):
        data = get_study(store, sid)
        if format == "json":
            return Response(
                json.dumps(data, ensure_ascii=False, indent=2),
                media_type="application/json",
            )
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "study_id",
                "trial_id",
                "attempt_id",
                "seed",
                "status",
                "parameters",
                "metrics",
                "error",
            ]
        )
        for trial in data["trials"]:
            for r in trial["repeats"]:
                values = [
                    sid,
                    trial["id"],
                    r["id"],
                    r["seed"],
                    r["status"],
                    json.dumps(trial["params"]),
                    json.dumps(r["metrics"]),
                    r["error"] or "",
                ]
                writer.writerow(
                    [
                        (
                            "'" + v
                            if isinstance(v, str)
                            and v.lstrip().startswith(("=", "+", "-", "@"))
                            else v
                        )
                        for v in values
                    ]
                )
        return Response(
            "\ufeff" + output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{sid}.csv"'},
        )

    @app.get("/api/attempts/{aid}/artifacts/{name}")
    def artifact(
        aid: str,
        name: Literal[
            "config.json",
            "manifest.json",
            "metrics.json",
            "result-manifest.json",
            "stdout.log",
            "progress.jsonl",
            "daily_metrics.json",
        ],
    ):
        if not store.query("SELECT id FROM trial_attempts WHERE id=?", (aid,)):
            raise HTTPException(404, "执行记录不存在")
        path = store.root / "experiments" / aid / name
        if not path.is_file():
            raise HTTPException(404, "文件尚未生成")
        return FileResponse(path, filename=name)

    @app.post("/api/studies/{sid}/analyze")
    def analysis(sid: str, req: AnalysisSpec):
        return analyze(store, sid, req)

    @app.post("/api/analyses", status_code=202)
    def natural_analysis(req: AnalysisRequest, request: Request):
        get_study(store, req.study_id)
        return enqueue("analysis", req, request)
