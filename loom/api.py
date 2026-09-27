import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from .config import Settings, ROOT
from .store import Store, now, IdempotencyConflict
from .observability import IngressGuard, new_metrics, metrics_snapshot
from .schemas import RunRequest
from .documents import inspect_pdf
from .model import LocalModel
from .retrieval import Retriever
from .engine import ResearchEngine
from .platform import PlatformEngine
from .routing import ModelRouter
from .workspace_api import install_routes
from .worker import Worker, TERMINAL
from .export import markdown_report


class ReviewRequest(BaseModel):
    item_id: str = Field(max_length=100)
    verdict: Literal["accepted", "needs_revision"]
    note: str = Field(default="", max_length=2000)


def create_app(settings=None, model=None, start_worker=True):
    settings = settings or Settings()
    store = Store(settings.data_dir)
    local_model = model or LocalModel(settings)
    model = model or ModelRouter(store,settings,local_model)
    retriever = Retriever(store, settings)
    engine = PlatformEngine(store, settings, model, retriever)
    worker = Worker(store, settings, engine)

    @asynccontextmanager
    async def lifespan(app):
        if start_worker and settings.worker_enabled:
            worker.start()
        yield
        if start_worker and settings.worker_enabled:
            worker.stop()

    app = FastAPI(title="研途 Research Loom", version="0.5.0", lifespan=lifespan)
    app.state.store, app.state.worker, app.state.model = store, worker, model
    metrics = new_metrics()
    install_routes(app,store,settings,worker,model)

    @app.exception_handler(ValueError)
    async def invalid(request,exc):
        return JSONResponse(status_code=422,content={"detail":str(exc)[:1500]})

    @app.exception_handler(IdempotencyConflict)
    async def conflict(request,exc):
        return JSONResponse(status_code=409,content={"detail":str(exc)})

    @app.get("/api/metrics")
    def metrics_view():
        return {**metrics_snapshot(metrics),"tasks":store.query("SELECT status,COUNT(*) n FROM runs GROUP BY status"),"model_attempts":store.query("SELECT status,COUNT(*) n FROM model_attempts GROUP BY status"),"execution_slots":1,"worker_embedded":settings.worker_enabled and start_worker}

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(OverflowError)
    async def full(request, exc):
        return JSONResponse(status_code=429, content={"detail": str(exc)}, headers={"Retry-After": "15"})

    @app.middleware("http")
    async def local_origin(request, call_next):
        if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse(status_code=403, content={"detail": "仅接受同源页面提交"})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web" / "index.html")

    @app.get("/api/status")
    def status():
        return {"model": "Qwen3-VL-4B-Instruct", "local_only": not bool(settings.remote_url and settings.remote_model and settings.remote_key), "worker_embedded":settings.worker_enabled and start_worker,"active_execution":store.query("SELECT id,owner,epoch,lease_until,updated FROM runs WHERE status='running' LIMIT 1"),"model_available": settings.model_path is not None, "model_loaded": getattr(model, "model", None) is not None, "load_seconds": getattr(model, "load_seconds", None), "current_generation": getattr(model, "current_generation", None), "dense_enabled": settings.use_dense, "dense_error": retriever.dense_error, "queue_size": store.query("SELECT COUNT(*) n FROM runs WHERE status IN ('queued','running')")[0]["n"], "queue_limit": settings.queue_limit}

    @app.get("/api/papers")
    def papers():
        return store.papers()

    @app.post("/api/papers", status_code=202)
    async def upload(file: UploadFile = File(...)):
        filename = (file.filename or "paper.pdf").replace("\\", "/").split("/")[-1]
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(415, "请上传 PDF 格式论文")
        uploads = store.root / "uploads"
        uploads.mkdir(exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="incoming_", suffix=".pdf", dir=uploads)
        path = Path(name)
        digest, size = hashlib.sha256(), 0
        try:
            with os.fdopen(fd, "wb") as stream:
                while chunk := await file.read(1024*1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, "论文不能超过 40 MB")
                    digest.update(chunk)
                    stream.write(chunk)
            if size < 5:
                raise HTTPException(422, "PDF 文件为空或损坏")
            try:
                page_count, title = await asyncio.to_thread(inspect_pdf, path, settings.max_pages)
            except Exception as exc:
                raise HTTPException(422, f"无法读取 PDF：{exc}") from exc
            if title == path.stem:
                title = Path(filename).stem
            sha = digest.hexdigest()
            existing = store.query("SELECT id FROM papers WHERE sha=?", (sha,))
            if existing:
                paper = store.paper(existing[0]["id"])
                active = store.query("SELECT id,request FROM runs WHERE kind='ingest' AND status IN ('queued','running')")
                run_id = next((r["id"] for r in active if json.loads(r["request"])["paper_id"] == paper["id"]), None)
                if paper["status"] != "ready" and run_id is None:
                    run_id = worker.enqueue("ingest", {"paper_id": paper["id"]})
                return {"paper": public_paper(paper), "run_id": run_id, "duplicate": True}
            final_path = uploads / f"{sha}.pdf"
            try:
                os.link(path, final_path)  # 同卷原子发布；并发重复上传不能替换正在读取的 PDF。
            except FileExistsError:
                pass
            paper = store.add_paper(sha, filename[:240], title, final_path, page_count)
            run_id = worker.enqueue("ingest", {"paper_id": paper["id"]})
            return {"paper": public_paper(paper), "run_id": run_id, "duplicate": False}
        finally:
            await file.close()
            path.unlink(missing_ok=True)

    def public_paper(paper):
        return {k: v for k, v in paper.items() if k != "path"}

    @app.get("/api/papers/{paper_id}")
    def paper_details(paper_id: str):
        paper = public_paper(store.paper(paper_id))
        pages = store.query("SELECT page,captions FROM pages WHERE paper_id=? ORDER BY page", (paper_id,))
        for page in pages:
            page["captions"] = json.loads(page["captions"])
        return {**paper, "page_index": pages}

    @app.get("/api/papers/{paper_id}/pdf")
    def pdf(paper_id: str):
        paper = store.paper(paper_id)
        return FileResponse(paper["path"], media_type="application/pdf", filename=paper["filename"], content_disposition_type="inline")

    @app.post("/api/papers/{paper_id}/resume", status_code=202)
    def resume_paper(paper_id: str):
        paper = store.paper(paper_id)
        if paper["status"] == "ready":
            raise HTTPException(409, "论文已解析完成")
        active = store.query("SELECT id,request FROM runs WHERE kind='ingest' AND status IN ('queued','running')")
        existing = next((r["id"] for r in active if json.loads(r["request"])["paper_id"] == paper_id), None)
        return {"run_id": existing or worker.enqueue("ingest", {"paper_id": paper_id})}

    @app.get("/api/papers/{paper_id}/pages/{page}/image")
    def page_image(paper_id: str, page: int):
        store.paper(paper_id)
        rows = store.query("SELECT image FROM pages WHERE paper_id=? AND page=?", (paper_id, page))
        if not rows:
            raise HTTPException(404, "页面尚未解析")
        return FileResponse(rows[0]["image"], media_type="image/png")

    @app.get("/api/evidence/{evidence_id}")
    def evidence(evidence_id: str):
        rows = store.get_evidence(ids=[evidence_id])
        if not rows:
            raise HTTPException(404, "证据不存在")
        return rows[0]

    @app.post("/api/runs", status_code=202)
    def create_run(req: RunRequest, request: Request):
        key = request.headers.get("idempotency-key")
        if key is not None and (not key.strip() or len(key)>128):
            raise HTTPException(422,"Idempotency-Key 应为 1–128 字符")
        for pid in req.paper_ids:
            paper = store.paper(pid)
            if paper["status"] != "ready":
                raise HTTPException(409, "请等待论文解析完成")
        if req.mode == "figure":
            if req.page is None or req.page > store.paper(req.paper_ids[0])["pages"]:
                raise HTTPException(422, "读图任务需要所选论文的有效页码")
        if req.parent_run_id:
            parent = store.run(req.parent_run_id)
            if parent["kind"] != "research" or parent["status"] != "completed":
                raise HTTPException(409, "只能基于已完成的研究任务继续讨论")
            if not set(parent["request"]["paper_ids"]) <= set(req.paper_ids):
                raise HTTPException(422, "继续讨论需要包含上一轮使用的论文")
        run_id = worker.enqueue("research", req.model_dump(), key=key)
        return {"run_id": run_id}

    @app.get("/api/runs")
    def runs():
        rows = store.query("SELECT id,kind,status,request,created,updated,error FROM runs ORDER BY created DESC LIMIT 100")
        for row in rows:
            row["request"] = json.loads(row["request"])
        return rows

    @app.get("/api/runs/{run_id}")
    def run_details(run_id: str):
        run = store.run(run_id)
        run["reviews"] = store.query("SELECT * FROM reviews WHERE run_id=? ORDER BY id", (run_id,))
        return run

    @app.get("/api/runs/{run_id}/trace")
    def trace(run_id: str, include_payloads: bool = False):
        run = store.run(run_id)
        columns = "*" if include_payloads else "id,run_id,epoch,stage,input_hash,status,usage,error,created,updated"
        attempts = store.query(f"SELECT {columns} FROM model_attempts WHERE run_id=? ORDER BY created,id",(run_id,))
        for attempt in attempts:
            for name in ("input","usage"):
                if attempt.get(name):
                    attempt[name] = json.loads(attempt[name])
        events = store.query("SELECT * FROM events WHERE run_id=? ORDER BY seq",(run_id,))
        for item in events:
            item["payload"] = json.loads(item["payload"])
        return {"run_id":run_id,"status":run["status"],"epoch":run["epoch"],"deadline":run["deadline"],"not_before":run["not_before"],"recoveries":run["recoveries"],"attempts":attempts,"events":events,"unknown_usage_attempts":sum(a["usage"] is None for a in attempts),"coverage":"本版本执行记录" if attempts else "没有模型调用账本：可能是旧任务、未调用模型或尚未开始，不能推断调用次数为零","guarantee":"已记录回答可回放；未知调用可能重复；单次提交不等于模型只执行一次"}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        store.request_cancel(run_id)
        worker.wake.set()
        return {"ok": True}

    @app.post("/api/runs/{run_id}/retry")
    def retry(run_id: str):
        store.retry(run_id,settings.queue_limit,settings.task_timeout)
        worker.wake.set()
        return {"run_id": run_id}

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request, after: int = 0):
        store.run(run_id)
        try:
            after = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            raise HTTPException(422, "Last-Event-ID 必须为数字")

        async def stream():
            cursor, heartbeat = after, 0
            while not await request.is_disconnected():
                rows = await asyncio.to_thread(store.events, run_id, cursor)
                for event in rows:
                    cursor = event["seq"]
                    yield f"id: {cursor}\nevent: progress\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                state = await asyncio.to_thread(store.run, run_id)
                if state["status"] in TERMINAL and len(rows) < 100:
                    yield "event: done\ndata: {}\n\n"
                    break
                heartbeat += 1
                if heartbeat % 20 == 0:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(.5)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/runs/{run_id}/reviews")
    def review(run_id: str, req: ReviewRequest):
        run = store.run(run_id)
        result = run.get("result") or {}
        ids = {item["id"] for item in result.get("findings", []) + result.get("directions", [])}
        if req.item_id not in ids:
            raise HTTPException(422, "复核条目不属于该任务")
        store.execute("INSERT INTO reviews(run_id,item_id,verdict,note,created) VALUES (?,?,?,?,?)", (run_id, req.item_id, req.verdict, req.note, now()))
        return {"ok": True}

    @app.get("/api/runs/{run_id}/export")
    def export(run_id: str, format: Literal["markdown", "json"] = "markdown"):
        run = store.run(run_id)
        if run["kind"] != "research" or run["status"] != "completed":
            raise HTTPException(409, "研究任务完成后可导出")
        run["reviews"] = store.query("SELECT * FROM reviews WHERE run_id=? ORDER BY id", (run_id,))
        body = json.dumps(run, ensure_ascii=False, indent=2) if format == "json" else markdown_report(run)
        suffix = "json" if format == "json" else "md"
        return Response(body, media_type="application/json" if format == "json" else "text/markdown", headers={"Content-Disposition": f'attachment; filename="research-{run_id}.{suffix}"'})

    app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=list(settings.allowed_hosts))
    app.add_middleware(IngressGuard,settings=settings,metrics=metrics)
    return app
