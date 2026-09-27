"""有界请求观测与入口保护；指标为单进程口径，模型吞吐另计。"""
import collections
import hmac
import json
import tempfile
import time
import uuid
from starlette.concurrency import run_in_threadpool


class IngressGuard:
    def __init__(self,app,settings,metrics):
        self.app,self.settings,self.metrics = app,settings,metrics
        self.active = self.sse = self.uploads = 0

    async def __call__(self,scope,receive,send):
        if scope["type"] != "http":
            return await self.app(scope,receive,send)
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        headers = dict(scope["headers"])
        path = scope["path"]
        is_sse = path.startswith("/api/runs/") and path.endswith("/events")
        is_upload = path=="/api/papers" and scope["method"]=="POST"
        status, admitted, sse_admitted, upload_admitted = 500, False, False, False
        body = None
        async def tracked(message):
            nonlocal status
            if message["type"]=="http.response.start":
                status = message["status"]
                message["headers"] = list(message.get("headers",[]))+[(b"x-request-id",request_id.encode()),(b"x-content-type-options",b"nosniff"),(b"referrer-policy",b"no-referrer"),(b"x-frame-options",b"DENY"),(b"content-security-policy",b"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")]
            await send(message)
        async def reject(code,detail):
            await tracked({"type":"http.response.start","status":code,"headers":[(b"content-type",b"application/json"),(b"retry-after",b"2")]})
            await tracked({"type":"http.response.body","body":json.dumps({"detail":detail,"request_id":request_id},ensure_ascii=False).encode()})
        try:
            if path.startswith("/api/") and self.settings.api_token:
                expected = ("Bearer "+self.settings.api_token).encode()
                if not hmac.compare_digest(headers.get(b"authorization",b""),expected):
                    return await reject(401,"需要有效的 API Bearer token")
            if self.active >= self.settings.ingress_limit or is_sse and self.sse >= self.settings.sse_limit or is_upload and self.uploads >= 2:
                return await reject(429,"并发连接达到本进程上限，请退避重试")
            if is_upload:
                self.uploads += 1
                upload_admitted = True
            maximum = self.settings.max_upload_bytes+1024*1024 if path=="/api/papers" else 65536
            try:
                length = int(headers.get(b"content-length",b"0"))
            except ValueError:
                return await reject(400,"无效 Content-Length")
            if length < 0:
                return await reject(400,"无效 Content-Length")
            if length > maximum:
                return await reject(413,"请求体超过限制")
            # 先有界分块接收，覆盖无 Content-Length 的传输。超过 1 MiB 自动落盘，
            # 避免上传文件完整常驻内存；之后再分块回放给 Starlette multipart 解析器。
            body_size = 0
            if scope["method"] in {"POST","PUT","PATCH","DELETE"}:
                self.active += 1
                admitted = True
                body = tempfile.SpooledTemporaryFile(max_size=1024*1024,mode="w+b")
                while True:
                    import asyncio
                    try:
                        message = await asyncio.wait_for(receive(),timeout=30)
                    except asyncio.TimeoutError:
                        return await reject(408,"请求体接收超时")
                    if message["type"]=="http.disconnect":
                        return
                    chunk = message.get("body",b"")
                    body_size += len(chunk)
                    if body_size>maximum:
                        return await reject(413,"请求体超过限制")
                    # 提前 rollover，避免大块先复制进 BytesIO；磁盘 I/O 不阻塞事件循环。
                    if body_size > 1024*1024:
                        await run_in_threadpool(body.rollover)
                    await run_in_threadpool(body.write, chunk)
                    if not message.get("more_body",False):
                        break
                await run_in_threadpool(body.seek, 0)
            if not admitted:
                self.active += 1
                admitted = True
            if is_sse:
                self.sse += 1
                sse_admitted = True
            delivered = False
            async def replay():
                nonlocal delivered
                if body is not None and not delivered:
                    chunk = await run_in_threadpool(body.read, 64*1024)
                    more = body.tell() < body_size
                    if not more:
                        delivered = True
                    return {"type":"http.request","body":chunk,"more_body":more}
                return await receive()
            await self.app(scope,replay,tracked)
        finally:
            if body is not None:
                await run_in_threadpool(body.close)
            if admitted:
                self.active -= 1
                if sse_admitted:
                    self.sse -= 1
            if upload_admitted:
                self.uploads -= 1
            route = getattr(scope.get("route"),"path","unmatched")
            self.metrics["requests"][(scope["method"],route,status)] += 1
            self.metrics["latencies"].append(time.monotonic()-started)
            self.metrics["active"] = self.active


def new_metrics():
    return {"requests":collections.Counter(),"latencies":collections.deque(maxlen=1000),"active":0}


def metrics_snapshot(metrics):
    values = sorted(metrics["latencies"])
    return {"scope":"single_api_process","requests":[{"method":k[0],"route":k[1],"status":k[2],"count":v} for k,v in metrics["requests"].items()],
            "active":metrics["active"],"recent_sample_size":len(values),
            "recent_request_p95_seconds":values[min(len(values)-1,int(len(values)*.95))] if values else None}
