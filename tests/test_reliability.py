import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
import sqlite3
import time
import pytest
from fastapi.testclient import TestClient
from loom.api import create_app
from loom.config import Settings
from loom.store import Store,LeaseLost,IdempotencyConflict
from loom.worker import Worker
from loom.observability import IngressGuard,new_metrics
from tests.reliability_support import child_worker,submit_many,ChangingModel


def finish_process(process):
    process.join(18)
    if process.is_alive():
        process.terminate()
        process.join(3)
        pytest.fail("实验子进程未在期限内退出")


@pytest.mark.parametrize("point,physical,chosen",[
    ("response_before_record",2,"proposal-2"),
    ("response_recorded",1,"proposal-1"),
    ("stage_committed",1,"proposal-1"),
])
def test_killed_process_and_automatic_resume(tmp_path,point,physical,chosen):
    store=Store(tmp_path)
    rid=store.create_run("research",{},ttl=60)
    context=mp.get_context("spawn")
    first=context.Process(target=child_worker,args=(str(tmp_path),point))
    first.start();finish_process(first)
    assert first.exitcode==83
    # 新 worker 自行扫描过期任务，无人工 retry/execute。
    second=context.Process(target=child_worker,args=(str(tmp_path),))
    second.start();finish_process(second)
    assert second.exitcode==0
    run=store.run(rid)
    assert run["status"]=="completed" and run["result"]["decision"]==chosen
    with sqlite3.connect(tmp_path/"physical-model.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]==physical
    assert len([e for e in store.events(rid) if e["type"]=="completed"])==1
    assert run["epoch"]==2 and run["recoveries"]==1
    if physical==2:
        unknown=store.query("SELECT * FROM model_attempts WHERE status='unknown'")
        assert len(unknown)==1 and unknown[0]["usage"] is None


@pytest.mark.parametrize("key,expected,rejections",[("lost-response-key",1,0),(None,7,25)])
def test_multiprocess_submission_and_capacity(tmp_path,key,expected,rejections):
    store=Store(tmp_path)
    ctx=mp.get_context("spawn");start=ctx.Event();result=ctx.Queue()
    children=[ctx.Process(target=submit_many,args=(str(tmp_path),start,result,key,7)) for _ in range(4)]
    for child in children:child.start()
    start.set()
    reports=[result.get(timeout=15) for _ in children]
    for child in children:
        finish_process(child);assert child.exitcode==0
    assert len(store.query("SELECT id FROM runs"))==expected
    assert sum(x[1] for x in reports)==rejections
    if key:
        assert len({rid for ids,_ in reports for rid in ids})==1
        with pytest.raises(IdempotencyConflict):store.create_run("research",{"question":"changed"},key=key)
    assert len(store.query("SELECT seq FROM events WHERE type='queued'"))==expected


def test_expired_writer_cannot_commit_or_renew(tmp_path):
    store=Store(tmp_path);rid=store.create_run("research",{})
    _,epoch=store.claim("old",30)
    store.execute("UPDATE runs SET lease_until=0 WHERE id=?",(rid,))
    assert store.claim("new",30)==(rid,epoch+1)
    assert not store.heartbeat(rid,"old",epoch,30)
    with store.leased(rid,"old",epoch):
        for operation in (lambda:store.update_run(rid,checkpoint={"bad":1}),lambda:store.event(rid,"bad",{}),lambda:store.finish(rid,"completed",result={"bad":1})):
            with pytest.raises(LeaseLost):operation()
    with store.leased(rid,"new",epoch+1):store.finish(rid,"completed",result={"ok":1})
    assert store.run(rid)["result"]=={"ok":1}
    assert not store.query("SELECT * FROM events WHERE type='bad'")


def test_concurrent_claim_has_single_winner(tmp_path):
    store=Store(tmp_path)
    for _ in range(5):store.create_run("research",{})
    with ThreadPoolExecutor(max_workers=12) as pool:
        results=list(pool.map(lambda i:Store(tmp_path).claim(str(i),30),range(12)))
    assert len([x for x in results if x])==1


def test_persisted_backoff_budget_and_deadline(tmp_path):
    store=Store(tmp_path);settings=Settings(data_dir=tmp_path,retry_delay=60,max_recoveries=1)
    class TimeoutEngine:
        def run(self,*args):raise TimeoutError("injected transient failure")
    worker=Worker(store,settings,TimeoutEngine());rid=worker.enqueue("research",{})
    worker.execute(rid)
    row=store.run(rid)
    assert row["status"]=="queued" and row["not_before"]>time.time()+50
    reopened=Store(tmp_path)
    assert reopened.claim("other",30) is None
    assert reopened.run(rid)["deadline"]==row["deadline"]
    store.execute("UPDATE runs SET not_before=0 WHERE id=?",(rid,))
    worker.execute(rid)
    assert store.run(rid)["status"]=="failed" and store.run(rid)["recoveries"]==1
    expiry=worker.enqueue("research",{})
    store.execute("UPDATE runs SET deadline=1 WHERE id=?",(expiry,))
    assert store.claim("deadline",30) is None
    assert store.run(expiry)["status"]=="failed"


def test_cancellation_beats_late_success_and_terminal_event_is_atomic(tmp_path):
    store=Store(tmp_path);rid=store.create_run("research",{});_,epoch=store.claim("w",30)
    store.request_cancel(rid)
    with store.leased(rid,"w",epoch):store.finish(rid,"completed",result={"late":1})
    assert store.run(rid)["status"]=="cancelled" and store.run(rid)["result"] is None
    second=store.create_run("research",{});_,epoch=store.claim("w",30)
    store.execute("CREATE TRIGGER fail_event BEFORE INSERT ON events WHEN NEW.type='completed' BEGIN SELECT RAISE(ABORT,'injected'); END")
    with store.leased(second,"w",epoch),pytest.raises(sqlite3.IntegrityError):
        store.finish(second,"completed",result={"x":1})
    assert store.run(second)["status"]=="running" and store.run(second)["result"] is None


def test_auth_host_origin_body_limits_and_trace(tmp_path):
    settings=Settings(data_dir=tmp_path,use_dense=False,api_token="local-test-secret",allowed_hosts=("testserver",))
    app=create_app(settings,model=ChangingModel(tmp_path),start_worker=False)
    with TestClient(app) as c:
        assert c.get("/api/runs").status_code==401
        headers={"authorization":"Bearer local-test-secret"}
        assert c.get("/api/runs",headers={**headers,"host":"evil.example"}).status_code==400
        assert c.post("/api/runs",json={},headers={**headers,"origin":"https://evil.example"}).status_code==403
        assert c.post("/api/runs",content=b"x"*65537,headers=headers).status_code==413
        rid=app.state.store.create_run("research",{})
        trace=c.get(f"/api/runs/{rid}/trace",headers=headers)
        assert trace.status_code==200 and trace.headers["x-request-id"]
        assert trace.json()["events"][0]["type"]=="queued"
        assert c.get("/api/metrics",headers=headers).status_code==200


def test_sse_connection_backpressure_and_disconnect_release():
    async def scenario():
        entered,release=asyncio.Event(),asyncio.Event()
        async def held(scope,receive,send):
            entered.set();await release.wait()
            await send({"type":"http.response.start","status":200,"headers":[]})
            await send({"type":"http.response.body","body":b""})
        guard=IngressGuard(held,Settings(sse_limit=1),new_metrics())
        scope={"type":"http","path":"/api/runs/r/events","method":"GET","headers":[]}
        async def receive():return {"type":"http.request","body":b""}
        async def ignore(message):pass
        first=asyncio.create_task(guard(scope,receive,ignore));await entered.wait()
        messages=[]
        async def collect(message):messages.append(message)
        await guard(scope,receive,collect)
        assert messages[0]["status"]==429
        release.set();await first
        assert guard.sse==0 and guard.active==0
    asyncio.run(scenario())


def test_alive_but_stalled_worker_stops_renewing_lease(tmp_path):
    import threading
    store=Store(tmp_path)
    release=threading.Event()
    class HungEngine:
        def run(self,*args):
            release.wait(5)
            return {"late":True}
    settings=Settings(data_dir=tmp_path,lease_seconds=.3,stall_timeout=.15)
    worker=Worker(store,settings,HungEngine());rid=worker.enqueue("research",{})
    worker.start()
    try:
        assert worker.stalled.wait(3)
        until=store.run(rid)["lease_until"]
        time.sleep(max(0,until-time.time())+.05)
        assert store.claim("replacement",30)==(rid,2)
        release.set()
        worker.stop()
        assert store.run(rid)["owner"]=="replacement" and store.run(rid)["result"] is None
    finally:
        release.set();worker.stop()


def test_chunked_body_limit_cannot_be_bypassed():
    async def scenario():
        called=False;messages=[]
        async def forbidden(*args):
            nonlocal called
            called=True
        guard=IngressGuard(forbidden,Settings(),new_metrics())
        async def receive():return {"type":"http.request","body":b"x"*40000,"more_body":True}
        async def send(message):messages.append(message)
        await guard({"type":"http","path":"/api/runs","method":"POST","headers":[]},receive,send)
        assert messages[0]["status"]==413 and not called and guard.active==0
    asyncio.run(scenario())


def test_response_lost_after_api_commit_reuses_same_task(tmp_path):
    settings=Settings(data_dir=tmp_path,use_dense=False,allowed_hosts=("testserver",))
    app=create_app(settings,model=ChangingModel(tmp_path),start_worker=False)
    store=app.state.store
    paper=store.add_paper("b"*64,"fixture.pdf","fixture",tmp_path/"unused.pdf",1)
    store.execute("UPDATE papers SET status='ready' WHERE id=?",(paper["id"],))
    class DropOnce:
        dropped=False
        async def __call__(self,scope,receive,send):
            async def interrupted(message):
                if scope["type"]=="http" and scope["path"]=="/api/runs" and message["type"]=="http.response.start" and message["status"]==202 and not self.dropped:
                    self.dropped=True
                    raise ConnectionError("injected response loss after durable commit")
                await send(message)
            await app(scope,receive,interrupted)
    wrapper=DropOnce()
    with TestClient(wrapper) as client:
        body={"paper_ids":[paper["id"]]};headers={"Idempotency-Key":"lost-ack"}
        with pytest.raises(ConnectionError):client.post("/api/runs",json=body,headers=headers)
        rows=store.query("SELECT id FROM runs")
        assert len(rows)==1
        response=client.post("/api/runs",json=body,headers=headers)
        assert response.status_code==202 and response.json()["run_id"]==rows[0]["id"]
        assert len(store.query("SELECT * FROM runs"))==1
