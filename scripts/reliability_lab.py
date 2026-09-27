"""同主机真实 HTTP 压测；独立实验目录、两个 API 进程、不调用 GPU。"""
import asyncio
import collections
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import uuid
import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from loom.store import Store
from loom.schemas import RunRequest


async def measure(base,name,method,path,count,concurrency,body=None,headers=None,duration=None):
    semaphore=asyncio.Semaphore(concurrency)
    samples=[]
    statuses=collections.Counter()
    ids=[]
    async with httpx.AsyncClient(base_url=base,trust_env=False,timeout=30,limits=httpx.Limits(max_connections=concurrency,max_keepalive_connections=concurrency)) as client:
        async def one(i):
            async with semaphore:
                started=time.perf_counter()
                try:
                    response=await client.request(method,path,json=body,headers=headers)
                    statuses[str(response.status_code)]+=1
                    if response.status_code==202:ids.append(response.json()["run_id"])
                except httpx.HTTPError as exc:
                    statuses[type(exc).__name__]+=1
                samples.append(time.perf_counter()-started)
        started=time.perf_counter()
        if duration:
            async def runner():
                while time.perf_counter()-started < duration:
                    await one(0)
            await asyncio.gather(*(runner() for _ in range(concurrency)))
            count=len(samples)
        else:
            await asyncio.gather(*(one(i) for i in range(count)))
        elapsed=time.perf_counter()-started
    ordered=sorted(samples)
    successful=sum(v for k,v in statuses.items() if k.startswith("2"))
    return {"name":name,"requests":count,"concurrency":concurrency,"elapsed_seconds":round(elapsed,4),
            "all_response_qps":round(count/elapsed,2),"accepted_or_read_success_qps":round(successful/elapsed,2),
            "latency_p50_ms":round(ordered[int(len(ordered)*.5)]*1000,2),
            "latency_p95_ms":round(ordered[min(len(ordered)-1,int(len(ordered)*.95))]*1000,2),
            "status_counts":dict(statuses),"distinct_accepted_run_ids":len(set(ids))}


def main():
    folder=ROOT/"data"/"reliability"/(time.strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True)
    store=Store(folder)
    # API 验证固定研究元数据；不声称完成了论文推理。
    paper=store.add_paper("a"*64,"pressure-fixture.pdf","Synthetic pressure fixture",folder/"unused.pdf",1)
    store.execute("UPDATE papers SET status='ready' WHERE id=?",(paper["id"],))
    request=RunRequest(paper_ids=[paper["id"]],question="Pressure fixture only").model_dump()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    base=f"http://127.0.0.1:{port}"
    env={**os.environ,"LOOM_PRESSURE_DIR":str(folder),"PYTHONUTF8":"1"}
    log=(folder/"server.log").open("w",encoding="utf-8")
    process=subprocess.Popen([sys.executable,"-m","uvicorn","tests.reliability_support:pressure_app","--factory","--host","127.0.0.1","--port",str(port),"--workers","2","--no-access-log"],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    try:
        deadline=time.monotonic()+30
        while True:
            try:
                if httpx.get(base+"/api/status",trust_env=False,timeout=1).status_code==200:break
            except httpx.HTTPError:pass
            if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError("Pressure server did not start; inspect server.log")
            time.sleep(.2)
        reports=[]
        reports.append(asyncio.run(measure(base,"sustained_metadata_reads_30s","GET","/api/status",0,16,duration=30)))
        for count,concurrency in ((400,16),(800,64)):
            reports.append(asyncio.run(measure(base,"metadata_reads","GET","/api/status",count,concurrency)))
        reports.append(asyncio.run(measure(base,"duplicate_submission","POST","/api/runs",64,32,request,{"Idempotency-Key":"durable-request-1"})))
        assert reports[-1]["distinct_accepted_run_ids"]==1 and reports[-1]["status_counts"]=={"202":64}
        reports.append(asyncio.run(measure(base,"unique_submission_overload","POST","/api/runs",100,32,request)))
        rows=store.query("SELECT status,COUNT(*) n FROM runs GROUP BY status")
        assert rows==[{"status":"queued","n":12}],rows
        assert reports[-1]["status_counts"].get("202")==11 and reports[-1]["status_counts"].get("429")==89
        assert not any(k.startswith("5") or not k.isdigit() for r in reports for k in r["status_counts"])
        output={"timestamp":time.strftime("%Y-%m-%d %H:%M:%S %z"),"python":sys.version,"platform":platform.platform(),"api_workers":2,
                "database":"same-host SQLite WAL","gpu_calls":0,"workload":"finite closed-loop HTTP requests, synthetic paper metadata, no inference",
                "results":reports,"queue_after":rows,"source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"loom").glob("*.py")}}
        (folder/"report.json").write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding="utf-8")
        print(json.dumps(output,indent=2,ensure_ascii=False))
        print("REPORT",folder/"report.json")
    finally:
        # 只终止本脚本创建的测试服务树；不触及用户的 8010 服务。
        if process.poll() is None:
            subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True,check=False)
            process.wait(timeout=10)
        log.close()


if __name__=="__main__":main()
