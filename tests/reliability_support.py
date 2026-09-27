"""仅供故障实验的独立、不确定模型；生产入口不导入此模块。"""
import json
import os
from pathlib import Path
import sqlite3
import time
from pydantic import BaseModel
from loom.config import Settings
from loom.execution import StageExecutor
from loom.store import Store
from loom.worker import Worker


class Decision(BaseModel):
    decision: str


class ChangingModel:
    def __init__(self,root):
        self.root=Path(root)

    def generate(self,*args,**kwargs):
        # 独立于业务数据库：像外部推理服务一样记录实际调用，重试回答不同。
        with sqlite3.connect(self.root/"physical-model.sqlite3") as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY AUTOINCREMENT)")
            number=db.execute("INSERT INTO calls DEFAULT VALUES").lastrowid
        return json.dumps({"decision":f"proposal-{number}"}),{"input_tokens":10,"output_tokens":5,"seconds":.001}


class ProbeEngine:
    def __init__(self,store,settings,fault=None):
        self.store,self.settings,self.fault=store,settings,fault

    def run(self,run_id,emit,cancelled):
        cp=self.store.run(run_id)["checkpoint"]
        cp.setdefault("calls",0)
        cp.setdefault("usage",[])
        executor=StageExecutor(self.store,self.settings,ChangingModel(self.store.root),self.fault)
        return executor.generate(run_id,cp,"decision","test only","choose a proposal",Decision,"fault-probe-v1",emit,cancelled)


def child_worker(root,fault_point=None):
    store=Store(Path(root))
    settings=Settings(data_dir=Path(root),use_dense=False,lease_seconds=.6,task_timeout=60)
    def fault(point):
        if point==fault_point:
            os._exit(83)  # 只在本实验子进程中模拟强制退出；无 HTTP 开关。
    worker=Worker(store,settings,ProbeEngine(store,settings,fault))
    worker.start()
    deadline=time.monotonic()+12
    while time.monotonic()<deadline:
        rows=store.query("SELECT status FROM runs")
        if rows and all(r["status"] in {"completed","failed","cancelled"} for r in rows):
            worker.stop()
            return
        time.sleep(.03)
    worker.stop()
    raise RuntimeError("Fault probe worker deadline exceeded")


def submit_many(root,start,result,key,limit,count=8):
    store=Store(Path(root))
    start.wait(10)
    accepted,rejected=[],0
    for i in range(count):
        try:
            accepted.append(store.create_run("research",{"question":"same"},limit=limit,key=key))
        except OverflowError:
            rejected+=1
    result.put((accepted,rejected))


def pressure_app():
    """仅用于本地 HTTP 压测，独立数据目录且不运行 GPU。"""
    from loom.api import create_app
    settings=Settings(data_dir=Path(os.environ["LOOM_PRESSURE_DIR"]),use_dense=False,worker_enabled=False)
    return create_app(settings=settings,model=ChangingModel(settings.data_dir),start_worker=False)
