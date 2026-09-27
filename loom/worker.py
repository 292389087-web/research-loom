import logging
import threading
import time
from datetime import datetime
from .documents import ingest
from .store import LeaseLost, uid
from .gpu import ResourceBusy

TERMINAL = {"completed", "failed", "cancelled"}


class Worker:
    """轮询派发、租约心跳与 epoch 隔离；API 和执行进程可独立运行。"""
    def __init__(self, store, settings, engine):
        self.store, self.settings, self.engine = store, settings, engine
        self.owner = uid("worker")
        self.stalled = threading.Event()
        self.shutdown = threading.Event()
        self.wake = threading.Event()
        self.thread = None
        self.submit_lock = threading.Lock()

    def start(self):
        self.thread = threading.Thread(target=self.loop, daemon=True, name="loom-worker")
        self.thread.start()

    def stop(self):
        self.shutdown.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=8)

    def enqueue(self, kind, request, key=None):
        if self.settings.demo_mode:
            raise ValueError('CPU 演示模式只提供确定性数据分析；模型任务请正常启动服务')
        run_id = self.store.create_run(kind, request, self.settings.queue_limit, key, self.settings.task_timeout)
        self.wake.set()
        return run_id

    def loop(self):
        while not self.shutdown.is_set():
            try:
                acquired = self.store.claim(self.owner,self.settings.lease_seconds,max_recoveries=self.settings.max_recoveries)
                if acquired:
                    self.execute(acquired[0],acquired[1])
                    continue
            except Exception:
                logging.exception("Dispatcher failed; retry polling")
            self.wake.wait(.5)
            self.wake.clear()

    def execute(self, run_id, epoch=None):
        if epoch is None:
            acquired = self.store.claim(self.owner,self.settings.lease_seconds,run_id,self.settings.max_recoveries)
            if not acquired:
                return
            _, epoch = acquired
        stopped, lost = threading.Event(), threading.Event()
        def heartbeat():
            while not stopped.wait(max(.05,self.settings.lease_seconds/3)):
                try:
                    state = self.store.run(run_id)
                    last_progress = datetime.fromisoformat(state["updated"]).timestamp()
                    if time.time()-last_progress > self.settings.stall_timeout or state["deadline"] and time.time()>=state["deadline"]:
                        logging.error("Worker made no progress or exceeded deadline: %s",run_id)
                        self.stalled.set()
                        lost.set()
                        self.shutdown.set()
                        return  # 不再续租；独立 worker 主进程终止阻塞计算并由监督进程重启。
                    if not self.store.heartbeat(run_id,self.owner,epoch,self.settings.lease_seconds):
                        lost.set()
                        return
                except Exception:
                    logging.exception("Heartbeat failed for %s",run_id)
                    lost.set()
                    return
        monitor = threading.Thread(target=heartbeat,daemon=True,name="loom-heartbeat")
        monitor.start()
        try:
            with self.store.leased(run_id,self.owner,epoch):
                run = self.store.run(run_id)
                def cancelled():
                    if lost.is_set():
                        raise LeaseLost("执行权已丢失")
                    state = self.store.run(run_id)
                    if self.shutdown.is_set() or state["cancel"]:
                        raise InterruptedError("任务已取消或服务正在停止")
                    if state["deadline"] and time.time() >= state["deadline"]:
                        raise TimeoutError("任务超过持久化总期限")
                def emit(kind,payload):
                    self.store.event(run_id,kind,{**payload,"epoch":epoch,"owner":self.owner})
                try:
                    cancelled()
                    result = ingest(self.store,run["request"]["paper_id"],emit,cancelled) if run["kind"]=="ingest" else self.engine.run(run_id,emit,cancelled)
                    cancelled()
                    self.store.finish(run_id,"completed",result=result)
                except LeaseLost:
                    raise
                except ResourceBusy as exc:
                    self.store.finish(run_id,"queued",error=str(exc),delay=3)
                except InterruptedError:
                    restarting = self.shutdown.is_set() and not self.store.run(run_id)["cancel"]
                    self.store.finish(run_id,"queued" if restarting else "cancelled")
                except Exception as exc:
                    logging.exception("Task %s epoch %s failed",run_id,epoch)
                    state = self.store.run(run_id)
                    transient = isinstance(exc,(TimeoutError,ConnectionError))
                    retry = transient and state["recoveries"] < self.settings.max_recoveries and (not state["deadline"] or time.time()+self.settings.retry_delay < state["deadline"])
                    if retry:
                        self.store.execute("UPDATE runs SET recoveries=recoveries+1 WHERE id=?",(run_id,))
                        delay = self.settings.retry_delay*(2**state["recoveries"])
                        self.store.finish(run_id,"queued",error=f"暂时错误，按持久计划重试：{str(exc)[:1500]}",delay=delay)
                    else:
                        if run["kind"]=="ingest":
                            self.store.execute("UPDATE papers SET status='failed' WHERE id=?",(run["request"]["paper_id"],))
                        self.store.finish(run_id,"failed",error=f"{type(exc).__name__}: {str(exc)[:1800]}")
        except LeaseLost:
            logging.warning("Discarded stale execution %s epoch=%s",run_id,epoch)
        finally:
            stopped.set()
            monitor.join(timeout=1)
