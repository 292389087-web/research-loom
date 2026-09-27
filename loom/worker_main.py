"""独立执行进程：python -m loom.worker_main。仅一个进程加载本机 GPU。"""
import logging
import time
import os
import sys
import subprocess
import argparse
from .config import Settings
from .store import Store
from .model import LocalModel
from .retrieval import Retriever
from .platform import PlatformEngine
from .routing import ModelRouter
from .worker import Worker


def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()
    store = Store(settings.data_dir)
    engine = PlatformEngine(store,settings,ModelRouter(store,settings,LocalModel(settings)),Retriever(store,settings))
    worker = Worker(store,settings,engine)
    worker.start()
    try:
        while True:
            if worker.stalled.is_set():
                logging.error("Terminating stalled worker process to release model resources")
                os._exit(75)
            time.sleep(.5)
    except KeyboardInterrupt:
        worker.stop()


def supervise():
    """同主机监督器，只管理它创建的一个执行进程；旧进程退出后才启动替代者。"""
    restarts=0
    while True:
        child=subprocess.Popen([sys.executable,"-m","loom.worker_main"],creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            code=child.wait()
        except KeyboardInterrupt:
            child.terminate()
            child.wait(timeout=10)
            return
        if code==0 or restarts>=5:
            return
        restarts+=1
        time.sleep(min(2**restarts,30))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--supervise",action="store_true")
    args=parser.parse_args()
    supervise() if args.supervise else main()
