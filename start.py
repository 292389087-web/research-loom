"""python start.py --port 8010；默认仅允许本机连接。"""
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument('--demo',action='store_true',help='独立数据目录的 CPU 数据分析演示，不执行模型或训练')
    parser.add_argument("--api-only",action="store_true",help="API 不加载模型执行任务，另启 python -m loom.worker_main")
    parser.add_argument("--workers",type=int,default=1,help="仅 --api-only 可使用多个 API 进程")
    args = parser.parse_args()
    if args.demo:
        from pathlib import Path
        args.api_only=True
        os.environ['LOOM_DEMO']='1'
        os.environ['LOOM_DATA_DIR']=str(Path(__file__).resolve().parent/'data'/'demo')
        for key in ('URL','MODEL','KEY'):
            os.environ['LOOM_REMOTE_'+key]=''
    if args.workers<1 or args.workers>1 and not args.api_only:
        parser.error("多个 API 进程必须配合 --api-only，避免复制模型")
    if args.api_only:
        os.environ["LOOM_WORKER"] = "0"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import uvicorn
    uvicorn.run("loom.api:create_app", factory=True,host="127.0.0.1", port=args.port, log_level="info",workers=args.workers)
