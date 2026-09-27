"""真实本地模型验收：上传 PDF 并提交研究，原始结果写入本项目 data/evals。"""
import argparse
import json
from pathlib import Path
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True, help="本地研究论文 PDF 路径")
    parser.add_argument("--mode", choices=["understand","plan","question","figure"], default="plan")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    args = parser.parse_args()
    out = ROOT / "data" / "evals"
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.base_url, timeout=90) as client:
        with args.pdf.open("rb") as stream:
            response = client.post("/api/papers", files={"file": (args.pdf.name,stream,"application/pdf")})
        response.raise_for_status()
        uploaded = response.json()
        if uploaded["run_id"]:
            wait(client, uploaded["run_id"])
        question = "请结合框架图解释方法如何解决问题，整理作者采用的评价指标、基线与消融。"
        if args.mode == "plan":
            question += "从方法机制与边界条件出发，提出可证伪的研究切入点；算力只作为约束，不默认轻量化。"
        body = {"paper_ids":[uploaded["paper"]["id"]],"mode":args.mode,"question":question,"profile":{"interests":"人工智能方法与应用","experience":"熟悉 PyTorch 与强化学习","compute":"RTX 4070 Ti SUPER 16GB，单 GPU","timeline":"8 周","goal":"理解论文，设计可证伪的后续研究"}}
        if args.mode == "figure":
            body["page"] = 12
            body["bbox"] = [0,0,1,.65]
        response = client.post("/api/runs", json=body)
        response.raise_for_status()
        run_id = response.json()["run_id"]
        print("RESEARCH_RUN", run_id, flush=True)
        result = wait(client,run_id)
        (out/f"{run_id}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print("RESULT", json.dumps(result["result"].get("quality",{}),ensure_ascii=False),flush=True)


def wait(client,run_id):
    last = ""
    deadline = time.monotonic()+1800
    while time.monotonic()<deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        value=f"{run_id} {run['status']} {list((run.get('checkpoint') or {}).keys())}"
        if value != last:
            print(value,flush=True); last=value
        if run["status"] in ("completed","failed","cancelled"):
            if run["status"] != "completed":
                raise RuntimeError(run.get("error") or run["status"])
            return run
        time.sleep(3)
    raise TimeoutError("验收任务超时；可在界面检查持久化状态")


if __name__ == "__main__":
    main()
