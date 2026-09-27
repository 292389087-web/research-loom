"""真实本地 Qwen 引用推荐与调用后显存释放验收；不调用远程 API。"""

import json
import subprocess
import time
from pathlib import Path
import httpx

root = Path(__file__).resolve().parents[1]
out = root / "data" / "evals"
out.mkdir(exist_ok=True)


def memory():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()


with httpx.Client(base_url="http://127.0.0.1:8010", timeout=30) as client:
    papers = client.get("/api/papers").json()
    paper = next(p for p in papers if "LoRA" in p["title"])
    draft = client.post(
        "/api/drafts",
        json={
            "title": "Engineering validation: LoRA citation",
            "text": "Parameter-efficient adaptation can reduce the number of trainable parameters while preserving the pretrained model. Low-rank updates are one approach to this problem.",
        },
    ).json()
    response = client.post(
        "/api/citations",
        json={
            "draft_id": draft["id"],
            "paper_ids": [paper["id"]],
            "model_policy": "local",
        },
    )
    response.raise_for_status()
    rid = response.json()["run_id"]
    started = time.time()
    before = memory()
    samples = []
    print("run_id=" + rid, flush=True)
    while time.time() - started < 400:
        run = client.get("/api/runs/" + rid).json()
        samples.append(
            {"seconds": round(time.time() - started, 1), "compute_memory": memory()}
        )
        if run["status"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(3)
    trace = client.get(f"/api/runs/{rid}/trace").json()
    result = {
        "run_id": rid,
        "status": run["status"],
        "error": run["error"],
        "seconds": round(time.time() - started, 2),
        "memory_before": before,
        "memory_after": memory(),
        "samples": samples,
        "trace": trace,
        "result": run["result"],
    }
    (out / "hybrid-local-validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "run_id",
                    "status",
                    "error",
                    "seconds",
                    "memory_before",
                    "memory_after",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    assert run["status"] == "completed", run["error"]
    assert all(
        a["usage"].get("provider") == "local" for a in trace["attempts"] if a["usage"]
    )
