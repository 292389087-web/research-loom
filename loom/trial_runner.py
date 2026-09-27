"""可信的单进程训练入口。包装进程自己持有 GPU 锁，调度器退出不释放该锁。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import runpy
import shutil
import sys
import threading
import time
from contextlib import redirect_stdout, redirect_stderr
from .gpu import gpu_slot, ResourceBusy
from .store import Store, dump
from .experiments import (
    get_study,
    resolve_file,
    event,
    dataset_snapshot,
    loom_runtime_hashes,
)
from .training import atomic_json
from .artifact_finalization import record_execution, validate_outputs


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_digest(folder, name, max_bytes):
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError("Required experiment artifact missing or invalid: " + name)
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"Experiment artifact exceeds frozen size limit: {name} ({size} > {max_bytes} bytes)")
    return digest(path)


def run_attempt(store, aid):
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT a.*,t.study_id,t.params FROM trial_attempts a JOIN trials t ON t.id=a.trial_id WHERE a.id=?",
            (aid,),
        ).fetchone()
        if not row or row["status"] != "queued":
            return
        row = dict(row)
        state = db.execute('SELECT status,stop,deadline FROM studies WHERE id=?',(row['study_id'],)).fetchone()
        if state['status'] != 'running' or state['stop'] or (state['deadline'] and time.time() >= state['deadline']):
            return
        db.execute(
            "UPDATE trial_attempts SET status='running',execution_state='running',artifact_state='pending',pid=?,started=? WHERE id=? AND status='queued'",
            (os.getpid(), time.time(), aid),
        )
    study = get_study(store, row["study_id"])
    spec = study["spec"]
    manifest = study["manifest"]
    folder = store.root / "experiments" / aid
    folder.mkdir(parents=True, exist_ok=True)
    done = threading.Event()
    started = time.time()

    def finish(status, metrics=None, error=None):
        store.execute(
            "UPDATE trial_attempts SET status=?,metrics=?,error=?,finished=?, artifact_state=CASE WHEN execution_state='completed' THEN ? ELSE 'not_checked' END, execution_state=CASE WHEN execution_state='running' THEN ? ELSE execution_state END WHERE id=? AND status='running' AND pid=?",
            (
                status,
                dump(metrics) if metrics is not None else None,
                error,
                time.time(),
                "accepted" if status=="completed" else "failed",
                status,
                aid,
                os.getpid(),
            ),
        )

    def monitor():
        while not done.wait(0.25):
            try:
                current = store.query(
                    "SELECT stop,deadline FROM studies WHERE id=?", (study["id"],)
                )[0]
                timeout = time.time() - started >= spec["trial_timeout"] or (
                    current["deadline"] and time.time() >= current["deadline"]
                )
                if current["stop"] or timeout:
                    finish(
                        "cancelled" if current["stop"] else "failed",
                        error=(
                            "cancelled"
                            if current["stop"]
                            else "execution deadline exceeded"
                        ),
                    )
                    os._exit(124)  # 入口在当前进程；退出后 OS 回收显存和文件锁。
            except Exception:
                # DB 失联不能让预算无限失效。
                if time.time() - started >= spec["trial_timeout"]:
                    os._exit(125)

    threading.Thread(target=monitor, daemon=True).start()
    try:
        if (
            manifest.get("loom_runtime_hashes")
            and manifest["loom_runtime_hashes"] != loom_runtime_hashes()
        ):
            raise ValueError("Research Loom runtime changed after the study was frozen")
        source = folder / "source"
        source.mkdir(exist_ok=True)
        root = Path(manifest["adapter"]["root"]).resolve()
        for name, expected in manifest.get(
            "source_hashes", manifest.get("hashes", {})
        ).items():
            original = resolve_file(root, name)
            if digest(original) != expected:
                raise ValueError("Registered source or dataset changed: " + name)
            if name in manifest["adapter"]["source_files"]:
                target = (source / name).resolve()
                if not target.is_relative_to(source.resolve()):
                    raise ValueError("Unsafe snapshot path")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                if digest(target) != expected:
                    raise ValueError("Source changed during snapshot: " + name)
        from .preflight import asset_snapshot
        if asset_snapshot(manifest["adapter"]) != manifest.get("assets", {}):
            raise ValueError("Registered model/assets changed after study freeze")
        current_datasets = [
            dataset_snapshot(item) for item in manifest["adapter"].get("datasets", [])
        ]
        if current_datasets != manifest.get("datasets", []):
            raise ValueError("Registered dataset changed after the study was frozen")
        config = {
            "attempt_id": aid,
            "study_id": study["id"],
            "seed": row["seed"],
            "parameters": json.loads(row["params"]),
            "objective": spec["objective"],
            "constraints": spec["constraints"],
            "protocol": spec["protocol"],
            "data_root": current_datasets[0]["root"] if current_datasets else str(root),
            "datasets": current_datasets,
        }
        atomic_json(folder / "config.json", config)
        atomic_json(
            folder / "manifest.json",
            {
                **manifest,
                "python": sys.version,
                "platform": platform.platform(),
                "config_sha256": digest(folder / "config.json"),
            },
        )
        while True:
            try:
                slot = gpu_slot(store.root)
                slot.__enter__()
                event(
                    store,
                    study["id"],
                    "gpu_acquired",
                    {"attempt_id": aid, "pid": os.getpid()},
                )
                sys.path.insert(0, str(source))
                sys.argv = [
                    str(source / manifest["adapter"]["entrypoint"]),
                    "--config",
                    str(folder / "config.json"),
                    "--output",
                    str(folder),
                ]
                with (folder / "stdout.log").open(
                    "w", encoding="utf-8", buffering=1
                ) as log, redirect_stdout(log), redirect_stderr(log):
                    try:
                        runpy.run_path(sys.argv[0], run_name="__main__")
                    except SystemExit as exc:
                        if exc.code not in (None, 0):
                            raise RuntimeError(
                                "Trainer exited: " + str(exc.code)
                            ) from exc
                receipt = record_execution(store, study, aid, folder)
                metrics, result = validate_outputs(folder, study, receipt)
                atomic_json(folder / "result-manifest.json", result)
                finish("completed", metrics)
                # 不再运行下一项；进程退出才回收训练模型占据的显存。
                done.set()
                os._exit(0)
            except ResourceBusy:
                time.sleep(0.25)
    except BaseException as exc:
        finish("failed", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
    finally:
        done.set()
        os._exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    run_attempt(Store(Path(args.data_dir)), args.attempt)
