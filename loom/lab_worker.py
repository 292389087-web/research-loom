"""单主机实验调度器：重启接管状态，不重跑身份未知的训练。"""

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from .config import Settings, ROOT
from .store import Store, uid, now, dump
from .gpu import gpu_slot, ResourceBusy
from .experiments import get_study, next_parameters, summarize, event


def alive(attempt):
    if not attempt["pid"]:
        return False
    try:
        proc = psutil.Process(attempt["pid"])
        return (
            abs(proc.create_time() - attempt["started"]) < 30
            and attempt["id"] in proc.cmdline()
            and proc.is_running()
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


class LabWorker:
    def __init__(self, store):
        self.store = store
        self.children = {}

    def tick(self):
        self.store.execute(
            "INSERT INTO service_heartbeats VALUES ('lab',?,?) ON CONFLICT(name) DO UPDATE SET pid=excluded.pid,updated=excluded.updated",
            (os.getpid(), time.time()),
        )
        for aid, child in list(self.children.items()):
            if child.poll() is not None:
                child.wait()
                del self.children[aid]
        # 接管孤儿记录；未知结果保留为失败，不把重复执行当作新 seed。
        active = self.store.query("SELECT * FROM trial_attempts WHERE status='running'")
        for attempt in active:
            if not alive(attempt):
                self.store.execute(
                    "UPDATE trial_attempts SET status='failed',execution_state=CASE WHEN execution_state='running' THEN 'unknown' ELSE execution_state END,artifact_state=CASE WHEN execution_state='completed' THEN 'pending' ELSE 'not_checked' END,error='Execution process lost; outcome unknown; no automatic rerun',finished=? WHERE id=? AND status='running'",
                    (time.time(), attempt["id"]),
                )
                continue
            state = self.store.query(
                "SELECT s.stop,s.deadline,s.spec FROM studies s JOIN trials t ON t.study_id=s.id WHERE t.id=?",
                (attempt["trial_id"],),
            )[0]
            limit = json.loads(state["spec"])["trial_timeout"]
            expired = time.time() - attempt["started"] >= limit or bool(
                state["deadline"] and time.time() >= state["deadline"]
            )
            if state["stop"] or expired:
                # 独立于训练线程的 watchdog，覆盖持有 GIL 的原生代码阻塞。
                try:
                    process = psutil.Process(attempt["pid"])
                    if alive(attempt):
                        process.kill()
                        process.wait(timeout=5)
                except psutil.NoSuchProcess:
                    pass
                self.store.execute(
                    "UPDATE trial_attempts SET execution_state=CASE WHEN execution_state='running' THEN ? ELSE execution_state END,artifact_state=CASE WHEN execution_state='completed' THEN 'pending' ELSE 'not_checked' END,status=?,error=?,finished=? WHERE id=? AND status='running'",
                    (
                        "cancelled" if state["stop"] else "failed",
                        "cancelled" if state["stop"] else "failed",
                        (
                            "External watchdog: cancelled"
                            if state["stop"]
                            else "External watchdog: deadline exceeded"
                        ),
                        time.time(),
                        attempt["id"],
                    ),
                )
        studies = self.store.query(
            "SELECT id FROM studies WHERE status IN ('running','pausing') OR (status='paused' AND (stop=1 OR deadline<=?)) ORDER BY created", (time.time(),)
        )
        for entry in studies:
            study = get_study(self.store, entry["id"])
            spec = study["spec"]
            sid = study["id"]
            exhausted = bool(study["deadline"] and time.time() >= study["deadline"])
            if study["stop"] or exhausted:
                status = "cancelled" if study["stop"] else "budget_exhausted"
                self.store.execute(
                    "UPDATE trial_attempts SET status='cancelled',execution_state='cancelled',artifact_state='not_checked',error=? WHERE status='queued' AND trial_id IN (SELECT id FROM trials WHERE study_id=?)",
                    (status, sid),
                )
                if any(
                    r["status"] == "running"
                    for t in study["trials"]
                    for r in t["repeats"]
                ):
                    continue
                self.store.execute(
                    "UPDATE trials SET status='cancelled',error=? WHERE study_id=? AND status IN ('queued','running')",
                    (status, sid),
                )
                self.store.execute(
                    "UPDATE studies SET status=? WHERE id=?", (status, sid)
                )
                event(self.store, sid, status, {})
                continue
            if study['status'] == 'pausing':
                if not any(r['status']=='running' for t in study['trials'] for r in t['repeats']):
                    self.store.execute("UPDATE studies SET status='paused' WHERE id=? AND status='pausing'", (sid,))
                    event(self.store,sid,'paused',{'message':'已在 seed 边界暂停；原始总期限继续计时'})
                continue
            pending = next(
                (t for t in study["trials"] if t["status"] in ("queued", "running")),
                None,
            )
            if pending:
                repeats = pending["repeats"]
                if any(r["status"] == "running" for r in repeats):
                    return
                queued = next((r for r in repeats if r["status"] == "queued"), None)
                if queued:
                    if queued["id"] not in self.children:
                        if queued["launches"] >= 3:
                            self.store.execute(
                                "UPDATE trial_attempts SET status='failed',execution_state='not_started',artifact_state='not_checked',error='Runner failed to claim after three launches',finished=? WHERE id=? AND status='queued'",
                                (time.time(), queued["id"]),
                            )
                            return
                        self.store.execute(
                            "UPDATE trial_attempts SET launches=launches+1 WHERE id=? AND status='queued'",
                            (queued["id"],),
                        )
                        child = subprocess.Popen(
                            [
                                sys.executable,
                                "-X",
                                "utf8",
                                "-m",
                                "loom.trial_runner",
                                "--data-dir",
                                str(self.store.root),
                                "--attempt",
                                queued["id"],
                            ],
                            cwd=ROOT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        self.children[queued["id"]] = child
                    return
                good = all(r["status"] == "completed" for r in repeats)
                metrics = summarize(spec, repeats) if good else None
                error = (
                    None
                    if good
                    else "; ".join(
                        (r["error"] or r["status"])
                        for r in repeats
                        if r["status"] != "completed"
                    )[:1500]
                )
                self.store.execute(
                    "UPDATE trials SET status=?,metrics=?,error=? WHERE id=?",
                    (
                        "completed" if good else "failed",
                        dump(metrics) if metrics else None,
                        error,
                        pending["id"],
                    ),
                )
                event(
                    self.store,
                    sid,
                    "trial_finished",
                    {"trial_id": pending["id"], "metrics": metrics, "error": error},
                )
                if not good and spec.get('batch_mode') and spec.get('failure_policy','pause') == 'pause':
                    self.store.execute("UPDATE studies SET status='paused' WHERE id=? AND status='running'",(sid,))
                    event(self.store,sid,'paused_after_failure',{'trial_id':pending['id'],'message':'失败组合已保留；人工恢复后继续下一组，不自动重跑'})
                return
            params = (
                next_parameters(study, self.store)
                if len(study["trials"]) < spec["max_trials"]
                else None
            )
            if params is None:
                status = (
                    "completed"
                    if any(t["status"] == "completed" for t in study["trials"])
                    else "failed"
                )
                if spec.get('batch_mode') and status == 'completed' and any(t['status']!='completed' for t in study['trials']):
                    status = 'completed_with_failures'
                self.store.execute(
                    "UPDATE studies SET status=? WHERE id=?", (status, sid)
                )
                event(
                    self.store,
                    sid,
                    "search_finished",
                    {
                        "status": status,
                        "reason": "trial budget or search space exhausted",
                    },
                )
                continue
            tid = uid("trial")
            with self.store.db() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "INSERT INTO trials(id,study_id,number,params,status,created) VALUES (?,?,?,?,?,?)",
                    (tid, sid, len(study["trials"]), dump(params), "running", now()),
                )
                for seed in spec["seeds"]:
                    db.execute(
                        "INSERT INTO trial_attempts(id,trial_id,seed,status,execution_state,artifact_state) VALUES (?,?,?,?,'queued','not_checked')",
                        (uid("attempt"), tid, seed, "queued"),
                    )
                db.execute(
                    "INSERT INTO study_events(study_id,time,type,payload) VALUES (?,?,?,?)",
                    (
                        sid,
                        now(),
                        "trial_planned",
                        dump({"trial_id": tid, "params": params}),
                    ),
                )
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Settings().data_dir)
    args = parser.parse_args()
    store = Store(args.data_dir)
    worker = LabWorker(store)
    try:
        with gpu_slot(store.root / "lab-dispatch"):
            while True:
                try:
                    worker.tick()
                except Exception:
                    logging.exception("Experiment dispatch failed")
                time.sleep(0.5)
    except ResourceBusy:
        raise SystemExit(
            "An experiment dispatcher is already active for this data directory"
        )


if __name__ == "__main__":
    main()
