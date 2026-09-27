"""SQLite 是唯一任务/证据事实源；每次操作独立连接，页级事务可恢复。"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import uuid
import hashlib
import threading
import time


class LeaseLost(RuntimeError):
    """执行权已过期，旧进程不得再提交任何状态。"""


class IdempotencyConflict(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def uid(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def dump(value):
    return json.dumps(value, ensure_ascii=False)


class Store:
    def __init__(self, data_dir: Path):
        self.local = threading.local()
        self.root = data_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "loom.sqlite3"
        with self.db() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS papers (
              id TEXT PRIMARY KEY, sha TEXT UNIQUE NOT NULL, filename TEXT NOT NULL,
              title TEXT NOT NULL, path TEXT NOT NULL, pages INTEGER NOT NULL,
              status TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS pages (
              paper_id TEXT NOT NULL REFERENCES papers(id), page INTEGER NOT NULL,
              width REAL, height REAL, text TEXT, image TEXT, captions TEXT,
              PRIMARY KEY(paper_id,page));
            CREATE TABLE IF NOT EXISTS evidence (
              id TEXT PRIMARY KEY, paper_id TEXT NOT NULL REFERENCES papers(id),
              page INTEGER NOT NULL, bbox TEXT NOT NULL, kind TEXT NOT NULL,
              content TEXT NOT NULL, metadata TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS evidence_paper ON evidence(paper_id,page);
            CREATE TABLE IF NOT EXISTS embeddings (
              evidence_id TEXT PRIMARY KEY REFERENCES evidence(id) ON DELETE CASCADE,
              model TEXT NOT NULL, vector BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
              request TEXT NOT NULL, checkpoint TEXT NOT NULL DEFAULT '{}',
              result TEXT, error TEXT, created TEXT NOT NULL, updated TEXT NOT NULL,
              cancel INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS events (
              seq INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL REFERENCES runs(id), time TEXT NOT NULL,
              type TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS events_run ON events(run_id,seq);
            CREATE TABLE IF NOT EXISTS reviews (
              id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id),
              item_id TEXT NOT NULL, verdict TEXT NOT NULL, note TEXT NOT NULL, created TEXT NOT NULL);
            """)
            # 向后兼容迁移；旧任务保留原 ID、checkpoint 和结果。
            db.execute("BEGIN IMMEDIATE")
            columns = {r[1] for r in db.execute("PRAGMA table_info(runs)")}
            for name, definition in {
                "owner": "TEXT", "epoch": "INTEGER NOT NULL DEFAULT 0",
                "lease_until": "REAL NOT NULL DEFAULT 0", "not_before": "REAL NOT NULL DEFAULT 0",
                "deadline": "REAL NOT NULL DEFAULT 0", "recoveries": "INTEGER NOT NULL DEFAULT 0",
            }.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
            db.execute("CREATE INDEX IF NOT EXISTS runs_dispatch ON runs(status,not_before,created)")
            db.execute("CREATE TABLE IF NOT EXISTS submissions (key TEXT PRIMARY KEY, digest TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES runs(id), created TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS model_attempts (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), epoch INTEGER NOT NULL,
                stage TEXT NOT NULL, input_hash TEXT NOT NULL, input TEXT NOT NULL,
                status TEXT NOT NULL, raw_output TEXT, usage TEXT, error TEXT,
                created TEXT NOT NULL, updated TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS attempts_run ON model_attempts(run_id,stage,created)")
            from .workspace import install
            install(db)
            db.execute("PRAGMA user_version=3")

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            lease = getattr(self.local, "lease", None)
            if lease:
                # 锁内检查 fencing token，避免检查通过后执行权再被其他进程接管。
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT status,owner,epoch,lease_until FROM runs WHERE id=?", (lease[0],)).fetchone()
                if not row or row["status"] != "running" or (row["owner"], row["epoch"]) != lease[1:] or row["lease_until"] <= time.time():
                    raise LeaseLost("执行租约已失效，禁止旧执行者提交")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def query(self, sql, args=()):
        with self.db() as db:
            return [dict(r) for r in db.execute(sql, args).fetchall()]

    def execute(self, sql, args=()):
        with self.db() as db:
            db.execute(sql, args)

    def paper(self, paper_id):
        rows = self.query("SELECT * FROM papers WHERE id=?", (paper_id,))
        if not rows:
            raise KeyError("论文不存在")
        return rows[0]

    def papers(self):
        return self.query("SELECT id,sha,filename,title,pages,status,created FROM papers ORDER BY created DESC")

    def add_paper(self, sha, filename, title, path, pages):
        paper_id = "p_" + sha[:20]
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO papers VALUES (?,?,?,?,?,?,?,?)", (paper_id, sha, filename, title, str(path), pages, "queued", now()))
            row = dict(db.execute("SELECT * FROM papers WHERE sha=?", (sha,)).fetchone())
        return row

    def save_page(self, paper_id, page, width, height, text, image, captions, evidence):
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO pages VALUES (?,?,?,?,?,?,?)", (paper_id, page, width, height, text, image, dump(captions)))
            for ev in evidence:
                db.execute("INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?,?,?)", (ev["id"], paper_id, page, dump(ev["bbox"]), ev["kind"], ev["content"], dump(ev.get("metadata", {}))))

    def get_evidence(self, ids=None, paper_ids=None):
        if ids is not None and not ids or paper_ids is not None and not paper_ids:
            return []
        sql, args = "SELECT * FROM evidence", []
        if ids is not None:
            sql += " WHERE id IN (" + ",".join("?" for _ in ids) + ")"
            args = ids
        elif paper_ids is not None:
            sql += " WHERE paper_id IN (" + ",".join("?" for _ in paper_ids) + ") AND kind='text'"
            args = paper_ids
        rows = self.query(sql + " ORDER BY paper_id,page,id", args)
        for row in rows:
            row["bbox"] = json.loads(row["bbox"])
            row["metadata"] = json.loads(row["metadata"])
        return rows

    def visual_evidence(self, paper_id, page, bbox, content, metadata):
        eid = uid("v")
        self.execute("INSERT INTO evidence VALUES (?,?,?,?,?,?,?)", (eid, paper_id, page, dump(bbox), "visual", content, dump(metadata)))
        return self.get_evidence(ids=[eid])[0]

    def create_run(self, kind, request, limit=12, key=None, ttl=3600):
        digest = hashlib.sha256(dump({"kind": kind, "request": request}).encode()).hexdigest()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if key:
                previous = db.execute("SELECT * FROM submissions WHERE key=?", (key,)).fetchone()
                if previous:
                    if previous["digest"] != digest:
                        raise IdempotencyConflict("同一个 Idempotency-Key 不能提交不同请求")
                    return previous["run_id"]
            # 同一 PDF 的解析任务也按业务身份去重。
            if kind == "ingest":
                existing = db.execute("SELECT id FROM runs WHERE kind='ingest' AND status IN ('queued','running') AND json_extract(request,'$.paper_id')=?", (request["paper_id"],)).fetchone()
                if existing:
                    return existing["id"]
            count = db.execute("SELECT COUNT(*) FROM runs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= limit:
                raise OverflowError("持久任务队列已满，请等待已有任务完成")
            run_id = uid("r")
            db.execute("INSERT INTO runs(id,kind,status,request,created,updated,deadline) VALUES (?,?,?,?,?,?,?)", (run_id, kind, "queued", dump(request), now(), now(), time.time()+ttl))
            self._event(db, run_id, "queued", {"message": "请求与队列事件已原子保存"})
            if key:
                db.execute("INSERT INTO submissions VALUES (?,?,?,?)", (key, digest, run_id, now()))
            return run_id

    @staticmethod
    def _event(db, run_id, kind, payload):
        db.execute("INSERT INTO events(run_id,time,type,payload) VALUES (?,?,?,?)", (run_id, now(), kind, dump(payload)))

    @contextmanager
    def leased(self, run_id, owner, epoch):
        previous = getattr(self.local, "lease", None)
        self.local.lease = (run_id, owner, epoch)
        try:
            yield
        finally:
            self.local.lease = previous

    def claim(self, owner, seconds, run_id=None, max_recoveries=2):
        """同主机事务抢占；一个执行槽位，过期接管递增 epoch。"""
        t = time.time()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            # 截止时间属于任务，重启不延长；取消排队任务无需占用执行槽。
            for row in db.execute("SELECT id,cancel FROM runs WHERE status='queued' AND (cancel=1 OR (deadline>0 AND deadline<=?))", (t,)).fetchall():
                state = "cancelled" if row["cancel"] else "failed"
                db.execute("UPDATE runs SET status=?,error=?,updated=? WHERE id=?", (state, "任务取消或超过总期限", now(), row["id"]))
                self._event(db, row["id"], state, {"message": "任务取消或超过总期限"})
            if db.execute("SELECT 1 FROM runs WHERE status='running' AND lease_until>?", (t,)).fetchone():
                return None
            sql = "SELECT * FROM runs WHERE ((status='queued' AND not_before<=?) OR (status='running' AND lease_until<=?))"
            args = [t,t]
            if run_id:
                sql += " AND id=?"
                args.append(run_id)
            row = db.execute(sql + " ORDER BY created,id LIMIT 1", args).fetchone()
            if not row:
                return None
            expired = row["status"] == "running"
            if expired and (row["recoveries"] >= max_recoveries or row["deadline"] and row["deadline"] <= t):
                db.execute("UPDATE runs SET status='failed',error=?,owner=NULL,lease_until=0,updated=? WHERE id=?", ("故障恢复预算耗尽或任务超过总期限", now(), row["id"]))
                self._event(db,row["id"],"failed",{"message":"故障恢复预算耗尽或任务超过总期限"})
                return None
            epoch = row["epoch"]+1
            db.execute("UPDATE runs SET status='running',owner=?,epoch=?,lease_until=?,recoveries=recoveries+?,updated=? WHERE id=?", (owner,epoch,t+seconds,int(expired),now(),row["id"]))
            self._event(db,row["id"],"lease_recovered" if expired else "started",{"message":"过期任务从持久阶段恢复" if expired else "任务取得执行租约", "owner":owner,"epoch":epoch})
            return row["id"], epoch

    def heartbeat(self, run_id, owner, epoch, seconds):
        with self.db() as db:
            return db.execute("UPDATE runs SET lease_until=? WHERE id=? AND status='running' AND owner=? AND epoch=? AND lease_until>?", (time.time()+seconds,run_id,owner,epoch,time.time())).rowcount == 1

    def finish(self, run_id, status, result=None, error=None, delay=0):
        # 必须在 leased 上下文内调用，结果和终态事件在同一事务提交。
        if not getattr(self.local,"lease",None):
            raise LeaseLost("终态提交需要执行租约")
        with self.db() as db:
            if status == "completed" and db.execute("SELECT cancel FROM runs WHERE id=?",(run_id,)).fetchone()[0]:
                status, result = "cancelled", None
            db.execute("UPDATE runs SET status=?,result=?,error=?,owner=NULL,lease_until=0,not_before=?,updated=? WHERE id=?", (status,dump(result) if result is not None else None,error,time.time()+delay,now(),run_id))
            self._event(db,run_id,status,{"message":error or ("任务完成，结果已保存" if status=="completed" else "阶段已保存"),"epoch":self.local.lease[2],"retry_at":time.time()+delay if delay else None})

    def retry(self, run_id, limit, ttl):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE id=?",(run_id,)).fetchone()
            if not row:
                raise KeyError("任务不存在")
            if row["status"] not in {"failed","cancelled"}:
                raise IdempotencyConflict("只有失败或取消的任务可重试")
            if db.execute("SELECT COUNT(*) FROM runs WHERE status IN ('queued','running')").fetchone()[0] >= limit:
                raise OverflowError("队列已满")
            # 显式人工重试给新的任务期限/恢复预算；模型调用总预算不重置。
            db.execute("UPDATE runs SET status='queued',cancel=0,error=NULL,not_before=0,deadline=?,recoveries=0,updated=? WHERE id=?", (time.time()+ttl,now(),run_id))
            self._event(db,run_id,"queued",{"message":"人工重试，保留模型输出与累计调用次数"})

    def request_cancel(self, run_id):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM runs WHERE id=?",(run_id,)).fetchone()
            if not row:
                raise KeyError("任务不存在")
            if row["status"] in {"queued","running"}:
                db.execute("UPDATE runs SET cancel=1,updated=? WHERE id=?",(now(),run_id))
                self._event(db,run_id,"cancellation_requested",{"message":"取消请求已持久接收，等待执行者停止"})

    def run(self, run_id):
        rows = self.query("SELECT * FROM runs WHERE id=?", (run_id,))
        if not rows:
            raise KeyError("任务不存在")
        row = rows[0]
        for key in ("request", "checkpoint", "result"):
            row[key] = json.loads(row[key]) if row[key] else None
        return row

    def update_run(self, run_id, **values):
        allowed = {"status", "checkpoint", "result", "error", "cancel"}
        if not values or not set(values) <= allowed:
            raise ValueError("无效任务字段")
        values["updated"] = now()
        for key in ("checkpoint", "result"):
            if key in values:
                values[key] = dump(values[key])
        self.execute("UPDATE runs SET " + ",".join(f"{k}=?" for k in values) + " WHERE id=?", [*values.values(), run_id])

    def event(self, run_id, kind, payload):
        self.execute("INSERT INTO events(run_id,time,type,payload) VALUES (?,?,?,?)", (run_id, now(), kind, dump(payload)))

    def events(self, run_id, after=0):
        rows = self.query("SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT 100", (run_id, after))
        for row in rows:
            row["payload"] = json.loads(row["payload"])
        return rows
