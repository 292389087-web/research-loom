"""按所选论文限定范围的 BM25 + 本地 BGE-M3 + RRF。向量持久化。"""
import hashlib
import re
import numpy as np
from rank_bm25 import BM25Okapi


def tokens(text):
    text = text.lower()
    english = re.findall(r"[a-z0-9][a-z0-9_-]*", text)
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return english + chinese + [a+b for a, b in zip(chinese, chinese[1:])]


class Retriever:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings
        self.embedder = None
        self.dense_error = None

    def dense(self, evidence, queries, emit, cancelled):
        path = self.settings.embedding_path
        if not self.settings.use_dense or path is None or self.dense_error:
            return None
        try:
            if self.embedder is None:
                emit("retrieval", {"message": "加载本地 BGE-M3，构建可复用的论文向量"})
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(str(path), device="cpu", local_files_only=True)
                self.embedder.max_seq_length = 512
            version = hashlib.sha256((str(path) + str((path / "config.json").stat().st_mtime_ns)).encode()).hexdigest()[:16]
            cached = {r["evidence_id"]: np.frombuffer(r["vector"], dtype=np.float32) for r in self.store.query("SELECT * FROM embeddings WHERE model=?", (version,))}
            missing = [e for e in evidence if e["id"] not in cached]
            for i in range(0, len(missing), 12):
                cancelled()
                batch = missing[i:i+12]
                vecs = self.embedder.encode([e["content"] for e in batch], normalize_embeddings=True, show_progress_bar=False, batch_size=12)
                with self.store.db() as db:
                    for ev, vec in zip(batch, vecs):
                        vec = np.asarray(vec, dtype=np.float32)
                        cached[ev["id"]] = vec
                        db.execute("INSERT OR REPLACE INTO embeddings VALUES (?,?,?)", (ev["id"], version, vec.tobytes()))
                emit("retrieval", {"message": f"论文向量 {min(i+12, len(missing))}/{len(missing)}", "indexed": min(i+12, len(missing))})
            q = np.asarray(self.embedder.encode(queries, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
            return np.stack([cached[e["id"]] for e in evidence]) @ q.T
        except InterruptedError:
            raise
        except Exception as exc:
            self.dense_error = str(exc)
            emit("warning", {"message": "稠密检索不可用，明确降级为 BM25", "detail": str(exc)[:200]})
            return None

    def search(self, paper_ids, queries, emit, cancelled, limit=28, char_budget=14000):
        evidence = self.store.get_evidence(paper_ids=paper_ids)
        if not evidence:
            return [], {"mode": "visual_only", "candidates": 0}
        bm25 = BM25Okapi([tokens(e["content"]) or ["empty"] for e in evidence])
        sparse = np.stack([bm25.get_scores(tokens(q)) for q in queries], axis=1)
        dense = self.dense(evidence, queries, emit, cancelled)
        scores = np.zeros(len(evidence))
        for matrix in ([sparse, dense] if dense is not None else [sparse]):
            for qi in range(matrix.shape[1]):
                for rank, idx in enumerate(np.argsort(-matrix[:, qi])[:60]):
                    scores[idx] += 1/(60+rank+1)
        ranked = list(np.argsort(-scores))
        # 每篇论文保留最相关片段与首页上下文，防止多论文任务被单篇支配。
        forced = []
        for paper_id in paper_ids:
            forced += [i for i in ranked if evidence[i]["paper_id"] == paper_id][:3]
            forced += [i for i, e in enumerate(evidence) if e["paper_id"] == paper_id and e["page"] == 1][:2]
        selected, used, size = [], set(), 0
        for i in forced + ranked:
            if i in used:
                continue
            ev = evidence[i]
            if size + len(ev["content"]) > char_budget:
                continue
            selected.append(ev)
            used.add(i)
            size += len(ev["content"])
            if len(selected) >= limit:
                break
        return selected, {"mode": "BGE-M3 + BM25 + RRF" if dense is not None else "BM25", "candidates": len(evidence), "selected": len(selected), "characters": size, "dense_error": self.dense_error}
