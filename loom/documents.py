"""PDF 原生文本与版面证据。论文内容始终视为资料，绝不执行其中指令。"""
import re
import fitz
from pathlib import Path

PARSER_VERSION = "blocks-v1"


def inspect_pdf(path: Path, max_pages: int):
    # 无效 PDF 在 Windows 上也不应持有文件句柄，防止清理时掩盖原异常。
    with fitz.open(stream=path.read_bytes(), filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("暂不支持加密 PDF，请先提供可读取的版本")
        if not 1 <= len(doc) <= max_pages:
            raise ValueError(f"PDF 页数须为 1–{max_pages}")
        title = (doc.metadata or {}).get("title", "").strip()
        if not title or len(title) > 250 or "latex" in title.lower():
            title = path.stem
        return len(doc), title


def ingest(store, paper_id, emit, cancelled):
    paper = store.paper(paper_id)
    target = store.root / "pages" / paper_id
    target.mkdir(parents=True, exist_ok=True)
    complete = {p["page"] for p in store.query("SELECT page FROM pages WHERE paper_id=?", (paper_id,))}
    store.execute("UPDATE papers SET status='parsing' WHERE id=?", (paper_id,))
    with fitz.open(paper["path"]) as doc:
        for index, page in enumerate(doc):
            cancelled()
            number = index + 1
            if number in complete:
                continue
            width, height = page.rect.width, page.rect.height
            raw = page.get_text("blocks", sort=True)
            evidence, captions = [], []
            for block_index, block in enumerate(raw):
                x0, y0, x1, y1, text, *_ = block
                text = text.strip()
                if not text or len(text) < 12:
                    continue
                if y0 > height * .965 or (y1 < height * .055 and len(text) < 180):
                    continue
                bbox = [max(0, x0 / width), max(0, y0 / height), min(1, x1 / width), min(1, y1 / height)]
                if re.search(r"(?:Fig(?:ure)?\.?\s*\d|TABLE\s*\d|图\s*\d)", text, re.I):
                    captions.append({"text": text[:600], "bbox": bbox})
                for part, start in enumerate(range(0, len(text), 1800)):
                    evidence.append({"id": f"{paper_id}_p{number}_b{block_index}_{part}", "bbox": bbox, "kind": "text", "content": text[start:start + 1800], "metadata": {"parser": PARSER_VERSION}})
            image = target / f"{number}.png"
            page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False).save(image)
            store.save_page(paper_id, number, width, height, page.get_text(sort=True), str(image), captions, evidence)
            emit("page_parsed", {"message": f"已解析第 {number}/{len(doc)} 页", "paper_id": paper_id, "page": number, "total": len(doc), "text_blocks": len(evidence)})
    store.execute("UPDATE papers SET status='ready' WHERE id=?", (paper_id,))
    return {"paper_id": paper_id, "pages": paper["pages"], "evidence_count": len(store.get_evidence(paper_ids=[paper_id]))}


def render_region(store, paper_id, page_number, bbox=None):
    paper = store.paper(paper_id)
    if not 1 <= page_number <= paper["pages"]:
        raise ValueError("页码超出论文范围")
    with fitz.open(paper["path"]) as doc:
        page = doc[page_number - 1]
        rect = page.rect
        if bbox:
            rect = fitz.Rect(bbox[0]*rect.width, bbox[1]*rect.height, bbox[2]*rect.width, bbox[3]*rect.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), clip=rect, alpha=False)
        from PIL import Image
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
