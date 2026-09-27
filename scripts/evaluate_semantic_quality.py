"""在冻结人工标注上生成语义质量 JSON/Markdown 报告。"""

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loom.evaluation import evaluate, file_sha256, load_jsonl, render_markdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="输出 JSON；同时生成同名 .md")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--example", action="store_true", help="仅作格式演示，不是质量结果")
    mode.add_argument("--metadata", type=Path, help="正式候选报告的来源配置；仍需人工确认标注")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("--output 必须为 .json，避免覆盖 Markdown 报告")
    inputs = {p.resolve() for p in (args.gold, args.predictions, args.metadata) if p}
    if args.output.resolve() in inputs or args.output.with_suffix('.md').resolve() in inputs:
        parser.error("输出不能覆盖输入")
    metadata = {}
    if args.metadata:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        required = ("dataset_version", "split", "model_version", "prompt_version", "retrieval_config", "generated_at", "annotation_protocol")
        if not isinstance(metadata, dict) or any(not metadata.get(k) for k in required):
            parser.error("metadata 缺少数据版本、split、模型/Prompt 版本、检索配置、生成时间或标注协议")
    report = evaluate(load_jsonl(args.gold), load_jsonl(args.predictions))
    report.update(report_kind="example" if args.example else "annotated_candidate",
                  evaluated_at=datetime.now(timezone.utc).isoformat(), metadata=metadata,
                  evaluator_sha256=file_sha256(ROOT / "loom" / "evaluation.py"))
    report["inputs"] = {
        "gold": str(args.gold),
        "gold_sha256": file_sha256(args.gold),
        "predictions": str(args.predictions),
        "predictions_sha256": file_sha256(args.predictions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
