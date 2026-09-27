"""检查当前/历史文档状态标记和已知过期表述。"""

from pathlib import Path


CURRENT = (
    "README.md",
    "config/README.md",
    "docs/CURRENT_STATUS.md",
    "docs/EXPERIMENT_ASSISTANT.md",
    "docs/HISTORICAL_INDEX.md",
    "docs/RELIABILITY_OPERATIONS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/SEMANTIC_EVALUATION.md",
    "docs/WORKSPACE_OPERATIONS.md",
    "docs/BATCH_EXPERIMENTS.md",
    "docs/README.md",
    "docs/ARCHITECTURE.md",
    "docs/ENGINEERING_HARDENING_20260922.md",
)
def historical_documents(root: Path) -> tuple[str, ...]:
    """归档集合以目录实际内容为准，便于快照增删。"""
    return tuple(f"docs/archive/{path.name}" for path in sorted((root / "docs" / "archive").glob("*.md")))

FORBIDDEN = {
    "README.md": ("自动实验由独立调度器", "已加入混合模型路由、文献元数据、引用推荐、自动实验"),
    "docs/WORKSPACE_OPERATIONS.md": ("需要自动实验时", "点击“启动已冻结计划”后自动继续预算内试验"),
    "docs/EXPERIMENT_ASSISTANT.md": ("仍待独立修复",),
    "config/README.md": ("当前没有真实密钥",),
}


def check(root: Path) -> list[str]:
    failures = []
    for relative in CURRENT:
        path = root / relative
        if not path.is_file():
            failures.append(f"当前文档缺失: {relative}")
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "文档状态：当前" not in text[:500]:
            failures.append(f"当前文档缺少状态标记: {relative}")
    for relative in historical_documents(root):
        path = root / relative
        if not path.is_file():
            failures.append(f"历史文档缺失: {relative}")
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "文档状态：历史" not in text[:500]:
            failures.append(f"历史文档缺少归档标记: {relative}")
    for relative, phrases in FORBIDDEN.items():
        if not (root / relative).is_file():
            continue  # 已由缺失文件检查报告，不用 FileNotFoundError 遮蔽其他问题。
        text = (root / relative).read_text(encoding="utf-8-sig")
        for phrase in phrases:
            if phrase in text:
                failures.append(f"{relative} 仍包含过期表述: {phrase}")
    ignore = (root / ".gitignore").read_text(encoding="utf-8-sig").splitlines()
    if "config/deepseek.local.toml" not in ignore:
        failures.append(".gitignore 缺少 config/deepseek.local.toml")
    return failures


def main():
    root = Path(__file__).resolve().parents[1]
    failures = check(root)
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"documentation contract ok: {len(CURRENT)} current, {len(historical_documents(root))} historical")


if __name__ == "__main__":
    main()
