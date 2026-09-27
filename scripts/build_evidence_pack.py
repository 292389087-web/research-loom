"""按显式白名单构建可公开核验的证据包，并生成 SHA-256 manifest。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys


SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"(?i)(?:api[_-]?key|authorization|access[_-]?token|secret[_-]?key)[\"']?\s*[=:]\s*[\"']?(?:bearer\s+)?[A-Za-z0-9_./+=-]{20,}"),
    re.compile(rb"(?i)bearer\s+[A-Za-z0-9_./+=-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
FORBIDDEN_NAMES = {".env", "deepseek.local.toml"}
FORBIDDEN_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contains_secret(path: Path) -> bool:
    carry = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sample = carry + chunk
            if any(pattern.search(sample) for pattern in SECRET_PATTERNS):
                return True
            carry = sample[-256:]
    return False


def safe_source(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"证据文件越界或不存在: {path}")
    name = resolved.name.lower()
    if name in FORBIDDEN_NAMES or name.startswith('.env.') or name.endswith('.local.toml') or resolved.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"证据包禁止包含密钥文件: {resolved.name}")
    if contains_secret(resolved):
        raise ValueError(f"疑似包含密钥，拒绝打包: {resolved.relative_to(root)}")
    return resolved


def collect(root: Path, spec: dict) -> list[Path]:
    root = root.resolve()
    if not isinstance(spec, dict) or not isinstance(spec.get('entries'), list):
        raise ValueError('证据 spec 需要 entries 数组')
    selected = {}
    for entry in spec.get("entries", []):
        if not isinstance(entry, dict):
            raise ValueError('证据条目必须是对象')
        pattern = entry.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("每个证据条目都需要 pattern")
        if Path(pattern).is_absolute() or '..' in Path(pattern).parts:
            raise ValueError('证据 pattern 不允许绝对路径或父目录')
        matches = [path for path in root.glob(pattern) if path.is_file()]
        if not matches and not entry.get("optional", False):
            raise ValueError(f"必需证据不存在: {pattern}")
        for path in matches:
            resolved = safe_source(root, path)
            selected[resolved.relative_to(root).as_posix()] = resolved
    return [selected[name] for name in sorted(selected)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--spec", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "evidence-pack.example.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("输出目录必须不存在或为空，避免旧证据混入")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    files = collect(root, spec)
    if not files:
        raise ValueError('证据包不能为空')
    if any(p.is_relative_to(output) for p in files):
        raise ValueError('输出目录不能包含来源文件')
    if any(p.relative_to(root).as_posix() == 'manifest.json' for p in files):
        raise ValueError('根 manifest.json 是保留路径')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for source in files:
        relative = source.relative_to(root)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        safe_source(output, target)  # 也扫描实际交付副本，避免扫描后来源变化漏检。
        records.append({
            "path": relative.as_posix(),
            "bytes": target.stat().st_size,
            "sha256": digest(target),
        })
    manifest = {
        "schema": "research-loom-evidence-pack-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "privacy_review": "required_before_publication",
        "files": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{output} ({len(records)} files)")


if __name__ == "__main__":
    main()
