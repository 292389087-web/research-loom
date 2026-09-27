"""校验证据包 manifest、文件大小和 SHA-256。"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    args = parser.parse_args()
    root = args.pack.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "research-loom-evidence-pack-v1":
        raise ValueError("不支持的证据包 schema")
    records = manifest.get('files')
    if not isinstance(records, list) or not records:
        raise ValueError('证据 manifest 必须包含非空 files')
    seen = set()
    for record in records:
        relative = record['path']
        if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('证据路径必须是包内相对路径')
        candidate = root / relative
        path = candidate.resolve()
        if path in seen or path == root / 'manifest.json':
            raise ValueError('重复或保留证据路径')
        seen.add(path)
        if candidate.is_symlink() or any(p.is_symlink() for p in candidate.parents if p.is_relative_to(root)):
            raise ValueError('证据包不允许符号链接')
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"证据文件缺失或越界: {record['path']}")
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"证据文件校验失败: {record['path']}")
    actual = {p.resolve() for p in root.rglob('*') if p.is_file() and p != root / 'manifest.json'}
    if actual != seen:
        raise ValueError('证据包存在未列入 manifest 的文件')
    print(f"verified {len(manifest.get('files', []))} files")


if __name__ == "__main__":
    main()
