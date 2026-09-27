"""仅本地 CLI 登记可信代码，不接受浏览器上传可执行脚本。"""

import argparse
from pathlib import Path
from .config import Settings
from .store import Store
from .experiments import Trainer, resolve_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Settings().data_dir)
    args = parser.parse_args()
    adapter = Trainer.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    adapter.root = str(Path(adapter.root).resolve())
    root = Path(adapter.root)
    for name in adapter.source_files + adapter.dataset_files:
        resolve_file(root, name)
    for dataset in adapter.datasets:
        if not Path(dataset.root).resolve().is_dir():
            raise ValueError(f"数据目录不存在: {dataset.root}")
    if adapter.entrypoint not in adapter.source_files:
        raise ValueError("Entrypoint must be listed in source_files")
    store = Store(args.data_dir)
    directory = store.root / "trainers"
    directory.mkdir(exist_ok=True)
    target = directory / (adapter.id + ".json")
    target.write_text(adapter.model_dump_json(indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
