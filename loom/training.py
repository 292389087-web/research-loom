"""可选训练适配器 SDK：配置读取、结构化进度和原子指标发布。"""

import argparse
import json
import math
from numbers import Real
import os
from pathlib import Path
import time


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def numeric_metrics(metrics):
    if (
        not isinstance(metrics, dict)
        or not metrics
        or len(metrics) > 100
        or any(
            not isinstance(k, str)
            or len(k) > 100
            or isinstance(v, bool)
            or not isinstance(v, Real)
            or not math.isfinite(v)
            for k, v in metrics.items()
        )
    ):
        raise ValueError("Metrics must be a bounded mapping of names to finite numbers")
    return {key: float(value) for key, value in metrics.items()}


class TrainingContext:
    def __init__(self, config, output):
        self.config = json.loads(Path(config).read_text(encoding="utf-8"))
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args()
        return cls(args.config, args.output)

    def report(self, step, metrics, axis="environment_step"):
        if (
            type(step) is not int
            or step < 0
            or axis
            not in (
                "environment_step",
                "optimizer_step",
                "episode",
                "simulation_seconds",
            )
        ):
            raise ValueError("Invalid progress coordinate")
        path = self.output / "progress.jsonl"
        if path.exists() and path.stat().st_size > 10 * 1024 * 1024:
            raise OverflowError("Progress artifact exceeds 10 MB")
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "step": step,
                        "axis": axis,
                        "metrics": numeric_metrics(metrics),
                    },
                    allow_nan=False,
                )
                + "\n"
            )

    def finish(self, metrics):
        atomic_json(self.output / "metrics.json", numeric_metrics(metrics))
