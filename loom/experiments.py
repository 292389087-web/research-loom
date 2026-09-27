"""固定评价协议与可信训练适配器；试验结果是唯一优化历史。"""

import hashlib
from contextlib import nullcontext
from importlib.metadata import version, PackageNotFoundError
import json
import math
from pathlib import Path
import statistics
import time
from typing import Literal
from pydantic import Field, model_validator
from .schemas import Strict
from .store import uid, now, dump, IdempotencyConflict
from .experiment_design import Scope, MetricRule, check_derived, check_rules


def loom_runtime_hashes():
    root = Path(__file__).resolve().parent
    names = (
        "experiments.py",
        "experiment_design.py",
        "batch_experiments.py",
        "trial_runner.py",
        "training.py",
        "lab_worker.py",
        "adaptive_controller.py",
        "gpu.py",
        "store.py",
        "preflight.py",
        "artifact_finalization.py",
    )
    return {name: file_digest(root / name) for name in names}


class Trainer(Strict):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,60}$")
    name: str = Field(max_length=160)
    root: str
    entrypoint: str
    source_files: list[str] = Field(min_length=1, max_length=100)
    artifact_files: list[str] = Field(default_factory=list, max_length=30)
    # Freeze the trainer's per-file allowance with the study. Neural-network
    # checkpoints routinely exceed the former hard-coded 10 MiB limit.
    artifact_max_bytes: int = Field(default=256 * 1024 * 1024, ge=1, le=2 * 1024**3)
    required_modules: list[str] = Field(default_factory=list, max_length=50)
    required_assets: dict[str, str] = Field(default_factory=dict, max_length=30)
    evaluation_phase: Literal['training', 'evaluation', 'unspecified'] = 'unspecified'
    timeseries_artifact: str | None = None
    timeseries_axis: str = 'day'
    timeseries_metadata: dict[str, dict[str, str]] = Field(default_factory=dict, max_length=50)
    datasets: list["Dataset"] = Field(default_factory=list, max_length=10)
    # 旧适配器兼容字段；新登记应使用 datasets，以便目录级资料留下快照。
    dataset_files: list[str] = Field(default_factory=list, max_length=30)
    parameters: dict[str, list[float]] = Field(min_length=1, max_length=24)
    metrics: list[str] = Field(min_length=1, max_length=30)
    fixture: bool = False
    parameter_groups: dict[str, Literal['reward', 'hyperparameter', 'protocol']] = Field(default_factory=dict)
    parameter_labels: dict[str, str] = Field(default_factory=dict)
    metric_metadata: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def metadata_keys(self):
        if not set(self.parameter_groups) <= set(self.parameters) or not set(self.parameter_labels) <= set(self.parameters):
            raise ValueError('参数元数据包含未登记字段')
        if not set(self.metric_metadata) <= set(self.metrics):
            raise ValueError('指标元数据包含未登记指标')
        if self.timeseries_artifact and self.timeseries_artifact not in self.artifact_files:
            raise ValueError('时间序列文件必须列入产物清单')
        return self


class Dataset(Strict):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,60}$")
    root: str
    pattern: str = Field(default="*.csv", min_length=1, max_length=200)


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset_snapshot(dataset):
    root = Path(dataset["root"]).resolve()
    if not root.is_dir():
        raise ValueError(f"数据目录不存在: {root}")
    files = sorted(path for path in root.glob(dataset["pattern"]) if path.is_file())
    if not files:
        raise ValueError(f"数据集 {dataset['id']} 没有匹配 {dataset['pattern']} 的文件")
    if len(files) > 20_000:
        raise ValueError("单个数据集最多允许 20,000 个文件")
    manifest = hashlib.sha256()
    size = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        item = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        }
        manifest.update((dump(item) + "\n").encode())
        size += item["bytes"]
    return {
        "id": dataset["id"],
        "root": str(root),
        "pattern": dataset["pattern"],
        "files": len(files),
        "bytes": size,
        "manifest_sha256": manifest.hexdigest(),
    }


def dataset_snapshots(adapter):
    return [dataset_snapshot(item) for item in adapter.get("datasets", [])]


class StudySpec(Strict):
    batch_mode: Literal['grid', 'explicit'] | None = None
    configurations: list[dict[str, float]] = Field(default_factory=list, max_length=100)
    failure_policy: Literal['continue', 'pause'] = 'pause'
    project_name: str = Field(default='我的研究', min_length=1, max_length=100)
    adjustment_scope: Scope = 'baseline'
    evaluation_metrics: list[str] = Field(default_factory=list, max_length=30)
    metric_rules: list[MetricRule] = Field(default_factory=list, max_length=20)
    review_mode: bool = False
    parent_study_id: str | None = None
    researcher_note: str = Field(default="", max_length=3000)
    name: str = Field(min_length=1, max_length=160)
    trainer_id: str
    source_run_id: str | None = None
    allow_remote: bool = False
    parameters: dict[str, list[float]] = Field(min_length=1, max_length=24)
    objective: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,60}$")
    direction: Literal["minimize", "maximize"] = "minimize"
    constraints: dict[str, float] = Field(default_factory=dict, max_length=12)
    seeds: list[int] = Field(default=[11, 23, 37], min_length=1, max_length=10)
    max_trials: int = Field(default=8, ge=1, le=100)
    budget_seconds: int = Field(default=3600, ge=1, le=604800)
    trial_timeout: int = Field(default=600, ge=1, le=86400)
    protocol: str = Field(min_length=10, max_length=2500)
    sampler: Literal["random", "tpe"] = "random"
    search_seed: int = Field(default=42, ge=0, le=2147483647)
    controller: Literal["tpe", "api_bounded"] = "tpe"
    controller_max_calls: int = Field(default=0, ge=0, le=20)
    initial_parameters: dict[str, float] | None = None

    @model_validator(mode="after")
    def bounded(self):
        if not self.review_mode and not self.batch_mode and self.objective is None:
            raise ValueError('历史搜索模式需要单一目标；人工审核模式可不排序')
        if self.review_mode and ((self.adjustment_scope == 'baseline') == bool(self.parent_study_id)):
            raise ValueError('基线不能指定父轮次；调整或重复验证必须选择基准轮')
        if self.review_mode and (self.max_trials != 1 or any(len(v) != 1 for v in self.parameters.values()) or self.controller == "api_bounded"):
            raise ValueError("实验助手每轮只运行一组人工确认的参数；不能自动搜索或调用控制器")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seed 不能重复")
        for key, values in self.parameters.items():
            if (
                not values
                or len(values) > 100
                or any(not math.isfinite(v) for v in values)
            ):
                raise ValueError("参数候选须为有限数值，且每项最多 100 个")
            if len(values) != len(set(values)):
                raise ValueError("参数候选不能重复")
        if any(not math.isfinite(v) for v in self.constraints.values()):
            raise ValueError("约束必须有限")
        if self.controller == "api_bounded":
            if not self.allow_remote:
                raise ValueError("API 控制器需要明确允许外发本计划的汇总指标")
            if self.controller_max_calls < 1:
                raise ValueError("API 控制器至少需要一次受限调用额度")
        if self.initial_parameters is not None:
            if set(self.initial_parameters) != set(self.parameters):
                raise ValueError("初始参数必须覆盖且只覆盖本计划参数")
            for key, value in self.initial_parameters.items():
                if value not in self.parameters[key]:
                    raise ValueError("初始参数必须属于已冻结候选集")
        if self.batch_mode:
            if self.review_mode or self.parent_study_id or self.controller == 'api_bounded' or self.controller_max_calls or self.initial_parameters:
                raise ValueError('批次只执行预先确认的矩阵，不使用模型控制器、父轮次或建议模式')
            if self.batch_mode == 'grid' and self.configurations:
                raise ValueError('网格模式不能同时填写手工组合')
            from .batch_experiments import expand_matrix
            self.max_trials = len(expand_matrix(self.model_dump()))
        elif self.configurations:
            raise ValueError('手工组合仅用于批次模式')
        return self


def resolve_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("适配器文件不在登记目录内或不存在")
    return path


def trainers(store):
    directory = store.root / "trainers"
    directory.mkdir(exist_ok=True)
    result = []
    for file in directory.glob("*.json"):
        try:
            result.append(
                Trainer.model_validate_json(
                    file.read_text(encoding="utf-8")
                ).model_dump()
            )
        except ValueError:
            continue
    return result


def trainer(store, identity):
    matches = [x for x in trainers(store) if x["id"] == identity]
    if not matches:
        raise KeyError("未登记训练适配器，请先提供训练代码并使用本地注册工具")
    return matches[0]


def event(store, study_id, kind, payload):
    store.execute(
        "INSERT INTO study_events(study_id,time,type,payload) VALUES (?,?,?,?)",
        (study_id, now(), kind, dump(payload)),
    )


def create_study(store, spec, connection=None, prepare_only=False):
    adapter = trainer(store, spec.trainer_id)
    batch = None
    if spec.batch_mode:
        from .batch_experiments import preview
        batch = preview(spec, adapter)
    root = Path(adapter["root"]).resolve()
    if adapter["entrypoint"] not in adapter["source_files"]:
        raise ValueError("入口文件必须在代码快照清单中")
    if not set(spec.parameters) <= set(adapter["parameters"]):
        raise ValueError("参数不属于已登记适配器")
    for key, values in spec.parameters.items():
        if not set(values) <= set(adapter["parameters"][key]):
            raise ValueError("参数范围超出已登记边界")
    if (spec.objective is not None and spec.objective not in adapter["metrics"]) or not set(spec.constraints) <= set(
        adapter["metrics"]
    ):
        raise ValueError("评价指标不在适配器声明中")
    if spec.source_run_id:
        source = store.run(spec.source_run_id)
        if source["status"] != "completed":
            raise ValueError("只能绑定已完成的研究计划")
    if not set(spec.evaluation_metrics + [r.metric for r in spec.metric_rules]) <= set(adapter['metrics']):
        raise ValueError('观察指标或约束不在适配器声明中')
    if spec.review_mode and set(spec.parameters) != set(adapter['parameters']):
        raise ValueError('本轮必须提供适配器的完整参数')
    parent = get_study(store, spec.parent_study_id) if spec.parent_study_id else None
    if parent:
        check_derived(parent, spec, adapter)
    hashes = {}
    for name in adapter["source_files"] + adapter["dataset_files"]:
        path = resolve_file(root, name)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        hashes[name] = h.hexdigest()
    packages = {}
    for package in ("torch", "numpy", "optuna", "transformers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    manifest = {
        "adapter": adapter,
        "source_hashes": hashes,
        "datasets": dataset_snapshots(adapter),
        "loom_runtime_hashes": loom_runtime_hashes(),
        "packages": packages,
        "schema": "study-v1",
        "evaluation": adapter.get('evaluation_phase', 'unspecified'),
        "checkpoint_mode": "fresh_repeat_no_implicit_resume",
    }
    if batch:
        manifest['batch'] = batch
    sid = uid("study")
    if parent:
        for key in ('source_hashes', 'datasets', 'packages'):
            if manifest[key] != parent['manifest'].get(key):
                raise ValueError('代码、数据或环境已改变，请建立新基线')
    from .preflight import asset_snapshot
    manifest['assets'] = asset_snapshot(adapter)
    if parent and manifest['assets'] != parent['manifest'].get('assets', {}):
        raise ValueError('模型或资源文件已改变，请建立新基线')
    if prepare_only:
        return manifest
    return persist_study(store, spec, manifest, connection, sid)


def persist_study(store, spec, manifest, connection=None, sid=None):
    sid = sid or uid('study')
    with (nullcontext(connection) if connection is not None else store.db()) as db:
        db.execute(
            "INSERT INTO studies(id,spec,manifest,status,created) VALUES (?,?,?,?,?)",
            (sid, dump(spec.model_dump()), dump(manifest), "draft", now()),
        )
        db.execute(
            "INSERT INTO study_events(study_id,time,type,payload) VALUES (?,?,?,?)",
            (
                sid,
                now(),
                "planned",
                dump({"message": "计划已冻结，启动前可核对预算、参数和评价协议"}),
            ),
        )
    return {"id": sid} if connection is not None else get_study(store, sid)


def get_study(store, sid):
    rows = store.query("SELECT * FROM studies WHERE id=?", (sid,))
    if not rows:
        raise KeyError("实验计划不存在")
    result = rows[0]
    for key in ("spec", "manifest"):
        result[key] = json.loads(result[key])
    trials = store.query(
        "SELECT * FROM trials WHERE study_id=? ORDER BY number", (sid,)
    )
    for trial in trials:
        trial["params"] = json.loads(trial["params"])
        trial["metrics"] = json.loads(trial["metrics"]) if trial["metrics"] else None
        trial["repeats"] = store.query(
            "SELECT * FROM trial_attempts WHERE trial_id=? ORDER BY seed",
            (trial["id"],),
        )
        for repeat in trial["repeats"]:
            repeat["metrics"] = (
                json.loads(repeat["metrics"]) if repeat["metrics"] else None
            )
    result["trials"] = trials
    events = store.query(
        "SELECT * FROM study_events WHERE study_id=? ORDER BY seq", (sid,)
    )
    for e in events:
        e["payload"] = json.loads(e["payload"])
    result["events"] = events
    return result


def start_study(store, sid):
    from .preflight import check_preflight
    frozen = get_study(store, sid)
    if frozen['status'] != 'draft':
        raise IdempotencyConflict('计划已启动；重复启动不会产生额外试验')
    preflight = check_preflight(frozen['manifest']['adapter'], frozen['manifest'])
    event(store, sid, 'preflight', preflight)
    if not preflight['passed']:
        raise ValueError('训练预检失败：' + '; '.join(preflight['errors']))
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM studies WHERE id=?", (sid,)).fetchone()
        if not row:
            raise KeyError("实验计划不存在")
        if row["status"] != "draft":
            raise IdempotencyConflict("计划已启动；重复启动不会产生额外试验")
        if (
            db.execute(
                "SELECT COUNT(*) FROM studies WHERE status IN ('running','pausing')"
            ).fetchone()[0]
            >= 8
        ):
            raise OverflowError("实验队列已满，最多 8 个已启动计划")
        spec = json.loads(row["spec"])
        db.execute(
            "UPDATE studies SET status='running',deadline=? WHERE id=?",
            (time.time() + spec["budget_seconds"], sid),
        )
        db.execute(
            "INSERT INTO study_events(study_id,time,type,payload) VALUES (?,?,?,?)",
            (
                sid,
                now(),
                "started",
                dump({"message": "开始执行已确认计划，自动推进预算内试验"}),
            ),
        )


def next_parameters(study, store=None):
    if study['spec'].get('batch_mode'):
        matrix = study['manifest']['batch']['matrix']
        number = len(study['trials'])
        return matrix[number] if number < len(matrix) else None
    if study['spec'].get('review_mode'):
        return {k:v[0] for k,v in study['spec']['parameters'].items()} if not study['trials'] else None
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec = study["spec"]
    number = len(study["trials"])
    if number == 0 and spec.get("initial_parameters"):
        return spec["initial_parameters"]
    sampler = (
        optuna.samplers.RandomSampler(seed=spec["search_seed"] + number)
        if spec["sampler"] == "random"
        else optuna.samplers.TPESampler(
            seed=spec["search_seed"] + number, n_startup_trials=5
        )
    )
    distributions = {
        k: optuna.distributions.CategoricalDistribution(v)
        for k, v in spec["parameters"].items()
    }
    used = {json.dumps(t["params"], sort_keys=True) for t in study["trials"]}
    # The remote critic is an experiment reviewer, not an unconstrained
    # optimiser.  After the baseline it sees local one-parameter neighbours of
    # the best measured feasible configuration.  This preserves attribution:
    # a bad selection can be rejected from evidence without conflating a dozen
    # simultaneous reward/PPO changes.
    if spec.get("controller") == "api_bounded":
        completed = [
            t for t in study["trials"]
            if t["status"] == "completed" and t["metrics"] and t["metrics"]["feasible"]
        ]
        if completed:
            incumbent = sorted(
                completed,
                key=lambda t: t["metrics"]["objective_mean"],
                reverse=spec["direction"] == "maximize",
            )[0]["params"]
            candidates = []
            for key, values in spec["parameters"].items():
                for value in values:
                    if value == incumbent[key]:
                        continue
                    proposal = {**incumbent, key: value}
                    if json.dumps(proposal, sort_keys=True) not in used:
                        candidates.append(proposal)
                        if len(candidates) >= 8:
                            break
                if len(candidates) >= 8:
                    break
            if candidates:
                calls = sum(1 for e in study["events"] if e["type"] == "controller_decision")
                if calls < spec.get("controller_max_calls", 0) and store is not None:
                    from .adaptive_controller import choose_candidate

                    selected = choose_candidate(store, study, candidates)
                    if selected is not None:
                        return selected
                return candidates[0]
            return None
    optimizer = optuna.create_study(direction=spec["direction"], sampler=sampler)
    for t in study["trials"]:
        if t["status"] == "completed" and t["metrics"] and t["metrics"]["feasible"]:
            optimizer.add_trial(
                optuna.trial.create_trial(
                    params=t["params"],
                    distributions=distributions,
                    value=t["metrics"]["objective_mean"],
                )
            )
    candidates = []
    for _ in range(100):
        proposal = optimizer.ask(distributions).params
        if json.dumps(proposal, sort_keys=True) not in used and proposal not in candidates:
            candidates.append(proposal)
            if len(candidates) >= 8:
                break
    if not candidates:
        return None
    if candidates:
        return candidates[0]
    return None


def summarize(spec, repeats):
    values = [r["metrics"][spec["objective"]] for r in repeats] if spec.get('objective') else []
    checks = [{'seed':r.get('seed'), 'checks':check_rules(r['metrics'], spec)} for r in repeats]
    feasible = all(c['passed'] for r in checks for c in r['checks'])
    return {
        "objective_mean": statistics.mean(values) if values else None,
        "objective_std": statistics.stdev(values) if len(values) > 1 else None,
        "repeat_count": len(repeats),
        "feasible": feasible,
        "constraint_checks": checks,
        "metrics_std": {k: statistics.stdev([r['metrics'][k] for r in repeats]) if len(repeats)>1 else None
                        for k in repeats[0]['metrics'] if all(k in r['metrics'] for r in repeats)},
        "metrics_mean": {
            k: statistics.mean([r["metrics"][k] for r in repeats])
            for k in repeats[0]["metrics"]
            if all(k in r["metrics"] for r in repeats)
        },
    }


class AnalysisSpec(Strict):
    operation: Literal["rank", "failures", "summary", "sensitivity"] = "rank"
    parameter: str | None = None
    feasible_only: bool = True
    limit: int = Field(default=20, ge=1, le=100)


def analyze(store, sid, spec, identity=None):
    if identity:
        existing = store.query("SELECT result FROM analyses WHERE id=?", (identity,))
        if existing:
            return json.loads(existing[0]["result"])
    study = get_study(store, sid)
    trials = study["trials"]
    valid = [
        t
        for t in trials
        if t["status"] == "completed"
        and t["metrics"]
        and (not spec.feasible_only or t["metrics"]["feasible"])
    ]
    if spec.operation == "failures":
        rows = [
            {"trial_id": t["id"], "error": t["error"], "params": t["params"]}
            for t in trials
            if t["status"] == "failed"
        ]
    elif spec.operation == "summary":
        rows = [
            {"status": state, "count": sum(t["status"] == state for t in trials)}
            for state in ("queued", "running", "completed", "failed", "cancelled")
        ]
    elif spec.operation == "sensitivity":
        if not study['spec'].get('objective'):
            raise ValueError('未设置排序指标，请使用科研数据分析选择观察指标')
        if spec.parameter not in study["spec"]["parameters"]:
            raise ValueError("请选择已登记参数")
        rows = []
        for v in study["spec"]["parameters"][spec.parameter]:
            group = [t for t in valid if t["params"][spec.parameter] == v]
            if group:
                rows.append(
                    {
                        "parameter": spec.parameter,
                        "value": v,
                        "mean": statistics.mean(
                            t["metrics"]["objective_mean"] for t in group
                        ),
                        "trials": [t["id"] for t in group],
                        "note": "探索性关联，其他参数可能变化，不能据此推断因果",
                    }
                )
    else:
        if not study['spec'].get('objective'):
            raise ValueError('此轮未设置排序指标，请在科研数据分析中显式选择指标后排序')
        valid.sort(
            key=lambda t: t["metrics"]["objective_mean"],
            reverse=study["spec"]["direction"] == "maximize",
        )
        rows = [
            {"trial_id": t["id"], "params": t["params"], **t["metrics"]} for t in valid
        ]
    snapshot = hashlib.sha256(dump(trials).encode()).hexdigest()
    aid = identity or uid("analysis")
    result = {
        "id": aid,
        "study_id": sid,
        "operation": spec.model_dump(),
        "rows": rows[: spec.limit],
        "snapshot": snapshot,
        "lineage": [t["id"] for t in trials],
        "protocol": study["spec"]["protocol"],
        "note": "仅分析本计划记录，不跨实验协议排名；当前结果不是独立测试集结论。",
    }
    store.execute(
        "INSERT INTO analyses VALUES (?,?,?,?,?)",
        (aid, sid, dump(spec.model_dump()), dump(result), now()),
    )
    return result
