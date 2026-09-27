"""Bounded remote critic for post-trial experiment selection.

The API receives only a compact, user-authorised summary of completed local
trials.  It chooses an identifier from a server-generated candidate set; it
cannot write code, access a checkpoint, alter the data set, or invent a value
outside the frozen study manifest.  A malformed/unavailable API falls back to
the deterministic first candidate and leaves an audit event.
"""

import hashlib
import json
from pathlib import Path

from .config import Settings
from .routing import RemoteModel
from .store import dump, now


def _digest(value):
    return hashlib.sha256(dump(value).encode()).hexdigest()


def _daily_trace(store, attempt_id):
    path = store.root / "experiments" / attempt_id / "daily_metrics.json"
    if not path.is_file() or path.stat().st_size > 1_000_000:
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(rows, list):
        return []
    allowed = {
        "day", "daily_reward", "mean_step_reward", "grid_energy_kwh",
        "solar_used_kwh", "completed_tasks_delta", "completed_tasks_total",
        "queue_size", "migrations_delta", "max_gpu_temp_c",
        "mean_gpu_temp_c", "sla_violation_rate", "reward_components_mean",
    }
    return [{k: row[k] for k in allowed if k in row} for row in rows[-10:] if isinstance(row, dict)]


def _history(store, study):
    rows = []
    for trial in study["trials"]:
        if trial["status"] != "completed" or not trial["metrics"]:
            continue
        traces = [
            _daily_trace(store, repeat["id"])
            for repeat in trial["repeats"]
            if repeat["status"] == "completed"
        ]
        rows.append(
            {
                "trial_id": trial["id"],
                "parameters": trial["params"],
                "aggregate": {
                    key: value
                    for key, value in trial["metrics"].get("metrics_mean", {}).items()
                    if key
                    in {
                        "evaluation_loss", "cumulative_grid_energy_kwh",
                        "max_gpu_temp_c", "sla_violation_rate", "mean_reward",
                        "reward_slope_per_day", "completed_tasks", "queue_size",
                        "migration_count",
                    }
                },
                "daily_traces": traces,
            }
        )
    return rows[-8:]


def _event(store, study_id, payload):
    store.execute(
        "INSERT INTO study_events(study_id,time,type,payload) VALUES (?,?,?,?)",
        (study_id, now(), "controller_decision", dump(payload)),
    )


def choose_candidate(store, study, candidates):
    """Return a candidate chosen by DeepSeek, or None for local fallback."""
    spec = study["spec"]
    if not spec.get("allow_remote"):
        return None
    settings = Settings()
    remote = RemoteModel(settings)
    completed = [
        trial for trial in study["trials"]
        if trial["status"] == "completed" and trial["metrics"] and trial["metrics"]["feasible"]
    ]
    incumbent = sorted(
        completed,
        key=lambda trial: trial["metrics"]["objective_mean"],
        reverse=spec["direction"] == "maximize",
    )[0]["params"] if completed else {}
    indexed = [
        {
            "candidate_id": f"candidate_{index}",
            "parameters": value,
            "changed_parameters": [
                key
                for key, candidate_value in value.items()
                if candidate_value != incumbent.get(key)
            ],
        }
        for index, value in enumerate(candidates)
    ]
    request = {
        "protocol": spec["protocol"],
        "objective": spec["objective"],
        "direction": spec["direction"],
        "constraints": spec["constraints"],
        "history": _history(store, study),
        "candidates": indexed,
    }
    prompt = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    base_audit = {
        "controller": "api_bounded-v1",
        "input_sha256": _digest(request),
        "candidate_count": len(indexed),
        "remote_configured": remote.configured,
        "model": settings.remote_model or None,
        "request_options": remote.request_options() if remote.configured else {},
    }
    if not remote.configured:
        _event(store, study["id"], {**base_audit, "outcome": "fallback", "reason": "API unavailable"})
        return None
    system = (
        "You are an experiment critic. Select exactly one candidate_id from the "
        "provided candidates using only the completed local-trial summaries. "
        "Do not claim significance, do not change the protocol, and do not invent "
        "parameters. Reward values under different reward profiles are diagnostic "
        "only; rank by the declared external objective and constraints. Return a JSON "
        "object with candidate_id, rationale (at most 120 Chinese characters), and "
        "diagnostic_signals (a short array)."
    )
    try:
        text, usage = remote.generate(system, prompt, max_tokens=8192)
        parsed = json.loads(text)
        selected = parsed.get("candidate_id") if isinstance(parsed, dict) else None
        match = next((x for x in indexed if x["candidate_id"] == selected), None)
        if match is None:
            raise ValueError("API returned an unknown candidate_id")
        _event(
            store,
            study["id"],
            {
                **base_audit,
                "outcome": "selected",
                "selected_candidate_id": selected,
                "selected_parameters": match["parameters"],
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "response": {
                    "rationale": str(parsed.get("rationale", ""))[:500],
                    "diagnostic_signals": parsed.get("diagnostic_signals", [])[:8],
                },
                "usage": usage,
            },
        )
        return match["parameters"]
    except (ValueError, TypeError, json.JSONDecodeError, ConnectionError, TimeoutError) as exc:
        _event(
            store,
            study["id"],
            {**base_audit, "outcome": "fallback", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"},
        )
        return None
