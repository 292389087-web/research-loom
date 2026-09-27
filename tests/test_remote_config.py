import json
from pathlib import Path
import httpx
import pytest
from loom.config import ROOT, Settings
from loom.routing import RemoteModel, ModelRouter
from loom.store import Store
from loom.execution import StageExecutor
from loom.experiments import AnalysisSpec


def preset(tmp_path, monkeypatch):
    # A developer may have a real key in the parent shell.  This fixture tests
    # the file path explicitly and must never inherit that secret.
    for name in (
        "LOOM_REMOTE_KEY", "LOOM_REMOTE_URL", "LOOM_REMOTE_MODEL",
        "LOOM_REMOTE_VISION", "LOOM_REMOTE_THINKING",
        "LOOM_REMOTE_REASONING_EFFORT", "LOOM_REMOTE_THINKING_TOKENS",
        "LOOM_REMOTE_OUTPUT_BUDGET",
    ):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "deepseek.toml"
    template = (ROOT / "config/deepseek.example.toml").read_text(encoding="utf-8")
    path.write_text(
        template.replace('api_key = ""', 'api_key = "fixture-key"'), encoding="utf-8"
    )
    monkeypatch.setenv("LOOM_REMOTE_CONFIG", str(path))
    return path


def test_only_key_needed_and_env_override(tmp_path, monkeypatch):
    preset(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert (
        s.remote_model == "deepseek-flash"
        and s.remote_url == "https://api.deepseek.com"
    )
    assert (
        s.remote_key == "fixture-key"
        and s.remote_vision
        and s.remote_output_budget == 131072
    )
    assert "fixture-key" not in repr(s)
    monkeypatch.setenv("LOOM_REMOTE_KEY", "")
    assert not RemoteModel(Settings()).configured


def test_invalid_config_does_not_echo_secret(tmp_path, monkeypatch):
    path = preset(tmp_path, monkeypatch)
    path.write_text("[remote]\napi_key = fixture-secret-invalid-toml", encoding="utf-8")
    with pytest.raises(ValueError) as err:
        Settings()
    assert "fixture-secret" not in str(err.value)


def test_deepseek_reasoning_request_ledger_and_budget(tmp_path, monkeypatch):
    preset(tmp_path, monkeypatch)
    s = Settings(data_dir=tmp_path / "data", use_dense=False)
    store = Store(s.data_dir)
    seen = []

    def respond(req):
        seen.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "model": "deepseek-flash",
                "choices": [
                    {
                        "message": {"content": '{"operation":"summary"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 200,
                    "completion_tokens_details": {"reasoning_tokens": 150},
                },
            },
        )

    router = ModelRouter(
        store, s, object(), RemoteModel(s, httpx.MockTransport(respond))
    )
    rid = store.create_run("analysis", {"allow_remote": True, "model_policy": "api"})
    _, epoch = store.claim("test", 30, rid)
    cp = {"calls": 0, "usage": []}
    with store.leased(rid, "test", epoch):
        executor = StageExecutor(store, s, router)
        executor.generate(
            rid,
            cp,
            "分析计划",
            "JSON only",
            "Return a JSON plan",
            AnalysisSpec,
            "test",
            lambda *x: None,
            lambda: None,
            max_tokens=400,
        )
        assert cp["remote_output_reserved"] == 32768
        s.remote_output_budget = 32768
        with pytest.raises(ValueError, match="预算"):
            executor.generate(
                rid,
                cp,
                "another",
                "JSON only",
                "JSON",
                AnalysisSpec,
                "test",
                lambda *x: None,
                lambda: None,
                max_tokens=400,
            )
    assert len(seen) == 1 and seen[0]["max_tokens"] == 32768
    assert (
        seen[0]["thinking"] == {"type": "enabled"}
        and seen[0]["reasoning_effort"] == "high"
    )
    assert cp["usage"][0]["reasoning_tokens"] == 150
    assert "fixture-key" not in json.dumps(
        store.query("SELECT input,usage FROM model_attempts")
    )
