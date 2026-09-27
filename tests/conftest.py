"""测试不读取开发者的真实服务密钥或本机供应商配置。"""

import pytest


@pytest.fixture(autouse=True)
def isolate_remote_settings(monkeypatch, tmp_path):
    for name in (
        "URL",
        "MODEL",
        "KEY",
        "VISION",
        "THINKING",
        "REASONING_EFFORT",
        "THINKING_TOKENS",
        "OUTPUT_BUDGET",
    ):
        monkeypatch.delenv("LOOM_REMOTE_" + name, raising=False)
    monkeypatch.setenv("LOOM_REMOTE_CONFIG", str(tmp_path / "no-remote-config.toml"))
