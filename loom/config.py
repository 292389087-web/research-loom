from dataclasses import dataclass, field
from pathlib import Path
import os
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def remote_option(name, env_name, default):
    """环境变量优先；所有服务从同一文件读取，不依赖当前工作目录。"""
    if env_name in os.environ:
        raw = os.environ[env_name]
    else:
        path = Path(
            os.environ.get(
                "LOOM_REMOTE_CONFIG", ROOT / "config" / "deepseek.local.toml"
            )
        )
        try:
            raw = (
                tomllib.loads(path.read_text(encoding="utf-8-sig"))
                .get("remote", {})
                .get(name, default)
                if path.is_file()
                else default
            )
        except (OSError, ValueError, AttributeError):
            raise ValueError(
                "远程配置文件无法读取或 TOML 格式错误，请检查 config/deepseek.local.toml"
            ) from None
    if isinstance(default, bool):
        if raw in (True, "1", "true"):
            return True
        if raw in (False, "0", "false"):
            return False
        raise ValueError(f"远程配置 {name} 必须为布尔值")
    if isinstance(default, int):
        try:
            value = int(raw)
        except (ValueError, TypeError):
            raise ValueError(f"远程配置 {name} 必须为整数") from None
        if not 0 <= value <= 393216:
            raise ValueError(f"远程配置 {name} 超出范围")
        return value
    if not isinstance(raw, str):
        raise ValueError(f"远程配置 {name} 必须为字符串")
    return raw.strip()


def cached_model(name: str, env_name: str) -> Path | None:
    """只找本地权重，不依赖旧项目代码，也不自动下载。"""
    explicit = os.environ.get(env_name)
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if (path / "config.json").is_file() else None
    cache = Path.home() / ".cache"
    patterns = [
        (
            cache / "modelscope" / "models",
            f"{name.replace('/', '--')}/snapshots/*/config.json",
        ),
        (
            cache / "huggingface" / "hub",
            f"models--{name.replace('/', '--')}/snapshots/*/config.json",
        ),
    ]
    for base, pattern in patterns:
        found = sorted(base.glob(pattern))
        if found:
            return found[-1].parent
    return None


@dataclass
class Settings:
    demo_mode: bool = field(default_factory=lambda: os.environ.get('LOOM_DEMO','0') == '1')
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("LOOM_DATA_DIR", ROOT / "data"))
    )
    max_upload_bytes: int = 40 * 1024 * 1024
    max_pages: int = 100
    queue_limit: int = 12
    max_model_calls: int = 8
    generation_timeout: int = 150
    max_image_pixels: int = 786432
    use_dense: bool = field(
        default_factory=lambda: os.environ.get("LOOM_DENSE", "1") == "1"
    )
    worker_enabled: bool = field(
        default_factory=lambda: os.environ.get("LOOM_WORKER", "1") == "1"
    )
    lease_seconds: float = 30
    task_timeout: float = 3600
    max_recoveries: int = 2
    retry_delay: float = 5
    api_token: str = field(default_factory=lambda: os.environ.get("LOOM_API_TOKEN", ""))
    ingress_limit: int = 64
    sse_limit: int = 16
    allowed_hosts: tuple = ("127.0.0.1", "localhost", "[::1]")
    stall_timeout: float = 600
    remote_url: str = field(
        default_factory=lambda: remote_option("base_url", "LOOM_REMOTE_URL", "")
    )
    remote_model: str = field(
        default_factory=lambda: remote_option("model", "LOOM_REMOTE_MODEL", "")
    )
    remote_key: str = field(
        default_factory=lambda: remote_option("api_key", "LOOM_REMOTE_KEY", ""),
        repr=False,
    )
    remote_vision: bool = field(
        default_factory=lambda: remote_option("vision", "LOOM_REMOTE_VISION", False)
    )
    remote_thinking: str = field(
        default_factory=lambda: remote_option(
            "thinking", "LOOM_REMOTE_THINKING", "enabled"
        )
    )
    remote_reasoning_effort: str = field(
        default_factory=lambda: remote_option(
            "reasoning_effort", "LOOM_REMOTE_REASONING_EFFORT", "low"
        )
    )
    remote_thinking_tokens: int = field(
        default_factory=lambda: remote_option(
            "thinking_max_tokens", "LOOM_REMOTE_THINKING_TOKENS", 8192
        )
    )
    remote_call_limit: int = 4
    remote_output_budget: int = field(
        default_factory=lambda: remote_option(
            "output_token_budget", "LOOM_REMOTE_OUTPUT_BUDGET", 7000
        )
    )

    @property
    def model_path(self):
        return cached_model("Qwen/Qwen3-VL-4B-Instruct", "LOOM_MODEL_PATH")

    @property
    def embedding_path(self):
        return cached_model("Xorbits/bge-m3", "LOOM_EMBEDDING_PATH") or cached_model(
            "BAAI/bge-m3", "LOOM_EMBEDDING_PATH"
        )
