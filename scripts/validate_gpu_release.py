"""一次真实本地生成，随后交接给一个 CUDA 分配探针。不是科研训练结果。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import torch
from loom.config import Settings
from loom.store import Store
from loom.model import LocalModel
from loom.routing import ModelRouter, RoutedModel
from loom.gpu import gpu_slot, gpu_busy

settings = Settings()
store = Store(settings.data_dir)
router = ModelRouter(store, settings, LocalModel(settings))
before = {
    "allocated_bytes": torch.cuda.memory_allocated(),
    "reserved_bytes": torch.cuda.memory_reserved(),
}
decision = router.choose({}, "release-check", "Return JSON", busy=False)
text, usage = RoutedModel(router, decision).generate(
    "Return a JSON object only.", 'Return {"gpu_handoff":"checked"}.', max_tokens=40
)
released = (
    usage["after_unload"]["allocated_bytes"] < 32 * 1024**2
    and usage["after_unload"]["reserved_bytes"] < 64 * 1024**2
)
assert not gpu_busy(store.root)
with gpu_slot(store.root):
    probe = torch.ones(1024 * 1024, device="cuda")
    torch.cuda.synchronize()
    success = probe.sum().item() == 1024 * 1024
    del probe
    torch.cuda.empty_cache()
result = {
    "purpose": "engineering validation, not paper training",
    "before": before,
    "route": decision,
    "output": text,
    "usage": usage,
    "weights_released_within_threshold": released,
    "thresholds": {"allocated_bytes": 32 * 1024**2, "reserved_bytes": 64 * 1024**2},
    "next_cuda_allocation_succeeded": success,
    "limitation": "PyTorch allocator measures this process; CUDA context and other applications are not counted. Small retained allocations are measured, not treated as zero.",
}
target = store.root / "evals" / "gpu-release-validation.json"
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
assert success and released
