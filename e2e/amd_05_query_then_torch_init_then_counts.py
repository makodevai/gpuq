"""
AMD Scenario 5: mirrors the original bug report — gpuq.query(visible_only=False)
first, then torch.cuda.init(), then both counts.

gpuq initialises the HIP runtime first (via save_visible, env var removed), so
torch sees all GPUs at init time — but still reports only 1 because HIP_VISIBLE_DEVICES
is restored before torch runs.

EXPECTED: PASS — gpuq initialises HIP runtime before torch.

Run with: HIP_VISIBLE_DEVICES=0 python e2e/amd_05_query_then_torch_init_then_counts.py
"""

import subprocess, sys, os

_env = {
    k: v
    for k, v in os.environ.items()
    if k not in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
}
TOTAL = int(
    subprocess.check_output(
        [sys.executable, "-c", "import gpuq; print(gpuq.count(visible_only=False))"],
        env=_env,
    )
    .decode()
    .strip()
)

print(f"System has {TOTAL} AMD GPU(s), VISIBLE=1 (HIP_VISIBLE_DEVICES=0)")

import gpuq

gpus = gpuq.query(gpuq.Provider.HIP, visible_only=False)
assert len(gpus) == TOTAL, f"gpuq.query(all): expected {TOTAL}, got {len(gpus)}"

import torch

torch.cuda.init()
assert torch.cuda.device_count() == 1, (
    f"torch.cuda.device_count(): expected 1, got {torch.cuda.device_count()}"
)

assert gpuq.count(visible_only=True) == 1, (
    f"gpuq.count(visible=True): expected 1, got {gpuq.count(visible_only=True)}"
)
assert gpuq.count(visible_only=False) == TOTAL, (
    f"gpuq.count(visible=False): expected {TOTAL}, got {gpuq.count(visible_only=False)}"
)

print("OK")
