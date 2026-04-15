"""
AMD Scenario 4: torch before and after gpuq — same as CUDA scenario 4 but on HIP.

torch initialises HIP runtime first (1 GPU).  gpuq then tries to count all GPUs.
Without a ROCm-SMI backend, gpuq's amdGetDeviceCount() sees the already-restricted
runtime and returns 1.

EXPECTED: FAIL on gpuq.count(visible_only=False) until a ROCm-SMI backend is added.

Run with: HIP_VISIBLE_DEVICES=0 python e2e/amd_04_torch_sandwich.py
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

import torch
import gpuq

torch.cuda.init()
assert torch.cuda.device_count() == 1, (
    f"torch before gpuq: expected 1, got {torch.cuda.device_count()}"
)

assert gpuq.count(visible_only=True) == 1, (
    f"gpuq.count(visible=True): expected 1, got {gpuq.count(visible_only=True)}"
)
assert gpuq.count(visible_only=False) == TOTAL, (
    f"gpuq.count(visible=False): expected {TOTAL}, got {gpuq.count(visible_only=False)}"
)

gpus = gpuq.query(gpuq.Provider.HIP, visible_only=False)
assert len(gpus) == TOTAL, f"gpuq.query(all): expected {TOTAL}, got {len(gpus)}"

# gpuq's save_visible temporarily removed HIP_VISIBLE_DEVICES — must be restored
assert torch.cuda.device_count() == 1, (
    f"torch after gpuq: expected 1, got {torch.cuda.device_count()}"
)

print("OK")
