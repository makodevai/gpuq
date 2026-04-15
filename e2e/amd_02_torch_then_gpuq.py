"""
AMD Scenario 2: torch initialises HIP runtime first (with HIP_VISIBLE_DEVICES=0,
so only 1 GPU), then gpuq tries to count all GPUs.

Unlike the CUDA side (which has an NVML backend to bypass the runtime), gpuq's
AMD path has no ROCm-SMI equivalent yet.  The HIP runtime is already initialised
with 1 GPU when gpuq calls amdGetDeviceCount(), so it also sees only 1.

EXPECTED: FAIL on gpuq.count(visible_only=False) until a ROCm-SMI backend is added.

Run with: HIP_VISIBLE_DEVICES=0 python e2e/amd_02_torch_then_gpuq.py
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

torch.cuda.init()  # on ROCm this initialises the HIP runtime with 1 GPU
assert torch.cuda.device_count() == 1, (
    f"torch.cuda.device_count(): expected 1, got {torch.cuda.device_count()}"
)

import gpuq

assert gpuq.count(visible_only=False) == TOTAL, (
    f"gpuq.count(visible=False): expected {TOTAL}, got {gpuq.count(visible_only=False)}"
)
assert gpuq.count(visible_only=True) == 1, (
    f"gpuq.count(visible=True): expected 1, got {gpuq.count(visible_only=True)}"
)

gpus_all = gpuq.query(gpuq.Provider.HIP, visible_only=False)
assert len(gpus_all) == TOTAL, f"gpuq.query(all): expected {TOTAL}, got {len(gpus_all)}"

gpus_visible = gpuq.query(gpuq.Provider.HIP, visible_only=True)
assert len(gpus_visible) == 1, (
    f"gpuq.query(visible): expected 1, got {len(gpus_visible)}"
)

print("OK")
