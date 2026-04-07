"""
AMD Scenario 1: gpuq sees all GPUs first, then torch initialises.

gpuq calls save_visible() which removes HIP_VISIBLE_DEVICES before touching
the HIP runtime, so the runtime initialises with all physical GPUs visible.
Torch initialises afterwards into that already-initialised runtime.

EXPECTED: PASS — gpuq initialises HIP runtime first (without restriction).

Run with: HIP_VISIBLE_DEVICES=0 python e2e/amd_01_gpuq_all_then_torch_then_gpuq_visible.py
"""

import subprocess, sys, os

# Detect total GPU count via a clean subprocess (no HIP_VISIBLE_DEVICES).
_env = {k: v for k, v in os.environ.items()
        if k not in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}
TOTAL = int(subprocess.check_output(
    [sys.executable, '-c', 'import gpuq; print(gpuq.count(visible_only=False))'],
    env=_env).decode().strip())

print(f"System has {TOTAL} AMD GPU(s), VISIBLE=1 (HIP_VISIBLE_DEVICES=0)")

import gpuq

gpus = gpuq.query(gpuq.Provider.HIP, visible_only=False)
assert len(gpus) == TOTAL, f"gpuq.query(all): expected {TOTAL}, got {len(gpus)}"

import torch
torch.cuda.init()  # on ROCm this initialises the HIP runtime
assert torch.cuda.device_count() == 1, \
    f"torch.cuda.device_count(): expected 1, got {torch.cuda.device_count()}"

assert gpuq.count(visible_only=True) == 1, \
    f"gpuq.count(visible=True): expected 1, got {gpuq.count(visible_only=True)}"
assert gpuq.count(visible_only=False) == TOTAL, \
    f"gpuq.count(visible=False): expected {TOTAL}, got {gpuq.count(visible_only=False)}"

print("OK")
