"""
AMD Scenario 3: gpuq and torch calls interleaved — gpuq goes first each time.

gpuq's visible count call triggers save_visible() → HIP runtime initialises
with all GPUs.  Torch follows into the already-full runtime.  Subsequent gpuq
calls still see all GPUs.  gpuq's save_visible() temporarily removes
HIP_VISIBLE_DEVICES; this must not leave torch seeing the wrong count.

EXPECTED: PASS — gpuq initialises the HIP runtime before torch in each round.

Run with: HIP_VISIBLE_DEVICES=0 python e2e/amd_03_interleaved.py
"""

import subprocess, sys, os

_env = {k: v for k, v in os.environ.items()
        if k not in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}
TOTAL = int(subprocess.check_output(
    [sys.executable, '-c', 'import gpuq; print(gpuq.count(visible_only=False))'],
    env=_env).decode().strip())

print(f"System has {TOTAL} AMD GPU(s), VISIBLE=1 (HIP_VISIBLE_DEVICES=0)")

import gpuq
import torch

# gpuq first: triggers save_visible + HIP runtime init with all GPUs
assert gpuq.count(visible_only=True) == 1, \
    f"gpuq.count(visible=True): expected 1, got {gpuq.count(visible_only=True)}"

torch.cuda.init()
assert torch.cuda.device_count() == 1, \
    f"torch after gpuq init: expected 1, got {torch.cuda.device_count()}"

assert gpuq.count(visible_only=False) == TOTAL, \
    f"gpuq.count(visible=False): expected {TOTAL}, got {gpuq.count(visible_only=False)}"

# gpuq's save_visible temporarily removed HIP_VISIBLE_DEVICES — must be restored
assert torch.cuda.device_count() == 1, \
    f"torch after gpuq all-count: expected 1, got {torch.cuda.device_count()}"

assert gpuq.count(visible_only=True) == 1, \
    f"gpuq.count(visible=True) at end: expected 1, got {gpuq.count(visible_only=True)}"

print("OK")
