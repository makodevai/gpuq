"""
Scenario 4: torch before and after gpuq — gpuq's save_visible temporarily removes
CUDA_VISIBLE_DEVICES; verify torch's device_count is unaffected on both sides.

Run with: CUDA_VISIBLE_DEVICES=0 python e2e/04_torch_sandwich.py
"""

import torch
import gpuq

# torch before gpuq
torch.cuda.init()
assert torch.cuda.device_count() == 1, f"torch before gpuq: expected 1, got {torch.cuda.device_count()}"

# gpuq in the middle
assert gpuq.count(visible_only=True) == 1, f"gpuq visible: expected 1, got {gpuq.count(visible_only=True)}"
assert gpuq.count(visible_only=False) == 8, f"gpuq total: expected 8, got {gpuq.count(visible_only=False)}"

gpus = gpuq.query(visible_only=False)
assert len(gpus) == 8, f"gpuq query all: expected 8, got {len(gpus)}"

# torch after gpuq — env var must have been restored
assert torch.cuda.device_count() == 1, f"torch after gpuq: expected 1, got {torch.cuda.device_count()}"

print("OK")
