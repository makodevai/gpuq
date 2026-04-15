"""
Scenario 2: torch initialises first (libcudart sees only 1 GPU via CUDA_VISIBLE_DEVICES),
then gpuq — does torch's early init prevent gpuq from seeing all 8 GPUs?

Run with: CUDA_VISIBLE_DEVICES=0 python e2e/02_torch_then_gpuq.py
"""

import torch

torch.cuda.init()
assert torch.cuda.device_count() == 1, (
    f"Expected torch to see 1 GPU, got {torch.cuda.device_count()}"
)

import gpuq

assert gpuq.count(visible_only=False) == 8, (
    f"Expected gpuq total=8, got {gpuq.count(visible_only=False)}"
)
assert gpuq.count(visible_only=True) == 1, (
    f"Expected gpuq visible=1, got {gpuq.count(visible_only=True)}"
)

gpus_all = gpuq.query(visible_only=False)
assert len(gpus_all) == 8, f"Expected 8 total GPUs from query, got {len(gpus_all)}"

gpus_visible = gpuq.query(visible_only=True)
assert len(gpus_visible) == 1, (
    f"Expected 1 visible GPU from query, got {len(gpus_visible)}"
)

print("OK")
