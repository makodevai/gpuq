"""
Scenario 1: gpuq sees all GPUs first (removes CUDA_VISIBLE_DEVICES internally),
then torch initialises — does gpuq's early libcudart init break torch's visible count?

Run with: CUDA_VISIBLE_DEVICES=0 python e2e/01_gpuq_all_then_torch_then_gpuq_visible.py
"""

import gpuq

gpus = gpuq.query(visible_only=False)
assert len(gpus) == 8, f"Expected 8 total GPUs, got {len(gpus)}"

import torch

torch.cuda.init()
assert torch.cuda.device_count() == 1, (
    f"Expected torch to see 1 GPU, got {torch.cuda.device_count()}"
)

assert gpuq.count(visible_only=True) == 1, (
    f"Expected gpuq visible=1, got {gpuq.count(visible_only=True)}"
)
assert gpuq.count(visible_only=False) == 8, (
    f"Expected gpuq total=8, got {gpuq.count(visible_only=False)}"
)

print("OK")
