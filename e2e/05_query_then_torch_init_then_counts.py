"""
Scenario 5: mirrors the original failing test from the bug report.
gpuq.query(visible_only=False) first, then explicit torch.cuda.init(), then both counts.

Run with: CUDA_VISIBLE_DEVICES=0 python e2e/05_query_then_torch_init_then_counts.py
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
