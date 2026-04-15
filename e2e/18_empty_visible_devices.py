"""
Scenario 18: empty CUDA_VISIBLE_DEVICES — visible count is 0, total unchanged.

Run with: CUDA_VISIBLE_DEVICES= python e2e/18_empty_visible_devices.py
"""

import gpuq

assert gpuq.count(visible_only=True) == 0, (
    f"Expected 0 visible, got {gpuq.count(visible_only=True)}"
)

total = gpuq.count(visible_only=False)
assert total == 8, f"Expected 8 total, got {total}"

gpus = gpuq.query(visible_only=True)
assert len(gpus) == 0, f"Expected empty query, got {len(gpus)}"

print("OK")
