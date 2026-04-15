"""
Scenario 15: provider-specific count.

Run with: python e2e/15_provider_filtering.py
"""

import gpuq

cuda_count = gpuq.count(gpuq.Provider.CUDA, visible_only=False)
hip_count = gpuq.count(gpuq.Provider.HIP, visible_only=False)
total = gpuq.count(visible_only=False)
print(f"cuda={cuda_count}, hip={hip_count}, total={total}")

assert cuda_count + hip_count == total, (
    f"cuda ({cuda_count}) + hip ({hip_count}) != total ({total})"
)

print("OK")
