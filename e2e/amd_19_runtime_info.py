"""
AMD Scenario 19: hip_info runtime properties.

Run with: python e2e/amd_19_runtime_info.py
"""

import gpuq

g = gpuq.get(0, visible_only=False)
print(f"provider: {g.provider.name}")

assert g.provider == gpuq.Provider.HIP, f"Expected HIP provider, got {g.provider.name}"
info = g.hip_info
print(f"hip_info: {info}")
assert info is not None, "hip_info should not be None for HIP GPU"
assert g.cuda_info is None, "cuda_info should be None for HIP GPU"

print("OK")
