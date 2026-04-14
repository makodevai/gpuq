"""
Scenario 19: cuda_info / hip_info runtime properties.

Run with: python e2e/19_runtime_info.py
"""

import gpuq

g = gpuq.get(0, visible_only=False)
print(f"provider: {g.provider.name}")

if g.provider == gpuq.Provider.CUDA:
    info = g.cuda_info
    print(f"cuda_info: {info}")
    assert info is not None, "cuda_info should not be None for CUDA GPU"
    assert g.hip_info is None, "hip_info should be None for CUDA GPU"
elif g.provider == gpuq.Provider.HIP:
    info = g.hip_info
    print(f"hip_info: {info}")
    assert info is not None, "hip_info should not be None for HIP GPU"
    assert g.cuda_info is None, "cuda_info should be None for HIP GPU"

print("OK")
