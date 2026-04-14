"""
Scenario 17: required= raises RuntimeError for missing provider.

Run with: python e2e/17_required_raises.py

On NVIDIA box: requires HIP (should fail).
On AMD box: requires CUDA (should fail).
"""

import gpuq

if gpuq.hascuda() and not gpuq.hasamd():
    missing = gpuq.Provider.HIP
elif gpuq.hasamd() and not gpuq.hascuda():
    missing = gpuq.Provider.CUDA
else:
    print("SKIP: both providers present, cannot test missing provider")
    exit(0)

try:
    gpuq.query(required=missing)
    assert False, f"Should have raised RuntimeError for missing {missing.name}"
except RuntimeError as e:
    print(f"Correctly raised: {e}")

print("OK")
