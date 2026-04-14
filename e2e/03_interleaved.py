"""
Scenario 3: interleaved calls — gpuq and torch alternate, checking neither corrupts the other.

Run with: CUDA_VISIBLE_DEVICES=0 python e2e/03_interleaved.py
"""

import gpuq
import torch

# gpuq visible count first (this internally calls save_visible + cudaGetDeviceCount)
assert gpuq.count(visible_only=True) == 1, f"Expected gpuq visible=1, got {gpuq.count(visible_only=True)}"

# torch init after gpuq has already touched libcudart
torch.cuda.init()
assert torch.cuda.device_count() == 1, f"Expected torch to see 1 GPU, got {torch.cuda.device_count()}"

# gpuq all-count after torch init
assert gpuq.count(visible_only=False) == 8, f"Expected gpuq total=8, got {gpuq.count(visible_only=False)}"

# torch again — gpuq's save_visible temporarily removed the env var; make sure it was restored
assert torch.cuda.device_count() == 1, f"Expected torch still 1 GPU after gpuq calls, got {torch.cuda.device_count()}"

# final gpuq visible
assert gpuq.count(visible_only=True) == 1, f"Expected gpuq visible=1 at end, got {gpuq.count(visible_only=True)}"

print("OK")
