"""
Scenario 20: non-monotonic CUDA_VISIBLE_DEVICES — gpuq sorts visible indices.

Run with: CUDA_VISIBLE_DEVICES=5,1,3 python e2e/20_nonmonotonic_ordering.py
"""

import gpuq

gpus = gpuq.query(visible_only=True)
print([(g.index, g.system_index) for g in gpus])

# gpuq sorts: [5,1,3] -> [1,3,5] -> local 0,1,2
assert len(gpus) == 3, f"Expected 3 visible, got {len(gpus)}"
assert gpus[0].system_index == 1 and gpus[0].index == 0
assert gpus[1].system_index == 3 and gpus[1].index == 1
assert gpus[2].system_index == 5 and gpus[2].index == 2

print("OK")
