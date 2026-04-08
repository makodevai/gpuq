"""
Scenario 13: get() — single GPU by visible index.

Run with: CUDA_VISIBLE_DEVICES=2,7 python e2e/13_get_single_gpu.py
"""

import gpuq

g0 = gpuq.get(0, visible_only=True)
print(repr(g0))
assert g0.index == 0 and g0.system_index == 2, \
    f"get(0): expected (0, 2), got ({g0.index}, {g0.system_index})"

g1 = gpuq.get(1, visible_only=True)
print(repr(g1))
assert g1.index == 1 and g1.system_index == 7, \
    f"get(1): expected (1, 7), got ({g1.index}, {g1.system_index})"

print("OK")
