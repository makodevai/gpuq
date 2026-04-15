"""
Scenario 11: query visible_only=True — verify index mapping.

Run with: CUDA_VISIBLE_DEVICES=3,6 python e2e/11_query_visible_index_mapping.py
"""

import gpuq

gpus = gpuq.query(visible_only=True)
print([(g.index, g.system_index, g.is_visible) for g in gpus])

assert len(gpus) == 2, f"Expected 2 visible, got {len(gpus)}"
assert gpus[0].index == 0 and gpus[0].system_index == 3, (
    f"gpu[0]: expected (0, 3), got ({gpus[0].index}, {gpus[0].system_index})"
)
assert gpus[1].index == 1 and gpus[1].system_index == 6, (
    f"gpu[1]: expected (1, 6), got ({gpus[1].index}, {gpus[1].system_index})"
)
assert all(g.is_visible for g in gpus)

print("OK")
