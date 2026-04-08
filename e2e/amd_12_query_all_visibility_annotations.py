"""
AMD Scenario 12: query visible_only=False — all GPUs with visibility annotations.

Run with: HIP_VISIBLE_DEVICES=1,4 python e2e/amd_12_query_all_visibility_annotations.py
"""

import gpuq

gpus = gpuq.query(visible_only=False)
print([(g.system_index, g.index, g.is_visible) for g in gpus])

assert len(gpus) == 8, f"Expected 8 total, got {len(gpus)}"

vis = [g for g in gpus if g.is_visible]
assert len(vis) == 2, f"Expected 2 visible, got {len(vis)}"
assert vis[0].system_index == 1 and vis[0].index == 0
assert vis[1].system_index == 4 and vis[1].index == 1

invis = [g for g in gpus if not g.is_visible]
assert all(g.index is None for g in invis), \
    f"Non-visible GPUs should have index=None"

print("OK")
