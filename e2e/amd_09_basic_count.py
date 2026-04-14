"""
AMD Scenario 9: basic count — no env restriction.

Run with: python e2e/amd_09_basic_count.py
(no HIP_VISIBLE_DEVICES set)
"""

import gpuq

total = gpuq.count(visible_only=False)
visible = gpuq.count(visible_only=True)
print(f"total={total}, visible={visible}")
assert total == 8, f"Expected 8 total, got {total}"
assert visible == 8, f"Expected 8 visible (no restriction), got {visible}"

print("OK")
