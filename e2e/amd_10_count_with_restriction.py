"""
AMD Scenario 10: count with HIP_VISIBLE_DEVICES restriction.

Run with: HIP_VISIBLE_DEVICES=2,5 python e2e/amd_10_count_with_restriction.py
"""

import gpuq

total = gpuq.count(visible_only=False)
visible = gpuq.count(visible_only=True)
print(f"total={total}, visible={visible}")
assert total == 8, f"Expected 8 total, got {total}"
assert visible == 2, f"Expected 2 visible, got {visible}"

print("OK")
