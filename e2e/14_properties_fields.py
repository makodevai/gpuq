"""
Scenario 14: Properties fields — uuid, name, hardware props, asdict.

Run with: python e2e/14_properties_fields.py
"""

import gpuq

g = gpuq.get(0, visible_only=False)
print(f"name={g.name}, uuid={g.uuid}, major={g.major}, minor={g.minor}")
print(f"memory={g.total_memory}")

assert g.name != "", "name should not be empty"
assert g.uuid is not None, "uuid should not be None"
assert g.total_memory > 0, f"total_memory should be > 0, got {g.total_memory}"

d = g.asdict()
assert "uuid" in d and "name" in d and "total_memory" in d
print(f"asdict keys: {list(d.keys())}")

print("OK")
