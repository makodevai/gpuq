"""
Scenario 16: checkcuda / hascuda / checkamd / hasamd.

Run with: python e2e/16_check_has_provider.py

Expected: on NVIDIA box, hascuda=True; on AMD box, hasamd=True.
At least one provider must be present.
"""

import gpuq

print(f"checkcuda: {repr(gpuq.checkcuda())}")
print(f"hascuda:   {gpuq.hascuda()}")
print(f"checkamd:  {repr(gpuq.checkamd())}")
print(f"hasamd:    {gpuq.hasamd()}")

assert gpuq.hascuda() or gpuq.hasamd(), "At least one provider should be available"

print("OK")
