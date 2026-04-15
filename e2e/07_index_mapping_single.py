"""
Scenario 7: single visible GPU — index mapping sanity check.

CUDA_VISIBLE_DEVICES=0 on a multi-GPU system: only GPU 0 is visible,
local index 0 maps to system_index 0.

Run with: CUDA_VISIBLE_DEVICES=0 python e2e/07_index_mapping_single.py
"""

import subprocess, sys, os

_env = {
    k: v
    for k, v in os.environ.items()
    if k not in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
}
TOTAL = int(
    subprocess.check_output(
        [sys.executable, "-c", "import gpuq; print(gpuq.count(visible_only=False))"],
        env=_env,
    )
    .decode()
    .strip()
)

print(f"System has {TOTAL} GPU(s), CUDA_VISIBLE_DEVICES=0")

import gpuq

visible = gpuq.query(visible_only=True)
assert len(visible) == 1, f"visible query: expected 1, got {len(visible)}"
assert visible[0].index == 0, f"visible[0].index: expected 0, got {visible[0].index}"
assert visible[0].system_index == 0, (
    f"visible[0].system_index: expected 0, got {visible[0].system_index}"
)
assert visible[0].is_visible

all_gpus = gpuq.query(visible_only=False)
assert len(all_gpus) == TOTAL, f"all query: expected {TOTAL}, got {len(all_gpus)}"

for gpu in all_gpus:
    if gpu.system_index == 0:
        assert gpu.index == 0, f"system_index=0: expected local 0, got {gpu.index}"
        assert gpu.is_visible
    else:
        assert gpu.index is None, (
            f"system_index={gpu.system_index}: expected None, got {gpu.index}"
        )
        assert not gpu.is_visible

print("OK")
