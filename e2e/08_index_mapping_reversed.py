"""
Scenario 8: non-monotonic CUDA_VISIBLE_DEVICES — gpuq sorts visible indices.

CUDA_VISIBLE_DEVICES=6,2,4 on an 8-GPU system:
  gpuq parses and sorts the list to [2, 4, 6], so:
  system_index 2 -> local index 0
  system_index 4 -> local index 1
  system_index 6 -> local index 2

Run with: CUDA_VISIBLE_DEVICES=6,2,4 python e2e/08_index_mapping_reversed.py
"""

import subprocess, sys, os

_env = {k: v for k, v in os.environ.items()
        if k not in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}
TOTAL = int(subprocess.check_output(
    [sys.executable, '-c', 'import gpuq; print(gpuq.count(visible_only=False))'],
    env=_env).decode().strip())

RAW_IDS = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
VISIBLE_IDS = sorted(RAW_IDS)  # gpuq sorts visible indices
print(f"System has {TOTAL} GPU(s), CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} -> sorted {VISIBLE_IDS}")

import gpuq

visible = gpuq.query(visible_only=True)
assert len(visible) == len(VISIBLE_IDS), \
    f"visible query: expected {len(VISIBLE_IDS)}, got {len(visible)}"

for local_idx, gpu in enumerate(visible):
    assert gpu.index == local_idx, \
        f"visible[{local_idx}].index: expected {local_idx}, got {gpu.index}"
    assert gpu.system_index == VISIBLE_IDS[local_idx], \
        f"visible[{local_idx}].system_index: expected {VISIBLE_IDS[local_idx]}, got {gpu.system_index}"
    assert gpu.is_visible

all_gpus = gpuq.query(visible_only=False)
assert len(all_gpus) == TOTAL, f"all query: expected {TOTAL}, got {len(all_gpus)}"

for gpu in all_gpus:
    if gpu.system_index in VISIBLE_IDS:
        expected_local = VISIBLE_IDS.index(gpu.system_index)
        assert gpu.index == expected_local, \
            f"GPU system_index={gpu.system_index}: expected local {expected_local}, got {gpu.index}"
        assert gpu.is_visible
    else:
        assert gpu.index is None
        assert not gpu.is_visible

print("OK")
