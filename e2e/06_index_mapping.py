"""
Scenario 6: visible-device index mapping.

CUDA_VISIBLE_DEVICES selects a subset of GPUs.  gpuq must map global
(system-wide) indices to local (visible) indices correctly.

E.g. CUDA_VISIBLE_DEVICES=4,5,6 on an 8-GPU system:
  system_index 4 -> local index 0
  system_index 5 -> local index 1
  system_index 6 -> local index 2
  all other GPUs -> index is None, is_visible is False

Run with: CUDA_VISIBLE_DEVICES=4,5,6 python e2e/06_index_mapping.py
"""

import subprocess, sys, os

_env = {k: v for k, v in os.environ.items()
        if k not in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}
TOTAL = int(subprocess.check_output(
    [sys.executable, '-c', 'import gpuq; print(gpuq.count(visible_only=False))'],
    env=_env).decode().strip())

VISIBLE_IDS = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
print(f"System has {TOTAL} GPU(s), CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

import gpuq

# -- visible_only=True: only the selected GPUs, with local indices 0..N-1 --
visible = gpuq.query(visible_only=True)
assert len(visible) == len(VISIBLE_IDS), \
    f"visible query: expected {len(VISIBLE_IDS)}, got {len(visible)}"

for local_idx, gpu in enumerate(visible):
    assert gpu.index == local_idx, \
        f"visible[{local_idx}].index: expected {local_idx}, got {gpu.index}"
    assert gpu.system_index == VISIBLE_IDS[local_idx], \
        f"visible[{local_idx}].system_index: expected {VISIBLE_IDS[local_idx]}, got {gpu.system_index}"
    assert gpu.is_visible, \
        f"visible[{local_idx}].is_visible: expected True"

# -- visible_only=False: all GPUs, visible ones have local index, others None --
all_gpus = gpuq.query(visible_only=False)
assert len(all_gpus) == TOTAL, \
    f"all query: expected {TOTAL}, got {len(all_gpus)}"

visible_count = 0
for gpu in all_gpus:
    if gpu.system_index in VISIBLE_IDS:
        expected_local = VISIBLE_IDS.index(gpu.system_index)
        assert gpu.index == expected_local, \
            f"GPU system_index={gpu.system_index}: expected local index {expected_local}, got {gpu.index}"
        assert gpu.is_visible, \
            f"GPU system_index={gpu.system_index}: expected is_visible=True"
        visible_count += 1
    else:
        assert gpu.index is None, \
            f"GPU system_index={gpu.system_index}: expected index None, got {gpu.index}"
        assert not gpu.is_visible, \
            f"GPU system_index={gpu.system_index}: expected is_visible=False"

assert visible_count == len(VISIBLE_IDS), \
    f"visible count in all_gpus: expected {len(VISIBLE_IDS)}, got {visible_count}"

print("OK")
