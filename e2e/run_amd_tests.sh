#!/usr/bin/env bash
set -uo pipefail

# ============================================================================
# AMD e2e test runner for gpuq (MI355X, 8 GPUs)
#
# Runs all existing AMD e2e tests plus additional CUDA_VISIBLE_DEVICES-on-AMD
# scenarios that probe the env-var fallback path and torch poisoning bug.
# ============================================================================

PYTHON="${PYTHON:-python3}"
PASS=0
FAIL=0
SKIP=0
ERRORS=()

run() {
    local label="$1"
    shift
    local env_vars=()
    local script=""

    # Separate env vars from the script path
    while [[ $# -gt 1 ]]; do
        env_vars+=("$1")
        shift
    done
    script="$1"

    printf "%-70s " "$label"

    # Build the env command
    local cmd="env"
    # Always clean slate: unset all GPU visibility vars first
    cmd+=" -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES"
    for v in "${env_vars[@]}"; do
        cmd+=" $v"
    done
    cmd+=" $PYTHON $script"

    output=$(eval "$cmd" 2>&1)
    rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "PASS"
        ((PASS++))
    else
        echo "FAIL"
        ((FAIL++))
        ERRORS+=("$label")
        # Print indented output for failures
        echo "$output" | sed 's/^/    /'
        echo ""
    fi
}

skip() {
    local label="$1"
    local reason="$2"
    printf "%-70s SKIP (%s)\n" "$label" "$reason"
    ((SKIP++))
}

echo "========================================================================"
echo "AMD e2e test suite for gpuq"
echo "========================================================================"
echo ""

# Quick sanity: make sure gpuq loads and sees GPUs
TOTAL=$($PYTHON -c "
import os
os.environ.pop('HIP_VISIBLE_DEVICES', None)
os.environ.pop('CUDA_VISIBLE_DEVICES', None)
os.environ.pop('ROCR_VISIBLE_DEVICES', None)
import gpuq; print(gpuq.count(visible_only=False))
" 2>&1)

if [[ "$TOTAL" -lt 1 ]] 2>/dev/null; then
    echo "FATAL: gpuq sees $TOTAL GPUs — cannot run tests."
    exit 1
fi
echo "Detected $TOTAL AMD GPUs"
echo ""

# --------------------------------------------------------------------------
# Section 1: Existing AMD e2e tests
# --------------------------------------------------------------------------
echo "--- Existing AMD e2e tests ---"
echo ""

run "amd_01 gpuq_all then torch (HIP_VIS=0)" \
    "HIP_VISIBLE_DEVICES=0" e2e/amd_01_gpuq_all_then_torch_then_gpuq_visible.py

run "amd_02 torch then gpuq (HIP_VIS=0)" \
    "HIP_VISIBLE_DEVICES=0" e2e/amd_02_torch_then_gpuq.py

run "amd_03 interleaved (HIP_VIS=0)" \
    "HIP_VISIBLE_DEVICES=0" e2e/amd_03_interleaved.py

run "amd_04 torch sandwich (HIP_VIS=0)" \
    "HIP_VISIBLE_DEVICES=0" e2e/amd_04_torch_sandwich.py

run "amd_05 query then torch then counts (HIP_VIS=0)" \
    "HIP_VISIBLE_DEVICES=0" e2e/amd_05_query_then_torch_init_then_counts.py

run "amd_06 index mapping (HIP_VIS=4,5,6)" \
    "HIP_VISIBLE_DEVICES=4,5,6" e2e/amd_06_index_mapping.py

run "amd_07 index mapping single (HIP_VIS=0)" \
    "HIP_VISIBLE_DEVICES=0" e2e/amd_07_index_mapping_single.py

run "amd_08 index mapping reversed (HIP_VIS=6,2,4)" \
    "HIP_VISIBLE_DEVICES=6,2,4" e2e/amd_08_index_mapping_reversed.py

run "amd_09 basic count (no env)" \
    "" e2e/amd_09_basic_count.py

run "amd_10 count with restriction (HIP_VIS=2,5)" \
    "HIP_VISIBLE_DEVICES=2,5" e2e/amd_10_count_with_restriction.py

run "amd_11 query visible index mapping (HIP_VIS=3,6)" \
    "HIP_VISIBLE_DEVICES=3,6" e2e/amd_11_query_visible_index_mapping.py

run "amd_12 query all visibility annotations (HIP_VIS=1,4)" \
    "HIP_VISIBLE_DEVICES=1,4" e2e/amd_12_query_all_visibility_annotations.py

run "amd_13 get single gpu (HIP_VIS=2,7)" \
    "HIP_VISIBLE_DEVICES=2,7" e2e/amd_13_get_single_gpu.py

run "amd_14 properties fields (no env)" \
    "" e2e/amd_14_properties_fields.py

run "amd_18 empty visible devices (HIP_VIS=)" \
    "HIP_VISIBLE_DEVICES=" e2e/amd_18_empty_visible_devices.py

run "amd_19 runtime info (no env)" \
    "" e2e/amd_19_runtime_info.py

run "amd_20 nonmonotonic ordering (HIP_VIS=5,1,3)" \
    "HIP_VISIBLE_DEVICES=5,1,3" e2e/amd_20_nonmonotonic_ordering.py

# --------------------------------------------------------------------------
# Section 2: CUDA_VISIBLE_DEVICES fallback on AMD
#
# When HIP_VISIBLE_DEVICES is unset, gpuq falls back to CUDA_VISIBLE_DEVICES
# for HIP visibility (impl.py: parsed_hip = parsed_cuda). These tests check
# that this fallback works and doesn't poison torch.
# --------------------------------------------------------------------------
echo ""
echo "--- CUDA_VISIBLE_DEVICES fallback on AMD (torch poisoning tests) ---"
echo ""

# Generate inline test scripts via heredocs to avoid cluttering the e2e dir.
# Each test runs in a subprocess with a clean env.

TORCH_POISON_SCRIPT=$(cat <<'PYEOF'
import subprocess, sys, os

# Detect total from clean subprocess
_env = {k: v for k, v in os.environ.items()
        if k not in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}
TOTAL = int(subprocess.check_output(
    [sys.executable, '-c', 'import gpuq; print(gpuq.count(visible_only=False))'],
    env=_env).decode().strip())

cuda_vis = os.environ.get('CUDA_VISIBLE_DEVICES', None)
hip_vis = os.environ.get('HIP_VISIBLE_DEVICES', None)
expected_visible = None

if hip_vis is not None:
    if hip_vis == '':
        expected_visible = 0
    else:
        expected_visible = len([x for x in hip_vis.split(',') if x.strip()])
elif cuda_vis is not None:
    if cuda_vis == '':
        expected_visible = 0
    else:
        expected_visible = len([x for x in cuda_vis.split(',') if x.strip()])
else:
    expected_visible = TOTAL

print(f"TOTAL={TOTAL}, CUDA_VISIBLE_DEVICES={cuda_vis!r}, HIP_VISIBLE_DEVICES={hip_vis!r}, expected_visible={expected_visible}")

import gpuq

# gpuq first — this is where poisoning can happen
gpus = gpuq.query(visible_only=False)
assert len(gpus) == TOTAL, f"gpuq.query(vis=F): expected {TOTAL}, got {len(gpus)}"

import torch
if expected_visible > 0:
    torch.cuda.init()
    tc = torch.cuda.device_count()
    assert tc == expected_visible, \
        f"torch.cuda.device_count(): expected {expected_visible}, got {tc}"

assert gpuq.count(visible_only=True) == expected_visible, \
    f"gpuq.count(vis=T): expected {expected_visible}, got {gpuq.count(visible_only=True)}"
assert gpuq.count(visible_only=False) == TOTAL, \
    f"gpuq.count(vis=F): expected {TOTAL}, got {gpuq.count(visible_only=False)}"

print("OK")
PYEOF
)

# Write it to a temp file so run() can use it
TMPSCRIPT=$(mktemp /tmp/gpuq_torch_test_XXXXXX.py)
echo "$TORCH_POISON_SCRIPT" > "$TMPSCRIPT"

# -- gpuq before torch, CUDA_VISIBLE_DEVICES only (HIP unset) --
run "gpuq->torch, CUDA_VIS=0 (single GPU fallback)" \
    "CUDA_VISIBLE_DEVICES=0" "$TMPSCRIPT"

run "gpuq->torch, CUDA_VIS=3 (mid-range GPU fallback)" \
    "CUDA_VISIBLE_DEVICES=3" "$TMPSCRIPT"

run "gpuq->torch, CUDA_VIS=0,1 (two GPUs fallback)" \
    "CUDA_VISIBLE_DEVICES=0,1" "$TMPSCRIPT"

run "gpuq->torch, CUDA_VIS=2,5,7 (three GPUs fallback)" \
    "CUDA_VISIBLE_DEVICES=2,5,7" "$TMPSCRIPT"

run "gpuq->torch, CUDA_VIS=7,3,1 (non-monotonic fallback)" \
    "CUDA_VISIBLE_DEVICES=7,3,1" "$TMPSCRIPT"

run "gpuq->torch, CUDA_VIS= (empty = 0 visible)" \
    "CUDA_VISIBLE_DEVICES=" "$TMPSCRIPT"

# -- gpuq before torch, HIP_VISIBLE_DEVICES set (no fallback needed) --
run "gpuq->torch, HIP_VIS=0 (single GPU, explicit)" \
    "HIP_VISIBLE_DEVICES=0" "$TMPSCRIPT"

run "gpuq->torch, HIP_VIS=2,5 (two GPUs, explicit)" \
    "HIP_VISIBLE_DEVICES=2,5" "$TMPSCRIPT"

run "gpuq->torch, HIP_VIS=7,3,1 (non-monotonic, explicit)" \
    "HIP_VISIBLE_DEVICES=7,3,1" "$TMPSCRIPT"

# -- Both set: HIP should take precedence over CUDA --
run "gpuq->torch, CUDA_VIS=0,1 + HIP_VIS=3 (HIP wins)" \
    "CUDA_VISIBLE_DEVICES=0,1" "HIP_VISIBLE_DEVICES=3" "$TMPSCRIPT"

run "gpuq->torch, CUDA_VIS=0 + HIP_VIS=2,4,6 (HIP wins)" \
    "CUDA_VISIBLE_DEVICES=0" "HIP_VISIBLE_DEVICES=2,4,6" "$TMPSCRIPT"

# --------------------------------------------------------------------------
# Section 3: torch before gpuq with CUDA_VISIBLE_DEVICES
# --------------------------------------------------------------------------
echo ""
echo "--- torch before gpuq with CUDA_VISIBLE_DEVICES ---"
echo ""

TORCH_FIRST_SCRIPT=$(cat <<'PYEOF'
import subprocess, sys, os

_env = {k: v for k, v in os.environ.items()
        if k not in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}
TOTAL = int(subprocess.check_output(
    [sys.executable, '-c', 'import gpuq; print(gpuq.count(visible_only=False))'],
    env=_env).decode().strip())

cuda_vis = os.environ.get('CUDA_VISIBLE_DEVICES', None)
hip_vis = os.environ.get('HIP_VISIBLE_DEVICES', None)

if hip_vis is not None:
    if hip_vis == '':
        expected_visible = 0
    else:
        expected_visible = len([x for x in hip_vis.split(',') if x.strip()])
elif cuda_vis is not None:
    if cuda_vis == '':
        expected_visible = 0
    else:
        expected_visible = len([x for x in cuda_vis.split(',') if x.strip()])
else:
    expected_visible = TOTAL

print(f"TOTAL={TOTAL}, CUDA_VIS={cuda_vis!r}, HIP_VIS={hip_vis!r}, expected_visible={expected_visible}")

# torch first — HIP runtime inits with restriction
import torch
torch.cuda.init()
tc = torch.cuda.device_count()
assert tc == expected_visible, \
    f"torch.cuda.device_count(): expected {expected_visible}, got {tc}"

# gpuq after torch
import gpuq
assert gpuq.count(visible_only=True) == expected_visible, \
    f"gpuq.count(vis=T): expected {expected_visible}, got {gpuq.count(visible_only=True)}"
assert gpuq.count(visible_only=False) == TOTAL, \
    f"gpuq.count(vis=F): expected {TOTAL}, got {gpuq.count(visible_only=False)}"

# torch should still see the same count after gpuq ran
tc2 = torch.cuda.device_count()
assert tc2 == expected_visible, \
    f"torch after gpuq: expected {expected_visible}, got {tc2}"

print("OK")
PYEOF
)

TMPSCRIPT2=$(mktemp /tmp/gpuq_torch_first_XXXXXX.py)
echo "$TORCH_FIRST_SCRIPT" > "$TMPSCRIPT2"

run "torch->gpuq, CUDA_VIS=0 (single GPU)" \
    "CUDA_VISIBLE_DEVICES=0" "$TMPSCRIPT2"

run "torch->gpuq, CUDA_VIS=3 (mid-range GPU)" \
    "CUDA_VISIBLE_DEVICES=3" "$TMPSCRIPT2"

run "torch->gpuq, CUDA_VIS=0,1,2 (three GPUs)" \
    "CUDA_VISIBLE_DEVICES=0,1,2" "$TMPSCRIPT2"

run "torch->gpuq, HIP_VIS=0 (single, explicit)" \
    "HIP_VISIBLE_DEVICES=0" "$TMPSCRIPT2"

run "torch->gpuq, CUDA_VIS=0 + HIP_VIS=2,4 (HIP wins)" \
    "CUDA_VISIBLE_DEVICES=0" "HIP_VISIBLE_DEVICES=2,4" "$TMPSCRIPT2"

# --------------------------------------------------------------------------
# Section 4: interleaved calls with CUDA_VISIBLE_DEVICES
# --------------------------------------------------------------------------
echo ""
echo "--- Interleaved gpuq/torch with CUDA_VISIBLE_DEVICES ---"
echo ""

INTERLEAVED_SCRIPT=$(cat <<'PYEOF'
import subprocess, sys, os

_env = {k: v for k, v in os.environ.items()
        if k not in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}
TOTAL = int(subprocess.check_output(
    [sys.executable, '-c', 'import gpuq; print(gpuq.count(visible_only=False))'],
    env=_env).decode().strip())

cuda_vis = os.environ.get('CUDA_VISIBLE_DEVICES', None)
hip_vis = os.environ.get('HIP_VISIBLE_DEVICES', None)

if hip_vis is not None:
    if hip_vis == '':
        expected_visible = 0
    else:
        expected_visible = len([x for x in hip_vis.split(',') if x.strip()])
elif cuda_vis is not None:
    if cuda_vis == '':
        expected_visible = 0
    else:
        expected_visible = len([x for x in cuda_vis.split(',') if x.strip()])
else:
    expected_visible = TOTAL

print(f"TOTAL={TOTAL}, CUDA_VIS={cuda_vis!r}, HIP_VIS={hip_vis!r}, expected_visible={expected_visible}")

import gpuq
import torch

# Round 1: gpuq visible count
vc1 = gpuq.count(visible_only=True)
assert vc1 == expected_visible, f"R1 gpuq vis: expected {expected_visible}, got {vc1}"

# Round 2: torch init
torch.cuda.init()
tc1 = torch.cuda.device_count()
assert tc1 == expected_visible, f"R2 torch: expected {expected_visible}, got {tc1}"

# Round 3: gpuq total (triggers save_visible — clears and restores env)
ac = gpuq.count(visible_only=False)
assert ac == TOTAL, f"R3 gpuq total: expected {TOTAL}, got {ac}"

# Round 4: torch still correct after gpuq cleared/restored env?
tc2 = torch.cuda.device_count()
assert tc2 == expected_visible, f"R4 torch after gpuq total: expected {expected_visible}, got {tc2}"

# Round 5: gpuq visible still correct?
vc2 = gpuq.count(visible_only=True)
assert vc2 == expected_visible, f"R5 gpuq vis: expected {expected_visible}, got {vc2}"

print("OK")
PYEOF
)

TMPSCRIPT3=$(mktemp /tmp/gpuq_interleaved_XXXXXX.py)
echo "$INTERLEAVED_SCRIPT" > "$TMPSCRIPT3"

run "interleaved, CUDA_VIS=0 (single GPU)" \
    "CUDA_VISIBLE_DEVICES=0" "$TMPSCRIPT3"

run "interleaved, CUDA_VIS=3,5 (two GPUs)" \
    "CUDA_VISIBLE_DEVICES=3,5" "$TMPSCRIPT3"

run "interleaved, HIP_VIS=0 (single, explicit)" \
    "HIP_VISIBLE_DEVICES=0" "$TMPSCRIPT3"

run "interleaved, CUDA_VIS=0 + HIP_VIS=1,2 (HIP wins)" \
    "CUDA_VISIBLE_DEVICES=0" "HIP_VISIBLE_DEVICES=1,2" "$TMPSCRIPT3"

# --------------------------------------------------------------------------
# Cleanup and summary
# --------------------------------------------------------------------------
rm -f "$TMPSCRIPT" "$TMPSCRIPT2" "$TMPSCRIPT3"

echo ""
echo "========================================================================"
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "========================================================================"

if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo ""
    echo "FAILED:"
    for e in "${ERRORS[@]}"; do
        echo "  - $e"
    done
    exit 1
else
    echo "All tests passed."
    exit 0
fi
