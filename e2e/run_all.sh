#!/usr/bin/env bash
#
# Run all CUDA e2e tests for gpuq on an 8×H100 machine.
#
# Each test is invoked with exactly the CUDA_VISIBLE_DEVICES value
# it was designed for (as documented in its docstring).
#
# Tests 06 and 08 are parametric (they parse CVD themselves),
# so we exercise them with additional CVD permutations.
#
# Usage: bash e2e/run_all.sh [--stop-on-fail]

set -euo pipefail

STOP_ON_FAIL=0
[[ "${1:-}" == "--stop-on-fail" ]] && STOP_ON_FAIL=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

PASSED=0
FAILED=0
SKIPPED=0
FAILURES=()

# ── helpers ──────────────────────────────────────────────────────────────────

run_test() {
    local desc="$1"
    local script="$2"
    shift 2
    # remaining args are env overrides (VAR=val), or "__UNSET_CVD__" to unset it

    local env_args=()
    local unset_cvd=0
    for arg in "$@"; do
        if [[ "$arg" == "__UNSET_CVD__" ]]; then
            unset_cvd=1
        else
            env_args+=("$arg")
        fi
    done

    printf "  %-70s " "$desc"

    # build the env: start clean (no CVD/HVD/RVD), then apply overrides
    local cmd_prefix=()
    cmd_prefix+=(env)
    if [[ $unset_cvd -eq 1 ]]; then
        cmd_prefix+=(-u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES)
    fi
    for e in "${env_args[@]+"${env_args[@]}"}"; do
        cmd_prefix+=("$e")
    done

    local output
    if output=$("${cmd_prefix[@]}" "$PYTHON" "$SCRIPT_DIR/$script" 2>&1); then
        printf "\033[32mPASS\033[0m\n"
        PASSED=$((PASSED + 1))
    else
        local rc=$?
        printf "\033[31mFAIL\033[0m (exit %d)\n" "$rc"
        # indent output for readability
        echo "$output" | sed 's/^/    | /'
        FAILED=$((FAILED + 1))
        FAILURES+=("$desc")
        if [[ $STOP_ON_FAIL -eq 1 ]]; then
            echo ""
            echo "STOPPING (--stop-on-fail)"
            exit 1
        fi
    fi
}

section() {
    echo ""
    echo "━━━ $1 ━━━"
}

# ── tests: no CUDA_VISIBLE_DEVICES (all GPUs visible) ───────────────────────

section "No CUDA_VISIBLE_DEVICES (all 8 GPUs visible)"

run_test "09  basic count (no restriction)" \
    09_basic_count.py __UNSET_CVD__

run_test "14  properties & fields" \
    14_properties_fields.py __UNSET_CVD__

run_test "15  provider filtering" \
    15_provider_filtering.py __UNSET_CVD__

run_test "16  check/has provider" \
    16_check_has_provider.py __UNSET_CVD__

run_test "17  required raises for missing provider" \
    17_required_raises.py __UNSET_CVD__

run_test "19  runtime info" \
    19_runtime_info.py __UNSET_CVD__

# ── tests: empty CUDA_VISIBLE_DEVICES (0 GPUs visible) ──────────────────────

section "Empty CUDA_VISIBLE_DEVICES (0 visible)"

run_test "18  empty CVD" \
    18_empty_visible_devices.py CUDA_VISIBLE_DEVICES=

# ── tests: single GPU visible ────────────────────────────────────────────────
# test 07 is designed for CVD=0 only (hardcodes system_index==0)

section "Single GPU visible"

run_test "07  single GPU CVD=0" \
    07_index_mapping_single.py CUDA_VISIBLE_DEVICES=0

# ── tests: torch interaction ─────────────────────────────────────────────────
# tests 01-05 are designed for a single-GPU CVD

section "Torch + gpuq interaction (CVD=0)"

run_test "01  gpuq all → torch → gpuq visible  CVD=0" \
    01_gpuq_all_then_torch_then_gpuq_visible.py CUDA_VISIBLE_DEVICES=0

run_test "02  torch → gpuq                     CVD=0" \
    02_torch_then_gpuq.py CUDA_VISIBLE_DEVICES=0

run_test "03  interleaved                       CVD=0" \
    03_interleaved.py CUDA_VISIBLE_DEVICES=0

run_test "04  torch sandwich                    CVD=0" \
    04_torch_sandwich.py CUDA_VISIBLE_DEVICES=0

run_test "05  query → torch init → counts       CVD=0" \
    05_query_then_torch_init_then_counts.py CUDA_VISIBLE_DEVICES=0

section "Torch + gpuq interaction (CVD=5)"

run_test "01  gpuq all → torch → gpuq visible  CVD=5" \
    01_gpuq_all_then_torch_then_gpuq_visible.py CUDA_VISIBLE_DEVICES=5

run_test "02  torch → gpuq                     CVD=5" \
    02_torch_then_gpuq.py CUDA_VISIBLE_DEVICES=5

run_test "03  interleaved                       CVD=5" \
    03_interleaved.py CUDA_VISIBLE_DEVICES=5

run_test "04  torch sandwich                    CVD=5" \
    04_torch_sandwich.py CUDA_VISIBLE_DEVICES=5

run_test "05  query → torch init → counts       CVD=5" \
    05_query_then_torch_init_then_counts.py CUDA_VISIBLE_DEVICES=5

# ── tests: two GPUs ──────────────────────────────────────────────────────────

section "Two GPUs visible"

run_test "10  count CVD=2,5" \
    10_count_with_restriction.py CUDA_VISIBLE_DEVICES=2,5

run_test "11  query visible index mapping CVD=3,6" \
    11_query_visible_index_mapping.py CUDA_VISIBLE_DEVICES=3,6

run_test "12  all visibility annotations CVD=1,4" \
    12_query_all_visibility_annotations.py CUDA_VISIBLE_DEVICES=1,4

run_test "13  get single gpu CVD=2,7" \
    13_get_single_gpu.py CUDA_VISIBLE_DEVICES=2,7

# ── tests: three GPUs (including non-monotonic) ─────────────────────────────
# tests 06 and 08 are parametric — they parse CVD themselves

section "Three GPUs visible (various orderings)"

run_test "06  index mapping CVD=4,5,6" \
    06_index_mapping.py CUDA_VISIBLE_DEVICES=4,5,6

run_test "08  reversed order CVD=6,2,4" \
    08_index_mapping_reversed.py CUDA_VISIBLE_DEVICES=6,2,4

run_test "20  non-monotonic CVD=5,1,3" \
    20_nonmonotonic_ordering.py CUDA_VISIBLE_DEVICES=5,1,3

run_test "08  reversed order CVD=7,3,0" \
    08_index_mapping_reversed.py CUDA_VISIBLE_DEVICES=7,3,0

# ── tests: four GPUs (half the machine) ──────────────────────────────────────

section "Four GPUs visible (half machine)"

run_test "06  index mapping CVD=0,1,2,3 (first half)" \
    06_index_mapping.py CUDA_VISIBLE_DEVICES=0,1,2,3

run_test "06  index mapping CVD=4,5,6,7 (second half)" \
    06_index_mapping.py CUDA_VISIBLE_DEVICES=4,5,6,7

run_test "06  index mapping CVD=0,2,4,6 (even)" \
    06_index_mapping.py CUDA_VISIBLE_DEVICES=0,2,4,6

run_test "06  index mapping CVD=1,3,5,7 (odd)" \
    06_index_mapping.py CUDA_VISIBLE_DEVICES=1,3,5,7

run_test "08  reversed order CVD=7,5,3,1" \
    08_index_mapping_reversed.py CUDA_VISIBLE_DEVICES=7,5,3,1

# ── tests: all 8 GPUs explicitly ─────────────────────────────────────────────

section "All 8 GPUs explicitly listed"

run_test "06  index mapping CVD=0,1,2,3,4,5,6,7" \
    06_index_mapping.py CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

run_test "08  reversed order CVD=7,6,5,4,3,2,1,0" \
    08_index_mapping_reversed.py CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0

# ── tests: edge cases ───────────────────────────────────────────────────────

section "Edge cases"

# sparse: only edges
run_test "06  index mapping CVD=0,7 (edges only)" \
    06_index_mapping.py CUDA_VISIBLE_DEVICES=0,7

# seven GPUs (all but one)
for skip in $(seq 0 7); do
    cvd=""
    for g in $(seq 0 7); do
        [[ $g -eq $skip ]] && continue
        [[ -n "$cvd" ]] && cvd+=","
        cvd+="$g"
    done
    run_test "06  index mapping CVD=$cvd (all but $skip)" \
        06_index_mapping.py CUDA_VISIBLE_DEVICES=$cvd
done

# ── summary ──────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
printf "  \033[32mPASSED: %d\033[0m   \033[31mFAILED: %d\033[0m   SKIPPED: %d\n" "$PASSED" "$FAILED" "$SKIPPED"
echo "═══════════════════════════════════════════════════════════"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo ""
    echo "Failed tests:"
    for f in "${FAILURES[@]}"; do
        echo "  ✗ $f"
    done
    echo ""
    exit 1
fi

echo ""
echo "All tests passed."
exit 0
