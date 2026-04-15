#!/usr/bin/env python3
"""
Test script for AMD MI355X: find HIP_VISIBLE_DEVICES values that break gpuq.count().

This script runs gpuq.count() under various HIP_VISIBLE_DEVICES settings in
isolated subprocesses. It probes edge cases that can cause incorrect results,
crashes, or hangs.

Run with:  python e2e/amd_test_hip_visible_devices_breaking.py
"""

import os
import sys
import json
import subprocess
import textwrap

PYTHON = sys.executable
TIMEOUT = 30  # seconds per subprocess

# ---------------------------------------------------------------------------
# Helper: run gpuq.count() in a subprocess with a given env overlay
# ---------------------------------------------------------------------------

_PROBE_SCRIPT = textwrap.dedent("""\
    import gpuq, json, sys
    try:
        total   = gpuq.count(visible_only=False)
        visible = gpuq.count(visible_only=True)
        queried = len(gpuq.query(visible_only=False))
        queried_vis = len(gpuq.query(visible_only=True))
    except Exception as e:
        json.dump({"error": f"{type(e).__name__}: {e}"}, sys.stdout)
        sys.exit(0)
    json.dump({
        "total": total,
        "visible": visible,
        "queried": queried,
        "queried_vis": queried_vis,
    }, sys.stdout)
""")


def run_probe(env_overlay: dict[str, str | None], label: str) -> dict:
    """Run the probe script in a subprocess with the given env vars.

    Returns a dict with the probe results or an error description.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    }
    for k, v in env_overlay.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v

    try:
        proc = subprocess.run(
            [PYTHON, "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"label": label, "status": "TIMEOUT"}

    if proc.returncode != 0:
        return {
            "label": label,
            "status": "CRASH",
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip()[-500:],
        }

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "label": label,
            "status": "BAD_OUTPUT",
            "stdout": proc.stdout.strip()[-500:],
            "stderr": proc.stderr.strip()[-500:],
        }

    data["label"] = label
    data["status"] = "OK"
    return data


# ---------------------------------------------------------------------------
# Step 1: discover baseline (no env var restrictions)
# ---------------------------------------------------------------------------

print("=" * 72)
print("AMD MI355X — HIP_VISIBLE_DEVICES breaking-behavior test")
print("=" * 72)

baseline = run_probe({}, "baseline (no env vars)")
if baseline["status"] != "OK":
    print(f"FATAL: cannot even get a baseline: {baseline}")
    sys.exit(1)

TOTAL = baseline["total"]
print(f"\nBaseline: {TOTAL} GPUs detected")
print(f"  count(visible_only=False) = {baseline['total']}")
print(f"  count(visible_only=True)  = {baseline['visible']}")
print(f"  len(query(visible_only=False)) = {baseline['queried']}")
print(f"  len(query(visible_only=True))  = {baseline['queried_vis']}")

if TOTAL == 0:
    print("No AMD GPUs detected — nothing to test.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Step 2: define test cases
# ---------------------------------------------------------------------------

cases: list[tuple[str, dict[str, str | None]]] = []


def case(label: str, **env: str | None):
    cases.append((label, env))


# --- Basic valid values ---
case("single GPU 0", HIP_VISIBLE_DEVICES="0")
case("single GPU last", HIP_VISIBLE_DEVICES=str(TOTAL - 1))
case("all GPUs ascending", HIP_VISIBLE_DEVICES=",".join(str(i) for i in range(TOTAL)))
case(
    "all GPUs descending",
    HIP_VISIBLE_DEVICES=",".join(str(i) for i in reversed(range(TOTAL))),
)

# --- Empty / whitespace ---
case("empty string", HIP_VISIBLE_DEVICES="")
case("whitespace only", HIP_VISIBLE_DEVICES="   ")
case("comma only", HIP_VISIBLE_DEVICES=",")
case("multiple commas", HIP_VISIBLE_DEVICES=",,,")

# --- Out-of-range indices ---
case("one past last", HIP_VISIBLE_DEVICES=str(TOTAL))
case("way out of range", HIP_VISIBLE_DEVICES="99")
case("very large index", HIP_VISIBLE_DEVICES="999999")
case("valid + out-of-range", HIP_VISIBLE_DEVICES=f"0,{TOTAL + 10}")
case("only out-of-range", HIP_VISIBLE_DEVICES=f"{TOTAL},{TOTAL + 1}")

# --- Negative indices ---
case("negative -1", HIP_VISIBLE_DEVICES="-1")
case("negative mixed", HIP_VISIBLE_DEVICES="-1,0,1")

# --- Duplicates ---
case("duplicate same GPU", HIP_VISIBLE_DEVICES="0,0,0")
case("duplicates mixed", HIP_VISIBLE_DEVICES="0,1,0,1")

# --- Leading/trailing whitespace and padding ---
case("leading space", HIP_VISIBLE_DEVICES=" 0")
case("trailing space", HIP_VISIBLE_DEVICES="0 ")
case("spaces around comma", HIP_VISIBLE_DEVICES="0 , 1")
case("trailing comma", HIP_VISIBLE_DEVICES="0,1,")
case("leading comma", HIP_VISIBLE_DEVICES=",0,1")

# --- ROCR_VISIBLE_DEVICES (not handled by save_visible!) ---
case("ROCR restricts to 1 GPU", ROCR_VISIBLE_DEVICES="0")
case("ROCR restricts to 2 GPUs", ROCR_VISIBLE_DEVICES="0,1")
case("ROCR + HIP both set", ROCR_VISIBLE_DEVICES="0", HIP_VISIBLE_DEVICES="0,1,2")
case("ROCR empty", ROCR_VISIBLE_DEVICES="")
case("ROCR out-of-range", ROCR_VISIBLE_DEVICES="99")

# --- CUDA_VISIBLE_DEVICES fallback (hip defaults to cuda if HIP not set) ---
case("CUDA set, HIP unset", CUDA_VISIBLE_DEVICES="0,1")
case(
    "CUDA set, HIP explicitly unset",
    CUDA_VISIBLE_DEVICES="0,1",
    HIP_VISIBLE_DEVICES=None,
)
case("CUDA empty, HIP unset", CUDA_VISIBLE_DEVICES="")
case("CUDA out-of-range, HIP unset", CUDA_VISIBLE_DEVICES="99")
case(
    "CUDA + HIP both set, different",
    CUDA_VISIBLE_DEVICES="0,1",
    HIP_VISIBLE_DEVICES="2,3",
)
case(
    "CUDA + HIP + ROCR all set",
    CUDA_VISIBLE_DEVICES="0",
    HIP_VISIBLE_DEVICES="1",
    ROCR_VISIBLE_DEVICES="2",
)

# --- ROCR_VISIBLE_DEVICES vs count(visible_only=False) ---
# This is the most likely bug: save_visible() clears HIP_VISIBLE_DEVICES but
# NOT ROCR_VISIBLE_DEVICES. If ROCR restricts the runtime, c_count() will
# return fewer GPUs than expected even with HIP_VISIBLE_DEVICES cleared.
case("ROCR=0, count should still be total", ROCR_VISIBLE_DEVICES="0")
case("ROCR=0,1, count should still be total", ROCR_VISIBLE_DEVICES="0,1")
case(
    "ROCR=0 + HIP=0, count should still be total",
    ROCR_VISIBLE_DEVICES="0",
    HIP_VISIBLE_DEVICES="0",
)


# ---------------------------------------------------------------------------
# Step 3: run all cases
# ---------------------------------------------------------------------------

print(f"\nRunning {len(cases)} test cases...\n")

results = []
failures = []

for label, env_overlay in cases:
    result = run_probe(env_overlay, label)
    results.append(result)

    env_desc = " ".join(f"{k}={v!r}" for k, v in env_overlay.items() if v is not None)
    unset = [k for k, v in env_overlay.items() if v is None]
    if unset:
        env_desc += " (unset: " + ",".join(unset) + ")"

    is_broken = False
    detail = ""

    if result["status"] == "TIMEOUT":
        is_broken = True
        detail = "TIMEOUT — process hung"
    elif result["status"] == "CRASH":
        is_broken = True
        detail = f"CRASH (rc={result['returncode']}): {result.get('stderr', '')[:200]}"
    elif result["status"] == "BAD_OUTPUT":
        is_broken = True
        detail = f"BAD_OUTPUT: stdout={result.get('stdout', '')[:100]}"
    elif result["status"] == "OK":
        if "error" in result:
            # Python-level exception (caught inside probe)
            detail = f"EXCEPTION: {result['error']}"
            is_broken = True
        else:
            # Check invariants
            issues = []

            # count(visible_only=False) should == len(query(visible_only=False))
            if result["total"] != result["queried"]:
                issues.append(
                    f"count(vis=F)={result['total']} != len(query(vis=F))={result['queried']}"
                )

            # count(visible_only=True) should == len(query(visible_only=True))
            if result["visible"] != result["queried_vis"]:
                issues.append(
                    f"count(vis=T)={result['visible']} != len(query(vis=T))={result['queried_vis']}"
                )

            # count(visible_only=False) should always equal the true total
            # (since save_visible clears env vars before querying)
            if result["total"] != TOTAL:
                issues.append(
                    f"count(vis=F)={result['total']} != baseline total={TOTAL}"
                )

            # visible count should never exceed total
            if result["visible"] > TOTAL:
                issues.append(f"count(vis=T)={result['visible']} > total={TOTAL}")

            if issues:
                is_broken = True
                detail = "; ".join(issues)
            else:
                detail = (
                    f"total={result['total']}, visible={result['visible']}, "
                    f"queried={result['queried']}, queried_vis={result['queried_vis']}"
                )

    status_marker = "FAIL" if is_broken else "PASS"
    print(f"[{status_marker}] {label}")
    print(f"       env: {env_desc}")
    print(f"       {detail}")
    print()

    if is_broken:
        failures.append(result)

# ---------------------------------------------------------------------------
# Step 4: summary
# ---------------------------------------------------------------------------

print("=" * 72)
print(
    f"Results: {len(results) - len(failures)} passed, {len(failures)} failed "
    f"out of {len(results)} total"
)
print("=" * 72)

if failures:
    print("\nFAILED CASES:")
    for f in failures:
        print(f"  - {f['label']}: {f.get('status', 'UNKNOWN')}")
    print()
    sys.exit(1)
else:
    print("\nAll cases passed.")
    sys.exit(0)
