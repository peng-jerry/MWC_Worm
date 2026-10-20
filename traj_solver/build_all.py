"""
build_all.py — regenerate every output file for all 4 scenarios.

Usage:
    python build_all.py              # all scenarios, all outputs
    python build_all.py --no-render  # skip GIF rendering (violation check only)

Outputs written, one file per scenario (floor_to_wall, wall_to_ceiling, outside, thin_edge):
    outputs/gifs/<scenario>.gif        animate.py
    outputs/speeds/<scenario>.png      plot_speeds.py
    outputs/phases/<scenario>.png      plot_phases.py
    outputs/snapshots/<scenario>.png   plot_snapshot.py
    timings/<scenario>.csv             export_timing.py   (read by Motor/)
"""

import subprocess
import sys
import time
import os

_DIR     = os.path.dirname(os.path.abspath(__file__))
_PY      = sys.executable
SCENARIOS = ["floor_to_wall", "wall_to_ceiling", "outside", "thin_edge"]
NO_RENDER = "--no-render" in sys.argv

_passed = []
_failed = []


def run(label, cmd):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=_DIR)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  ✓  done in {elapsed:.1f}s")
        _passed.append(label)
    else:
        print(f"  ✗  FAILED (exit {result.returncode})")
        _failed.append(label)


# ── animate (one call per scenario) ───────────────────────────────────────────
for sc in SCENARIOS:
    cmd = [_PY, "animate.py", sc]
    if NO_RENDER:
        cmd.append("--no-render")
    run(f"animate             {sc}", cmd)

# ── plot_speeds (one call per scenario) ───────────────────────────────────────
for sc in SCENARIOS:
    run(f"plot_speeds         {sc}", [_PY, "plot_speeds.py", sc])

# ── export_timing (single call handles all scenarios) ────────────────────────
run("export_timing       all", [_PY, "export_timing.py", "all"])

# ── plot_phases (single call handles all scenarios) ──────────────────────────
run("plot_phases         all", [_PY, "plot_phases.py", "all"])

# ── plot_snapshot (single call handles all scenarios) ────────────────────────
run("plot_snapshot       all", [_PY, "plot_snapshot.py", "all"])

# ── Summary ───────────────────────────────────────────────────────────────────
total = len(_passed) + len(_failed)
print(f"\n{'═'*60}")
print(f"  {len(_passed)}/{total} steps passed")
if _failed:
    print("  FAILED:")
    for f in _failed:
        print(f"    ✗  {f}")
else:
    print("  All outputs up to date.")
print(f"{'═'*60}\n")

sys.exit(1 if _failed else 0)
