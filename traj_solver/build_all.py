"""
build_all.py — regenerate every output file for all 4 scenarios.

Usage:
    python build_all.py              # all scenarios, all outputs
    python build_all.py --no-render  # skip GIF rendering (violation check only)

Outputs written:
    gifs/           floor_to_wall.gif, wall_to_ceiling.gif, outside.gif, thin_edge.gif
    wheel_omega/    wheel_omega_ftw.png, _wtc.png, _out.png, _te.png
    timings/        floor_to_wall.csv, wall_to_ceiling.csv, outside.csv, thin_edge.csv
    keyframes/      keyframes_floor_to_wall.png, ..._wall_to_ceiling.png, ..._outside.png, ..._thin_edge.png
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


# ── animate_transition (one call per scenario) ────────────────────────────────
for sc in SCENARIOS:
    cmd = [_PY, "animate_transition.py", sc]
    if NO_RENDER:
        cmd.append("--no-render")
    run(f"animate_transition  {sc}", cmd)

# ── wheel_omega (one call per scenario) ───────────────────────────────────────
for sc in SCENARIOS:
    run(f"wheel_omega         {sc}", [_PY, "wheel_omega.py", sc])

# ── robot_timing (single call handles all scenarios) ─────────────────────────
run("robot_timing        all", [_PY, "robot_timing.py", "all"])

# ── keyframes (single call handles all scenarios) ────────────────────────────
run("keyframes           all", [_PY, "keyframes.py", "all"])

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
