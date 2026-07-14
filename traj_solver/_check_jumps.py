"""Run all 4 scenarios and collect the built-in check output."""
import subprocess, sys, os
from pathlib import Path

HERE = str(Path(__file__).parent)

for scenario in ["floor_to_wall", "wall_to_ceiling", "outside", "thin_edge"]:
    result = subprocess.run(
        [sys.executable, "animate_transition.py", scenario, "--no-render"],
        capture_output=True, text=True, cwd=HERE,
    )
    print(f"\n{'='*56}")
    print(f"{scenario}")
    # Print only lines from the check section (skip per-frame "ok" noise)
    for line in result.stdout.splitlines():
        if any(tok in line for tok in ("LIMIT", "SIGN", "JUMP", "OMEGA", "Check:", "issue")):
            print(line)
    if result.returncode != 0 and not result.stdout.strip():
        print(f"  ERROR: {result.stderr[-400:]}")

print("\nDone.")
