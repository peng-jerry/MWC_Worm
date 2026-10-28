"""
build_all.py — regenerate every output file for every transition in core/scenarios.py.

Usage:
    python build_all.py [scenario ...]               default: all (-h for help)
    python build_all.py [scenario ...] --no-render   everything except the GIFs
                                                     (animate.py still checks)

Runs the scripts in generate/; outputs written, one file per transition:
    outputs/gifs/<scenario>.gif        generate/animate.py
    outputs/speeds/<scenario>.png      generate/plot_speeds.py
    outputs/phases/<scenario>.png      generate/plot_phases.py
    outputs/snapshots/<scenario>.png   generate/plot_snapshot.py
    timings/<scenario>.csv             generate/export_timing.py   (read by Motor/)

The checks (checks/) are run separately.

Each step runs as its own process; the first solves every transition (or
loads it from .cache/) and the rest reuse the cache.  A step fails when its
script exits non-zero — generate/animate.py does when its check finds a problem — and
the build exits non-zero if any step failed.
"""

import subprocess
import sys
import time

from core.engine.cli import TRAJ_SOLVER_DIR, parse_args
from core.scenarios import SCENARIOS


def run(label, script, args):
    """Run one step — `script` with `args`, from traj_solver/ — and return
    whether it succeeded."""
    print(f"\n{'─' * 60}\n  {label}\n{'─' * 60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script, *args], cwd=TRAJ_SOLVER_DIR)
    if result.returncode == 0:
        print(f"  ✓  done in {time.time() - t0:.1f}s")
    else:
        print(f"  ✗  FAILED (exit {result.returncode})")
    return result.returncode == 0


def main():
    names, args = parse_args(__doc__, [(("--no-render",), dict(
        action="store_true", help="skip the GIFs (animate.py still solves and checks)"))])
    render = [] if not args.no_render else ["--no-render"]
    everything = "all" if names == list(SCENARIOS) else " ".join(names)
    # The animation and speed plots one transition at a time (so a failure names
    # it), then the steps that take every transition in one call.
    steps = ([(f"animate             {n}", "generate/animate.py", [n, *render]) for n in names]
             + [(f"plot_speeds         {n}", "generate/plot_speeds.py", [n]) for n in names]
             + [(f"{script:20s}{everything}", f"generate/{script}.py", names)
                for script in ("export_timing", "plot_phases", "plot_snapshot")])
    failed = [label for label, script, step_args in steps if not run(label, script, step_args)]

    print(f"\n{'═' * 60}\n  {len(steps) - len(failed)}/{len(steps)} steps passed")
    if failed:
        print("  FAILED:")
        for label in failed:
            print(f"    ✗  {label}")
    else:
        print("  All outputs up to date.")
    print(f"{'═' * 60}\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
