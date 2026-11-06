"""
cli.py — what the scripts share (build_all.py, generate/, checks/): reading the
transitions named on the command line, and where their files go.

Every script takes any number of transition names from core/scenarios.py, or
`all` (also what no name means), plus its own options, and prints its
docstring for -h / --help.  It works through them with run_each, so one
transition that cannot be planned does not stop the others.
"""

import argparse
import os

from ..scenarios import SCENARIOS
from .planner import PlanError

TRAJ_SOLVER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMINGS_DIR = os.path.join(TRAJ_SOLVER_DIR, "timings")      # export_timing.py → Motor/


def output_dir(kind):
    """traj_solver/outputs/<kind> (gifs, speeds, phases, snapshots)."""
    return os.path.join(TRAJ_SOLVER_DIR, "outputs", kind)


def parse_args(doc, options=()):
    """
    Read a script's command line: transition names (or `all`, the default)
    and the script's own `options`, each a (flags, argparse keyword
    arguments) pair.  `doc` is the script's docstring, shown by --help.

    Returns (names, args): the transitions to work on — in SCENARIOS order
    for `all`, otherwise as given — and the parsed options.  An unknown name
    stops the script with the list of choices.
    """
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", nargs="*", metavar="scenario",
                        help=f"{', '.join(SCENARIOS)} or all (the default)")
    for flags, kwargs in options:
        parser.add_argument(*flags, **kwargs)
    args = parser.parse_args()
    unknown = [n for n in args.scenarios if n != "all" and n not in SCENARIOS]
    if unknown:
        parser.error(f"unknown transition(s) {', '.join(unknown)}; choose from {', '.join(SCENARIOS)} or all")
    if not args.scenarios or "all" in args.scenarios:
        return list(SCENARIOS), args
    return list(dict.fromkeys(args.scenarios)), args


def run_each(names, work):
    """
    Call work(name) for each transition in turn.  One that cannot be planned
    or solved (planner.PlanError, whose message says why) is reported and
    skipped, and the rest still run; work returning False also counts as a
    failure.  Returns the names that failed, after listing them, so a script
    can end with sys.exit(1 if run_each(...) else 0).
    """
    failed = []
    for name in names:
        try:
            ok = work(name) is not False
        except PlanError as e:
            why = str(e).removeprefix(f"{name}: ")      # the message names the transition already
            print(f"\n{name}: cannot be planned or solved — {why}")
            ok = False
        if not ok:
            failed.append(name)
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
    return failed
