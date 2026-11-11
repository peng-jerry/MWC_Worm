"""
preflight.py — go/no-go: may each timing file be played on the robot?

Usage:
    python checks/preflight.py [transition ...]     default: all (-h for help)

A transition is GO only if every item holds:

  file       timings/<transition>.csv exists and matches the current solve
  solve      the solve passes its check (core/engine/solve_check.py): hard
             rules, joint limits, bend direction, joint steps, wheel rolling
             speed, driven wheels on the solid
  played     played as Motor/playback.py plays it (core/engine/playback_sim.py),
             every hard rule holds at every update
  speed      joint speed on the played curves within MAX_JOINT_SPEED
  accel      joint motor acceleration on the played curves within JOINT_ACC_MAX
  position   joint motor positions, along the curves too, within
             ±POS_LIMIT_MOTOR, the window the motors accept
  wheels     wheel motor speeds within WHEEL_VEL_MAX
  travel     each driven wheel's speed, as sent, adds up to the plan's travel
             within TRAVEL_TOL (core/engine/playback_sim.py)

The limits are in core/motor_limits.py.  Prints each transition's items and
a GO / NO-GO line, and exits non-zero if any transition is NO-GO.
"""

import os
import sys

import numpy as np

# traj_solver/ (this script's parent folder) on the path, so `core` imports from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine.cli import run_each, TIMINGS_DIR, parse_args
from core.engine.playback_sim import CURVE_POINTS, TRAVEL_TOL, on_curve, simulate
from core.engine.solve_check import check as check_solve
from core.engine.timing_file import read_current
from core.engine.trajectory import solve_trajectory
from core.motor_calibration import JOINTS, WHEELS
from core.motor_limits import JOINT_ACC_MAX, MAX_JOINT_SPEED, POS_LIMIT_MOTOR, WHEEL_VEL_MAX
from core.transitions import TRANSITIONS


def items(name):
    """[(passed, item, detail)] for transition `name`, in the order of the docstring."""
    traj = solve_trajectory(TRANSITIONS[name])
    out = []
    cols = read_current(traj, os.path.join(TIMINGS_DIR, f"{name}.csv"))
    out.append((cols is not None, "file", "matches the current solve" if cols is not None
                else "missing or out of date: run generate/export_timing.py"))
    problems, _ = check_solve(traj)
    out.append((not problems, "solve", "passes its check" if not problems else
                f"{len(problems)} problem(s): python generate/animate.py {name} --no-render lists them"))
    if cols is None:
        return out                                   # nothing to play

    r = simulate(traj, cols)
    broken = "; ".join(f"{rule} at {len(ts)} updates" for rule, ts in r["broken"].items())
    out.append((not r["broken"], "played", broken or "every hard rule holds at every update"))
    speed, j, t, phase, _ = r["speed"]
    out.append((speed <= MAX_JOINT_SPEED, "speed", f"{np.degrees(speed):.1f}°/s on q{j + 1} "
                f"(limit {np.degrees(MAX_JOINT_SPEED):.1f}°/s), t = {t:.2f} s, {phase}"))

    # Motor units along the whole of every curve.  The calibration is linear,
    # so these are the curves Motor/playback.py sends.
    times = cols["time_s"]
    q = np.stack([cols[f"{n}_rad"] for n in JOINTS], axis=1)
    v = np.stack([cols[f"d{n}_rad_s"] for n in JOINTS], axis=1)
    kk = np.repeat(np.arange(len(times) - 1), CURVE_POINTS)
    aa = np.tile(np.linspace(0.0, 1.0, CURVE_POINTS), len(times) - 1)
    pos, _, acc = on_curve(times, q, v, kk, aa)
    i, j = np.unravel_index(np.argmax(np.abs(acc)), acc.shape)
    out.append((abs(acc[i, j]) <= JOINT_ACC_MAX, "accel", f"{abs(acc[i, j]):.0f} rad/s² on q{j + 1} "
                f"(limit {JOINT_ACC_MAX:g} rad/s²)"))
    names = list(JOINTS)
    beyond = [n for c, n in enumerate(names) if np.abs(pos[:, c]).max() > POS_LIMIT_MOTOR]
    i, j = np.unravel_index(np.argmax(np.abs(pos)), pos.shape)
    worst = f"{pos[i, j]:+.2f} rad on {names[j]} (limit ±{POS_LIMIT_MOTOR:g} rad)"
    out.append((not beyond, "position", worst + (f"; outside on {', '.join(beyond)}" if beyond else "")))
    wheel = {n: np.abs(cols[f"w_{n}_rad_s"]).max() for n, m in WHEELS.items() if m is not None}
    n = max(wheel, key=wheel.get)
    out.append((wheel[n] <= WHEEL_VEL_MAX, "wheels", f"{wheel[n]:.2f} rad/s on {n} "
                f"(limit {WHEEL_VEL_MAX:g} rad/s)"))
    e, n, t, phase = r["travel"]
    out.append((e <= TRAVEL_TOL, "travel", f"{e * 1000:.2f} mm off the plan on {n} "
                f"(limit {TRAVEL_TOL * 1000:g} mm), t = {t:.2f} s, {phase}"))
    return out


def main():
    names, _ = parse_args(__doc__)
    verdicts = {}

    def one(name):
        results = items(name)
        go = all(passed for passed, _, _ in results)
        verdicts[name] = go
        print(f"\n=== {name}: {'GO' if go else 'NO-GO'}")
        for passed, item, detail in results:
            print(f"  {'ok  ' if passed else 'FAIL'}  {item:9s} {detail}")
        return go

    failed = run_each(names, one)
    go = [n for n in names if verdicts.get(n)]
    print(f"\nGO: {', '.join(go) or 'none'}" + (f"   NO-GO: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
