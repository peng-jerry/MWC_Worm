"""
check_playback.py — play a timing file the way the robot does, and measure
what happens between the waypoints.

Usage:
    python checks/check_playback.py [transition ...] [--rate R]     default: all (-h for help)

Motor/playback.py sends every joint, PLAYBACK_HZ times a second, the point on
the cubic Hermite curve through the two surrounding waypoints (position and
speed at both ends).  The solver checked only the waypoints themselves; this
script follows the same curves and checks the motion in between:

  chain gap     each joint follows its own curve, so between waypoints the
                chain no longer closes exactly: its far end (P3, from the
                front assembly and the interpolated joints) misses the back
                assembly's chain joint, which is where the plan puts it at that
                moment.  The joints are self-locking worms and will not give
                way, so the gap has to go into wheel slip or flex in the links.
  clearance     the smallest gap between the chain and the surfaces, l1 and
                l3, and the chain and each assembly (constraints.py) — it can
                dip below its value at the waypoints
  hard rules    clearance ≥ HARD_CLEARANCE, joint limits, bend direction
  joint speed   and acceleration of the curves, against the motor's limit
  wheel speed   and acceleration (wheel speeds change linearly between
                waypoints)
  wheel travel  how far each driven wheel's speed, as sent, adds up to away
                from the plan's: the wheels run open loop, so nothing else
                corrects it

The geometry is checked at every playback update; speeds and accelerations
are found along the whole of each curve.

By default the timing files in timings/ are checked; a file that no longer
matches the solved transition is reported (run export_timing.py).  With
--rate R the transitions are solved at R waypoints per second and checked
without writing any file, to compare waypoint rates.

Exits non-zero if a hard rule breaks, a joint goes over the motors' speed limit,
a wheel's travel strays more than TRAVEL_TOL (2 mm) from the plan, or a timing
file is missing or out of date.
"""

import os
import sys

import numpy as np

# traj_solver/ (this script's parent folder) on the path, so `core` imports from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine.cli import run_each, TIMINGS_DIR, parse_args
from core.engine.playback_sim import TRAVEL_TOL, simulate
from core.engine.timing_file import read_current, waypoints
from core.engine.trajectory import solve_trajectory
from core.motor_limits import MAX_JOINT_SPEED, PLAYBACK_HZ
from core.params import HARD_CLEARANCE, SOFT_CLEARANCE
from core.transitions import TRANSITIONS


def report(name, traj, r, source):
    """Print the results of simulate() for transition `name`; `source` says
    where the waypoints came from (a timing file, or solved for the check)."""
    times = r["times"]
    print(f"\n=== {name}   ({len(times)} waypoints, {(len(times) - 1) / times[-1]:g} per second, "
          f"{times[-1]:.1f} s; {source}; played at {PLAYBACK_HZ} Hz)")
    gap, twist, t, ph = r["gap"]
    print(f"  chain gap           max {gap * 1000:.1f} mm, {np.degrees(twist):.2f}°   "
          f"(t = {t:.2f} s, {ph})")
    worst = min(r["clearance"], key=lambda g: r["clearance"][g][0])
    for g, (c, c_wp, t, ph) in r["clearance"].items():
        flag = "  ← below hard" if c < HARD_CLEARANCE else ("  ← below soft" if c < SOFT_CLEARANCE else "")
        mark = "*" if g == worst else " "
        print(f"  clearance {g:15s}{mark}{c * 1000:5.1f} mm (waypoints {c_wp * 1000:5.1f} mm)   "
              f"t = {t:.2f} s, {ph}{flag}")
    if r["broken"]:
        for rule, ts in r["broken"].items():
            print(f"  HARD RULE BROKEN    {rule}: {len(ts)} updates, t = {ts[0]:.2f} … {ts[-1]:.2f} s")
    else:
        print("  hard rules          met at every update")
    s, j, t, ph, step = r["speed"]
    over = "  ← over the motor limit" if s > MAX_JOINT_SPEED else ""
    print(f"  joint speed         max {np.degrees(s):.1f}°/s on q{j + 1} (t = {t:.2f} s, {ph})   "
          f"waypoint steps {np.degrees(step):.1f}°/s, motor {np.degrees(MAX_JOINT_SPEED):.1f}°/s{over}")
    acc, j, t, ph = r["accel"]
    print(f"  joint acceleration  max {np.degrees(acc):.0f}°/s² on q{j + 1} (t = {t:.2f} s, {ph})")
    wa, wheel, t, ph = r["wheel_accel"]
    print(f"  wheel motor         max {r['wheel_speed']:.2f} rad/s;  acceleration max {wa:.1f} rad/s² "
          f"({wa * traj.transition.wg.wheel_r:.2f} m/s² at the rim) on {wheel} (t = {t:.2f} s, {ph})")
    e, wheel, t, ph = r["travel"]
    flag = "  ← over the tolerance" if e > TRAVEL_TOL else ""
    print(f"  wheel travel        max {e * 1000:.2f} mm off the plan on {wheel} (t = {t:.2f} s, {ph}), "
          f"tolerance {TRAVEL_TOL * 1000:g} mm{flag}")


def main():
    names, args = parse_args(__doc__, [(("--rate",), dict(
        type=float, metavar="R", help="solve at R waypoints per second instead of reading timings/ "
                                      "(writes no files)"))])
    rate = args.rate

    def one(name):
        if rate is None:
            traj = solve_trajectory(TRANSITIONS[name])
            cols = read_current(traj, os.path.join(TIMINGS_DIR, f"{name}.csv"))
            if cols is None:
                print(f"\n=== {name}: timings/{name}.csv is missing or out of date — "
                      f"run generate/export_timing.py first")
                return False
            source = f"timings/{name}.csv"
        else:
            traj = solve_trajectory(TRANSITIONS[name], rate)
            cols, source = waypoints(traj), "solved for this check"
        r = simulate(traj, cols)
        report(name, traj, r, source)
        return not r["broken"] and r["speed"][0] <= MAX_JOINT_SPEED and r["travel"][0] <= TRAVEL_TOL

    sys.exit(1 if run_each(names, one) else 0)


if __name__ == "__main__":
    main()
