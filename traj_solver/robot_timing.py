"""
robot_timing.py — export each transition as a timing file for the robot
(Motor/homing.py, Motor/playback.py).

Usage:
    python robot_timing.py [scenario | all]     default: all

Writes timings/<scenario>.csv: a header of '#' lines (units, conventions,
the calibration used and how to interpolate), then one row per waypoint:

  time_s                     seconds from the start of the transition
  q1_rad … q4_rad            joint motor positions (rad)
  dq1_rad_s … dq4_rad_s      joint motor speeds at the waypoint (rad/s)
  w_front_a_rad_s,           wheel motor speeds (rad/s): front arm A, front arm B,
  w_front_b_rad_s, …         back arm A, back arm B

Every value is in its motor's own convention (motor_calibration.py).  With
the calibration table at its pass-through defaults that is the display
convention (kinematics.q_display), counter-clockwise positive in the side view.

Joint speeds
------------
Between waypoints the playback moves each joint along the cubic through its
position and speed at both ends (cubic Hermite interpolation).  The speed at
a waypoint uses the steps on both sides of it, by the monotone ("PCHIP") rule
of Fritsch and Carlson: a weighted harmonic mean of the two step speeds when
they have the same sign, and 0 where the joint stops or turns back.  So the
cubic never overshoots a waypoint (every waypoint was checked by the
solver), and a joint holding still between two waypoints stays still.  The
first and last waypoints have speed 0: the robot starts and ends at rest
(every phase eases in and out of rest, scenarios.ease).

Wheel speeds
------------
From the motion plan at each waypoint's instant, relative to the wheel's own
assembly — what the wheel's motor must turn at (scenarios.wheel_speeds).
Between waypoints the playback changes them linearly.

The waypoint rate is timing.SAMPLE_RATE (6 by default; override with e.g.
TRAJ_SAMPLE_RATE=3).
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinematics import q_display
from motor_calibration import JOINTS, WHEELS
from scenarios import SCENARIOS, wheel_speeds
from trajectory import solve_trajectory

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, "timings")


def joint_speeds(times, qs):
    """
    Speed (rad/s) of each joint at each waypoint, for angles qs (N, J) at
    times (N,): the monotone rule of the module docstring, 0 at both ends.
    """
    h = np.diff(times)[:, None]
    step = np.diff(qs, axis=0) / h                   # speed over each step
    s0, s1, h0, h1 = step[:-1], step[1:], h[:-1], h[1:]
    w0, w1 = 2 * h1 + h0, h1 + 2 * h0                # Fritsch–Butland weights
    v = np.zeros_like(qs)
    with np.errstate(divide="ignore", invalid="ignore"):
        v[1:-1] = np.where(s0 * s1 > 0, (w0 + w1) / (w0 / s0 + w1 / s1), 0.0)
    return v


def _header(scn, traj):
    """The '#' lines at the top of a timing file."""
    joint_cal = "\n".join(f"#   {name:8s} {m.offset:+.6f}  {m.sign:+d}  {m.ratio:g}"
                          for name, m in JOINTS.items())
    wheel_cal = "\n".join(f"#   {name:8s}     -      {m.sign:+d}  {m.ratio:g}"
                          for name, m in WHEELS.items())
    return f"""\
# {scn.label}: timing file for Motor/playback.py
# Written by traj_solver/robot_timing.py from solve {traj.key[:16]}
# {len(traj)} waypoints, {traj.rate:g} per second, {traj.times[-1]:.3f} s
#
# Columns
#   time_s                      seconds from the start of the transition
#   q1_rad .. q4_rad            joint motor positions (rad)
#   dq1_rad_s .. dq4_rad_s      joint motor speeds at the waypoint (rad/s)
#   w_front_a_rad_s ..          wheel motor speeds (rad/s): front arm A, front arm B,
#                               back arm A, back arm B
#
# Between waypoints each joint follows the cubic through its position and speed
# at both ends (cubic Hermite); wheel speeds change linearly.  The assemblies
# come to rest at every phase boundary, and the whole robot is at rest at the
# first and last waypoint.
#
# Values are in each motor's own convention (traj_solver/motor_calibration.py):
#   joint motor position = offset + sign * ratio * joint angle
#   motor speed          = sign * ratio * joint or wheel speed
# joint angle: q1/q4 0 = link straight up from its calf, q2/q3 0 = straight,
#   counter-clockwise positive in the side view (as in the plots)
# wheel speed: relative to the wheel's own assembly, counter-clockwise positive
#   in the side view
#   motor    offset_rad  sign  ratio
{joint_cal}
{wheel_cal}
"""


def waypoints(traj):
    """The columns of a solved transition's timing file, {column name: (N,)
    array} in file order and in motor units."""
    # Joint angles in display convention, unwrapped so ±π crossings stay continuous.
    angles = np.unwrap(q_display(traj.Q), axis=0)
    rates = joint_speeds(traj.times, angles)
    _, wheel = wheel_speeds(traj.scenario, traj.phases, traj.times)

    columns = {"time_s": traj.times}
    for j, (name, m) in enumerate(JOINTS.items()):
        columns[f"{name}_rad"] = m.position(angles[:, j])
    for j, (name, m) in enumerate(JOINTS.items()):
        columns[f"d{name}_rad_s"] = m.speed(rates[:, j])
    for j, (name, m) in enumerate(WHEELS.items()):
        columns[f"w_{name}_rad_s"] = m.speed(wheel[:, j])
    return columns


def export(name, out_dir=OUT_DIR):
    """Solve one transition (or load it from the cache) and write its timing
    file; returns the output path."""
    scn = SCENARIOS[name]
    traj = solve_trajectory(scn)
    columns = waypoints(traj)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    with open(path, "w", newline="") as fh:
        fh.write(_header(scn, traj))
        writer = csv.writer(fh)
        writer.writerow(list(columns))
        for row in zip(*columns.values()):
            writer.writerow([f"{v:.6f}" for v in row])
    print(f"{name}: {traj.times[-1]:.1f} s, {len(traj)} waypoints at {traj.rate:g} Hz → {path}")
    return path


def read_timing(path):
    """The columns of a timing file, {column name: (N,) array}, in motor units."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    return {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg != "all" and arg not in SCENARIOS:
        print(f"Unknown scenario '{arg}'. Choose: {list(SCENARIOS)} or 'all'")
        sys.exit(1)
    for name in (SCENARIOS if arg == "all" else [arg]):
        export(name)


if __name__ == "__main__":
    main()
