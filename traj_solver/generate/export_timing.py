"""
export_timing.py — export each transition as a timing file for the robot
(Motor/homing.py, Motor/playback.py).

Usage:
    python generate/export_timing.py [scenario ...]     default: all (-h for help)

Writes timings/<scenario>.csv: a header of '#' lines (units, conventions,
the calibration used and how to interpolate), then one row per waypoint:

  time_s                     seconds from the start of the transition
  q1_rad … q4_rad            joint motor positions (rad)
  dq1_rad_s … dq4_rad_s      joint motor speeds at the waypoint (rad/s)
  w_front_a_rad_s,           wheel motor speeds (rad/s), one per wheel with a
  w_back_a_rad_s             motor (motor_calibration.WHEELS): front and back
                             arm A.  Arm B wheels spin freely.

Every value is in its motor's own convention (core/motor_calibration.py).  With
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
(every phase eases in and out of rest, planner.ease).

Wheel speeds
------------
From the motion plan at each waypoint's instant, relative to the wheel's own
assembly — what the wheel's motor must turn at (planner.wheel_speeds).
Between waypoints the playback changes them linearly.

The waypoint rate is params.SAMPLE_RATE (6 by default; override with e.g.
TRAJ_SAMPLE_RATE=3).
"""

import csv
import os
import sys

# traj_solver/ (this script's parent folder) on the path, so `core` imports from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine.cli import TIMINGS_DIR, parse_args
from core.engine.timing_file import waypoints        # the columns (and joint speeds, by the rule above)
from core.engine.trajectory import solve_trajectory
from core.motor_calibration import JOINTS, WHEELS
from core.scenarios import SCENARIOS

OUT_DIR = TIMINGS_DIR


def _header(scn, traj):
    """The '#' lines at the top of a timing file."""
    joint_cal = "\n".join(f"#   {name:8s} {m.offset:+.6f}  {m.sign:+d}  {m.ratio:g}"
                          for name, m in JOINTS.items())
    wheel_cal = "\n".join(f"#   {name:8s}     -      {m.sign:+d}  {m.ratio:g}" if m is not None
                          else f"#   {name:8s} no motor (spins freely)"
                          for name, m in WHEELS.items())
    return f"""\
# {scn.label}: timing file for Motor/playback.py
# Written by traj_solver/generate/export_timing.py from solve {traj.key[:16]}
# {len(traj)} waypoints, {traj.rate:g} per second, {traj.times[-1]:.3f} s
#
# Columns
#   time_s                      seconds from the start of the transition
#   q1_rad .. q4_rad            joint motor positions (rad)
#   dq1_rad_s .. dq4_rad_s      joint motor speeds at the waypoint (rad/s)
#   w_<wheel>_rad_s             wheel motor speeds (rad/s), one per wheel with a motor
#
# Between waypoints each joint follows the cubic through its position and speed
# at both ends (cubic Hermite); wheel speeds change linearly.  The assemblies
# come to rest at every phase boundary, and the whole robot is at rest at the
# first and last waypoint.
#
# Values are in each motor's own convention (traj_solver/core/motor_calibration.py):
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


def main():
    names, _ = parse_args(__doc__)
    for name in names:
        export(name)


if __name__ == "__main__":
    main()
