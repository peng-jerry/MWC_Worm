"""
export_timing.py — export each transition as a timing file for the robot
(Motor/homing.py, Motor/playback.py).

Usage:
    python generate/export_timing.py [transition ...]     default: all (-h for help)

A solve that fails its check (core/engine/solve_check.py) is not written: any
file already there stays as it was, and the script exits non-zero.

Writes timings/<transition>.csv: a header of '#' lines (units, conventions,
the calibration used and how to interpolate), then one row per waypoint:

  time_s                     seconds from the start of the transition
  q1_rad … q4_rad            joint motor positions (rad)
  dq1_rad_s … dq4_rad_s      joint motor speeds at the waypoint (rad/s)
  w_front_a_rad_s,           wheel motor speeds (rad/s), one per wheel with a
  w_back_a_rad_s             motor (motor_calibration.WHEELS): front and back
                             arm A.  Arm B wheels spin freely.

Every value is in its motor's own convention (core/motor_calibration.py): the
display convention (kinematics.q_display), counter-clockwise positive in the
side view, with each motor's gear ratio, direction and zero offset applied.  So
the joint columns are 30 times the joint angles.

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
assembly — what the wheel's motor must turn at (wheels.wheel_speeds).
Between waypoints the playback changes them linearly.

The waypoint rate is params.SAMPLE_RATE (6 by default; override with e.g.
TRAJ_SAMPLE_RATE=3).
"""

import os
import sys

# traj_solver/ (this script's parent folder) on the path, so `core` imports from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine.cli import run_each, TIMINGS_DIR, parse_args
from core.engine.solve_check import check
from core.engine.timing_file import write_timing     # the columns (and joint speeds, by the rule above)
from core.engine.trajectory import solve_trajectory
from core.transitions import TRANSITIONS

OUT_DIR = TIMINGS_DIR


def export(name, out_dir=OUT_DIR):
    """Solve one transition (or load it from the cache) and write its timing
    file.  Returns the output path, or False if the solve fails its check."""
    trn = TRANSITIONS[name]
    traj = solve_trajectory(trn)
    problems, _ = check(traj)
    if problems:
        print(f"{name}: not written, the solve has {len(problems)} problem(s) "
              f"(python generate/animate.py {name} --no-render lists them)")
        return False
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    write_timing(traj, path)
    print(f"{name}: {traj.times[-1]:.1f} s, {len(traj)} waypoints at {traj.rate:g} Hz → {path}")
    return path


def main():
    names, _ = parse_args(__doc__)
    sys.exit(1 if run_each(names, export) else 0)


if __name__ == "__main__":
    main()
