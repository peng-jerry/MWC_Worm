"""
robot_timing.py — export per-frame joint angles and wheel speeds for physical robot control.

Usage:
    python robot_timing.py [scenario | all]     default: all

Outputs one CSV per scenario to traj_solver/timings/ with 13 columns:

  time_s | q1_rad  dq1_rad_s  w_fl_rad_s | q2_rad  dq2_rad_s  w_fr_rad_s |
          | q3_rad  dq3_rad_s  w_bl_rad_s | q4_rad  dq4_rad_s  w_br_rad_s

  time_s        seconds from the start of the transition
  q1–q4         joint angles (rad), display/calf-up convention (0 = straight up from calf)
  dq1–dq4       joint angular velocity (rad/s): change since the previous frame ÷ frame interval
  w_fl / w_fr   front assembly arm-A / arm-B wheel ω (rad/s), CCW positive
  w_bl / w_br   back assembly  arm-A / arm-B wheel ω (rad/s), CCW positive

Column grouping pairs each chain joint with one wheel:
  q1 ↔ front arm-A wheel    q2 ↔ front arm-B wheel
  q3 ↔ back arm-A wheel     q4 ↔ back arm-B wheel

Wheel ω is the rolling rate on whichever surface the wheel touches
(scenarios.wheel_spin): ω = (ẑ × n)·Δcentre / (r Δt) for surface normal n,
and 0 while a wheel is at the corner, in the air or touching both surfaces.

The waypoint rate is timing.SAMPLE_RATE (6 by default; override with e.g.
TRAJ_SAMPLE_RATE=3).  Transition durations come from the speed limits in
timing.py and the phases in scenarios.py.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from animate_transition import solve_trajectory
from kinematics import q_display
from scenarios import SCENARIOS, wheel_spin

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, "timings")

HEADER = [
    "time_s",
    "q1_rad", "dq1_rad_s", "w_fl_rad_s",
    "q2_rad", "dq2_rad_s", "w_fr_rad_s",
    "q3_rad", "dq3_rad_s", "w_bl_rad_s",
    "q4_rad", "dq4_rad_s", "w_br_rad_s",
]


def export(name, out_dir=OUT_DIR):
    """Solve one transition and write its CSV; returns the output path."""
    scn = SCENARIOS[name]
    traj = solve_trajectory(scn)

    # Joint angles in display convention, unwrapped so ±π crossings stay continuous.
    qs = np.unwrap(np.array([q_display(q) for q in traj.Q]), axis=0)
    # Joint velocity over the step that ends at each frame (0 at frame 0).
    dqs = np.diff(qs, axis=0, prepend=qs[[0]]) / np.diff(traj.times, prepend=-1.0)[:, None]
    dqs[0] = 0.0
    spin = wheel_spin(scn, traj.poses, traj.times)          # (N, 4) fl, fr, bl, br

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for k, t in enumerate(traj.times):
            row = [f"{t:.6f}"]
            for j in range(4):
                row += [f"{qs[k, j]:.6f}", f"{dqs[k, j]:.6f}", f"{spin[k, j]:.6f}"]
            writer.writerow(row)
    print(f"{name}: {traj.times[-1]:.1f} s, {len(traj)} frames at {traj.rate:g} Hz → {path}")
    return path


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg != "all" and arg not in SCENARIOS:
        print(f"Unknown scenario '{arg}'. Choose: {list(SCENARIOS)} or 'all'")
        sys.exit(1)
    for name in (SCENARIOS if arg == "all" else [arg]):
        export(name)


if __name__ == "__main__":
    main()
