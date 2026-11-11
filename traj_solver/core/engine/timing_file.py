"""
timing_file.py — the columns of a timing file (timings/<transition>.csv).

Shared by generate/export_timing.py, which writes them (write_timing), and the
checks that read them back (checks/check_playback.py, checks/preflight.py).  The file's format (columns, units, conventions, how to interpolate) is
described in export_timing.py.
"""

import csv
import os

import numpy as np

from ..motor_calibration import JOINTS, WHEELS
from .kinematics import q_display
from .trajectory import joint_speeds
from .wheels import wheel_speeds

FILE_TOL = 2e-6     # a timing file matches the solve within its 6-decimal rounding


def waypoints(traj):
    """The columns of a solved transition's timing file, {column name: (N,)
    array} in file order and in motor units."""
    # Joint angles in display convention, unwrapped so ±π crossings stay continuous.
    angles = np.unwrap(q_display(traj.Q), axis=0)
    rates = joint_speeds(traj.times, angles)
    _, wheel = wheel_speeds(traj.transition, traj.phases, traj.times)

    columns = {"time_s": traj.times}
    for j, (name, m) in enumerate(JOINTS.items()):
        columns[f"{name}_rad"] = m.position(angles[:, j])
    for j, (name, m) in enumerate(JOINTS.items()):
        columns[f"d{name}_rad_s"] = m.speed(rates[:, j])
    for j, (name, m) in enumerate(WHEELS.items()):       # same order as wheels.WHEELS
        if m is not None:
            columns[f"w_{name}_rad_s"] = m.speed(wheel[:, j])
    return columns


def _header(traj):
    """The '#' lines at the top of a timing file."""
    trn = traj.transition
    joint_cal = "\n".join(f"#   {name:8s} {m.offset:+.6f}  {m.sign:+d}  {m.ratio:g}"
                          for name, m in JOINTS.items())
    wheel_cal = "\n".join(f"#   {name:8s}     -      {m.sign:+d}  {m.ratio:g}" if m is not None
                          else f"#   {name:8s} no motor (spins freely)"
                          for name, m in WHEELS.items())
    return f"""\
# {trn.label}: timing file for Motor/playback.py
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


def write_timing(traj, path):
    """Write the timing file of solved transition `traj` to `path`: the '#'
    header, then one row per waypoint, every value to 6 decimals."""
    columns = waypoints(traj)
    with open(path, "w", newline="") as fh:
        fh.write(_header(traj))
        writer = csv.writer(fh)
        writer.writerow(list(columns))
        for row in zip(*columns.values()):
            writer.writerow([f"{v:.6f}" for v in row])


def read_timing(path):
    """The columns of a timing file, {column name: (N,) array}, in motor units."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    return {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}


def read_current(traj, path):
    """The columns of the timing file at `path`, or None if it is missing or
    does not match the solved trajectory `traj` (it is out of date)."""
    if not os.path.exists(path):
        return None
    cols, expected = read_timing(path), waypoints(traj)
    if list(cols) != list(expected) or len(cols["time_s"]) != len(expected["time_s"]):
        return None
    if any(np.abs(cols[c] - expected[c]).max() > FILE_TOL for c in cols):
        return None
    return cols
