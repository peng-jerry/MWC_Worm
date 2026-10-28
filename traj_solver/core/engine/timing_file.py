"""
timing_file.py — the columns of a timing file (timings/<scenario>.csv).

Shared by generate/export_timing.py, which writes them, and
checks/check_playback.py, which reads them back and plays them as the robot
does.  The file's format (columns, units, conventions, how to interpolate) is
described in export_timing.py.
"""

import csv

import numpy as np

from ..motor_calibration import JOINTS, WHEELS
from .kinematics import q_display
from .planner import wheel_speeds
from .trajectory import joint_speeds


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
    for j, (name, m) in enumerate(WHEELS.items()):       # same order as planner.WHEELS
        if m is not None:
            columns[f"w_{name}_rad_s"] = m.speed(wheel[:, j])
    return columns


def read_timing(path):
    """The columns of a timing file, {column name: (N,) array}, in motor units."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    return {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}
