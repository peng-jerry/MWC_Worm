"""
trajectory.py — solve a transition's joint angles once, and reuse the result.

  solve_trajectory   samples the phases at timing.SAMPLE_RATE, fixes both ends
                     to the cruise posture, and lets trajectory_ik choose every
                     frame's joint angles together.  If a joint would have to
                     move faster than the limit, that phase is slowed down and
                     the trajectory is solved again.

Cache
-----
A solve takes from about 2 s (inside corners) to 20 s (outside corner, thin
edge) and every script needs the same result, so each solved transition is
saved in .cache/.  Its key is built from the transition's name, the sample
rate, the numpy and scipy versions, and the source of this module and every
module it imports from this directory (found by reading the imports, so a
new module is picked up automatically).  Any change to the code or the
parameters therefore changes the key, and a stale result is never reused.
Older entries for the same transition and rate are deleted when a new one is
saved.

Set TRAJ_NO_CACHE=1 to solve afresh for one run (the result is still saved).
"""

import ast
import hashlib
import os
import pickle
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy

from kinematics import q_display
from scenarios import (Scenario, l1, l2, l3, plan_phases, phase_starts, sample_poses,
                       cruise_posture, follow_gap)
from timing import SAMPLE_RATE, PLAN_JOINT_SPEED
from trajectory_ik import solve_trajectory_dp

_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_DIR, ".cache")

# A phase is slowed down when a joint exceeds PLAN_JOINT_SPEED by more than
# RETIME_TOLERANCE in it, as the robot plays it (_phase_peak_speeds); the
# trajectory is then solved again, at most MAX_RETIMES times.
RETIME_TOLERANCE = 0.005
MAX_RETIMES = 5


@dataclass
class Trajectory:
    """
    A solved transition, one entry per frame.

    times    (N,) seconds
    poses    assembly poses per frame (dicts, see scenarios.sample_poses)
    Q        (N, 4) joint angles (rad)
    infos    per-frame diagnostics (trajectory_ik.solve_trajectory_dp)
    phases   the phases with their final durations
    retimes  how many times phases were slowed down for the joint speed limit
    gap      gap (m) the partner aims to keep while following a pivot, or None when
             it parks (scenarios.follow_gap)
    key      the cache key: identifies the code and parameters that made it
    cached   True when it was loaded from the cache instead of solved

    Iterating yields (q, info, pose) for each frame.
    """
    scenario: Scenario
    phases: list
    times: np.ndarray
    poses: list
    Q: np.ndarray
    infos: list
    retimes: int
    gap: Optional[float]
    key: str = ""
    cached: bool = False

    @property
    def rate(self):
        """Frames per second."""
        return (len(self.times) - 1) / self.times[-1]

    @property
    def phase_starts(self):
        """Start time of each phase plus the end time."""
        return phase_starts(self.phases)

    def phase_at(self, t):
        """Index of the phase running at time t."""
        return min(int(np.searchsorted(self.phase_starts, t, side="right")) - 1,
                   len(self.phases) - 1)

    def __len__(self):
        return len(self.times)

    def __iter__(self):
        return iter(zip(self.Q, self.infos, self.poses))


def joint_speeds(times, qs):
    """
    Speed (rad/s) of each joint at each waypoint, for angles qs (N, J) at
    times (N,), as the robot plays them (robot_timing.py): the monotone
    ("PCHIP") rule of Fritsch and Carlson — a weighted harmonic mean of the
    step speeds on both sides when they have the same sign, 0 where the joint
    stops or turns back — and 0 at both ends.
    """
    h = np.diff(times)[:, None]
    step = np.diff(qs, axis=0) / h                   # speed over each step
    s0, s1, h0, h1 = step[:-1], step[1:], h[:-1], h[1:]
    w0, w1 = 2 * h1 + h0, h1 + 2 * h0                # Fritsch–Butland weights
    v = np.zeros_like(qs)
    with np.errstate(divide="ignore", invalid="ignore"):
        v[1:-1] = np.where(s0 * s1 > 0, (w0 + w1) / (w0 / s0 + w1 / s1), 0.0)
    return v


def _phase_peak_speeds(times, Q, phases):
    """
    Fastest joint speed (rad/s) within each phase, on the curve the robot
    plays: between waypoints, the cubic through both ends' positions and
    joint_speeds (cubic Hermite, Motor/playback.py).  Near a speed peak it
    is faster than the step between the waypoints.  Each segment between two
    waypoints counts toward the phase that contains its midpoint.
    """
    D = np.unwrap(np.array([q_display(q) for q in Q]), axis=0)
    V = joint_speeds(times, D)
    a = np.linspace(0.0, 1.0, 33)[:, None, None]     # through each segment, finer than 100 Hz
    # Time derivative of the cubic Hermite (basis derivatives h00', h10', h11').
    speed = ((6 * a * a - 6 * a) * (D[:-1] - D[1:]) / np.diff(times)[:, None]
             + (3 * a * a - 4 * a + 1) * V[:-1] + (3 * a * a - 2 * a) * V[1:])
    speed = np.abs(speed).max(axis=(0, 2))
    mid = 0.5 * (times[1:] + times[:-1])
    starts = phase_starts(phases)
    return np.array([speed[(mid > starts[i]) & (mid < starts[i + 1])].max(initial=0.0)
                     for i in range(len(phases))])


def _solve(scn, rate):
    """The solve itself (see solve_trajectory)."""
    gap = follow_gap(scn, rate) if scn.partner_follows else None
    min_durations = None
    for retimes in range(MAX_RETIMES + 1):
        phases = plan_phases(scn, min_durations, gap)
        times, poses = sample_poses(scn, phases, rate)
        frame_rate = (len(times) - 1) / times[-1]
        Q, infos = solve_trajectory_dp(
            poses, scn.env, scn.wg, l1, l2, l3,
            dt=1.0 / frame_rate,
            max_step=PLAN_JOINT_SPEED / frame_rate,
            q_start=cruise_posture(scn, poses[0]),
            q_end=cruise_posture(scn, poses[-1]),
        )
        peaks = _phase_peak_speeds(times, Q, phases)
        too_fast = peaks > PLAN_JOINT_SPEED * (1 + RETIME_TOLERANCE)
        if not too_fast.any() or retimes == MAX_RETIMES:
            break
        min_durations = [p.duration * (pk / PLAN_JOINT_SPEED if fast else 1.0)
                         for p, pk, fast in zip(phases, peaks, too_fast)]
    return Trajectory(scn, phases, times, poses, Q, infos, retimes, gap)


# ------------------------------------------------------------------ #
#  Cache                                                              #
# ------------------------------------------------------------------ #

def _solver_sources():
    """{module name: source bytes} for this module and every module it
    imports from this directory, directly or indirectly."""
    sources, todo = {}, [os.path.splitext(os.path.basename(__file__))[0]]
    while todo:
        name = todo.pop()
        if name in sources:
            continue
        with open(os.path.join(_DIR, name + ".py"), "rb") as fh:
            sources[name] = fh.read()
        for node in ast.walk(ast.parse(sources[name])):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            else:
                continue
            todo += [n for n in names if os.path.exists(os.path.join(_DIR, n + ".py"))]
    return sources


def cache_key(scn, rate=SAMPLE_RATE):
    """The cache key for transition `scn` sampled at `rate` (see the module docstring)."""
    h = hashlib.sha256()
    for part in (scn.name, repr(float(rate)), np.__version__, scipy.__version__):
        h.update(part.encode() + b"\0")
    for name, src in sorted(_solver_sources().items()):
        h.update(name.encode() + b"\0" + src + b"\0")
    return h.hexdigest()


def _cache_path(scn, rate, key):
    return os.path.join(CACHE_DIR, f"{scn.name}_{float(rate):g}fps_{key[:16]}.pkl")


def _load(scn, rate, key):
    """The cached trajectory, or None if there is none (or it does not rebuild)."""
    path = _cache_path(scn, rate, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        # Phases hold functions, which are not saved: rebuild them from the
        # saved durations and check they come out the same.
        phases = plan_phases(scn, d["durations"], d["gap"])
        if not np.allclose([p.duration for p in phases], d["durations"]):
            return None
    except Exception:                        # unreadable or from an older format
        return None
    return Trajectory(scn, phases, d["times"], d["poses"], d["Q"], d["infos"], d["retimes"],
                      d["gap"], key, cached=True)


def _save(traj, rate):
    """Save a solved trajectory and delete older entries for the same transition and rate."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(traj.scenario, rate, traj.key)
    stem = os.path.basename(path).rsplit("_", 1)[0] + "_"
    d = {"durations": [p.duration for p in traj.phases], "gap": traj.gap, "times": traj.times,
         "poses": traj.poses, "Q": traj.Q, "infos": traj.infos, "retimes": traj.retimes}
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        pickle.dump(d, fh)
    os.replace(tmp, path)
    for f in os.listdir(CACHE_DIR):
        if f.startswith(stem) and f.endswith(".pkl") and f != os.path.basename(path):
            os.remove(os.path.join(CACHE_DIR, f))


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #

def solve_trajectory(scn, rate=SAMPLE_RATE):
    """
    Plan the transition `scn`, sample it at `rate` frames per second and solve
    every frame's joint angles together, with both ends fixed to the cruise
    posture.  Returns a Trajectory, from the cache when the code and
    parameters are unchanged since it was last solved.

    At an outside corner or edge the follow gap is chosen first
    (scenarios.follow_gap).  The phases are then timed from the wheel and
    pivot speeds alone (scenarios.plan_phases).  If the solved joints must
    move faster than timing.PLAN_JOINT_SPEED in some phase — as the robot
    plays them, between waypoints too — that phase is lengthened in
    proportion and the trajectory is solved again.
    """
    key = cache_key(scn, rate)
    if os.environ.get("TRAJ_NO_CACHE", "") in ("", "0"):
        traj = _load(scn, rate, key)
        if traj is not None:
            return traj
    traj = _solve(scn, rate)
    traj.key = key
    _save(traj, rate)
    return traj
