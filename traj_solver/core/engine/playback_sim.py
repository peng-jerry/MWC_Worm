"""
playback_sim.py — a timing file played the way Motor/playback.py plays it,
and what happens between the waypoints.  checks/check_playback.py reports it
and checks/preflight.py judges it.
"""

import numpy as np

from .constraints import CLEARANCE_GROUPS, evaluate
from .hermite import hermite, hermite_acceleration, hermite_speed
from .kinematics import q_from_display, wrap
from .planner import plan_pose
from .wheels import wheel_speeds
from ..motor_calibration import JOINTS, WHEELS
from ..motor_limits import PLAYBACK_HZ
from ..robot import l1, l2, l3

CURVE_POINTS = 51           # points per curve segment for speed and acceleration
TRAVEL_TOL = 0.002          # m; how far a driven wheel's travel, as sent, may stray from the plan's


def on_curve(times, q, v, k, a):
    """
    Position, speed and acceleration on the curve of segment k (from waypoint
    k to k + 1) at fraction a ∈ [0, 1] of the way along it: the curve
    Motor/playback.py plays (core/engine/hermite.py).  q, v (N, J): waypoint
    positions and speeds; k, a (M,).  Returns three (M, J) arrays.
    """
    segment = (q[k], v[k], q[k + 1], v[k + 1], (times[k + 1] - times[k])[:, None], a[:, None])
    return hermite(*segment), hermite_speed(*segment), hermite_acceleration(*segment)


def playback_points(times, rate=PLAYBACK_HZ):
    """Times of the playback updates, and for each its segment k and fraction
    a — the way Motor/playback.py picks them."""
    t = np.minimum(np.arange(int(round(times[-1] * rate)) + 1) / rate, times[-1])
    k = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    a = np.clip((t - times[k]) / (times[k + 1] - times[k]), 0.0, 1.0)
    return t, k, a


def simulate(traj, cols):
    """Measure the playback of timing-file columns `cols` (motor units) of the
    solved trajectory `traj`.  Returns a dict of results."""
    trn = traj.transition
    times = cols["time_s"]
    # Back to the solver's conventions: display angles (rad) and side-view speeds.
    q = np.stack([m.angle(cols[f"{n}_rad"]) for n, m in JOINTS.items()], axis=1)
    v = np.stack([m.rate(cols[f"d{n}_rad_s"]) for n, m in JOINTS.items()], axis=1)
    driven = {n: m for n, m in WHEELS.items() if m is not None}
    w = np.stack([m.rate(cols[f"w_{n}_rad_s"]) for n, m in driven.items()], axis=1)

    def phase(t):
        """Name of the phase running at time t."""
        return traj.phases[traj.phase_at(t)].name

    # Geometry at every playback update.
    t, k, a = playback_points(times)
    pos, _, _ = on_curve(times, q, v, k, a)
    Qs = q_from_display(pos)
    poses = [plan_pose(trn, traj.phases, s) for s in t]
    x1, y1, th1, x2, y2, th2 = (np.array([p[key] for p in poses])
                                for key in ("x1", "y1", "theta1", "x2", "y2", "theta2"))
    phi1 = th1 + Qs[:, 0]
    phi2 = phi1 + Qs[:, 1]
    phi3 = phi2 + Qs[:, 2]
    end_x = x1 + l1 * np.cos(phi1) + l2 * np.cos(phi2) + l3 * np.cos(phi3)
    end_y = y1 + l1 * np.sin(phi1) + l2 * np.sin(phi2) + l3 * np.sin(phi3)
    gap = np.hypot(end_x - x2, end_y - y2)
    twist = np.abs(wrap(phi3 + Qs[:, 3] - th2))

    clearance = {g: np.empty(len(t)) for g in CLEARANCE_GROUPS}
    broken = {}
    for i, (Qi, p) in enumerate(zip(Qs, poses)):
        ev = evaluate(Qi[None], p, trn.env, trn.wg, l1, l2, l3)
        for g in CLEARANCE_GROUPS:
            clearance[g][i] = ev.clearance[g][0]
        for rule, amount in ev.violations(0).items():
            broken.setdefault(rule, []).append(t[i])
    at_waypoints = {g: min(info["clearance"][g] for info in traj.infos) for g in CLEARANCE_GROUPS}

    # Speed and acceleration along the whole of every curve.
    kk = np.repeat(np.arange(len(times) - 1), CURVE_POINTS)
    aa = np.tile(np.linspace(0.0, 1.0, CURVE_POINTS), len(times) - 1)
    tt = times[kk] + aa * (times[kk + 1] - times[kk])
    _, vel, acc = on_curve(times, q, v, kk, aa)
    i_vel, j_vel = np.unravel_index(np.argmax(np.abs(vel)), vel.shape)
    i_acc, j_acc = np.unravel_index(np.argmax(np.abs(acc)), acc.shape)
    step_speed = np.abs(np.diff(q, axis=0) / np.diff(times)[:, None]).max()

    w_acc = np.abs(np.diff(w, axis=0) / np.diff(times)[:, None])
    k_wacc, j_wacc = np.unravel_index(np.argmax(w_acc), w_acc.shape)

    # Wheel travel: what each driven wheel's motor is sent (every update's
    # speed, held until the next) against the plan's wheel speed, added up
    # over the transition.  The wheels run open loop, so this is how far the
    # wheels alone would take each assembly off the plan.
    _, planned = wheel_speeds(trn, traj.phases, t)
    planned = planned[:, [j for j, m in enumerate(WHEELS.values()) if m is not None]]
    sent = w[k] + a[:, None] * (w[k + 1] - w[k])
    dt = np.diff(t)[:, None]
    off = np.cumsum(np.vstack([np.zeros((1, w.shape[1])), sent[:-1] * dt - 0.5 * (planned[1:] + planned[:-1]) * dt]),
                    axis=0) * trn.wg.wheel_r
    i_off, j_off = np.unravel_index(np.argmax(np.abs(off)), off.shape)

    i_gap = int(np.argmax(gap))
    return {
        "times": times,
        "gap": (gap[i_gap], twist.max(), t[i_gap], phase(t[i_gap])),
        "clearance": {g: (c.min(), at_waypoints[g], t[int(np.argmin(c))], phase(t[int(np.argmin(c))]))
                      for g, c in clearance.items()},
        "broken": broken,
        "speed": (abs(vel[i_vel, j_vel]), j_vel, tt[i_vel], phase(tt[i_vel]), step_speed),
        "accel": (abs(acc[i_acc, j_acc]), j_acc, tt[i_acc], phase(tt[i_acc])),
        "wheel_speed": np.abs(w).max(),
        "wheel_accel": (w_acc[k_wacc, j_wacc], list(driven)[j_wacc], times[k_wacc], phase(times[k_wacc])),
        "travel": (abs(off[i_off, j_off]), list(driven)[j_off], t[i_off], phase(t[i_off])),
    }
