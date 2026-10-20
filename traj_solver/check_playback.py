"""
check_playback.py — play a timing file the way the robot does, and measure
what happens between the waypoints.

Usage:
    python check_playback.py [scenario | all] [--rate R]     default: all

Motor/playback.py sends every joint, PLAYBACK_HZ times a second, the point on
the cubic Hermite curve through the two surrounding waypoints (position and
speed at both ends).  The solver checked only the waypoints themselves; this
script follows the same curves and checks the motion in between:

  chain gap     each joint follows its own curve, so between waypoints the
                chain no longer closes exactly: its far end (P3, from the
                front assembly and the interpolated joints) misses the back
                assembly's chain joint, which is where the plan puts it at that
                moment.  The motors' compliance has to absorb the gap.
  clearance     the smallest gap between the chain and the surfaces, l1 and
                l3, and the chain and each assembly (constraints.py) — it can
                dip below its value at the waypoints
  hard rules    clearance ≥ HARD_CLEARANCE, joint limits, bend direction
  joint speed   and acceleration of the curves, against the motor's limit
  wheel speed   and acceleration (wheel speeds change linearly between
                waypoints)

The geometry is checked at every playback update; speeds and accelerations
are found along the whole of each curve.

By default the timing files in timings/ are checked; a file that no longer
matches the solved transition is reported (run export_timing.py).  With
--rate R the transitions are solved at R waypoints per second and checked
without writing any file, to compare waypoint rates.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.constraints import CLEARANCE_GROUPS, HARD_CLEARANCE, SOFT_CLEARANCE, evaluate
from core.params import MAX_JOINT_SPEED
from core.scenarios import SCENARIOS, l1, l2, l3, plan_pose
from core.trajectory import solve_trajectory
from export_timing import OUT_DIR, read_timing, waypoints
from motor_calibration import JOINTS, WHEELS

PLAYBACK_HZ = 100           # Motor/playback.py INTERP_HZ
CURVE_POINTS = 51           # points per curve segment for speed and acceleration
FILE_TOL = 2e-6             # a timing file matches the solve within its 6-decimal rounding

# Display convention → solver joint angles (inverse of kinematics.q_display).
_SHIFT = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2])


def _wrap(a):
    """Wrap angles to [-π, π)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def hermite(times, q, v, k, a):
    """
    Position, speed and acceleration on the cubic Hermite curve of segment k
    (from waypoint k to k + 1) at fraction a ∈ [0, 1] of the way along it.
    q, v (N, J): waypoint positions and speeds; k, a (M,).  Returns three (M, J)
    arrays.  Same curve as hermite() in Motor/playback.py.
    """
    T = (times[k + 1] - times[k])[:, None]
    a = a[:, None]
    q0, q1, m0, m1 = q[k], q[k + 1], T * v[k], T * v[k + 1]
    pos = ((2 * a**3 - 3 * a**2 + 1) * q0 + (a**3 - 2 * a**2 + a) * m0
           + (-2 * a**3 + 3 * a**2) * q1 + (a**3 - a**2) * m1)
    vel = ((6 * a**2 - 6 * a) * (q0 - q1) + (3 * a**2 - 4 * a + 1) * m0 + (3 * a**2 - 2 * a) * m1) / T
    acc = ((12 * a - 6) * (q0 - q1) + (6 * a - 4) * m0 + (6 * a - 2) * m1) / T**2
    return pos, vel, acc


def playback_points(times, rate=PLAYBACK_HZ):
    """Times of the playback updates, and for each its segment k and fraction
    a — the way Motor/playback.py picks them."""
    t = np.minimum(np.arange(int(round(times[-1] * rate)) + 1) / rate, times[-1])
    k = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    a = np.clip((t - times[k]) / (times[k + 1] - times[k]), 0.0, 1.0)
    return t, k, a


def _from_file(traj):
    """The timing file's columns, or None if it does not match the solved trajectory."""
    path = os.path.join(OUT_DIR, f"{traj.scenario.name}.csv")
    if not os.path.exists(path):
        return None
    cols, expected = read_timing(path), waypoints(traj)
    if list(cols) != list(expected) or len(cols["time_s"]) != len(expected["time_s"]):
        return None
    if any(np.abs(cols[c] - expected[c]).max() > FILE_TOL for c in cols):
        return None
    return cols


def check(traj, cols):
    """Measure the playback of timing-file columns `cols` (motor units) of the
    solved trajectory `traj`.  Returns a dict of results."""
    scn = traj.scenario
    times = cols["time_s"]
    # Back to the solver's conventions: display angles (rad) and side-view speeds.
    q = np.stack([m.angle(cols[f"{n}_rad"]) for n, m in JOINTS.items()], axis=1)
    v = np.stack([m.rate(cols[f"d{n}_rad_s"]) for n, m in JOINTS.items()], axis=1)
    driven = {n: m for n, m in WHEELS.items() if m is not None}
    w = np.stack([m.rate(cols[f"w_{n}_rad_s"]) for n, m in driven.items()], axis=1)

    def phase(t):
        return traj.phases[traj.phase_at(t)].name

    # Geometry at every playback update.
    t, k, a = playback_points(times)
    pos, _, _ = hermite(times, q, v, k, a)
    Qs = pos + _SHIFT                                   # solver joint angles
    poses = [plan_pose(scn, traj.phases, s) for s in t]
    x1, y1, th1, x2, y2, th2 = (np.array([p[key] for p in poses])
                                for key in ("x1", "y1", "theta1", "x2", "y2", "theta2"))
    phi1 = th1 + Qs[:, 0]
    phi2 = phi1 + Qs[:, 1]
    phi3 = phi2 + Qs[:, 2]
    end_x = x1 + l1 * np.cos(phi1) + l2 * np.cos(phi2) + l3 * np.cos(phi3)
    end_y = y1 + l1 * np.sin(phi1) + l2 * np.sin(phi2) + l3 * np.sin(phi3)
    gap = np.hypot(end_x - x2, end_y - y2)
    twist = np.abs(_wrap(phi3 + Qs[:, 3] - th2))

    clearance = {g: np.empty(len(t)) for g in CLEARANCE_GROUPS}
    broken = {}
    for i, (Qi, p) in enumerate(zip(Qs, poses)):
        ev = evaluate(Qi[None], p, scn.env, scn.wg, l1, l2, l3)
        for g in CLEARANCE_GROUPS:
            clearance[g][i] = ev.clearance[g][0]
        for rule, amount in ev.violations(0).items():
            broken.setdefault(rule, []).append(t[i])
    at_waypoints = {g: min(info["clearance"][g] for info in traj.infos) for g in CLEARANCE_GROUPS}

    # Speed and acceleration along the whole of every curve.
    kk = np.repeat(np.arange(len(times) - 1), CURVE_POINTS)
    aa = np.tile(np.linspace(0.0, 1.0, CURVE_POINTS), len(times) - 1)
    tt = times[kk] + aa * (times[kk + 1] - times[kk])
    _, vel, acc = hermite(times, q, v, kk, aa)
    i_vel, j_vel = np.unravel_index(np.argmax(np.abs(vel)), vel.shape)
    i_acc, j_acc = np.unravel_index(np.argmax(np.abs(acc)), acc.shape)
    step_speed = np.abs(np.diff(q, axis=0) / np.diff(times)[:, None]).max()

    w_acc = np.abs(np.diff(w, axis=0) / np.diff(times)[:, None])
    k_wacc, j_wacc = np.unravel_index(np.argmax(w_acc), w_acc.shape)

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
    }


def report(name, traj, r, source):
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
        print(f"  hard rules          met at every update")
    s, j, t, ph, step = r["speed"]
    over = "  ← over the motor limit" if s > MAX_JOINT_SPEED else ""
    print(f"  joint speed         max {np.degrees(s):.1f}°/s on q{j + 1} (t = {t:.2f} s, {ph})   "
          f"waypoint steps {np.degrees(step):.1f}°/s, motor {np.degrees(MAX_JOINT_SPEED):.1f}°/s{over}")
    acc, j, t, ph = r["accel"]
    print(f"  joint acceleration  max {np.degrees(acc):.0f}°/s² on q{j + 1} (t = {t:.2f} s, {ph})")
    wa, wheel, t, ph = r["wheel_accel"]
    print(f"  wheel motor         max {r['wheel_speed']:.2f} rad/s;  acceleration max {wa:.1f} rad/s² "
          f"({wa * traj.scenario.wg.wheel_r:.2f} m/s² at the rim) on {wheel} (t = {t:.2f} s, {ph})")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rate = None
    if "--rate" in sys.argv:
        rate = float(sys.argv[sys.argv.index("--rate") + 1])
        args = [a for a in args if a != sys.argv[sys.argv.index("--rate") + 1]]
    arg = args[0] if args else "all"
    if arg != "all" and arg not in SCENARIOS:
        print(f"Unknown scenario '{arg}'. Choose: {list(SCENARIOS)} or 'all'")
        sys.exit(1)
    for name in (SCENARIOS if arg == "all" else [arg]):
        if rate is None:
            traj = solve_trajectory(SCENARIOS[name])
            cols = _from_file(traj)
            if cols is None:
                print(f"\n=== {name}: timings/{name}.csv is missing or out of date — "
                      f"run export_timing.py first")
                continue
            source = f"timings/{name}.csv"
        else:
            traj = solve_trajectory(SCENARIOS[name], rate)
            cols, source = waypoints(traj), "solved for this check"
        report(name, traj, check(traj, cols), source)


if __name__ == "__main__":
    main()
