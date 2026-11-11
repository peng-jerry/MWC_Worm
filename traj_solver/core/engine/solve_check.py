"""
solve_check.py — what a playback on the robot would hit, for a solved
transition.  generate/animate.py reports it, and generate/export_timing.py
writes no timing file for a solve that fails it.
"""

import numpy as np

from .kinematics import q_display, wrap
from .wheels import WHEELS, wheel_speeds, without_drive
from ..motor_limits import MAX_JOINT_SPEED
from ..params import JOINT_HARD_LIMIT_DEG, JOINT_SOFT_LIMIT_DEG, ROLL_SPEED


def check(traj):
    """
    What a playback on the robot would hit (limits from params.py, rules from constraints.py).
    Returns (problems, warnings), each a list of report lines.

    Problems, one line per frame (or step) where they occur:
      HARD   a hard rule broken (clearance, joint limit or bend direction)
      LIMIT  joint beyond the hard limit (±110°)
      SIGN   q2 or q3 bending the wrong way (> 0)
      JUMP   joint step at or above the joint speed limit
      OMEGA  rolling wheel faster than ROLL_SPEED
      DRIVE  a driven wheel off the solid while its assembly moves (one line
             per wheel)
    Warnings, one line per joint:
      SOFT   joint beyond the soft limit (±90°): allowed, but costed, so the
             solver only goes there when the geometry needs it
    """
    problems, soft = [], {}                                  # soft: {joint: [(t, angle °)]}
    names = ["q1", "q2", "q3", "q4"]
    jump = MAX_JOINT_SPEED / traj.rate                       # rad per frame
    omega_max = ROLL_SPEED / traj.transition.wg.wheel_r * 1.01
    for k, (q, info, _) in enumerate(traj):
        t = traj.times[k]
        for rule, amount in info["violations"].items():
            unit = "°" if rule in ("joint limit", "bend") else " mm"
            value = np.degrees(amount) if unit == "°" else amount * 1000
            problems.append(f"  HARD    t={t:5.2f}s  {rule} by {value:.1f}{unit}")
        for j, v in enumerate(np.degrees(q_display(q))):
            if abs(v) > JOINT_HARD_LIMIT_DEG:
                problems.append(f"  LIMIT   t={t:5.2f}s  {names[j]}={v:+.1f}°")
            elif abs(v) > JOINT_SOFT_LIMIT_DEG:
                soft.setdefault(j, []).append((t, v))
        for j in (1, 2):
            if q[j] > 0.0:
                problems.append(f"  SIGN    t={t:5.2f}s  {names[j]}={np.degrees(q[j]):+.2f}°")
    for k in range(1, len(traj)):
        d = wrap(traj.Q[k] - traj.Q[k - 1])
        for j in range(4):
            if abs(d[j]) >= jump:
                problems.append(f"  JUMP    t={traj.times[k]:5.2f}s  {names[j]}  Δ={np.degrees(abs(d[j])):.2f}°"
                                f" ({np.degrees(abs(d[j])) * traj.rate:.1f}°/s)")
    spin, _ = wheel_speeds(traj.transition, traj.phases, traj.times)
    for k, j in zip(*np.nonzero(np.abs(spin) > omega_max)):
        problems.append(f"  OMEGA   t={traj.times[k]:5.2f}s  {WHEELS[j]}  ω={spin[k, j]:+.2f} rad/s")
    stranded = without_drive(traj.transition, traj.phases, traj.times)
    for j in np.nonzero(stranded.any(axis=0))[0]:
        t = traj.times[stranded[:, j]]
        problems.append(f"  DRIVE   {WHEELS[j]} off the solid while its assembly moves: {len(t)} frames, "
                        f"t = {t[0]:.2f} … {t[-1]:.2f} s")
    warnings = [f"  SOFT    {names[j]} past ±{JOINT_SOFT_LIMIT_DEG:g}° in {len(hits)} frames between "
                f"t = {hits[0][0]:.2f} and {hits[-1][0]:.2f} s, peak {max(hits, key=lambda h: abs(h[1]))[1]:+.1f}°"
                for j, hits in sorted(soft.items())]
    return problems, warnings
