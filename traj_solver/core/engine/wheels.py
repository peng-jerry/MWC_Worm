"""
wheels.py — which wheels touch the solid, and how fast each must turn, from a
transition's motion plan.

  wheel_speeds      each wheel's spin in the world and relative to its assembly
  wheels_touching   which wheels touch the solid at each time
  without_drive     where a driven wheel is off the solid while its assembly moves
"""

import numpy as np

from .constraints import wheel_centers
from .kinematics import wrap
from .planner import phase_starts, plan_pose
from ..motor_calibration import WHEELS as WHEEL_MOTORS
from ..params import FEASIBILITY_TOL

WHEELS = ("front arm A", "front arm B", "back arm A", "back arm B")
NORMAL_SPEED = 1e-6     # m/s; a wheel moving toward or away from a contact faster than this is not rolling on it
MOVING_SPEED = 1e-3     # m/s and rad/s; an assembly moving or turning faster than this needs its drive


def wheel_speeds(trn, phases, times, h=1e-4, tol=FEASIBILITY_TOL):
    """
    How fast each wheel turns at each time, from the motion plan (rad/s,
    counter-clockwise positive in the side view).  Returns (rolling, motor),
    each (N, 4) in the order of WHEELS:

      rolling  the wheel's spin in the world
      motor    its spin relative to its own assembly — rolling minus the
               assembly's turn rate — which is what the wheel's motor must
               turn at.  The two differ while an assembly pivots.

    Velocities are taken from the plan at each instant (central differences
    over ±h seconds, which is continuous between frames), not from the steps
    between frames.

    A wheel touches the solid where its centre is one wheel radius (within
    tol) from a face or from a corner.  For a contact whose direction from the
    touching point to the wheel centre is n, rolling without slipping gives

        ω = (ẑ × n) · v / r        (v the wheel centre's velocity)

    on a face and over a corner alike: over a corner the wheel turns with the
    assembly about the touching point.  Only a contact the wheel moves along
    counts: one it moves toward or away from (|v · n| > NORMAL_SPEED) it is
    arriving at or leaving, within tol of it but not rolling on it.  Then:

      one contact      ω from it
      two contacts     wedged into an inside corner: the mean, which is 0
                       while the wheel stays wedged (it cannot roll along
                       both faces at once)
      none             in the air: the motor speed is interpolated in time
                       between the nearest contacts before and after, so the
                       wheel is already turning at the right speed when it
                       lands
    """
    rolling, turn = _contact_rolling(trn, phases, times, h, tol)
    motor = rolling - turn
    for j in range(4):
        air = np.isnan(motor[:, j])
        if air.all():
            motor[:, j] = 0.0
        elif air.any():
            motor[air, j] = np.interp(times[air], times[~air], motor[~air, j])
    return motor + turn, motor


def wheels_touching(trn, phases, times, h=1e-4, tol=FEASIBILITY_TOL):
    """
    Which wheels touch the solid at each time, (N, 4) bool in the order of
    WHEELS: where wheel_speeds takes a wheel's speed from a contact, rather
    than ramping it through the air.
    """
    return ~np.isnan(_contact_rolling(trn, phases, times, h, tol)[0])


def without_drive(trn, phases, times, h=1e-4, tol=FEASIBILITY_TOL):
    """
    Where a driven wheel (one with a motor, motor_calibration.WHEELS) is off
    the solid while its assembly moves or turns, (N, 4) bool in the order of
    WHEELS.  Only a driven wheel can push its assembly along, so there the
    assembly has no drive of its own.  Free wheels are never flagged.
    """
    rolling, turn = _contact_rolling(trn, phases, times, h, tol)
    end = phase_starts(phases)[-1]
    speed = np.empty((len(times), 2))                 # front, back chain joint (m/s)
    for k, t in enumerate(times):
        ta, tb = max(t - h, 0.0), min(t + h, end)
        a, b = plan_pose(trn, phases, ta), plan_pose(trn, phases, tb)
        speed[k] = [np.hypot(b["x1"] - a["x1"], b["y1"] - a["y1"]),
                    np.hypot(b["x2"] - a["x2"], b["y2"] - a["y2"])]
        speed[k] /= tb - ta
    moving = (np.repeat(speed, 2, axis=1) > MOVING_SPEED) | (np.abs(turn) > MOVING_SPEED)
    driven = np.array([m is not None for m in WHEEL_MOTORS.values()])
    return np.isnan(rolling) & moving & driven


def _contact_rolling(trn, phases, times, h, tol):
    """
    Each wheel's rolling speed from its contacts (wheel_speeds), NaN where it
    touches nothing, and each wheel's assembly turn rate; both (N, 4) rad/s.
    """
    r = trn.wg.wheel_r
    segments = trn.env.face_segments()
    end = phase_starts(phases)[-1]

    def centres(pose):
        return np.array([*wheel_centers(pose["x1"], pose["y1"], pose["theta1"], trn.wg, side=1),
                         *wheel_centers(pose["x2"], pose["y2"], pose["theta2"], trn.wg, side=-1)])

    def headings(pose):
        return np.array([pose["theta1"], pose["theta1"], pose["theta2"], pose["theta2"]])

    rolling = np.full((len(times), 4), np.nan)        # NaN until a contact sets it
    turn = np.empty((len(times), 4))
    for k, t in enumerate(times):
        ta, tb = max(t - h, 0.0), min(t + h, end)
        before, now, after = (plan_pose(trn, phases, s) for s in (ta, t, tb))
        c = centres(now)
        v = (centres(after) - centres(before)) / (tb - ta)
        turn[k] = wrap(headings(after) - headings(before)) / (tb - ta)
        for j in range(4):
            rates, points = [], []
            for a, b in segments:
                ab = b - a
                p = a + np.clip((c[j] - a) @ ab / (ab @ ab), 0.0, 1.0) * ab   # nearest point
                d = float(np.linalg.norm(c[j] - p))
                if abs(d - r) > tol or any(np.allclose(p, q, atol=1e-9) for q in points):
                    continue        # not touching, or a corner already counted from the other face
                points.append(p)
                n = (c[j] - p) / d
                if abs(v[j] @ n) <= NORMAL_SPEED:
                    rates.append(np.array([-n[1], n[0]]) @ v[j] / r)
            if rates:
                rolling[k, j] = np.mean(rates)
    return rolling, turn
