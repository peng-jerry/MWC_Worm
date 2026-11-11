"""
follow_gap.py — the gap the partner of a pivoting assembly aims to keep at an
outside corner or edge: the one that keeps the joints farthest from their
limits.  Inside corners do not use it: there the partner parks.

  follow_gap         searches the gaps, planning and scoring each one
  peak_joint_angle   the score: the largest joint angle a plan forces
"""

import math

import numpy as np

from .chain_ik import solve_branch_batch
from .constraints import evaluate
from .kinematics import q_display
from .planner import plan_phases, sample_poses
from .transition import PlanError
from ..params import CRUISE_GAP, FOLLOW_GAP_STEP, FOLLOW_GAP_TOLERANCE, SAMPLE_RATE
from ..robot import l1, l2, l3


def peak_joint_angle(trn, phases, rate=SAMPLE_RATE):
    """
    The largest joint angle a planned transition forces on the chain (display
    convention, radians), or inf if some pose has no valid configuration.

    The phases are sampled at `rate` frames per second — by default the
    trajectory solver's rate, so exactly the poses it will solve.  At each
    pose every closed-chain configuration on a 1° q1 grid (on the elbow
    branch the solver searches) that meets all the hard rules is a candidate, and the pose
    forces at least the smallest largest-joint-angle among them.  Each pose
    is judged on its own, so this is a bound the trajectory solver can only
    match or exceed — but it is quick enough to compare many plans.
    """
    try:
        _, poses = sample_poses(trn, phases, rate)
    except ValueError:                       # out of reach, or an assembly collides
        return math.inf
    env = trn.env
    q1 = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    peak = 0.0
    for p in poses:
        args = (p["x1"], p["y1"], p["theta1"], p["x2"], p["y2"], p["theta2"])
        Q = solve_branch_batch(q1, +1.0, *args, l1, l2, l3)
        Q = Q[~np.isnan(Q[:, 0])]
        if len(Q) == 0:
            return math.inf
        feasible = evaluate(Q, p, env, trn.wg, l1, l2, l3).feasible
        if not feasible.any():
            return math.inf
        peak = max(peak, float(np.abs(q_display(Q[feasible])).max(axis=1).min()))
    return peak


def follow_gap(trn, rate=SAMPLE_RATE):
    """
    The gap (m) the partner aims to keep while following a pivot at an
    outside corner or edge: the one that keeps the joints farthest from
    their limits.

    Gaps are planned and scored by peak_joint_angle on a grid
    FOLLOW_GAP_STEP apart, up to the cruise gap.  The robot will not hold
    the planned gap exactly, so each gap is judged by its worst score within
    ±FOLLOW_GAP_TOLERANCE.  That keeps the choice away from the edge of the
    range of gaps that work, where the best score often lies.

    To save time the grid is searched coarse to fine: first every
    2 × FOLLOW_GAP_TOLERANCE — a gap can only be chosen if the whole
    ±tolerance range around it works, so no such range is missed — then
    every FOLLOW_GAP_STEP around the best of those.
    """
    n = int(CRUISE_GAP / FOLLOW_GAP_STEP + 1e-9)        # grid: gap i = i × FOLLOW_GAP_STEP
    k = int(round(FOLLOW_GAP_TOLERANCE / FOLLOW_GAP_STEP))     # tolerance, in grid steps
    scores = {}

    def score(i):
        if i not in scores:
            try:
                scores[i] = peak_joint_angle(trn, plan_phases(trn, gap=i * FOLLOW_GAP_STEP), rate)
            except ValueError:               # no place on the surface at that gap
                scores[i] = math.inf
        return scores[i]

    def worst(i):
        return max(score(j) for j in range(max(1, i - k), min(n, i + k) + 1))

    # Coarse gaps that work, best first; refine around each until one holds
    # up within the tolerance.
    coarse = sorted((i for i in range(1, n + 1, 2 * k) if math.isfinite(score(i))), key=score)
    for c in coarse:
        best = min(range(max(1, c - 2 * k), min(n, c + 2 * k) + 1), key=worst)
        if math.isfinite(worst(best)):
            return best * FOLLOW_GAP_STEP
    raise PlanError(f"{trn.name}: no follow gap up to the cruise gap gives every pose a valid "
                     f"configuration within ±{FOLLOW_GAP_TOLERANCE * 1000:.0f} mm")
