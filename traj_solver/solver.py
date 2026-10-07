"""
solver.py — closed-chain inverse kinematics for one pair of assembly poses.

The chain
---------
The front assembly sits at (x1, y1, θ1) and the back assembly at (x2, y2, θ2).
Four joints connect them through three links:

    P0 --l1-- P1 --l2-- P2 --l3-- P3
    q1 at P0 (relative to θ1), q2 at P1, q3 at P2, q4 at P3 (sets the back
    assembly's orientation)

Closing the chain fixes three things — P3's x and y, and the back assembly's
orientation θ2 — while there are four joints, so one degree of freedom is
left free.  This module parametrises that free DOF by q1:

  * Fix q1.  P1 is then known, and links l2 and l3 must reach from P1 to P3:
    a two-link problem solved exactly by the law of cosines.  It has two
    solutions, the elbow branches (sign = +1 gives q3 <= 0, the inward bend
    every scenario uses).
  * q4 then follows from the orientation, so every (q1, branch) candidate
    closes the chain exactly — there is no kinematic residual to minimise.

Choosing q1 is therefore a 1-D search.  `solve_ik` does it for a single pose
(grid over q1 on both branches, then a bounded 1-D refinement); the
trajectory solver (trajectory_ik.py) does it for every frame at once.

Shared building blocks used by trajectory_ik.py:
    solve_branch_batch   closed-form q for many q1 values on one branch
    posture_cost         neutral-posture tie-breaker
    prepare_pose         snap assembly poses onto their surfaces
"""

import numpy as np
from scipy.optimize import minimize_scalar

from kinematics import forward_kinematics, reachability_check
from constraints import (WheelGeometry,
                         apply_floor_constraint, apply_wall_constraint,
                         apply_ceiling_constraint, apply_outside_constraint,
                         apply_thin_edge_constraint,
                         CONSTRAINT_SETS)
from constraints_batch import evaluate_batch


# Weight of the neutral-posture term ||q_display||² (see posture_cost).  It is
# only a tie-breaker between candidates the other costs rate (nearly) equally,
# so it is kept small: at weight ~1 it would compete with the smoothness term
# and pull the joints toward neutral even when the assemblies are still.
POSTURE_WEIGHT = 1e-3

# Display convention (kinematics.q_display): q1 and q4 are measured from
# "straight up from the calf", q2 and q3 as-is.  q_display = wrap(q + shift).
_DISPLAY_SHIFT = np.array([-np.pi / 2, 0.0, 0.0, -np.pi / 2])

# Cost returned for q1 values where the two-link sub-chain cannot close, so
# the bounded 1-D refinement steers away from them.
_UNREACHABLE = 1e30


def _wrap(a):
    """Wrap angles to [-π, π)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ #
#  Closed-form configurations along the free DOF                      #
# ------------------------------------------------------------------ #

def solve_branch_batch(q1, sign, x1, y1, theta1, x2, y2, theta2, l1, l2, l3):
    """
    Joint angles for an array of q1 values on one elbow branch.

    Parameters
    ----------
    q1 : array-like (K,)
        Values of the free joint q1 (radians).
    sign : +1.0 or -1.0
        Elbow branch of the l2/l3 sub-chain; +1 gives q3 <= 0.
    x1, y1, theta1, x2, y2, theta2 : float
        Effective assembly poses (after prepare_pose).
    l1, l2, l3 : float
        Link lengths.

    Returns
    -------
    Q : ndarray (K, 4)
        [q1, q2, q3, q4] per row, wrapped to [-π, π).  Rows are NaN where the
        sub-chain cannot reach P3 from that q1.  Every finite row closes the
        chain exactly.
    """
    q1 = np.asarray(q1, dtype=float)
    phi1 = theta1 + q1                       # absolute direction of link l1
    # Vector from P1 to P3: the target for the two-link sub-chain (l2, l3).
    px = x2 - (x1 + l1 * np.cos(phi1))
    py = y2 - (y1 + l1 * np.sin(phi1))
    d2 = px ** 2 + py ** 2
    d = np.sqrt(d2)

    # Law of cosines for the elbow angle: cos(q3) = (d² - l2² - l3²) / (2 l2 l3).
    # |cos q3| > 1 means P3 is out of the sub-chain's reach for this q1.
    cos_q3 = (d2 - l2 ** 2 - l3 ** 2) / (2.0 * l2 * l3)
    reachable = np.abs(cos_q3) <= 1.0

    # Law of cosines for the angle beta between P1→P3 and link l2:
    # cos(beta) = (d² + l2² - l3²) / (2 d l2).  Clipped for rounding at full
    # extension; d = 0 only happens for unreachable rows, which are masked.
    alpha = np.arctan2(py, px)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_beta = np.clip((d2 + l2 ** 2 - l3 ** 2) / (2.0 * d * l2), -1.0, 1.0)
    beta = np.arccos(cos_beta)

    # Branch: l2 turns by +beta (sign +1) or -beta from P1→P3, and the elbow
    # bends back by the opposite sign so l3 ends on P3.
    phi2 = alpha + sign * beta
    q3 = -sign * np.arccos(np.clip(cos_q3, -1.0, 1.0))
    phi3 = phi2 + q3

    # Relative joint angles; q4 makes the back assembly's orientation theta2.
    Q = _wrap(np.stack([q1, phi2 - phi1, phi3 - phi2, theta2 - phi3], axis=-1))
    Q[~reachable] = np.nan
    return Q


def _solve_branch(q1, sign, *kin_args):
    """solve_branch_batch for one q1; returns the q-vector or None if unreachable."""
    q = solve_branch_batch(np.array([q1], dtype=float), sign, *kin_args)[0]
    return None if np.isnan(q[0]) else q


def posture_cost(Q):
    """
    Neutral-posture tie-breaker for candidate rows Q (K, 4): POSTURE_WEIGHT ×
    ||q_display||².  Display zero (q1/q4 straight up from the calf, q2/q3
    straight) is the centre of the joint-limit range used in constraints.py.
    """
    qd = _wrap(Q + _DISPLAY_SHIFT)
    return POSTURE_WEIGHT * np.sum(qd * qd, axis=1)


# ------------------------------------------------------------------ #
#  Assembly poses                                                     #
# ------------------------------------------------------------------ #

def prepare_pose(x1, y1, theta1, x2, y2, theta2, wg, constraint_set,
                 wall_x=0.0, ceiling_y=3.0):
    """
    Snap both assembly poses so their wheels touch the constraint set's
    surface(s), using the constraints.apply_*_constraint functions.  The
    *_exact sets and "none" use the given positions unchanged.

    Returns the effective (x1, y1, x2, y2); orientations are not changed.
    """
    if constraint_set == "floor":
        y1, y2 = apply_floor_constraint(x1, theta1, x2, theta2, wg)
    elif constraint_set == "wall":
        x1, y1, x2, y2 = apply_wall_constraint(x1, y1, theta1, x2, y2, theta2, wg, wall_x)
    elif constraint_set == "ceiling":
        x1, y1, x2, y2 = apply_ceiling_constraint(x1, y1, theta1, x2, y2, theta2, wg, wall_x, ceiling_y)
    elif constraint_set == "outside":
        x1, y1, x2, y2 = apply_outside_constraint(x1, y1, theta1, x2, y2, theta2, wg, wall_x, ceiling_y)
    elif constraint_set == "thin_edge":
        # thin_edge repurposes ceiling_y as edge_y and wall_x as edge_x.
        x1, y1, x2, y2 = apply_thin_edge_constraint(x1, y1, theta1, x2, y2, theta2, wg, ceiling_y, wall_x)
    return x1, y1, x2, y2


# ------------------------------------------------------------------ #
#  Single-pose solve                                                  #
# ------------------------------------------------------------------ #

def solve_ik(
    x1, y1, theta1,
    x2, y2, theta2,
    l1, l2, l3,
    wg: WheelGeometry = None,
    constraint_set: str = "none",
    wall_x: float = 0.0,
    ceiling_y: float = 3.0,
    q_init=None,
    smooth_weight: float = 0.0,
    n_grid=360,
    tol=1e-7,
):
    """
    Joint angles [q1, q2, q3, q4] connecting the two wheel assemblies at one pose.

    For whole trajectories use animate_transition.solve_trajectory, which
    chooses every frame's joint angles together (trajectory_ik.py).  This
    function is for single poses: snapshots, and showing a keyframe pose with
    the joint angles closest to a given reference (q_init).

    Cost of a candidate q (lowest wins):
        constraint penalty (constraints_batch.evaluate_batch)
      + smooth_weight · ||wrap(q - q_init)||²      (only when q_init is given)
      + posture_cost(q)

    Parameters
    ----------
    x1, y1, theta1, x2, y2, theta2 : float
        Front / back assembly poses (radians); snapped to the constraint
        set's surfaces by prepare_pose before solving.
    l1, l2, l3 : float
        Link lengths.
    wg : WheelGeometry, optional
        Wheel/wishbone geometry for the collision terms (default WheelGeometry()).
    constraint_set : str
        One of constraints.CONSTRAINT_SETS.
    wall_x, ceiling_y : float
        Environment geometry (thin_edge reuses them as edge_x, edge_y).
    q_init : array-like (4,), optional
        Reference configuration to stay close to.
    smooth_weight : float
        Weight on the distance to q_init.
    n_grid : int
        q1 grid points over [-π, π) for the global sweep; the best grid
        candidate is then refined continuously within one grid step.
    tol : float
        Kinematic residual below which the solve counts as successful.

    Returns
    -------
    q : ndarray (4,)
        Joint angles in radians.
    success : bool
        True if the kinematic residual is below tol (always, unless the
        assemblies are out of reach).
    info : dict
        residual_norm, position_error, orientation_error (rad);
        constraint_set; x1, y1, x2, y2 (effective poses after prepare_pose);
        cost (total candidate cost); penalty (constraint part); branch
        (elbow sign); extension (|P3 - P0| / (l1 + l2 + l3));
        feasible (q satisfies every hard constraint); violations ({hard
        term: raw value} q breaks); grid_feasible / grid_candidates (how many
        grid candidates satisfy every hard constraint); blocking (when none
        do: the hard terms the least-violating candidate still breaks, i.e.
        why the pose itself is infeasible).
    """
    if constraint_set not in CONSTRAINT_SETS:
        raise ValueError(f"Unknown constraint set '{constraint_set}'. "
                         f"Valid options: {CONSTRAINT_SETS}")
    if wg is None:
        wg = WheelGeometry()

    x1, y1, x2, y2 = prepare_pose(x1, y1, theta1, x2, y2, theta2, wg, constraint_set,
                                  wall_x, ceiling_y)
    if not reachability_check(x1, y1, x2, y2, l1, l2, l3):
        dist = np.hypot(x2 - x1, y2 - y1)
        raise ValueError(
            f"Target ({x2:.3f}, {y2:.3f}) is unreachable: "
            f"distance {dist:.3f} > total link length {l1+l2+l3:.3f}."
        )

    kin_args = (x1, y1, theta1, x2, y2, theta2, l1, l2, l3)
    q_ref = np.asarray(q_init, dtype=float) if q_init is not None else None

    def evaluate(Q):
        """Penalty batch and total cost for candidate rows Q."""
        Q = np.atleast_2d(Q)
        pb = evaluate_batch(Q, *kin_args, wg, constraint_set,
                            wall_x=wall_x, ceiling_y=ceiling_y)
        cost = pb.total + posture_cost(Q)
        if q_ref is not None and smooth_weight != 0.0:
            dq = _wrap(Q - q_ref)
            cost = cost + smooth_weight * np.sum(dq * dq, axis=1)
        return pb, cost

    # --- Global sweep: q1 grid × both branches, scored in one batch ---
    # Rows are ordered (q1_0, +1), (q1_0, -1), (q1_1, +1), ... so argmin breaks
    # ties toward lower q1 and the +1 branch.
    q1_grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    signs = (+1.0, -1.0)
    Q = np.stack([solve_branch_batch(q1_grid, s, *kin_args) for s in signs],
                 axis=1).reshape(-1, 4)
    valid = np.flatnonzero(~np.isnan(Q[:, 0]))
    if valid.size == 0:
        raise RuntimeError("No closed-chain configuration found on the q1 grid.")

    grid_pb, grid_cost = evaluate(Q[valid])
    i = int(np.argmin(grid_cost))
    best_q1   = q1_grid[valid[i] // 2]
    best_sign = signs[valid[i] % 2]
    best_q    = Q[valid[i]]
    best_cost = float(grid_cost[i])

    # --- Continuous refinement of q1 on the winning branch ---
    # Every q1 on the branch closes the chain exactly, so this only moves
    # along the free DOF; the grid optimum lies within one step of the true one.
    def branch_cost(q1):
        q_vec = _solve_branch(q1, best_sign, *kin_args)
        return _UNREACHABLE if q_vec is None else float(evaluate(q_vec)[1][0])

    step = 2 * np.pi / n_grid
    res = minimize_scalar(branch_cost, bounds=(best_q1 - step, best_q1 + step),
                          method="bounded", options={"xatol": 1e-8})
    if res.fun < best_cost:
        best_q    = _solve_branch(res.x, best_sign, *kin_args)
        best_cost = float(res.fun)

    # --- Diagnostics ---
    positions, theta_end = forward_kinematics(best_q, x1, y1, theta1, l1, l2, l3)
    p3 = positions[-1]
    pos_err = float(np.hypot(p3[0] - x2, p3[1] - y2))
    ang_err = float(abs(_wrap(theta_end - theta2)))
    residual_norm = float(np.sqrt(pos_err ** 2 + ang_err ** 2))

    best_pb, _ = evaluate(best_q)
    grid_ok = grid_pb.feasible
    blocking = {}
    if not grid_ok.any():
        blocking = grid_pb.violations(int(np.argmin(grid_pb.hard_violation)))

    info = {
        "residual_norm":     residual_norm,
        "position_error":    pos_err,
        "orientation_error": ang_err,
        "constraint_set":    constraint_set,
        "x1": x1, "y1": y1,
        "x2": x2, "y2": y2,
        "cost":            best_cost,
        "penalty":         float(best_pb.total[0]),
        "branch":          best_sign,
        "extension":       float(np.hypot(x2 - x1, y2 - y1) / (l1 + l2 + l3)),
        "feasible":        bool(best_pb.feasible[0]),
        "violations":      best_pb.violations(0),
        "grid_feasible":   int(grid_ok.sum()),
        "grid_candidates": int(valid.size),
        "blocking":        blocking,
    }
    return best_q, residual_norm < tol, info
