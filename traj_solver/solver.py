"""
Inverse kinematics solver: find q1-q4 given front/back assembly poses.

Strategy
--------
The problem has 3 hard constraints (x2, y2, theta2) and 4 unknowns (q1-q4),
leaving 1 DOF.  We exploit the structure:

  - For any fixed q1, the remaining chain (l2, l3) is a standard 2-link IK
    problem that can be solved analytically (law of cosines).  That gives
    exact (q2, q3), and q4 follows from the orientation constraint.
  - We sweep q1 over a dense grid, keep all feasible solutions, then pick
    the one that minimises the sum of squared joint angles (neutral posture).
  - A final BFGS polish is applied to the best candidate.
"""

import numpy as np
from scipy.optimize import minimize

from kinematics import forward_kinematics, reachability_check
from constraints import (WheelGeometry, total_penalty,
                         apply_floor_constraint, apply_wall_constraint,
                         apply_ceiling_constraint, apply_outside_constraint,
                         CONSTRAINT_SETS)


# ------------------------------------------------------------------ #
#  Two-link analytical IK                                             #
# ------------------------------------------------------------------ #

def _two_link_ik(px, py, l2, l3):
    """
    Solve the 2-link planar IK: place the tip of (l2, l3) at (px, py).

    Returns list of (phi2, phi3) absolute-angle pairs (0, 1, or 2 solutions).
    phi2 is the absolute angle of link l2 from the X-axis; phi3 for l3.
    """
    d2 = px ** 2 + py ** 2
    d = np.sqrt(d2)

    # Law of cosines: cos(q3) = (d^2 - l2^2 - l3^2) / (2*l2*l3)
    cos_q3 = (d2 - l2 ** 2 - l3 ** 2) / (2.0 * l2 * l3)
    if abs(cos_q3) > 1.0:
        return []  # out of reach

    alpha = np.arctan2(py, px)  # direction from base to target

    # Law of cosines: cos(beta) = (d^2 + l2^2 - l3^2) / (2*d*l2)
    cos_beta = (d2 + l2 ** 2 - l3 ** 2) / (2.0 * d * l2)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)  # numerical safety
    beta = np.arccos(cos_beta)

    solutions = []
    for sign in (+1.0, -1.0):           # elbow-up and elbow-down
        phi2 = alpha + sign * beta
        q3 = -sign * np.arccos(cos_q3)  # opposite sign: phi2=alpha+beta → q3 bends back
        phi3 = phi2 + q3
        solutions.append((phi2, phi3))
    return solutions


# ------------------------------------------------------------------ #
#  Full chain solver                                                  #
# ------------------------------------------------------------------ #

def _objective(q, x1, y1, theta1, x2, y2, theta2, l1, l2, l3, wg, constraint_set, wall_x, ceiling_y):
    """Kinematic residual + constraint penalty (for BFGS polish)."""
    positions, theta_end = forward_kinematics(q, x1, y1, theta1, l1, l2, l3)
    p3 = positions[-1]
    dx = p3[0] - x2
    dy = p3[1] - y2
    dtheta = np.arctan2(np.sin(theta_end - theta2), np.cos(theta_end - theta2))
    kin = float(dx ** 2 + dy ** 2 + dtheta ** 2)
    pen = total_penalty(q, x1, y1, theta1, x2, y2, theta2, l1, l2, l3,
                        wg, constraint_set, wall_x=wall_x, ceiling_y=ceiling_y)
    return kin + pen


def _solve_for_q1(q1, x1, y1, theta1, x2, y2, theta2, l1, l2, l3):
    """
    Given q1, analytically solve for q2, q3, q4.
    Returns list of q-vectors (possibly empty, or 1-2 solutions).
    """
    phi1 = theta1 + q1
    p1x = x1 + l1 * np.cos(phi1)
    p1y = y1 + l1 * np.sin(phi1)

    # Remaining target for the 2-link chain rooted at p1
    dx = x2 - p1x
    dy = y2 - p1y

    results = []
    for phi2, phi3 in _two_link_ik(dx, dy, l2, l3):
        q2 = phi2 - phi1
        q3 = phi3 - phi2
        q4 = theta2 - phi3
        # Wrap to [-pi, pi]
        q_vec = np.array([q1, q2, q3, q4])
        q_vec = (q_vec + np.pi) % (2 * np.pi) - np.pi
        results.append(q_vec)
    return results


def solve_ik(
    x1, y1, theta1,
    x2, y2, theta2,
    l1, l2, l3,
    wg: WheelGeometry = None,
    constraint_set: str = "none",
    wall_x: float = 0.0,
    ceiling_y: float = 3.0,
    q_init=None,
    n_grid=360,
    tol=1e-7,
):
    """
    Find joint angles [q1, q2, q3, q4] connecting the two wheel assemblies.

    Parameters
    ----------
    x1, y1, theta1 : float
        Front assembly position and orientation (radians).
    x2, y2, theta2 : float
        Back assembly position and orientation (radians).
    l1, l2, l3 : float
        Link lengths (must be positive).
    wg : WheelGeometry, optional
        Wheel/wishbone geometry needed for collision checking.
        Defaults to WheelGeometry() if not provided.
    constraint_set : {"none", "floor"}
        Active constraint set.  "none" runs a purely kinematic IK.
        "floor" auto-adjusts y1/y2 for wheel-ground contact and penalises
        the linkage for going below the floor.
    q_init : array-like of shape (4,), optional
        Warm-start guess for the joint angles.
    n_grid : int
        Number of q1 values sampled in [-pi, pi] for the global grid sweep.
    tol : float
        Kinematic residual norm below which the solution is declared successful.

    Returns
    -------
    q : ndarray of shape (4,)
        Best joint angles [q1, q2, q3, q4] in radians.
    success : bool
        True if the kinematic residual norm is below `tol`.
    info : dict
        'residual_norm', 'position_error', 'orientation_error' (rad),
        'constraint_set', 'y1', 'y2' (effective assembly heights used).
    """
    if constraint_set not in CONSTRAINT_SETS:
        raise ValueError(f"Unknown constraint set '{constraint_set}'. "
                         f"Valid options: {CONSTRAINT_SETS}")

    if wg is None:
        wg = WheelGeometry()

    # --- Pre-processing: adjust poses so wheels contact the right surface(s) ---
    if constraint_set == "floor":
        y1, y2 = apply_floor_constraint(x1, theta1, x2, theta2, wg)
    elif constraint_set == "wall":
        x1, y1, x2, y2 = apply_wall_constraint(x1, y1, theta1, x2, y2, theta2, wg, wall_x)
    elif constraint_set == "ceiling":
        x1, y1, x2, y2 = apply_ceiling_constraint(x1, y1, theta1, x2, y2, theta2, wg, wall_x, ceiling_y)
    elif constraint_set == "outside":
        x1, y1, x2, y2 = apply_outside_constraint(x1, y1, theta1, x2, y2, theta2, wg, wall_x, ceiling_y)

    if not reachability_check(x1, y1, x2, y2, l1, l2, l3):
        dist = np.hypot(x2 - x1, y2 - y1)
        raise ValueError(
            f"Target ({x2:.3f}, {y2:.3f}) is unreachable: "
            f"distance {dist:.3f} > total link length {l1+l2+l3:.3f}."
        )

    kin_args  = (x1, y1, theta1, x2, y2, theta2, l1, l2, l3)
    full_args = (*kin_args, wg, constraint_set, wall_x, ceiling_y)

    # --- Grid sweep over q1 ---
    candidates = []
    for q1_val in np.linspace(-np.pi, np.pi, n_grid, endpoint=False):
        for q_vec in _solve_for_q1(q1_val, *kin_args):
            kin_cost = float(np.dot(q_vec, q_vec))
            pen_cost = total_penalty(q_vec, *kin_args, wg, constraint_set,
                                     wall_x=wall_x, ceiling_y=ceiling_y)
            candidates.append((kin_cost + pen_cost, q_vec))

    if q_init is not None:
        q0 = np.asarray(q_init, dtype=float)
        pen = total_penalty(q0, *kin_args, wg, constraint_set,
                            wall_x=wall_x, ceiling_y=ceiling_y)
        candidates.append((float(np.dot(q0, q0)) + pen, q0))

    if not candidates:
        raise RuntimeError("No feasible configuration found on the q1 grid.")

    candidates.sort(key=lambda t: t[0])
    top_seeds = [q for _, q in candidates[:min(10, len(candidates))]]

    # --- Polish ---
    best_q   = top_seeds[0].copy()
    best_val = np.inf

    if constraint_set == "outside":
        # The segment-crossing penalty has a zero-gradient minimum at the corner
        # where finite-difference BFGS loses precision.  Powell (derivative-free)
        # handles this landscape cleanly and converges to machine-epsilon IK.
        for q0 in top_seeds[:3]:
            result = minimize(
                _objective, q0, args=full_args, method="Powell",
                options={"xtol": 1e-12, "ftol": 1e-12, "maxiter": 5_000},
            )
            if result.fun < best_val:
                best_val = result.fun
                best_q   = result.x
    else:
        for q0 in top_seeds:
            result = minimize(
                _objective, q0, args=full_args, method="BFGS",
                options={"gtol": 1e-14, "maxiter": 5_000},
            )
            if result.fun < best_val:
                best_val = result.fun
                best_q   = result.x

    best_q = (best_q + np.pi) % (2 * np.pi) - np.pi

    # --- Diagnostics ---
    positions, theta_end = forward_kinematics(best_q, x1, y1, theta1, l1, l2, l3)
    p3 = positions[-1]
    pos_err = float(np.hypot(p3[0] - x2, p3[1] - y2))
    ang_err = float(abs(np.arctan2(
        np.sin(theta_end - theta2), np.cos(theta_end - theta2)
    )))
    residual_norm = float(np.sqrt(pos_err ** 2 + ang_err ** 2))

    info = {
        "residual_norm":     residual_norm,
        "position_error":    pos_err,
        "orientation_error": ang_err,
        "constraint_set":    constraint_set,
        "x1": x1, "y1": y1,
        "x2": x2, "y2": y2,
    }
    return best_q, residual_norm < tol, info
