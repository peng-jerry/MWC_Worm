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
                         apply_thin_edge_constraint,
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

def _smooth_cost(q: np.ndarray, q_ref: np.ndarray, smooth_weight: float) -> float:
    """Squared angle-wrapped distance from q to q_ref, scaled by smooth_weight."""
    if smooth_weight == 0.0:
        return 0.0
    dq = (q - q_ref + np.pi) % (2 * np.pi) - np.pi
    return smooth_weight * float(np.dot(dq, dq))


def _objective(q, x1, y1, theta1, x2, y2, theta2, l1, l2, l3,
               wg, constraint_set, wall_x, ceiling_y, q_ref, smooth_weight,
               pen_weight=1e4):
    """Kinematic residual + constraint penalty + trajectory smoothness (for BFGS polish)."""
    positions, theta_end = forward_kinematics(q, x1, y1, theta1, l1, l2, l3)
    p3 = positions[-1]
    dx = p3[0] - x2
    dy = p3[1] - y2
    dtheta = np.arctan2(np.sin(theta_end - theta2), np.cos(theta_end - theta2))
    kin = float(dx ** 2 + dy ** 2 + dtheta ** 2)
    pen = total_penalty(q, x1, y1, theta1, x2, y2, theta2, l1, l2, l3,
                        wg, constraint_set, wall_x=wall_x, ceiling_y=ceiling_y,
                        weight=pen_weight)
    return kin + pen + _smooth_cost(q, q_ref, smooth_weight)


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
    smooth_weight: float = 0.0,
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
        Warm-start guess for the joint angles.  When provided, the candidate
        scoring also penalises deviation from this reference by smooth_weight.
    smooth_weight : float
        Weight on the trajectory-smoothness term ||q - q_init||² (angle-wrapped).
        Only applied when q_init is provided.  A value of 1.0 adds a term
        comparable in scale to the neutral-posture cost ||q||².
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
    elif constraint_set == "wall_exact":
        pass  # exact positions from keyframes; skip contact adjustment
    elif constraint_set == "outside_exact":
        pass  # exact pivot positions from keyframes; skip contact adjustment
    elif constraint_set == "thin_edge_exact":
        pass  # exact pivot positions from keyframes; skip contact adjustment
    elif constraint_set == "thin_edge":
        # ceiling_y is repurposed as edge_y; wall_x as edge_x (right terminus).
        x1, y1, x2, y2 = apply_thin_edge_constraint(x1, y1, theta1, x2, y2, theta2, wg, ceiling_y, wall_x)

    if not reachability_check(x1, y1, x2, y2, l1, l2, l3):
        dist = np.hypot(x2 - x1, y2 - y1)
        raise ValueError(
            f"Target ({x2:.3f}, {y2:.3f}) is unreachable: "
            f"distance {dist:.3f} > total link length {l1+l2+l3:.3f}."
        )

    kin_args  = (x1, y1, theta1, x2, y2, theta2, l1, l2, l3)

    # Smoothness reference: only meaningful when a previous solution is available.
    q_ref      = np.asarray(q_init, dtype=float) if q_init is not None else np.zeros(4)
    eff_smooth = smooth_weight if q_init is not None else 0.0
    full_args  = (*kin_args, wg, constraint_set, wall_x, ceiling_y, q_ref, eff_smooth)

    # --- Grid sweep over q1 ---
    # For the "outside" / "thin_edge" constraints, track the lowest
    # (penalty + smooth) grid candidate separately and use it directly —
    # BFGS can drift off the constraint surface in these penalty landscapes.
    candidates = []
    best_pen_val = np.inf
    best_pen_q   = None

    for q1_val in np.linspace(-np.pi, np.pi, n_grid, endpoint=False):
        for q_vec in _solve_for_q1(q1_val, *kin_args):
            kin_cost    = float(np.dot(q_vec, q_vec))
            pen_cost    = total_penalty(q_vec, *kin_args, wg, constraint_set,
                                        wall_x=wall_x, ceiling_y=ceiling_y)
            smooth_cost = _smooth_cost(q_vec, q_ref, eff_smooth)
            total_cost  = kin_cost + pen_cost + smooth_cost
            candidates.append((total_cost, q_vec))
            if constraint_set in ("outside", "outside_exact",
                                    "thin_edge", "thin_edge_exact"):
                # Tie-break by kin_cost (neutral posture) when penalties are equal.
                # Weight 0.001 is negligible during warm frames where smooth_cost
                # dominates, but matters at cold restarts where smooth_cost=0 and
                # pen≈0 for many candidates — without it, grid order determines the
                # winner, which can land on a degenerate branch far from neutral.
                combined = pen_cost + smooth_cost + 0.001 * kin_cost
                if combined < best_pen_val:
                    best_pen_val = combined
                    best_pen_q   = q_vec.copy()

    if not candidates:
        raise RuntimeError("No feasible configuration found on the q1 grid.")

    candidates.sort(key=lambda t: t[0])
    top_seeds = [q for _, q in candidates[:min(10, len(candidates))]]

    # --- Polish ---
    best_q   = top_seeds[0].copy()
    best_val = np.inf

    if constraint_set in ("outside", "outside_exact",
                           "thin_edge", "thin_edge_exact"):
        # Grid candidates are analytically exact (zero kinematic residual).
        # Powell / BFGS can drift off the constraint surface in crossing-penalty
        # landscapes where local minima exist at non-zero kinematic residual
        # (e.g. chain routes the wrong way around the corner/terminus).
        # Use the best-penalty grid candidate directly to guarantee zero residual.
        best_q = best_pen_q if best_pen_q is not None else top_seeds[0].copy()

    else:
        # Polish with smooth_weight=0 and pen_weight=0 so BFGS minimises only
        # the kinematic residual.  With pen_weight=1e4 (default), the hard-
        # penalty gradient dominates at kin_res≈0 (where ∂kin/∂q≈0), causing
        # BFGS to drift away from analytically-exact grid seeds.
        # grid scoring still uses eff_smooth so the warm branch is preferred;
        # BFGS itself must use 0 to avoid the false-minimum at q_prev (see
        # solver notes: f(q_prev)=kin_res²≈0 dominates 200·||Δq||² for small
        # per-frame geometry changes, freezing the solver).
        wall_smooth = 0.0
        polish_args = (*kin_args, wg, constraint_set, wall_x, ceiling_y,
                       q_ref, wall_smooth, 0.0)   # pen_weight=0
        converged = []
        for q0 in top_seeds:
            result = minimize(
                _objective, q0, args=polish_args, method="BFGS",
                options={"gtol": 1e-14, "maxiter": 5_000},
            )
            q_cand = (result.x + np.pi) % (2 * np.pi) - np.pi
            # Measure kinematic residual directly (result.fun == kin_res² here).
            pos_c, th_c = forward_kinematics(q_cand, x1, y1, theta1, l1, l2, l3)
            pe = float(np.hypot(pos_c[-1][0] - x2, pos_c[-1][1] - y2))
            ae = float(abs(np.arctan2(np.sin(th_c - theta2), np.cos(th_c - theta2))))
            if float(np.sqrt(pe**2 + ae**2)) < tol:
                sel = result.fun + _smooth_cost(q_cand, q_ref, eff_smooth)
                converged.append((sel, q_cand))

        if converged:
            converged.sort(key=lambda t: t[0])
            best_q = converged[0][1]
        # else: keep top_seeds[0] — the analytically-exact grid seed

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
