"""
trajectory_ik.py — joint angles for a whole trajectory, chosen together by
dynamic programming.

Why a whole-trajectory solve
----------------------------
At every frame the chain has one free degree of freedom (see solver.py): for
any q1 on an elbow branch, solver.solve_branch_batch gives the exact q2..q4
that close the chain.  Solving frame by frame means choosing each frame's q1
without knowing what later frames will need, so the robot can be cornered
into a joint step faster than the motors allow.  Choosing all frames together
avoids that: the solver can start moving early to stay within the limit.

The problem
-----------
Sample each frame's free DOF as a set of candidate configurations.  Pick one
candidate per frame to minimise

    Σ_k node(q_k)  +  Σ_k edge(q_{k-1}, q_k)

  node(q)      constraint penalty (constraints_batch.evaluate_batch)
               + posture tie-breaker (solver.posture_cost)
               + INFEASIBLE_COST if q violates any hard constraint
  edge(qa, qb) smooth_weight · ||wrap(qb - qa)||²
               + STEP_VIOLATION_COST + STEP_EXCESS_WEIGHT · excess
                 if any joint moves more than max_step

The cost of a path is a sum of terms that each depend on one frame or on two
consecutive frames, so the exact minimum over all candidate paths is found by
the Viterbi recursion:

    J_0(s) = node_0(s)
    J_k(s) = node_k(s) + min over s' of [ J_{k-1}(s') + edge(s', s) ]

keeping the minimising s' for each s and tracing back from the best final
state.  Violations are made very expensive rather than impossible, so a path
always exists: if the keyframes demand something infeasible, the result is
the least-violating path and the per-frame diagnostics say where.

Passes
------
A single fine grid over all of q1 would be too slow, so the search narrows:

  coarse   q1 every 1° (COARSE_GRID) on both elbow branches.  Only q1 changes
           up to BAND_FACTOR × max_step between frames are considered (a
           larger change would break the step limit anyway).
  wide     WIDE_PASS[1] values of q1 within ±WIDE_PASS[0] coarse cells of the
           current path, on its branch, all transitions between consecutive
           frames.  Repeated, centred on the new path each time, until no
           frame's q1 moves more than WIDE_CONVERGED cells (at most
           WIDE_REPEATS times).  The current path is always a candidate, so
           the total cost can only fall.  Repeating lets the path move several
           cells away from the coarse solution where the coarse grid was too
           crude to resolve the step limit.
  sharp    SHARP_PASS: the same, narrow and dense, for the final resolution
           (0.01 coarse cell).

Every pass uses the exact step limit.  Allowing the coarse pass slack for its
discretisation hides real violations that the narrower passes cannot remove.

Pinned frames (q_start / q_end) have a single candidate: the given posture.
"""

import numpy as np

from constraints_batch import evaluate_batch
from kinematics import forward_kinematics
from solver import solve_branch_batch, posture_cost

# Penalties that make violations dominate every other cost term.
INFEASIBLE_COST     = 1e6   # per frame whose configuration breaks a hard constraint
STEP_VIOLATION_COST = 1e6   # per transition in which any joint exceeds max_step
STEP_EXCESS_WEIGHT  = 1e8   # per radian beyond max_step, to rank unavoidable violations
PIN_WEIGHT          = 1e6   # coarse pass: draw a pinned frame to its grid point nearest the pin

# Search resolution.
COARSE_GRID    = 360          # coarse q1 grid points over 2π (1° cells)
BAND_FACTOR    = 3.0          # coarse: largest q1 change per frame considered, × max_step
WIDE_PASS      = (3.0, 61)    # (half-width in cells, points): ±3 cells at 0.1-cell spacing
WIDE_REPEATS   = 8            # most wide passes
WIDE_CONVERGED = 0.01         # wide passes stop once no frame's q1 moves more (cells)
SHARP_PASS     = (0.15, 31)   # ±0.15 cell at 0.01-cell spacing

_SIGNS = (+1.0, -1.0)         # elbow branches; +1 gives q3 <= 0


def _wrap(a):
    """Wrap angles to [-π, π)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _pose_args(pose):
    """(x1, y1, theta1, x2, y2, theta2) from a pose dict."""
    return (pose["x1"], pose["y1"], pose["theta1"],
            pose["x2"], pose["y2"], pose["theta2"])


# ------------------------------------------------------------------ #
#  Costs                                                              #
# ------------------------------------------------------------------ #

def _node_costs(Q, pose, wg, l1, l2, l3):
    """
    Node cost of each candidate row of Q (K, 4) at one frame's pose.

    Rows that are NaN (the chain cannot close for that q1) cost inf.
    Returns (cost (K,), PenaltyBatch of the finite rows or None, finite-row mask).
    """
    cost = np.full(len(Q), np.inf)
    ok = ~np.isnan(Q[:, 0])
    if not ok.any():
        return cost, None, ok
    pb = evaluate_batch(Q[ok], *_pose_args(pose), l1, l2, l3, wg, pose["constraint_set"],
                        wall_x=pose["wall_x"], ceiling_y=pose["ceiling_y"])
    cost[ok] = pb.total + posture_cost(Q[ok]) + INFEASIBLE_COST * ~pb.feasible
    return cost, pb, ok


def _edge_costs(Qb, Qa, smooth_weight, max_step):
    """
    Cost of moving from configurations Qa to Qb (any broadcastable shapes
    ending in 4).  NaN where either side is NaN; callers treat NaN as inf.
    """
    dq = _wrap(Qb - Qa)
    excess = np.sum(np.maximum(np.abs(dq) - max_step, 0.0), axis=-1)
    return (smooth_weight * np.sum(dq * dq, axis=-1)
            + STEP_VIOLATION_COST * (excess > 1e-12) + STEP_EXCESS_WEIGHT * excess)


def _pinned_candidate(q_pin, pose, l1, l2, l3, k):
    """
    The exact configuration for a pinned frame, as (q1, branch sign, q).

    Re-solves the chain at the frame's pose from q_pin's q1 on both branches
    and keeps the branch that reproduces q_pin.  Raises ValueError if q_pin
    does not close the chain at that pose (to within 1e-6 rad).
    """
    q_pin = np.asarray(q_pin, dtype=float)
    best = None
    for s in _SIGNS:
        q = solve_branch_batch(np.array([q_pin[0]]), s, *_pose_args(pose), l1, l2, l3)[0]
        if np.isnan(q[0]):
            continue
        err = float(np.abs(_wrap(q - q_pin)).max())
        if best is None or err < best[0]:
            best = (err, s, q)
    if best is None or best[0] > 1e-6:
        off = "unreachable" if best is None else f"off by {np.degrees(best[0]):.3f}°"
        raise ValueError(f"Pinned posture for frame {k} does not close the chain at "
                         f"that frame's pose ({off}).")
    return q_pin[0], best[1], best[2]


# ------------------------------------------------------------------ #
#  Passes                                                             #
# ------------------------------------------------------------------ #

def _coarse_pass(poses, wg, l1, l2, l3, n_grid, smooth_weight, max_step, pins):
    """
    Viterbi over the coarse q1 grid on both branches.

    States per frame are (branch b, grid index m).  A predecessor (b', m') is
    considered only if |m - m'| (cyclically) is within the band, so each state
    has 2 × (2·band + 1) predecessors; all of them are scored at once.

    pins: {frame: exact pinned q}; those frames are drawn to the nearest grid
    point by PIN_WEIGHT (the wide/sharp passes then use the exact pin).

    Returns (q1 per frame, branch sign per frame, grid cell size (rad),
    per-frame grid diagnostics).
    """
    N, M = len(poses), n_grid
    cell = 2 * np.pi / M
    q1_grid = np.linspace(-np.pi, np.pi, M, endpoint=False)

    # Candidates and node costs for every frame.
    Q = np.empty((N, 2, M, 4))
    node = np.empty((N, 2, M))
    diag = []
    for k, pose in enumerate(poses):
        for b, s in enumerate(_SIGNS):
            Q[k, b] = solve_branch_batch(q1_grid, s, *_pose_args(pose), l1, l2, l3)
        cost, pb, ok = _node_costs(Q[k].reshape(-1, 4), pose, wg, l1, l2, l3)
        if pb is None:
            raise RuntimeError(f"Frame {k}: no closed-chain configuration on the q1 grid "
                               f"(assemblies out of reach).")
        feasible = pb.feasible
        diag.append({
            "grid_candidates": int(ok.sum()),
            "grid_feasible":   int(feasible.sum()),
            # When no candidate is feasible the pose itself is infeasible;
            # report what still breaks for the least-violating candidate.
            "blocking":        {} if feasible.any()
                               else pb.violations(int(np.argmin(pb.hard_violation))),
        })
        if k in pins:
            dist = np.sum(_wrap(Q[k].reshape(-1, 4) - pins[k]) ** 2, axis=1)
            cost = cost + PIN_WEIGHT * np.where(ok, dist, 0.0)
        node[k] = cost.reshape(2, M)

    # MP[o, m] is the predecessor grid index when q1 changes by offsets[o]
    # cells arriving at index m (the grid wraps around at ±π).
    band = max(1, int(np.ceil(BAND_FACTOR * max_step / cell)))
    idx = np.arange(M)
    offsets = np.arange(-band, band + 1)
    MP = (idx[None, :] - offsets[:, None]) % M

    J = node[0].copy()                              # best path cost ending at each state
    back = np.zeros((N, 2, M), dtype=np.int64)      # its predecessor, as b' * M + m'
    for k in range(1, N):
        best = np.full((2, M), np.inf)
        arg = np.zeros((2, M), dtype=np.int64)
        for bp in range(2):
            Jp = J[bp][MP]                          # (offsets, M)
            if not np.isfinite(Jp).any():
                continue
            Qp = Q[k - 1, bp][MP]                   # (offsets, M, 4)
            for b in range(2):
                cand = Jp + _edge_costs(Q[k, b][None], Qp, smooth_weight, max_step)
                cand = np.where(np.isnan(cand), np.inf, cand)
                o = np.argmin(cand, axis=0)
                c = cand[o, idx]
                better = c < best[b]
                best[b, better] = c[better]
                arg[b, better] = bp * M + MP[o, idx][better]
        J = best + node[k]
        back[k] = arg

    if not np.isfinite(J).any():
        raise RuntimeError("No connected path through the q1 grid: consecutive frames have "
                           "no configurations within the coarse search band.")
    b, m = divmod(int(np.argmin(J)), M)
    path = [(b, m)]
    for k in range(N - 1, 0, -1):
        b, m = divmod(int(back[k, b, m]), M)
        path.append((b, m))
    path.reverse()
    q1 = np.array([q1_grid[m] for _, m in path])
    signs = np.array([_SIGNS[b] for b, _ in path])
    return q1, signs, cell, diag


def _local_pass(poses, wg, l1, l2, l3, q1_path, signs, half_width, points,
                smooth_weight, max_step, pinned):
    """
    Viterbi over `points` values of q1 per frame within ±half_width (rad) of
    q1_path, on each frame's branch, with all transitions between consecutive
    frames.  The centre value (offset 0) is always a candidate, so the result
    costs no more than q1_path.

    pinned: {frame: (q1, sign, q)}; those frames have just that candidate.
    Returns (q1 per frame, Q (N, 4)).
    """
    offsets = np.linspace(-half_width, half_width, points)
    layers = []                                     # (q1 values, Q, node cost) per frame
    for k, pose in enumerate(poses):
        if k in pinned:
            q1k, _, qk = pinned[k]
            q1s, Qk = np.array([q1k]), qk[None, :]
        else:
            q1s = q1_path[k] + offsets
            Qk = solve_branch_batch(q1s, signs[k], *_pose_args(pose), l1, l2, l3)
        cost, _, _ = _node_costs(Qk, pose, wg, l1, l2, l3)
        layers.append((q1s, Qk, cost))

    J = layers[0][2]
    backs = []
    for k in range(1, len(layers)):
        Qa, Qb, cost = layers[k - 1][1], layers[k][1], layers[k][2]
        # tot[i, j]: best cost to frame k-1 candidate i, then on to frame k candidate j
        tot = J[:, None] + _edge_costs(Qb[None, :, :], Qa[:, None, :], smooth_weight, max_step)
        tot = np.where(np.isnan(tot), np.inf, tot)
        a = np.argmin(tot, axis=0)
        J = tot[a, np.arange(len(Qb))] + cost
        backs.append(a)

    i = int(np.argmin(J))
    picks = [i]
    for a in reversed(backs):
        i = int(a[i])
        picks.append(i)
    picks.reverse()
    q1 = np.array([layers[k][0][i] for k, i in enumerate(picks)])
    Q = np.array([layers[k][1][i] for k, i in enumerate(picks)])
    return q1, Q


def _frame_info(Q, poses, signs, diag, wg, l1, l2, l3, max_step):
    """Per-frame diagnostics for the final path (same keys as solver.solve_ik's info)."""
    infos = []
    for k, (q, pose) in enumerate(zip(Q, poses)):
        pb = evaluate_batch(q[None, :], *_pose_args(pose), l1, l2, l3, wg,
                            pose["constraint_set"], wall_x=pose["wall_x"],
                            ceiling_y=pose["ceiling_y"])
        positions, theta_end = forward_kinematics(q, pose["x1"], pose["y1"], pose["theta1"],
                                                  l1, l2, l3)
        pos_err = float(np.hypot(positions[-1][0] - pose["x2"], positions[-1][1] - pose["y2"]))
        ang_err = float(abs(_wrap(theta_end - pose["theta2"])))
        step_excess = 0.0
        if k > 0:
            step_excess = float(max(0.0, np.abs(_wrap(q - Q[k - 1])).max() - max_step))
        infos.append({
            "residual_norm":     float(np.hypot(pos_err, ang_err)),
            "position_error":    pos_err,
            "orientation_error": ang_err,
            "constraint_set":    pose["constraint_set"],
            "x1": pose["x1"], "y1": pose["y1"],
            "x2": pose["x2"], "y2": pose["y2"],
            "penalty":           float(pb.total[0]),
            "branch":            float(signs[k]),
            "extension":         float(np.hypot(pose["x2"] - pose["x1"], pose["y2"] - pose["y1"])
                                       / (l1 + l2 + l3)),
            "feasible":          bool(pb.feasible[0]),
            "violations":        pb.violations(0),
            "step_excess":       step_excess,
            **diag[k],
        })
    return infos


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #

def solve_trajectory_dp(poses, wg, l1, l2, l3, *, max_step, smooth_weight,
                        n_grid=None, q_start=None, q_end=None):
    """
    Joint angles for every frame, optimised over the whole trajectory.

    Parameters
    ----------
    poses : list of dict
        Assembly poses per frame (scenarios.sample_poses):
        x1, y1, theta1, x2, y2, theta2, constraint_set, wall_x, ceiling_y.
    wg : WheelGeometry
        Wheel/wishbone geometry for the collision terms.
    l1, l2, l3 : float
        Link lengths.
    max_step : float
        Per-joint step limit between consecutive frames (rad).
    smooth_weight : float
        Weight on ||Δq||² between consecutive frames.
    n_grid : int, optional
        Coarse q1 grid points over 2π (default COARSE_GRID).
    q_start, q_end : array-like (4,), optional
        Fix the first / last frame to this posture.  It must close the chain
        at that frame's pose.

    Returns
    -------
    Q : ndarray (N, 4)
        Joint angles per frame (radians, wrapped).
    infos : list of dict
        Per-frame diagnostics: the keys of solver.solve_ik's info (except
        cost) plus step_excess, the largest joint step beyond max_step from
        the previous frame (rad).
    """
    N = len(poses)
    n_grid = n_grid or COARSE_GRID
    pinned = {}
    if q_start is not None:
        pinned[0] = _pinned_candidate(q_start, poses[0], l1, l2, l3, 0)
    if q_end is not None:
        pinned[N - 1] = _pinned_candidate(q_end, poses[N - 1], l1, l2, l3, N - 1)

    q1, signs, cell, diag = _coarse_pass(poses, wg, l1, l2, l3, n_grid, smooth_weight,
                                         max_step, {k: p[2] for k, p in pinned.items()})
    for k, (q1k, sk, _) in pinned.items():
        q1[k], signs[k] = q1k, sk

    for _ in range(WIDE_REPEATS):
        q1_new, Q = _local_pass(poses, wg, l1, l2, l3, q1, signs, WIDE_PASS[0] * cell,
                                WIDE_PASS[1], smooth_weight, max_step, pinned)
        moved = np.abs(q1_new - q1).max()
        q1 = q1_new
        if moved <= WIDE_CONVERGED * cell:
            break
    q1, Q = _local_pass(poses, wg, l1, l2, l3, q1, signs, SHARP_PASS[0] * cell,
                        SHARP_PASS[1], smooth_weight, max_step, pinned)

    return Q, _frame_info(Q, poses, signs, diag, wg, l1, l2, l3, max_step)
