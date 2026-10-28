"""
trajectory_ik.py — joint angles for a whole trajectory, chosen together by
dynamic programming.

Why a whole-trajectory solve
----------------------------
At every frame the chain has one free degree of freedom (see chain_ik.py): for
any q1 on an elbow branch, chain_ik.solve_branch_batch gives the exact q2..q4
that close the chain.  Solving frame by frame means choosing each frame's q1
without knowing what later frames will need, so the robot can be cornered
into a joint step faster than the motors allow.  Choosing all frames together
avoids that: the solver can start moving early to stay within the limit.

The problem
-----------
Sample each frame's free DOF as a set of candidate configurations.  Pick one
candidate per frame to minimise

    Σ_k node(q_k)  +  Σ_k edge(q_{k-1}, q_k)

  node(q)      INFEASIBLE_COST if q breaks a hard rule (constraints.py),
               plus VIOLATION_WEIGHT × by how much (to rank such candidates)
               + dt × MARGIN_WEIGHT × the soft score (clearance and joint
                 margins, each a unitless fraction 0 … 1, squared)
               + dt × POSTURE_WEIGHT × distance from neutral (tie-breaker)
  edge(qa, qb) dt × SMOOTH_WEIGHT × Σ_j (joint step / max_step)²
               + STEP_VIOLATION_COST + STEP_EXCESS_WEIGHT · excess
                 if any joint moves more than max_step

Every soft term is multiplied by the frame interval dt, so the total is a
cost per second of motion: the weights mean the same thing at any sample
rate.  "Margins first": MARGIN_WEIGHT is far larger than SMOOTH_WEIGHT, so
the solver gives up smoothness before it gives up clearance or joint range.

The cost of a path is a sum of terms that each depend on one frame or on two
consecutive frames, so the exact minimum over all candidate paths is found by
the Viterbi recursion:

    J_0(s) = node_0(s)
    J_k(s) = node_k(s) + min over s' of [ J_{k-1}(s') + edge(s', s) ]

keeping the minimising s' for each s and tracing back from the best final
state.  Violations are made very expensive rather than impossible, so a path
always exists: if the poses demand something infeasible, the result is the
least-violating path and the per-frame diagnostics say where.

Passes
------
A single fine grid over all of q1 would be too slow, so the search narrows:

  coarse   q1 every 1° (COARSE_GRID).  Only q1 changes
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

Only the elbow branch that bends the chain away from the surface (q3 <= 0)
is searched: the other one breaks the bend rule (constraints.py) everywhere.
"""

import numpy as np

from .chain_ik import solve_branch_batch, posture_cost
from .constraints import evaluate
from .kinematics import forward_kinematics, wrap
from ..params import (BAND_FACTOR, COARSE_GRID, MARGIN_WEIGHT, POSTURE_WEIGHT, SHARP_PASS,
                      SMOOTH_WEIGHT, WIDE_CONVERGED, WIDE_PASS, WIDE_REPEATS)

# Hard rules: violations dominate every other cost term.
INFEASIBLE_COST     = 1e6   # per frame whose configuration breaks a hard rule
VIOLATION_WEIGHT    = 1e6   # per metre / radian of hard-rule violation (ranks infeasible candidates)
STEP_VIOLATION_COST = 1e6   # per transition in which any joint exceeds max_step
STEP_EXCESS_WEIGHT  = 1e8   # per radian beyond max_step, to rank unavoidable violations
PIN_WEIGHT          = 1e6   # coarse pass: draw a pinned frame to its grid point nearest the pin

# The soft-term weights (MARGIN_WEIGHT, SMOOTH_WEIGHT, POSTURE_WEIGHT) and the
# search resolution (COARSE_GRID … SHARP_PASS) are tunable, so they are set in
# params.py.  The costs above are not: they only need to dwarf every soft term.

_SIGNS = (+1.0,)              # elbow branches searched; +1 gives q3 <= 0 (see the module docstring)


def _pose_args(pose):
    """(x1, y1, theta1, x2, y2, theta2) from a pose dict."""
    return (pose["x1"], pose["y1"], pose["theta1"],
            pose["x2"], pose["y2"], pose["theta2"])


# ------------------------------------------------------------------ #
#  Costs                                                              #
# ------------------------------------------------------------------ #

class _Problem:
    """What every cost needs: the environment, robot and timing (dt: seconds
    between frames; max_step: largest joint step allowed between frames, rad)."""

    def __init__(self, env, wg, l1, l2, l3, dt, max_step):
        self.env, self.wg, self.links = env, wg, (l1, l2, l3)
        self.dt, self.max_step = dt, max_step

    def node_costs(self, Q, pose):
        """
        Node cost of each candidate row of Q (K, 4) at one frame's pose.

        Rows that are NaN (the chain cannot close for that q1) cost inf.
        Returns (cost (K,), Evaluation of the finite rows or None, finite-row mask).
        """
        cost = np.full(len(Q), np.inf)
        ok = ~np.isnan(Q[:, 0])
        if not ok.any():
            return cost, None, ok
        ev = evaluate(Q[ok], pose, self.env, self.wg, *self.links)
        cost[ok] = (INFEASIBLE_COST * ~ev.feasible + VIOLATION_WEIGHT * ev.violation
                    + self.dt * (MARGIN_WEIGHT * ev.soft + POSTURE_WEIGHT * posture_cost(Q[ok])))
        return cost, ev, ok

    def edge_costs(self, Qb, Qa):
        """
        Cost of moving from configurations Qa to Qb (any broadcastable shapes
        ending in 4).  NaN where either side is NaN; callers treat NaN as inf.
        """
        dq = wrap(Qb - Qa)
        excess = np.sum(np.maximum(np.abs(dq) - self.max_step, 0.0), axis=-1)
        rate = dq / self.max_step                    # joint speed ÷ planned speed limit
        return (self.dt * SMOOTH_WEIGHT * np.sum(rate * rate, axis=-1)
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
        err = float(np.abs(wrap(q - q_pin)).max())
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

def _coarse_pass(poses, prob, n_grid, pins):
    """
    Viterbi over the coarse q1 grid on every branch in _SIGNS.

    States per frame are (branch b, grid index m).  A predecessor (b', m') is
    considered only if |m - m'| (cyclically) is within the band, so each state
    has (branches) × (2·band + 1) predecessors; all of them are scored at once.

    pins: {frame: exact pinned q}; those frames are drawn to the nearest grid
    point by PIN_WEIGHT (the wide/sharp passes then use the exact pin).

    Returns (q1 per frame, branch sign per frame, grid cell size (rad),
    per-frame grid diagnostics).
    """
    N, M, nb = len(poses), n_grid, len(_SIGNS)
    l1, l2, l3 = prob.links
    cell = 2 * np.pi / M
    q1_grid = np.linspace(-np.pi, np.pi, M, endpoint=False)

    # Candidates and node costs for every frame.
    Q = np.empty((N, nb, M, 4))
    node = np.empty((N, nb, M))
    diag = []
    for k, pose in enumerate(poses):
        for b, s in enumerate(_SIGNS):
            Q[k, b] = solve_branch_batch(q1_grid, s, *_pose_args(pose), l1, l2, l3)
        cost, ev, ok = prob.node_costs(Q[k].reshape(-1, 4), pose)
        if ev is None:
            raise RuntimeError(f"Frame {k}: no closed-chain configuration on the q1 grid "
                               f"(assemblies out of reach).")
        feasible = ev.feasible
        diag.append({
            "grid_candidates": int(ok.sum()),
            "grid_feasible":   int(feasible.sum()),
            # When no candidate is feasible the pose itself is infeasible;
            # report what still breaks for the least-violating candidate.
            "blocking":        {} if feasible.any()
                               else ev.violations(int(np.argmin(ev.violation))),
        })
        if k in pins:
            dist = np.sum(wrap(Q[k].reshape(-1, 4) - pins[k]) ** 2, axis=1)
            cost = cost + PIN_WEIGHT * np.where(ok, dist, 0.0)
        node[k] = cost.reshape(nb, M)

    # MP[o, m] is the predecessor grid index when q1 changes by offsets[o]
    # cells arriving at index m (the grid wraps around at ±π).
    band = max(1, int(np.ceil(BAND_FACTOR * prob.max_step / cell)))
    idx = np.arange(M)
    offsets = np.arange(-band, band + 1)
    MP = (idx[None, :] - offsets[:, None]) % M

    J = node[0].copy()                              # best path cost ending at each state
    back = np.zeros((N, nb, M), dtype=np.int64)     # its predecessor, as b' * M + m'
    for k in range(1, N):
        best = np.full((nb, M), np.inf)
        arg = np.zeros((nb, M), dtype=np.int64)
        for bp in range(nb):
            Jp = J[bp][MP]                          # (offsets, M)
            if not np.isfinite(Jp).any():
                continue
            Qp = Q[k - 1, bp][MP]                   # (offsets, M, 4)
            for b in range(nb):
                cand = Jp + prob.edge_costs(Q[k, b][None], Qp)
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


def _local_pass(poses, prob, q1_path, signs, half_width, points, pinned):
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
            Qk = solve_branch_batch(q1s, signs[k], *_pose_args(pose), *prob.links)
        cost, _, _ = prob.node_costs(Qk, pose)
        layers.append((q1s, Qk, cost))

    J = layers[0][2]
    backs = []
    for k in range(1, len(layers)):
        Qa, Qb, cost = layers[k - 1][1], layers[k][1], layers[k][2]
        # tot[i, j]: best cost to frame k-1 candidate i, then on to frame k candidate j
        tot = J[:, None] + prob.edge_costs(Qb[None, :, :], Qa[:, None, :])
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


def _frame_info(Q, poses, signs, diag, prob):
    """Per-frame diagnostics for the final path."""
    l1, l2, l3 = prob.links
    infos = []
    for k, (q, pose) in enumerate(zip(Q, poses)):
        ev = evaluate(q[None, :], pose, prob.env, prob.wg, l1, l2, l3)
        positions, theta_end = forward_kinematics(q, pose["x1"], pose["y1"], pose["theta1"],
                                                  l1, l2, l3)
        pos_err = float(np.hypot(positions[-1][0] - pose["x2"], positions[-1][1] - pose["y2"]))
        ang_err = float(abs(wrap(theta_end - pose["theta2"])))
        step_excess = 0.0
        if k > 0:
            step_excess = float(max(0.0, np.abs(wrap(q - Q[k - 1])).max() - prob.max_step))
        infos.append({
            "residual_norm":     float(np.hypot(pos_err, ang_err)),
            "position_error":    pos_err,
            "orientation_error": ang_err,
            "x1": pose["x1"], "y1": pose["y1"],
            "x2": pose["x2"], "y2": pose["y2"],
            "branch":            float(signs[k]),
            "extension":         float(np.hypot(pose["x2"] - pose["x1"], pose["y2"] - pose["y1"])
                                       / (l1 + l2 + l3)),
            "feasible":          bool(ev.feasible[0]),
            "violations":        ev.violations(0),
            "clearance":         {g: float(c[0]) for g, c in ev.clearance.items()},
            "margin_score":      float(ev.soft[0]),
            "step_excess":       step_excess,
            **diag[k],
        })
    return infos


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #

def solve_trajectory_dp(poses, env, wg, l1, l2, l3, *, dt, max_step,
                        n_grid=None, q_start=None, q_end=None):
    """
    Joint angles for every frame, optimised over the whole trajectory.

    Parameters
    ----------
    poses : list of dict
        Assembly poses per frame (planner.sample_poses): x1, y1, theta1,
        x2, y2, theta2.
    env : environment.Environment
        The solid around the transition.
    wg : WheelGeometry
        Wheel assembly geometry (for the robot's outline).
    l1, l2, l3 : float
        Link lengths.
    dt : float
        Time between frames (s).
    max_step : float
        Per-joint step limit between consecutive frames (rad).
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
        Per-frame diagnostics: residual (position_error, orientation_error,
        residual_norm), branch, extension (gap / chain length), feasible,
        violations {rule: amount}, clearance {group: m}, margin_score (the
        soft score), step_excess (largest joint step beyond max_step, rad),
        grid_candidates / grid_feasible / blocking (coarse-grid diagnostics).
    """
    prob = _Problem(env, wg, l1, l2, l3, dt, max_step)
    N = len(poses)
    n_grid = n_grid or COARSE_GRID
    pinned = {}
    if q_start is not None:
        pinned[0] = _pinned_candidate(q_start, poses[0], l1, l2, l3, 0)
    if q_end is not None:
        pinned[N - 1] = _pinned_candidate(q_end, poses[N - 1], l1, l2, l3, N - 1)

    q1, signs, cell, diag = _coarse_pass(poses, prob, n_grid,
                                         {k: p[2] for k, p in pinned.items()})
    for k, (q1k, sk, _) in pinned.items():
        q1[k], signs[k] = q1k, sk

    for _ in range(WIDE_REPEATS):
        q1_new, Q = _local_pass(poses, prob, q1, signs, WIDE_PASS[0] * cell, WIDE_PASS[1], pinned)
        moved = np.abs(q1_new - q1).max()
        q1 = q1_new
        if moved <= WIDE_CONVERGED * cell:
            break
    q1, Q = _local_pass(poses, prob, q1, signs, SHARP_PASS[0] * cell, SHARP_PASS[1], pinned)

    return Q, _frame_info(Q, poses, signs, diag, prob)
