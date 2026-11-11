"""
planner.py — how the robot crosses a transition, planned from its geometry in five or six phases.

The transitions themselves are defined in core/transitions.py; this module
turns any such definition into a motion plan: the phases (plan_phases), the
poses at any moment (plan_pose, sample_poses) and the standard posture at both
ends (cruise_posture).  Beside it: transition.py (the definition), placement.py
(where an assembly goes), follow_gap.py (the gap at an outside corner or edge)
and wheels.py (wheel speeds, from this plan).

Every transition carries the robot from one surface to another across an
inside corner (floor → wall, wall → ceiling), an outside corner (top of a
block → its side wall) or the end of a thin plate (top → underside).  All of
them run through the same phases:

  1 approach    both assemblies roll toward the transition until the
                leading one is ready to pivot
  2 pivot       the leading assembly turns onto the new surface
  3 roll clear  the leading assembly rolls along the new surface, clear of
                the transition
  4 roll up     the trailing assembly rolls up to the transition
  5 pivot       the trailing assembly turns onto the new surface
  6 depart      both roll away to the standard end

At an inside corner, 3 and 4 move the assemblies one after the other so the
chain never folds up between the two surfaces — unless that would take the
chain out of reach (a corner wider than a right angle), when they roll
together instead (plan_phases).  At an outside corner or edge
they are a single phase, "both roll", in which the two assemblies roll at the
same time (Transition.rolls_together), so those transitions have five phases.

Each phase starts and ends at rest: its speed ramps up and back down along
an S-curve (ease, params.RAMP_TIME), so the robot never changes speed
abruptly.

"Gap" means the distance between the two chain joints (P0 on the front
assembly, P3 on the back one).  What the partner of a pivoting assembly does
depends on the transition (Transition.partner_follows):

  inside corner   the partner parks.  Approach and depart roll both
                  assemblies together at the cruise gap; in between, the gap
                  is whatever the geometry makes it.
  outside corner  the partner follows the pivoting assembly, rolling to keep
  or edge         the gap at a set value, but never backwards: where keeping
                  it exactly would make the partner back up, it eases to a
                  stop early and waits, and the gap opens a little instead
                  (_without_backsteps).  Approach closes the gap from the
                  cruise gap to that value and depart opens it again.  The
                  value is computed for each transition (follow_gap.py): the gap
                  that keeps the joints farthest from their limits.

Every transition starts with its leading chain joint START_DISTANCE before the
end of its approach surface and ends with its trailing chain joint
END_DISTANCE along the depart surface, both times CRUISE_GAP apart; the
solver also fixes the joint angles at both ends to the same cruise posture
(cruise_posture).  The parameters are identical for all transitions; nothing
below is tuned per transition.

Positions come from geometry only (placement.py):
  * resting on a surface: the chain joint sits where the lowest wheel just
    touches the surface (rest_point)
  * pivoting at an inside corner: the assembly stays pressed into both
    surfaces while it turns (pivot_pose)
  * pivoting at an outside corner or edge: the assembly's trailing wheel rolls
    around the solid's outline — round the corner, along the end face if
    the solid has a thickness, round the next corner — while the assembly
    turns at an even rate (pivot_pose)
so changing the link lengths, wheel and arm geometry, or where the surfaces
are, changes the motion automatically.  The plan is checked as it is sampled:
the chain joints stay within the chain's reach (no farther apart than its
length, no closer than it can fold, CHAIN_MIN_SPAN), neither assembly may
enter the solid (wheels may touch it) and the two assemblies keep the hard
clearance from each other (check_assemblies).
Phase durations come from the speed limits in params.py (see
_phase_duration); the trajectory solver stretches a phase further if the
joints cannot keep up.

Assemblies: "front" is assembly 1 (chain joint P0; pose x1, y1, theta1;
side +1) and "back" is assembly 2 (P3; x2, y2, theta2; side -1).  Angles are
radians, counter-clockwise positive.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from .chain_ik import solve_branch_batch, posture_cost
from .constraints import evaluate
from .geometry import vec
from .placement import along, check_assemblies, pivot_pose, rest_point, roll_to_gap
from .transition import PlanError, surface_orientation
from ..params import (CRUISE_GAP, END_DISTANCE, JOINT_HARD_LIMIT_DEG, PIVOT_RATE, RAMP_TIME,
                     SCHEDULE_STEP, ROLL_SPEED, START_DISTANCE)
from ..robot import CHAIN_LENGTH, l1, l2, l3


# ── The robot (every dimension is in robot.py) ────────────────────────────────

def _chain_min_span():
    """
    The shortest gap the chain can fold to: q2 and q3 bent the allowed way
    (≤ 0, constraints.py) and no further than the hard joint limit.  With l2
    along x, P3 − P0 = (l2 + l1 cos q2 + l3 cos q3, l3 sin q3 − l1 sin q2).
    """
    q = np.radians(np.linspace(-JOINT_HARD_LIMIT_DEG, 0.0, 221))
    q2, q3 = np.meshgrid(q, q)
    return float(np.hypot(l2 + l1 * np.cos(q2) + l3 * np.cos(q3),
                          l3 * np.sin(q3) - l1 * np.sin(q2)).min())


CHAIN_MIN_SPAN = _chain_min_span()   # ≈ 0.33 m (depends on the joint limit too, so it lives here)

# The motion plan every transition shares (start and end distances, cruise
# gap, follow-gap search) is set in params.py.

SIDE = {"front": +1, "back": -1}


def _lerp(a, b, f):
    """The point a fraction f of the way from a to b."""
    return a + f * (b - a)


# ── Phases ────────────────────────────────────────────────────────────────────

@dataclass
class Phase:
    """
    One phase of a transition.

    pose(f), f ∈ [0, 1] through the phase, returns
    (leading chain joint, leading θ, trailing chain joint, trailing θ).
    """
    name: str
    duration: float
    pose: Callable


def ease(tau, ramp):
    """
    How much of a phase's motion is done (0 … 1) at fraction tau ∈ [0, 1] of
    its time.  The speed ramps up from rest over the first `ramp` of the time
    along a raised-cosine S-curve, holds, and ramps back down to rest over
    the last `ramp` (a fraction ≤ 0.5; at 0.5 there is no constant part).
    Speed and acceleration are continuous, and the peak speed is
    1 / (1 − ramp) times the average.
    """
    r = min(ramp, 0.5)
    if r <= 0.0:
        return tau
    peak = 1.0 / (1.0 - r)

    def ramp_up(x):                        # motion done x into the first ramp (x ≤ r)
        return peak * (x / 2 - r / (2 * math.pi) * math.sin(math.pi * x / r))

    if tau < r:
        return ramp_up(tau)
    if tau > 1.0 - r:
        return 1.0 - ramp_up(1.0 - tau)
    return peak * (tau - r / 2)


def _phase_duration(pose, minimum):
    """
    Seconds for a phase: long enough that, easing in and out (ease), neither
    assembly rolls faster than ROLL_SPEED or turns faster than PIVOT_RATE
    anywhere in it — the time at constant speed plus RAMP_TIME — and at
    least `minimum`.  Rounded up to whole SCHEDULE_STEPs.
    """
    f = np.linspace(0.0, 1.0, 401)
    samples = [pose(x) for x in f]
    df = f[1] - f[0]
    roll = max(float(np.max(np.linalg.norm(np.diff(np.array([s[k] for s in samples]), axis=0),
                                            axis=1))) for k in (0, 2)) / df
    turn = max(float(np.max(np.abs(np.diff(np.array([s[k] for s in samples])))))
               for k in (1, 3)) / df
    seconds = max(max(roll / ROLL_SPEED, turn / PIVOT_RATE) + RAMP_TIME, minimum)
    return max(1, math.ceil(seconds / SCHEDULE_STEP - 1e-9)) * SCHEDULE_STEP


def _without_backsteps(trn, pose, k, surface, side, approaching):
    """
    A pivot phase `pose` with its following assembly — element k of the
    pose, resting on `surface` — changed so that it never rolls backwards.

    Keeping the gap exactly can make the follower back up: through the thin
    edge's half turn the pivoting chain joint swings out past the end of the
    plate and then back under it, and the follower has to retreat to stay
    `gap` from it.  Instead the follower waits: approaching the transition,
    it never gets nearer to it than it will be later; leaving it, it never
    comes back toward it.  It is then never nearer the pivoting joint than
    following exactly would put it, so the gap opens where it waits.  Its
    stops and starts are eased: its position is smoothed over RAMP_TIME by a
    raised-cosine moving average, which keeps it moving one way and leaves
    the ends of the phase where they were.

    Returns `pose` itself when the follower never rolls backwards.
    """
    f = np.linspace(0.0, 1.0, 501)
    u = np.array([along(surface, pose(x)[k]) for x in f])        # distance from the transition
    held = np.maximum.accumulate(u[::-1])[::-1] if approaching else np.maximum.accumulate(u)
    if np.allclose(held, u, rtol=0.0, atol=1e-9):
        return pose
    # RAMP_TIME as a fraction of the phase: between its own ramps a phase
    # covers f at 1 / (duration − RAMP_TIME) per second (ease).
    width = RAMP_TIME / max(_phase_duration(pose, 0.0) - RAMP_TIME, RAMP_TIME)
    n = max(1, round(width / (f[1] - f[0]) / 2))
    w = 1.0 + np.cos(np.linspace(-np.pi, np.pi, 2 * n + 3)[1:-1])    # raised cosine, 2n + 1 taps
    # Point-reflected past both ends, so the ends stay where they are.
    padded = np.concatenate([2 * held[0] - held[n:0:-1], held, 2 * held[-1] - held[-2:-n - 2:-1]])
    smooth = np.convolve(padded, w / w.sum(), mode="valid")

    def softened(x):
        p = list(pose(x))
        p[k] = rest_point(trn, surface, float(np.interp(x, f, smooth)), side)
        return tuple(p)

    return softened


def _within_reach(*phases, samples=101):
    """
    Whether the chain can span the gap between the two chain joints all
    through these phases: never farther apart than CHAIN_LENGTH nor closer
    than CHAIN_MIN_SPAN — the test sample_poses applies to every frame.
    """
    for pose in phases:
        for f in np.linspace(0.0, 1.0, samples):
            lead, _, trail, _ = pose(f)
            if not CHAIN_MIN_SPAN <= float(np.linalg.norm(vec(lead) - vec(trail))) <= CHAIN_LENGTH:
                return False
    return True


def plan_phases(trn, min_durations=None, gap=None):
    """
    The phases of a transition: six, or five when the assemblies roll
    together between the pivots — always at an outside corner or edge
    (Transition.rolls_together), and at an inside corner when rolling one after
    the other would take the chain out of reach (_within_reach).

    Raises PlanError if the transition cannot be planned.

    gap: outside corners and edges only, where it is required — the gap (m)
    the partner aims to keep while following a pivot (it opens a little where
    keeping it would make the partner roll backwards; follow_gap.py chooses
    it).  Ignored at inside corners, where the partner parks.
    min_durations: optional lower bound on each phase's duration (seconds),
    used by the trajectory solver to slow a phase the joints cannot keep up with.
    """
    lead, trail = SIDE[trn.lead], -SIDE[trn.lead]
    trail_name = "back" if trn.lead == "front" else "front"
    A, B = trn.approach, trn.depart
    thA = surface_orientation(A)
    thB = thA + trn.turn
    C = CRUISE_GAP
    follows = trn.partner_follows
    if not follows:
        gap = C
    elif gap is None:
        raise ValueError(f"{trn.name}: an outside corner or edge needs its follow gap (follow_gap.py)")

    lead_ready = pivot_pose(trn, lead, 0.0)[0]       # leading assembly about to pivot
    trail_ready = pivot_pose(trn, trail, 0.0)[0]     # trailing assembly about to pivot
    trail_across = pivot_pose(trn, trail, 1.0)[0]    # trailing assembly after its pivot
    if along(A, lead_ready) >= START_DISTANCE:
        raise PlanError(f"{trn.name}: START_DISTANCE is too short — the leading assembly "
                         f"pivots {along(A, lead_ready):.3f} m from the transition")
    if along(B, trail_across) >= END_DISTANCE:
        raise PlanError(f"{trn.name}: END_DISTANCE is too short — the trailing assembly "
                         f"finishes its pivot {along(B, trail_across):.3f} m from the transition")

    # Where the partner of a pivoting assembly is.  Following: `gap` from the
    # pivoting chain joint, wherever it is now.  Parked: a cruise gap from
    # where that joint is at the moment the partner stands still — the start
    # of the leading pivot, the end of the trailing one.
    def trail_partner(J):
        return roll_to_gap(trn, A, trail, J if follows else lead_ready, gap)

    def lead_partner(T):
        return roll_to_gap(trn, B, lead, T if follows else trail_across, gap)

    # 2 the leading assembly pivots; the trailing one parks or follows.
    def lead_pivot(f):
        J, theta = pivot_pose(trn, lead, f)
        return J, theta, trail_partner(J), thA

    # 5 the trailing assembly pivots; the leading one parks or follows.
    def trail_pivot(f):
        T, theta = pivot_pose(trn, trail, f)
        return lead_partner(T), thB, T, theta

    # A follower never rolls backwards; the gap opens instead.
    if follows:
        lead_pivot = _without_backsteps(trn, lead_pivot, 2, A, trail, approaching=True)
        trail_pivot = _without_backsteps(trn, trail_pivot, 0, B, lead, approaching=False)

    # 1 approach: from the standard start until the leading assembly is in
    #   position to pivot.  Parked: both roll together at the cruise gap.
    #   Following: the trailing assembly closes the gap to `gap`.
    lead_start = rest_point(trn, A, START_DISTANCE, lead)
    trail_start = rest_point(trn, A, START_DISTANCE + C, trail)
    trail_waiting = lead_pivot(0.0)[2]

    def approach(f):
        return _lerp(lead_start, lead_ready, f), thA, _lerp(trail_start, trail_waiting, f), thA

    # 3, 4: from the end of phase 2 to the start of phase 5.  At an inside
    #   corner the leading assembly rolls along the new surface first (3),
    #   then the trailing one rolls up to the transition (4), which keeps the
    #   chain from folding up between the two surfaces — if the chain can span
    #   the gap all the way.  Otherwise (a corner wider than a right angle), and
    #   always at an outside corner or edge, both do it at once, in one phase.
    lead_across, _, trail_after, _ = lead_pivot(1.0)
    lead_waiting = trail_pivot(0.0)[0]

    def lead_rolls(f):
        return _lerp(lead_across, lead_waiting, f), thB, trail_after, thA

    def trail_rolls(f):
        return lead_waiting, thB, _lerp(trail_after, trail_ready, f), thA

    def both_roll(f):
        return _lerp(lead_across, lead_waiting, f), thB, _lerp(trail_after, trail_ready, f), thA

    # 6 depart: both roll to the standard end.  Parked: together at the
    #   cruise gap.  Following: the gap opens from `gap` to the cruise gap.
    lead_leaving = trail_pivot(1.0)[0]
    lead_end = rest_point(trn, B, END_DISTANCE + C, lead)
    trail_end = rest_point(trn, B, END_DISTANCE, trail)

    def depart(f):
        return _lerp(lead_leaving, lead_end, f), thB, _lerp(trail_across, trail_end, f), thB

    if trn.rolls_together or not _within_reach(lead_rolls, trail_rolls):
        middle = [("both roll", both_roll)]
    else:
        middle = [(f"{trn.lead} rolls clear", lead_rolls),
                  (f"{trail_name} rolls up", trail_rolls)]
    specs = [("approach", approach),
             (f"{trn.lead} pivots", lead_pivot),
             *middle,
             (f"{trail_name} pivots", trail_pivot),
             ("depart", depart)]
    minimum = list(min_durations) if min_durations is not None else [0.0] * len(specs)
    return [Phase(name, _phase_duration(fn, minimum[i]), fn)
            for i, (name, fn) in enumerate(specs)]


def phase_starts(phases):
    """Start time of each phase plus the end time, shape (len(phases) + 1,)."""
    return np.concatenate([[0.0], np.cumsum([p.duration for p in phases])])


def phase_index(starts, t):
    """Index of the phase running at time t, given phase_starts."""
    return min(int(np.searchsorted(starts, t, side="right")) - 1, len(starts) - 2)


def plan_pose(trn, phases, t):
    """
    The planned assembly poses at time t (s; clamped to the plan), as a dict
    of x1, y1, theta1 (front) and x2, y2, theta2 (back).  The plan is
    continuous in time, so it can be evaluated between frames.
    """
    starts = phase_starts(phases)
    t = min(max(t, 0.0), starts[-1])
    i = phase_index(starts, t)
    tau = min(max((t - starts[i]) / phases[i].duration, 0.0), 1.0)
    lead_xy, lead_th, trail_xy, trail_th = phases[i].pose(ease(tau, RAMP_TIME / phases[i].duration))
    if trn.lead == "front":
        (p1, th1), (p2, th2) = (lead_xy, lead_th), (trail_xy, trail_th)
    else:
        (p1, th1), (p2, th2) = (trail_xy, trail_th), (lead_xy, lead_th)
    return {"x1": p1[0], "y1": p1[1], "theta1": th1,
            "x2": p2[0], "y2": p2[1], "theta2": th2}


def sample_poses(trn, phases, rate):
    """
    Assembly poses at evenly spaced times through the phases, about `rate`
    frames per second (exactly, when the total duration is a whole number of
    frame intervals).

    Returns (times (N,), poses): each pose is a dict of x1, y1, theta1 (front)
    and x2, y2, theta2 (back) — the input the trajectory solver expects.
    Raises ValueError if the chain joints end up out of the chain's reach
    (farther apart than CHAIN_LENGTH or closer than CHAIN_MIN_SPAN) or an
    assembly collides (check_assemblies).
    """
    starts = phase_starts(phases)
    total = starts[-1]
    times = np.linspace(0.0, total, int(round(total * rate)) + 1)
    poses = []
    for t in times:
        pose = plan_pose(trn, phases, t)
        name = phases[phase_index(starts, t)].name
        gap = float(np.hypot(pose["x2"] - pose["x1"], pose["y2"] - pose["y1"]))
        if gap > CHAIN_LENGTH:
            raise PlanError(f"{trn.name}: chain joints {gap:.3f} m apart at t = {t:.2f} s "
                             f"({name}), beyond the chain's reach")
        if gap < CHAIN_MIN_SPAN:
            raise PlanError(f"{trn.name}: chain joints {gap:.3f} m apart at t = {t:.2f} s "
                             f"({name}), closer than the chain can fold ({CHAIN_MIN_SPAN:.3f} m)")
        problems = check_assemblies(trn, pose)
        if problems:
            listed = ", ".join(f"{k} ({v * 1000:+.1f} mm)" for k, v in problems.items())
            raise PlanError(f"{trn.name}: at t = {t:.2f} s ({name}): {listed}")
        poses.append(pose)
    return times, poses


# ── Standard posture ──────────────────────────────────────────────────────────

def cruise_posture(trn, pose):
    """
    The standard joint angles at the start and end of every transition: both
    middle joints bent equally (q2 = q3), the arch bending away from the
    surface (elbow branch +1).  Among the configurations at `pose` that satisfy
    this and every hard rule, returns the one closest to neutral.
    """
    args = (pose["x1"], pose["y1"], pose["theta1"],
            pose["x2"], pose["y2"], pose["theta2"], l1, l2, l3)
    env = trn.env

    def bend_difference(q1):
        q = solve_branch_batch(np.array([q1]), +1.0, *args)[0]
        return q[1] - q[2]

    grid = np.linspace(-np.pi, np.pi, 3600, endpoint=False)
    Q = solve_branch_batch(grid, +1.0, *args)
    g = Q[:, 1] - Q[:, 2]
    best = None
    for i in range(len(grid) - 1):
        # a sign change of q2 - q3 between neighbours (ignoring ±π wrap jumps)
        if np.isfinite(g[i]) and np.isfinite(g[i + 1]) and g[i] * g[i + 1] < 0 \
                and abs(g[i] - g[i + 1]) < 1.0:
            root = brentq(bend_difference, grid[i], grid[i + 1], xtol=1e-14)
            q = solve_branch_batch(np.array([root]), +1.0, *args)[0]
            cost = float(posture_cost(q[None])[0])
            if evaluate(q[None], pose, env, trn.wg, l1, l2, l3).feasible[0] \
                    and (best is None or cost < best[0]):
                best = (cost, q)
    if best is None:
        raise PlanError(f"{trn.name}: no feasible cruise posture at this pose")
    return best[1]
