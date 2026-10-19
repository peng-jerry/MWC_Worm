"""
scenarios.py — the four surface transitions, each built from geometry in five or six phases.

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
chain never folds up between the two surfaces.  At an outside corner or edge
they are a single phase, "both roll", in which the two assemblies roll at the
same time (Scenario.rolls_together), so those transitions have five phases.

Each phase starts and ends at rest: its speed ramps up and back down along
an S-curve (ease, timing.RAMP_TIME), so the robot never changes speed
abruptly.

"Gap" means the distance between the two chain joints (P0 on the front
assembly, P3 on the back one).  What the partner of a pivoting assembly does
depends on the transition (Scenario.partner_follows):

  inside corner   the partner parks.  Approach and depart roll both
                  assemblies together at the cruise gap; in between, the gap
                  is whatever the geometry makes it.
  outside corner  the partner follows the pivoting assembly, rolling to keep
  or edge         the gap at a set value, but never backwards: where keeping
                  it exactly would make the partner back up, it eases to a
                  stop early and waits, and the gap opens a little instead
                  (_without_backsteps).  Approach closes the gap from the
                  cruise gap to that value and depart opens it again.  The
                  value is computed for each transition (follow_gap): the gap
                  that keeps the joints farthest from their limits.

Every transition starts with its leading chain joint START_DISTANCE before the
end of its approach surface and ends with its trailing chain joint
END_DISTANCE along the depart surface, both times CRUISE_SEPARATION apart; the
solver also fixes the joint angles at both ends to the same cruise posture
(cruise_posture).  The parameters are identical for all transitions; nothing
below is tuned per transition.

Positions come from geometry only:
  * resting on a surface: the chain joint sits where the lowest wheel just
    touches the surface (rest_point)
  * pivoting at an inside corner: the assembly stays pressed into both
    surfaces while it turns (pivot_pose)
  * pivoting at an outside corner or edge: the assembly's trailing wheel rolls
    around the solid's outline — round the corner, along the end face if
    the solid has a thickness, round the next corner (pivot_pose)
so changing the link lengths, wheel and arm geometry, or where the surfaces
are, changes the motion automatically.  The plan is checked as it is sampled:
the chain joints stay within the chain's reach (no farther apart than its
length, no closer than it can fold, CHAIN_MIN_SPAN), neither assembly may
enter the solid (wheels may touch it) and the two assemblies keep the hard
clearance from each other (check_assemblies).
Phase durations come from the speed limits in timing.py (see
_phase_duration); the trajectory solver stretches a phase further if the
joints cannot keep up.

Assemblies: "front" is assembly 1 (chain joint P0; pose x1, y1, theta1;
side +1) and "back" is assembly 2 (P3; x2, y2, theta2; side -1).  Angles are
radians, counter-clockwise positive.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from constraints import (FEASIBILITY_TOL, HARD_CLEARANCE, JOINT_HARD_LIMIT_DEG, WheelGeometry,
                         assembly_parts, evaluate, wheel_centers)
from environment import Environment, Surface
from kinematics import q_display
from solver import solve_branch_batch, posture_cost
from timing import SAMPLE_RATE, SCHEDULE_STEP, SLIDE_SPEED, PIVOT_RATE, RAMP_TIME


# ── The robot ─────────────────────────────────────────────────────────────────
l1 = 0.250                   # front link (m)
l2 = 0.500                   # middle link (m)
l3 = 0.250                   # back link (m)
CHAIN_LENGTH = l1 + l2 + l3  # 1.00 m


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


CHAIN_MIN_SPAN = _chain_min_span()   # ≈ 0.33 m

# Wheel assemblies (measured dimensions in constraints.py).  The arms can be
# rotated, so each kind of transition uses the arm layout that suits it.
wg = WheelGeometry()               # arm B on the inboard side of each assembly
_ARM_A = abs(wg.arm_a1)            # ≈ 17.8°
_ARM_B = abs(wg.arm_b1)            # = 45°
wg_outside = WheelGeometry(        # both assemblies: arm B wheel on the −x side
    bar_len=wg.bar_len, wheel_r=wg.wheel_r,
    arm_a1=_ARM_A, arm_b1=-_ARM_B,
    arm_a2=_ARM_A, arm_b2=-_ARM_B,
    calf=wg.calf, bar_len_a=wg.bar_len_a,
)
wg_thin_edge = WheelGeometry(      # both assemblies: arm B wheel on the +x side
    bar_len=wg.bar_len, wheel_r=wg.wheel_r,
    arm_a1=-_ARM_A, arm_b1=_ARM_B,
    arm_a2=-_ARM_A, arm_b2=_ARM_B,
    calf=wg.calf, bar_len_a=wg.bar_len_a,
)

# ── Motion plan: the same for every transition ────────────────────────────────
START_DISTANCE        = 0.40                  # m, leading chain joint before the transition at the start
END_DISTANCE          = 0.40                  # m, trailing chain joint past the transition at the end
CRUISE_SEPARATION     = 0.80 * CHAIN_LENGTH   # gap between chain joints at the start and end
FOLLOW_GAP_STEP       = 0.005                 # m, spacing of the follow gaps tried (follow_gap)
FOLLOW_GAP_TOLERANCE  = 0.010                 # m, how far the robot's real gap may be off the plan

SIDE = {"front": +1, "back": -1}

THIN_EDGE_THICKNESS = 0.0034                  # m, the plate of the thin-edge transition


# ── Transition definitions ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Scenario:
    """
    One transition: the two surfaces, how the robot turns between them and
    which assembly crosses first.

    approach, depart  the surface the robot arrives on and leaves on
                      (environment.Surface); their origins are where they
                      end at the transition
    turn              each assembly's rotation during its pivot (rad, + =
                      CCW); takes the approach orientation to the departure
                      orientation
    corner            "inside" (concave corner) or "outside" (convex corner
                      or the end of a plate); decides how an assembly pivots
                      and whether its partner parks or follows
    lead              "front" or "back": the assembly that crosses first
    xlim, ylim        drawing only: fixed (low, high) plot limits; None on
                      either side fits that side to the robot
    """
    name: str
    label: str
    approach: Surface
    depart: Surface
    turn: float
    corner: str
    lead: str
    wg: WheelGeometry
    output: str
    xlim: Tuple[Optional[float], Optional[float]] = (None, None)
    ylim: Tuple[Optional[float], Optional[float]] = (None, None)

    @property
    def env(self) -> Environment:
        """The solid around the transition (environment.Environment)."""
        return Environment(self.approach, self.depart, self.corner)

    @property
    def partner_follows(self):
        """
        Whether the partner of a pivoting assembly rolls after it (True) or
        stays parked (False).  At an inside corner the pivoting assembly stays
        tucked into the corner, so its chain joint barely moves and the
        partner can stay where it is.  At an outside corner or edge it swings
        around the outside of the transition, carrying its chain joint
        away — beyond the chain's reach if the partner stayed put.
        """
        return self.corner == "outside"

    @property
    def rolls_together(self):
        """
        Whether both assemblies roll at the same time between the two pivots
        (True: one phase) or one after the other (False: the leading one
        rolls clear, then the trailing one rolls up).  At an inside corner,
        rolling both at once would bring them toward the corner together and
        fold the chain up between the two surfaces.  At an outside corner or
        edge, rolling both at once keeps the gap at or below the follow gap
        all the way (each moves in a straight line, so the gap is largest at
        the ends, where it is the follow gap); rolling them in turn would
        stretch the chain over the transition instead.
        """
        return self.corner == "outside"

    def __post_init__(self):
        if abs(_wrap(surface_orientation(self.approach) + self.turn
                     - surface_orientation(self.depart))) > 1e-9:
            raise ValueError(f"{self.name}: turn {np.degrees(self.turn):.1f}° does not take the "
                             f"approach orientation to the departure orientation")
        if self.lead not in SIDE:
            raise ValueError(f"{self.name}: lead must be front or back")
        self.env                                    # validates the corner and the surfaces


# ── Geometry ──────────────────────────────────────────────────────────────────

def _wrap(a):
    """Wrap angles to [-π, π)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _vec(v):
    return np.asarray(v, dtype=float)


def _up(theta):
    """The 'up' direction of an assembly at orientation theta (away from its wheels)."""
    return np.array([-math.sin(theta), math.cos(theta)])


def surface_orientation(surface):
    """Assembly orientation θ for resting on `surface` (its wheels point into it)."""
    n = surface.normal
    return math.atan2(n[1], n[0]) - math.pi / 2


def wheel_offsets(theta, wg, side):
    """Wheel centres relative to the chain joint, shape (2, 2): arm A, arm B."""
    return np.array(wheel_centers(0.0, 0.0, theta, wg, side))


def rest_point(scn, surface, u, side):
    """
    Chain joint of an assembly resting on `surface`, u metres from the
    surface's end along it.  The lowest wheel just touches it.
    """
    n, t = _vec(surface.normal), _vec(surface.tangent)
    W = wheel_offsets(surface_orientation(surface), scn.wg, side)
    height = scn.wg.wheel_r - np.min(W @ n)        # chain joint height above the surface
    return _vec(surface.origin) + u * t + height * n


def along(surface, joint):
    """How far a chain joint is from the end of `surface`, measured along it."""
    return float((joint - _vec(surface.origin)) @ _vec(surface.tangent))


def _outline(scn):
    """
    The path of the trailing wheel's centre around an outside corner or edge,
    as pieces ("turn", corner point, θ from, θ to) and ("roll", from, to, θ),
    each with its length.  The wheel turns round the approach surface's end,
    rolls along the end face if the solid has one, and turns round the
    depart surface's end.
    """
    r = scn.wg.wheel_r
    thA = surface_orientation(scn.approach)
    oA, oB = _vec(scn.approach.origin), _vec(scn.depart.origin)
    end_faces = [f for f in scn.env.faces if np.isfinite(f.length)]
    if not end_faces:
        return [(("turn", oA, thA, thA + scn.turn), r * abs(scn.turn))]
    nE = end_faces[0].normal
    # The end face's orientation, reached from the approach orientation by
    # turning the same way as the whole pivot.
    dE = _wrap(math.atan2(nE[1], nE[0]) - math.pi / 2 - thA)
    if dE * scn.turn < 0 or abs(dE) > abs(scn.turn) + 1e-9:
        raise ValueError(f"{scn.name}: the end face does not lie between the two surfaces")
    thE = thA + dE
    return [(("turn", oA, thA, thE), r * abs(dE)),
            (("roll", oA, oB, thE), float(np.linalg.norm(oB - oA))),
            (("turn", oB, thE, thA + scn.turn), r * abs(scn.turn - dE))]


def pivot_pose(scn, side, f):
    """
    Chain joint and orientation of an assembly pivoting across the transition,
    f ∈ [0, 1] through the pivot.

    Inside corner: the orientation turns at an even rate and the assembly
    stays pressed into both surfaces — for each surface the wheel nearest to
    it just touches it.  That fixes the joint's distance from both surfaces.

    Outside corner or edge: the trailing wheel (the last to leave the
    approach surface) rolls around the solid's outline (_outline): its centre
    stays one wheel radius from the solid, in the assembly's "up" direction,
    as if the assembly stood on a surface that turns with it.  f runs evenly
    along that path.
    """
    thA = surface_orientation(scn.approach)
    r = scn.wg.wheel_r
    if scn.corner == "inside":
        theta = thA + f * scn.turn
        W = wheel_offsets(theta, scn.wg, side)
        O = _vec(scn.approach.origin)
        nA, nB = _vec(scn.approach.normal), _vec(scn.depart.normal)
        rhs = [nA @ O + r - np.min(W @ nA), nB @ O + r - np.min(W @ nB)]
        return np.linalg.solve(np.array([nA, nB]), rhs), theta

    trailing = int(np.argmax(wheel_offsets(thA, scn.wg, side) @ _vec(scn.approach.tangent)))
    pieces = _outline(scn)
    s = f * sum(length for _, length in pieces)
    for k, (piece, length) in enumerate(pieces):
        if s <= length or k == len(pieces) - 1:
            g = min(s / length, 1.0) if length > 0 else 1.0
            break
        s -= length
    if piece[0] == "turn":
        _, corner, th0, th1 = piece
        theta = th0 + g * (th1 - th0)
        centre = corner + r * _up(theta)
    else:
        _, start, end, theta = piece
        centre = start + g * (end - start) + r * _up(theta)
    return centre - wheel_offsets(theta, scn.wg, side)[trailing], theta


def roll_to_separation(scn, surface, side, other, separation):
    """
    Chain joint of an assembly resting on `surface` exactly `separation` from
    the chain joint `other`, choosing the position farther from the
    surface's end.
    """
    base = rest_point(scn, surface, 0.0, side)
    t = _vec(surface.tangent)
    # |base + u·t - other| = separation  →  u² + 2bu + c = 0
    d = base - other
    b = t @ d
    c = d @ d - separation ** 2
    disc = b * b - c
    if disc < 0:
        raise ValueError(f"{scn.name}: no position on the surface is {separation:.3f} m from the "
                         f"other chain joint")
    return base + (-b + math.sqrt(disc)) * t


def _lerp(a, b, f):
    return a + f * (b - a)


def check_assemblies(scn, pose):
    """
    Problems with where a pose puts the two wheel assemblies, as
    {description: gap (m)}; empty when there are none.

      * no part of either assembly may enter the solid (wheels touch it,
        so a gap down to -FEASIBILITY_TOL is allowed)
      * the two assemblies stay HARD_CLEARANCE apart, measured through free
        space (on opposite faces of a plate, around its end)
    """
    env = scn.env
    front = assembly_parts(pose["x1"], pose["y1"], pose["theta1"], scn.wg, +1)
    back = assembly_parts(pose["x2"], pose["y2"], pose["theta2"], scn.wg, -1)
    names = [f"front {n}" for n in front] + [f"back {n}" for n in back]
    a, b, r = (np.array([part[i] for part in [*front.values(), *back.values()]]) for i in range(3))
    problems = {}
    for name, gap in zip(names, env.clearance(a, b, r)):
        if gap < -FEASIBILITY_TOL:
            problems[f"{name} in the surface"] = float(gap)
    # Every front part against every back part: (7, 1) against (1, 7).
    n = len(front)
    gap = (env.separation(a[:n, None], b[:n, None], a[None, n:], b[None, n:],
                          within=HARD_CLEARANCE + r[:n, None] + r[None, n:])
           - r[:n, None] - r[None, n:]).min()
    if gap < HARD_CLEARANCE - FEASIBILITY_TOL:
        problems["front and back assemblies too close"] = float(gap)
    return problems


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
    assembly rolls faster than SLIDE_SPEED or turns faster than PIVOT_RATE
    anywhere in it — the time at constant speed plus RAMP_TIME — and at
    least `minimum`.  Rounded up to whole SCHEDULE_STEPs.
    """
    f = np.linspace(0.0, 1.0, 401)
    samples = [pose(x) for x in f]
    df = f[1] - f[0]
    slide = max(float(np.max(np.linalg.norm(np.diff(np.array([s[k] for s in samples]), axis=0),
                                            axis=1))) for k in (0, 2)) / df
    turn = max(float(np.max(np.abs(np.diff(np.array([s[k] for s in samples])))))
               for k in (1, 3)) / df
    seconds = max(max(slide / SLIDE_SPEED, turn / PIVOT_RATE) + RAMP_TIME, minimum)
    return max(1, math.ceil(seconds / SCHEDULE_STEP - 1e-9)) * SCHEDULE_STEP


def _without_backsteps(scn, pose, k, surface, side, approaching):
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
        p[k] = rest_point(scn, surface, float(np.interp(x, f, smooth)), side)
        return tuple(p)

    return softened


def plan_phases(scn, min_durations=None, gap=None):
    """
    The phases of a transition: six, or five when the assemblies roll
    together between the pivots (Scenario.rolls_together).

    gap: outside corners and edges only — the gap (m) the partner aims to
    keep while following a pivot (it opens a little where keeping it would
    make the partner roll backwards).  Computed by follow_gap(scn) when not
    given.  Ignored at inside corners, where the partner parks.
    min_durations: optional lower bound on each phase's duration (seconds),
    used by the trajectory solver to slow a phase the joints cannot keep up with.
    """
    lead, trail = SIDE[scn.lead], -SIDE[scn.lead]
    trail_name = "back" if scn.lead == "front" else "front"
    A, B = scn.approach, scn.depart
    thA = surface_orientation(A)
    thB = thA + scn.turn
    C = CRUISE_SEPARATION
    follows = scn.partner_follows
    if not follows:
        gap = C
    elif gap is None:
        gap = follow_gap(scn)

    lead_ready = pivot_pose(scn, lead, 0.0)[0]       # leading assembly about to pivot
    trail_ready = pivot_pose(scn, trail, 0.0)[0]     # trailing assembly about to pivot
    trail_across = pivot_pose(scn, trail, 1.0)[0]    # trailing assembly after its pivot
    if along(A, lead_ready) >= START_DISTANCE:
        raise ValueError(f"{scn.name}: START_DISTANCE is too short — the leading assembly "
                         f"pivots {along(A, lead_ready):.3f} m from the transition")
    if along(B, trail_across) >= END_DISTANCE:
        raise ValueError(f"{scn.name}: END_DISTANCE is too short — the trailing assembly "
                         f"finishes its pivot {along(B, trail_across):.3f} m from the transition")

    # Where the partner of a pivoting assembly is.  Following: `gap` from the
    # pivoting chain joint, wherever it is now.  Parked: a cruise gap from
    # where that joint is at the moment the partner stands still — the start
    # of the leading pivot, the end of the trailing one.
    def trail_partner(J):
        return roll_to_separation(scn, A, trail, J if follows else lead_ready, gap)

    def lead_partner(T):
        return roll_to_separation(scn, B, lead, T if follows else trail_across, gap)

    # 2 the leading assembly pivots; the trailing one parks or follows.
    def lead_pivot(f):
        J, theta = pivot_pose(scn, lead, f)
        return J, theta, trail_partner(J), thA

    # 5 the trailing assembly pivots; the leading one parks or follows.
    def trail_pivot(f):
        T, theta = pivot_pose(scn, trail, f)
        return lead_partner(T), thB, T, theta

    # A follower never rolls backwards; the gap opens instead.
    if follows:
        lead_pivot = _without_backsteps(scn, lead_pivot, 2, A, trail, approaching=True)
        trail_pivot = _without_backsteps(scn, trail_pivot, 0, B, lead, approaching=False)

    # 1 approach: from the standard start until the leading assembly is in
    #   position to pivot.  Parked: both roll together at the cruise gap.
    #   Following: the trailing assembly closes the gap to `gap`.
    lead_start = rest_point(scn, A, START_DISTANCE, lead)
    trail_start = rest_point(scn, A, START_DISTANCE + C, trail)
    trail_waiting = lead_pivot(0.0)[2]

    def approach(f):
        return _lerp(lead_start, lead_ready, f), thA, _lerp(trail_start, trail_waiting, f), thA

    # 3, 4: from the end of phase 2 to the start of phase 5.  At an inside
    #   corner the leading assembly rolls along the new surface first (3),
    #   then the trailing one rolls up to the transition (4), which keeps the
    #   chain from folding up between the two surfaces.  At an outside corner
    #   or edge both do it at once, in one phase (Scenario.rolls_together).
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
    lead_end = rest_point(scn, B, END_DISTANCE + C, lead)
    trail_end = rest_point(scn, B, END_DISTANCE, trail)

    def depart(f):
        return _lerp(lead_leaving, lead_end, f), thB, _lerp(trail_across, trail_end, f), thB

    if scn.rolls_together:
        middle = [("both roll", both_roll)]
    else:
        middle = [(f"{scn.lead} rolls clear", lead_rolls),
                  (f"{trail_name} rolls up", trail_rolls)]
    specs = [("approach", approach),
             (f"{scn.lead} pivots", lead_pivot),
             *middle,
             (f"{trail_name} pivots", trail_pivot),
             ("depart", depart)]
    minimum = list(min_durations) if min_durations is not None else [0.0] * len(specs)
    return [Phase(name, _phase_duration(fn, minimum[i]), fn)
            for i, (name, fn) in enumerate(specs)]


def phase_starts(phases):
    """Start time of each phase plus the end time, shape (len(phases) + 1,)."""
    return np.concatenate([[0.0], np.cumsum([p.duration for p in phases])])


def _phase_index(starts, t):
    """Index of the phase running at time t, given phase_starts."""
    return min(int(np.searchsorted(starts, t, side="right")) - 1, len(starts) - 2)


def plan_pose(scn, phases, t):
    """
    The planned assembly poses at time t (s; clamped to the plan), as a dict
    of x1, y1, theta1 (front) and x2, y2, theta2 (back).  The plan is
    continuous in time, so it can be evaluated between frames.
    """
    starts = phase_starts(phases)
    t = min(max(t, 0.0), starts[-1])
    i = _phase_index(starts, t)
    tau = min(max((t - starts[i]) / phases[i].duration, 0.0), 1.0)
    lead_xy, lead_th, trail_xy, trail_th = phases[i].pose(ease(tau, RAMP_TIME / phases[i].duration))
    if scn.lead == "front":
        (p1, th1), (p2, th2) = (lead_xy, lead_th), (trail_xy, trail_th)
    else:
        (p1, th1), (p2, th2) = (trail_xy, trail_th), (lead_xy, lead_th)
    return {"x1": p1[0], "y1": p1[1], "theta1": th1,
            "x2": p2[0], "y2": p2[1], "theta2": th2}


def sample_poses(scn, phases, rate):
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
        pose = plan_pose(scn, phases, t)
        name = phases[_phase_index(starts, t)].name
        gap = float(np.hypot(pose["x2"] - pose["x1"], pose["y2"] - pose["y1"]))
        if gap > CHAIN_LENGTH:
            raise ValueError(f"{scn.name}: chain joints {gap:.3f} m apart at t = {t:.2f} s "
                             f"({name}), beyond the chain's reach")
        if gap < CHAIN_MIN_SPAN:
            raise ValueError(f"{scn.name}: chain joints {gap:.3f} m apart at t = {t:.2f} s "
                             f"({name}), closer than the chain can fold ({CHAIN_MIN_SPAN:.3f} m)")
        problems = check_assemblies(scn, pose)
        if problems:
            listed = ", ".join(f"{k} ({v * 1000:+.1f} mm)" for k, v in problems.items())
            raise ValueError(f"{scn.name}: at t = {t:.2f} s ({name}): {listed}")
        poses.append(pose)
    return times, poses


# ── The follow gap (outside corners and edges) ────────────────────────────────

def peak_joint_angle(scn, phases, rate=SAMPLE_RATE):
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
        _, poses = sample_poses(scn, phases, rate)
    except ValueError:                       # out of reach, or an assembly collides
        return math.inf
    env = scn.env
    q1 = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    peak = 0.0
    for p in poses:
        args = (p["x1"], p["y1"], p["theta1"], p["x2"], p["y2"], p["theta2"])
        Q = solve_branch_batch(q1, +1.0, *args, l1, l2, l3)
        Q = Q[~np.isnan(Q[:, 0])]
        if len(Q) == 0:
            return math.inf
        feasible = evaluate(Q, p, env, scn.wg, l1, l2, l3).feasible
        if not feasible.any():
            return math.inf
        peak = max(peak, float(np.abs(q_display(Q[feasible])).max(axis=1).min()))
    return peak


def follow_gap(scn, rate=SAMPLE_RATE):
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
    n = int(CRUISE_SEPARATION / FOLLOW_GAP_STEP + 1e-9)        # grid: gap i = i × FOLLOW_GAP_STEP
    k = int(round(FOLLOW_GAP_TOLERANCE / FOLLOW_GAP_STEP))     # tolerance, in grid steps
    scores = {}

    def score(i):
        if i not in scores:
            try:
                scores[i] = peak_joint_angle(scn, plan_phases(scn, gap=i * FOLLOW_GAP_STEP), rate)
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
    raise ValueError(f"{scn.name}: no follow gap up to the cruise gap gives every pose a valid "
                     f"configuration within ±{FOLLOW_GAP_TOLERANCE * 1000:.0f} mm")


# ── Standard posture ──────────────────────────────────────────────────────────

def cruise_posture(scn, pose):
    """
    The standard joint angles at the start and end of every transition: both
    middle joints bent equally (q2 = q3), the arch bending away from the
    surface (elbow branch +1).  Among the configurations at `pose` that satisfy
    this and every hard rule, returns the one closest to neutral.
    """
    args = (pose["x1"], pose["y1"], pose["theta1"],
            pose["x2"], pose["y2"], pose["theta2"], l1, l2, l3)
    env = scn.env

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
            if evaluate(q[None], pose, env, scn.wg, l1, l2, l3).feasible[0] \
                    and (best is None or cost < best[0]):
                best = (cost, q)
    if best is None:
        raise ValueError(f"{scn.name}: no feasible cruise posture at this pose")
    return best[1]


# ── Wheel contact ─────────────────────────────────────────────────────────────

WHEELS = ("front arm A", "front arm B", "back arm A", "back arm B")
LEAVING_SPEED = 1e-6    # m/s; a wheel moving away from a contact faster than this is leaving it


def wheel_speeds(scn, phases, times, h=1e-4, tol=FEASIBILITY_TOL):
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
    assembly about the touching point.  A contact the wheel is moving away
    from (v · n > LEAVING_SPEED) is being left and does not count.  Then:

      one contact      ω from it
      two contacts     wedged into an inside corner: the mean, which is 0
                       while the wheel stays wedged (it cannot roll along
                       both faces at once)
      none             in the air: the motor speed is interpolated in time
                       between the nearest contacts before and after, so the
                       wheel is already turning at the right speed when it
                       lands
    """
    r = scn.wg.wheel_r
    segments = scn.env.face_segments()
    end = phase_starts(phases)[-1]

    def centres(pose):
        return np.array([*wheel_centers(pose["x1"], pose["y1"], pose["theta1"], scn.wg, side=1),
                         *wheel_centers(pose["x2"], pose["y2"], pose["theta2"], scn.wg, side=-1)])

    def headings(pose):
        return np.array([pose["theta1"], pose["theta1"], pose["theta2"], pose["theta2"]])

    rolling = np.full((len(times), 4), np.nan)        # NaN until a contact sets it
    turn = np.empty((len(times), 4))
    for k, t in enumerate(times):
        ta, tb = max(t - h, 0.0), min(t + h, end)
        before, now, after = (plan_pose(scn, phases, s) for s in (ta, t, tb))
        c = centres(now)
        v = (centres(after) - centres(before)) / (tb - ta)
        turn[k] = _wrap(headings(after) - headings(before)) / (tb - ta)
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
                if v[j] @ n <= LEAVING_SPEED:
                    rates.append(np.array([-n[1], n[0]]) @ v[j] / r)
            if rates:
                rolling[k, j] = np.mean(rates)

    motor = rolling - turn
    for j in range(4):
        air = np.isnan(motor[:, j])
        if air.all():
            motor[:, j] = 0.0
        elif air.any():
            motor[air, j] = np.interp(times[air], times[~air], motor[~air, j])
    return motor + turn, motor


# ── The four transitions ──────────────────────────────────────────────────────

SCENARIOS = {
    "floor_to_wall": Scenario(
        name="floor_to_wall", label="Floor to wall (inside corner)",
        approach=Surface(origin=(0.0, 0.0), tangent=(1, 0), normal=(0, 1)),   # floor, robot at x > 0
        depart=Surface(origin=(0.0, 0.0), tangent=(0, 1), normal=(1, 0)),     # wall, robot above the floor
        turn=-np.pi / 2, corner="inside", lead="front", wg=wg,
        output="floor_to_wall.mp4"),
    "wall_to_ceiling": Scenario(
        name="wall_to_ceiling", label="Wall to ceiling (inside corner)",
        approach=Surface(origin=(0.0, 1.5), tangent=(0, -1), normal=(1, 0)),  # wall, robot below the ceiling
        depart=Surface(origin=(0.0, 1.5), tangent=(1, 0), normal=(0, -1)),   # ceiling, robot at x > 0
        turn=-np.pi / 2, corner="inside", lead="front", wg=wg,
        output="wall_to_ceiling.mp4"),
    "outside": Scenario(
        name="outside", label="Outside corner (ceiling top to exterior wall)",
        approach=Surface(origin=(0.0, 0.55), tangent=(1, 0), normal=(0, 1)),  # top of the block, robot at x > 0
        depart=Surface(origin=(0.0, 0.55), tangent=(0, -1), normal=(-1, 0)),  # exterior wall, robot below the top
        turn=+np.pi / 2, corner="outside", lead="front", wg=wg_outside,
        output="outside.mp4"),
    "thin_edge": Scenario(
        name="thin_edge", label="Thin edge (top to underside)",
        approach=Surface(origin=(0.50, 0.50), tangent=(-1, 0), normal=(0, 1)),     # top face
        depart=Surface(origin=(0.50, 0.50 - THIN_EDGE_THICKNESS),                  # underside
                       tangent=(-1, 0), normal=(0, -1)),
        turn=-np.pi, corner="outside", lead="back", wg=wg_thin_edge,
        output="thin_edge.mp4", xlim=(None, 1.1)),
}
