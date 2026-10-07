"""
scenarios.py — the four surface transitions, each built from geometry in five phases.

Every transition carries the robot across a transition point: an inside
corner (floor → wall, wall → ceiling), an outside corner (ceiling top →
exterior wall) or the end of a thin edge (top → underside).  All of them run
through the same five phases:

  1 approach    both assemblies roll toward the transition point; their gap
                closes from CRUISE_SEPARATION to TRANSITION_SEPARATION
  2 pivot       the leading assembly turns onto the new surface
  3 both move   the leading assembly rolls clear of the transition point,
                then the trailing assembly rolls up to it
  4 pivot       the trailing assembly turns onto the new surface
  5 depart      both roll away; the gap opens back to CRUISE_SEPARATION

"Gap" and "separation" mean the distance between the two chain joints (P0 on
the front assembly, P3 on the back one).  While an assembly pivots, the other
one rolls just enough to keep the separation at TRANSITION_SEPARATION.  In
phase 3 the leading assembly moves first so the chain never folds up between
the two surfaces.

Every transition starts with its leading chain joint START_DISTANCE before the
transition point and ends with its trailing chain joint END_DISTANCE past it,
both times CRUISE_SEPARATION apart; the solver also fixes the joint angles at
both ends to the same cruise posture (cruise_posture).  The parameters are
identical for all transitions; nothing below is tuned per transition.

Positions come from geometry only:
  * resting on a surface: the chain joint sits where the lowest wheel just
    touches the surface (rest_point)
  * pivoting at an inside corner: the assembly stays pressed into both
    surfaces while it turns (pivot_point)
  * pivoting at an outside corner or edge: the assembly's trailing wheel rolls
    around the transition point while the body turns (pivot_point)
so changing the link lengths, wheel and arm geometry, or where the surfaces
are, changes the motion automatically.  Phase durations come from the speed
limits in timing.py (see _phase_duration); the trajectory solver stretches a
phase further if the joints cannot keep up.

Assemblies: "front" is assembly 1 (chain joint P0; pose x1, y1, theta1;
side +1) and "back" is assembly 2 (P3; x2, y2, theta2; side -1).  Angles are
radians, counter-clockwise positive.
"""

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import brentq

from constraints import WheelGeometry, wheel_centers
from constraints_batch import evaluate_batch
from solver import solve_branch_batch, posture_cost
from timing import SCHEDULE_STEP, SLIDE_SPEED, PIVOT_RATE


# ── The robot ─────────────────────────────────────────────────────────────────
l1 = 0.200                   # front link (m)
l2 = 2 * l1                  # middle link (m)
l3 = l1                      # back link (m)
CHAIN_LENGTH = l1 + l2 + l3  # 0.80 m

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
START_DISTANCE        = 0.40                  # m, leading chain joint before the point at the start
END_DISTANCE          = 0.40                  # m, trailing chain joint past the point at the end
CRUISE_SEPARATION     = 0.80 * CHAIN_LENGTH   # gap between chain joints at the start and end
TRANSITION_SEPARATION = 0.53 * CHAIN_LENGTH   # gap while crossing (phases 2–4); chosen for the
                                              # largest joint-limit margin over all transitions

SIDE = {"front": +1, "back": -1}


# ── Transition definitions ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Surface:
    """
    A flat surface through the transition point.

    tangent  unit vector along the surface, pointing away from the transition
             point into the part the robot drives on
    normal   unit vector out of the surface, into free space
    """
    tangent: Tuple[float, float]
    normal: Tuple[float, float]


@dataclass(frozen=True)
class Scenario:
    """
    One transition: the geometry of the transition point, which assembly
    crosses first, and the environment used for the constraint penalties.

    turn            each assembly's rotation during its pivot (rad, + = CCW);
                    takes the approach orientation to the departure orientation
    corner          "inside" (concave corner) or "outside" (convex corner or
                    the end of an edge); decides how an assembly pivots
    lead            "front" or "back": the assembly that crosses first
    constraint_set  penalty set in constraints.py; wall_x and ceiling_y are its
                    environment (thin_edge uses them as edge_x and edge_y)
    """
    name: str
    label: str
    point: Tuple[float, float]
    approach: Surface
    depart: Surface
    turn: float
    corner: str
    lead: str
    wg: WheelGeometry
    constraint_set: str
    wall_x: float
    ceiling_y: float
    output: str

    def __post_init__(self):
        for s in (self.approach, self.depart):
            t, n = np.asarray(s.tangent, float), np.asarray(s.normal, float)
            if abs(np.linalg.norm(t) - 1) > 1e-9 or abs(np.linalg.norm(n) - 1) > 1e-9 or abs(t @ n) > 1e-9:
                raise ValueError(f"{self.name}: surface tangent and normal must be "
                                 f"perpendicular unit vectors")
        if abs(_wrap(surface_orientation(self.approach) + self.turn
                     - surface_orientation(self.depart))) > 1e-9:
            raise ValueError(f"{self.name}: turn {np.degrees(self.turn):.1f}° does not take the "
                             f"approach orientation to the departure orientation")
        if self.corner not in ("inside", "outside") or self.lead not in SIDE:
            raise ValueError(f"{self.name}: corner must be inside/outside, lead front/back")


# ── Geometry ──────────────────────────────────────────────────────────────────

def _wrap(a):
    """Wrap angles to [-π, π)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _vec(v):
    return np.asarray(v, dtype=float)


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
    transition point along the surface.  The lowest wheel just touches it.
    """
    n, t = _vec(surface.normal), _vec(surface.tangent)
    W = wheel_offsets(surface_orientation(surface), scn.wg, side)
    height = scn.wg.wheel_r - np.min(W @ n)        # chain joint height above the surface
    return _vec(scn.point) + u * t + height * n


def along(scn, surface, joint):
    """How far a chain joint is from the transition point, measured along `surface`."""
    return float((joint - _vec(scn.point)) @ _vec(surface.tangent))


def pivot_point(scn, side, theta):
    """
    Chain joint of an assembly pivoting at the transition point, at orientation theta.

    Inside corner: the assembly is pressed into both surfaces, so for each
    surface the wheel nearest to it just touches it.  That fixes the joint's
    distance from both surfaces.

    Outside corner or edge: the trailing wheel (the last to leave the approach
    surface) rolls around the transition point.  Its centre stays one wheel
    radius from the point, in the assembly's "up" direction, as if the
    assembly stood on a surface that turns with it.
    """
    O, r = _vec(scn.point), scn.wg.wheel_r
    W = wheel_offsets(theta, scn.wg, side)
    if scn.corner == "inside":
        nA, nB = _vec(scn.approach.normal), _vec(scn.depart.normal)
        rhs = [nA @ O + r - np.min(W @ nA), nB @ O + r - np.min(W @ nB)]
        return np.linalg.solve(np.array([nA, nB]), rhs)
    W_rest = wheel_offsets(surface_orientation(scn.approach), scn.wg, side)
    trailing = int(np.argmax(W_rest @ _vec(scn.approach.tangent)))
    up = np.array([-math.sin(theta), math.cos(theta)])
    return O + r * up - W[trailing]


def roll_to_separation(scn, surface, side, other, separation):
    """
    Chain joint of an assembly resting on `surface` exactly `separation` from
    the chain joint `other`, choosing the position farther from the
    transition point.
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


def _phase_duration(pose, rotation, minimum):
    """
    Seconds for a phase: long enough that neither assembly rolls faster than
    SLIDE_SPEED, a pivot turns no faster than PIVOT_RATE, and at least
    `minimum`.  Rounded up to whole SCHEDULE_STEPs.
    """
    f = np.linspace(0.0, 1.0, 201)
    samples = [pose(x) for x in f]
    peak = 0.0                                     # fastest motion, metres per unit of f
    for k in (0, 2):                               # leading and trailing chain joints
        path = np.array([s[k] for s in samples])
        peak = max(peak, float(np.max(np.linalg.norm(np.diff(path, axis=0), axis=1))) / (f[1] - f[0]))
    seconds = max(peak / SLIDE_SPEED, rotation / PIVOT_RATE, minimum)
    return max(1, math.ceil(seconds / SCHEDULE_STEP - 1e-9)) * SCHEDULE_STEP


def plan_phases(scn, min_durations=None):
    """
    The five phases of a transition.

    min_durations: optional lower bound on each phase's duration (seconds),
    used by the trajectory solver to slow a phase the joints cannot keep up with.
    """
    lead, trail = SIDE[scn.lead], -SIDE[scn.lead]
    trail_name = "back" if scn.lead == "front" else "front"
    A, B = scn.approach, scn.depart
    thA = surface_orientation(A)
    thB = thA + scn.turn
    S, C = TRANSITION_SEPARATION, CRUISE_SEPARATION
    minimum = list(min_durations) if min_durations is not None else [0.0] * 5

    # 1 approach: from the standard start until the leading assembly is in
    #   position to pivot; the trailing one closes up to the crossing gap.
    lead_start = rest_point(scn, A, START_DISTANCE, lead)
    trail_start = rest_point(scn, A, START_DISTANCE + C, trail)
    lead_ready = pivot_point(scn, lead, thA)
    if along(scn, A, lead_ready) >= START_DISTANCE:
        raise ValueError(f"{scn.name}: START_DISTANCE is too short — the leading assembly "
                         f"pivots {along(scn, A, lead_ready):.3f} m from the transition point")
    trail_waiting = roll_to_separation(scn, A, trail, lead_ready, S)

    def approach(f):
        return _lerp(lead_start, lead_ready, f), thA, _lerp(trail_start, trail_waiting, f), thA

    # 2 the leading assembly pivots; the trailing one keeps the gap.
    def lead_pivot(f):
        theta = thA + f * scn.turn
        J = pivot_point(scn, lead, theta)
        return J, theta, roll_to_separation(scn, A, trail, J, S), thA

    # 3 both move: the leading assembly rolls along the new surface until the
    #   gap would be right for the trailing pivot, then the trailing assembly
    #   rolls up to the transition point.  Moving them one after the other keeps
    #   the chain from folding up between the two surfaces.
    lead_across, _, trail_after, _ = lead_pivot(1.0)
    trail_ready = pivot_point(scn, trail, thA)
    lead_waiting = roll_to_separation(scn, B, lead, trail_ready, S)
    lead_leg = np.linalg.norm(lead_waiting - lead_across)
    trail_leg = np.linalg.norm(trail_ready - trail_after)
    split = lead_leg / max(lead_leg + trail_leg, 1e-12)

    def both_move(f):
        if f <= split and split > 0:
            return _lerp(lead_across, lead_waiting, f / split), thB, trail_after, thA
        g = (f - split) / (1 - split) if split < 1 else 1.0
        return lead_waiting, thB, _lerp(trail_after, trail_ready, g), thA

    # 4 the trailing assembly pivots; the leading one keeps the gap.
    def trail_pivot(f):
        theta = thA + f * scn.turn
        T = pivot_point(scn, trail, theta)
        return roll_to_separation(scn, B, lead, T, S), thB, T, theta

    # 5 depart: both roll to the standard end; the gap opens to the cruise gap.
    lead_leaving, _, trail_across, _ = trail_pivot(1.0)
    lead_end = rest_point(scn, B, END_DISTANCE + C, lead)
    trail_end = rest_point(scn, B, END_DISTANCE, trail)
    if along(scn, B, trail_across) >= END_DISTANCE:
        raise ValueError(f"{scn.name}: END_DISTANCE is too short — the trailing assembly "
                         f"finishes its pivot {along(scn, B, trail_across):.3f} m from the point")

    def depart(f):
        return _lerp(lead_leaving, lead_end, f), thB, _lerp(trail_across, trail_end, f), thB

    specs = [("approach", approach, 0.0),
             (f"{scn.lead} pivots", lead_pivot, abs(scn.turn)),
             ("both move", both_move, 0.0),
             (f"{trail_name} pivots", trail_pivot, abs(scn.turn)),
             ("depart", depart, 0.0)]
    return [Phase(name, _phase_duration(fn, rotation, minimum[i]), fn)
            for i, (name, fn, rotation) in enumerate(specs)]


def phase_starts(phases):
    """Start time of each phase plus the end time, shape (len(phases) + 1,)."""
    return np.concatenate([[0.0], np.cumsum([p.duration for p in phases])])


def sample_poses(scn, phases, rate):
    """
    Assembly poses at evenly spaced times through the phases, about `rate`
    frames per second (exactly, when the total duration is a whole number of
    frame intervals).

    Returns (times (N,), poses): each pose is a dict of x1, y1, theta1 (front),
    x2, y2, theta2 (back), constraint_set, wall_x, ceiling_y — the input the
    trajectory solver and the penalty functions expect.
    """
    starts = phase_starts(phases)
    total = starts[-1]
    times = np.linspace(0.0, total, int(round(total * rate)) + 1)
    poses = []
    for t in times:
        i = min(int(np.searchsorted(starts, t, side="right")) - 1, len(phases) - 1)
        f = min(max((t - starts[i]) / phases[i].duration, 0.0), 1.0)
        lead_xy, lead_th, trail_xy, trail_th = phases[i].pose(f)
        if scn.lead == "front":
            (p1, th1), (p2, th2) = (lead_xy, lead_th), (trail_xy, trail_th)
        else:
            (p1, th1), (p2, th2) = (trail_xy, trail_th), (lead_xy, lead_th)
        gap = float(np.hypot(*(p2 - p1)))
        if gap > CHAIN_LENGTH:
            raise ValueError(f"{scn.name}: chain joints {gap:.3f} m apart at t = {t:.2f} s "
                             f"({phases[i].name}), beyond the chain's reach")
        poses.append({"x1": p1[0], "y1": p1[1], "theta1": th1,
                      "x2": p2[0], "y2": p2[1], "theta2": th2,
                      "constraint_set": scn.constraint_set,
                      "wall_x": scn.wall_x, "ceiling_y": scn.ceiling_y})
    return times, poses


# ── Standard posture ──────────────────────────────────────────────────────────

def cruise_posture(scn, pose):
    """
    The standard joint angles at the start and end of every transition: both
    middle joints bent equally (q2 = q3), the arch bending away from the
    surface (elbow branch +1).  Among the configurations at `pose` that satisfy
    this and every hard constraint, returns the one closest to neutral.
    """
    args = (pose["x1"], pose["y1"], pose["theta1"],
            pose["x2"], pose["y2"], pose["theta2"], l1, l2, l3)

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
            pb = evaluate_batch(q[None], *args[:6], l1, l2, l3, scn.wg, pose["constraint_set"],
                                wall_x=pose["wall_x"], ceiling_y=pose["ceiling_y"])
            cost = float(posture_cost(q[None])[0])
            if pb.feasible[0] and (best is None or cost < best[0]):
                best = (cost, q)
    if best is None:
        raise ValueError(f"{scn.name}: no feasible cruise posture at this pose")
    return best[1]


# ── Wheel contact ─────────────────────────────────────────────────────────────

def wheel_spin(scn, poses, times, tol=0.004):
    """
    Spin of each wheel (rad/s, counter-clockwise positive), shape (N, 4):
    front arm A, front arm B, back arm A, back arm B.

    A wheel is rolling on a surface when its centre is one wheel radius from
    the surface line (within tol) and on the surface's side of the transition
    point, in this frame and the previous one, and it is not touching the
    other surface.  Rolling without slipping gives
        ω = (ẑ × n) · Δcentre / (r Δt)
    with n the surface normal.  Wheels at the corner, in the air or touching
    both surfaces get 0.  Frame 0 copies frame 1.
    """
    O, r = _vec(scn.point), scn.wg.wheel_r
    surfaces = [(_vec(s.normal), _vec(s.tangent)) for s in (scn.approach, scn.depart)]

    def touching(w):
        """Index of the single surface the wheel centre w touches, or None."""
        hits = [i for i, (n, t) in enumerate(surfaces)
                if abs(n @ (w - O) - r) <= tol and t @ (w - O) >= -tol]
        return hits[0] if len(hits) == 1 else None

    centres = np.array([[*wheel_centers(p["x1"], p["y1"], p["theta1"], scn.wg, side=1),
                         *wheel_centers(p["x2"], p["y2"], p["theta2"], scn.wg, side=-1)]
                        for p in poses])                       # (N, 4 wheels, 2)
    spin = np.zeros((len(poses), 4))
    for k in range(1, len(poses)):
        dt = times[k] - times[k - 1]
        for j in range(4):
            s_now, s_prev = touching(centres[k, j]), touching(centres[k - 1, j])
            if s_now is not None and s_now == s_prev:
                n = surfaces[s_now][0]
                roll_dir = np.array([-n[1], n[0]])             # ẑ × n
                spin[k, j] = roll_dir @ (centres[k, j] - centres[k - 1, j]) / (r * dt)
    spin[0] = spin[1]
    return spin


# ── The four transitions ──────────────────────────────────────────────────────

SCENARIOS = {
    "floor_to_wall": Scenario(
        name="floor_to_wall", label="Floor to wall (inside corner)",
        point=(0.0, 0.0),
        approach=Surface(tangent=(1, 0), normal=(0, 1)),     # floor, robot at x > 0
        depart=Surface(tangent=(0, 1), normal=(1, 0)),       # wall, robot above the floor
        turn=-np.pi / 2, corner="inside", lead="front", wg=wg,
        constraint_set="wall", wall_x=0.0, ceiling_y=2.0, output="floor_to_wall.mp4"),
    "wall_to_ceiling": Scenario(
        name="wall_to_ceiling", label="Wall to ceiling (inside corner)",
        point=(0.0, 1.5),
        approach=Surface(tangent=(0, -1), normal=(1, 0)),    # wall, robot below the ceiling
        depart=Surface(tangent=(1, 0), normal=(0, -1)),      # ceiling, robot at x > 0
        turn=-np.pi / 2, corner="inside", lead="front", wg=wg,
        constraint_set="ceiling", wall_x=0.0, ceiling_y=1.5, output="wall_to_ceiling.mp4"),
    "outside": Scenario(
        name="outside", label="Outside corner (ceiling top to exterior wall)",
        point=(0.0, 0.55),
        approach=Surface(tangent=(1, 0), normal=(0, 1)),     # top of the block, robot at x > 0
        depart=Surface(tangent=(0, -1), normal=(-1, 0)),     # exterior wall, robot below the top
        turn=+np.pi / 2, corner="outside", lead="front", wg=wg_outside,
        constraint_set="outside", wall_x=0.0, ceiling_y=0.55, output="outside.mp4"),
    "thin_edge": Scenario(
        name="thin_edge", label="Thin edge (top to underside)",
        point=(0.44, 0.50),
        approach=Surface(tangent=(-1, 0), normal=(0, 1)),    # top face, robot left of the end
        depart=Surface(tangent=(-1, 0), normal=(0, -1)),     # underside, robot left of the end
        turn=-np.pi, corner="outside", lead="back", wg=wg_thin_edge,
        constraint_set="thin_edge", wall_x=0.44, ceiling_y=0.50, output="thin_edge.mp4"),
}
