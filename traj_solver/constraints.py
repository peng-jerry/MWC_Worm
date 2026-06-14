"""
Constraint sets for the wheeled-robot IK solver.

Constraint sets
---------------
"none"  : Purely kinematic IK — no physical constraints (useful for debugging).
"floor" : Wheels coincident with floor (y = 0); linkage stays above floor.

The no-self-collision rule (linkage must not intersect wheels or wishbone bars)
is active for every set except "none".
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from kinematics import forward_kinematics


# ------------------------------------------------------------------ #
#  Robot geometry                                                     #
# ------------------------------------------------------------------ #

@dataclass
class WheelGeometry:
    """Physical dimensions of the wheel assemblies."""
    bar_len: float = 0.25       # length of each wishbone arm (joint → wheel)
    wheel_r: float = 0.12       # wheel radius
    spread:  float = np.pi / 4  # half-angle of the V opening


CONSTRAINT_SETS = ("none", "floor", "wall", "ceiling", "outside")


# ------------------------------------------------------------------ #
#  Wheel center helpers (must match visualize._draw_assembly)        #
# ------------------------------------------------------------------ #

def wheel_centers(
    cx: float, cy: float, theta: float, wg: WheelGeometry
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (left, right) wheel center positions for one assembly."""
    c = theta - np.pi / 2          # rearward centreline; down at theta=0
    left  = np.array([cx + wg.bar_len * np.cos(c - wg.spread),
                      cy + wg.bar_len * np.sin(c - wg.spread)])
    right = np.array([cx + wg.bar_len * np.cos(c + wg.spread),
                      cy + wg.bar_len * np.sin(c + wg.spread)])
    return left, right


def floor_y_for_assembly(theta: float, wg: WheelGeometry) -> float:
    """
    Return the y of the assembly joint such that the lowest wheel centre
    is exactly at y = wheel_r  (wheel tangent to the floor at y=0).
    """
    c = theta - np.pi / 2
    y_offsets = [wg.bar_len * np.sin(c - wg.spread),
                 wg.bar_len * np.sin(c + wg.spread)]
    return wg.wheel_r - min(y_offsets)


def wall_x_for_assembly(theta: float, wg: WheelGeometry,
                         wall_x: float = 0.0) -> float:
    """
    Return the x of the assembly joint such that the leftmost wheel centre
    is exactly at x = wall_x + wheel_r  (wheel tangent to the wall).
    """
    c = theta - np.pi / 2
    x_offsets = [wg.bar_len * np.cos(c - wg.spread),
                 wg.bar_len * np.cos(c + wg.spread)]
    return wall_x + wg.wheel_r - min(x_offsets)


def ceiling_y_for_assembly(theta: float, wg: WheelGeometry,
                            ceiling_y: float = 3.0) -> float:
    """
    Return the y of the assembly joint such that the highest wheel centre
    is exactly at y = ceiling_y - wheel_r  (wheel tangent to ceiling from below).
    """
    c = theta - np.pi / 2
    y_offsets = [wg.bar_len * np.sin(c - wg.spread),
                 wg.bar_len * np.sin(c + wg.spread)]
    return ceiling_y - wg.wheel_r - max(y_offsets)


def wall_x_exterior_for_assembly(theta: float, wg: WheelGeometry,
                                  wall_x: float = 0.0) -> float:
    """
    Return the x of the assembly joint such that the rightmost wheel centre
    is exactly at x = wall_x - wheel_r  (wheel tangent to wall from the left,
    i.e. the exterior face of the wall).
    """
    c = theta - np.pi / 2
    x_offsets = [wg.bar_len * np.cos(c - wg.spread),
                 wg.bar_len * np.cos(c + wg.spread)]
    return wall_x - wg.wheel_r - max(x_offsets)


def _surface_contact_adjust(
    x: float, y: float, theta: float,
    wg: WheelGeometry, wall_x: float = 0.0,
    ceiling_y: float = None,
) -> Tuple[float, float]:
    """
    Auto-adjust (x, y) so wheels are tangent to whichever surface(s) they face.

    Detection uses the V's centreline direction c = theta - π/2:
      |sin(c)| > |cos(c)|  AND  sin(c) < 0  →  wheels face down  → floor contact
      |sin(c)| > |cos(c)|  AND  sin(c) > 0  →  wheels face up    → ceiling contact
      |cos(c)| > |sin(c)|  AND  cos(c) < 0  →  wheels face left  → wall contact

    Pass ceiling_y=None  (default) to enable floor detection instead of ceiling.
    Pass ceiling_y=<value> to enable ceiling detection instead of floor.
    Wall detection is always active.
    """
    c = theta - np.pi / 2
    cos_c = np.cos(c)
    sin_c = np.sin(c)

    if ceiling_y is None:
        if abs(sin_c) > abs(cos_c) and sin_c < 0:   # primarily downward → floor
            y = floor_y_for_assembly(theta, wg)
    else:
        if abs(sin_c) > abs(cos_c) and sin_c > 0:   # primarily upward → ceiling
            y = ceiling_y_for_assembly(theta, wg, ceiling_y)

    if abs(cos_c) > abs(sin_c) and cos_c < 0:       # primarily leftward → wall
        x = wall_x_for_assembly(theta, wg, wall_x)

    return x, y


# ------------------------------------------------------------------ #
#  Geometric primitives                                               #
# ------------------------------------------------------------------ #

def _seg_point_dist_sq(a: np.ndarray, b: np.ndarray,
                        p: np.ndarray) -> float:
    """Squared minimum distance from point p to segment a→b."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        d = p - a
        return float(np.dot(d, d))
    t = np.clip(float(np.dot(p - a, ab)) / denom, 0.0, 1.0)
    d = p - (a + t * ab)
    return float(np.dot(d, d))


def _segs_cross_interior(a: np.ndarray, b: np.ndarray,
                          c: np.ndarray, d: np.ndarray,
                          eps: float = 1e-6) -> bool:
    """
    True if segments a→b and c→d cross at a point strictly interior to
    both (both parametric coords in (eps, 1-eps)).  Shared endpoints and
    parallel / collinear pairs return False.
    """
    r = b - a
    s = d - c
    rxs = float(r[0] * s[1] - r[1] * s[0])
    if abs(rxs) < 1e-12:
        return False
    t = float(((c[0]-a[0]) * s[1] - (c[1]-a[1]) * s[0]) / rxs)
    u = float(((c[0]-a[0]) * r[1] - (c[1]-a[1]) * r[0]) / rxs)
    return eps < t < 1 - eps and eps < u < 1 - eps


# ------------------------------------------------------------------ #
#  Angular cone helpers                                               #
# ------------------------------------------------------------------ #

def _wrap(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _cone_penetration(direction: float, center: float, half_width: float) -> float:
    """
    Return how far `direction` is inside a cone centred on `center`
    with half-angle `half_width`.  Positive means inside; zero means outside.
    """
    return max(0.0, half_width - abs(_wrap(direction - center)))


def _v_cone_entry_penalty(
    positions: np.ndarray,
    theta_end: float,
    theta1: float,
    wg: WheelGeometry,
) -> float:
    """
    Penalty for a linkage arm arriving at an apex joint from inside
    the wishbone's angular cone.

    At q1 (P0): the first link P0→P1 must leave the apex pointing
                OUTSIDE the front V's cone.
    At q4 (P3): the last link P2→P3 must arrive at the apex from
                OUTSIDE the back V's cone — checked as the direction
                from P3 toward P2.

    This catches the case (not caught by _segs_cross_interior) where
    the link approaches the apex through the V interior, since the
    only crossing with a V arm happens at the shared endpoint (t=1,u=0)
    which is excluded by the eps guard.
    """
    P = positions
    penalty = 0.0

    # --- Front V at q1 = P0 ---
    front_center = theta1 - np.pi / 2          # cone opens rearward/downward
    dir_01 = np.arctan2(float(P[1][1] - P[0][1]),
                        float(P[1][0] - P[0][0]))
    pen = _cone_penetration(dir_01, front_center, wg.spread)
    penalty += pen ** 2

    # --- Back V at q4 = P3 ---
    back_center = theta_end - np.pi / 2
    # Direction FROM the apex TOWARD P2 (i.e. where the last link comes from)
    dir_3to2 = np.arctan2(float(P[2][1] - P[3][1]),
                          float(P[2][0] - P[3][0]))
    pen = _cone_penetration(dir_3to2, back_center, wg.spread)
    penalty += pen ** 2

    return penalty


# ------------------------------------------------------------------ #
#  Individual penalty terms                                           #
# ------------------------------------------------------------------ #

def _collision_penalty(
    positions: np.ndarray,
    theta_end: float,
    x1: float, y1: float, theta1: float,
    wg: WheelGeometry,
) -> float:
    """
    Penalty for the linkage penetrating any wheel circle or crossing any
    wishbone bar.  Uses the interior-crossing test so shared endpoints
    (q1/q4 are roots of both the linkage and the wishbone) don't trigger
    false positives.
    """
    P = positions  # P[0]=q1 … P[3]=q4

    # Wheel centres
    lf, rf = wheel_centers(x1, y1, theta1, wg)
    lb, rb = wheel_centers(float(P[3][0]), float(P[3][1]), theta_end, wg)
    all_wheels = (lf, rf, lb, rb)

    # Wishbone bars (root, tip)
    bars = [(P[0], lf), (P[0], rf), (P[3], lb), (P[3], rb)]

    # Linkage segments
    segs = [(P[0], P[1]), (P[1], P[2]), (P[2], P[3])]

    penalty = 0.0
    r2 = wg.wheel_r ** 2

    for a, b in segs:
        # --- vs wheel circles ---
        for w in all_wheels:
            d2 = _seg_point_dist_sq(a, b, w)
            if d2 < r2:
                # Smooth quadratic penalty proportional to penetration depth
                penetration = wg.wheel_r - np.sqrt(max(d2, 0.0))
                penalty += penetration ** 2

        # --- vs wishbone bars ---
        # _segs_cross_interior uses eps > 0, so shared endpoints at t=0/u=0
        # are automatically excluded — no false positives at q1 / q4.
        for bar_root, bar_tip in bars:
            if _segs_cross_interior(a, b, bar_root, bar_tip):
                penalty += 1.0  # unit penalty per crossing

    return penalty


def _no_overlap_penalty(
    positions: np.ndarray,
    theta_end: float,
    x1: float, y1: float, theta1: float,
    wg: WheelGeometry,
) -> float:
    """
    Penalty for:
      (a) Any two wheel circles overlapping each other.
      (b) Non-adjacent linkage segments crossing each other
          (only P0→P1 vs P2→P3; adjacent pairs share an endpoint
           and _segs_cross_interior's eps guard already excludes those).
    """
    P = positions

    lf, rf = wheel_centers(x1, y1, theta1, wg)
    lb, rb = wheel_centers(float(P[3][0]), float(P[3][1]), theta_end, wg)
    all_wheels = (lf, rf, lb, rb)

    penalty = 0.0
    min_wheel_dist = 2.0 * wg.wheel_r  # wheels touch when centres are 2r apart

    # --- (a) wheel-wheel overlap ---
    for i in range(len(all_wheels)):
        for j in range(i + 1, len(all_wheels)):
            diff = all_wheels[i] - all_wheels[j]
            dist = float(np.sqrt(np.dot(diff, diff)))
            if dist < min_wheel_dist:
                penalty += (min_wheel_dist - dist) ** 2

    # --- (b) link self-intersection (non-adjacent pair only) ---
    if _segs_cross_interior(P[0], P[1], P[2], P[3]):
        penalty += 1.0

    return penalty


def _floor_penalty(positions: np.ndarray) -> float:
    """
    Penalty for intermediate joints P1, P2 going below the floor (y < 0).
    """
    penalty = 0.0
    for pos in positions[1:3]:
        depth = -float(pos[1])
        if depth > 0.0:
            penalty += depth ** 2
    return penalty


def _wall_penalty(
    positions: np.ndarray,
    theta_end: float,
    x1: float, y1: float, theta1: float,
    wg: WheelGeometry,
    wall_x: float = 0.0,
) -> float:
    """
    Penalty for any wheel centre or intermediate linkage joint penetrating
    the wall (x < wall_x).  Wheel centres must stay at x >= wall_x + wheel_r;
    joints P1/P2 must stay at x >= wall_x.
    """
    P = positions
    lf, rf = wheel_centers(x1, y1, theta1, wg)
    lb, rb = wheel_centers(float(P[3][0]), float(P[3][1]), theta_end, wg)

    penalty = 0.0

    for w in (lf, rf, lb, rb):
        depth = (wall_x + wg.wheel_r) - float(w[0])
        if depth > 0.0:
            penalty += depth ** 2

    for pos in positions[1:3]:          # P1, P2
        depth = wall_x - float(pos[0])
        if depth > 0.0:
            penalty += depth ** 2

    return penalty


def _outside_corner_penalty(
    positions: np.ndarray,
    wall_x: float = 0.0,
    ceiling_y: float = 3.0,
) -> float:
    """
    Penalty for intermediate joints P1/P2 entering either forbidden region:

      1. Solid corner block  (x < wall_x  AND y > ceiling_y)
      2. Room interior       (x > wall_x  AND y < ceiling_y)

    Valid regions for an outside-corner wrap are:
      - Above ceiling (y >= ceiling_y, x >= wall_x): transition from P0
      - Exterior wrap (x <= wall_x,   y <= ceiling_y): transition to P3

    Physically, the linkage must not cross through the ceiling surface
    (going from above to below while still right of the wall) nor through
    the wall surface (going from right to left while below the ceiling).
    """
    penalty = 0.0
    for pos in positions[1:3]:
        x = float(pos[0])
        y = float(pos[1])
        # Solid corner block: x < wall_x AND y > ceiling_y
        x_left  = wall_x - x      # > 0 when left of wall
        y_above = y - ceiling_y   # > 0 when above ceiling
        if x_left > 0.0 and y_above > 0.0:
            penalty += x_left ** 2 + y_above ** 2
        # Room interior: x > wall_x AND y < ceiling_y
        x_right = x - wall_x      # > 0 when right of wall
        y_below = ceiling_y - y   # > 0 when below ceiling
        if x_right > 0.0 and y_below > 0.0:
            penalty += x_right ** 2 + y_below ** 2
    return penalty


def _outside_surface_crossing_penalty(
    positions: np.ndarray,
    wall_x: float = 0.0,
    ceiling_y: float = 3.0,
) -> float:
    """
    Penalty for any link *segment* crossing the two physical surfaces that
    form the outside corner:

      - Ceiling surface: y = ceiling_y  for  x >= wall_x  (horizontal, rightward)
      - Wall surface:    x = wall_x     for  y <= ceiling_y  (vertical, downward)

    For a segment that straddles one surface, the penalty equals the squared
    distance of the crossing point from the corner (wall_x, ceiling_y).
    This is minimised to zero when the segment passes exactly through the
    corner point — the only physically valid transition path.
    """
    penalty = 0.0
    n = len(positions)
    for i in range(n - 1):
        ax, ay = float(positions[i][0]), float(positions[i][1])
        bx, by = float(positions[i + 1][0]), float(positions[i + 1][1])

        # Ceiling line crossing: segment straddles y = ceiling_y
        # Penalise by squared x-distance from the corner.  Minimum = 0 when
        # the segment passes exactly through (wall_x, ceiling_y).
        # This fires whether the crossing is right of wall (physical ceiling)
        # OR left of wall (solid corner extension) — both deviate from the corner.
        ay_rel = ay - ceiling_y
        by_rel = by - ceiling_y
        if ay_rel * by_rel < 0.0:
            t = ay_rel / (ay_rel - by_rel)
            x_cross = ax + t * (bx - ax)
            penalty += (x_cross - wall_x) ** 2

        # Wall line crossing: segment straddles x = wall_x
        # Penalise by squared y-distance from the corner.
        ax_rel = ax - wall_x
        bx_rel = bx - wall_x
        if ax_rel * bx_rel < 0.0:
            t = ax_rel / (ax_rel - bx_rel)
            y_cross = ay + t * (by - ay)
            penalty += (ceiling_y - y_cross) ** 2

    return penalty


def _ceiling_penalty(
    positions: np.ndarray,
    theta_end: float,
    x1: float, y1: float, theta1: float,
    wg: WheelGeometry,
    ceiling_y: float = 3.0,
) -> float:
    """
    Penalty for any wheel centre or intermediate linkage joint penetrating
    the ceiling (y > ceiling_y).  Wheel centres must stay at y <= ceiling_y - wheel_r;
    joints P1/P2 must stay at y <= ceiling_y.
    """
    P = positions
    lf, rf = wheel_centers(x1, y1, theta1, wg)
    lb, rb = wheel_centers(float(P[3][0]), float(P[3][1]), theta_end, wg)

    penalty = 0.0

    for w in (lf, rf, lb, rb):
        depth = float(w[1]) - (ceiling_y - wg.wheel_r)
        if depth > 0.0:
            penalty += depth ** 2

    for pos in positions[1:3]:          # P1, P2
        depth = float(pos[1]) - ceiling_y
        if depth > 0.0:
            penalty += depth ** 2

    return penalty


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #

def total_penalty(
    q: np.ndarray,
    x1: float, y1: float, theta1: float,
    x2: float, y2: float, theta2: float,
    l1: float, l2: float, l3: float,
    wg: WheelGeometry,
    constraint_set: str,
    wall_x: float = 0.0,
    ceiling_y: float = 3.0,
    weight: float = 1e4,
) -> float:
    """
    Weighted constraint penalty for the given joint angles.

    Parameters
    ----------
    constraint_set : {"none", "floor", "wall", "ceiling"}
        Active constraint set.
    wall_x : float
        X-coordinate of the vertical wall (used for "wall" and "ceiling" sets).
    ceiling_y : float
        Y-coordinate of the ceiling (only used for "ceiling" set).
    weight : float
        Overall scaling applied to all penalty terms.
    """
    if constraint_set not in CONSTRAINT_SETS:
        raise ValueError(
            f"Unknown constraint set '{constraint_set}'. "
            f"Valid options: {CONSTRAINT_SETS}"
        )
    if constraint_set == "none":
        return 0.0

    positions, theta_end = forward_kinematics(q, x1, y1, theta1, l1, l2, l3)

    # General constraints
    pen  = _collision_penalty(positions, theta_end, x1, y1, theta1, wg)
    pen += _no_overlap_penalty(positions, theta_end, x1, y1, theta1, wg)
    # V-cone penalty is skipped for "outside": the linkage necessarily exits
    # the apex through the V opening (toward the corner), which is valid.
    if constraint_set != "outside":
        pen += _v_cone_entry_penalty(positions, theta_end, theta1, wg)

    if constraint_set in ("floor", "wall"):
        pen += _floor_penalty(positions)

    if constraint_set in ("wall", "ceiling"):
        pen += _wall_penalty(positions, theta_end, x1, y1, theta1, wg, wall_x)

    if constraint_set == "ceiling":
        pen += _ceiling_penalty(positions, theta_end, x1, y1, theta1, wg, ceiling_y)

    if constraint_set == "outside":
        pen += _outside_corner_penalty(positions, wall_x, ceiling_y)
        pen += _outside_surface_crossing_penalty(positions, wall_x, ceiling_y)

    return weight * pen


# ------------------------------------------------------------------ #
#  Pre-processing: pose adjustment for constrained sets              #
# ------------------------------------------------------------------ #

def apply_floor_constraint(
    x1: float, theta1: float,
    x2: float, theta2: float,
    wg: WheelGeometry,
) -> Tuple[float, float]:
    """
    Compute y1, y2 such that the lowest wheel of each assembly is tangent
    to the floor (y = 0).

    Returns y1, y2.
    """
    return floor_y_for_assembly(theta1, wg), floor_y_for_assembly(theta2, wg)


def apply_wall_constraint(
    x1: float, y1: float, theta1: float,
    x2: float, y2: float, theta2: float,
    wg: WheelGeometry,
    wall_x: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    Auto-adjust (x1, y1) and (x2, y2) so each assembly's wheels are tangent
    to whichever surface(s) they face (floor at y=0, wall at x=wall_x).

    Returns (x1, y1, x2, y2).
    """
    x1, y1 = _surface_contact_adjust(x1, y1, theta1, wg, wall_x)
    x2, y2 = _surface_contact_adjust(x2, y2, theta2, wg, wall_x)
    return x1, y1, x2, y2


def apply_ceiling_constraint(
    x1: float, y1: float, theta1: float,
    x2: float, y2: float, theta2: float,
    wg: WheelGeometry,
    wall_x: float = 0.0,
    ceiling_y: float = 3.0,
) -> Tuple[float, float, float, float]:
    """
    Auto-adjust (x1, y1) and (x2, y2) so each assembly's wheels are tangent
    to whichever surface(s) they face (ceiling at y=ceiling_y, wall at x=wall_x).

      wheels primarily upward  (sin(theta-π/2) > 0) → ceiling contact (y auto-set)
      wheels primarily leftward (cos(theta-π/2) < 0) → wall contact   (x auto-set)

    Returns (x1, y1, x2, y2).
    """
    x1, y1 = _surface_contact_adjust(x1, y1, theta1, wg, wall_x, ceiling_y)
    x2, y2 = _surface_contact_adjust(x2, y2, theta2, wg, wall_x, ceiling_y)
    return x1, y1, x2, y2


def _outside_contact_adjust(
    x: float, y: float, theta: float,
    wg: WheelGeometry, wall_x: float = 0.0, ceiling_y: float = 3.0,
) -> Tuple[float, float]:
    """
    Surface-contact adjustment for the "outside" corner geometry.

    The assembly is on the EXTERIOR of the wall/ceiling corner:
      wheels primarily downward (sin(c) < 0) → resting ON TOP of ceiling
                                                y set so lowest wheel at ceiling_y + wheel_r
      wheels primarily rightward (cos(c) > 0) → pressing against exterior wall face
                                                x set so rightmost wheel at wall_x - wheel_r
    """
    c = theta - np.pi / 2
    cos_c = np.cos(c)
    sin_c = np.sin(c)

    if abs(sin_c) > abs(cos_c) and sin_c < 0:   # wheels face DOWN → on top of ceiling
        y = ceiling_y + floor_y_for_assembly(theta, wg)

    if abs(cos_c) > abs(sin_c) and cos_c > 0:   # wheels face RIGHT → exterior wall face
        x = wall_x_exterior_for_assembly(theta, wg, wall_x)

    return x, y


def apply_outside_constraint(
    x1: float, y1: float, theta1: float,
    x2: float, y2: float, theta2: float,
    wg: WheelGeometry,
    wall_x: float = 0.0,
    ceiling_y: float = 3.0,
) -> Tuple[float, float, float, float]:
    """
    Auto-adjust poses for the exterior corner: ceiling top + exterior wall face.

      wheels primarily downward  (sin(theta-π/2) < 0) → on top of ceiling  (y auto-set)
      wheels primarily rightward (cos(theta-π/2) > 0) → exterior wall face (x auto-set)

    Returns (x1, y1, x2, y2).
    """
    x1, y1 = _outside_contact_adjust(x1, y1, theta1, wg, wall_x, ceiling_y)
    x2, y2 = _outside_contact_adjust(x2, y2, theta2, wg, wall_x, ceiling_y)
    return x1, y1, x2, y2
