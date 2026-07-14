"""
Constraint sets for the wheeled-robot IK solver.

Constraint sets
---------------
"none"         : Purely kinematic IK — no physical constraints (debugging only).
"floor"        : Wheels on floor (y = 0); linkage stays above floor.
"wall"         : Inside floor-wall corner (left wall + floor).
"ceiling"      : Inside wall-ceiling corner (left wall + ceiling).
"outside"      : Exterior top-left corner (ceiling top + left wall exterior);
                 contact position computed from theta by _outside_contact_adjust.
"outside_exact": Same penalties as "outside" but exact (x, y) positions passed
                 through unchanged — used during pivot phases where the contact
                 formula would give the wrong surface.
"thin_edge"    : On top of a thin horizontal edge; wraps around the right terminus.
"thin_edge_exact": Same as thin_edge but positions passed through unchanged.

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
    bar_len: float = 0.140      # arm length from V-apex to each wheel centre
    wheel_r: float = 0.050      # wheel radius
    spread:  float = np.pi / 4  # half-angle of the V opening
    calf:    float = 0.042      # vertical strut from V-apex up to elbow
    thigh:   float = 0.08951      # horizontal strut from elbow to chain joint (q1/q4)


CONSTRAINT_SETS = ("none", "floor", "wall", "wall_exact", "ceiling", "outside", "outside_exact", "thin_edge", "thin_edge_exact")


# ------------------------------------------------------------------ #
#  Wheel center helpers (must match visualize._draw_assembly)        #
# ------------------------------------------------------------------ #

def _v_apex(cx: float, cy: float, theta: float, wg: WheelGeometry,
            side: int = 1) -> Tuple[float, float]:
    """
    Return the V-apex (bar midpoint) position given the chain-joint (cx, cy).

    The L-bracket connecting the chain joint to the V-apex consists of:
      thigh — horizontal in the assembly frame, pointing AWAY from the chain
              (+theta direction for assembly 1 / side=+1; opposite for side=-1)
      calf  — perpendicular to theta, pointing toward the surface
              (direction theta - π/2, i.e. downward when theta=0)

    side=+1  assembly 1 (q1): thigh points backward  (-theta direction from joint)
    side=-1  assembly 2 (q4): thigh points forward   (+theta direction from joint)
    """
    vx = cx - side * wg.thigh * np.cos(theta) + wg.calf * np.sin(theta)
    vy = cy - side * wg.thigh * np.sin(theta) - wg.calf * np.cos(theta)
    return vx, vy


def wheel_centers(
    cx: float, cy: float, theta: float, wg: WheelGeometry, side: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (left, right) wheel center positions for one assembly.

    side=+1 for assembly 1 (q1 joint), side=-1 for assembly 2 (q4 joint).
    """
    vx, vy = _v_apex(cx, cy, theta, wg, side)
    c = theta - np.pi / 2          # rearward centreline; down at theta=0
    left  = np.array([vx + wg.bar_len * np.cos(c - wg.spread),
                      vy + wg.bar_len * np.sin(c - wg.spread)])
    right = np.array([vx + wg.bar_len * np.cos(c + wg.spread),
                      vy + wg.bar_len * np.sin(c + wg.spread)])
    return left, right


def floor_y_for_assembly(theta: float, wg: WheelGeometry, side: int = 1) -> float:
    """
    Return the y of the assembly joint such that the lowest wheel centre
    is exactly at y = wheel_r  (wheel tangent to the floor at y=0).
    """
    c = theta - np.pi / 2
    y_offsets = [wg.bar_len * np.sin(c - wg.spread),
                 wg.bar_len * np.sin(c + wg.spread)]
    y_vapex = wg.wheel_r - min(y_offsets)  # V-apex y that gives floor contact
    return y_vapex + side * wg.thigh * np.sin(theta) + wg.calf * np.cos(theta)


def wall_x_for_assembly(theta: float, wg: WheelGeometry,
                         wall_x: float = 0.0, side: int = 1) -> float:
    """
    Return the x of the assembly joint such that the leftmost wheel centre
    is exactly at x = wall_x + wheel_r  (wheel tangent to the wall).
    """
    c = theta - np.pi / 2
    x_offsets = [wg.bar_len * np.cos(c - wg.spread),
                 wg.bar_len * np.cos(c + wg.spread)]
    x_vapex = wall_x + wg.wheel_r - min(x_offsets)
    return x_vapex + side * wg.thigh * np.cos(theta) - wg.calf * np.sin(theta)


def ceiling_y_for_assembly(theta: float, wg: WheelGeometry,
                            ceiling_y: float = 0.55, side: int = 1) -> float:
    """
    Return the y of the assembly joint such that the highest wheel centre
    is exactly at y = ceiling_y - wheel_r  (wheel tangent to ceiling from below).
    """
    c = theta - np.pi / 2
    y_offsets = [wg.bar_len * np.sin(c - wg.spread),
                 wg.bar_len * np.sin(c + wg.spread)]
    y_vapex = ceiling_y - wg.wheel_r - max(y_offsets)
    return y_vapex + side * wg.thigh * np.sin(theta) + wg.calf * np.cos(theta)


def wall_x_exterior_for_assembly(theta: float, wg: WheelGeometry,
                                  wall_x: float = 0.0, side: int = 1) -> float:
    """
    Return the x of the assembly joint such that the rightmost wheel centre
    is exactly at x = wall_x - wheel_r  (wheel tangent to wall from the left,
    i.e. the exterior face of the wall).
    """
    c = theta - np.pi / 2
    x_offsets = [wg.bar_len * np.cos(c - wg.spread),
                 wg.bar_len * np.cos(c + wg.spread)]
    x_vapex = wall_x - wg.wheel_r - max(x_offsets)
    return x_vapex + side * wg.thigh * np.cos(theta) - wg.calf * np.sin(theta)


def _surface_contact_adjust(
    x: float, y: float, theta: float,
    wg: WheelGeometry, wall_x: float = 0.0,
    ceiling_y: float = None, side: int = 1,
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
            y = floor_y_for_assembly(theta, wg, side)
    else:
        if abs(sin_c) > abs(cos_c) and sin_c > 0:   # primarily upward → ceiling
            y = ceiling_y_for_assembly(theta, wg, ceiling_y, side)

    if abs(cos_c) > abs(sin_c) and cos_c < 0:       # primarily leftward → wall
        x = wall_x_for_assembly(theta, wg, wall_x, side)

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

def _joint_sign_penalty(q: np.ndarray, require_negative: bool = True,
                        constant: float = 1.0) -> float:
    """
    Penalty for q2 or q3 violating their required sign.

    require_negative=True  (wall, ceiling, thin_edge): penalise qi > 0
    require_negative=False (outside): penalise qi < 0

    constant=1.0 (default): hard barrier — even a tiny violation costs 1e4
    constant=0.0: soft quadratic — penalty grows smoothly from 0 at the
                  boundary, letting the solver cross q=0 without a cliff
                  (used for thin_edge_exact so the pivot phases can
                  transition smoothly without branch-switch teleportation)
    """
    penalty = 0.0
    for qi in (float(q[1]), float(q[2])):
        if require_negative and qi > 0.0:
            penalty += constant + qi * qi
        elif not require_negative and qi < 0.0:
            penalty += constant + qi * qi
    return penalty


def _collision_penalty(
    positions: np.ndarray,
    theta_end: float,
    x1: float, y1: float, theta1: float,
    wg: WheelGeometry,
    bar_cross_weight: float = 1.0,
) -> float:
    """
    Penalty for the linkage penetrating any wheel circle or crossing any
    wishbone bar.  Uses the interior-crossing test so shared endpoints
    (q1/q4 are roots of both the linkage and the wishbone) don't trigger
    false positives.
    """
    P = positions  # P[0]=q1 … P[3]=q4

    # Wheel centres (side=+1 for front assembly, side=-1 for back assembly)
    lf, rf = wheel_centers(x1, y1, theta1, wg, side=1)
    lb, rb = wheel_centers(float(P[3][0]), float(P[3][1]), theta_end, wg, side=-1)
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
                penalty += bar_cross_weight

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

    lf, rf = wheel_centers(x1, y1, theta1, wg, side=1)
    lb, rb = wheel_centers(float(P[3][0]), float(P[3][1]), theta_end, wg, side=-1)
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
    lf, rf = wheel_centers(x1, y1, theta1, wg, side=1)
    lb, rb = wheel_centers(float(P[3][0]), float(P[3][1]), theta_end, wg, side=-1)

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
    Penalty for intermediate joints P1/P2 entering the solid block interior.

    Geometry: solid block occupies x >= wall_x AND y <= ceiling_y (lower-right).
    The exterior corner space (x < wall_x AND y > ceiling_y) is valid — the
    chain may route through it when transitioning from ceiling-top to wall.

    Forbidden region: x > wall_x AND y < ceiling_y (inside the solid block).
    """
    penalty = 0.0
    for pos in positions[1:3]:
        x = float(pos[0])
        y = float(pos[1])
        # Solid block interior: x > wall_x AND y < ceiling_y
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
        # The physical ceiling surface exists only for x >= wall_x.  Only
        # penalise when the crossing occurs to the right of (or at) the wall,
        # i.e. the link actually pierces the ceiling material.  Crossings at
        # x < wall_x are in the exterior corner space where no ceiling exists —
        # the corner penalty already guards against solid-block intrusion there.
        ay_rel = ay - ceiling_y
        by_rel = by - ceiling_y
        if ay_rel * by_rel < 0.0:
            t = ay_rel / (ay_rel - by_rel)
            x_cross = ax + t * (bx - ax)
            if x_cross >= wall_x:
                penalty += (x_cross - wall_x) ** 2

        # Wall line crossing: segment straddles x = wall_x.
        # Penalise crossings that occur below the ceiling (y ≤ ceiling_y)
        # by squared distance from the corner (wall_x, ceiling_y).  These
        # crossings hit the exterior wall face, which is solid material.
        # Crossings above the ceiling (y > ceiling_y) pass through open
        # exterior-corner space (no wall material there) — no penalty.
        #
        # Exception: skip the P0→P1 segment (i==0).  When the front assembly
        # is slightly right of wall_x (e.g. still transitioning from ceiling to
        # wall), the first link of the folded configuration must cross x=wall_x
        # briefly — this is a geometry artifact of the finite x1 position and
        # the penalty vanishes naturally as x1 → wall_x.  Penalising it here
        # would give the wide-elbow branch an unfair advantage and prevent the
        # branch switch from happening at the right frame.
        if i > 0:
            ax_rel = ax - wall_x
            bx_rel = bx - wall_x
            if ax_rel * bx_rel < 0.0:
                t = ax_rel / (ax_rel - bx_rel)
                y_cross = ay + t * (by - ay)
                if y_cross <= ceiling_y:  # wall only exists below the ceiling
                    penalty += (y_cross - ceiling_y) ** 2

    return penalty


def _outside_clearance_penalty(
    positions: np.ndarray,
    wall_x: float = 0.0,
    ceiling_y: float = 3.0,
) -> float:
    """
    Soft repulsion keeping P1/P2 away from the two exterior surfaces of the
    outside corner.

    Two valid exterior regions and their nearby surfaces:
      - Above ceiling (y > ceiling_y, x > wall_x): push away from y = ceiling_y
      - Exterior wall (x < wall_x, y < ceiling_y): push away from x = wall_x

    Uses 1/d² with a D_MIN floor so the penalty stays finite at contact.
    """
    D_MIN = 0.01
    penalty = 0.0
    for pos in positions[1:3]:
        x, y = float(pos[0]), float(pos[1])
        if y > ceiling_y and x > wall_x:
            d = max(y - ceiling_y, D_MIN)
            penalty += 1.0 / (d * d)
        if x < wall_x and y < ceiling_y:
            d = max(wall_x - x, D_MIN)
            penalty += 1.0 / (d * d)
    return penalty


def _thin_edge_crossing_penalty(
    positions: np.ndarray,
    edge_x: float = 1.0,
    edge_y: float = 1.5,
) -> float:
    """
    Penalty for any link segment crossing y=edge_y to the left of edge_x.

    The thin horizontal edge extends leftward from its right terminus at
    (edge_x, edge_y).  The only valid crossing path is around the right
    side (x >= edge_x), so any segment that crosses y=edge_y at x < edge_x
    is penalised by the squared x-deficit from the terminus.
    """
    penalty = 0.0
    n = len(positions)
    for i in range(n - 1):
        ax, ay = float(positions[i][0]), float(positions[i][1])
        bx, by = float(positions[i + 1][0]), float(positions[i + 1][1])
        ay_rel = ay - edge_y
        by_rel = by - edge_y
        if ay_rel * by_rel < 0.0:
            t = ay_rel / (ay_rel - by_rel)
            x_cross = ax + t * (bx - ax)
            deficit = edge_x - x_cross
            if deficit > 0.0:
                penalty += deficit ** 2
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
    lf, rf = wheel_centers(x1, y1, theta1, wg, side=1)
    lb, rb = wheel_centers(float(P[3][0]), float(P[3][1]), theta_end, wg, side=-1)

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
    outside_clearance_weight: float = 2.0,
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
    # For outside/thin_edge the linkage must navigate around the corner and
    # will naturally brush the wishbone bars; use a softer bar-crossing weight
    # so smooth continuity is preferred over a large jump to avoid the bar.
    _bar_w = 0.001 if constraint_set in ("outside", "outside_exact", "thin_edge", "thin_edge_exact") else 1.0
    pen  = _collision_penalty(positions, theta_end, x1, y1, theta1, wg, _bar_w)
    pen += _no_overlap_penalty(positions, theta_end, x1, y1, theta1, wg)
    # wall/ceiling: joints must bend inward (q2<0, q3<0)
    # outside: q2 sign omitted — near the wrap boundary at q2≈±180° the
    #   wrapped representation flips sign discontinuously, causing false
    #   rejections.  q3 does NOT flip sign during the outside wrap, so we
    #   enforce q3 < 0 (elbow-up) separately.
    # thin_edge: joints must bend outward (q2>0, q3>0) during normal travel
    #   but the sign flips discontinuously near the 180° wrap; omit for
    #   thin_edge_exact (pivot phases) to avoid false rejections.
    if constraint_set in ("wall", "wall_exact", "ceiling"):
        pen += _joint_sign_penalty(q, require_negative=True)
    elif constraint_set in ("outside", "outside_exact"):
        q3 = float(q[2])
        if q3 > 0.0:
            # "outside_exact" is used during pivot transitions where q3 can
            # legitimately cross 0; use a soft quadratic so the solver can
            # pass through q3=0 without a discontinuous cost cliff.
            # "outside" uses a hard barrier (same as wall/ceiling).
            if constraint_set == "outside_exact":
                pen += q3 * q3
            else:
                pen += 1.0 + q3 * q3
    elif constraint_set == "thin_edge":
        pen += _joint_sign_penalty(q, require_negative=True, constant=1.0)
    elif constraint_set == "thin_edge_exact":
        # Soft quadratic — no constant base so the solver can cross q=0
        # without hitting a cliff, preventing branch-switch teleportation.
        pen += _joint_sign_penalty(q, require_negative=True, constant=0.0)
    # thin_edge / thin_edge_exact: q2 < 0, q3 < 0 preferred (elbow-up arch)
    # V-cone penalty is skipped for outside/thin_edge: the linkage necessarily
    # exits the apex through the V opening toward the surface, which is valid.
    if constraint_set not in ("outside", "outside_exact", "thin_edge", "thin_edge_exact"):
        pen += _v_cone_entry_penalty(positions, theta_end, theta1, wg)

    if constraint_set in ("floor", "wall", "wall_exact"):
        pen += _floor_penalty(positions)

    if constraint_set in ("wall", "wall_exact", "ceiling"):
        pen += _wall_penalty(positions, theta_end, x1, y1, theta1, wg, wall_x)
        _qlim = 2.0 * np.pi / 3.0  # 120°
        for _qi_idx in (0, 3):
            _exc = abs(float(q[_qi_idx])) - _qlim
            if _exc > 0.0:
                pen += _exc * _exc

    if constraint_set == "ceiling":
        pen += _ceiling_penalty(positions, theta_end, x1, y1, theta1, wg, ceiling_y)

    if constraint_set in ("outside", "outside_exact"):
        pen += _outside_corner_penalty(positions, wall_x, ceiling_y)
        pen += _outside_surface_crossing_penalty(positions, wall_x, ceiling_y)
        # Soft quadratic penalty for |q1| or |q4| approaching the physical
        # 135° limit.  Onset at 120° (= 2π/3) so the global search strongly
        # prefers configurations that stay within physical range.
        _q_soft_lim = 2.0 * np.pi / 3.0  # 120°
        for _qi in (float(q[0]), float(q[3])):
            _exc = abs(_qi) - _q_soft_lim
            if _exc > 0.0:
                pen += _exc * _exc

    if constraint_set in ("thin_edge", "thin_edge_exact"):
        # wall_x is repurposed as edge_x (right terminus); ceiling_y as edge_y (edge height).
        pen += _thin_edge_crossing_penalty(positions, wall_x, ceiling_y)

    result = weight * pen
    if constraint_set in ("outside", "outside_exact"):
        result += outside_clearance_weight * _outside_clearance_penalty(
            positions, wall_x, ceiling_y)
    return result


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
    x1, y1 = _surface_contact_adjust(x1, y1, theta1, wg, wall_x, side=1)
    x2, y2 = _surface_contact_adjust(x2, y2, theta2, wg, wall_x, side=-1)
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
    x1, y1 = _surface_contact_adjust(x1, y1, theta1, wg, wall_x, ceiling_y, side=1)
    x2, y2 = _surface_contact_adjust(x2, y2, theta2, wg, wall_x, ceiling_y, side=-1)
    return x1, y1, x2, y2


def _outside_contact_adjust(
    x: float, y: float, theta: float,
    wg: WheelGeometry, wall_x: float = 0.0, ceiling_y: float = 0.55,
    side: int = 1,
) -> Tuple[float, float]:
    """
    Surface-contact adjustment for the "outside" corner geometry.

    Physical pivot rules — no smoothstep blend:
      theta <= 0    : ceiling-top, back (right) wheel sets y
      0 < theta < pi/2:
        - front (left) wheel not yet at wall → back wheel on ceiling sets y
        - front wheel has reached wall       → front wheel on wall sets x
      theta >= pi/2 : wall mode, standard x_exterior formula
    """
    if theta >= np.pi / 2:
        # Both wheels fully wall-facing: use standard wall-exterior positioning.
        x = wall_x_exterior_for_assembly(theta, wg, wall_x, side)
        y = min(y, ceiling_y)
    else:
        # Ceiling or double-contact pivot: enforce back-wheel ceiling contact on y.
        # back (right) wheel: V-apex offset + arm offset gives wheel y.
        # joint_y = ceiling_y + wheel_r - vy_offset - dy_R
        vy_offset = -side * wg.thigh * np.sin(theta) - wg.calf * np.cos(theta)
        c = float(theta) - np.pi / 2
        dy_R = wg.bar_len * float(np.sin(c + wg.spread))
        y = ceiling_y + wg.wheel_r - vy_offset - dy_R

    # Corner-arc clearance: no wheel circle may overlap the solid-block corner.
    # Wheel positions come from V-apex, so compute V-apex first.
    vx, vy = _v_apex(x, y, theta, wg, side)
    c = float(theta) - np.pi / 2
    worst_deficit = 0.0
    best_vdx_w = best_vdy_w = 0.0
    for ws in (-1.0, +1.0):          # ws = wheel side: -1 left, +1 right
        dx_w = wg.bar_len * float(np.cos(c + ws * wg.spread))
        dy_w = wg.bar_len * float(np.sin(c + ws * wg.spread))
        wx = vx + dx_w
        wy = vy + dy_w
        dist = float(np.hypot(wx - wall_x, wy - ceiling_y))
        deficit = wg.wheel_r - dist
        if deficit > worst_deficit:
            worst_deficit = deficit
            best_vdx_w, best_vdy_w = dx_w, dy_w

    if worst_deficit > 0.0:
        wx = vx + best_vdx_w
        wy = vy + best_vdy_w
        dist = float(np.hypot(wx - wall_x, wy - ceiling_y))
        if dist > 1e-9:
            scale = wg.wheel_r / dist
            # Push V-apex so this wheel is at distance wheel_r from corner.
            vx = wall_x + (wx - wall_x) * scale - best_vdx_w
            vy = ceiling_y + (wy - ceiling_y) * scale - best_vdy_w
            # Recover joint position from new V-apex.
            x = vx + side * wg.thigh * np.cos(theta) - wg.calf * np.sin(theta)
            y = vy + side * wg.thigh * np.sin(theta) + wg.calf * np.cos(theta)

    # Clamp: assembly centre must not enter solid block (x < wall_x AND y > ceiling_y).
    if x < wall_x and y > ceiling_y:
        x = wall_x

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
    x1, y1 = _outside_contact_adjust(x1, y1, theta1, wg, wall_x, ceiling_y, side=1)
    x2, y2 = _outside_contact_adjust(x2, y2, theta2, wg, wall_x, ceiling_y, side=-1)
    return x1, y1, x2, y2


def _thin_edge_contact_adjust(
    x: float, y: float, theta: float,
    wg: WheelGeometry, edge_y: float = 0.40, edge_x: float = np.inf,
    side: int = 1,
) -> Tuple[float, float]:
    """
    Surface-contact adjustment for the thin-edge constraint.

    Continuous blend from L-wheel-on-top to R-wheel-on-bottom using sin(c)
    as the blend parameter.  sin(c) = -1 when V points down (top surface),
    +1 when V points up (bottom surface).  Smoothstep avoids the three
    discontinuities that the old pivot-branching logic produced.
    """
    c = theta - np.pi / 2
    sin_c = float(np.sin(c))

    dy_L = wg.bar_len * float(np.sin(c - wg.spread))
    dy_R = wg.bar_len * float(np.sin(c + wg.spread))

    # vy_correction converts V-apex y → joint y
    vy_corr = side * wg.thigh * float(np.sin(theta)) + wg.calf * float(np.cos(theta))

    y_L_top = edge_y + wg.wheel_r - dy_L + vy_corr   # joint y: L wheel on TOP surface
    y_R_bot = edge_y - wg.wheel_r - dy_R + vy_corr   # joint y: R wheel on BOTTOM surface

    # Smoothstep blend: 0 → top (sin_c = -1), 1 → bottom (sin_c = +1)
    t = (sin_c + 1.0) * 0.5
    w = t * t * (3.0 - 2.0 * t)
    y = y_L_top * (1.0 - w) + y_R_bot * w

    return x, y


def apply_thin_edge_constraint(
    x1: float, y1: float, theta1: float,
    x2: float, y2: float, theta2: float,
    wg: WheelGeometry,
    edge_y: float = 1.5,
    edge_x: float = np.inf,
) -> Tuple[float, float, float, float]:
    """
    Auto-adjust poses for the thin-edge constraint.

    The thin horizontal edge is at y=edge_y with its right terminus at x=edge_x.
    The linkage wraps around the terminus; assemblies are on opposite faces.

    Uses pivot-wheel contact: the wheel that remains on the flat surface side
    (x ≤ edge_x) defines the assembly height.  See _thin_edge_contact_adjust.

    x is not modified.  Returns (x1, y1, x2, y2).
    """
    x1, y1 = _thin_edge_contact_adjust(x1, y1, theta1, wg, edge_y, edge_x, side=1)
    x2, y2 = _thin_edge_contact_adjust(x2, y2, theta2, wg, edge_y, edge_x, side=-1)
    return x1, y1, x2, y2
