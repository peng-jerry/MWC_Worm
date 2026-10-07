"""
constraints_batch.py — batched constraint evaluation for the IK solvers.

evaluate_batch() scores K candidate joint vectors for one pair of assembly
poses in a single numpy pass (the solvers score hundreds of candidates per
frame, so a per-candidate Python loop would dominate the run time).  Its
total reproduces constraints.total_penalty for every candidate, and it also
returns:

  - the raw value of every penalty term, so callers can see why a candidate
    costs what it does, and
  - whether each term is hard (must be ~zero for a physically valid pose) or
    soft (a preference), so callers can tell infeasible from merely expensive.

Penalty terms (raw values; total = 1e4 × Σ terms except outside_clearance,
which is weighted 2.0):

  wheel_penetration         links entering a wheel circle: Σ depth² (m²)
  bar_crossing              links crossing a wishbone bar: count (× 0.001 on
                            the outside/thin_edge sets, where the chain wraps
                            around the corner and may brush the bars)
  wheel_overlap             two wheels overlapping: Σ overlap² (m²)
  link_self_crossing        first and last links crossing: count
  bend_sign                 q2/q3 bending the wrong way: 1 + q² per joint
                            (q² only on the *_exact sets)
  v_cone                    end links leaving through a wishbone's V: rad²
  floor / wall / ceiling    wheels or P1/P2 inside that surface: Σ depth² (m²)
  outside_corner            P1/P2 inside the outside corner's solid block (m²)
  outside_surface_crossing  links crossing the ceiling top / exterior wall
                            away from the corner point (m²)
  thin_edge_crossing        links crossing the thin edge left of its terminus (m²)
  joint_soft_limit          |q_display| beyond 90°: rad² (× 8 on wall/ceiling sets)
  joint_hard_limit          |q_display| beyond 105°: 1 + rad² per joint
  outside_clearance         1/d² push of P1/P2 away from the outside-corner
                            surfaces (always > 0 near them)

constraints.py holds the scalar reference implementation of every term.  Any
change to a term must be made in both places; test_constraints_batch.py
checks that they agree.
"""

from dataclasses import dataclass

import numpy as np

from constraints import CONSTRAINT_SETS, WheelGeometry, wheel_centers

# A hard term counts as violated above this raw value.  Geometric terms are
# squared depths (m²), so this is ≈1 mm of penetration; crossing, sign and
# hard-limit terms carry a 1.0 step and always exceed it.
HARD_TOL = 1e-6

_OUTSIDE   = ("outside", "outside_exact")
_THIN      = ("thin_edge", "thin_edge_exact")
_WALL_CEIL = ("wall", "wall_exact", "ceiling", "ceiling_exact")


def active_terms(constraint_set: str) -> dict:
    """Ordered {term name: "hard" | "soft"} for the terms a constraint set uses.

    Classification follows the existing penalty design: terms the original
    code deliberately softened (wishbone-bar crossings on outside/thin_edge,
    the bend sign on the *_exact sets) stay soft.
    """
    if constraint_set not in CONSTRAINT_SETS:
        raise ValueError(f"Unknown constraint set '{constraint_set}'. "
                         f"Valid options: {CONSTRAINT_SETS}")
    if constraint_set == "none":
        return {}

    wraps = constraint_set in _OUTSIDE + _THIN
    terms = {
        "wheel_penetration":  "hard",
        "bar_crossing":       "soft" if wraps else "hard",
        "wheel_overlap":      "hard",
        "link_self_crossing": "hard",
    }
    if constraint_set in _WALL_CEIL or constraint_set in ("outside", "thin_edge"):
        terms["bend_sign"] = "hard"
    elif constraint_set in ("outside_exact", "thin_edge_exact"):
        terms["bend_sign"] = "soft"
    if not wraps:
        terms["v_cone"] = "hard"
    if constraint_set in ("floor", "wall", "wall_exact"):
        terms["floor"] = "hard"
    if constraint_set in ("wall", "wall_exact", "ceiling"):
        terms["wall"] = "hard"
    if constraint_set in ("ceiling", "ceiling_exact"):
        terms["ceiling"] = "hard"
    if constraint_set in _OUTSIDE:
        terms["outside_corner"] = "hard"
        terms["outside_surface_crossing"] = "hard"
    if constraint_set in _THIN:
        terms["thin_edge_crossing"] = "hard"
    terms["joint_soft_limit"] = "soft"
    terms["joint_hard_limit"] = "hard"
    if constraint_set in _OUTSIDE:
        terms["outside_clearance"] = "soft"
    return terms


@dataclass
class PenaltyBatch:
    """Penalty terms for K candidates of one pose."""
    total: np.ndarray   # (K,) equals constraints.total_penalty per candidate
    terms: dict         # term name -> (K,) raw value (before the overall weight)
    kinds: dict         # term name -> "hard" | "soft"

    @property
    def hard_names(self):
        return [n for n, k in self.kinds.items() if k == "hard"]

    @property
    def hard_violation(self) -> np.ndarray:
        """(K,) sum of the raw hard terms."""
        out = np.zeros(len(self.total))
        for n in self.hard_names:
            out += self.terms[n]
        return out

    @property
    def feasible(self) -> np.ndarray:
        """(K,) True where every hard term is within HARD_TOL."""
        ok = np.ones(len(self.total), dtype=bool)
        for n in self.hard_names:
            ok &= self.terms[n] <= HARD_TOL
        return ok

    def violations(self, i: int) -> dict:
        """{hard term: raw value} for the hard terms candidate i violates."""
        return {n: float(self.terms[n][i]) for n in self.hard_names
                if self.terms[n][i] > HARD_TOL}


# ------------------------------------------------------------------ #
#  Vectorised geometry (rows are candidates; points are (K, 2))      #
# ------------------------------------------------------------------ #

def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _fk_batch(Q, x1, y1, theta1, l1, l2, l3):
    """Joint positions P0..P3 (each (K, 2)) and back-assembly orientation (K,)."""
    phi1 = theta1 + Q[:, 0]
    phi2 = phi1 + Q[:, 1]
    phi3 = phi2 + Q[:, 2]
    p0 = np.broadcast_to(np.array([x1, y1], dtype=float), (len(Q), 2))
    p1 = p0 + l1 * np.stack([np.cos(phi1), np.sin(phi1)], axis=1)
    p2 = p1 + l2 * np.stack([np.cos(phi2), np.sin(phi2)], axis=1)
    p3 = p2 + l3 * np.stack([np.cos(phi3), np.sin(phi3)], axis=1)
    return (p0, p1, p2, p3), phi3 + Q[:, 3]


def _wheel_centers_batch(cx, cy, theta, wg: WheelGeometry, side: int):
    """constraints.wheel_centers for arrays of joint positions and orientations."""
    vx = cx + wg.calf * np.sin(theta)
    vy = cy - wg.calf * np.cos(theta)
    c = theta - np.pi / 2
    aa, ab = (wg.arm_a1, wg.arm_b1) if side == 1 else (wg.arm_a2, wg.arm_b2)
    arm_a = np.stack([vx + wg.bar_len_a * np.cos(c + aa),
                      vy + wg.bar_len_a * np.sin(c + aa)], axis=1)
    arm_b = np.stack([vx + wg.bar_len * np.cos(c + ab),
                      vy + wg.bar_len * np.sin(c + ab)], axis=1)
    return arm_a, arm_b


def _seg_point_dist_sq(a, b, p):
    """Squared distance from points p to segments a→b."""
    abx = b[:, 0] - a[:, 0]
    aby = b[:, 1] - a[:, 1]
    apx = p[:, 0] - a[:, 0]
    apy = p[:, 1] - a[:, 1]
    denom = abx * abx + aby * aby
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
    t = np.where(denom < 1e-12, 0.0, t)
    dx = p[:, 0] - (a[:, 0] + t * abx)
    dy = p[:, 1] - (a[:, 1] + t * aby)
    return dx * dx + dy * dy


def _segs_cross_interior(a, b, c, d, eps=1e-6):
    """True where segments a→b and c→d cross strictly inside both."""
    rx, ry = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    sx, sy = d[:, 0] - c[:, 0], d[:, 1] - c[:, 1]
    cax, cay = c[:, 0] - a[:, 0], c[:, 1] - a[:, 1]
    rxs = rx * sy - ry * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (cax * sy - cay * sx) / rxs
        u = (cax * ry - cay * rx) / rxs
    return ((np.abs(rxs) >= 1e-12)
            & (eps < t) & (t < 1 - eps) & (eps < u) & (u < 1 - eps))


def _sq_pos(x):
    """x² where x > 0, else 0."""
    return np.where(x > 0.0, x * x, 0.0)


def _line_crossing(a, b, axis, level):
    """Where segments a→b straddle coordinate `axis` == level: (mask, other coordinate there)."""
    other = 1 - axis
    a_rel = a[:, axis] - level
    b_rel = b[:, axis] - level
    mask = a_rel * b_rel < 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = a_rel / (a_rel - b_rel)
    return mask, a[:, other] + t * (b[:, other] - a[:, other])


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #

def evaluate_batch(
    Q: np.ndarray,
    x1: float, y1: float, theta1: float,
    x2: float, y2: float, theta2: float,
    l1: float, l2: float, l3: float,
    wg: WheelGeometry,
    constraint_set: str,
    wall_x: float = 0.0,
    ceiling_y: float = 3.0,
    weight: float = 1e4,
    outside_clearance_weight: float = 2.0,
) -> PenaltyBatch:
    """
    Evaluate every penalty term for the candidate joint vectors Q (K, 4).

    Arguments match constraints.total_penalty; x2, y2, theta2 are accepted
    for the same signature (the back assembly pose comes from the chain).
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    K = len(Q)
    kinds = active_terms(constraint_set)
    if not kinds:
        return PenaltyBatch(np.zeros(K), {}, {})

    (p0, p1, p2, p3), theta_end = _fk_batch(Q, x1, y1, theta1, l1, l2, l3)
    lf, rf = (np.broadcast_to(w, (K, 2))
              for w in wheel_centers(x1, y1, theta1, wg, side=1))
    lb, rb = _wheel_centers_batch(p3[:, 0], p3[:, 1], theta_end, wg, side=-1)
    wheels = (lf, rf, lb, rb)
    segs = ((p0, p1), (p1, p2), (p2, p3))
    mids = (p1, p2)                    # intermediate joints checked against surfaces
    r = wg.wheel_r
    terms = {}

    # --- Linkage vs wheels and wishbone bars (constraints._collision_penalty) ---
    pen = np.zeros(K)
    bars = np.zeros(K)
    for a, b in segs:
        for w in wheels:
            d2 = _seg_point_dist_sq(a, b, w)
            pen += np.where(d2 < r * r, (r - np.sqrt(np.maximum(d2, 0.0))) ** 2, 0.0)
        for root, tip in ((p0, lf), (p0, rf), (p3, lb), (p3, rb)):
            bars += _segs_cross_interior(a, b, root, tip)
    terms["wheel_penetration"] = pen
    terms["bar_crossing"] = (0.001 if constraint_set in _OUTSIDE + _THIN else 1.0) * bars

    # --- Wheel overlap and link self-crossing (constraints._no_overlap_penalty) ---
    overlap = np.zeros(K)
    for i in range(4):
        for j in range(i + 1, 4):
            diff = wheels[i] - wheels[j]
            dist = np.sqrt(diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1])
            overlap += np.where(dist < 2.0 * r, (2.0 * r - dist) ** 2, 0.0)
    terms["wheel_overlap"] = overlap
    terms["link_self_crossing"] = _segs_cross_interior(p0, p1, p2, p3).astype(float)

    # --- Bend direction (constraints._joint_sign_penalty and the outside q3 rule) ---
    q2, q3 = Q[:, 1], Q[:, 2]
    if constraint_set in _WALL_CEIL or constraint_set in _THIN:
        const = 0.0 if constraint_set == "thin_edge_exact" else 1.0
        terms["bend_sign"] = (np.where(q2 > 0.0, const + q2 * q2, 0.0)
                              + np.where(q3 > 0.0, const + q3 * q3, 0.0))
    elif constraint_set in _OUTSIDE:
        const = 0.0 if constraint_set == "outside_exact" else 1.0
        terms["bend_sign"] = np.where(q3 > 0.0, const + q3 * q3, 0.0)

    # --- Links entering a wishbone cone (constraints._v_cone_entry_penalty) ---
    if "v_cone" in kinds:
        c1 = theta1 - np.pi / 2
        dir_01 = np.arctan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
        front = np.maximum(0.0, abs(wg.arm_b1 - wg.arm_a1) / 2.0
                           - np.abs(_wrap(dir_01 - (c1 + (wg.arm_a1 + wg.arm_b1) / 2.0))))
        c2 = theta_end - np.pi / 2
        dir_32 = np.arctan2(p2[:, 1] - p3[:, 1], p2[:, 0] - p3[:, 0])
        back = np.maximum(0.0, abs(wg.arm_b2 - wg.arm_a2) / 2.0
                          - np.abs(_wrap(dir_32 - (c2 + (wg.arm_a2 + wg.arm_b2) / 2.0))))
        terms["v_cone"] = front ** 2 + back ** 2

    # --- Environment surfaces ---
    if "floor" in kinds:
        terms["floor"] = sum(_sq_pos(-p[:, 1]) for p in mids)
    if "wall" in kinds:
        terms["wall"] = (sum(_sq_pos((wall_x + r) - w[:, 0]) for w in wheels)
                         + sum(_sq_pos(wall_x - p[:, 0]) for p in mids))
    if "ceiling" in kinds:
        terms["ceiling"] = (sum(_sq_pos(w[:, 1] - (ceiling_y - r)) for w in wheels)
                            + sum(_sq_pos(p[:, 1] - ceiling_y) for p in mids))
    if "outside_corner" in kinds:
        corner = np.zeros(K)
        for p in mids:
            x_right = p[:, 0] - wall_x
            y_below = ceiling_y - p[:, 1]
            corner += np.where((x_right > 0.0) & (y_below > 0.0),
                               x_right ** 2 + y_below ** 2, 0.0)
        terms["outside_corner"] = corner
    if "outside_surface_crossing" in kinds:
        crossing = np.zeros(K)
        for i, (a, b) in enumerate(segs):
            hit, x_cross = _line_crossing(a, b, axis=1, level=ceiling_y)
            crossing += np.where(hit & (x_cross >= wall_x), (x_cross - wall_x) ** 2, 0.0)
            if i > 0:   # P0→P1 is exempt from the wall test, as in the scalar code
                hit, y_cross = _line_crossing(a, b, axis=0, level=wall_x)
                crossing += np.where(hit & (y_cross <= ceiling_y),
                                     (y_cross - ceiling_y) ** 2, 0.0)
        terms["outside_surface_crossing"] = crossing
    if "thin_edge_crossing" in kinds:
        # wall_x is repurposed as edge_x (right terminus); ceiling_y as edge_y.
        crossing = np.zeros(K)
        for a, b in segs:
            hit, x_cross = _line_crossing(a, b, axis=1, level=ceiling_y)
            crossing += np.where(hit, _sq_pos(wall_x - x_cross), 0.0)
        terms["thin_edge_crossing"] = crossing

    # --- Joint limits, display convention (constraints._joint_limit_penalty and
    #     constraints._joint_hard_limit_penalty) ---
    shift = np.array([-np.pi / 2, 0.0, 0.0, -np.pi / 2])
    soft = np.abs(Q + shift) - np.radians(90.0)
    lim_mult = 8.0 if constraint_set in _WALL_CEIL else 1.0
    terms["joint_soft_limit"] = lim_mult * np.sum(_sq_pos(soft), axis=1)
    hard = np.abs(_wrap(Q + shift)) - np.radians(105.0)
    terms["joint_hard_limit"] = np.sum(np.where(hard > 0.0, 1.0 + hard * hard, 0.0), axis=1)

    # --- Outside-corner clearance (constraints._outside_clearance_penalty) ---
    if "outside_clearance" in kinds:
        clear = np.zeros(K)
        for p in mids:
            x, y = p[:, 0], p[:, 1]
            above = (y > ceiling_y) & (x > wall_x)
            beside = (x < wall_x) & (y < ceiling_y)
            d_above = np.maximum(y - ceiling_y, 0.01)
            d_beside = np.maximum(wall_x - x, 0.01)
            clear += np.where(above, 1.0 / (d_above * d_above), 0.0)
            clear += np.where(beside, 1.0 / (d_beside * d_beside), 0.0)
        terms["outside_clearance"] = clear

    # Same weighting as total_penalty: everything but the clearance term is
    # scaled by `weight`; the clearance term has its own weight.
    pen_sum = np.zeros(K)
    for name in kinds:
        if name != "outside_clearance":
            pen_sum += terms[name]
    total = weight * pen_sum
    if "outside_clearance" in kinds:
        total = total + outside_clearance_weight * terms["outside_clearance"]

    return PenaltyBatch(total, terms, kinds)
