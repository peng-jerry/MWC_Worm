"""
constraints.py — the robot's physical outline and the rules every
configuration must satisfy.

Robot model
-----------
Side view, with every part in one plane.  Each part is a capsule: a segment
thickened by a radius (a wheel is a capsule of zero length).

  chain      links l1, l2, l3 between the joints P0..P3, each 30 mm wide.
             The links are 27 mm bars with joints 15 mm in radius; the
             joint size is used along the whole link (1.5 mm conservative
             per side), so the links' ends are the joints.
  assembly   for each wheel assembly (front at P0, back at P3):
               calf             chain joint → V-apex
               block            the black block across the calf
               plate            the top of the wheel plate
               arm A, arm B     V-apex → each wheel
               wheel A, wheel B
             The calf, arms and wheels follow WheelGeometry, so a
             rearranged arm layout carries its outline with it.  Part
             widths were read off the CAD side view (constants below).

What can touch what (from the robot's construction)
---------------------------------------------------
  * Parts that meet at a joint sit in different layers and never collide:
    l1–l2, l2–l3, and l1 / l3 with the calf they are mounted on.
  * l1 and l3 collide.  Their ends are the joints, so this also covers
    every pair of joints that can meet.
  * Every link collides with every other part of either assembly.
  * The two assemblies collide with each other and with the surfaces.
    Both assemblies are placed by the motion plan, so that is checked
    there (scenarios.check_assemblies), not per configuration.
  * Parts on opposite faces of a plate cannot reach each other through it:
    their gap is measured around the plate (environment.separation).

Rules for one configuration (evaluate)
--------------------------------------
  hard   clearance ≥ HARD_CLEARANCE: chain ↔ surfaces, l1 ↔ l3, links ↔
         either assembly
         |joint| ≤ JOINT_HARD_LIMIT_DEG (display convention)
         q2, q3 ≤ 0: the chain arches away from the surface
  soft   clearance below SOFT_CLEARANCE, and |joint| beyond
         JOINT_SOFT_LIMIT_DEG.  Each is scored by how far it has gone from
         the soft value toward the hard one, as a fraction 0 … 1, squared —
         so every soft term is unitless and on the same scale.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from kinematics import q_display


# ------------------------------------------------------------------ #
#  Wheel assembly geometry                                            #
# ------------------------------------------------------------------ #

# Measured assembly dimensions (side view, theta=0, both wheels on the floor):
#
#              spine axis (chain joint)
#                  |
#                  |  calf
#                  |
#                V-apex
#               /   \
#        arm B /     \ arm A
#        45°  /       \
#           (B)-------(A)
#            |<-63.0->|<-20.19->|   (apex is 20.19 mm behind wheel A)
#            |<----- 83.19 ---->|
#
# SPINE_HEIGHT is spine axis → wheel-centre line; the V-apex sits
# APEX_HEIGHT above the wheel centres, the calf covers the rest.
SPINE_HEIGHT = 0.17685                              # spine axis to wheel centre (m)
WHEELBASE    = 0.08319                              # wheel centre-to-centre (m)
ARM_A_DX     = 0.02019                              # V-apex to arm A wheel, horizontal (m)
ARM_B_ANGLE  = np.pi / 4                            # arm B elevation at its wheel (45°)
ARM_B_DX     = WHEELBASE - ARM_A_DX                 # V-apex to arm B wheel, horizontal (0.063 m)
APEX_HEIGHT  = ARM_B_DX * np.tan(ARM_B_ANGLE)       # V-apex above wheel centres (0.063 m)
CALF_LEN     = SPINE_HEIGHT - APEX_HEIGHT           # 0.11385 m
BAR_LEN_A    = np.hypot(ARM_A_DX, APEX_HEIGHT)      # ≈ 0.06616 m
BAR_LEN_B    = np.hypot(ARM_B_DX, APEX_HEIGHT)      # ≈ 0.08910 m
ARM_A_OFFSET = np.arctan2(ARM_A_DX, APEX_HEIGHT)    # ≈ 17.77° from vertical
ARM_B_OFFSET = np.arctan2(ARM_B_DX, APEX_HEIGHT)    # = 45°   from vertical


@dataclass
class WheelGeometry:
    """Physical dimensions of the wheel assemblies."""
    bar_len:   float = BAR_LEN_B  # arm B length from V-apex to wheel centre
    wheel_r:   float = 0.025      # wheel radius (50 mm diameter)
    # Per-assembly arm offsets from centreline c = theta − π/2.
    # Wheel direction from axle = c + arm_angle.  Positive → +x at θ=0; negative → −x.
    # Arm A and arm B always lean to opposite sides of the centreline.
    arm_a1: float = -ARM_A_OFFSET    # front assembly, arm A (~−17.8°, outboard at θ=0)
    arm_b1: float =  ARM_B_OFFSET    # front assembly, arm B (+45°, inboard at θ=0)
    arm_a2: float =  ARM_A_OFFSET    # back assembly,  arm A (~+17.8°, outboard at θ=0)
    arm_b2: float = -ARM_B_OFFSET    # back assembly,  arm B (−45°, inboard at θ=0)
    calf:   float = CALF_LEN     # strut from chain joint to V-apex (perpendicular to theta)
    bar_len_a: float = BAR_LEN_A # arm A length; pass None to auto-derive from bar_len/arm_b1/arm_a1

    def __post_init__(self):
        if self.bar_len_a is None:
            # arm_a length that puts the arm_a wheel at the same surface height as
            # arm_b at theta=0 (equal y-projection → simultaneous floor contact):
            #   bar_len_a * cos(arm_a1) = bar_len * cos(arm_b1)
            self.bar_len_a = self.bar_len * abs(np.cos(self.arm_b1)) / abs(np.cos(self.arm_a1))


def wheel_centers(
    cx: float, cy: float, theta: float, wg: WheelGeometry, side: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (arm_a, arm_b) wheel center positions for one assembly.

    The calf runs from the chain joint (cx, cy) to the V-apex, perpendicular
    to theta (straight down at theta = 0); the arms leave the apex at their
    offsets from c = theta − π/2.  side=+1 for assembly 1 (q1 joint),
    side=-1 for assembly 2 (q4 joint).
    """
    vx = cx + wg.calf * np.sin(theta)
    vy = cy - wg.calf * np.cos(theta)
    c = theta - np.pi / 2
    aa, ab = (wg.arm_a1, wg.arm_b1) if side == 1 else (wg.arm_a2, wg.arm_b2)
    arm_a = np.array([vx + wg.bar_len_a * np.cos(c + aa),
                      vy + wg.bar_len_a * np.sin(c + aa)])
    arm_b = np.array([vx + wg.bar_len   * np.cos(c + ab),
                      vy + wg.bar_len   * np.sin(c + ab)])
    return arm_a, arm_b


# ------------------------------------------------------------------ #
#  Physical outline (side view, from the CAD model)                   #
# ------------------------------------------------------------------ #

JOINT_RADIUS = 0.015      # m; links are drawn and checked as 30 mm-wide capsules
CALF_RADIUS  = 0.016      # m; calf 32 mm wide
# Bars across the calf: (depth of their centre line below the chain joint,
# half the length of that line, radius).  Overall size in the comments.
BLOCK = (0.071, 0.020, 0.017)   # black block: 74 × 34 mm, 54–88 mm below the joint
PLATE = (0.108, 0.015, 0.018)   # top of the wheel plate: 66 × 36 mm, 90–126 mm below
ARM_A_RADIUS = 0.018      # m; the wheel plate along arm A, 36 mm wide
ARM_B_RADIUS = 0.010      # m; arm B, 20 mm wide


def assembly_parts(x, y, theta, wg: WheelGeometry, side: int) -> dict:
    """
    Capsules (end a, end b, radius) of one wheel assembly whose chain joint
    is at (x, y) with orientation theta, keyed by part name.
    """
    joint = np.array([x, y], dtype=float)
    down = np.array([np.sin(theta), -np.cos(theta)])        # chain joint → V-apex
    across = np.array([np.cos(theta), np.sin(theta)])
    apex = joint + wg.calf * down
    wheel_a, wheel_b = wheel_centers(x, y, theta, wg, side)

    def bar(depth, half, radius):
        centre = joint + depth * down
        return centre - half * across, centre + half * across, radius

    return {
        "calf":    (joint, apex, CALF_RADIUS),
        "block":   bar(*BLOCK),
        "plate":   bar(*PLATE),
        "arm A":   (apex, wheel_a, ARM_A_RADIUS),
        "arm B":   (apex, wheel_b, ARM_B_RADIUS),
        "wheel A": (wheel_a, wheel_a, wg.wheel_r),
        "wheel B": (wheel_b, wheel_b, wg.wheel_r),
    }


def chain_points(Q, x1, y1, theta1, l1, l2, l3):
    """Joint positions P0..P3, each (K, 2), for candidate rows Q (K, 4)."""
    Q = np.atleast_2d(Q)
    phi1 = theta1 + Q[:, 0]
    phi2 = phi1 + Q[:, 1]
    phi3 = phi2 + Q[:, 2]
    p0 = np.broadcast_to(np.array([x1, y1], dtype=float), (len(Q), 2))
    p1 = p0 + l1 * np.stack([np.cos(phi1), np.sin(phi1)], axis=1)
    p2 = p1 + l2 * np.stack([np.cos(phi2), np.sin(phi2)], axis=1)
    p3 = p2 + l3 * np.stack([np.cos(phi3), np.sin(phi3)], axis=1)
    return p0, p1, p2, p3


# ------------------------------------------------------------------ #
#  Limits                                                             #
# ------------------------------------------------------------------ #

# Joint limits in the display convention (kinematics.q_display: q1/q4 0° =
# link straight up from the calf, q2/q3 0° = straight).  Set only here —
# the solver, the playback check, check_ik.py and the plots all use these.
JOINT_SOFT_LIMIT_DEG = 90.0    # beyond this the solver pays a growing cost
JOINT_HARD_LIMIT_DEG = 110.0   # never exceeded (the ±110° stop also keeps
                               # each motor off the calf)

# Gap between the outlines of any two parts that can collide, and between
# the chain and the surfaces.
HARD_CLEARANCE = 0.015    # m, never less
SOFT_CLEARANCE = 0.050    # m, less than this costs

FEASIBILITY_TOL = 1e-3    # m or rad by which a hard rule may be missed (rounding)

CLEARANCE_GROUPS = ("surface", "l1-l3", "front assembly", "back assembly")


@dataclass
class Evaluation:
    """
    The rules evaluated for K candidate configurations of one pose.

    clearance  {group: (K,) smallest gap in that group (m)}; groups are
               CLEARANCE_GROUPS
    joints     (K, 4) |joint angle|, display convention (rad)
    bend       (K,) the larger of q2, q3 (rad; > 0 bends the wrong way)
    """
    clearance: dict
    joints: np.ndarray
    bend: np.ndarray

    def excess(self) -> dict:
        """{rule: (K,) amount by which each candidate breaks it} (0 = met)."""
        out = {f"clearance {g}": np.maximum(0.0, HARD_CLEARANCE - c)
               for g, c in self.clearance.items()}
        out["joint limit"] = np.maximum(0.0, self.joints - np.radians(JOINT_HARD_LIMIT_DEG)).sum(axis=1)
        out["bend"] = np.maximum(0.0, self.bend)
        return out

    @property
    def violation(self) -> np.ndarray:
        """(K,) total amount by which the hard rules are broken (m and rad)."""
        return sum(self.excess().values())

    @property
    def feasible(self) -> np.ndarray:
        """(K,) True where every hard rule holds (within FEASIBILITY_TOL)."""
        return np.all([e <= FEASIBILITY_TOL for e in self.excess().values()], axis=0)

    def violations(self, i: int) -> dict:
        """{rule: amount} for the hard rules candidate i breaks."""
        return {name: float(e[i]) for name, e in self.excess().items() if e[i] > FEASIBILITY_TOL}

    @property
    def soft(self) -> np.ndarray:
        """(K,) the soft score: Σ over clearance groups and joints of the
        fraction of the way from the soft value to the hard one, squared."""
        score = np.zeros(len(self.bend))
        for c in self.clearance.values():
            score += np.clip((SOFT_CLEARANCE - c) / (SOFT_CLEARANCE - HARD_CLEARANCE), 0.0, 1.0) ** 2
        soft, hard = np.radians(JOINT_SOFT_LIMIT_DEG), np.radians(JOINT_HARD_LIMIT_DEG)
        score += (np.clip((self.joints - soft) / (hard - soft), 0.0, 1.0) ** 2).sum(axis=1)
        return score


def evaluate(Q, pose, env, wg: WheelGeometry, l1, l2, l3) -> Evaluation:
    """
    Check candidate joint vectors Q (K, 4) at one pose.

    pose  dict with the assemblies' chain joints and orientations x1, y1,
          theta1 (front) and x2, y2, theta2 (back), as from
          scenarios.sample_poses.  The candidates are expected to close the
          chain (solver.solve_branch_batch), so both assemblies are placed
          from the pose.
    env   environment.Environment of the transition
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    p0, p1, p2, p3 = chain_points(Q, pose["x1"], pose["y1"], pose["theta1"], l1, l2, l3)
    starts, ends = np.stack([p0, p1, p2], axis=1), np.stack([p1, p2, p3], axis=1)   # links, (K, 3, 2)
    r = JOINT_RADIUS

    clearance = {
        "surface": env.clearance(starts, ends, r).min(axis=1),
        "l1-l3":   env.separation(p0, p1, p2, p3, within=SOFT_CLEARANCE + 2 * r) - 2 * r,
    }
    for group, x, y, theta, side, own_link in (
            ("front assembly", pose["x1"], pose["y1"], pose["theta1"], +1, 0),
            ("back assembly",  pose["x2"], pose["y2"], pose["theta2"], -1, 2)):
        parts = assembly_parts(x, y, theta, wg, side)
        pa, pb, pr = (np.array([part[i] for part in parts.values()]) for i in range(3))
        # Every link against every part at once: (K, 3 links, P parts).
        gap = env.separation(starts[:, :, None], ends[:, :, None], pa, pb,
                             within=SOFT_CLEARANCE + r + pr) - r - pr
        gap[:, own_link, list(parts).index("calf")] = np.inf    # a link passes its own calf
        clearance[group] = gap.min(axis=(1, 2))

    return Evaluation(clearance=clearance,
                      joints=np.abs(q_display(Q)),
                      bend=np.maximum(Q[:, 1], Q[:, 2]))
