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
               motor            the motor across the calf
               arm A, arm B     V-apex → each wheel
               wheel A, wheel B
             The calf, arms and wheels follow WheelGeometry, so a
             rearranged arm layout carries its outline with it.  Part
             widths were read off the CAD side view; every dimension is
             in robot.py.

What can touch what (from the robot's construction)
---------------------------------------------------
  * Parts that meet at a joint sit in different layers and never collide:
    l1–l2, l2–l3, and l1 / l3 with the calf they are mounted on.
  * l1 and l3 collide.  Their ends are the joints, so this also covers
    every pair of joints that can meet.
  * Every link collides with every other part of either assembly.  A link
    is measured against its own assembly (l1 the front, l3 the back) only
    beyond its mount joint, from the edge of the joint housing (JOINT_RADIUS
    along the link): the joint is fixed to that assembly, so its own gap to
    the assembly's parts never changes and says nothing about the motion.
  * The two assemblies collide with each other and with the surfaces.
    Both assemblies are placed by the motion plan, so that is checked
    there (planner.check_assemblies), not per configuration.
  * Parts on opposite faces of a plate cannot reach each other through it:
    their gap is measured around the plate (environment.separation).

Rules for one configuration (evaluate; the limits are in params.py)
--------------------------------------------------------------------
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

from .kinematics import q_display
from ..params import (FEASIBILITY_TOL, HARD_CLEARANCE, JOINT_HARD_LIMIT_DEG, JOINT_SOFT_LIMIT_DEG,
                     SOFT_CLEARANCE)
from ..robot import (ARM_A_RADIUS, ARM_B_RADIUS, CALF_RADIUS, JOINT_RADIUS, MOTOR_DEPTH,
                    MOTOR_HEIGHT, MOTOR_WIDTH, WheelGeometry)


# ------------------------------------------------------------------ #
#  Wheel assemblies (dimensions in robot.py)                          #
# ------------------------------------------------------------------ #

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
#  Physical outline (part sizes in robot.py)                          #
# ------------------------------------------------------------------ #

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

    def bar(depth, width, height):
        """A part across the calf, centred `depth` below the chain joint: a capsule
        `height` thick whose rounded ends are part of its overall `width`."""
        centre, half = joint + depth * down, (width - height) / 2
        return centre - half * across, centre + half * across, height / 2

    return {
        "calf":    (joint, apex, CALF_RADIUS),
        "motor":   bar(MOTOR_DEPTH, MOTOR_WIDTH, MOTOR_HEIGHT),
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

# The joint and clearance limits themselves are in params.py.
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
          planner.sample_poses.  The candidates are expected to close the
          chain (chain_ik.solve_branch_batch), so both assemblies are placed
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
    # Against its own assembly a link starts at the edge of its mount joint's
    # housing (see the module docstring): l1 leaves P0, l3 leaves P3.
    starts_front = starts.copy()
    starts_front[:, 0] = p0 + (JOINT_RADIUS / l1) * (p1 - p0)
    ends_back = ends.copy()
    ends_back[:, 2] = p3 - (JOINT_RADIUS / l3) * (p3 - p2)
    for group, x, y, theta, side, a, b, own_link in (
            ("front assembly", pose["x1"], pose["y1"], pose["theta1"], +1, starts_front, ends, 0),
            ("back assembly",  pose["x2"], pose["y2"], pose["theta2"], -1, starts, ends_back, 2)):
        parts = assembly_parts(x, y, theta, wg, side)
        pa, pb, pr = (np.array([part[i] for part in parts.values()]) for i in range(3))
        # Every link against every part at once: (K, 3 links, P parts).
        gap = env.separation(a[:, :, None], b[:, :, None], pa, pb,
                             within=SOFT_CLEARANCE + r + pr) - r - pr
        gap[:, own_link, list(parts).index("calf")] = np.inf    # a link passes its own calf
        clearance[group] = gap.min(axis=(1, 2))

    return Evaluation(clearance=clearance,
                      joints=np.abs(q_display(Q)),
                      bend=np.maximum(Q[:, 1], Q[:, 2]))
