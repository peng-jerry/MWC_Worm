"""
robot.py — every dimension of the robot, in one place.

Edit the measured values here; everything else is derived from them: the
chain's reach, where the wheels sit, the outline the clearance rules check
(engine/constraints.assembly_parts) and every plan (engine/planner.py).  Lengths in
metres, angles in radians, all in the side view.

  Chain              l1, l2, l3
  Wheel assemblies   SPINE_HEIGHT, WHEELBASE, ARM_A_DX, ARM_B_ANGLE,
                     WHEEL_RADIUS → WheelGeometry, and the arm layout each
                     kind of transition uses (wg_inside, wg_outside,
                     wg_thin_edge)
  Outline            how wide each part is, for the clearance rules

Nothing here imports from the rest of the solver.
"""

from dataclasses import dataclass

import numpy as np


# ── Chain ─────────────────────────────────────────────────────────────────────
l1 = 0.2415                   # front link (m), joint centre to joint centre
l2 = 0.4997                   # middle link (m)
l3 = 0.2415                   # back link (m)
CHAIN_LENGTH = l1 + l2 + l3   # straight-chain length


# ── Wheel assemblies ──────────────────────────────────────────────────────────
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
WHEEL_RADIUS = 0.025                                # m (50 mm wheels)

# Derived from the measurements above (values at the time of writing).
ARM_B_DX     = WHEELBASE - ARM_A_DX                 # V-apex to arm B wheel, horizontal (0.063 m)
APEX_HEIGHT  = ARM_B_DX * np.tan(ARM_B_ANGLE)       # V-apex above wheel centres (0.063 m)
CALF_LEN     = SPINE_HEIGHT - APEX_HEIGHT           # 0.11385 m
BAR_LEN_A    = np.hypot(ARM_A_DX, APEX_HEIGHT)      # ≈ 0.06616 m
BAR_LEN_B    = np.hypot(ARM_B_DX, APEX_HEIGHT)      # ≈ 0.08910 m
ARM_A_OFFSET = np.arctan2(ARM_A_DX, APEX_HEIGHT)    # ≈ 17.77° from vertical
ARM_B_OFFSET = np.arctan2(ARM_B_DX, APEX_HEIGHT)    # = 45°   from vertical


@dataclass
class WheelGeometry:
    """
    The dimensions of the wheel assemblies, and which way each one's arms
    point (the arm layout).  The defaults are the measured values above with
    arm B on the inboard side of each assembly.
    """
    bar_len:   float = BAR_LEN_B     # arm B length from V-apex to wheel centre
    wheel_r:   float = WHEEL_RADIUS  # wheel radius
    # Per-assembly arm offsets from centreline c = theta − π/2.
    # Wheel direction from axle = c + arm_angle.  Positive → +x at θ=0; negative → −x.
    # Arm A and arm B always lean to opposite sides of the centreline.
    arm_a1: float = -ARM_A_OFFSET    # front assembly, arm A (~−17.8°, outboard at θ=0)
    arm_b1: float =  ARM_B_OFFSET    # front assembly, arm B (+45°, inboard at θ=0)
    arm_a2: float =  ARM_A_OFFSET    # back assembly,  arm A (~+17.8°, outboard at θ=0)
    arm_b2: float = -ARM_B_OFFSET    # back assembly,  arm B (−45°, inboard at θ=0)
    calf:   float = CALF_LEN         # strut from chain joint to V-apex (perpendicular to theta)
    bar_len_a: float = BAR_LEN_A     # arm A length; pass None to auto-derive from bar_len/arm_b1/arm_a1

    def __post_init__(self):
        if self.bar_len_a is None:
            # arm_a length that puts the arm_a wheel at the same surface height as
            # arm_b at theta=0 (equal y-projection → simultaneous floor contact):
            #   bar_len_a * cos(arm_a1) = bar_len * cos(arm_b1)
            self.bar_len_a = self.bar_len * abs(np.cos(self.arm_b1)) / abs(np.cos(self.arm_a1))


# The arms can be rotated, so each kind of transition uses the layout that suits it.
wg_inside = WheelGeometry()        # inside corners: arm B on the inboard side of each assembly
_ARM_A = abs(wg_inside.arm_a1)     # ≈ 17.8°
_ARM_B = abs(wg_inside.arm_b1)     # = 45°
wg_outside = WheelGeometry(        # outside corner: both assemblies' arm B wheel on the −x side
    bar_len=wg_inside.bar_len, wheel_r=wg_inside.wheel_r,
    arm_a1=_ARM_A, arm_b1=-_ARM_B,
    arm_a2=_ARM_A, arm_b2=-_ARM_B,
    calf=wg_inside.calf, bar_len_a=wg_inside.bar_len_a,
)
wg_thin_edge = WheelGeometry(      # thin edge: both assemblies' arm B wheel on the +x side
    bar_len=wg_inside.bar_len, wheel_r=wg_inside.wheel_r,
    arm_a1=-_ARM_A, arm_b1=_ARM_B,
    arm_a2=-_ARM_A, arm_b2=_ARM_B,
    calf=wg_inside.calf, bar_len_a=wg_inside.bar_len_a,
)


# ── Outline (side view, from the CAD model) ───────────────────────────────────
# Every part is a capsule: a segment thickened by a radius (constraints.py).
JOINT_RADIUS = 0.015      # m; links are drawn and checked as 30 mm-wide capsules
CALF_RADIUS  = 0.016      # m; calf 32 mm wide
# The motor across the calf, centred on the calf line.  Checked as a capsule
# MOTOR_HEIGHT thick whose rounded ends are part of MOTOR_WIDTH.
MOTOR_DEPTH  = 0.061      # m; its centre below the chain joint
MOTOR_WIDTH  = 0.06551    # m; overall, along the assembly (across the calf)
MOTOR_HEIGHT = 0.034      # m; overall, along the calf (so 44–78 mm below the joint)
ARM_A_RADIUS = 0.018      # m; the wheel plate along arm A, 36 mm wide
ARM_B_RADIUS = 0.010      # m; arm B, 20 mm wide
