"""
params.py — the settings you tune, in one place: sampling, speeds, the
joint and clearance limits, and the motion plan every transition shares.

Also the solver's weights and search resolution, and how phases are slowed
down for the speed limit.  Elsewhere in core/: the robot's dimensions are in
robot.py, and the transitions themselves (their surfaces, turns and arm
layouts) in scenarios.py.  Nothing in core/engine/ needs editing to change any
of these.

All times are in seconds and lengths in metres.  engine/planner.py works out how
long each phase of a transition lasts from the speeds below (and stretches
a phase if the joints cannot keep up).  The finished motion is then sampled
at SAMPLE_RATE frames (waypoints) per second.  Per-frame limits are always
derived from these per-second values at the actual sample rate, never
hard-coded.

SAMPLE_RATE can be overridden for one run without editing this file:
    TRAJ_SAMPLE_RATE=3 python checks/check_ik.py
"""

import os

import numpy as np

from .robot import CHAIN_LENGTH

# ── Sampling ──────────────────────────────────────────────────────────────────
# Waypoints per second.  Between waypoints the MCU interpolates each joint on
# its own, which opens the closed chain slightly; 6 fps keeps that gap to a few
# millimetres (it was 5–16 mm at 3 fps with linear interpolation).
SAMPLE_RATE = float(os.environ.get("TRAJ_SAMPLE_RATE", 6.0))

# Phase durations are rounded up to whole multiples of this step, so every
# phase boundary falls exactly on a frame at 3 or 6 fps.
SCHEDULE_STEP = 1 / 3

# ── Speeds ────────────────────────────────────────────────────────────────────
MAX_JOINT_SPEED  = np.radians(21.6)   # rad/s, the joint speed limit (motor setting)
PLAN_JOINT_SPEED = np.radians(21.0)   # rad/s, what the solver plans to: ~3% margin
                                      # under MAX_JOINT_SPEED for tracking error
SLIDE_SPEED      = 0.105              # m/s, top speed of any assembly while rolling
PIVOT_RATE       = np.radians(15.0)   # rad/s, how fast an assembly turns while pivoting

# Every phase starts and ends at rest: its speed ramps up from rest over the
# first RAMP_TIME seconds along an S-curve and back down over the last
# RAMP_TIME (planner.ease), so each phase is RAMP_TIME longer than at
# constant speed.  Peak acceleration is π/2 × speed / RAMP_TIME: at 0.5 s,
# 0.33 m/s² for an assembly at SLIDE_SPEED and 66°/s² for a joint at 21°/s.
RAMP_TIME        = 0.5                # s

# ── Joint limits ──────────────────────────────────────────────────────────────
# In the display convention (kinematics.q_display: q1/q4 0° = link straight
# up from the calf, q2/q3 0° = straight).  The solver, the checks and the
# plots all use these.
JOINT_SOFT_LIMIT_DEG = 90.0    # beyond this the solver pays a growing cost
JOINT_HARD_LIMIT_DEG = 110.0   # never exceeded (the ±110° stop also keeps
                               # each motor off the calf)

# ── Clearance ─────────────────────────────────────────────────────────────────
# Gap between the outlines of any two parts that can collide, and between
# the chain and the surfaces (constraints.evaluate).
HARD_CLEARANCE = 0.015    # m, never less
SOFT_CLEARANCE = 0.035    # m, less than this costs (was 50 mm: most gaps are set by the geometry, and
                          # costing 35–50 mm only bent the elbows further for no real clearance)

FEASIBILITY_TOL = 1e-3    # m or rad by which a hard rule may be missed (rounding)

# ── Motion plan: the same for every transition (planner.plan_phases) ───────
START_DISTANCE        = 0.40                  # m, leading chain joint before the transition at the start
END_DISTANCE          = 0.40                  # m, trailing chain joint past the transition at the end
CRUISE_SEPARATION     = 0.80 * CHAIN_LENGTH   # gap between chain joints at the start and end
FOLLOW_GAP_STEP       = 0.005                 # m, spacing of the follow gaps tried (planner.follow_gap)
FOLLOW_GAP_TOLERANCE  = 0.010                 # m, how far the robot's real gap may be off the plan

# ── Retiming (engine/trajectory.py) ───────────────────────────────────────────
# A phase is slowed down when a joint exceeds PLAN_JOINT_SPEED by more than
# RETIME_TOLERANCE (as a fraction) on the curve the robot plays, and the
# transition is solved again, at most MAX_RETIMES times.
RETIME_TOLERANCE = 0.005
MAX_RETIMES = 5

# ── Solver (engine/trajectory_ik.py) ──────────────────────────────────────────
# What the solver trades against what, per second of motion: every frame
# costs MARGIN_WEIGHT × its soft score (clearance and joint margins, each 0 at
# the soft limit to 1 at the hard one, squared) and every step between frames
# SMOOTH_WEIGHT × Σ (joint speed / PLAN_JOINT_SPEED)².  "Margins first":
# MARGIN_WEIGHT is far larger, so the solver gives up smoothness before
# clearance or joint range.  The posture term only breaks ties.
MARGIN_WEIGHT  = 1000.0     # soft score 1 = one clearance or joint at its hard limit
SMOOTH_WEIGHT  = 1.0        # 1 = one joint moving at the planned speed limit
POSTURE_WEIGHT = 1e-3       # per rad² from the neutral posture; a tie-breaker only

# Search resolution: finer is slower and rarely better.  The search runs on a
# coarse q1 grid, then narrows around the best path (trajectory_ik's docstring).
COARSE_GRID    = 360          # coarse q1 grid points over 2π (1° cells)
BAND_FACTOR    = 3.0          # coarse: largest q1 change per frame considered, × the step limit
WIDE_PASS      = (3.0, 61)    # (half-width in cells, points): ±3 cells at 0.1-cell spacing
WIDE_REPEATS   = 8            # most wide passes
WIDE_CONVERGED = 0.01         # wide passes stop once no frame's q1 moves more (cells)
SHARP_PASS     = (0.15, 31)   # ±0.15 cell at 0.01-cell spacing
