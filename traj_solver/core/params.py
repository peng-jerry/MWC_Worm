"""
params.py — speeds, limits and sampling shared by planning, solving and export.

All times are in seconds.  scenarios.py works out how long each phase of a
transition lasts from the speeds below (and stretches a phase if the joints
cannot keep up).  The finished motion is then sampled at SAMPLE_RATE frames
(waypoints) per second.  Per-frame limits are always derived from these
per-second values at the actual sample rate, never hard-coded.

SAMPLE_RATE can be overridden for one run without editing this file:
    TRAJ_SAMPLE_RATE=3 python check_ik.py
"""

import os

import numpy as np

# ── Sampling ──────────────────────────────────────────────────────────────────
# Waypoints per second.  Between waypoints the MCU interpolates each joint on
# its own, which opens the closed chain slightly; 6 fps keeps that gap to a few
# millimetres (it was 5–16 mm at 3 fps with linear interpolation).
SAMPLE_RATE = float(os.environ.get("TRAJ_SAMPLE_RATE", 6.0))

# Phase durations are rounded up to whole multiples of this step, so every
# phase boundary falls exactly on a frame at 3 or 6 fps.
SCHEDULE_STEP = 1 / 3

# ── Speeds and limits ─────────────────────────────────────────────────────────
MAX_JOINT_SPEED  = np.radians(21.6)   # rad/s, the joint speed limit (motor setting)
PLAN_JOINT_SPEED = np.radians(21.0)   # rad/s, what the solver plans to: ~3% margin
                                      # under MAX_JOINT_SPEED for tracking error
SLIDE_SPEED      = 0.105              # m/s, top speed of any assembly while rolling
PIVOT_RATE       = np.radians(15.0)   # rad/s, how fast an assembly turns while pivoting

# Every phase starts and ends at rest: its speed ramps up from rest over the
# first RAMP_TIME seconds along an S-curve and back down over the last
# RAMP_TIME (scenarios.ease), so each phase is RAMP_TIME longer than at
# constant speed.  Peak acceleration is π/2 × speed / RAMP_TIME: at 0.5 s,
# 0.33 m/s² for an assembly at SLIDE_SPEED and 66°/s² for a joint at 21°/s.
RAMP_TIME        = 0.5                # s
