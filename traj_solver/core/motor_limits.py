"""
motor_limits.py — the limits the motors run under and how often they are
commanded, in one place.

Read by the solver's checks and by the scripts that drive the robot
(Motor/playback.py, homing.py, xbox_live.py), which put this folder on their
path, so it imports nothing.  It is apart from motor_calibration.py and
params.py because neither the solve nor its cache depends on these values.
"""

import math

MAX_JOINT_SPEED = math.radians(21.6)   # rad/s at the joint, the speed limit the motors are set to

# Rad at the motor: ±95.3° at the joint, through the 30:1 worm.  The BEAR silently
# ignores position-limit writes above ~50 rad (measured on this hardware), so
# 49.9 rad is the largest safe ceiling.
POS_LIMIT_MOTOR = 49.9

# Commands per second during playback.  If the PC cannot send all goals within
# one update, the late count Motor/playback.py prints at the end says so: lower
# this if that is large.
PLAYBACK_HZ = 100

# Caps Motor/playback.py sets on the motors.  It still sets them from its own
# copies (JOINT_ACC_MAX and WHEEL_VEL_MAX there): keep the two equal until it
# reads these instead.
JOINT_ACC_MAX = 120.0   # rad/s² at a joint motor
WHEEL_VEL_MAX = 5.0     # rad/s at a wheel motor
