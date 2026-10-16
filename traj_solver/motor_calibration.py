"""
motor_calibration.py — how each motor measures the angles and speeds in the
timing files.

robot_timing.py writes the timing files in each motor's own convention:

    joint motor position = offset + sign × ratio × joint angle
    joint motor speed    =          sign × ratio × joint speed
    wheel motor speed    =          sign × ratio × wheel speed

  joint angle   display convention (kinematics.q_display): q1 / q4 are 0
                with the link straight up from its calf, q2 / q3 are 0 when
                straight; radians, counter-clockwise positive in the side view
                (the view of every plot and animation)
  wheel speed   rad/s, counter-clockwise positive in the side view, relative
                to the assembly the wheel is mounted on

  offset  the motor's position reading (rad) with the joint at 0.  Not used
          for wheels.
  sign    +1 if the motor turns counter-clockwise in the side view when its
          reading increases, −1 if clockwise.  A wheel motor seen from the
          other side of the robot has the opposite sign to its partner.
  ratio   motor turns per joint (or wheel) turn: 1 when the motor drives it
          directly.  BEAR actuators report their output shaft (after the
          built-in 20:1), so this is only gearing beyond the actuator.

Every entry is a pass-through (offset 0, sign +1, ratio 1) until measured on
the robot.  After changing this file, run robot_timing.py again (the solve
itself is cached and does not depend on it).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Motor:
    offset: float = 0.0     # rad
    sign: int = +1
    ratio: float = 1.0

    def position(self, angle):
        """Motor position (rad) for a joint angle (rad, display convention)."""
        return self.offset + self.sign * self.ratio * angle

    def speed(self, rate):
        """Motor speed (rad/s) for a joint or wheel speed (rad/s, side view)."""
        return self.sign * self.ratio * rate

    def angle(self, position):
        """Joint angle (rad, display convention) for a motor position: the inverse of position()."""
        return (position - self.offset) / (self.sign * self.ratio)

    def rate(self, speed):
        """Joint or wheel speed (rad/s, side view) for a motor speed: the inverse of speed()."""
        return speed / (self.sign * self.ratio)


# The chain joints, q1 (front) … q4 (back).
JOINTS = {
    "q1": Motor(),
    "q2": Motor(),
    "q3": Motor(),
    "q4": Motor(),
}

# The wheels: arm A and arm B of the front and back assemblies.
WHEELS = {
    "front_a": Motor(),
    "front_b": Motor(),
    "back_a":  Motor(),
    "back_b":  Motor(),
}
