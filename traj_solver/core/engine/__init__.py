"""
engine — the solver's implementation.  It takes every value from the files in
core/ (robot.py, params.py, transitions.py, motor_calibration.py,
motor_limits.py), so nothing here needs editing to change the robot, a setting
or a transition.

Each module imports only from core/ and from the ones listed above it:

  hermite.py         the curve the robot plays between waypoints (also read by Motor/,
                     so it imports nothing)
  geometry.py        2-D segment and capsule distance tests
  environment.py     the solid around a transition: clearance, distance through free space
  kinematics.py      forward kinematics, display convention, angle wrapping
  constraints.py     wheel positions, the outline as capsules, hard and soft rules (evaluate)
  chain_ik.py        closes the chain exactly for given q1 values (one frame)
  transition.py      the Transition each entry of transitions.py is: which assembly
                     leads, whether its partner follows, the checks on a definition
  placement.py       where an assembly goes: resting, pivoting or at a set gap; pose checks
  planner.py         a transition's motion plan: phases, easing, poses, cruise posture
  follow_gap.py      the gap the partner keeps at an outside corner or edge
  wheels.py          wheel contact and wheel speeds, from the plan
  trajectory_ik.py   joint angles for every frame at once, by dynamic programming
  trajectory.py      solve_trajectory: sampling, retiming, the solve cache
  drawing.py         matplotlib drawing of a transition and the robot
  cli.py             what the scripts share: the transitions named on the command
                     line (and --help), and where outputs and timing files go
  solve_check.py     what a playback on the robot would hit, for a solved transition
  timing_file.py     a timing file's columns: computed for generate/export_timing.py,
                     read back by the checks, and whether a file is the current solve's
  playback_sim.py    a timing file played the way Motor/playback.py plays it
"""
