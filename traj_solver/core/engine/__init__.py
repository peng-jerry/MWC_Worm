"""
engine — the solver's implementation.  It takes every value from the files in
core/ (robot.py, params.py, scenarios.py), so nothing here needs editing to
change the robot, a setting or a transition.

Each module imports only from core/ and from the ones listed above it:

  geometry.py        2-D segment and capsule distance tests
  environment.py     the solid around a transition: clearance, distance through free space
  kinematics.py      forward kinematics, display convention, angle wrapping
  constraints.py     wheel positions, the outline as capsules, hard and soft rules (evaluate)
  chain_ik.py        closes the chain exactly for given q1 values (one frame)
  planner.py         a transition's motion plan: phases, follow gap, plan checks,
                     cruise posture, wheel speeds; the Scenario a transition is defined by
  trajectory_ik.py   joint angles for every frame at once, by dynamic programming
  trajectory.py      solve_trajectory: sampling, retiming, the solve cache
  drawing.py         matplotlib drawing of a transition and the robot
  cli.py             what the scripts share: the transitions named on the command
                     line (and --help), and where outputs and timing files go
  timing_file.py     a timing file's columns: computed for generate/export_timing.py,
                     read back for checks/check_playback.py
"""
