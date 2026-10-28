"""
core — the trajectory solver as a library.  Nothing here is run directly:
the scripts (traj_solver/build_all.py, generate/, checks/) import it.

Everything you might change is in this folder; the code that uses it is in
engine/, which should not need editing.

  robot.py       every dimension of the robot: links, wheel assemblies, arm
                 layouts, part sizes
  params.py      every setting you tune: sampling, speeds, joint and clearance
                 limits, the motion plan every transition shares, the solver's
                 weights and search resolution
  scenarios.py   the transitions, one entry each, and how to add one
  motor_calibration.py
                 each motor's zero offset, direction and gear ratio (also
                 read by Motor/); only the timing files depend on it

  engine/        the implementation (see engine/__init__.py)

robot.py, params.py and motor_calibration.py import nothing else from core; scenarios.py builds its
entries with engine.planner.Scenario.
"""
