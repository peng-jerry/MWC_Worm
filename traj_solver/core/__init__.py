"""
core — the trajectory solver as a library.  Nothing here is run directly:
the scripts in traj_solver/ import it.

Each module imports only from the ones listed above it:

  geometry.py        2-D segment and capsule distance tests
  environment.py     the solid around a transition: clearance, distance through free space
  kinematics.py      forward kinematics, display convention
  constraints.py     robot outline (CAD), WheelGeometry, hard and soft rules (evaluate)
  params.py          sample rate, speed limits, ramp time
  chain_ik.py        closes the chain exactly for given q1 values (one frame)
  scenarios.py       the four transitions: surfaces, phases, follow gap, wheel speeds
  trajectory_ik.py   joint angles for every frame at once, by dynamic programming
  trajectory.py      solve_trajectory: sampling, retiming, the solve cache
  drawing.py         matplotlib drawing of a transition and the robot
"""
