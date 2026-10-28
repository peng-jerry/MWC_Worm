"""
scenarios.py — the transitions the robot makes.  Add or change one here.

Each entry of SCENARIOS describes one transition by its two surfaces.  The
motion is planned from that geometry alone (engine/planner.py), so nothing
else needs to know about a new entry: every script, plot, check and the
robot's Motor/ scripts pick it up by its name.

Adding a transition
-------------------
1. Copy the entry closest to it below (an inside corner: floor_to_wall; an
   outside corner: outside; the end of a plate: thin_edge) under a new key.
   `name` must equal the key; the output files are named after it.
2. Describe its two surfaces, in metres in the side view (the plots' x, y):
     approach   the surface the robot arrives on
     depart     the surface it leaves on
   each a Surface(origin, tangent, normal):
     origin     where the surface ends at the transition: the corner, or the
                edge of a plate
     tangent    unit vector along the surface, pointing away from the origin,
                onto the part the robot drives on
     normal     unit vector out of the surface, into free space
   An inside corner's two surfaces share their origin.  A plate's are its two
   faces; the segment between their origins becomes its end face.
3. Set
     corner     "inside" (a concave corner) or "outside" (a convex corner, or
                the end of a plate)
     turn       how far each assembly rotates while it pivots, counter-
                clockwise positive: the short way round at a corner (±90° for
                a square one), ±180° around a plate's end — signed the way the
                robot goes round it
     wg         the arm layout: wg_inside, wg_outside or wg_thin_edge from
                robot.py, or a new WheelGeometry defined there
   Which assembly leads is worked out from the surfaces (Scenario.lead).
4. Check and build, in traj_solver/:
     python checks/check_constraints.py             samples every plan, the new one too
     python generate/animate.py <name> --no-render  solves it, prints its phases, checks it
     python build_all.py                            every output, the new one included
   then add a row to the transitions table in traj_solver/README.md.

A mistake in the definition stops the scripts straight away with a message
naming the transition: a tangent and normal that are not perpendicular unit
vectors, a turn that does not take one surface to the other (or goes the long
way round a corner), an inside corner whose surfaces do not meet, a plate's
end face not between its faces, or surfaces the chain cannot arch away from.
The plan is then checked as it is sampled: if the chain would be out of reach
or an assembly would hit a surface or the other assembly, the message says
when; START_DISTANCE / END_DISTANCE (params.py) may need to be longer.

What can be modelled: two flat surfaces meeting at one corner, or the two
faces of a plate.  Curved surfaces, steps (two corners) and gaps cannot.
"""

import numpy as np

from .engine.environment import Surface
from .engine.planner import Scenario
from .robot import wg_inside, wg_outside, wg_thin_edge

THIN_EDGE_THICKNESS = 0.0034      # m, the plate of the thin-edge transition

SCENARIOS = {
    "floor_to_wall": Scenario(
        name="floor_to_wall", label="Floor to wall (inside corner)",
        approach=Surface(origin=(0.0, 0.0), tangent=(1, 0), normal=(0, 1)),   # floor, robot at x > 0
        depart=Surface(origin=(0.0, 0.0), tangent=(0, 1), normal=(1, 0)),     # wall, robot above the floor
        turn=-np.pi / 2, corner="inside", wg=wg_inside),
    "wall_to_ceiling": Scenario(
        name="wall_to_ceiling", label="Wall to ceiling (inside corner)",
        approach=Surface(origin=(0.0, 1.5), tangent=(0, -1), normal=(1, 0)),  # wall, robot below the ceiling
        depart=Surface(origin=(0.0, 1.5), tangent=(1, 0), normal=(0, -1)),   # ceiling, robot at x > 0
        turn=-np.pi / 2, corner="inside", wg=wg_inside),
    "outside": Scenario(
        name="outside", label="Outside corner (top of a block to its exterior wall)",
        approach=Surface(origin=(0.0, 0.55), tangent=(1, 0), normal=(0, 1)),  # top of the block, robot at x > 0
        depart=Surface(origin=(0.0, 0.55), tangent=(0, -1), normal=(-1, 0)),  # exterior wall, robot below the top
        turn=+np.pi / 2, corner="outside", wg=wg_outside),
    "thin_edge": Scenario(
        name="thin_edge", label="Thin edge (top to underside)",
        approach=Surface(origin=(0.50, 0.50), tangent=(-1, 0), normal=(0, 1)),     # top face
        depart=Surface(origin=(0.50, 0.50 - THIN_EDGE_THICKNESS),                  # underside
                       tangent=(-1, 0), normal=(0, -1)),
        turn=-np.pi, corner="outside", wg=wg_thin_edge, xlim=(None, 1.1)),

    # A template: the reverse of floor_to_wall.  It was tried (it solves in
    # 36.7 s and passes the check) but is not part of the set — uncomment it
    # to add it.
    # "wall_to_floor": Scenario(
    #     name="wall_to_floor", label="Wall to floor (inside corner)",
    #     approach=Surface(origin=(0.0, 0.0), tangent=(0, 1), normal=(1, 0)),   # wall, robot above the floor
    #     depart=Surface(origin=(0.0, 0.0), tangent=(1, 0), normal=(0, 1)),     # floor, robot at x > 0
    #     turn=+np.pi / 2, corner="inside", wg=wg_inside),
}
