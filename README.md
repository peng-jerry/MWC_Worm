# MWC Worm — 2D Wheeled-Robot IK Solver

Inverse kinematics for a planar serial-chain robot that connects two independently-mounted wheel assemblies. The robot navigates surfaces that share a corner: floor, wall, ceiling, and exterior (outside) corners.

## Robot topology

```
Front assembly (P0, θ₁)
  └─ q1 ─ l1 ─ q2 ─ l2 ─ q3 ─ l3 ─ q4
                                       Back assembly (P3, θ₂)
```

Four joint angles (q1–q4) connect the two wheel assemblies through three rigid links. Each assembly is a wishbone (V-shape): two diagonal bars radiate from the joint apex to a pair of wheels. The system has 3 pose constraints (x₂, y₂, θ₂) and 4 unknowns, leaving 1 redundant DOF that the solver resolves by minimising ‖q‖².

Default parameters (edit in `main.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| l1, l2, l3 | 1.2, 1.0, 1.2 m | Link lengths |
| bar\_len | 0.25 m | Wishbone arm length (joint → wheel) |
| wheel\_r | 0.12 m | Wheel radius |
| spread | π/4 rad | Half-angle of the V opening |

## Constraint sets

| Set | Front assembly | Back assembly |
|-----|---------------|---------------|
| `none` | free | free (purely kinematic) |
| `floor` | wheels on floor (y = 0) | wheels on floor |
| `wall` | floor or wall contact | floor or wall contact |
| `ceiling` | ceiling or wall contact | ceiling or wall contact |
| `outside` | on top of ceiling | on exterior face of wall |

Contact surfaces are auto-detected from the assembly orientation θ and the y-coordinate is adjusted so the outermost wheel is tangent to the surface. For `outside`, the linkage is penalised for entering the solid corner block or the room interior, and is driven to pass exactly through the corner point (wall\_x, ceiling\_y).

## Usage

```bash
cd traj_solver
python main.py
```

Edit `CONSTRAINT_SET` and the pose hints at the top of `main.py` to explore configurations. A matplotlib figure is saved to `robot_config.png`.

## Solver

The solver uses three stages:

1. **Grid sweep** — q1 is sampled over 360 values in [−π, π]. For each q1 the remaining chain is solved analytically (law of cosines), yielding up to 2 exact solutions per grid point.
2. **Penalty ranking** — each candidate is scored by ‖q‖² + weighted constraint penalty and the top seeds are selected.
3. **Polish** — BFGS (most constraint sets) or Powell (outside corner, where the crossing penalty has a zero-gradient minimum) refines the best seeds to machine precision.

## Constraint feasibility limits

For the `outside` corner with the default wheel geometry, the back assembly y-coordinate must satisfy **y₂ ≤ 1.99** (approximately). Above this limit the linkage's corner-wrapping segment collides with the right back wheel; there is no valid IK solution. The default `y2 = 1.5` sits safely inside this range with a wheel clearance of ~0.14.

## File structure

```
traj_solver/
  kinematics.py    forward kinematics and reachability check
  constraints.py   penalty terms for each constraint set; surface-contact helpers
  solver.py        grid sweep + analytic IK + BFGS/Powell polish
  visualize.py     matplotlib drawing of robot, wheels, and surface lines
  main.py          entry point — configure constraint set and pose hints here
```
