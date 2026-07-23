# MWC Worm — 2D Wheeled-Robot IK Solver & Transition Animator

Inverse kinematics for a planar serial-chain robot that connects two independently-mounted wheel assemblies. The robot navigates surfaces that share a corner — floor, interior wall, interior ceiling, exterior corner, and thin horizontal edges — by solving a 4-DOF chain between the two assemblies at each animation frame.

## Robot topology

```
Front assembly (P0, θ₁)
  └─ q1 ─ l1 ─ q2 ─ l2 ─ q3 ─ l3 ─ q4
                                       Back assembly (P3, θ₂)
```

Four joint angles (q1–q4) connect the two wheel assemblies through three rigid links. Each assembly is a wishbone (V-shape): two diagonal bars radiate from the joint apex to a pair of wheels. The system has 3 pose constraints (x₂, y₂, θ₂) and 4 unknowns, leaving 1 redundant DOF that the solver resolves by minimising ‖q‖² subject to surface-contact penalties.

## Robot parameters (1.0 m chain)

| Parameter | Value | Description |
|-----------|-------|-------------|
| l1, l2, l3 | 0.25, 0.50, 0.25 m | Link lengths (total 1.00 m) |
| bar\_len | 0.200 m | Wishbone arm length (joint apex → wheel centre) |
| wheel\_r | 0.050 m | Wheel radius |
| spread | π/4 rad | Half-angle of the V opening |
| thigh | 0.08951 m | Chain-joint offset along θ |
| calf | 0.042 m | Chain-joint offset perpendicular to θ |

Environment constants: `WALL_X = 0.0`, `CEILING_Y = 0.55`, `EDGE_X = 0.44`, `EDGE_Y = 0.50`.

## Scenarios

Four transition sequences are fully implemented and share a single set of keyframes in `animate_transition.SCENARIO_CONFIG`:

| Scenario | Description | Violations |
|----------|-------------|------------|
| `floor_to_wall` | Inside bottom-left corner: floor → interior wall | CLEAN |
| `wall_to_ceiling` | Inside top-left corner: interior wall → ceiling | 21 (pre-existing speed limits) |
| `outside` | Outside top-left corner: ceiling top → exterior wall | CLEAN |
| `thin_edge` | Thin horizontal edge: top surface → bottom surface | CLEAN |

Violation thresholds: joint-angle jump ≥ 10°/frame (JUMP), wheel angular velocity > 0.70 rad/frame (OMEGA, equivalent to 0.035 m/frame at wheel\_r = 0.050).

## Constraint sets

| Set | Behaviour |
|-----|-----------|
| `wall` | Interior corner: wheels snapped to floor or interior wall based on θ |
| `ceiling` | Interior corner: wheels snapped to interior wall or ceiling based on θ |
| `outside` | Exterior corner: ceiling-top or exterior-wall contact; crossing penalty keeps linkage outside the solid block |
| `outside_exact` | Same penalties as `outside` but exact (x, y) positions from keyframes pass through unchanged |
| `thin_edge_exact` | Exact positions for thin-edge pivot phases; no surface snap |
| `floor` | Wheels on floor (y = 0) |
| `none` | Purely kinematic — no surface constraints |

## Usage

```bash
cd traj_solver

# Run a full transition animation (saves MP4 or GIF):
python animate_transition.py floor_to_wall
python animate_transition.py wall_to_ceiling
python animate_transition.py outside
python animate_transition.py thin_edge

# Skip rendering (violation check only, much faster):
python animate_transition.py floor_to_wall --no-render

# Wheel angular velocity + joint angle graph (saves PNG):
python wheel_omega.py floor_to_wall
python wheel_omega.py wall_to_ceiling
python wheel_omega.py outside
python wheel_omega.py thin_edge

# Single-frame IK explorer:
python main.py
```

Output files (written to `traj_solver/`):

| File | Contents |
|------|----------|
| `floor_to_wall.mp4` | Floor-to-wall animation |
| `wall_to_ceiling.mp4` | Wall-to-ceiling animation |
| `outside.mp4` | Outside-corner animation |
| `thin_edge.mp4` | Thin-edge animation |
| `wheel_omega_ftw.png` | ω + joint-angle graph — floor_to_wall |
| `wheel_omega_wtc.png` | ω + joint-angle graph — wall_to_ceiling |
| `wheel_omega_out.png` | ω + joint-angle graph — outside |
| `wheel_omega_te.png` | ω + joint-angle graph — thin_edge |

## Solver

1. **Grid sweep** — q1 is sampled over a dense grid in [−π, π]. For each q1 the remaining chain is solved analytically (law of cosines), yielding up to 2 exact solutions per grid point. Outside/thin-edge scenarios use 6× normal grid density.
2. **Candidate scoring** — each solution is ranked by ‖q‖² + constraint penalty + smooth_weight·‖q − q_prev‖² (trajectory smoothness term, only active after the first frame).
3. **Polish** — BFGS refines the top seeds to machine precision. Outside/thin-edge scenarios skip BFGS and use the best-penalty grid candidate directly, because the crossing-penalty landscape contains spurious local minima.

Key solver constants (per scenario):

| Scenario | smooth\_weight | cold\_at |
|----------|---------------|---------|
| floor\_to\_wall | 200.0 | None |
| wall\_to\_ceiling | 60.0 | [0.29] |
| outside | 200.0 | None |
| thin\_edge | 30.0 | [0.50, 0.61] |

## Architecture

```
traj_solver/
  animate_transition.py  trajectory solver + animation renderer; exports
                         SCENARIO_CONFIG, wg, l1, l2, l3, WALL_X, CEILING_Y,
                         EDGE_X, EDGE_Y for import by wheel_omega.py
  wheel_omega.py         wheel ω + joint-angle graphs; imports all scenario
                         data from animate_transition (single source of truth)
  solver.py              grid sweep + analytic IK + BFGS polish
  constraints.py         penalty terms, surface-contact helpers, WheelGeometry
  kinematics.py          forward kinematics and reachability check
  visualize.py           matplotlib robot drawing
  main.py                single-frame IK explorer (configure pose hints here)
```

`animate_transition.py` and `wheel_omega.py` always operate on the same keyframes: `wheel_omega.py` imports `SCENARIO_CONFIG` from `animate_transition.py` rather than maintaining its own copy.
