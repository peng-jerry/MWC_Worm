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
| wheelbase | 0.1263 m | Centre-to-centre distance between arm-A and arm-B wheels |
| bar\_len | 0.1777 m | Arm-B length from V-apex to wheel centre (hypot(0.1263, 0.125)) |
| bar\_len\_a | 0.125 m | Arm-A length from V-apex to wheel centre |
| wheel\_r | 0.050 m | Wheel radius |
| calf | 0.0921 m | Strut length from chain joint to V-apex (perpendicular to θ) |

Environment constants: `WALL_X = 0.0`, `CEILING_Y = 0.55` (outside scenario), `EDGE_X = 0.44`, `EDGE_Y = 0.50`. Wall-to-ceiling uses a local ceiling height of 1.50 m.

Arm-B lean angle (±45.3°) varies by scenario:

| Scenario | arm\_b1 (front) | arm\_b2 (back) |
|----------|----------------|----------------|
| floor\_to\_wall, wall\_to\_ceiling | +45.3° | −45.3° |
| outside | −45.3° | −45.3° |
| thin\_edge | +45.3° | +45.3° |

## Conventions

### Assembly orientation θ

θ is the forward-pointing direction of the assembly. Key values:

| θ | Orientation | Typical contact surface |
|---|-------------|------------------------|
| 0 | pointing right | floor |
| −π/2 | pointing down | interior wall (ascending) |
| π | pointing left | interior ceiling |
| π/2 | pointing up | exterior wall (descending) |

### Arm-A and arm-B

Each V-shaped assembly has two arms radiating from the V-apex:

- **Arm-A** (0.125 m): straight arm, always at 0° relative to the centreline `c = θ − π/2`. At θ=0 this arm points straight down.
- **Arm-B** (0.1777 m): angled arm, at ±45.3° relative to `c`. The sign is scenario-specific (see table above).

In the timing CSV, "left" means the arm-A wheel and "right" means the arm-B wheel for both assemblies. In a 2D model these are the leading and trailing wheels depending on direction of travel.

### Joint angle convention

q2 and q3 are standard chain angles (elbow joints, always ≤ 0 in a valid configuration). q1 and q4 use a **calf-up display convention**:

```
q1_display = q1_raw − π/2     (0° = link pointing straight up out of the front calf)
q4_display = q4_raw − π/2     (0° = last link pointing straight up out of the back calf)
```

All angles in the timing CSV, graphs, and keyframe labels use the display convention. The raw solver angles differ by π/2.

### Wheel ω sign convention

ω > 0 is CCW. The rolling formula depends on which surface the wheel is on:

| Surface | Formula | Positive ω = |
|---------|---------|--------------|
| Floor | −Δx / R | moving left |
| Interior wall | +Δy / R | moving up |
| Interior ceiling | +Δx / R | moving right |
| Ceiling top (outside scenario) | −Δx / R | moving left |
| Exterior wall | −Δy / R | moving down |
| Edge top | −Δx / R | moving left |
| Edge bottom | +Δx / R | moving right |

ω = 0 during pivot frames (θ changing) and frames where the wheel is not in contact with a surface.

## Scenarios

Four transition sequences are fully implemented and share a single set of keyframes in `animate_transition.SCENARIO_CONFIG`:

| Scenario | Description | Violations |
|----------|-------------|------------|
| `floor_to_wall` | Inside bottom-left corner: floor → interior wall | CLEAN |
| `wall_to_ceiling` | Inside top-left corner: interior wall → ceiling | CLEAN |
| `outside` | Outside top-left corner: ceiling top → exterior wall | CLEAN |
| `thin_edge` | Thin horizontal edge: top surface → bottom surface | CLEAN |

Violation thresholds: joint-angle jump ≥ 7.2°/frame (JUMP), wheel angular velocity > 3.77 rad/frame (OMEGA hard ceiling; operational slide target is 0.70 rad/frame = 0.035 m/frame at wheel\_r = 0.050).

## Constraint sets

| Set | Behaviour |
|-----|-----------|
| `wall` | Inside floor-wall corner: wheels snapped to floor or interior wall based on θ |
| `ceiling` | Inside wall-ceiling corner: wheels snapped to interior wall or ceiling based on θ |
| `wall_exact` / `ceiling_exact` | Same penalties as wall/ceiling but exact (x, y) passed through unchanged |
| `outside` | Exterior corner: ceiling-top or exterior-wall contact; crossing penalty keeps linkage outside the solid block |
| `outside_exact` | Same penalties as `outside` but exact (x, y) positions from keyframes pass through unchanged |
| `thin_edge_exact` | Thin-edge pivot phases; positions passed through unchanged |
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

# Export per-frame joint angles and wheel speeds (saves CSV):
python robot_timing.py                  # all 4 scenarios
python robot_timing.py floor_to_wall    # single scenario

# Render all keyframe poses as a grid (saves PNG):
python keyframes.py                     # all 4 scenarios
python keyframes.py outside             # single scenario

# Single-frame IK explorer:
python main.py
python main.py outside
```

Output files (all written relative to `traj_solver/`):

| File | Contents |
|------|----------|
| `gifs/floor_to_wall.gif` | Floor-to-wall animation (MP4 if ffmpeg available) |
| `gifs/wall_to_ceiling.gif` | Wall-to-ceiling animation |
| `gifs/outside.gif` | Outside-corner animation |
| `gifs/thin_edge.gif` | Thin-edge animation |
| `wheel_omega/wheel_omega_ftw.png` | ω + joint-angle graph — floor_to_wall |
| `wheel_omega/wheel_omega_wtc.png` | ω + joint-angle graph — wall_to_ceiling |
| `wheel_omega/wheel_omega_out.png` | ω + joint-angle graph — outside |
| `wheel_omega/wheel_omega_te.png` | ω + joint-angle graph — thin_edge |
| `robot_configs/robot_config_{scenario}.png` | Single-frame snapshot — main.py |
| `keyframes/keyframes_{scenario}.png` | Grid of all keyframe poses |
| `timings/{scenario}.csv` | Per-frame joint angles and wheel speeds for robot control |

### Timing CSV columns

Each CSV has 90 rows (one per frame) and 13 columns. `PHYSICAL_FPS = 3` fr/s, derived from the motor speed limit: 21.6 °/s ÷ 7.2 °/fr = 3 fr/s. To adapt to a different motor, set `PHYSICAL_FPS = motor_max_deg_per_s / 7.2` in `robot_timing.py`; only the time scaling and velocity values change — the joint angle columns are unaffected.

| Column | Description |
|--------|-------------|
| `time_s` | Elapsed time in seconds (`frame / PHYSICAL_FPS`) |
| `q1_rad` | Joint 1 angle (rad), calf-up convention — 0 = link straight up from calf |
| `dq1_rad_s` | Joint 1 angular velocity (rad/s) = Δq × PHYSICAL\_FPS |
| `w_fl_rad_s` | Front-left wheel ω (rad/s), CCW positive |
| `q2_rad` | Joint 2 angle (rad) |
| `dq2_rad_s` | Joint 2 angular velocity (rad/s) |
| `w_fr_rad_s` | Front-right wheel ω (rad/s) |
| `q3_rad` | Joint 3 angle (rad) |
| `dq3_rad_s` | Joint 3 angular velocity (rad/s) |
| `w_bl_rad_s` | Back-left wheel ω (rad/s) |
| `q4_rad` | Joint 4 angle (rad), calf-up convention |
| `dq4_rad_s` | Joint 4 angular velocity (rad/s) |
| `w_br_rad_s` | Back-right wheel ω (rad/s) |

Each joint is paired with its nearest wheel: q1 ↔ front-left (arm-A), q2 ↔ front-right (arm-B), q3 ↔ back-left, q4 ↔ back-right. Wheel ω is 0 during pivot and air frames.

## Solver

1. **Grid sweep** — q1 is sampled over n\_grid × 6 points in [−π, π]. For each q1 the remaining chain is solved analytically (law of cosines), yielding up to 2 exact solutions per grid point. (The ×6 comes from 3 sub-intervals × 2 analytic branches per point.)
2. **Candidate scoring** — each solution is ranked by ‖q‖² + constraint penalty + smooth\_weight·‖q − q\_prev‖² (smoothness term, active after the first frame).
3. **Polish** — BFGS refines the top seeds to machine precision. For `outside` and `thin_edge` constraint sets BFGS is skipped and the best-penalty grid candidate is used directly, because the crossing-penalty landscape contains spurious local minima.

**`smooth_weight`** controls how strongly each frame is pulled toward the previous frame's solution. Higher values produce smoother joint trajectories but can lock the solver onto a suboptimal IK branch once it has committed to one. Lower values give the solver more freedom to find globally better poses but risk frame-to-frame jumps.

**`cold_at`** is a normalised time value (or list of values) at which the warm-start is dropped and the solver searches globally. Use this at rotation boundaries where the robot legitimately needs to switch IK branches. Without it, the smoothness term can prevent the solver from finding the correct post-rotation configuration.

Key solver parameters per scenario:

| Scenario | n\_grid | smooth\_weight | cold\_at |
|----------|---------|---------------|---------|
| floor\_to\_wall | 120 | 20.0 | None |
| wall\_to\_ceiling | 60 | 100.0 | None |
| outside | 60 | 200.0 | None |
| thin\_edge | 60 | 30.0 | None |

## Architecture

```
traj_solver/
  animate_transition.py  trajectory solver + animation renderer; exports
                         SCENARIO_CONFIG (including per-scenario WheelGeometry),
                         wg, wg_outside, wg_thin_edge, l1, l2, l3, WALL_X,
                         CEILING_Y, EDGE_X, EDGE_Y, N_FRAMES, and all WTC/FTW/OUT
                         phase timing constants
  wheel_omega.py         wheel ω + joint-angle graphs
  robot_timing.py        per-frame joint angles + wheel speeds → timings/*.csv
  keyframes.py           grid of all keyframe poses → keyframes/*.png
  solver.py              grid sweep + analytic IK + BFGS polish
  constraints.py         penalty terms, surface-contact helpers, WheelGeometry
  kinematics.py          forward kinematics and reachability check
  visualize.py           matplotlib robot drawing
  main.py                single-frame IK explorer
```

All scripts import `SCENARIO_CONFIG` from `animate_transition.py` as the single source of truth for keyframes, solver parameters, and wheel geometry.
