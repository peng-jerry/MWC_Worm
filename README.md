# MWC Worm — 2D Wheeled-Robot Trajectory Solver & Transition Animator

Motion planning for a planar robot made of two wheel assemblies joined by a three-link chain. The robot crosses between surfaces that meet at a corner: floor → interior wall, interior wall → ceiling, over an exterior corner, and around the end of a thin plate. For each transition, `traj_solver/` plans how both assemblies move (from geometry alone), chooses the joint angles for the whole transition at once, checks every rule, and exports waypoint files that `Motor/playback.py` plays on the robot.

## Robot

```
Front assembly (P0, θ₁)
  └─ q1 ─ l1 ─ q2 ─ l2 ─ q3 ─ l3 ─ q4
                                       Back assembly (P3, θ₂)
```

Four joints (q1–q4) connect the two wheel assemblies through three rigid links. Placing both assemblies fixes where the chain must end (x₂, y₂, θ₂), which leaves one free degree of freedom: for any q1, q2–q4 follow in closed form. The solver chooses q1 for every frame of a transition together (see [Solver](#solver)).

Each assembly is a V: a calf runs from the chain joint down to the V-apex, and two arms run from the apex to the wheels.

| Parameter | Value | Description |
|-----------|-------|-------------|
| l1, l2, l3 | 0.25, 0.50, 0.25 m | Link lengths (chain 1.00 m) |
| Chain span | 0.329 – 1.00 m | Gap between the chain joints the chain can reach: folded with q2 = q3 at the 110° limit, to straight |
| Wheel radius | 0.025 m | 50 mm wheels |
| Spine height | 0.177 m | Chain joint to the wheel-centre line |
| Calf | 0.114 m | Chain joint to V-apex |
| Wheelbase | 0.083 m | Arm A wheel to arm B wheel |
| Arm A | 0.066 m at 17.8° | From the calf line |
| Arm B | 0.089 m at 45° | From the calf line |

The robot's outline (30 mm links, calf, block, wheel plate, arms, wheels) is taken from the CAD side view and lives in `traj_solver/constraints.py`. Joints are KBMB02 BEAR actuators driving each chain joint through a 30:1 worm; the arm A wheels are driven directly and the arm B wheels spin freely.

The arms can be mounted either way round, so each kind of transition uses the layout that suits it:

| Transition | Arm B (front, back) |
|------------|---------------------|
| floor\_to\_wall, wall\_to\_ceiling | inboard: +45°, −45° |
| outside | both toward −x: −45°, −45° |
| thin\_edge | both toward +x: +45°, +45° |

## Conventions

**Assembly orientation θ** (counter-clockwise positive, 0 = standing on a floor with its wheels down):

| θ | Wheels point | Surface |
|---|--------------|---------|
| 0 | down | floor, top of the block, top of the plate |
| −π/2 | left (−x) | interior wall |
| ±π | up | ceiling, underside of the plate |
| π/2 | right (+x) | exterior wall |

**Joint angles** — q2 and q3 are the elbows: 0 = straight, and always ≤ 0 (the chain arches away from the surface). q1 and q4 use a calf-up display convention:

```
q1_display = q1_raw − π/2     (0° = link pointing straight up out of the front calf)
q4_display = q4_raw − π/2     (0° = last link pointing straight up out of the back calf)
```

All angles in the timing files, graphs, animations and reports use the display convention.

**Gap** — the straight-line distance between the two chain joints, P0 and P3, measured through space (across a plate too, not along the surface). It sets how much the 1.00 m chain has to arch: on flat ground the cruise gap of 0.80 m gives q = (−37°, −53°, −53°, −37°).

**Wheel speeds** — counter-clockwise positive in the side view. A rolling wheel turns at ω = (ẑ × n)·v / r, for its centre's velocity v and the direction n from the contact (a face or a corner) to its centre. The motor speed is that minus the assembly's own turn rate, which is what the motor turns at: a wheel hinging over a corner with its assembly reads 0. A wheel wedged into an inside corner reads 0; a wheel in the air is ramped between the speeds of its contacts before and after.

## Transitions

All four transitions follow the same requirements; nothing is tuned per transition.

**Common to every transition**

- **Start and end**: the leading chain joint starts 0.40 m before the transition and the trailing one ends 0.40 m past it, 0.80 m apart both times, in the same cruise posture (q2 = q3, arched away from the surface, as close to neutral as the rules allow).
- **Geometry only**: every position comes from the surfaces, links, arms and wheels. Resting on a surface, the lowest wheel just touches it. Changing a dimension or a surface changes the motion automatically.
- **Motion**: every phase starts and ends at rest, ramping speed along an S-curve over 0.5 s. Phase durations come from the rolling speed (0.105 m/s) and pivot rate (15°/s), rounded up to 1/3 s; a phase is lengthened if a joint would exceed 21°/s as the robot plays it (motor limit 21.6°/s).
- **Output**: 6 waypoints per second, played as cubic Hermite curves at 100 Hz, and checked between waypoints too.

**Phases**

| # | Phase | What happens |
|---|-------|--------------|
| 1 | approach | both assemblies roll toward the transition until the leading one is ready to pivot |
| 2 | pivot | the leading assembly turns onto the new surface |
| 3 | roll clear | the leading assembly rolls along the new surface, clear of the transition |
| 4 | roll up | the trailing assembly rolls up to the transition |
| 5 | pivot | the trailing assembly turns onto the new surface |
| 6 | depart | both roll away to the standard end |

**Inside corners vs exterior transitions**

| | Inside corner (floor → wall, wall → ceiling) | Exterior (outside corner, thin edge) |
|---|---|---|
| Pivot | the assembly stays pressed into both surfaces, turning at an even rate | the trailing wheel rolls around the solid's outline (round the corner, along the end face if there is one, round the next corner) |
| Partner during a pivot | **parks** | **follows** at the follow gap, never rolling backwards: where keeping the gap exactly would make it back up, it eases to a stop and waits, and the gap opens a little instead |
| Gap in approach / depart | stays 0.80 m | closes from 0.80 m to the follow gap / opens back to 0.80 m |
| Phases 3 and 4 | **one after the other** (leading first), so the chain does not fold up into the corner | **one phase, "both roll"**: both roll at once, so the chain does not stretch over the edge |
| Number of phases | 6 | 5 |

The **follow gap** is chosen per exterior transition (`scenarios.follow_gap`): gaps up to 0.80 m are searched on a 5 mm grid (every 20 mm first, then finer around the best), each as a complete plan, and scored by the smallest peak joint angle any valid configuration can achieve over all its poses. Each gap is judged by its worst score within ±10 mm, since the robot will not hold the gap exactly.

**The four transitions**

| Transition | From → to | Turn | Leads | Gap during pivots | Duration | Peak joint |
|------------|-----------|------|-------|-------------------|----------|------------|
| `floor_to_wall` | floor → interior wall | −90° | front | partner parks | 37.3 s | 90.0° |
| `wall_to_ceiling` | interior wall → ceiling | −90° | front | partner parks | 37.3 s | 90.0° |
| `outside` | top of a block → its exterior wall | +90° | front | 0.635 m | 30.3 s | 87.1° |
| `thin_edge` | top of a 3.4 mm plate → its underside | −180° | back | 0.435 m (0.434 – 0.472) | 45.3 s | 102.8° |

Durations and peaks are from the solve at 6 waypoints/s; `python animate_transition.py <name> --no-render` prints the current phase timing. The thin edge's half turn needs joints past the 90° soft limit (but within the 110° hard limit).

## Rules

Every configuration is checked against the robot's outline (`constraints.py`), the plan against the surfaces (`scenarios.py`).

| | Rule |
|---|------|
| **Hard** (never broken) | 15 mm clearance: chain ↔ surfaces, l1 ↔ l3, links ↔ either assembly, the two assemblies ↔ each other |
| | no part of an assembly enters the solid (wheels may touch it) |
| | every joint within ±110° |
| | q2, q3 ≤ 0: the chain arches away from the surface |
| | the gap within the chain's span, 0.329 – 1.00 m |
| | no joint steps faster than 21°/s from one waypoint to the next |
| **Soft** (in this order) | keep 50 mm of clearance and joints within ±90° |
| | smooth joint motion |
| | near-neutral posture (tie-breaker) |

## Solver

1. **Plan** (`scenarios.py`): the phases of the transition and their durations, from geometry and the speed limits. For an exterior transition the follow gap is chosen first.
2. **Sample** the plan at `timing.SAMPLE_RATE` (6 per second).
3. **Joint angles** (`trajectory_ik.py`): for each frame, candidate q1 values each give an exact closed chain (`solver.py`, one elbow branch). One candidate per frame is chosen for the whole transition at once, by dynamic programming (Viterbi), minimising
   - per frame: a very large cost for breaking a hard rule, + 1000 × the soft margins (clearance and joint limits, each 0 … 1 from the soft to the hard value, squared), + a tiny posture term;
   - per step between frames: joint speed², and a very large cost for exceeding 21°/s.

   The search runs on a 1° q1 grid, then narrows to 0.01°. The first and last frames are pinned to the cruise posture. Because all frames are chosen together, the solver can start a move early to stay within the speed limit.
4. **Retime** (`trajectory.py`): if a joint, on the curve the robot plays, exceeds 21°/s in a phase, that phase is lengthened in proportion and the transition solved again (at most 5 times).
5. **Cache**: each solved transition is saved in `traj_solver/.cache/`, keyed on the source of every solver module, the sample rate and the numpy/scipy versions, so a stale result is never reused. `TRAJ_NO_CACHE=1` forces a fresh solve.

## Usage

```bash
cd traj_solver

# Regenerate every output (GIFs, wheel plots, timing files, keyframes):
python build_all.py
python build_all.py --no-render         # skip the GIFs

# Solve, check and animate one transition (MP4 if ffmpeg is installed, else GIF):
python animate_transition.py floor_to_wall
python animate_transition.py thin_edge --no-render   # timing + check only

# Wheel speed, joint angle and joint speed against time (saves PNG):
python wheel_omega.py outside

# Waypoint files for the robot (saves CSV):
python robot_timing.py                  # all 4 transitions
python robot_timing.py floor_to_wall    # one transition

# Play the timing files as Motor/playback.py does (100 Hz) and measure the motion
# between waypoints: chain gap, clearance, joint/wheel speed and acceleration:
python playback_check.py                # all 4 transitions, from timings/*.csv
python playback_check.py --rate 12      # solve at 12 waypoints/s instead (writes no files)

# The pose at every phase boundary, as a grid (saves PNG):
python keyframes.py                     # all 4 transitions
python keyframes.py outside             # one transition

# One frame: the end of phase 3, with its joint angles (saves PNG):
python main.py outside

# Joint-angle quality report, compared with ik_baseline.npz:
python check_ik.py
python check_ik.py --save-baseline      # accept the current solution as the baseline

# Checks of the geometry, rules and plans against hand-worked cases:
python test_constraints.py

# Another sample rate for one run:
TRAJ_SAMPLE_RATE=3 python check_ik.py
```

Output files (all written relative to `traj_solver/`):

| File | Contents |
|------|----------|
| `gifs/{transition}.gif` | Animation of the solved transition (`.mp4` instead if ffmpeg is available) |
| `wheel_omega/wheel_omega_{ftw,wtc,out,te}.png` | Wheel motor speed, joint angles and joint speeds against time |
| `keyframes/keyframes_{transition}.png` | The pose at the start and at the end of every phase |
| `robot_configs/robot_config_{transition}.png` | One frame (end of phase 3) — `main.py` |
| `timings/{transition}.csv` | Waypoints for the robot: joint positions and speeds, wheel speeds (`Motor/playback.py`) |

### Timing files

`robot_timing.py` writes one row per waypoint at `timing.SAMPLE_RATE` (6 per second by default; override with `TRAJ_SAMPLE_RATE=3`). The file starts with `#` lines giving units, conventions, the calibration used and how to interpolate; readers skip them.

| Column | Description |
|--------|-------------|
| `time_s` | Seconds from the start of the transition |
| `q1_rad` … `q4_rad` | Joint motor positions (rad) |
| `dq1_rad_s` … `dq4_rad_s` | Joint motor speeds at the waypoint (rad/s) |
| `w_front_a_rad_s`, `w_back_a_rad_s` | Wheel motor speeds (rad/s) of the driven wheels, front and back arm A (arm B wheels spin freely) |

All values are in each motor's own convention, set in `traj_solver/motor_calibration.py` (`position = offset + sign × ratio × angle`). The ratios are known — 30 for every chain joint (30:1 worm), 1 for the arm A wheels (direct drive) — so joint columns are motor positions and speeds, 30× the joint angles. Offsets and signs stay pass-throughs (0, +1) until measured on the robot; angles are in the display convention: q1/q4 0 = link straight up from its calf, q2/q3 0 = straight, counter-clockwise positive in the side view. After changing the calibration, re-run `robot_timing.py`. `Motor/homing.py` and `Motor/playback.py` read the same table to show joint angles and apply tolerances at the joint.

- **Joint speeds** use the steps on both sides of each waypoint (monotone "PCHIP" rule), so the cubic Hermite curve that `playback.py` follows between waypoints never overshoots a waypoint, and a joint holding still stays still. First and last waypoints have speed 0. The solver's speed check (step 4 of [Solver](#solver)) measures the same curve.
- **Wheel speeds** are taken from the motion plan at each waypoint's instant and are relative to the wheel's own assembly, which is what its motor turns at. A wheel rolling over a corner with its assembly reads 0; a wheel in the air ramps linearly to its landing speed.

`Motor/homing.py` moves the joints to the first row; `Motor/playback.py` plays the file at 100 Hz with cubic Hermite interpolation, timed from `time_s`.

## Architecture

```
traj_solver/
  scenarios.py           the four transitions: surfaces, phases, follow gap, plan checks,
                         cruise posture, wheel speeds
  environment.py         the solid around a transition: clearance, distance through free space
  geometry.py            2-D segment and capsule distance tests
  constraints.py         robot outline (CAD), WheelGeometry, hard and soft rules (evaluate)
  kinematics.py          forward kinematics, display convention
  solver.py              closed-form chain closure for given q1 values
  trajectory_ik.py       joint angles for a whole transition by dynamic programming
  trajectory.py          solve_trajectory: sampling, cruise-posture pins, retiming, cache
  timing.py              sample rate, speed limits, ramp time
  animate_transition.py  built-in check + GIF/MP4 renderer
  wheel_omega.py         wheel speed / joint angle / joint speed plots
  robot_timing.py        waypoint files for the robot → timings/*.csv
  motor_calibration.py   per-motor zero offset, direction sign, gear ratio
  playback_check.py      simulates Motor/playback.py and measures between waypoints
  keyframes.py           pose at every phase boundary → keyframes/*.png
  main.py                one-frame snapshot → robot_configs/*.png
  visualize.py           matplotlib robot drawing
  check_ik.py            joint-angle quality report against ik_baseline.npz
  test_constraints.py    hand-worked checks of geometry, rules and plans
  build_all.py           regenerates every output
Motor/
  homing.py              moves the joints to the first waypoint
  playback.py            plays a timing file at 100 Hz
```
