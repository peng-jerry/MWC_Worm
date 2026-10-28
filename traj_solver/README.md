# traj_solver — planning the transitions

Part of [MWC Worm](../README.md).  For each transition, this folder plans how both wheel assemblies move (from geometry alone), chooses the joint angles for the whole transition at once, checks every rule, and writes animations, plots and the timing files that [`../Motor/`](../Motor/README.md) plays on the robot.

## Getting started

Needs Python 3 with numpy, scipy and matplotlib (developed on Python 3.12, numpy 2.4, scipy 1.17, matplotlib 3.10). ffmpeg is optional: with it the animations are MP4, without it GIF.

```bash
pip install numpy scipy matplotlib
python build_all.py          # solve all four transitions, check them, write every output
```

The first run solves every transition (a few minutes; the exterior ones take longest) and caches the results in `.cache/`; later runs reuse them until the solver code changes. Start reading at [Architecture](#architecture): `build_all.py` and the scripts in `generate/` and `checks/` are what you run, and `core/` is the library they share. To change the robot, a setting or a transition, edit the files in `core/` ([Parameters](#parameters), [Adding a transition](#adding-a-transition)); the code in `core/engine/` reads everything from them.

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
| l1, l2, l3 | 0.2415, 0.4997, 0.2415 m | Link lengths, joint centre to joint centre (chain 0.983 m) |
| Chain span | 0.335 – 0.983 m | Gap between the chain joints the chain can reach: folded with q2 = q3 at the 110° limit, to straight |
| Wheel radius | 0.025 m | 50 mm wheels |
| Spine height | 0.177 m | Chain joint to the wheel-centre line |
| Calf | 0.114 m | Chain joint to V-apex |
| Wheelbase | 0.083 m | Arm A wheel to arm B wheel |
| Arm A | 0.066 m at 17.8° | From the calf line |
| Arm B | 0.089 m at 45° | From the calf line |

The robot's outline (30 mm links, calf, motor, arms, wheels) is taken from the CAD side view. Every dimension of the robot (the values above, the arm layouts below and the outline) is set in one file, `core/robot.py`; see [Parameters](#parameters). Joints are KBMB02 BEAR actuators driving each chain joint through a 30:1 worm; the arm A wheels are driven directly and the arm B wheels spin freely.

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

**Gap** — the straight-line distance between the two chain joints, P0 and P3, measured through space (across a plate too, not along the surface). It sets how much the 0.983 m chain has to arch: on flat ground the cruise gap of 0.786 m (0.8 chain lengths) gives q = (−36°, −54°, −54°, −36°).

**Wheel speeds** — counter-clockwise positive in the side view. A rolling wheel turns at ω = (ẑ × n)·v / r, for its centre's velocity v and the direction n from the contact (a face or a corner) to its centre. The motor speed is that minus the assembly's own turn rate, which is what the motor turns at: a wheel hinging over a corner with its assembly reads 0. A wheel wedged into an inside corner reads 0; a wheel in the air is ramped between the speeds of its contacts before and after.

## Transitions

All four transitions follow the same requirements; nothing is tuned per transition.

**Common to every transition**

- **Start and end**: the leading chain joint starts 0.40 m before the transition and the trailing one ends 0.40 m past it, 0.786 m (0.8 chain lengths) apart both times, in the same cruise posture (q2 = q3, arched away from the surface, as close to neutral as the rules allow).
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
| Pivot | the assembly stays pressed into both surfaces, turning at an even rate | the trailing wheel rolls around the solid's outline (round the corner, along the end face if there is one, round the next corner) while the assembly turns at an even rate, so the turn never pauses on the end face |
| Partner during a pivot | **parks** | **follows** at the follow gap, never rolling backwards: where keeping the gap exactly would make it back up, it eases to a stop and waits, and the gap opens a little instead |
| Gap in approach / depart | stays 0.786 m | closes from 0.786 m to the follow gap / opens back to 0.786 m |
| Phases 3 and 4 | **one after the other** (leading first), so the chain does not fold up into the corner | **one phase, "both roll"**: both roll at once, so the chain does not stretch over the edge |
| Number of phases | 6 | 5 |

The **follow gap** is chosen per exterior transition (`planner.follow_gap`): gaps up to the cruise gap (0.786 m) are searched on a 5 mm grid (every 20 mm first, then finer around the best), each as a complete plan, and scored by the smallest peak joint angle any valid configuration can achieve over all its poses. Each gap is judged by its worst score within ±10 mm, since the robot will not hold the gap exactly.

**The four transitions**

| Transition | From → to | Turn | Leads | Gap during pivots | Duration | Peak joint |
|------------|-----------|------|-------|-------------------|----------|------------|
| `floor_to_wall` | floor → interior wall | −90° | front | partner parks | 36.7 s | 90.2° |
| `wall_to_ceiling` | interior wall → ceiling | −90° | front | partner parks | 36.7 s | 90.2° |
| `outside` | top of a block → its exterior wall | +90° | front | 0.635 m | 29.7 s | 90.5° |
| `thin_edge` | top of a 3.4 mm plate → its underside | −180° | back | 0.435 m (0.435 – 0.471) | 43.7 s | 103.3° |

Durations and peaks are from the solve at 6 waypoints/s; `python generate/animate.py <name> --no-render` prints the current phase timing. The thin edge's half turn needs joints past the 90° soft limit (but within the 110° hard limit).

## Adding a transition

The transitions are the entries of `SCENARIOS` in `core/scenarios.py`; that file's docstring is the step-by-step guide, and it ends with a commented-out template (`wall_to_floor`, the reverse of floor-to-wall, which was tried and solves). In short:

1. Copy the closest existing entry under a new key (`name` must match it; output files are named after it).
2. Give its two surfaces: `approach` (arrived on) and `depart` (left on), each as `Surface(origin, tangent, normal)`: `origin` where the surface ends at the transition, `tangent` along the surface away from the transition, `normal` into free space.
3. Set `corner` (`"inside"` or `"outside"`), `turn` (each assembly's rotation, counter-clockwise positive: the short way round a corner, ±180° round a plate's end) and `wg` (an arm layout from `core/robot.py`). Which assembly leads is worked out from the surfaces.
4. In this folder: `python checks/check_constraints.py`, then `python generate/animate.py <name> --no-render`, then `python build_all.py`. Add a row to the table under [Transitions](#transitions).

Nothing else changes: every script, plot and check, and `../Motor/homing.py` / `../Motor/playback.py` (which list the files in `timings/`), pick a new entry up by its name. A mistake in a definition stops the scripts straight away with a message naming the transition and the mistake; `check_constraints.py` tests that each kind of mistake is caught. What can be modelled: two flat surfaces meeting at one corner, or the two faces of a plate (not curved surfaces, steps or gaps).

## Rules

Every configuration is checked against the robot's outline (`core/engine/constraints.py`), the plan against the surfaces (`core/engine/planner.py`).

| | Rule |
|---|------|
| **Hard** (never broken) | 15 mm clearance: chain ↔ surfaces, l1 ↔ l3, links ↔ either assembly (a link against its own assembly from the edge of its mount joint, since the joint itself is fixed to it), the two assemblies ↔ each other |
| | no part of an assembly enters the solid (wheels may touch it) |
| | every joint within ±110° |
| | q2, q3 ≤ 0: the chain arches away from the surface |
| | the gap within the chain's span, 0.335 – 0.983 m |
| | no joint steps faster than 21°/s from one waypoint to the next |
| **Soft** (in this order) | keep 35 mm of clearance and joints within ±90° |
| | smooth joint motion |
| | near-neutral posture (tie-breaker) |

## Solver

All of the solver lives in `core/engine/`; everything it reads that you might change is in `core/`.

1. **Plan** (`core/engine/planner.py`): the phases of the transition and their durations, from geometry and the speed limits. For an exterior transition the follow gap is chosen first.
2. **Sample** the plan at `params.SAMPLE_RATE` (6 per second).
3. **Joint angles** (`core/engine/trajectory_ik.py`): for each frame, candidate q1 values each give an exact closed chain (`core/engine/chain_ik.py`, one elbow branch). One candidate per frame is chosen for the whole transition at once, by dynamic programming (Viterbi), minimising
   - per frame: a very large cost for breaking a hard rule, + 1000 × the soft margins (clearance and joint limits, each 0 … 1 from the soft to the hard value, squared), + a tiny posture term;
   - per step between frames: joint speed², and a very large cost for exceeding 21°/s.

   The search runs on a 1° q1 grid, then narrows to 0.01°. The first and last frames are pinned to the cruise posture. Because all frames are chosen together, the solver can start a move early to stay within the speed limit.
4. **Retime** (`core/engine/trajectory.py`): if a joint, on the curve the robot plays, exceeds 21°/s in a phase, that phase is lengthened in proportion and the transition solved again (at most 5 times).
5. **Cache**: each solved transition is saved in `.cache/`, keyed on the source of every solver module, the sample rate and the numpy/scipy versions, so a stale result is never reused. `TRAJ_NO_CACHE=1` forces a fresh solve.

## Usage

Every script takes any number of transition names from `core/scenarios.py` (`floor_to_wall`, `wall_to_ceiling`, `outside`, `thin_edge`), or `all`, and does every transition when given none; `-h` / `--help` prints its usage. `check_constraints.py` always checks every transition.

```bash
# Regenerate every output (everything in outputs/, plus timings/):
python build_all.py
python build_all.py --no-render         # everything but the GIFs
python build_all.py outside             # just one transition

# Solve, check and animate (MP4 if ffmpeg is installed, else GIF):
python generate/animate.py floor_to_wall
python generate/animate.py thin_edge --no-render   # phase timing + check only

# Wheel speed, joint angle and joint speed against time (saves PNG):
python generate/plot_speeds.py outside

# The pose at every phase boundary, as a grid (saves PNG):
python generate/plot_phases.py outside

# One frame: the end of phase 3, with its joint angles (saves PNG):
python generate/plot_snapshot.py outside

# Waypoint files for the robot (saves CSV):
python generate/export_timing.py

# Play the timing files as Motor/playback.py does (100 Hz) and measure the motion
# between waypoints: chain gap, clearance, joint/wheel speed and acceleration:
python checks/check_playback.py                # from timings/*.csv
python checks/check_playback.py --rate 12      # solve at 12 waypoints/s instead (writes no files)

# Joint-angle quality report, compared with ik_baseline.npz:
python checks/check_ik.py
python checks/check_ik.py --save-baseline      # accept the current solution as the baseline

# Checks of the geometry, rules and plans against hand-worked cases:
python checks/check_constraints.py

# Another sample rate for one run:
TRAJ_SAMPLE_RATE=3 python checks/check_ik.py
```

`animate.py`'s check reports, per frame, any broken hard rule (HARD), joint past the hard limit (LIMIT), elbow bent the wrong way (SIGN), joint step faster than the motor limit (JUMP) or wheel rolling faster than the rolling limit (OMEGA), and exits non-zero if there is any, so `build_all.py` marks that step failed. Joints past the 90° soft limit are allowed and only summarised, one line per joint (SOFT). `check_constraints.py` exits non-zero if a check fails.

Output files (all relative to this folder, one per transition). Everything in `outputs/` is generated and can be rebuilt with `build_all.py`:

| File | Contents | Written by |
|------|----------|------------|
| `outputs/gifs/{transition}.gif` | Animation of the solved transition (`.mp4` instead if ffmpeg is available) | `generate/animate.py` |
| `outputs/speeds/{transition}.png` | Wheel motor speed (the free arm B wheels at 0 while in the air), joint angles and joint speeds against time | `generate/plot_speeds.py` |
| `outputs/phases/{transition}.png` | The pose at the start and at the end of every phase | `generate/plot_phases.py` |
| `outputs/snapshots/{transition}.png` | One frame (end of phase 3) | `generate/plot_snapshot.py` |
| `timings/{transition}.csv` | Waypoints for the robot: joint positions and speeds, wheel speeds (read by `../Motor/homing.py` and `../Motor/playback.py`) | `generate/export_timing.py` |

### Timing files

`export_timing.py` writes one row per waypoint at `params.SAMPLE_RATE` (6 per second by default; override with `TRAJ_SAMPLE_RATE=3`). The file starts with `#` lines giving units, conventions, the calibration used and how to interpolate; readers skip them.

| Column | Description |
|--------|-------------|
| `time_s` | Seconds from the start of the transition |
| `q1_rad` … `q4_rad` | Joint motor positions (rad) |
| `dq1_rad_s` … `dq4_rad_s` | Joint motor speeds at the waypoint (rad/s) |
| `w_front_a_rad_s`, `w_back_a_rad_s` | Wheel motor speeds (rad/s) of the driven wheels, front and back arm A (arm B wheels spin freely) |

All values are in each motor's own convention, set in `core/motor_calibration.py` (`position = offset + sign × ratio × angle`). The ratios are known — 30 for every chain joint (30:1 worm), 1 for the arm A wheels (direct drive) — so joint columns are motor positions and speeds, 30× the joint angles. Offsets and signs stay pass-throughs (0, +1) until measured on the robot; angles are in the display convention: q1/q4 0 = link straight up from its calf, q2/q3 0 = straight, counter-clockwise positive in the side view. After changing the calibration, re-run `export_timing.py`. `../Motor/homing.py` and `../Motor/playback.py` read the same table to show joint angles and apply tolerances at the joint.

- **Joint speeds** use the steps on both sides of each waypoint (monotone "PCHIP" rule), so the cubic Hermite curve that `playback.py` follows between waypoints never overshoots a waypoint, and a joint holding still stays still. First and last waypoints have speed 0. The solver's speed check (step 4 of [Solver](#solver)) measures the same curve.
- **Wheel speeds** are taken from the motion plan at each waypoint's instant and are relative to the wheel's own assembly, which is what its motor turns at. A wheel rolling over a corner with its assembly reads 0; a wheel in the air ramps linearly to its landing speed.

The files are played on the robot by `../Motor/homing.py` and `../Motor/playback.py`; see [`../Motor/README.md`](../Motor/README.md).

## Parameters

Everything you might change is in `core/`: four files, below. Nothing in `core/engine/` needs editing. Changing any of them changes the solve cache's key, so the next run solves again; the motor calibration and the script settings don't need a re-solve.

| File | What it sets |
|------|--------------|
| `core/robot.py` | **Robot geometry**, all of it: link lengths `l1`, `l2`, `l3`; the wheel assemblies (`SPINE_HEIGHT`, `WHEELBASE`, `ARM_A_DX`, `ARM_B_ANGLE`, `WHEEL_RADIUS`); the arm layout each kind of transition uses (`wg_inside`, `wg_outside`, `wg_thin_edge`); the outline (`JOINT_RADIUS`, `CALF_RADIUS`, `MOTOR_DEPTH` / `MOTOR_WIDTH` / `MOTOR_HEIGHT`, `ARM_A_RADIUS`, `ARM_B_RADIUS`) |
| `core/params.py` | **Everything else you tune**: waypoints per second `SAMPLE_RATE` and phase-length rounding `SCHEDULE_STEP`; speeds (the joint limit `MAX_JOINT_SPEED`, the speed planned to `PLAN_JOINT_SPEED`, rolling `SLIDE_SPEED`, pivoting `PIVOT_RATE`) and the ease-in/out time `RAMP_TIME`; limits (joints `JOINT_SOFT_LIMIT_DEG` (90°) / `JOINT_HARD_LIMIT_DEG` (110°), clearance `SOFT_CLEARANCE` (35 mm) / `HARD_CLEARANCE` (15 mm), rounding allowance `FEASIBILITY_TOL`); and the plan every transition shares (`START_DISTANCE`, `END_DISTANCE`, the cruise gap `CRUISE_SEPARATION`, the follow-gap search `FOLLOW_GAP_STEP` / `FOLLOW_GAP_TOLERANCE`); how phases are slowed for the speed limit (`RETIME_TOLERANCE`, `MAX_RETIMES`); and the solver's trade-offs (`MARGIN_WEIGHT`, `SMOOTH_WEIGHT`, `POSTURE_WEIGHT`) and search resolution (`COARSE_GRID`, `BAND_FACTOR`, `WIDE_PASS`, `SHARP_PASS`, …) |
| `core/scenarios.py` | **The transitions**: each one's surfaces, turn and arm layout (`SCENARIOS`; the leading assembly is worked out from the surfaces), the thin edge's plate `THIN_EDGE_THICKNESS`; see [Adding a transition](#adding-a-transition) |
| `core/motor_calibration.py` | **Motors**: each joint and wheel motor's zero offset, direction and gear ratio (`JOINTS`, `WHEELS`), also read by `../Motor/`; re-run `export_timing.py` after changing it |
| Scripts | animation speed `GIF_FRAME_MS` / `GIF_FRAME_STEP` (`animate.py`); playback rate `PLAYBACK_HZ` (`check_playback.py`, matching `../Motor/playback.py`) |
| Environment | `TRAJ_SAMPLE_RATE=3` overrides `SAMPLE_RATE` for one run; `TRAJ_NO_CACHE=1` forces a fresh solve |

## Architecture

```
traj_solver/   (this folder)
  build_all.py           regenerates every output (runs the scripts in generate/); start here

  generate/              the scripts that write the outputs, one kind each
    animate.py           solve + built-in check + GIF/MP4  → outputs/gifs/
    plot_speeds.py       wheel speed / joint angle / joint speed plots  → outputs/speeds/
    plot_phases.py       pose at every phase boundary  → outputs/phases/
    plot_snapshot.py     one frame, end of phase 3  → outputs/snapshots/
    export_timing.py     waypoint files for the robot  → timings/
  checks/                the checks, each printing a report
    check_constraints.py hand-worked checks of geometry, rules, definitions and plans
    check_ik.py          joint-angle quality report against ik_baseline.npz (beside it)
    check_playback.py    simulates Motor/playback.py and measures between waypoints

  core/                  the library — imported by the scripts, never run directly
    robot.py             EDIT: every dimension of the robot (links, wheel assemblies,
                         arm layouts, part sizes)
    params.py            EDIT: every setting you tune (sampling, speeds, limits, the
                         shared motion plan, the solver's weights and search)
    scenarios.py         EDIT: the transitions, and how to add one
    motor_calibration.py EDIT: each motor's zero offset, direction sign and gear ratio
                         (also read by Motor/)
    engine/              the implementation; reads everything from the three files above.
                         Each module imports only from them and from the ones above it
      geometry.py        2-D segment and capsule distance tests
      environment.py     the solid around a transition: clearance, distance through free space
      kinematics.py      forward kinematics, display convention, angle wrapping
      constraints.py     wheel positions, the robot's outline as capsules, hard and soft rules (evaluate)
      chain_ik.py        closed-form chain closure for given q1 values (one frame)
      planner.py         a transition's motion plan: phases, follow gap, plan checks,
                         cruise posture, wheel speeds; the Scenario a transition is defined by
      trajectory_ik.py   joint angles for a whole transition by dynamic programming
      trajectory.py      solve_trajectory: sampling, cruise-posture pins, retiming, cache
      drawing.py         matplotlib drawing of a transition and the robot
      cli.py             what the scripts share: transition names on the command line
                         (and --help), where outputs and timing files go
      timing_file.py     a timing file's columns: computed for export_timing.py, read
                         back for check_playback.py

  outputs/               generated figures (gifs/, speeds/, phases/, snapshots/)
  timings/               timing files for the robot
  .cache/                solved transitions (gitignored; delete any time)
```
