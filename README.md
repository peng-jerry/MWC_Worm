# MWC Worm

A planar robot made of two wheel assemblies joined by a three-link chain, which crosses between surfaces that meet at a corner: from a floor onto a wall, from a wall onto the ceiling, over an exterior corner, and around the end of a thin plate. This repository plans those transitions and plays them on the robot.

```
Front assembly ── q1 ── l1 ── q2 ── l2 ── q3 ── l3 ── q4 ── Back assembly
```

Four joints (q1–q4) connect the two assemblies through three rigid links; each joint is a KBMB02 BEAR actuator driving it through a 30:1 worm. Each assembly stands on two wheels, one of them driven, and has a swerve motor that keeps its wheels pointing straight during a transition.

## What's here

| Folder | What it does | Read |
|--------|--------------|------|
| `traj_solver/` | Plans each transition from geometry alone, chooses the joint angles for the whole transition at once, checks every rule, and writes animations, plots and the timing files the robot plays | [`traj_solver/README.md`](traj_solver/README.md) |
| `Motor/` | Drives the BEAR motors: homes the joints and plays a timing file on the robot, live Xbox control, and motor setup and tests | [`Motor/README.md`](Motor/README.md) |
| `old/` | Animations from earlier iterations of the solver, kept for reference | |

## How the two fit together

1. `traj_solver/` solves a transition and writes `traj_solver/timings/<transition>.csv`: the joint positions and speeds and the driven wheels' speeds, 6 times a second, already in each motor's own units.
2. `Motor/homing.py` moves the joints to the file's first row and `Motor/playback.py` plays the file at 100 Hz.
3. Both sides convert between joint angles and motor readings with the same table, `traj_solver/core/motor_calibration.py`, and `traj_solver/checks/check_playback.py` simulates the playback before it runs on the robot.

## The transitions

| Transition | From → to |
|------------|-----------|
| `floor_to_wall` | floor → interior wall (inside corner) |
| `wall_to_ceiling` | interior wall → ceiling (inside corner) |
| `outside` | top of a block → its exterior wall (outside corner) |
| `thin_edge` | top of a 3.4 mm plate → its underside |

Each is one entry in `traj_solver/core/scenarios.py`, and adding another takes one more entry ([how](traj_solver/README.md#adding-a-transition)). Durations, phases and the rules every transition follows are in [`traj_solver/README.md`](traj_solver/README.md#transitions).

## Quick start

On any computer, to solve every transition and write all outputs (Python 3 with numpy, scipy and matplotlib):

```bash
pip install numpy scipy matplotlib
cd traj_solver
python build_all.py
```

On the computer wired to the motors, once the joints are calibrated ([`Motor/README.md`](Motor/README.md#playing-a-transition) explains how, and how to play a transition before they are):

```bash
cd Motor
python homing.py floor_to_wall
python playback.py floor_to_wall
```
