# MWC Worm

`traj_solver/` plans each transition and writes the timing files; `Motor/` plays them on the robot.
Start with `traj_solver/README.md`: Status lists what is known to be wrong, and Architecture is the map.

## Commands

Run these from `traj_solver/`. Every script takes transition names or `all`, and `-h`.

```bash
pip install -r requirements.txt
python build_all.py --no-render                       # solve, then write plots and timing files
python generate/animate.py <transition> --no-render   # one transition: its phases and check
python checks/check_constraints.py                    # geometry, rules, definitions, plans
python checks/check_ik.py                             # joint-angle quality report
python checks/check_playback.py                       # plays timings/*.csv as the robot does
python checks/preflight.py                            # go/no-go for each timing file
```

A timing file goes to the robot only if `preflight.py` says GO for it.

A cold solve of all four takes about 2.5 minutes, and `check_constraints.py` about 1. Later runs load solves from `.cache/`.

## After changing solver code

1. Run `check_constraints.py`, `build_all.py --no-render`, then `check_playback.py`.
2. If the motion should not have changed, check with `git diff traj_solver/timings`: only the `from solve` header line may differ, and `check_ik.py` should show 0.00° against its baseline.
3. If the motion changed and that is intended, accept it with `python checks/check_ik.py --save-baseline`.

Every script exits non-zero on a failure, and `export_timing.py` writes no timing file for a solve that fails its check.

Any edit to a solver source file, even a comment, changes the cache key. The next run solves again, and the timing files need re-exporting to update that header line.

## Rules

- Change settings in `core/` (`robot.py`, `params.py`, `transitions.py`, `motor_calibration.py`, `motor_limits.py`), not in `core/engine/`.
- Each engine module imports only from `core/` and the modules listed above it in `core/engine/__init__.py`. Keep that order.
- `outputs/` and `timings/` are generated. Rebuild them; never edit them by hand.
- `Motor/` imports `core/motor_calibration.py`, `core/motor_limits.py` and `core/engine/hermite.py`, so these must import nothing from the rest of `core/`. The calibration values are measured on the robot: do not change them unless asked.
- Scripts in `Motor/` move real hardware. Do not run them unless asked.
- Do not commit or push. Say which files to stage and suggest a message.
