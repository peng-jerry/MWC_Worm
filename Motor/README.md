# Motor — running the transitions on the robot

Part of [MWC Worm](../README.md). These scripts drive the robot's BEAR motors: they play the timing files written by [`../traj_solver/`](../traj_solver/README.md) on the robot, and help set up, test and drive the motors by hand.

## Requirements

- The robot's eight BEAR motors on the USB2BEAR adapter. Each script sets the serial port as `PORT` near its top (`/dev/serial/by-id/usb-Westwood_Robotics_USB2BEAR_Boosted_UB0104-if00-port0`).
- PyBEAR, the motors' Python library, is bundled in `WRBear_test-20260913T135535Z-1-001/WRBear_test/PyBEAR/`; the scripts add it to their path themselves. It needs pyserial and termcolor.
- evdev for the Xbox controller (`xbox_live.py`, `zero_q1.py`, `tests/xbox_test.py`), and tkinter for the diagnostics window (optional: without it `xbox_live.py` runs without the window).

```bash
pip install pyserial termcolor evdev
```

## The motors

| BEAR ID | Motor | In `playback.py` |
|---------|-------|------------------|
| 1 | front wheel (arm A) | velocity mode, follows `w_front_a_rad_s` |
| 2 | front swerve | position mode, held where it is at the start |
| 3, 4, 5, 6 | chain joints q1, q2, q3, q4 | position mode, follow `q*_rad` / `dq*_rad_s` |
| 7 | back swerve | position mode, held where it is at the start |
| 8 | back wheel (arm A) | velocity mode, follows `w_back_a_rad_s` |

The joint motors drive their joints through a 30:1 worm, so a motor turns 30 times as far as its joint; the arm A wheels are driven directly and the arm B wheels spin freely. Each motor's zero offset, direction and gear ratio is in [`../traj_solver/core/motor_calibration.py`](../traj_solver/core/motor_calibration.py), which the solver uses to write the timing files and these scripts use to read them, so the two always agree.

## Playing a transition

1. **Write the timing files** in `../traj_solver/`: `python build_all.py` (or just `python generate/export_timing.py`). Every transition with a file in `../traj_solver/timings/` can be played; the scripts find them there. The format is described in [the solver's README](../traj_solver/README.md#timing-files).
2. **Calibrate the joints**: in `motor_calibration.py`, set each joint motor's reading with its joint at 0 (link straight up from the calf for q1/q4, straight for q2/q3) and its direction, then write the timing files again. `homing.py`'s docstring shows how to read the offsets. `zero_q1.py` sets q1's zero on the motor itself instead (its `homing_offset` register). It does not call `save_config()` (see `assign_motor_id.py` on why that matters for configuration registers), so check that q1 still reads 0 after a power cycle.
3. **Straighten the swerves**, e.g. with `xbox_live.py`: the plan is in the side plane, so the wheels must point straight.
4. `python homing.py <transition>` moves q1–q4 to the file's first row, in position mode, at most 10.8°/s at each joint, until each is within 2°.
5. `python playback.py <transition>` plays the file.

**Before the joints are calibrated**, skip steps 2 and 4: place the robot in the transition's start pose by hand (`playback.py` prints it at startup) and run `python playback.py <transition> --start-pose`. Each joint motor's reading is then taken as the file's first row, so every joint moves relative to where it really is.

### What `playback.py` does

It plays the file at 100 Hz, timed from `time_s`: q1–q4 follow the cubic Hermite curve through each pair of waypoints (position and speed at both ends), both arm A wheels change speed linearly between waypoints, and the swerves hold the position they had when playback started. `../traj_solver/checks/check_playback.py` simulates exactly this before it runs on the robot.

`homing.py` and `playback.py` set each motor's current-loop gains (BEAR SDK Table 6) and its speed, acceleration and position limits on every run, so nothing carries over from an earlier script.

While playing, `playback.py` reads one motor's full status per update (one packet, motors in turn, so each about 12 times a second) and stops — wheels to 0, joints and swerves held where they are — if:

- a joint is more than 5° off its goal,
- a swerve drifts more than 0.2 rad,
- a motor reports an error or drops out, or
- a motor misses three reads in a row.

Every reading of every motor (enabled flag, goal, position, speed, currents, supply voltage, temperatures, error code) is saved to `logs/playback_<transition>_<time>.csv` (gitignored).

## Setting up the motors

- **Check which motors answer**: `python tests/ping_motors.py` (read-only, safe at any time; `python tests/ping_motors.py 1-8` for a narrower range).
- **Give a motor its ID**: motors are addressed by ID only, so two motors sharing one both act on every command. Renumber one motor at a time with everything else disconnected: `python assign_motor_id.py <new_id>`. It saves the ID so it survives a power cycle; `python tests/power_cycle_test.py` confirms that every ID does.
- **Move each motor briefly**: `python tests/motor_test.py` (or `python tests/motor_test.py 1,3,5`), with a prompt before each, since moving a joint or swerve can tip the robot.
- **Check the controller**: `python tests/xbox_test.py`.
- **Drive by hand**: `python xbox_live.py` — every motor in velocity mode from the Xbox controller, with the joints kept within ±95° by a software limit (the swerves are not limited), a diagnostics window, and **Start** as the emergency stop. The full control mapping is in its docstring.

## The scripts

```
Motor/
  homing.py              move q1–q4 to a timing file's first row
  playback.py            play a timing file on the robot (joints, both driven wheels,
                         swerves held); stops on a tracking error or motor fault; logs to logs/
  zero_q1.py             jog q1 straight up with the Xbox stick and store that as its zero
  xbox_live.py           drive all 8 motors live from an Xbox controller, with a diagnostics window
  diagnostics.py         that window: per-motor status, currents, voltage, temperatures, errors;
                         runs as its own process so drawing never delays control
  assign_motor_id.py     give one isolated motor a new permanent ID
  tests/
    ping_motors.py       read-only scan of which motor IDs answer
    motor_test.py        ping all 8 motors and nudge each one, with a prompt before each
    power_cycle_test.py  confirm motor IDs (and encoders) survive a power cycle
    xbox_test.py         check the controller's axes and buttons
  WRBear_test-…/         the BEAR SDK manual, PyBEAR and the original test notebooks
  logs/                  playback logs (gitignored)
```
