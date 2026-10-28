"""
zero_q1.py — jog q1 to straight-up with the Xbox controller, then store the zero.

Run with:
    python zero_q1.py

Controls:
  Left stick up/down → jog q1 slowly (BEAR 3, velocity mode)
  A button           → set the current pose as q1 = 0 (writes homing_offset)
  B button / Ctrl-C  → abort without changing the offset

Jog until q1 is straight up, then press A.  Torque is disabled once the zero
is captured.  The BEAR's own homing_offset register is set so present_position
reads 0 at that pose, matching the offset=0.0 default for q1 in
traj_solver/core/motor_calibration.py (shared with homing.py / playback.py).
"""

import sys
import os
import time
import math

from pybear import Manager

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "traj_solver", "core"))
from motor_calibration import JOINTS

try:
    import evdev
    from evdev import ecodes
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "evdev"], check=True)
    import evdev
    from evdev import ecodes

# ── Config ────────────────────────────────────────────────────────────────────
PORT     = "/dev/serial/by-id/usb-Westwood_Robotics_USB2BEAR_Boosted_UB0104-if00-port0"
BAUD     = 8_000_000
Q1_ID    = 3

GEAR_RATIO   = JOINTS["q1"].ratio   # motor turns per joint turn (30:1 worm)
JOG_VELOCITY = 0.15       # JOINT rad/s at full stick (~8.6°/s = 4.5 rad/s at the motor); partial stick is finer
JOG_I_MAX    = 2.0        # A
JOG_P_GAIN   = 3.0        # A per rad/s of velocity error — must be high enough to break joint stiction
STICK_DEADZONE = 0.12
CONTROL_HZ   = 50
ZERO_TOL     = math.radians(1.0)     # joint rad

AXIS_LEFT_Y = ecodes.ABS_Y
BTN_ZERO    = ecodes.BTN_SOUTH   # A
BTN_ABORT   = ecodes.BTN_EAST    # B

# ── Helpers ───────────────────────────────────────────────────────────────────
def read1(getter, motor_id):
    data_list, err = getter(motor_id)[0]
    return data_list[0], err

def read_pos():
    return read1(bear.get_present_position, Q1_ID)[0]

def disable_q1(tries=5, delay=0.02):
    for _ in range(tries):
        bear.set_torque_enable((Q1_ID, 0))
        state, err = read1(bear.get_torque_enable, Q1_ID)
        if err is not None and state == 0:
            return True
        time.sleep(delay)
    return False

def set_mode_confirmed(mode, tries=5, delay=0.02):
    for _ in range(tries):
        bear.set_mode((Q1_ID, mode))
        actual, err = read1(bear.get_mode, Q1_ID)
        if err is not None and actual == mode:
            return True
        time.sleep(delay)
    return False

def enable_q1(tries=5, delay=0.05):
    for _ in range(tries):
        bear.set_torque_enable((Q1_ID, 1))
        time.sleep(delay)
        en, err = read1(bear.get_torque_enable, Q1_ID)
        if en == 1 and err is not None:
            return True
    return False

# ── Connect ───────────────────────────────────────────────────────────────────
print("Connecting to BEAR bus...")
bear = Manager.BEAR(port=PORT, baudrate=BAUD, single_timeout=0.05, single_try_num=3)

_, err = bear.ping(Q1_ID)[0]
if err is None:
    print(f"BEAR {Q1_ID}: OFFLINE — check power and connection.")
    bear.close_port()
    sys.exit(1)
print(f"  BEAR {Q1_ID}: online")

print("\nDetecting Xbox controller...")
joy = None
for path in evdev.list_devices():
    dev = evdev.InputDevice(path)
    if any(k in dev.name for k in ("Xbox", "X-Box", "Microsoft")):
        joy = dev
        break
    dev.close()
if joy is None:
    print("No Xbox controller found. Connect it and re-run.")
    bear.close_port()
    sys.exit(1)
print(f"  Controller: {joy.name}  ({joy.path})")

absinfo = joy.absinfo(AXIS_LEFT_Y)
center  = (absinfo.max + absinfo.min) / 2
span    = (absinfo.max - absinfo.min) / 2
raw_y   = int(center)
buttons = set()

def poll():
    global raw_y
    while True:
        try:
            event = joy.read_one()
        except BlockingIOError:
            break
        if event is None:
            break
        if event.type == ecodes.EV_ABS and event.code == AXIS_LEFT_Y:
            raw_y = event.value
        elif event.type == ecodes.EV_KEY:
            if event.value:
                buttons.add(event.code)
            else:
                buttons.discard(event.code)

def stick():
    v = -(raw_y - center) / span          # raw ABS_Y increases downward; up = +1
    if abs(v) < STICK_DEADZONE:
        return 0.0
    return math.copysign((abs(v) - STICK_DEADZONE) / (1.0 - STICK_DEADZONE), v)

# ── Configure jog (velocity mode) ─────────────────────────────────────────────
disable_q1()
old_offset = read1(bear.get_homing_offset, Q1_ID)[0]
print(f"\nCurrent homing_offset = {old_offset:+.4f} rad")

bear.set_p_gain_id((Q1_ID, 0.358));  bear.set_i_gain_id((Q1_ID, 0.045));  bear.set_d_gain_id((Q1_ID, 0.0))
bear.set_p_gain_iq((Q1_ID, 0.358));  bear.set_i_gain_iq((Q1_ID, 0.045));  bear.set_d_gain_iq((Q1_ID, 0.0))
bear.set_p_gain_velocity((Q1_ID, JOG_P_GAIN)); bear.set_i_gain_velocity((Q1_ID, 0.0)); bear.set_d_gain_velocity((Q1_ID, 0.0))
bear.set_limit_i_max((Q1_ID, JOG_I_MAX))
bear.set_limit_velocity_max((Q1_ID, JOG_VELOCITY * GEAR_RATIO * 1.5))
bear.set_limit_position_min((Q1_ID, -16 * math.pi))
bear.set_limit_position_max((Q1_ID,  16 * math.pi))
bear.set_goal_velocity((Q1_ID, 0.0))
if not set_mode_confirmed(1):
    print("Could not confirm velocity mode on BEAR 3.")
    bear.close_port()
    sys.exit(1)
if not enable_q1():
    print("BEAR 3 would not enable (check E-STOP / error state).")
    bear.close_port()
    sys.exit(1)

print("\nJog q1 to straight up (q1 = 0):")
print("  Left stick up/down → jog")
print("  A → set zero here    B / Ctrl-C → abort\n")

# ── Jog loop ──────────────────────────────────────────────────────────────────
dt = 1.0 / CONTROL_HZ
last_status = 0.0
zero_requested = False

try:
    while True:
        t0 = time.time()
        poll()

        if BTN_ABORT in buttons:
            print("\nB pressed — aborting, offset unchanged.")
            break
        if BTN_ZERO in buttons:
            zero_requested = True
            break

        stick_v = stick()
        cmd_v   = stick_v * JOG_VELOCITY                       # joint rad/s
        bear.set_goal_velocity((Q1_ID, cmd_v * GEAR_RATIO))    # motor rad/s

        now = time.time()
        if now - last_status >= 0.25:
            last_status = now
            pos = read_pos()
            vel, _ = read1(bear.get_present_velocity, Q1_ID)
            iq, _  = read1(bear.get_present_iq, Q1_ID)
            print(f"\r  q1 {math.degrees(pos / GEAR_RATIO):+7.2f}°  stick {stick_v:+.2f}  "
                  f"cmd {cmd_v:+.3f} rad/s  vel {vel / GEAR_RATIO:+.3f} rad/s  iq {iq:+.2f} A   ", end="", flush=True)

        time.sleep(max(0.0, dt - (time.time() - t0)))

except KeyboardInterrupt:
    print("\nInterrupted — offset unchanged.")

finally:
    try:
        bear.set_goal_velocity((Q1_ID, 0.0))
    except Exception:
        pass

# ── Capture zero ──────────────────────────────────────────────────────────────
success = False
if zero_requested:
    if not disable_q1():
        print("\nWARNING: torque disable unconfirmed — writing offset anyway.")
    time.sleep(0.1)
    pos = read_pos()
    print(f"\n\nZero pose reads {math.degrees(pos / GEAR_RATIO):+.2f}° at the joint ({pos:+.4f} rad at the motor).")

    for new_offset in (old_offset - pos, old_offset + pos):
        bear.set_homing_offset((Q1_ID, new_offset))
        time.sleep(0.2)
        after = read_pos()
        print(f"  homing_offset = {new_offset:+.4f} → joint reads {math.degrees(after / GEAR_RATIO):+.2f}°")
        if abs(after) / GEAR_RATIO < ZERO_TOL:
            success = True
            break

    if success:
        print("\nCalibrated: q1 now reads ~0° at straight up.")
        print("Power-cycle the motor and re-check that it still reads ~0° "
              "(otherwise the offset isn't saved across power cycles).")
    else:
        bear.set_homing_offset((Q1_ID, old_offset))
        print("\nNeither sign gave ~0° — restored the original offset. "
              "The offset may not apply to present_position.")

disable_q1()
bear.close_port()
print("BEAR 3 disabled, port closed.")
