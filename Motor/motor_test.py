"""
motor_test.py — verify BEAR motor bus connectivity and spin each motor.

Run straight through with:
    python motor_test.py

Tests in order:
  1. Ping all active motors
  2. Spin each motor briefly to confirm direction
  3. Clean up (disable + close port)
"""

import sys
import os
import math
import time

# ── PyBEAR path ───────────────────────────────────────────────────────────────
_PYBEAR_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "WRBear_test-20260913T135535Z-1-001", "WRBear_test", "PyBEAR",
)
sys.path.insert(0, _PYBEAR_DIR)
from pybear import Manager

# ── Config — edit to match your setup ────────────────────────────────────────
PORT = "/dev/serial/by-id/usb-Westwood_Robotics_USB2BEAR_Boosted_UB0104-if00-port0"
BAUD = 8_000_000

ROLE_TO_ID = {
    "left_stick_speed":  3,
    "right_stick_speed": 1,
    "trigger_speed":     2,
}

SPIN_VELOCITY = 1.0   # rad/s used during motor spin test
SPIN_DURATION = 0.8   # seconds each direction
MOVE_THRESHOLD = 0.25  # fraction of SPIN_VELOCITY the motor must reach to pass (velocity loop is slow to ramp)

# ── Helpers ───────────────────────────────────────────────────────────────────
def read1(getter, motor_id):
    data_list, err = getter(motor_id)[0]
    return data_list[0], err

def disable_motor(bear, motor_id, tries=5, delay=0.02):
    for _ in range(tries):
        bear.set_torque_enable((motor_id, 0))
        state, err = read1(bear.get_torque_enable, motor_id)
        if err is not None and state == 0:
            return True
        time.sleep(delay)
    return False

def enable_motor(bear, motor_id, tries=5, delay=0.05):
    en, err = None, None
    for _ in range(tries):
        bear.set_torque_enable((motor_id, 1))
        time.sleep(delay)
        en, err = read1(bear.get_torque_enable, motor_id)
        if en == 1 and err is not None:
            return True, en, err
    return False, en, err

def set_mode_confirmed(bear, motor_id, mode, tries=5, delay=0.02):
    for _ in range(tries):
        bear.set_mode((motor_id, mode))
        actual, err = read1(bear.get_mode, motor_id)
        if err is not None and actual == mode:
            return True
        time.sleep(delay)
    return False

def disable_all(bear, ids):
    for mid in ids:
        try:
            bear.set_torque_enable((mid, 0))
        except Exception:
            pass

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def ok(msg):  print(f"  ✓  {msg}")
def warn(msg): print(f"  !  {msg}")
def fail(msg): print(f"  ✗  {msg}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Connect & ping
# ═══════════════════════════════════════════════════════════════════════════════
section("1 / 3  Connect to BEAR bus and ping motors")

bear = Manager.BEAR(port=PORT, baudrate=BAUD, single_timeout=0.05, single_try_num=3)
active_ids = sorted(set(ROLE_TO_ID.values()))
all_online = True
summary = []   # (label, passed)

for mid in active_ids:
    _, err = bear.ping(mid)[0]
    if err is None:
        fail(f"BEAR {mid}: no response — check power and wiring")
        all_online = False
    else:
        ok(f"BEAR {mid}: online")
    summary.append((f"BEAR {mid} ping", err is not None))

if not all_online:
    print("\nOne or more motors offline. Fix connections and re-run.")
    bear.close_port()
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Motor spin test
# ═══════════════════════════════════════════════════════════════════════════════
section("2 / 3  Motor spin test (velocity mode, brief)")
print(f"  Each motor will spin at ±{SPIN_VELOCITY} rad/s for {SPIN_DURATION}s each direction.")

input("\n  Press Enter when the robot is clear to move...")

MODE_VELOCITY = 1

for role, motor_id in ROLE_TO_ID.items():
    print(f"\n  [{role}] BEAR {motor_id}")

    if not disable_motor(bear, motor_id):
        warn("torque disable unconfirmed — continuing anyway")
    bear.set_p_gain_id((motor_id, 0.358));  bear.set_i_gain_id((motor_id, 0.045));  bear.set_d_gain_id((motor_id, 0.0))
    bear.set_p_gain_iq((motor_id, 0.358));  bear.set_i_gain_iq((motor_id, 0.045));  bear.set_d_gain_iq((motor_id, 0.0))
    bear.set_p_gain_velocity((motor_id, 0.5)); bear.set_i_gain_velocity((motor_id, 0.0)); bear.set_d_gain_velocity((motor_id, 0.0))
    bear.set_limit_i_max((motor_id, 2.0))
    bear.set_limit_velocity_max((motor_id, max(SPIN_VELOCITY * 1.5, 2.0)))
    bear.set_limit_position_min((motor_id, -16 * math.pi))
    bear.set_limit_position_max((motor_id,  16 * math.pi))
    bear.set_goal_velocity((motor_id, 0.0))
    mode_ok = set_mode_confirmed(bear, motor_id, MODE_VELOCITY)
    (ok if mode_ok else fail)("velocity mode " + ("confirmed" if mode_ok else "could not be confirmed"))
    summary.append((f"BEAR {motor_id} velocity mode", mode_ok))

    enabled, en, en_err = enable_motor(bear, motor_id)
    (ok if enabled else fail)(f"torque_enable readback = {en} (err={en_err})")
    summary.append((f"BEAR {motor_id} torque enable", enabled))

    measured = {}
    for label, vel in (("CW ", SPIN_VELOCITY), ("CCW", -SPIN_VELOCITY)):
        print(f"    Spinning {label} ({vel:+.1f} rad/s)...")
        bear.set_goal_velocity((motor_id, vel))
        time.sleep(SPIN_DURATION)
        pv, _ = read1(bear.get_present_velocity, motor_id)
        cur, _ = read1(bear.get_present_iq, motor_id)
        measured[label.strip()] = pv
        print(f"      present velocity {pv:+.3f} rad/s, iq {cur:+.3f} A")
    moved = (measured["CW"] > MOVE_THRESHOLD * SPIN_VELOCITY) and (measured["CCW"] < -MOVE_THRESHOLD * SPIN_VELOCITY)
    summary.append((f"BEAR {motor_id} spin CW/CCW", moved))

    # Stop
    bear.set_goal_velocity((motor_id, 0.0))
    time.sleep(0.2)
    bear.set_torque_enable((motor_id, 0))

    vel, _ = read1(bear.get_present_velocity, motor_id)
    ok(f"Stopped. Present velocity: {vel:+.3f} rad/s")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Cleanup
# ═══════════════════════════════════════════════════════════════════════════════
section("3 / 3  Cleanup")
disable_all(bear, range(1, 4))
bear.close_port()
ok("All motors disabled, port closed.")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
section("Summary")
for label, passed in summary:
    (ok if passed else fail)(label)

if all(passed for _, passed in summary):
    print("\nAll tests passed.")
else:
    print("\nSome tests failed — see ✗ items above.")
