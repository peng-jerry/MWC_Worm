"""
motor_test.py — verify BEAR motor bus connectivity and briefly move each motor.

Run straight through with:
    python motor_test.py

Or test only specific motors (comma-separated BEAR IDs):
    python motor_test.py 1,3,5

Tests in order:
  1. Ping all 8 motors (skips any that are offline, since the robot may be
     mid-setup with only some motors wired)
  2. Move each online motor briefly, one at a time, with a confirm/skip
     prompt before each — the robot sits unsupported on the ground, so
     moving q1-q4 or the swerve motors can shift its balance or tip it.
     Wheels get a normal spin test; joints/swerve get a slow, short nudge.
  3. Clean up (disable + close port)
"""

import sys
import os
import math
import time

# ── PyBEAR path ───────────────────────────────────────────────────────────────
_PYBEAR_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "WRBear_test-20260913T135535Z-1-001", "WRBear_test", "PyBEAR",
)
sys.path.insert(0, _PYBEAR_DIR)
from pybear import Manager

# ── Config — edit to match your setup ────────────────────────────────────────
PORT = "/dev/serial/by-id/usb-Westwood_Robotics_USB2BEAR_Boosted_UB0104-if00-port0"
BAUD = 8_000_000

# BEAR ID -> (role, motor type).  "wheel" gets the normal spin test;
# "joint" (chain joints + swerve) gets a much slower, shorter nudge, since an
# unsupported robot can shift or tip when those move.
ROLES = {
    1: ("front wheels",  "wheel"),
    2: ("front swerve",  "joint"),
    3: ("q1",             "joint"),
    4: ("q2",             "joint"),
    5: ("q3",             "joint"),
    6: ("q4",             "joint"),
    7: ("back swerve",   "joint"),
    8: ("back wheels",   "wheel"),
}

WHEEL_SPIN_VELOCITY = 1.0    # rad/s — wheel spin test
WHEEL_SPIN_DURATION = 0.8    # s each direction
JOINT_NUDGE_VELOCITY = 2.0   # rad/s at the motor — enough to break stiction and read reliably
JOINT_NUDGE_VELOCITY_OVERRIDE = {2: 1.0, 7: 1.0}   # front/back swerve — slower than the other joints
JOINT_NUDGE_DURATION = 0.3   # s each direction — short, same reason
MOVE_THRESHOLD = 0.25        # fraction of target velocity the motor must reach to pass

if len(sys.argv) == 1:
    SELECTED_IDS = set(ROLES)
elif len(sys.argv) == 2:
    try:
        SELECTED_IDS = {int(x) for x in sys.argv[1].split(",")}
    except ValueError:
        print("Usage: python motor_test.py [id1,id2,...]   e.g. python motor_test.py 1,3,5")
        sys.exit(1)
    unknown = SELECTED_IDS - set(ROLES)
    if unknown:
        print(f"Unknown motor ID(s): {sorted(unknown)}  (known IDs: {sorted(ROLES)})")
        sys.exit(1)
else:
    print("Usage: python motor_test.py [id1,id2,...]   e.g. python motor_test.py 1,3,5")
    sys.exit(1)

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
section(f"1 / 3  Connect to BEAR bus and ping motors ({', '.join(str(m) for m in sorted(SELECTED_IDS))})")

bear = Manager.BEAR(port=PORT, baudrate=BAUD, single_timeout=0.05, single_try_num=3)
summary = []   # (label, passed)
online_ids = []

for mid, (role, _) in ROLES.items():
    if mid not in SELECTED_IDS:
        continue
    _, err = bear.ping(mid)[0]
    if err is None:
        warn(f"BEAR {mid} [{role}]: no response — skipping (offline or not wired yet)")
    else:
        ok(f"BEAR {mid} [{role}]: online")
        online_ids.append(mid)
        summary.append((f"BEAR {mid} [{role}] ping", True))

if not online_ids:
    print("\nNo motors responded. Check power and wiring.")
    bear.close_port()
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Motor move test
# ═══════════════════════════════════════════════════════════════════════════════
section("2 / 3  Motor move test (velocity mode, one motor at a time)")
print(f"  Wheels: ±{WHEEL_SPIN_VELOCITY} rad/s for {WHEEL_SPIN_DURATION}s each direction.")
print(f"  Joints/swerve: ±{JOINT_NUDGE_VELOCITY} rad/s (motor) for {JOINT_NUDGE_DURATION}s each direction — slow and short.")
print("  The robot is unsupported on the ground — each motor is confirmed individually")
print("  before it moves, and only one motor is enabled at a time.")

MODE_VELOCITY = 1

for mid in online_ids:
    role, kind = ROLES[mid]
    vel = WHEEL_SPIN_VELOCITY if kind == "wheel" else JOINT_NUDGE_VELOCITY_OVERRIDE.get(mid, JOINT_NUDGE_VELOCITY)
    dur = WHEEL_SPIN_DURATION if kind == "wheel" else JOINT_NUDGE_DURATION

    print(f"\n  [{role}] BEAR {mid} ({kind})")
    note = "" if kind == "wheel" else " — may shift the robot's balance; brace it if needed"
    resp = input(f"    Press Enter to test{note}, or 's' + Enter to skip: ").strip().lower()
    if resp == "s":
        warn("skipped")
        continue

    if not disable_motor(bear, mid):
        warn("torque disable unconfirmed — continuing anyway")
    bear.set_p_gain_id((mid, 0.358));  bear.set_i_gain_id((mid, 0.045));  bear.set_d_gain_id((mid, 0.0))
    bear.set_p_gain_iq((mid, 0.358));  bear.set_i_gain_iq((mid, 0.045));  bear.set_d_gain_iq((mid, 0.0))
    bear.set_p_gain_velocity((mid, 0.5)); bear.set_i_gain_velocity((mid, 0.0)); bear.set_d_gain_velocity((mid, 0.0))
    bear.set_limit_i_max((mid, 2.0))
    bear.set_limit_velocity_max((mid, max(vel * 1.5, 2.0)))
    bear.set_limit_position_min((mid, -16 * math.pi))
    bear.set_limit_position_max((mid,  16 * math.pi))
    bear.set_goal_velocity((mid, 0.0))
    mode_ok = set_mode_confirmed(bear, mid, MODE_VELOCITY)
    (ok if mode_ok else fail)("velocity mode " + ("confirmed" if mode_ok else "could not be confirmed"))
    summary.append((f"BEAR {mid} [{role}] velocity mode", mode_ok))

    enabled, en, en_err = enable_motor(bear, mid)
    (ok if enabled else fail)(f"torque_enable readback = {en} (err={en_err})")
    summary.append((f"BEAR {mid} [{role}] torque enable", enabled))

    measured = {}
    for label, v in (("CW ", vel), ("CCW", -vel)):
        print(f"    Moving {label} ({v:+.2f} rad/s)...")
        bear.set_goal_velocity((mid, v))
        time.sleep(dur)
        pv, _ = read1(bear.get_present_velocity, mid)
        cur, _ = read1(bear.get_present_iq, mid)
        measured[label.strip()] = pv
        print(f"      present velocity {pv:+.3f} rad/s, iq {cur:+.3f} A")
    moved = (measured["CW"] > MOVE_THRESHOLD * vel) and (measured["CCW"] < -MOVE_THRESHOLD * vel)
    summary.append((f"BEAR {mid} [{role}] move CW/CCW", moved))

    # Stop
    bear.set_goal_velocity((mid, 0.0))
    time.sleep(0.2)
    bear.set_torque_enable((mid, 0))

    pv, _ = read1(bear.get_present_velocity, mid)
    ok(f"Stopped. Present velocity: {pv:+.3f} rad/s")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Cleanup
# ═══════════════════════════════════════════════════════════════════════════════
section("3 / 3  Cleanup")
disable_all(bear, ROLES.keys())
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
