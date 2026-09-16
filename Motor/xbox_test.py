"""
xbox_test.py — verify Xbox controller axes/buttons and BEAR motor responses.

Run straight through with:
    python xbox_test.py

Tests in order:
  1. Ping all active motors
  2. Detect Xbox controller and list axes/buttons
  3. Walk through each axis one at a time with prompts
  4. Spin each motor briefly to confirm direction
  5. Clean up (disable + close port)
"""

import sys
import os
import math
import time
import select

# ── PyBEAR path ───────────────────────────────────────────────────────────────
_PYBEAR_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "WRBear_test-20260913T135535Z-1-001", "WRBear_test", "PyBEAR",
)
sys.path.insert(0, _PYBEAR_DIR)
from pybear import Manager

try:
    import evdev
    from evdev import ecodes
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "evdev"], check=True)
    import evdev
    from evdev import ecodes

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

# ── Helpers ───────────────────────────────────────────────────────────────────
def read1(getter, motor_id):
    data_list, err = getter(motor_id)[0]
    return data_list[0], err

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
section("1 / 5  Connect to BEAR bus and ping motors")

bear = Manager.BEAR(port=PORT, baudrate=BAUD, single_timeout=0.05, single_try_num=3)
active_ids = sorted(set(ROLE_TO_ID.values()))
all_online = True

for mid in active_ids:
    _, err = bear.ping(mid)[0]
    if err is None:
        fail(f"BEAR {mid}: no response — check power and wiring")
        all_online = False
    else:
        ok(f"BEAR {mid}: online")

if not all_online:
    print("\nOne or more motors offline. Fix connections and re-run.")
    bear.close_port()
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Detect Xbox controller
# ═══════════════════════════════════════════════════════════════════════════════
section("2 / 5  Detect Xbox controller")

candidates = []
for path in evdev.list_devices():
    dev = evdev.InputDevice(path)
    if any(k in dev.name for k in ("Xbox", "X-Box", "Microsoft")):
        candidates.append(dev)
    else:
        dev.close()

if not candidates:
    print("\n  No Xbox controller found. All visible input devices:")
    for path in evdev.list_devices():
        d = evdev.InputDevice(path)
        print(f"    {d.path}  {d.name!r}")
        d.close()
    print("\n  Plug in the controller and re-run.")
    bear.close_port()
    sys.exit(1)

joy = candidates[0]
ok(f"Found: {joy.name}  ({joy.path})")

caps     = joy.capabilities(verbose=False)
abs_info = {c: joy.absinfo(c) for c, _ in caps.get(ecodes.EV_ABS, [])}
btn_codes = set(caps.get(ecodes.EV_KEY, []))

abs_names = []
for c in sorted(abs_info):
    n = ecodes.ABS[c]
    abs_names.append(n if isinstance(n, str) else n[0])
print(f"\n  ABS axes  ({len(abs_info)}): {abs_names}")
print(f"  Buttons   ({len(btn_codes)}): present")

AXIS_LEFT_Y  = ecodes.ABS_Y
AXIS_RIGHT_X = ecodes.ABS_RX
AXIS_LT      = ecodes.ABS_Z
AXIS_RT      = ecodes.ABS_RZ
BTN_KILL     = ecodes.BTN_EAST   # B button

needed = {AXIS_LEFT_Y, AXIS_RIGHT_X, AXIS_LT, AXIS_RT}
missing = needed - set(abs_info)
if missing:
    warn(f"Expected axes not found: {[ecodes.ABS[c] for c in missing]}")
    warn("Check axis mapping and update AXIS_* constants if needed.")
else:
    ok("All expected axes present (ABS_Y, ABS_RX, ABS_Z, ABS_RZ)")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Axis isolation tests (one at a time)
# ═══════════════════════════════════════════════════════════════════════════════
section("3 / 5  Axis isolation tests")
print("  You will be prompted to move each control. Keep everything else still.")

def read_all_abs():
    return {c: joy.absinfo(c).value for c in abs_info}

def abs_name(c):
    n = ecodes.ABS[c]
    return n if isinstance(n, str) else n[0]

def diff(before, after):
    return {c: (before[c], after[c]) for c in before if abs(after[c] - before[c]) > 5}

input("\n  Release ALL sticks and triggers, then press Enter...")
baseline = read_all_abs()

tests = [
    ("Left stick  UP/DOWN    (ABS_Y)",  AXIS_LEFT_Y,  "push UP then DOWN"),
    ("Right stick LEFT/RIGHT (ABS_RX)", AXIS_RIGHT_X, "push RIGHT then LEFT"),
    ("Left trigger  LT       (ABS_Z)",  AXIS_LT,      "pull LT fully"),
    ("Right trigger RT       (ABS_RZ)", AXIS_RT,      "pull RT fully"),
]

axis_results = {}
for label, expected_code, instruction in tests:
    input(f"\n  {label} — {instruction}, then press Enter...")
    current = read_all_abs()
    changed = diff(baseline, current)
    if not changed:
        warn(f"No axis moved. Try again and hold while pressing Enter.")
    else:
        for c, (b, a) in changed.items():
            flag = "  <-- EXPECTED" if c == expected_code else ""
            print(f"    code {c:3d} ({abs_name(c)}): {b} -> {a}{flag}")
        if expected_code in changed:
            ok(f"Confirmed on expected axis ({abs_name(expected_code)})")
            axis_results[expected_code] = True
        else:
            warn(f"Expected axis ({abs_name(expected_code)}) didn't move — update AXIS_* constants if needed")
            axis_results[expected_code] = False

# B button test
input("\n  Press and HOLD the B button, then press Enter...")
if BTN_KILL in btn_codes:
    ok("B button code present in capabilities")
else:
    warn("B button (BTN_EAST) not in capabilities — kill may not work")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Motor spin test
# ═══════════════════════════════════════════════════════════════════════════════
section("4 / 5  Motor spin test (velocity mode, brief)")
print(f"  Each motor will spin at ±{SPIN_VELOCITY} rad/s for {SPIN_DURATION}s each direction.")

input("\n  Press Enter when the robot is clear to move...")

MODE_VELOCITY = 1

for role, motor_id in ROLE_TO_ID.items():
    print(f"\n  [{role}] BEAR {motor_id}")

    bear.set_torque_enable((motor_id, 0))
    bear.set_p_gain_id((motor_id, 0.358));  bear.set_i_gain_id((motor_id, 0.045));  bear.set_d_gain_id((motor_id, 0.0))
    bear.set_p_gain_iq((motor_id, 0.358));  bear.set_i_gain_iq((motor_id, 0.045));  bear.set_d_gain_iq((motor_id, 0.0))
    bear.set_p_gain_velocity((motor_id, 0.5)); bear.set_i_gain_velocity((motor_id, 0.0)); bear.set_d_gain_velocity((motor_id, 0.0))
    bear.set_limit_i_max((motor_id, 2.0))
    bear.set_limit_velocity_max((motor_id, max(SPIN_VELOCITY * 1.5, 2.0)))
    bear.set_limit_position_min((motor_id, -16 * math.pi))
    bear.set_limit_position_max((motor_id,  16 * math.pi))
    bear.set_goal_velocity((motor_id, 0.0))
    bear.set_mode((motor_id, MODE_VELOCITY))
    bear.set_torque_enable((motor_id, 1))

    # CW
    print(f"    Spinning CW  (+{SPIN_VELOCITY} rad/s)...")
    bear.set_goal_velocity((motor_id,  SPIN_VELOCITY))
    time.sleep(SPIN_DURATION)

    # CCW
    print(f"    Spinning CCW (-{SPIN_VELOCITY} rad/s)...")
    bear.set_goal_velocity((motor_id, -SPIN_VELOCITY))
    time.sleep(SPIN_DURATION)

    # Stop
    bear.set_goal_velocity((motor_id, 0.0))
    time.sleep(0.2)
    bear.set_torque_enable((motor_id, 0))

    vel, _ = read1(bear.get_present_velocity, motor_id)
    ok(f"Stopped. Present velocity: {vel:+.3f} rad/s")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Cleanup
# ═══════════════════════════════════════════════════════════════════════════════
section("5 / 5  Cleanup")
disable_all(bear, range(1, 4))
bear.close_port()
ok("All motors disabled, port closed.")
print("\nAll tests complete.")
