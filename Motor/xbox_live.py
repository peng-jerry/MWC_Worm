"""
xbox_live.py — live Xbox controller → BEAR velocity control.

Run with:
    python xbox_live.py

Every run does a full disable → configure → enable sequence so the script
can be stopped and restarted without power-cycling the motors.

Control mapping:
  Left stick  (up/down)    → left_stick_speed motor  (up = CW)
  Right stick (left/right) → right_stick_speed motor (right = CW)
  LT / RT triggers         → trigger_speed motor     (RT = CW, LT = CCW)
  B button                 → emergency stop (disables all motors immediately)
  Ctrl-C / terminal stop   → same cleanup via finally block
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

try:
    import evdev
    from evdev import ecodes
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "evdev"], check=True)
    import evdev
    from evdev import ecodes

# ══════════════════════════════════════════════════════════════════════════════
# Config — edit to match your setup
# ══════════════════════════════════════════════════════════════════════════════
PORT = "/dev/serial/by-id/usb-Westwood_Robotics_USB2BEAR_Boosted_UB0104-if00-port0"
BAUD = 8_000_000

# Comment out any role whose motor isn't connected
ROLE_TO_ID = {
    "left_stick_speed":  3,
    "right_stick_speed": 1,
    "trigger_speed":     2,
}

ROLE_PARAMS = {
    "left_stick_speed": dict(
        limit_i_max        = 3.0,
        limit_velocity_max = 4.0,
        limit_position_min = -16 * math.pi,
        limit_position_max =  16 * math.pi,
        invert             = False,
    ),
    "right_stick_speed": dict(
        limit_i_max        = 3.0,
        limit_velocity_max = 4.0,
        limit_position_min = -16 * math.pi,
        limit_position_max =  16 * math.pi,
        invert             = False,
    ),
    "trigger_speed": dict(
        limit_i_max        = 2.0,
        limit_velocity_max = 6.0,
        limit_position_min = -16 * math.pi,
        limit_position_max =  16 * math.pi,
        invert             = False,
    ),
}

CONTROL_HZ              = 50
STATUS_INTERVAL_S       = 0.5
STICK_DEADZONE          = 0.12
TRIGGER_DEADZONE        = 0.05
HEALTH_CHECK_EVERY_N    = 25     # ticks between torque_enable checks (~2×/sec at 50 Hz)

# Axis mapping — update if xbox_test.py reports different codes
AXIS_LEFT_Y  = ecodes.ABS_Y
AXIS_RIGHT_X = ecodes.ABS_RX
AXIS_LT      = ecodes.ABS_Z
AXIS_RT      = ecodes.ABS_RZ
BTN_KILL     = ecodes.BTN_EAST   # B button

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
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

def set_mode_confirmed(bear, motor_id, mode, tries=5, delay=0.02):
    for _ in range(tries):
        bear.set_mode((motor_id, mode))
        actual, err = read1(bear.get_mode, motor_id)
        if err is not None and actual == mode:
            return True
        time.sleep(delay)
    return False

def _deadzone(v, dz, floor_zero=False):
    if abs(v) < dz:
        return 0.0
    span = 1.0 - dz
    lo = 0.0 if floor_zero else -1.0
    return math.copysign((abs(v) - dz) / span, v)

# ══════════════════════════════════════════════════════════════════════════════
# Controller state
# ══════════════════════════════════════════════════════════════════════════════
class ControllerState:
    def __init__(self, dev):
        self.dev = dev
        caps = dev.capabilities(verbose=False)
        self._absinfo = {c: dev.absinfo(c) for c, _ in caps.get(ecodes.EV_ABS, [])
                         if c in (AXIS_LEFT_Y, AXIS_RIGHT_X, AXIS_LT, AXIS_RT)}
        self.raw     = {c: 0 for c in self._absinfo}
        self.buttons = set()

    def poll(self):
        while True:
            try:
                event = self.dev.read_one()
            except BlockingIOError:
                break
            if event is None:
                break
            if event.type == ecodes.EV_ABS and event.code in self.raw:
                self.raw[event.code] = event.value
            elif event.type == ecodes.EV_KEY:
                if event.value:
                    self.buttons.add(event.code)
                else:
                    self.buttons.discard(event.code)

    def stick(self, code, invert=False):
        info = self._absinfo[code]
        center = (info.max + info.min) / 2
        span   = (info.max - info.min) / 2
        v = (self.raw[code] - center) / span
        if code == AXIS_LEFT_Y:
            v = -v          # raw ABS_Y increases downward; flip so up = +1
        if invert:
            v = -v
        return _deadzone(v, STICK_DEADZONE)

    def trigger(self, code, invert=False):
        info = self._absinfo[code]
        v = (self.raw[code] - info.min) / (info.max - info.min)
        if invert:
            v = 1.0 - v
        return _deadzone(v, TRIGGER_DEADZONE, floor_zero=True)

# ══════════════════════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════════════════════
print("Connecting to BEAR bus...")
bear = Manager.BEAR(port=PORT, baudrate=BAUD, single_timeout=0.05, single_try_num=3)

active_ids = sorted(set(ROLE_TO_ID.values()))
print("Pinging motors...")
for mid in active_ids:
    _, err = bear.ping(mid)[0]
    if err is None:
        print(f"  BEAR {mid}: OFFLINE — check power and connection.")
        bear.close_port()
        sys.exit(1)
    print(f"  BEAR {mid}: online")

print("\nDetecting Xbox controller...")
candidates = []
for path in evdev.list_devices():
    dev = evdev.InputDevice(path)
    if any(k in dev.name for k in ("Xbox", "X-Box", "Microsoft")):
        candidates.append(dev)
    else:
        dev.close()

if not candidates:
    print("No Xbox controller found. Plug it in and re-run.")
    bear.close_port()
    sys.exit(1)

joy = candidates[0]
print(f"  Controller: {joy.name}  ({joy.path})")

ctrl = ControllerState(joy)

# Always disable first so re-runs don't fight a previously-enabled motor
print("\nDisabling all motors before configure...")
for mid in active_ids:
    ok = disable_motor(bear, mid)
    print(f"  BEAR {mid}: {'disabled' if ok else 'disable unconfirmed — continuing anyway'}")

print("\nConfiguring velocity mode...")
MODE_VELOCITY = 1
for role, motor_id in ROLE_TO_ID.items():
    p = ROLE_PARAMS[role]
    bear.set_p_gain_id((motor_id, 0.358));       bear.set_i_gain_id((motor_id, 0.045));       bear.set_d_gain_id((motor_id, 0.0))
    bear.set_p_gain_iq((motor_id, 0.358));       bear.set_i_gain_iq((motor_id, 0.045));       bear.set_d_gain_iq((motor_id, 0.0))
    bear.set_p_gain_velocity((motor_id, 0.5));   bear.set_i_gain_velocity((motor_id, 0.0));   bear.set_d_gain_velocity((motor_id, 0.0))
    bear.set_limit_i_max((motor_id, p["limit_i_max"]))
    bear.set_limit_velocity_max((motor_id, p["limit_velocity_max"]))
    bear.set_limit_position_min((motor_id, p["limit_position_min"]))
    bear.set_limit_position_max((motor_id, p["limit_position_max"]))
    bear.set_goal_velocity((motor_id, 0.0))
    set_mode_confirmed(bear, motor_id, MODE_VELOCITY)
    bear.set_torque_enable((motor_id, 1))
    print(f"  [{role}] BEAR {motor_id}: enabled — "
          f"i_max={p['limit_i_max']} A, vel_max={p['limit_velocity_max']} rad/s")

# ══════════════════════════════════════════════════════════════════════════════
# Control loop
# ══════════════════════════════════════════════════════════════════════════════
print("\nControl loop running.")
print("  Left stick up/down     → left_stick_speed motor")
print("  Right stick left/right → right_stick_speed motor")
print("  LT / RT triggers       → trigger_speed motor")
print("  B button               → emergency stop")
print("  Ctrl-C                 → safe exit\n")

dt          = 1.0 / CONTROL_HZ
last_status = 0.0
tick        = 0
last_warned = {}

try:
    while True:
        t0 = time.time()
        ctrl.poll()

        if BTN_KILL in ctrl.buttons:
            print("\nB pressed — emergency stop.")
            break

        if tick % HEALTH_CHECK_EVERY_N == 0:
            for role, motor_id in ROLE_TO_ID.items():
                state, _ = read1(bear.get_torque_enable, motor_id)
                if state != 1 and last_warned.get(role) != state:
                    print(f"\n  [{role}] BEAR {motor_id} torque_enable={state} — "
                          f"motor not healthy, skipping commands. Re-run script to re-enable.")
                last_warned[role] = state

        for role, motor_id in ROLE_TO_ID.items():
            if last_warned.get(role, 1) != 1:
                continue
            p = ROLE_PARAMS[role]

            if role == "trigger_speed":
                lt  = ctrl.trigger(AXIS_LT)
                rt  = ctrl.trigger(AXIS_RT)
                cmd = rt - lt
            elif role == "left_stick_speed":
                cmd = ctrl.stick(AXIS_LEFT_Y)
            elif role == "right_stick_speed":
                cmd = ctrl.stick(AXIS_RIGHT_X)
            else:
                cmd = 0.0

            if p["invert"]:
                cmd = -cmd

            bear.set_goal_velocity((motor_id, cmd * p["limit_velocity_max"]))

        now = time.time()
        if now - last_status >= STATUS_INTERVAL_S:
            last_status = now
            parts = []
            for role, motor_id in ROLE_TO_ID.items():
                v, _ = read1(bear.get_present_velocity, motor_id)
                parts.append(f"{role}(id{motor_id}) {v:+.2f} rad/s")
            print("\r" + "  ".join(parts) + "   ", end="", flush=True)

        tick += 1
        elapsed = time.time() - t0
        time.sleep(max(0.0, dt - elapsed))

except KeyboardInterrupt:
    print("\nInterrupted — stopping.")

finally:
    print("Disabling all motors...")
    for motor_id in range(1, 4):    # always target 1-3, not just active roles
        try:
            bear.set_torque_enable((motor_id, 0))
        except Exception as e:
            print(f"  BEAR {motor_id}: disable failed ({e!r}) — pull power if still moving.")
    bear.close_port()
    print("All motors disabled, port closed.")
