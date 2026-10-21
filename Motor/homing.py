"""
homing.py — move q1 (left-stick motor) to the starting pose defined in a timing CSV.

Usage:
    python homing.py floor_to_wall
    python homing.py wall_to_ceiling
    python homing.py outside
    python homing.py thin_edge

Uses POSITION mode on the q1 motor only.  The w_fl wheel motor (right-stick
motor) is velocity-driven and has no start pose, so it is left untouched.
Motion is interpolated at a safe speed (capped at HOMING_VEL_MAX rad/s).

BEFORE RUNNING:
  1. Set JOINT_OFFSET[i] so that:
         bear_raw_position = q_display_rad + JOINT_OFFSET[i]
     i.e. JOINT_OFFSET[i] is the raw encoder reading when q_display == 0.
     Read it with:  python -c "from pybear import Manager; b=Manager.BEAR('/dev/ttyUSB0',8000000); print(b.get_present_position(3))"
  2. Confirm MOTOR_IDS matches your wiring (q1 = BEAR 3, the left-stick motor).
  3. Confirm PORT matches your serial port.
"""

import sys
import os
import csv
import time
import math

# ── PyBEAR path ───────────────────────────────────────────────────────────────
_PYBEAR_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "WRBear_test-20260913T135535Z-1-001", "WRBear_test", "PyBEAR",
)
sys.path.insert(0, _PYBEAR_DIR)
from pybear import Manager

# ── Hardware config ───────────────────────────────────────────────────────────
PORT       = "/dev/ttyUSB0"   # change to e.g. "COM9" on Windows
BAUDRATE   = 8_000_000

# Motor IDs for the homed joints: q1 = BEAR 3 (left-stick motor).
# BEAR 1 (right-stick motor) drives the w_fl wheel and is not homed.
MOTOR_IDS  = [3]

# Gear reduction between the BEAR output and the q1 joint (motor turns 30x per joint turn)
GEAR_RATIO = 30.0

# Raw (BEAR) position when the joint is at q_display = 0:
#     goal_raw = q_display_rad * GEAR_RATIO + JOINT_OFFSET[i]
JOINT_OFFSET = [0.0]

# ── Motion limits ─────────────────────────────────────────────────────────────
HOMING_VEL_MAX = 0.19   # joint rad/s  (half of 21.6 °/s = 0.377 rad/s, conservative)
IQ_MAX         = 1.5    # A — torque cap during homing
STEP_DT        = 0.05   # s — command update interval

# Position PID (same as set_position.py example)
P_GAIN_POS = 5.0
I_GAIN_POS = 0.0
D_GAIN_POS = 0.2
P_GAIN_IQ  = 0.02
I_GAIN_IQ  = 0.02
D_GAIN_IQ  = 0.0
P_GAIN_ID  = 0.02
I_GAIN_ID  = 0.02
D_GAIN_ID  = 0.0

# Arrival tolerance
ARRIVAL_TOL = math.radians(2.0)   # rad — 2°
SETTLE_TIME = 2.0                 # s — max wait for the motor to reach the target after the ramp

# q1 travels ±95° from straight up (±49.74 rad at the motor).  The BEAR silently ignores
# position limits above ~50 rad, so use 49.9 (≈95.3° at the joint) — the largest safe value.
Q1_LIMIT_MOTOR = 49.9
POS_LIMIT_MIN = -Q1_LIMIT_MOTOR
POS_LIMIT_MAX =  Q1_LIMIT_MOTOR
MOTOR_VEL_MAX = 1.5 * HOMING_VEL_MAX * GEAR_RATIO   # rad/s at the motor

# ── Timing CSV directory ──────────────────────────────────────────────────────
_TIMINGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "traj_solver", "timings",
)

SCENARIO_NAMES = ["floor_to_wall", "wall_to_ceiling", "outside", "thin_edge"]


def load_start_pose(scenario: str) -> list[float]:
    """Return [q1] in rad from the first row of the timing CSV."""
    csv_path = os.path.join(_TIMINGS_DIR, f"{scenario}.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(1)
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)
    return [float(row["q1_rad"])]


def read1(getter, motor_id):
    data_list, err = getter(motor_id)[0]
    return data_list[0], err


def set_mode_confirmed(bear, motor_id, mode, tries=5, delay=0.02):
    for _ in range(tries):
        bear.set_mode((motor_id, mode))
        actual, err = read1(bear.get_mode, motor_id)
        if err is not None and actual == mode:
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


def read_positions(bear, ids: list[int]) -> list[float]:
    """Read present_position for each motor; return list of raw floats."""
    return [bear.get_present_position(mid)[0][0][0] for mid in ids]


def display_to_raw(q_display: list[float]) -> list[float]:
    return [q * GEAR_RATIO + JOINT_OFFSET[i] for i, q in enumerate(q_display)]


def raw_to_display(raw: list[float]) -> list[float]:
    return [(r - JOINT_OFFSET[i]) / GEAR_RATIO for i, r in enumerate(raw)]


def interpolate(start: list[float], end: list[float], t: float) -> list[float]:
    """Linear interpolation; t in [0, 1]."""
    return [s + (e - s) * t for s, e in zip(start, end)]


def homing_duration(start_raw: list[float], end_raw: list[float]) -> float:
    """Time needed so the fastest joint stays within HOMING_VEL_MAX (joint rad/s)."""
    max_delta = max(abs(e - s) for s, e in zip(start_raw, end_raw)) / GEAR_RATIO
    return max(max_delta / HOMING_VEL_MAX, STEP_DT)


# ── Main ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2 or sys.argv[1] not in SCENARIO_NAMES:
    print(f"Usage: python homing.py <scenario>")
    print(f"  scenarios: {SCENARIO_NAMES}")
    sys.exit(1)

scenario = sys.argv[1]
target_display = load_start_pose(scenario)
target_raw     = display_to_raw(target_display)

print(f"\nScenario : {scenario}")
print(f"Target   : q1={math.degrees(target_display[0]):+.1f}°  (display convention)")

# Connect
bear = Manager.BEAR(port=PORT, baudrate=BAUDRATE, single_timeout=0.05, single_try_num=3)

# Ping all motors
print("\nPinging motors...")
for mid in MOTOR_IDS:
    resp = bear.ping(mid)[0][1]
    if resp is None:
        print(f"  Motor {mid}: OFFLINE — check power and connection.")
        sys.exit(1)
    print(f"  Motor {mid}: online")

# Configure all motors
print("\nConfiguring position mode...")
for mid in MOTOR_IDS:
    bear.set_p_gain_iq((mid, P_GAIN_IQ))
    bear.set_i_gain_iq((mid, I_GAIN_IQ))
    bear.set_d_gain_iq((mid, D_GAIN_IQ))
    bear.set_p_gain_id((mid, P_GAIN_ID))
    bear.set_i_gain_id((mid, I_GAIN_ID))
    bear.set_d_gain_id((mid, D_GAIN_ID))
    bear.set_p_gain_position((mid, P_GAIN_POS))
    bear.set_i_gain_position((mid, I_GAIN_POS))
    bear.set_d_gain_position((mid, D_GAIN_POS))
    bear.set_limit_position_min((mid, POS_LIMIT_MIN))
    bear.set_limit_position_max((mid, POS_LIMIT_MAX))
    lim_max, _ = read1(bear.get_limit_position_max, mid)
    if abs(lim_max - POS_LIMIT_MAX) > 1e-3:
        print(f"  WARNING: BEAR {mid} ignored the position limit (reads ±{lim_max:.3f}, wanted ±{POS_LIMIT_MAX:.3f})")
    bear.set_limit_velocity_max((mid, MOTOR_VEL_MAX))
    bear.set_limit_i_max((mid, IQ_MAX))
    if not set_mode_confirmed(bear, mid, 2):    # position mode
        print(f"  WARNING: could not confirm position mode on BEAR {mid}")

# Read current positions
current_raw = read_positions(bear, MOTOR_IDS)
print(f"\nCurrent  : q1={math.degrees(raw_to_display(current_raw)[0]):+.1f}°")

duration = homing_duration(current_raw, target_raw)
n_steps  = max(int(duration / STEP_DT), 1)
print(f"\nHoming in {duration:.1f}s  ({n_steps} steps × {STEP_DT*1000:.0f}ms)")

input("\nPress Enter to begin homing — make sure the robot has clearance.")

# Prime goal positions before enabling torque
for mid, pos in zip(MOTOR_IDS, current_raw):
    bear.set_goal_position((mid, pos))

# Enable torque
for mid in MOTOR_IDS:
    enabled, en, en_err = enable_motor(bear, mid)
    if not enabled:
        print(f"  WARNING: BEAR {mid} did not enable (torque_enable={en}, err={en_err} → "
              f"{bear.decode_error(en_err) if en_err else 'n/a'})")

# Ramp to target
t0 = time.time()
for step in range(1, n_steps + 1):
    t_norm = step / n_steps
    goals  = interpolate(current_raw, target_raw, t_norm)
    for mid, g in zip(MOTOR_IDS, goals):
        bear.set_goal_position((mid, g))
    elapsed = time.time() - t0
    wait    = step * STEP_DT - elapsed
    if wait > 0:
        time.sleep(wait)

# Verify arrival — give the motor up to SETTLE_TIME to converge
t_settle = time.time()
while True:
    final_raw = read_positions(bear, MOTOR_IDS)
    errors    = [abs(f - t) / GEAR_RATIO for f, t in zip(final_raw, target_raw)]   # joint rad
    if all(e < ARRIVAL_TOL for e in errors) or time.time() - t_settle > SETTLE_TIME:
        break
    time.sleep(0.1)
print(f"\nArrival error: q1={math.degrees(errors[0]):.2f}°  "
      f"(at {math.degrees(raw_to_display(final_raw)[0]):+.1f}°, target {math.degrees(target_display[0]):+.1f}°)")
for mid in MOTOR_IDS:
    iq, _ = read1(bear.get_present_iq, mid)
    en, err = read1(bear.get_torque_enable, mid)
    print(f"  BEAR {mid}: iq={iq:+.2f} A (cap {IQ_MAX} A), torque_enable={en}, err={err}")

if all(e < ARRIVAL_TOL for e in errors):
    print("Homing complete — q1 within tolerance.")
else:
    print("WARNING: q1 outside tolerance. If iq is at the cap, raise IQ_MAX; "
          "if iq is low, check torque_enable/err above and the position limits.")

input("\nPress Enter to disable torque.")
for mid in MOTOR_IDS:
    bear.set_torque_enable((mid, 0))
print("Done.")
