"""
homing.py — move all 4 chain joints to the starting pose defined in a timing CSV.

Usage:
    python homing.py floor_to_wall
    python homing.py wall_to_ceiling
    python homing.py outside
    python homing.py thin_edge

Uses POSITION mode.  Each motor is interpolated independently, no joint
faster than HOMING_JOINT_SPEED, so the joint with the furthest to go sets the
overall time.

The CSV is in each motor's own units (motor positions, after the 30:1 worm);
speeds, tolerances and printed angles here are at the joint, converted with
the same table robot_timing.py uses (traj_solver/motor_calibration.py).

BEFORE RUNNING:
  1. Calibrate in traj_solver/motor_calibration.py — offset = the motor's
     reading with the joint at 0 (link straight up from the calf for q1/q4,
     straight for q2/q3), plus direction sign — then re-run
     traj_solver/robot_timing.py.
     Read the offsets with:  python -c "from pybear import Manager; b=Manager.BEAR('/dev/ttyUSB0',8000000); print(b.get_present_position(1,2,3,4))"
  2. Confirm MOTOR_IDS matches your wiring.
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

# ── Motor calibration (shared with traj_solver/robot_timing.py) ───────────────
_TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "traj_solver")
sys.path.insert(0, _TRAJ_DIR)
from motor_calibration import JOINTS
CAL = [JOINTS[f"q{j}"] for j in range(1, 5)]      # q1 → q4

# ── Hardware config ───────────────────────────────────────────────────────────
PORT       = "/dev/ttyUSB0"   # change to e.g. "COM9" on Windows
BAUDRATE   = 8_000_000

# Motor IDs for joints q1 → q4 (change to match your wiring)
MOTOR_IDS  = [1, 2, 3, 4]

# ── Motion limits ─────────────────────────────────────────────────────────────
HOMING_JOINT_SPEED = math.radians(10.8)   # rad/s at the joint (half the 21.6 °/s limit)
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
ARRIVAL_TOL = math.radians(2.0)   # rad at the joint — 2°

# ── Timing CSV directory ──────────────────────────────────────────────────────
_TIMINGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "traj_solver", "timings",
)

SCENARIO_NAMES = ["floor_to_wall", "wall_to_ceiling", "outside", "thin_edge"]


def load_start_pose(scenario: str) -> list[float]:
    """Return [q1, q2, q3, q4] motor positions (rad) from the first row of the timing CSV."""
    csv_path = os.path.join(_TIMINGS_DIR, f"{scenario}.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(1)
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(line for line in fh if not line.startswith("#"))
        row = next(reader)
    return [float(row["q1_rad"]), float(row["q2_rad"]),
            float(row["q3_rad"]), float(row["q4_rad"])]


def read_positions(bear, ids: list[int]) -> list[float]:
    """Read present_position for each motor; return list of raw floats."""
    return [bear.get_present_position(mid)[0][0][0] for mid in ids]


def joint_degrees(raw: list[float]) -> str:
    """Motor positions as joint angles, e.g. 'q1=+12.3°  q2=…'."""
    return "  ".join(f"q{j + 1}={math.degrees(c.angle(r)):+.1f}°" for j, (c, r) in enumerate(zip(CAL, raw)))


def interpolate(start: list[float], end: list[float], t: float) -> list[float]:
    """Linear interpolation; t in [0, 1]."""
    return [s + (e - s) * t for s, e in zip(start, end)]


def homing_duration(start_raw: list[float], end_raw: list[float]) -> float:
    """Time needed so no joint moves faster than HOMING_JOINT_SPEED."""
    max_delta = max(abs(c.angle(e) - c.angle(s)) for c, s, e in zip(CAL, start_raw, end_raw))
    return max(max_delta / HOMING_JOINT_SPEED, STEP_DT)


# ── Main ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2 or sys.argv[1] not in SCENARIO_NAMES:
    print(f"Usage: python homing.py <scenario>")
    print(f"  scenarios: {SCENARIO_NAMES}")
    sys.exit(1)

scenario   = sys.argv[1]
target_raw = load_start_pose(scenario)

print(f"\nScenario : {scenario}")
print(f"Target   : {joint_degrees(target_raw)}  (joint angles)")

# Connect
bear = Manager.BEAR(port=PORT, baudrate=BAUDRATE)

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
    bear.set_mode((mid, 2))          # position mode
    bear.set_limit_i_max((mid, IQ_MAX))

# Read current positions
current_raw = read_positions(bear, MOTOR_IDS)
print(f"\nCurrent  : {joint_degrees(current_raw)}")

# The motors refuse positions outside their position limits; check the target first.
for j, (mid, goal) in enumerate(zip(MOTOR_IDS, target_raw)):
    lo = bear.get_limit_position_min(mid)[0][0][0]
    hi = bear.get_limit_position_max(mid)[0][0][0]
    if not lo <= goal <= hi:
        print(f"WARNING: q{j + 1} target {goal:+.2f} rad is outside motor {mid}'s position limits "
              f"[{lo:+.2f}, {hi:+.2f}] rad (motor units, after the worm)")

duration = homing_duration(current_raw, target_raw)
n_steps  = max(int(duration / STEP_DT), 1)
print(f"\nHoming in {duration:.1f}s  ({n_steps} steps × {STEP_DT*1000:.0f}ms)")

input("\nPress Enter to begin homing — make sure the robot has clearance.")

# Prime goal positions before enabling torque
for mid, pos in zip(MOTOR_IDS, current_raw):
    bear.set_goal_position((mid, pos))

# Enable torque
for mid in MOTOR_IDS:
    bear.set_torque_enable((mid, 1))

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

# Verify arrival
time.sleep(0.3)
final_raw = read_positions(bear, MOTOR_IDS)
errors    = [abs(c.angle(f) - c.angle(t)) for c, f, t in zip(CAL, final_raw, target_raw)]
print("\nArrival errors (joint): " + "  ".join(f"q{j + 1}={math.degrees(e):.2f}°"
                                              for j, e in enumerate(errors)))

if all(e < ARRIVAL_TOL for e in errors):
    print("Homing complete — all joints within tolerance.")
else:
    print("WARNING: one or more joints outside tolerance. Check motor_calibration.py offsets or IQ_MAX.")

input("\nPress Enter to disable torque.")
for mid in MOTOR_IDS:
    bear.set_torque_enable((mid, 0))
print("Done.")
