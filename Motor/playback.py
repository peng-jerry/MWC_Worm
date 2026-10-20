"""
playback.py — replay a timing CSV through the 4 chain-joint BEAR motors.

Usage:
    python playback.py floor_to_wall
    python playback.py wall_to_ceiling
    python playback.py outside
    python playback.py thin_edge

Run homing.py first to reach the start pose before executing this script.

Each CSV row is a waypoint at time time_s (traj_solver/export_timing.py).
Every 1/INTERP_HZ seconds the script sends each joint the point at that
moment on the cubic through its position and speed at the two surrounding
waypoints (cubic Hermite interpolation):

    q(α) = h00(α)·q[k] + h10(α)·T·dq[k] + h01(α)·q[k+1] + h11(α)·T·dq[k+1]

with α ∈ [0, 1] the fraction of the way from waypoint k to k+1 and T the time
between them.  The curve passes exactly through every waypoint with the
waypoint's speed, so the motion is smooth across waypoints.
traj_solver/check_playback.py simulates this same interpolation.

The CSV is already in each motor's own units (calibration lives in
traj_solver/motor_calibration.py: motor positions after each joint's 30:1
worm), so positions are sent as they are.  Tolerances and printed angles here
are at the joint, converted with the same table.
Wheel speed columns are not used yet.
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

# ── Motor calibration (shared with traj_solver/export_timing.py) ──────────────
_TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "traj_solver")
sys.path.insert(0, _TRAJ_DIR)
from motor_calibration import JOINTS
CAL = [JOINTS[f"q{j}"] for j in range(1, 5)]      # q1 → q4

# ══════════════════════════════════════════════════════════════════════════════
# Config — must match homing.py
# ══════════════════════════════════════════════════════════════════════════════
PORT      = "/dev/ttyUSB0"
BAUDRATE  = 8_000_000

# Motor IDs for q1 → q4
MOTOR_IDS = [1, 2, 3, 4]

# ── Playback timing ───────────────────────────────────────────────────────────
# Command update rate (Hz).  If the PC cannot send all goals within one
# update, the late count printed at the end says so — lower it if that is large.
INTERP_HZ = 100

# ── Position PID ──────────────────────────────────────────────────────────────
P_GAIN_POS = 5.0
I_GAIN_POS = 0.0
D_GAIN_POS = 0.2
P_GAIN_IQ  = 0.02
I_GAIN_IQ  = 0.02
D_GAIN_IQ  = 0.0
P_GAIN_ID  = 0.02
I_GAIN_ID  = 0.02
D_GAIN_ID  = 0.0
IQ_MAX     = 1.5      # A — torque cap during playback

# How far each joint may be from the start pose before playback asks (rad at the joint)
START_TOL  = math.radians(5.0)

# ── Output directory ──────────────────────────────────────────────────────────
_TIMINGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "traj_solver", "timings",
)

SCENARIO_NAMES = ["floor_to_wall", "wall_to_ceiling", "outside", "thin_edge"]

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def read1(getter, motor_id):
    data_list, err = getter(motor_id)[0]
    return data_list[0], err

def load_csv(scenario):
    """Return (times, positions, velocities): waypoint times (s) and, per
    waypoint, [q1..q4] positions (rad) and speeds (rad/s) in motor units."""
    path = os.path.join(_TIMINGS_DIR, f"{scenario}.csv")
    if not os.path.exists(path):
        print(f"CSV not found: {path}")
        sys.exit(1)
    times      = []
    positions  = []
    velocities = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            times.append(float(row["time_s"]))
            positions.append([float(row[f"q{j}_rad"]) for j in range(1, 5)])
            velocities.append([float(row[f"dq{j}_rad_s"]) for j in range(1, 5)])
    return times, positions, velocities

def hermite(q_k, dq_k, q_k1, dq_k1, T, alpha):
    """
    Cubic Hermite position at normalised time alpha in [0, 1]: the cubic that
    starts at q_k with speed dq_k and ends T seconds later at q_k1 with speed
    dq_k1.  Same formula as traj_solver/check_playback.py.
    """
    a2, a3 = alpha * alpha, alpha * alpha * alpha
    h00, h10 = 2 * a3 - 3 * a2 + 1, a3 - 2 * a2 + alpha
    h01, h11 = -2 * a3 + 3 * a2, a3 - a2
    return [h00 * q_k[j] + h10 * T * dq_k[j] + h01 * q_k1[j] + h11 * T * dq_k1[j]
            for j in range(len(q_k))]

def disable_motor(bear, motor_id, tries=5, delay=0.02):
    for _ in range(tries):
        bear.set_torque_enable((motor_id, 0))
        state, err = read1(bear.get_torque_enable, motor_id)
        if err is not None and state == 0:
            return True
        time.sleep(delay)
    return False

# ══════════════════════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════════════════════
if len(sys.argv) < 2 or sys.argv[1] not in SCENARIO_NAMES:
    print(f"Usage: python playback.py <scenario>")
    print(f"  scenarios: {SCENARIO_NAMES}")
    sys.exit(1)

scenario                  = sys.argv[1]
times, frames, velocities = load_csv(scenario)
n_frames                  = len(frames)

print(f"\nScenario : {scenario}")
print(f"Waypoints: {n_frames}  ({times[-1]:.1f}s, {(n_frames - 1) / times[-1]:g} per second)")
print(f"Command  : every {1000/INTERP_HZ:.0f}ms  (cubic Hermite between waypoints)")

# Start pose (motor units)
target_raw = frames[0]

print(f"\nStart pose (joint angles):")
for i, q in enumerate(target_raw):
    print(f"  q{i+1} = {math.degrees(CAL[i].angle(q)):+.1f}°  (motor {MOTOR_IDS[i]})")

# Connect
print("\nConnecting...")
bear = Manager.BEAR(port=PORT, baudrate=BAUDRATE, single_timeout=0.05, single_try_num=3)

for mid in MOTOR_IDS:
    _, err = bear.ping(mid)[0]
    if err is None:
        print(f"  BEAR {mid}: OFFLINE")
        bear.close_port()
        sys.exit(1)
    print(f"  BEAR {mid}: online")

# Configure position mode
print("\nConfiguring position mode...")
for mid in MOTOR_IDS:
    disable_motor(bear, mid)
    bear.set_p_gain_id((mid, P_GAIN_ID));  bear.set_i_gain_id((mid, I_GAIN_ID));  bear.set_d_gain_id((mid, D_GAIN_ID))
    bear.set_p_gain_iq((mid, P_GAIN_IQ));  bear.set_i_gain_iq((mid, I_GAIN_IQ));  bear.set_d_gain_iq((mid, D_GAIN_IQ))
    bear.set_p_gain_position((mid, P_GAIN_POS))
    bear.set_i_gain_position((mid, I_GAIN_POS))
    bear.set_d_gain_position((mid, D_GAIN_POS))
    bear.set_mode((mid, 2))           # position mode
    bear.set_limit_i_max((mid, IQ_MAX))

# The motors refuse positions outside their position limits; check the whole
# file (the curves between waypoints never pass beyond the waypoints).
outside = False
for j, mid in enumerate(MOTOR_IDS):
    lo = read1(bear.get_limit_position_min, mid)[0]
    hi = read1(bear.get_limit_position_max, mid)[0]
    need_lo = min(f[j] for f in frames)
    need_hi = max(f[j] for f in frames)
    if need_lo < lo or need_hi > hi:
        outside = True
        print(f"  WARNING: q{j+1} needs motor {mid} from {need_lo:+.2f} to {need_hi:+.2f} rad, "
              f"outside its position limits [{lo:+.2f}, {hi:+.2f}] rad")
if outside:
    input("Press Enter to continue anyway, Ctrl-C to abort: ")

# Check we're near the start pose before enabling: an enabled motor drives
# straight to its goal.
current_raw = [read1(bear.get_present_position, mid)[0] for mid in MOTOR_IDS]
errors = [abs(c.angle(cur) - c.angle(t)) for c, cur, t in zip(CAL, current_raw, target_raw)]
max_err = max(errors)

print(f"\nDistance from start pose (joint):")
for i, e in enumerate(errors):
    flag = "  ← WARNING" if e > START_TOL else ""
    print(f"  q{i+1}: {math.degrees(e):.1f}°{flag}")

if max_err > START_TOL:
    print(f"\nWARNING: joints are more than {math.degrees(START_TOL):.0f}° from the start pose;")
    print("enabling will drive them straight there.  Run homing.py first, or press Enter to continue anyway.")
    input("Press Enter to continue, Ctrl-C to abort: ")
else:
    print("  All joints within tolerance.")

print("\nEnabling motors...")
for mid, pos in zip(MOTOR_IDS, target_raw):
    bear.set_goal_position((mid, pos))   # prime goal before enabling
    bear.set_torque_enable((mid, 1))

# Countdown
print("\nStarting playback in:")
for i in (3, 2, 1):
    print(f"  {i}...")
    time.sleep(1.0)
print("  GO\n")

# ══════════════════════════════════════════════════════════════════════════════
# Playback loop  (cubic Hermite interpolation)
# ══════════════════════════════════════════════════════════════════════════════
n_ticks = int(round(times[-1] * INTERP_HZ))
k       = 0       # current segment: times[k] <= t <= times[k + 1]
late    = 0       # updates sent more than one update interval behind schedule

t_start = time.time()

try:
    for tick in range(n_ticks + 1):
        t = min(tick / INTERP_HZ, times[-1])
        wait = t_start + t - time.time()
        if wait > 0:
            time.sleep(wait)
        elif wait < -1.0 / INTERP_HZ:
            late += 1

        while k < n_frames - 2 and t >= times[k + 1]:
            k += 1
            print(f"\r  waypoint {k:3d}/{n_frames - 1}  t={time.time() - t_start:.2f}s",
                  end="", flush=True)
        T     = times[k + 1] - times[k]
        alpha = min(max((t - times[k]) / T, 0.0), 1.0)
        goals = hermite(frames[k], velocities[k], frames[k + 1], velocities[k + 1], T, alpha)
        for mid, g in zip(MOTOR_IDS, goals):
            bear.set_goal_position((mid, g))

    # Hold final pose
    for mid, g in zip(MOTOR_IDS, frames[-1]):
        bear.set_goal_position((mid, g))

    total = time.time() - t_start
    print(f"\n\nPlayback complete — {total:.2f}s of {times[-1]:.2f}s planned  ({n_frames} waypoints)")
    print(f"Late updates: {late} of {n_ticks + 1}"
          + ("  ← the PC could not keep up; lower INTERP_HZ" if late > 0.01 * (n_ticks + 1) else ""))

except KeyboardInterrupt:
    print("\n\nInterrupted — holding current position.")

finally:
    input("\nPress Enter to disable motors.")
    for mid in MOTOR_IDS:
        try:
            bear.set_torque_enable((mid, 0))
        except Exception as e:
            print(f"  BEAR {mid}: disable failed ({e!r})")
    bear.close_port()
    print("All motors disabled, port closed.")
