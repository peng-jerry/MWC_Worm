"""
playback.py — replay a timing CSV through the q1 and front_a BEAR motors.

Usage:
    python playback.py floor_to_wall
    python playback.py wall_to_ceiling
    python playback.py outside
    python playback.py thin_edge

Run homing.py first to reach the start pose before executing this script.

Motor mapping (matches the Xbox roles in xbox_live.py):
  BEAR 3 (left-stick motor)  → q1,       POSITION mode, follows q1_rad / dq1_rad_s
  BEAR 1 (right-stick motor) → front_a,  VELOCITY mode, follows w_front_a_rad_s
The other joints/wheels in the CSV (q2-q4, back_a) are not driven — only
these two motors are wired on this robot right now.

Each CSV row is a waypoint at time time_s (traj_solver/export_timing.py).
Every 1/INTERP_HZ seconds the script sends q1 the point at that moment on the
cubic through its position and speed at the two surrounding waypoints (cubic
Hermite interpolation):

    q(α) = h00(α)·q[k] + h10(α)·T·dq[k] + h01(α)·q[k+1] + h11(α)·T·dq[k+1]

with α ∈ [0, 1] the fraction of the way from waypoint k to k+1 and T the time
between them.  The curve passes exactly through every waypoint with the
waypoint's speed, so the motion is smooth across waypoints.
traj_solver/check_playback.py simulates this same interpolation for q1-q4.
The front_a wheel velocity is linearly interpolated between waypoints.

The CSV is already in each motor's own units (calibration lives in
traj_solver/motor_calibration.py: motor positions after each joint's 30:1
worm), so q1 positions are sent as they are.  Tolerances and printed angles
here are at the joint, converted with the same table.
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
from motor_calibration import JOINTS, WHEELS
Q1_CAL   = JOINTS["q1"]
WFA_CAL  = WHEELS["front_a"]

# ══════════════════════════════════════════════════════════════════════════════
# Config — must match homing.py
# ══════════════════════════════════════════════════════════════════════════════
PORT      = "/dev/ttyUSB0"
BAUDRATE  = 8_000_000

# Motor IDs
Q1_MOTOR_ID  = 3     # left-stick motor  → q1 joint (position mode)
WFA_MOTOR_ID = 1     # right-stick motor → front_a wheel (velocity mode)
MOTOR_IDS    = [Q1_MOTOR_ID, WFA_MOTOR_ID]

# q1 travels ±95° from straight up (±49.74 rad at the motor after the 30:1 worm).
# The BEAR silently ignores position-limit writes above ~50 rad (measured on
# this hardware), so 49.9 rad is the largest safe ceiling.
Q1_LIMIT_MOTOR = 49.9

# Flip to -1.0 if the wheel spins the wrong way relative to the CSV
WFA_SIGN = 1.0

# Velocity-mode settings for the front_a wheel (same as right_stick_speed in xbox_live.py)
WFA_P_GAIN_VEL = 0.5
WFA_IQ_MAX     = 3.0
WFA_VEL_MAX    = 4.0     # rad/s — CSV peaks at ~2

# ── Playback timing ───────────────────────────────────────────────────────────
# Command update rate (Hz).  If the PC cannot send all goals within one
# update, the late count printed at the end says so — lower it if that is large.
INTERP_HZ = 100

# ── Position PID (q1) ─────────────────────────────────────────────────────────
P_GAIN_POS = 5.0
I_GAIN_POS = 0.0
D_GAIN_POS = 0.2
P_GAIN_IQ  = 0.02
I_GAIN_IQ  = 0.02
D_GAIN_IQ  = 0.0
P_GAIN_ID  = 0.02
I_GAIN_ID  = 0.02
D_GAIN_ID  = 0.0
IQ_MAX     = 1.5      # A — torque cap during playback (q1)

# How far q1 may be from the start pose before playback asks (rad at the joint)
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
    """Return (times, q1 motor rad, dq1 motor rad/s, front_a wheel rad/s) — one entry per waypoint."""
    path = os.path.join(_TIMINGS_DIR, f"{scenario}.csv")
    if not os.path.exists(path):
        print(f"CSV not found: {path}")
        sys.exit(1)
    times, q1, dq1, wfa = [], [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            times.append(float(row["time_s"]))
            q1.append(float(row["q1_rad"]))
            dq1.append(float(row["dq1_rad_s"]))
            wfa.append(float(row["w_front_a_rad_s"]))
    return times, q1, dq1, wfa

def hermite(q_k, dq_k, q_k1, dq_k1, T, alpha):
    """
    Cubic Hermite position at normalised time alpha in [0, 1]: the cubic that
    starts at q_k with speed dq_k and ends T seconds later at q_k1 with speed
    dq_k1.  Same formula as traj_solver/check_playback.py.
    """
    a2, a3 = alpha * alpha, alpha * alpha * alpha
    h00, h10 = 2 * a3 - 3 * a2 + 1, a3 - 2 * a2 + alpha
    h01, h11 = -2 * a3 + 3 * a2, a3 - a2
    return h00 * q_k + h10 * T * dq_k + h01 * q_k1 + h11 * T * dq_k1

def set_mode_confirmed(bear, motor_id, mode, tries=5, delay=0.02):
    for _ in range(tries):
        bear.set_mode((motor_id, mode))
        actual, err = read1(bear.get_mode, motor_id)
        if err is not None and actual == mode:
            return True
        time.sleep(delay)
    return False

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

# ══════════════════════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════════════════════
if len(sys.argv) < 2 or sys.argv[1] not in SCENARIO_NAMES:
    print(f"Usage: python playback.py <scenario>")
    print(f"  scenarios: {SCENARIO_NAMES}")
    sys.exit(1)

scenario                   = sys.argv[1]
times, q1_frames, dq1_frames, wfa_frames = load_csv(scenario)
n_frames                   = len(q1_frames)

print(f"\nScenario : {scenario}")
print(f"Waypoints: {n_frames}  ({times[-1]:.1f}s, {(n_frames - 1) / times[-1]:g} per second)")
print(f"Command  : every {1000/INTERP_HZ:.0f}ms  (cubic Hermite q1, linear front_a)")

# Start pose
q1_start_raw = q1_frames[0]

print(f"\nStart pose:")
print(f"  q1       = {math.degrees(Q1_CAL.angle(q1_start_raw)):+.1f}°  (BEAR {Q1_MOTOR_ID}, position)")
print(f"  front_a  = {WFA_CAL.rate(wfa_frames[0]):+.2f} rad/s  (BEAR {WFA_MOTOR_ID}, velocity)")

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

# Configure motors
print("\nConfiguring q1 (position) and front_a (velocity)...")
for mid in MOTOR_IDS:
    disable_motor(bear, mid)
    bear.set_p_gain_id((mid, P_GAIN_ID));  bear.set_i_gain_id((mid, I_GAIN_ID));  bear.set_d_gain_id((mid, D_GAIN_ID))
    bear.set_p_gain_iq((mid, P_GAIN_IQ));  bear.set_i_gain_iq((mid, I_GAIN_IQ));  bear.set_d_gain_iq((mid, D_GAIN_IQ))

bear.set_p_gain_position((Q1_MOTOR_ID, P_GAIN_POS))
bear.set_i_gain_position((Q1_MOTOR_ID, I_GAIN_POS))
bear.set_d_gain_position((Q1_MOTOR_ID, D_GAIN_POS))
bear.set_limit_i_max((Q1_MOTOR_ID, IQ_MAX))
bear.set_limit_position_min((Q1_MOTOR_ID, -Q1_LIMIT_MOTOR))
bear.set_limit_position_max((Q1_MOTOR_ID, Q1_LIMIT_MOTOR))
_lim, _ = read1(bear.get_limit_position_max, Q1_MOTOR_ID)
if abs(_lim - Q1_LIMIT_MOTOR) > 1e-3:
    print(f"  WARNING: BEAR {Q1_MOTOR_ID} ignored the position limit (reads {_lim:.3f})")
if not set_mode_confirmed(bear, Q1_MOTOR_ID, 2):           # position mode
    print(f"  WARNING: could not confirm position mode on BEAR {Q1_MOTOR_ID}")

bear.set_p_gain_velocity((WFA_MOTOR_ID, WFA_P_GAIN_VEL))
bear.set_i_gain_velocity((WFA_MOTOR_ID, 0.0))
bear.set_d_gain_velocity((WFA_MOTOR_ID, 0.0))
bear.set_limit_i_max((WFA_MOTOR_ID, WFA_IQ_MAX))
bear.set_limit_velocity_max((WFA_MOTOR_ID, WFA_VEL_MAX))
bear.set_goal_velocity((WFA_MOTOR_ID, 0.0))
if not set_mode_confirmed(bear, WFA_MOTOR_ID, 1):          # velocity mode
    print(f"  WARNING: could not confirm velocity mode on BEAR {WFA_MOTOR_ID}")

# The q1 motor refuses positions outside its position limit; check the whole
# file (the curve between waypoints never passes beyond the waypoints).
need_lo, need_hi = min(q1_frames), max(q1_frames)
if need_lo < -Q1_LIMIT_MOTOR or need_hi > Q1_LIMIT_MOTOR:
    print(f"  WARNING: q1 needs BEAR {Q1_MOTOR_ID} from {need_lo:+.2f} to {need_hi:+.2f} rad, "
          f"outside its position limit ±{Q1_LIMIT_MOTOR:.2f} rad")
    input("Press Enter to continue anyway, Ctrl-C to abort: ")

# Check we're near the start pose before enabling: an enabled motor drives
# straight to its goal.
current_q1 = read1(bear.get_present_position, Q1_MOTOR_ID)[0]
q1_err = abs(Q1_CAL.angle(current_q1) - Q1_CAL.angle(q1_start_raw))

print(f"\nDistance from start pose:")
print(f"  q1: {math.degrees(q1_err):.1f}°{'  ← WARNING' if q1_err > START_TOL else ''}")

if q1_err > START_TOL:
    print(f"\nWARNING: q1 is more than {math.degrees(START_TOL):.0f}° from the start pose;")
    print("enabling will drive it straight there.  Run homing.py first, or press Enter to continue anyway.")
    input("Press Enter to continue, Ctrl-C to abort: ")
else:
    print("  q1 within tolerance.")

print("\nEnabling motors...")
bear.set_goal_position((Q1_MOTOR_ID, q1_start_raw))   # prime goal before enabling
enabled, en, en_err = enable_motor(bear, Q1_MOTOR_ID)
if not enabled:
    print(f"  WARNING: BEAR {Q1_MOTOR_ID} did not enable (torque_enable={en}, err={en_err})")
enabled, en, en_err = enable_motor(bear, WFA_MOTOR_ID)
if not enabled:
    print(f"  WARNING: BEAR {WFA_MOTOR_ID} did not enable (torque_enable={en}, err={en_err})")

# Countdown
print("\nStarting playback in:")
for i in (3, 2, 1):
    print(f"  {i}...")
    time.sleep(1.0)
print("  GO\n")

# ══════════════════════════════════════════════════════════════════════════════
# Playback loop  (cubic Hermite for q1, linear for front_a)
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

        q1_goal = hermite(q1_frames[k], dq1_frames[k], q1_frames[k + 1], dq1_frames[k + 1], T, alpha)
        bear.set_goal_position((Q1_MOTOR_ID, q1_goal))

        wfa_goal = wfa_frames[k] + alpha * (wfa_frames[k + 1] - wfa_frames[k])
        bear.set_goal_velocity((WFA_MOTOR_ID, WFA_SIGN * wfa_goal))

    # Hold final q1 pose, stop the wheel
    bear.set_goal_position((Q1_MOTOR_ID, q1_frames[-1]))
    bear.set_goal_velocity((WFA_MOTOR_ID, 0.0))

    total = time.time() - t_start
    print(f"\n\nPlayback complete — {total:.2f}s of {times[-1]:.2f}s planned  ({n_frames} waypoints)")
    print(f"Late updates: {late} of {n_ticks + 1}"
          + ("  ← the PC could not keep up; lower INTERP_HZ" if late > 0.01 * (n_ticks + 1) else ""))

except KeyboardInterrupt:
    print("\n\nInterrupted — holding current position.")

finally:
    try:
        bear.set_goal_velocity((WFA_MOTOR_ID, 0.0))   # never leave the wheel spinning
    except Exception as e:
        print(f"  BEAR {WFA_MOTOR_ID}: wheel stop failed ({e!r})")
    input("\nPress Enter to disable motors.")
    for mid in MOTOR_IDS:
        try:
            bear.set_torque_enable((mid, 0))
        except Exception as e:
            print(f"  BEAR {mid}: disable failed ({e!r})")
    bear.close_port()
    print("All motors disabled, port closed.")
