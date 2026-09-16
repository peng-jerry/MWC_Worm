"""
playback.py — replay a timing CSV through the 4 chain-joint BEAR motors.

Usage:
    python playback.py floor_to_wall
    python playback.py wall_to_ceiling
    python playback.py outside
    python playback.py thin_edge

Run homing.py first to reach the start pose before executing this script.

Each CSV row is a frame at PHYSICAL_FPS = 3 fr/s (333 ms apart).  Between
frames the script generates sub-steps at INTERP_HZ using Tustin's method
(trapezoidal velocity integration):

    q(α) = q[k]  +  T · (α·dq[k]  +  α²/2 · (dq[k+1] − dq[k]))

where α ∈ [0,1] is the normalised position within the frame interval and T is
the frame period.  Velocity ramps linearly between the two CSV samples, so
the position curve is a parabolic arc rather than a straight line.  At each
frame boundary q is hard-reset to the CSV value to prevent drift.

JOINT_OFFSET must match homing.py — both scripts use the same calibration.
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

# ══════════════════════════════════════════════════════════════════════════════
# Config — must match homing.py
# ══════════════════════════════════════════════════════════════════════════════
PORT      = "/dev/ttyUSB0"
BAUDRATE  = 8_000_000

# Motor IDs for q1 → q4
MOTOR_IDS = [1, 2, 3, 4]

# Raw encoder value when q_display == 0  (set from physical calibration)
JOINT_OFFSET = [0.0, 0.0, 0.0, 0.0]

# ── Playback timing ───────────────────────────────────────────────────────────
PHYSICAL_FPS = 3      # fr/s — must match robot_timing.py
INTERP_HZ    = 20     # command update rate (Hz); interpolates between CSV frames

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

# Arrival tolerance before playback starts (rad)
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

def display_to_raw(q_display):
    return [q + JOINT_OFFSET[i] for i, q in enumerate(q_display)]

def load_csv(scenario):
    """Return (positions, velocities) — each a list of 4-element lists [q1..q4]."""
    path = os.path.join(_TIMINGS_DIR, f"{scenario}.csv")
    if not os.path.exists(path):
        print(f"CSV not found: {path}")
        sys.exit(1)
    positions  = []
    velocities = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            positions.append([
                float(row["q1_rad"]),
                float(row["q2_rad"]),
                float(row["q3_rad"]),
                float(row["q4_rad"]),
            ])
            velocities.append([
                float(row["dq1_rad_s"]),
                float(row["dq2_rad_s"]),
                float(row["dq3_rad_s"]),
                float(row["dq4_rad_s"]),
            ])
    return positions, velocities

def tustin_interp(q_k, dq_k, dq_k1, T, alpha):
    """
    Trapezoidal (Tustin) position at normalised time alpha in [0, 1].

    Velocity ramps linearly from dq_k to dq_k1 over the frame interval T.
    Integrating gives a parabolic position arc:

        q(α) = q[k]  +  T · (α·dq[k]  +  α²/2 · (dq[k+1] − dq[k]))
    """
    return [
        q_k[j] + T * (alpha * dq_k[j] + 0.5 * alpha**2 * (dq_k1[j] - dq_k[j]))
        for j in range(len(q_k))
    ]

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

scenario           = sys.argv[1]
frames, velocities = load_csv(scenario)
n_frames           = len(frames)

print(f"\nScenario : {scenario}")
print(f"Frames   : {n_frames}  ({n_frames / PHYSICAL_FPS:.1f}s at {PHYSICAL_FPS} fr/s)")
print(f"Command  : every {1000/INTERP_HZ:.0f}ms  "
      f"(~{round(INTERP_HZ / PHYSICAL_FPS)} steps per CSV frame, Tustin interp)")

# Start pose
target_display = frames[0]
target_raw     = display_to_raw(target_display)

print(f"\nStart pose:")
for i, q in enumerate(target_display):
    print(f"  q{i+1} = {math.degrees(q):+.1f}°  (motor {MOTOR_IDS[i]})")

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

# Enable and check we're near the start pose
print("\nEnabling motors...")
for mid, pos in zip(MOTOR_IDS, target_raw):
    bear.set_goal_position((mid, pos))   # prime goal before enabling
    bear.set_torque_enable((mid, 1))

current_raw = [read1(bear.get_present_position, mid)[0] for mid in MOTOR_IDS]
errors = [abs(c - t) for c, t in zip(current_raw, target_raw)]
max_err = max(errors)

print(f"\nDistance from start pose:")
for i, e in enumerate(errors):
    flag = "  ← WARNING" if e > START_TOL else ""
    print(f"  q{i+1}: {math.degrees(e):.1f}°{flag}")

if max_err > START_TOL:
    print(f"\nWARNING: joints are more than {math.degrees(START_TOL):.0f}° from the start pose.")
    print("Run homing.py first, or press Enter to continue anyway.")
    input("Press Enter to continue, Ctrl-C to abort: ")
else:
    print("  All joints within tolerance.")

# Countdown
print("\nStarting playback in:")
for i in (3, 2, 1):
    print(f"  {i}...")
    time.sleep(1.0)
print("  GO\n")

# ══════════════════════════════════════════════════════════════════════════════
# Playback loop  (Tustin / trapezoidal interpolation)
# ══════════════════════════════════════════════════════════════════════════════
frame_dt = 1.0 / PHYSICAL_FPS                        # 0.333 s per CSV row
n_interp = max(1, round(frame_dt * INTERP_HZ))        # ~7 sub-steps per frame

t_start = time.time()

try:
    for fi in range(n_frames - 1):
        # Hard-reset to CSV position at frame boundary — prevents drift
        q_k   = display_to_raw(frames[fi])
        dq_k  = velocities[fi]
        dq_k1 = velocities[fi + 1]

        for step in range(n_interp):
            alpha = step / n_interp                   # 0 → just under 1
            t_cmd = t_start + (fi + alpha) * frame_dt
            # Tustin interp works in raw encoder space (q_k already offset-corrected)
            goals = tustin_interp(q_k, dq_k, dq_k1, frame_dt, alpha)
            for mid, g in zip(MOTOR_IDS, goals):
                bear.set_goal_position((mid, g))

            wait = t_cmd - time.time()
            if wait > 0:
                time.sleep(wait)

        elapsed = time.time() - t_start
        print(f"\r  frame {fi+1:3d}/{n_frames}  t={elapsed:.2f}s", end="", flush=True)

    # Hold final pose
    final_raw = display_to_raw(frames[-1])
    for mid, g in zip(MOTOR_IDS, final_raw):
        bear.set_goal_position((mid, g))

    total = time.time() - t_start
    print(f"\n\nPlayback complete — {total:.2f}s  ({n_frames} frames)")

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
