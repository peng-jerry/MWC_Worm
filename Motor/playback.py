"""
playback.py — replay a timing CSV through the q1 and w_fl BEAR motors.

Usage:
    python playback.py floor_to_wall
    python playback.py wall_to_ceiling
    python playback.py outside
    python playback.py thin_edge

Run homing.py first to reach the start pose before executing this script.

Motor mapping (matches the Xbox roles in xbox_live.py):
  BEAR 3 (left-stick motor)  → q1,   POSITION mode, follows q1_rad / dq1_rad_s
  BEAR 1 (right-stick motor) → w_fl, VELOCITY mode, follows w_fl_rad_s
The other joints/wheels in the CSV (q2-q4, w_fr, w_bl, w_br) are not driven.

Each CSV row is a frame at PHYSICAL_FPS = 3 fr/s (333 ms apart).  Between
frames the script generates sub-steps at INTERP_HZ.  q1 uses Tustin's method
(trapezoidal velocity integration):

    q(α) = q[k]  +  T · (α·dq[k]  +  α²/2 · (dq[k+1] − dq[k]))

where α ∈ [0,1] is the normalised position within the frame interval and T is
the frame period.  Velocity ramps linearly between the two CSV samples, so
the position curve is a parabolic arc rather than a straight line.  At each
frame boundary q is hard-reset to the CSV value to prevent drift.  The w_fl
velocity command is linearly interpolated between frames.

Q1_OFFSET must match JOINT_OFFSET[0] in homing.py — both use the same calibration.
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

# Motor IDs
Q1_MOTOR_ID  = 3     # left-stick motor  → q1 joint (position mode)
WFL_MOTOR_ID = 1     # right-stick motor → w_fl wheel (velocity mode)
MOTOR_IDS    = [Q1_MOTOR_ID, WFL_MOTOR_ID]

# Gear reduction between the BEAR output and the q1 joint (motor turns 30x per joint turn)
GEAR_RATIO = 30.0

# Raw (BEAR) position when q1_display == 0:  raw = q1_display * GEAR_RATIO + Q1_OFFSET
Q1_OFFSET = 0.0

# q1 travels ±95° from straight up (±49.74 rad at the motor)
Q1_LIMIT = 49.9                                # motor rad (≈95.3° at the joint); the BEAR silently ignores limits above ~50 rad
Q1_VEL_MAX = 15.0                              # motor rad/s — CSV peaks at ~11 (0.37 joint rad/s × 30)

# Flip to -1.0 if the wheel spins the wrong way relative to the CSV
WFL_SIGN = 1.0

# Velocity-mode settings for the w_fl wheel (same as right_stick_speed in xbox_live.py)
WFL_P_GAIN_VEL = 0.5
WFL_IQ_MAX     = 3.0
WFL_VEL_MAX    = 4.0     # rad/s — CSV peaks at 2.0

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

def display_to_raw(q1_display):
    return q1_display * GEAR_RATIO + Q1_OFFSET

def load_csv(scenario):
    """Return (q1 joint rad, dq1 motor rad/s, w_fl rad/s) — one float per CSV frame."""
    path = os.path.join(_TIMINGS_DIR, f"{scenario}.csv")
    if not os.path.exists(path):
        print(f"CSV not found: {path}")
        sys.exit(1)
    q1, dq1, w_fl = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            q1.append(float(row["q1_rad"]))
            dq1.append(float(row["dq1_rad_s"]) * GEAR_RATIO)   # motor rad/s
            w_fl.append(float(row["w_fl_rad_s"]))
    return q1, dq1, w_fl

def tustin_interp(q_k, dq_k, dq_k1, T, alpha):
    """
    Trapezoidal (Tustin) position at normalised time alpha in [0, 1].

    Velocity ramps linearly from dq_k to dq_k1 over the frame interval T.
    Integrating gives a parabolic position arc:

        q(α) = q[k]  +  T · (α·dq[k]  +  α²/2 · (dq[k+1] − dq[k]))
    """
    return q_k + T * (alpha * dq_k + 0.5 * alpha**2 * (dq_k1 - dq_k))

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

# ══════════════════════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════════════════════
if len(sys.argv) < 2 or sys.argv[1] not in SCENARIO_NAMES:
    print(f"Usage: python playback.py <scenario>")
    print(f"  scenarios: {SCENARIO_NAMES}")
    sys.exit(1)

scenario           = sys.argv[1]
q1_frames, dq1_frames, wfl_frames = load_csv(scenario)
n_frames           = len(q1_frames)

print(f"\nScenario : {scenario}")
print(f"Frames   : {n_frames}  ({n_frames / PHYSICAL_FPS:.1f}s at {PHYSICAL_FPS} fr/s)")
print(f"Command  : every {1000/INTERP_HZ:.0f}ms  "
      f"(~{round(INTERP_HZ / PHYSICAL_FPS)} steps per CSV frame, Tustin interp)")

# Start pose
q1_start_raw = display_to_raw(q1_frames[0])

print(f"\nStart pose:")
print(f"  q1   = {math.degrees(q1_frames[0]):+.1f}°  (BEAR {Q1_MOTOR_ID}, position)")
print(f"  w_fl = {wfl_frames[0]:+.2f} rad/s  (BEAR {WFL_MOTOR_ID}, velocity)")

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
print("\nConfiguring q1 (position) and w_fl (velocity)...")
for mid in MOTOR_IDS:
    disable_motor(bear, mid)
    bear.set_p_gain_id((mid, P_GAIN_ID));  bear.set_i_gain_id((mid, I_GAIN_ID));  bear.set_d_gain_id((mid, D_GAIN_ID))
    bear.set_p_gain_iq((mid, P_GAIN_IQ));  bear.set_i_gain_iq((mid, I_GAIN_IQ));  bear.set_d_gain_iq((mid, D_GAIN_IQ))

bear.set_p_gain_position((Q1_MOTOR_ID, P_GAIN_POS))
bear.set_i_gain_position((Q1_MOTOR_ID, I_GAIN_POS))
bear.set_d_gain_position((Q1_MOTOR_ID, D_GAIN_POS))
bear.set_limit_i_max((Q1_MOTOR_ID, IQ_MAX))
bear.set_limit_velocity_max((Q1_MOTOR_ID, Q1_VEL_MAX))
bear.set_limit_position_min((Q1_MOTOR_ID, Q1_OFFSET - Q1_LIMIT))
bear.set_limit_position_max((Q1_MOTOR_ID, Q1_OFFSET + Q1_LIMIT))
_lim, _ = read1(bear.get_limit_position_max, Q1_MOTOR_ID)
if abs(_lim - (Q1_OFFSET + Q1_LIMIT)) > 1e-3:
    print(f"  WARNING: BEAR {Q1_MOTOR_ID} ignored the position limit (reads {_lim:.3f})")
if not set_mode_confirmed(bear, Q1_MOTOR_ID, 2):           # position mode
    print(f"  WARNING: could not confirm position mode on BEAR {Q1_MOTOR_ID}")

bear.set_p_gain_velocity((WFL_MOTOR_ID, WFL_P_GAIN_VEL))
bear.set_i_gain_velocity((WFL_MOTOR_ID, 0.0))
bear.set_d_gain_velocity((WFL_MOTOR_ID, 0.0))
bear.set_limit_i_max((WFL_MOTOR_ID, WFL_IQ_MAX))
bear.set_limit_velocity_max((WFL_MOTOR_ID, WFL_VEL_MAX))
bear.set_goal_velocity((WFL_MOTOR_ID, 0.0))
if not set_mode_confirmed(bear, WFL_MOTOR_ID, 1):          # velocity mode
    print(f"  WARNING: could not confirm velocity mode on BEAR {WFL_MOTOR_ID}")

# Enable and check q1 is near the start pose
print("\nEnabling motors...")
current_q1 = read1(bear.get_present_position, Q1_MOTOR_ID)[0]
bear.set_goal_position((Q1_MOTOR_ID, q1_start_raw))   # prime goal before enabling
bear.set_torque_enable((Q1_MOTOR_ID, 1))
bear.set_torque_enable((WFL_MOTOR_ID, 1))

q1_err = abs(current_q1 - q1_start_raw) / GEAR_RATIO   # joint rad
print(f"\nDistance from start pose:")
print(f"  q1: {math.degrees(q1_err):.1f}°{'  ← WARNING' if q1_err > START_TOL else ''}")

if q1_err > START_TOL:
    print(f"\nWARNING: q1 is more than {math.degrees(START_TOL):.0f}° from the start pose.")
    print("Run homing.py first, or press Enter to continue anyway.")
    input("Press Enter to continue, Ctrl-C to abort: ")
else:
    print("  q1 within tolerance.")

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
        q_k   = display_to_raw(q1_frames[fi])
        dq_k  = dq1_frames[fi]
        dq_k1 = dq1_frames[fi + 1]
        w_k   = wfl_frames[fi]
        w_k1  = wfl_frames[fi + 1]

        for step in range(n_interp):
            alpha = step / n_interp                   # 0 → just under 1
            t_cmd = t_start + (fi + alpha) * frame_dt
            # Tustin interp works in raw encoder space (q_k already offset-corrected)
            bear.set_goal_position((Q1_MOTOR_ID, tustin_interp(q_k, dq_k, dq_k1, frame_dt, alpha)))
            bear.set_goal_velocity((WFL_MOTOR_ID, WFL_SIGN * (w_k + alpha * (w_k1 - w_k))))

            wait = t_cmd - time.time()
            if wait > 0:
                time.sleep(wait)

        elapsed = time.time() - t_start
        print(f"\r  frame {fi+1:3d}/{n_frames}  t={elapsed:.2f}s", end="", flush=True)

    # Hold final q1 pose, stop the wheel
    bear.set_goal_position((Q1_MOTOR_ID, display_to_raw(q1_frames[-1])))
    bear.set_goal_velocity((WFL_MOTOR_ID, 0.0))

    total = time.time() - t_start
    print(f"\n\nPlayback complete — {total:.2f}s  ({n_frames} frames)")

except KeyboardInterrupt:
    print("\n\nInterrupted — holding current position.")

finally:
    try:
        bear.set_goal_velocity((WFL_MOTOR_ID, 0.0))   # never leave the wheel spinning
    except Exception as e:
        print(f"  BEAR {WFL_MOTOR_ID}: wheel stop failed ({e!r})")
    input("\nPress Enter to disable motors.")
    for mid in MOTOR_IDS:
        try:
            bear.set_torque_enable((mid, 0))
        except Exception as e:
            print(f"  BEAR {mid}: disable failed ({e!r})")
    bear.close_port()
    print("All motors disabled, port closed.")
