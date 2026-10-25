"""
playback.py — replay a timing CSV on the robot: chain joints q1–q4, both
driven wheels, and the two swerve motors.

Usage:
    python playback.py <scenario>                 joint positions from the calibration table
    python playback.py <scenario> --start-pose    bench test before calibration (see below)
    scenarios: floor_to_wall, wall_to_ceiling, outside, thin_edge

Motors (BEAR IDs, as wired):
  q1 3, q2 4, q3 5, q4 6          POSITION mode, follow q*_rad / dq*_rad_s along cubic
                                  Hermite curves
  front wheel 1, back wheel 8     VELOCITY mode, follow w_front_a / w_back_a (arm A
                                  wheels), speed linear between waypoints
  front swerve 2, back swerve 7   POSITION mode, held where they are when playback
                                  starts.  The plan is in the side plane, so the wheels
                                  must keep pointing straight: align them first (e.g. with
                                  xbox_live.py).

Without --start-pose, run homing.py first: the joints are driven to the file's
absolute angles, which need calibrated zeros (traj_solver/motor_calibration.py).

--start-pose: until the joints are calibrated their motor readings mean nothing
absolute.  With this flag, place the robot by hand in the transition's start
pose (printed at startup); playback reads each joint motor and takes that
reading as the file's first row, so every joint moves relative to where it
really is.  Nothing in motor_calibration.py changes; it is only as accurate as
the placement and applies to this run only.

Each CSV row is a waypoint at time time_s (traj_solver/export_timing.py).
Every 1/INTERP_HZ seconds the script sends each joint the point at that
moment on the cubic through its position and speed at the two surrounding
waypoints:

    q(α) = h00(α)·q[k] + h10(α)·T·dq[k] + h01(α)·q[k+1] + h11(α)·T·dq[k+1]

with α ∈ [0, 1] the fraction of the way from waypoint k to k+1 and T the time
between them.  The curve passes exactly through every waypoint with the
waypoint's speed.  traj_solver/check_playback.py simulates the same curves.

The CSV is in each motor's own units (motor positions after each joint's 30:1
worm; wheel directions from their signs in motor_calibration.py), so values are
sent as they are.  Tolerances and printed angles here are at the joint,
converted with the same table.

Feedback: every update the script reads one motor's status registers in one
packet (diagnostics.STATS: enabled, goal, position, velocity, currents, supply
voltage, temperatures, plus the error code), taking the motors in turn.  It
stops early — wheels to 0, joints and swerves held where they are — if a joint
is more than TRACK_ABORT from where it should be, a swerve has drifted more
than SWERVE_DRIFT_ABORT, a motor reports an error or is no longer enabled, or
a motor misses READ_FAILS_ABORT reads in a row.  Wheels follow a speed, which
is logged but not checked (their velocity loop lags on every speed change and
a wheel can legitimately slip).  Every reading is written to
logs/playback_<scenario>_<time>.csv.
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

from diagnostics import STATS, ERROR_BITS, read_stats, decode_code

# ── Motor calibration (shared with traj_solver/export_timing.py) ──────────────
_TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "traj_solver")
sys.path.insert(0, _TRAJ_DIR)
from motor_calibration import JOINTS

# ══════════════════════════════════════════════════════════════════════════════
# Config — must match homing.py
# ══════════════════════════════════════════════════════════════════════════════
PORT      = "/dev/serial/by-id/usb-Westwood_Robotics_USB2BEAR_Boosted_UB0104-if00-port0"   # as xbox_live.py
BAUDRATE  = 8_000_000

# Motor IDs
JOINT_IDS  = {"q1": 3, "q2": 4, "q3": 5, "q4": 6}
WHEEL_IDS  = {"front_a": 1, "back_a": 8}            # arm A wheels; arm B wheels spin freely
SWERVE_IDS = {"front_swerve": 2, "back_swerve": 7}
NAMES      = {mid: name for ids in (JOINT_IDS, WHEEL_IDS, SWERVE_IDS) for name, mid in ids.items()}
ALL_IDS    = list(NAMES)

# Joints travel ±95° from their zero (±49.74 rad at the motor after the 30:1
# worm).  The BEAR silently ignores position-limit writes above ~50 rad
# (measured on this hardware), so 49.9 rad is the largest safe ceiling.
POS_LIMIT_MOTOR = 49.9

# Joint speed and acceleration caps at the motor, set every run: a motor
# otherwise keeps whatever the last script left (xbox_live.py sets its
# velocity caps to 4.0 rad/s, below what playback needs).
JOINT_VEL_MAX = {n: math.radians(21.6) * JOINTS[n].ratio for n in JOINT_IDS}   # 11.3 rad/s: the 21.6 °/s
                                                    # joint limit (traj_solver/core/params.py MAX_JOINT_SPEED)
JOINT_ACC_MAX = 120.0   # rad/s² — about twice the most any timing file needs (61 rad/s²).
                        # Position mode limits acceleration to this; the BEAR default
                        # (5 rad/s²) would make a joint lag ~2 s behind every speed-up.
JOINT_IQ_MAX  = 1.5     # A — torque cap during playback

# Velocity-mode settings for the wheels (as right_stick_speed in xbox_live.py)
WHEEL_P_GAIN_VEL = 0.5
WHEEL_IQ_MAX     = 3.0
WHEEL_VEL_MAX    = 5.0     # rad/s — the timing files peak at 4.19 (thin_edge)

# Swerves only hold their start position
SWERVE_IQ_MAX       = 1.5      # A
SWERVE_VEL_MAX      = 2.0      # rad/s — slow if one is ever pushed off and returns
SWERVE_DRIFT_ABORT  = 0.2      # rad at the motor — a swerve this far off stops playback

# ── Playback timing ───────────────────────────────────────────────────────────
# Command update rate (Hz).  If the PC cannot send all goals within one
# update, the late count printed at the end says so — lower it if that is large.
INTERP_HZ = 100

# ── Position PID (joints and swerves) ─────────────────────────────────────────
P_GAIN_POS = 5.0
I_GAIN_POS = 0.0
D_GAIN_POS = 0.2
# Current-loop gains, every motor: BEAR SDK Table 6 for the Koala Muscle Build V2
# (the values xbox_live.py runs on the robot).
P_GAIN_IQ  = 0.358
I_GAIN_IQ  = 0.045
D_GAIN_IQ  = 0.0
P_GAIN_ID  = 0.358
I_GAIN_ID  = 0.045
D_GAIN_ID  = 0.0

# How far each joint may be from the start pose before playback asks (rad at the joint)
START_TOL  = math.radians(5.0)

# ── Feedback while playing ────────────────────────────────────────────────────
TRACK_WARN       = math.radians(2.0)     # rad at the joint: a joint this far off is counted
TRACK_ABORT      = math.radians(5.0)     # rad at the joint: a joint this far off stops playback
READ_FAILS_ABORT = 3                     # reads in a row a motor may miss before playback stops
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

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

def read_checked(getter, motor_id):
    """(value, error code) from one read, or (None, None) if the motor did not answer."""
    try:
        data_list, err = getter(motor_id)[0]
        return data_list[0], err
    except Exception:
        return None, None

def load_csv(scenario):
    """Return (times, q, dq, w): waypoint times (s), and per joint / wheel name
    the list of positions (rad), speeds (rad/s) and wheel speeds (rad/s) in
    motor units, one entry per waypoint."""
    path = os.path.join(_TIMINGS_DIR, f"{scenario}.csv")
    if not os.path.exists(path):
        print(f"CSV not found: {path}")
        sys.exit(1)
    times = []
    q  = {n: [] for n in JOINT_IDS}
    dq = {n: [] for n in JOINT_IDS}
    w  = {n: [] for n in WHEEL_IDS}
    with open(path, newline="") as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            times.append(float(row["time_s"]))
            for n in JOINT_IDS:
                q[n].append(float(row[f"{n}_rad"]))
                dq[n].append(float(row[f"d{n}_rad_s"]))
            for n in WHEEL_IDS:
                w[n].append(float(row[f"w_{n}_rad_s"]))
    return times, q, dq, w

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

def deg(n, motor_rad):
    """Joint n's motor position as a joint angle in degrees."""
    return math.degrees(JOINTS[n].angle(motor_rad))

# ══════════════════════════════════════════════════════════════════════════════
# Startup
# ══════════════════════════════════════════════════════════════════════════════
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) != 1 or args[0] not in SCENARIO_NAMES:
    print(f"Usage: python playback.py <scenario> [--start-pose]")
    print(f"  scenarios: {SCENARIO_NAMES}")
    sys.exit(1)

scenario   = args[0]
start_pose = "--start-pose" in sys.argv
times, q_frames, dq_frames, w_frames = load_csv(scenario)
n_frames   = len(times)

print(f"\nScenario : {scenario}")
print(f"Waypoints: {n_frames}  ({times[-1]:.1f}s, {(n_frames - 1) / times[-1]:g} per second)")
print(f"Command  : every {1000/INTERP_HZ:.0f}ms  (cubic Hermite joints, linear wheel speeds, swerves held)")
print(f"Joints   : " + ("relative to the start pose you place (--start-pose)" if start_pose
                        else "absolute, from motor_calibration.py"))

print(f"\nStart pose:")
for n, mid in JOINT_IDS.items():
    print(f"  {n} = {deg(n, q_frames[n][0]):+6.1f}°  (BEAR {mid})")
if start_pose:
    input("\nPlace the robot in this start pose by hand, then press Enter: ")

# Connect
print("\nConnecting...")
bear = Manager.BEAR(port=PORT, baudrate=BAUDRATE, single_timeout=0.05, single_try_num=3)

for mid in ALL_IDS:
    _, err = bear.ping(mid)[0]
    if err is None:
        print(f"  BEAR {mid} ({NAMES[mid]}): OFFLINE")
        bear.close_port()
        sys.exit(1)
    print(f"  BEAR {mid} ({NAMES[mid]}): online")

# Configure motors
print("\nConfiguring joints and swerves (position), wheels (velocity)...")
for mid in ALL_IDS:
    disable_motor(bear, mid)
    bear.set_p_gain_id((mid, P_GAIN_ID));  bear.set_i_gain_id((mid, I_GAIN_ID));  bear.set_d_gain_id((mid, D_GAIN_ID))
    bear.set_p_gain_iq((mid, P_GAIN_IQ));  bear.set_i_gain_iq((mid, I_GAIN_IQ));  bear.set_d_gain_iq((mid, D_GAIN_IQ))

for n, mid in JOINT_IDS.items():
    bear.set_p_gain_position((mid, P_GAIN_POS))
    bear.set_i_gain_position((mid, I_GAIN_POS))
    bear.set_d_gain_position((mid, D_GAIN_POS))
    bear.set_limit_i_max((mid, JOINT_IQ_MAX))
    bear.set_limit_velocity_max((mid, JOINT_VEL_MAX[n]))
    bear.set_limit_acc_max((mid, JOINT_ACC_MAX))
    bear.set_limit_position_min((mid, -POS_LIMIT_MOTOR))
    bear.set_limit_position_max((mid, POS_LIMIT_MOTOR))
    _lim, _ = read1(bear.get_limit_position_max, mid)
    if abs(_lim - POS_LIMIT_MOTOR) > 1e-3:
        print(f"  WARNING: BEAR {mid} ({n}) ignored the position limit (reads {_lim:.3f})")
    if not set_mode_confirmed(bear, mid, 2):           # position mode
        print(f"  WARNING: could not confirm position mode on BEAR {mid} ({n})")

for n, mid in SWERVE_IDS.items():
    bear.set_p_gain_position((mid, P_GAIN_POS))
    bear.set_i_gain_position((mid, I_GAIN_POS))
    bear.set_d_gain_position((mid, D_GAIN_POS))
    bear.set_limit_i_max((mid, SWERVE_IQ_MAX))
    bear.set_limit_velocity_max((mid, SWERVE_VEL_MAX))
    if not set_mode_confirmed(bear, mid, 2):           # position mode
        print(f"  WARNING: could not confirm position mode on BEAR {mid} ({n})")

for n, mid in WHEEL_IDS.items():
    bear.set_p_gain_velocity((mid, WHEEL_P_GAIN_VEL))
    bear.set_i_gain_velocity((mid, 0.0))
    bear.set_d_gain_velocity((mid, 0.0))
    bear.set_limit_i_max((mid, WHEEL_IQ_MAX))
    bear.set_limit_velocity_max((mid, WHEEL_VEL_MAX))
    bear.set_goal_velocity((mid, 0.0))
    if not set_mode_confirmed(bear, mid, 1):           # velocity mode
        print(f"  WARNING: could not confirm velocity mode on BEAR {mid} ({n})")

# Where the joints and swerves are now.  With --start-pose, each joint's
# reading becomes the file's first row: offset = reading - first row.
present = {mid: read1(bear.get_present_position, mid)[0]
           for mid in list(JOINT_IDS.values()) + list(SWERVE_IDS.values())}
offset  = {n: (present[mid] - q_frames[n][0]) if start_pose else 0.0 for n, mid in JOINT_IDS.items()}
hold    = {n: present[mid] for n, mid in SWERVE_IDS.items()}

# The motors refuse positions outside their position limits; check the whole
# file (the curves between waypoints never pass beyond the waypoints).
outside = False
for n, mid in JOINT_IDS.items():
    lo = read1(bear.get_limit_position_min, mid)[0]
    hi = read1(bear.get_limit_position_max, mid)[0]
    need_lo, need_hi = min(q_frames[n]) + offset[n], max(q_frames[n]) + offset[n]
    if need_lo < lo or need_hi > hi:
        outside = True
        print(f"  WARNING: {n} needs BEAR {mid} from {need_lo:+.2f} to {need_hi:+.2f} rad, "
              f"outside its position limits [{lo:+.2f}, {hi:+.2f}] rad")
for n, mid in SWERVE_IDS.items():
    lo = read1(bear.get_limit_position_min, mid)[0]
    hi = read1(bear.get_limit_position_max, mid)[0]
    if not lo <= hold[n] <= hi:
        outside = True
        print(f"  WARNING: {n} (BEAR {mid}) is at {hold[n]:+.2f} rad, outside its position limits "
              f"[{lo:+.2f}, {hi:+.2f}] rad")
if outside:
    input("Press Enter to continue anyway, Ctrl-C to abort: ")

# The wheel motors clamp their speed at WHEEL_VEL_MAX; the file must stay under it.
for n in WHEEL_IDS:
    peak_speed = max(abs(v) for v in w_frames[n])
    if peak_speed > WHEEL_VEL_MAX:
        print(f"  WARNING: {n} needs up to {peak_speed:.2f} rad/s, above WHEEL_VEL_MAX = {WHEEL_VEL_MAX} rad/s "
              f"— the wheel would fall behind the plan")
        input("Press Enter to continue anyway, Ctrl-C to abort: ")

# Check the joints are near the start pose before enabling: an enabled motor
# drives straight to its goal.  (With --start-pose they are there by definition.)
if not start_pose:
    errors = {n: abs(JOINTS[n].angle(present[mid]) - JOINTS[n].angle(q_frames[n][0]))
              for n, mid in JOINT_IDS.items()}
    print(f"\nDistance from start pose (joint):")
    for n, e in errors.items():
        print(f"  {n}: {math.degrees(e):5.1f}°{'  ← WARNING' if e > START_TOL else ''}")
    if max(errors.values()) > START_TOL:
        print(f"\nWARNING: joints are more than {math.degrees(START_TOL):.0f}° from the start pose;")
        print("enabling will drive them straight there.  Run homing.py first, or press Enter to continue anyway.")
        input("Press Enter to continue, Ctrl-C to abort: ")
    else:
        print("  All joints within tolerance.")

print("\nEnabling motors...")
for n, mid in JOINT_IDS.items():
    bear.set_goal_position((mid, q_frames[n][0] + offset[n]))    # prime goal before enabling
for n, mid in SWERVE_IDS.items():
    bear.set_goal_position((mid, hold[n]))
for mid in ALL_IDS:
    enabled, en, en_err = enable_motor(bear, mid)
    if not enabled:
        print(f"  WARNING: BEAR {mid} ({NAMES[mid]}) did not enable (torque_enable={en}, err={en_err})")

# Countdown
print("\nStarting playback in:")
for i in (3, 2, 1):
    print(f"  {i}...")
    time.sleep(1.0)
print("  GO\n")

# ══════════════════════════════════════════════════════════════════════════════
# Playback loop  (cubic Hermite for the joints, linear for the wheels)
# ══════════════════════════════════════════════════════════════════════════════
n_ticks = int(round(times[-1] * INTERP_HZ))
k       = 0       # current segment: times[k] <= t <= times[k + 1]
late    = 0       # updates sent more than one update interval behind schedule
abort   = None    # why playback stopped early, or None
goal    = {n: q_frames[n][0] + offset[n] for n in JOINT_IDS}   # latest joint goals (motor rad)
command = {n: 0.0 for n in WHEEL_IDS}                          # latest wheel speeds (rad/s)
fails   = {mid: 0 for mid in ALL_IDS}                          # reads in a row each motor missed
track   = {n: 0.0 for n in JOINT_IDS}                          # latest tracking error (rad at the joint)
track_max = {n: 0.0 for n in JOINT_IDS}
warned  = 0
drift_max = {n: 0.0 for n in SWERVE_IDS}
peak    = {mid: {"iq": 0.0, "temp": float("-inf"), "volts": float("inf")} for mid in ALL_IDS}
log     = []      # one row per reading, columns LOG_COLUMNS
LOG_COLUMNS = (["t_s", "motor", "bear_id", "command", "joint_error_deg", "swerve_drift_rad"]
               + STATS + ["error_code"])

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
            worst = max(track, key=track.get)
            print(f"\r  waypoint {k:3d}/{n_frames - 1}  t={time.time() - t_start:.2f}s  "
                  f"worst joint {worst} off by {math.degrees(track[worst]):4.1f}°", end="", flush=True)
        T     = times[k + 1] - times[k]
        alpha = min(max((t - times[k]) / T, 0.0), 1.0)

        for n, mid in JOINT_IDS.items():
            goal[n] = hermite(q_frames[n][k], dq_frames[n][k], q_frames[n][k + 1], dq_frames[n][k + 1],
                              T, alpha) + offset[n]
            bear.set_goal_position((mid, goal[n]))
        for n, mid in WHEEL_IDS.items():
            command[n] = w_frames[n][k] + alpha * (w_frames[n][k + 1] - w_frames[n][k])
            bear.set_goal_velocity((mid, command[n]))

        # Feedback: one motor's status registers per update, taking turns.
        mid = ALL_IDS[tick % len(ALL_IDS)]
        n   = NAMES[mid]
        st  = read_stats(bear, mid)
        if st is None:
            fails[mid] += 1
            if fails[mid] >= READ_FAILS_ABORT:
                abort = f"BEAR {mid} ({n}) missed {fails[mid]} reads in a row"
        else:
            fails[mid] = 0
            joint_err = swerve_drift = ""
            if n in JOINT_IDS:
                cmd = goal[n]
                track[n] = abs(JOINTS[n].angle(st["present_position"]) - JOINTS[n].angle(goal[n]))
                track_max[n] = max(track_max[n], track[n])
                warned += track[n] > TRACK_WARN
                joint_err = math.degrees(track[n])
                if track[n] > TRACK_ABORT:
                    abort = (f"{n} is {math.degrees(track[n]):.1f}° from where it should be "
                             f"(limit {math.degrees(TRACK_ABORT):.0f}°)")
            elif n in SWERVE_IDS:
                cmd = hold[n]
                swerve_drift = abs(st["present_position"] - hold[n])
                drift_max[n] = max(drift_max[n], swerve_drift)
                if swerve_drift > SWERVE_DRIFT_ABORT:
                    abort = (f"{n} has turned {swerve_drift:.2f} rad from where it was held "
                             f"(limit {SWERVE_DRIFT_ABORT} rad)")
            else:
                cmd = command[n]
            if st["error"] & ERROR_BITS:
                abort = (f"BEAR {mid} ({n}) reports an error: {', '.join(decode_code(st['error'] & ERROR_BITS))} "
                         f"(code {st['error']})")
            if st["torque_enable"] != 1:
                abort = f"BEAR {mid} ({n}) is no longer enabled (torque_enable = {st['torque_enable']})"
            pk = peak[mid]
            pk["iq"] = max(pk["iq"], abs(st["present_iq"]))
            pk["temp"] = max(pk["temp"], st["winding_temperature"], st["powerstage_temperature"],
                             st["ic_temperature"])
            pk["volts"] = min(pk["volts"], st["input_voltage"])
            log.append([t, n, mid, cmd, joint_err, swerve_drift] + [st[s] for s in STATS] + [st["error"]])
        if abort:
            break

    total = time.time() - t_start
    if abort:
        # Stop where we are: wheels to 0, joints and swerves held at their measured positions (no jump).
        for mid in WHEEL_IDS.values():
            bear.set_goal_velocity((mid, 0.0))
        for mid in list(JOINT_IDS.values()) + list(SWERVE_IDS.values()):
            held, _ = read_checked(bear.get_present_position, mid)
            if held is not None:
                bear.set_goal_position((mid, held))
        print(f"\n\nABORTED at t = {t:.2f}s (waypoint {k}/{n_frames - 1}): {abort}")
    else:
        # Hold the final pose, stop the wheels (swerves keep holding)
        for n, mid in JOINT_IDS.items():
            bear.set_goal_position((mid, q_frames[n][-1] + offset[n]))
        for mid in WHEEL_IDS.values():
            bear.set_goal_velocity((mid, 0.0))
        print(f"\n\nPlayback complete — {total:.2f}s of {times[-1]:.2f}s planned  ({n_frames} waypoints)")
    print(f"Late updates: {late} of {tick + 1}"
          + ("  ← the PC could not keep up; lower INTERP_HZ" if late > 0.01 * (tick + 1) else ""))
    print("Joint tracking error, max: " + "  ".join(f"{n} {math.degrees(v):.2f}°" for n, v in track_max.items())
          + f"   (readings above {math.degrees(TRACK_WARN):.0f}°: {warned})")
    print("Swerve drift, max: " + "  ".join(f"{n} {v:.3f} rad" for n, v in drift_max.items()))
    caps = {**{mid: JOINT_IQ_MAX for mid in JOINT_IDS.values()},
            **{mid: WHEEL_IQ_MAX for mid in WHEEL_IDS.values()},
            **{mid: SWERVE_IQ_MAX for mid in SWERVE_IDS.values()}}
    for mid in ALL_IDS:
        pk = peak[mid]
        if pk["volts"] < float("inf"):
            print(f"  BEAR {mid} ({NAMES[mid]:12s}): peak |iq| {pk['iq']:.2f} A (cap {caps[mid]} A"
                  + (", reached" if pk["iq"] >= 0.98 * caps[mid] else "")
                  + f"), hottest {pk['temp']:.0f} °C, lowest supply {pk['volts']:.1f} V")

except KeyboardInterrupt:
    print("\n\nInterrupted — holding current position.")

finally:
    for mid in WHEEL_IDS.values():
        try:
            bear.set_goal_velocity((mid, 0.0))   # never leave a wheel spinning
        except Exception as e:
            print(f"  BEAR {mid}: wheel stop failed ({e!r})")
    if log:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, f"playback_{scenario}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        with open(log_path, "w", newline="") as fh:
            fh.write(f"# playback.py log: {scenario}, started "
                     f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start))}"
                     + (" (--start-pose)" if start_pose else "") + "\n"
                     f"# one row per reading; motors read in turn, each about "
                     f"{INTERP_HZ / len(ALL_IDS):g} times a second"
                     + (f"; ABORTED: {abort}" if abort else "") + "\n"
                     f"# motors: " + ", ".join(f"BEAR {mid} = {NAMES[mid]}" for mid in ALL_IDS) + "\n"
                     f"# command: joints goal position (rad at the motor), wheels goal speed (rad/s),"
                     f" swerves held position (rad)\n"
                     + (f"# --start-pose offsets (rad at the motor): "
                        + ", ".join(f"{n} {offset[n]:+.4f}" for n in JOINT_IDS) + "\n" if start_pose else "")
                     + f"# units: positions rad and speeds rad/s at the BEAR output (joints = 30 x the joint"
                     f" angle); currents A; voltage V; temperatures C\n"
                     f"# joints: position mode, P/I/D {P_GAIN_POS}/{I_GAIN_POS}/{D_GAIN_POS}, iq cap"
                     f" {JOINT_IQ_MAX} A, vel cap {JOINT_VEL_MAX['q1']:.2f} rad/s, acc cap {JOINT_ACC_MAX} rad/s2\n"
                     f"# wheels: velocity mode, P {WHEEL_P_GAIN_VEL}, iq cap {WHEEL_IQ_MAX} A, vel cap"
                     f" {WHEEL_VEL_MAX} rad/s; swerves: position hold, iq cap {SWERVE_IQ_MAX} A;"
                     f" current loop P/I {P_GAIN_IQ}/{I_GAIN_IQ} (all)\n"
                     f"# limits: joint tracking warn {math.degrees(TRACK_WARN):.0f} deg, abort"
                     f" {math.degrees(TRACK_ABORT):.0f} deg; swerve drift abort {SWERVE_DRIFT_ABORT} rad\n")
            writer = csv.writer(fh)
            writer.writerow(LOG_COLUMNS)
            writer.writerows([v if isinstance(v, str) else f"{v:.6g}" for v in row] for row in log)
        print(f"Feedback log → {log_path}")
    input("\nPress Enter to disable motors.")
    for mid in ALL_IDS:
        try:
            bear.set_torque_enable((mid, 0))
        except Exception as e:
            print(f"  BEAR {mid}: disable failed ({e!r})")
    bear.close_port()
    print("All motors disabled, port closed.")
