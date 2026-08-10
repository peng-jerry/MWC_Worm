"""
robot_timing.py — export per-frame joint angles and wheel speeds for physical robot control.

Usage:
    python robot_timing.py [scenario | all]     default: all

Outputs one CSV per scenario to traj_solver/timings/ with 13 columns:

  time_s | q1_rad  dq1_rad_s  w_fl_rad_s | q2_rad  dq2_rad_s  w_fr_rad_s |
          | q3_rad  dq3_rad_s  w_bl_rad_s | q4_rad  dq4_rad_s  w_br_rad_s

  time_s        seconds from start, step = 1 / PHYSICAL_FPS
  q1–q4         joint angles (rad), display/calf-up convention (0 = straight up from calf)
  dq1–dq4       joint angular velocity (rad/s) = Δq per frame × PHYSICAL_FPS
  w_fl / w_fr   front-left / front-right wheel ω (rad/s), CCW positive
  w_bl / w_br   back-left  / back-right  wheel ω (rad/s), CCW positive

Column grouping pairs each chain joint with the nearest wheel:
  q1 ↔ front-left   (chain joint at front assembly, arm-A wheel)
  q2 ↔ front-right  (second chain joint,            arm-B wheel)
  q3 ↔ back-left    (third chain joint,             arm-A wheel)
  q4 ↔ back-right   (chain joint at back assembly,  arm-B wheel)

Sign convention for wheel ω matches wheel_omega.py:
  floor        omega = −dx/R   (rolling left = +CCW)
  int. wall    omega = +dy/R   (rolling up   = +CCW)
  ceiling      omega = +dx/R   (rolling right = +CCW)
  ceil. top    omega = −dx/R   (rolling left  = +CCW)
  ext. wall    omega = −dy/R   (rolling down  = +CCW)
  edge top     omega = −dx/R
  edge bottom  omega = +dx/R
  pivot / air  omega = 0

Set PHYSICAL_FPS to match your robot's joint-speed limit:
    PHYSICAL_FPS = max_joint_deg_per_s / max_jump_threshold_deg_per_frame
    Example: 21.6 deg/s ÷ 7.2 deg/fr = 3 fr/s
"""

import sys
import os
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constraints import wheel_centers
from animate_transition import (
    solve_trajectory, SCENARIO_CONFIG,
    wg, l1, l2, l3,
    WALL_X, CEILING_Y, EDGE_X, EDGE_Y,
    N_FRAMES,
)
from kinematics import q_display as _q_display

# ── Physical robot control rate ───────────────────────────────────────────────
PHYSICAL_FPS = 3     # frames per second; 21.6 deg/s ÷ 7.2 deg/fr = 3 fr/s

R   = wg.wheel_r     # 0.050 m
TOL = 0.004

# ── Output directory ──────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, "timings")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Scenario selection ────────────────────────────────────────────────────────
_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
if _arg == "all":
    SCENARIOS = list(SCENARIO_CONFIG)
elif _arg in SCENARIO_CONFIG:
    SCENARIOS = [_arg]
else:
    print(f"Unknown scenario '{_arg}'. Choose: {list(SCENARIO_CONFIG)} or 'all'")
    sys.exit(1)

# ── Per-scenario surface-contact definitions ──────────────────────────────────
# Mirrors wheel_omega.py exactly.  Returns four closures:
#   on_A(w)          True if wheel w is on surface A
#   on_B(w)          True if wheel w is on surface B
#   formula_A(w, wp) signed ω contribution when on A (rad/frame)
#   formula_B(w, wp) signed ω contribution when on B (rad/frame)

def _contact_fns(scenario):
    if scenario == "floor_to_wall":
        def on_A(w):          return w[1] <= R + TOL
        def on_B(w):          return w[0] <= WALL_X + R + TOL
        def formula_A(w, wp): return -(w[0] - wp[0]) / R   # floor: left = +CCW
        def formula_B(w, wp): return  (w[1] - wp[1]) / R   # wall:  up   = +CCW

    elif scenario == "wall_to_ceiling":
        _cy = SCENARIO_CONFIG["wall_to_ceiling"]["keyframes"][0]["ceiling_y"]
        def on_A(w):          return w[0] <= WALL_X + R + TOL
        def on_B(w):          return w[1] >= _cy - R - TOL
        def formula_A(w, wp): return  (w[1] - wp[1]) / R   # wall:    up    = +CCW
        def formula_B(w, wp): return  (w[0] - wp[0]) / R   # ceiling: right = +CCW

    elif scenario == "outside":
        def on_A(w):          return w[1] >= CEILING_Y + R - TOL and w[0] >= WALL_X - TOL
        def on_B(w):          return abs(w[0] - (WALL_X - R)) <= TOL
        def formula_A(w, wp): return -(w[0] - wp[0]) / R   # ceil-top: left  = +CCW
        def formula_B(w, wp): return -(w[1] - wp[1]) / R   # ext-wall: down  = +CCW

    elif scenario == "thin_edge":
        _xt = R + TOL
        def on_A(w):          return abs(w[1] - (EDGE_Y + R)) <= TOL and w[0] <= EDGE_X + _xt
        def on_B(w):          return abs(w[1] - (EDGE_Y - R)) <= TOL and w[0] <= EDGE_X + _xt
        def formula_A(w, wp): return -(w[0] - wp[0]) / R   # edge top:    right = -CW
        def formula_B(w, wp): return  (w[0] - wp[0]) / R   # edge bottom: left  = -CW

    return on_A, on_B, formula_A, formula_B


def _omega_series(wheels, on_A, on_B, formula_A, formula_B):
    """Per-frame wheel ω in rad/frame; frame 0 mirrors frame 1."""
    omega = np.zeros(N_FRAMES)
    for f in range(1, N_FRAMES):
        w, wp = wheels[f], wheels[f - 1]
        a, b  = on_A(w), on_B(w)
        if a and not b and on_A(wp):
            omega[f] = formula_A(w, wp)
        elif b and not a and on_B(wp):
            omega[f] = formula_B(w, wp)
    omega[0] = omega[1]
    return omega


# ── CSV header ────────────────────────────────────────────────────────────────
HEADER = [
    "time_s",
    "q1_rad", "dq1_rad_s", "w_fl_rad_s",
    "q2_rad", "dq2_rad_s", "w_fr_rad_s",
    "q3_rad", "dq3_rad_s", "w_bl_rad_s",
    "q4_rad", "dq4_rad_s", "w_br_rad_s",
]

# ── Main loop ─────────────────────────────────────────────────────────────────
for scenario in SCENARIOS:
    cfg = SCENARIO_CONFIG[scenario]

    print(f"{scenario}  ({N_FRAMES} fr → {N_FRAMES / PHYSICAL_FPS:.1f}s @ {PHYSICAL_FPS} fps)")
    print("-" * 50)
    results = solve_trajectory(
        cfg["keyframes"], cfg["wg"], l1, l2, l3,
        n_frames=N_FRAMES, n_grid=cfg["n_grid"],
        smooth_weight=cfg["smooth_weight"], cold_at=cfg["cold_at"],
    )
    print("-" * 50)

    # Solved poses
    x1  = np.array([r[1]["x1"]     for r in results])
    y1  = np.array([r[1]["y1"]     for r in results])
    x2  = np.array([r[1]["x2"]     for r in results])
    y2  = np.array([r[1]["y2"]     for r in results])
    th1 = np.array([r[2]["theta1"] for r in results])
    th2 = np.array([r[2]["theta2"] for r in results])

    # Joint angles: display/calf-up convention, unwrapped
    qs_raw  = np.array([r[0] for r in results])
    qs_disp = np.stack([_q_display(q) for q in qs_raw])   # (N_FRAMES, 4) rad
    qs      = np.unwrap(qs_disp, axis=0)                   # remove ±π wrap artefacts

    # Joint velocity: rad/frame → rad/s
    dqs      = np.diff(qs, axis=0, prepend=qs[[0]])        # dqs[0] = 0
    dqs_rads = dqs * PHYSICAL_FPS

    # Wheel centres
    _wg = cfg["wg"]
    wl1 = np.array([wheel_centers(x1[i], y1[i], th1[i], _wg, side= 1)[0] for i in range(N_FRAMES)])
    wr1 = np.array([wheel_centers(x1[i], y1[i], th1[i], _wg, side= 1)[1] for i in range(N_FRAMES)])
    wl2 = np.array([wheel_centers(x2[i], y2[i], th2[i], _wg, side=-1)[0] for i in range(N_FRAMES)])
    wr2 = np.array([wheel_centers(x2[i], y2[i], th2[i], _wg, side=-1)[1] for i in range(N_FRAMES)])

    # Wheel ω: rad/frame → rad/s
    on_A, on_B, fA, fB = _contact_fns(scenario)
    w_fl = _omega_series(wl1, on_A, on_B, fA, fB) * PHYSICAL_FPS
    w_fr = _omega_series(wr1, on_A, on_B, fA, fB) * PHYSICAL_FPS
    w_bl = _omega_series(wl2, on_A, on_B, fA, fB) * PHYSICAL_FPS
    w_br = _omega_series(wr2, on_A, on_B, fA, fB) * PHYSICAL_FPS

    # Time column
    times = np.arange(N_FRAMES) / PHYSICAL_FPS

    # Write CSV
    out_path = os.path.join(OUT_DIR, f"{scenario}.csv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for i in range(N_FRAMES):
            writer.writerow([
                f"{times[i]:.6f}",
                f"{qs[i,0]:.6f}", f"{dqs_rads[i,0]:.6f}", f"{w_fl[i]:.6f}",
                f"{qs[i,1]:.6f}", f"{dqs_rads[i,1]:.6f}", f"{w_fr[i]:.6f}",
                f"{qs[i,2]:.6f}", f"{dqs_rads[i,2]:.6f}", f"{w_bl[i]:.6f}",
                f"{qs[i,3]:.6f}", f"{dqs_rads[i,3]:.6f}", f"{w_br[i]:.6f}",
            ])

    print(f"Saved → {out_path}  ({N_FRAMES} rows × {len(HEADER)} cols)")
    print()
