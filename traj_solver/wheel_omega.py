"""
Wheel angular-velocity + joint-angle graph for any of the 4 transition scenarios.

Keyframes and solver config are imported from animate_transition.SCENARIO_CONFIG
so this file always uses the same data as the animation renderer.

Usage:
    python wheel_omega.py [floor_to_wall|wall_to_ceiling|outside|thin_edge]

Default: floor_to_wall

Sign convention (universal physical — counterclockwise = positive):
    floor        contact below wheel:  omega = -dx/R   (left = +CCW)
    wall (int.)  contact right side:   omega = +dy/R   (up   = +CCW)
    ceiling      contact above wheel:  omega = +dx/R   (right = +CCW)
    ceiling-top  contact below wheel:  omega = -dx/R   (left = +CCW)
    ext. wall    contact right side:   omega = -dy/R   (down = +CCW)
    edge-top     contact below wheel:  omega = -dx/R   (right = -CW)
    edge-bottom  contact above wheel:  omega = +dx/R   (left  = -CW)
    pivot / air: omega = 0
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constraints import wheel_centers
from animate_transition import (
    solve_trajectory, SCENARIO_CONFIG,
    wg, l1, l2, l3,
    WALL_X, CEILING_Y, EDGE_X, EDGE_Y,
    N_FRAMES,
    _WTC_T_APPROACH, _WTC_T_ROT_END, _WTC_T_SLIDE_0, _WTC_T_SLIDE_END, _WTC_ROT_DT,
    _OUT_T_BACK_ARR, _OUT_T_FRONT_STOP, _OUT_T_ROT1_S, _OUT_T_ROT1_E,
    _OUT_T_BOTH_S, _OUT_T_ROT2_E,
)

R   = wg.wheel_r   # 0.050 m
TOL = 0.004

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "floor_to_wall"

if SCENARIO not in SCENARIO_CONFIG:
    print(f"Unknown scenario '{SCENARIO}'. "
          "Choose: floor_to_wall, wall_to_ceiling, outside, thin_edge")
    sys.exit(1)

cfg           = SCENARIO_CONFIG[SCENARIO]
KEYFRAMES     = cfg["keyframes"]
N_GRID        = cfg["n_grid"]
SMOOTH_WEIGHT = cfg["smooth_weight"]
COLD_AT       = cfg["cold_at"]

# ── Per-scenario contact detection ────────────────────────────────────────────

def t2f(t): return round(t * (N_FRAMES - 1)) + 1

if SCENARIO == "floor_to_wall":
    OUT_FILE = "wheel_omega_ftw.png"
    TITLE    = "floor_to_wall — all 90 frames"

    def on_A(w):   return w[1] <= R + TOL                # floor
    def on_B(w):   return w[0] <= WALL_X + R + TOL       # interior wall
    def formula_A(w, wp): return -(w[0] - wp[0]) / R     # floor: left = +CCW
    def formula_B(w, wp): return  (w[1] - wp[1]) / R     # wall:  up   = +CCW

    PHASES = [
        (t2f(0.150), "front stops"),
        (t2f(0.183), "front rotates"),
        (t2f(0.360), "front on wall"),
        (t2f(0.480), "both moving"),
        (t2f(0.775), "back at corner"),
        (t2f(0.808), "back rotates"),
        (t2f(0.977), "back on wall"),
    ]

elif SCENARIO == "wall_to_ceiling":
    _WTC_CY = SCENARIO_CONFIG["wall_to_ceiling"]["keyframes"][0]["ceiling_y"]
    OUT_FILE = "wheel_omega_wtc.png"
    TITLE    = "wall_to_ceiling — all 90 frames"

    def on_A(w):   return w[0] <= WALL_X + R + TOL       # interior wall
    def on_B(w):   return w[1] >= _WTC_CY - R - TOL      # interior ceiling
    def formula_A(w, wp): return  (w[1] - wp[1]) / R     # wall:    up    = +CCW
    def formula_B(w, wp): return  (w[0] - wp[0]) / R     # ceiling: right = +CCW

    PHASES = [
        (t2f(_WTC_T_APPROACH),        "front at corner"),
        (t2f(_WTC_T_APPROACH + 0.025), "front rotates"),
        (t2f(_WTC_T_ROT_END),          "front on ceiling"),
        (t2f(_WTC_T_SLIDE_0),           "both moving"),
        (t2f(_WTC_T_SLIDE_END),        "back at corner"),
        (t2f(_WTC_T_SLIDE_END + 0.025), "back rotates"),
        (t2f(_WTC_T_SLIDE_END + 6*_WTC_ROT_DT), "back on ceiling"),
    ]

elif SCENARIO == "outside":
    OUT_FILE = "wheel_omega_out.png"
    TITLE    = "outside corner — all 90 frames"

    # Ceiling surface ends at x=WALL_X — wheel must still be over it.
    def on_A(w):   return w[1] >= CEILING_Y + R - TOL and w[0] >= WALL_X - TOL
    # Exterior wall contact: wheel centre at x ≈ WALL_X − R.
    def on_B(w):   return abs(w[0] - (WALL_X - R)) <= TOL
    def formula_A(w, wp): return -(w[0] - wp[0]) / R     # ceil-top: left  = +CCW
    def formula_B(w, wp): return -(w[1] - wp[1]) / R     # ext-wall: down  = +CCW

    PHASES = [
        (t2f(_OUT_T_FRONT_STOP),          "front arrives"),
        (t2f(_OUT_T_ROT1_S),              "front rotates"),
        (t2f(_OUT_T_ROT1_E),              "front on wall"),
        (t2f(_OUT_T_BOTH_S),              "back starts"),
        (t2f(_OUT_T_BACK_ARR),            "back arrives"),
        (t2f(_OUT_T_BACK_ARR + 0.025),    "back rotates"),
        (t2f(_OUT_T_ROT2_E),              "back on wall"),
    ]

elif SCENARIO == "thin_edge":
    OUT_FILE = "wheel_omega_te.png"
    TITLE    = "thin_edge — all 90 frames"

    _X_TOL = R + TOL
    def on_A(w):   return abs(w[1] - (EDGE_Y + R)) <= TOL and w[0] <= EDGE_X + _X_TOL
    def on_B(w):   return abs(w[1] - (EDGE_Y - R)) <= TOL and w[0] <= EDGE_X + _X_TOL
    def formula_A(w, wp): return -(w[0] - wp[0]) / R     # top:    right = -CW
    def formula_B(w, wp): return  (w[0] - wp[0]) / R     # bottom: left  = -CW

    PHASES = [
        (t2f(0.200), "red stops"),
        (t2f(0.232), "red pivots"),
        (t2f(0.521), "red done"),
        (t2f(0.616), "green stops"),
        (t2f(0.641), "green pivots"),
        (t2f(0.941), "green done"),
    ]

# ── Solve trajectory ─────────────────────────────────────────────────────────
print(f"{SCENARIO}  ({N_FRAMES} frames)")
print("-" * 50)
results = solve_trajectory(
    KEYFRAMES, wg, l1, l2, l3,
    n_frames=N_FRAMES, n_grid=N_GRID,
    smooth_weight=SMOOTH_WEIGHT, cold_at=COLD_AT,
)
print("-" * 50)

# ── Extract positions and joint angles ───────────────────────────────────────
x1  = np.array([r[1]["x1"]     for r in results])
y1  = np.array([r[1]["y1"]     for r in results])
x2  = np.array([r[1]["x2"]     for r in results])
y2  = np.array([r[1]["y2"]     for r in results])
th1 = np.array([r[2]["theta1"] for r in results])
th2 = np.array([r[2]["theta2"] for r in results])
from kinematics import q_display as _q_display
_qs_raw = np.array([r[0] for r in results])                          # (N_FRAMES, 4)
_qs_disp_rad = np.stack([_q_display(q) for q in _qs_raw])            # (N_FRAMES, 4) radians
qs = np.degrees(np.unwrap(_qs_disp_rad, axis=0))                     # unwrap to remove ±180° wrap artefacts

# ── Wheel centers ─────────────────────────────────────────────────────────────
wl1 = np.array([wheel_centers(x1[i], y1[i], th1[i], wg, side= 1)[0] for i in range(N_FRAMES)])
wr1 = np.array([wheel_centers(x1[i], y1[i], th1[i], wg, side= 1)[1] for i in range(N_FRAMES)])
wl2 = np.array([wheel_centers(x2[i], y2[i], th2[i], wg, side=-1)[0] for i in range(N_FRAMES)])
wr2 = np.array([wheel_centers(x2[i], y2[i], th2[i], wg, side=-1)[1] for i in range(N_FRAMES)])

# ── Angular velocity ──────────────────────────────────────────────────────────
def compute_omega(wheels):
    """on_A / on_B and formula_A / formula_B defined per scenario above."""
    omega = np.zeros(N_FRAMES)
    for f in range(1, N_FRAMES):
        w, wp = wheels[f], wheels[f - 1]
        a = on_A(w)
        b = on_B(w)
        # Require previous frame also in contact: first-touch frames have no
        # prior rolling displacement, so ω=0.
        if a and not b and on_A(wp):
            omega[f] = formula_A(w, wp)
        elif b and not a and on_B(wp):
            omega[f] = formula_B(w, wp)
        # both (corner/pivot), neither, or first-touch → omega stays 0
    omega[0] = omega[1]
    return omega

omega_fl = compute_omega(wl1)   # front-left  (assembly 1)
omega_fr = compute_omega(wr1)   # front-right (assembly 1)
omega_bl = compute_omega(wl2)   # back-left   (assembly 2)
omega_br = compute_omega(wr2)   # back-right  (assembly 2)

# ── Plot ──────────────────────────────────────────────────────────────────────
frames = np.arange(1, N_FRAMES + 1)
fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1, 1]})
fig.suptitle(TITLE, fontsize=13, y=0.98)

ax.plot(frames, omega_fl, label="Front-left",  color="#1f77b4", lw=1.8)
ax.plot(frames, omega_fr, label="Front-right", color="#1f77b4", lw=1.8,
        linestyle="--", dashes=(6, 3))
ax.plot(frames, omega_bl, label="Back-left",   color="#d62728", lw=1.8)
ax.plot(frames, omega_br, label="Back-right",  color="#d62728", lw=1.8,
        linestyle="--", dashes=(6, 3))
ax.axhline(0, color="black", lw=0.6, linestyle=":")
_OMEGA_LIMIT = 30 * np.radians(7.5)    # 30× 7.5°/fr joint limit  ≈ 3.927 rad/frame
ax.axhline( _OMEGA_LIMIT, color="orange", lw=0.8, linestyle="--", alpha=0.6,
            label=f"ω limit ±{_OMEGA_LIMIT:.3f}")
ax.axhline(-_OMEGA_LIMIT, color="orange", lw=0.8, linestyle="--", alpha=0.6)
ax.set_ylabel("Angular velocity  (rad / frame)", fontsize=10)
ax.set_xlim(1, N_FRAMES)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(axis="y", alpha=0.3)

q_colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
for i, (name, col) in enumerate(zip(["q1", "q2", "q3", "q4"], q_colors)):
    ax2.plot(frames, qs[:, i], label=name, color=col, lw=1.8)
ax2.axhline(  0, color="black", lw=0.5, linestyle=":")
ax2.axhline(  90, color="red",  lw=0.8, linestyle="--", alpha=0.5, label="+90° limit")
ax2.axhline( -90, color="red",  lw=0.8, linestyle="--", alpha=0.5)
ax2.set_ylabel("Joint angle  (°)", fontsize=10)
ax2.legend(loc="upper right", fontsize=9, ncol=3)
ax2.grid(axis="y", alpha=0.3)

_DJOINT_LIM = 7.5    # °/frame joint limit
dqs = np.diff(qs, axis=0, prepend=qs[[0]])   # dqs[0]=0; dqs[i]=qs[i]−qs[i−1]
for i, (name, col) in enumerate(zip(["q1", "q2", "q3", "q4"], q_colors)):
    ax3.plot(frames, dqs[:, i], label=name, color=col, lw=1.8)
ax3.axhline(0, color="black", lw=0.5, linestyle=":")
ax3.axhline( _DJOINT_LIM, color="red", lw=0.8, linestyle="--", alpha=0.5,
             label=f"±{_DJOINT_LIM:.2f}°/fr limit")
ax3.axhline(-_DJOINT_LIM, color="red", lw=0.8, linestyle="--", alpha=0.5)
ax3.set_ylabel("Δ Joint angle  (°/frame)", fontsize=10)
ax3.set_xlabel("Frame", fontsize=10)
ax3.legend(loc="upper left", fontsize=9, ncol=3)
ax3.grid(axis="y", alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.97])

for f, label in PHASES:
    if 1 <= f <= N_FRAMES:
        for a in (ax, ax2, ax3):
            a.axvline(f, color="gray", lw=0.8, linestyle=":")
        yhi = ax.get_ylim()[1]
        ax.text(f + 0.4, yhi * 0.97, label,
                fontsize=7, color="gray", va="top", rotation=90)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_FILE)
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
