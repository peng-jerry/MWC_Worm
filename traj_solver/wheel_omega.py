"""
Wheel angular-velocity + joint-angle graph for any of the 4 transition scenarios.

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
from constraints import WheelGeometry, wheel_centers
from animate_transition import solve_trajectory

# ── Shared robot geometry ────────────────────────────────────────────────────
l1, l2, l3 = 0.25, 0.50, 0.25
wg = WheelGeometry(bar_len=0.200, wheel_r=0.050, spread=np.pi / 4,
                   calf=0.042, thigh=0.08951)
R   = wg.wheel_r   # 0.050 m
TOL = 0.004

WALL_X    = 0.0
CEILING_Y = 0.55
EDGE_X    = 0.35
EDGE_Y    = 0.40

N_FRAMES = 90

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "floor_to_wall"

# ── Per-scenario configuration ───────────────────────────────────────────────

if SCENARIO == "floor_to_wall":
    _FTW_CY = 2.0
    KEYFRAMES = [
        {"t": 0.000, "x1": 0.45, "y1": 0.0, "theta1": 0.0,
                     "x2": 1.20, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.150, "x1": 0.000, "y1": 0.0, "theta1": 0.0,
                     "x2": 0.750, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.175, "x1": 0.000, "y1": 0.0, "theta1": 0.0,
                     "x2": 0.750, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.200, "x1": 0.000, "y1": 0.0, "theta1": -np.pi / 12,
                     "x2": 0.750, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.225, "x1": 0.000, "y1": 0.0, "theta1": -np.pi / 6,
                     "x2": 0.750, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.250, "x1": 0.000, "y1": 0.0, "theta1": -np.pi / 4,
                     "x2": 0.750, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.275, "x1": 0.000, "y1": 0.0, "theta1": -np.pi / 3,
                     "x2": 0.750, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.300, "x1": 0.000, "y1": 0.0, "theta1": -5 * np.pi / 12,
                     "x2": 0.750, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.325, "x1": 0.000, "y1": 0.0, "theta1": -np.pi / 2,
                     "x2": 0.750, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.350, "x1": 0.000, "y1": 0.0, "theta1": -np.pi / 2,
                     "x2": 0.750, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.550, "x1": 0.000, "y1": 0.60, "theta1": -np.pi / 2,
                     "x2": 0.750, "y2": 0.0,  "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.700, "x1": 0.000, "y1": 0.90, "theta1": -np.pi / 2,
                     "x2": 0.300, "y2": 0.0,  "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.800, "x1": 0.000, "y1": 0.90, "theta1": -np.pi / 2,
                     "x2": 0.000, "y2": 0.0,  "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.825, "x1": 0.000, "y1": 0.90, "theta1": -np.pi / 2,
                     "x2": 0.000, "y2": 0.0,  "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.850, "x1": 0.000, "y1": 0.90, "theta1": -np.pi / 2,
                     "x2": 0.000, "y2": 0.0,  "theta2": -np.pi / 12,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.875, "x1": 0.000, "y1": 0.90, "theta1": -np.pi / 2,
                     "x2": 0.000, "y2": 0.0,  "theta2": -np.pi / 6,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.900, "x1": 0.000, "y1": 0.90, "theta1": -np.pi / 2,
                     "x2": 0.000, "y2": 0.0,  "theta2": -np.pi / 4,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.925, "x1": 0.000, "y1": 0.90, "theta1": -np.pi / 2,
                     "x2": 0.000, "y2": 0.0,  "theta2": -np.pi / 3,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.950, "x1": 0.000, "y1": 0.90, "theta1": -np.pi / 2,
                     "x2": 0.000, "y2": 0.0,  "theta2": -5 * np.pi / 12,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.975, "x1": 0.000, "y1": 0.90, "theta1": -np.pi / 2,
                     "x2": 0.000, "y2": 0.24, "theta2": -np.pi / 2,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 1.000, "x1": 0.000, "y1": 0.90, "theta1": -np.pi / 2,
                     "x2": 0.000, "y2": 0.24, "theta2": -np.pi / 2,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    ]
    N_GRID        = 120
    SMOOTH_WEIGHT = 30.0
    COLD_AT       = [0.215, 0.225, 0.237, 0.249, 0.260, 0.325, 0.550, 0.800]
    OUT_FILE      = "wheel_omega_ftw.png"
    TITLE         = "floor_to_wall — all 90 frames"

    def on_A(w):   return w[1] <= R + TOL                # floor
    def on_B(w):   return w[0] <= WALL_X + R + TOL       # interior wall
    def formula_A(w, wp): return -(w[0] - wp[0]) / R     # floor: left = +CCW
    def formula_B(w, wp): return  (w[1] - wp[1]) / R     # wall:  up   = +CCW

    def t2f(t): return round(t * (N_FRAMES - 1)) + 1
    PHASES = [
        (t2f(0.150), "front stops"),
        (t2f(0.175), "front rotates"),
        (t2f(0.325), "front on wall"),
        (t2f(0.350), "front climbs"),
        (t2f(0.550), "back moves"),
        (t2f(0.700), "back alone"),
        (t2f(0.800), "back at corner"),
        (t2f(0.825), "back rotates"),
    ]

elif SCENARIO == "wall_to_ceiling":
    _WTC_CY = 1.2
    KEYFRAMES = [
        {"t": 0.000, "x1": 0.0, "y1":  0.75,   "theta1": -np.pi / 2,
                     "x2": 0.0, "y2":  0.00,   "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.150, "x1": 0.0, "y1": _WTC_CY, "theta1": -np.pi / 2,
                     "x2": 0.0, "y2":  0.450,  "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.175, "x1": 0.0, "y1": _WTC_CY, "theta1": -np.pi / 2,
                     "x2": 0.0, "y2":  0.450,  "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.200, "x1": 0.0, "y1": _WTC_CY, "theta1": -7 * np.pi / 12,
                     "x2": 0.0, "y2":  0.450,  "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.225, "x1": 0.0, "y1": _WTC_CY, "theta1": -2 * np.pi / 3,
                     "x2": 0.0, "y2":  0.450,  "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.250, "x1": 0.0, "y1": _WTC_CY, "theta1": -3 * np.pi / 4,
                     "x2": 0.0, "y2":  0.450,  "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.275, "x1": 0.0, "y1": _WTC_CY, "theta1": -5 * np.pi / 6,
                     "x2": 0.0, "y2":  0.450,  "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.300, "x1": 0.0, "y1": _WTC_CY, "theta1": -11 * np.pi / 12,
                     "x2": 0.0, "y2":  0.450,  "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.325, "x1": 0.0, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0, "y2":  0.450,  "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.350, "x1": 0.0, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0, "y2":  0.450,  "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.575, "x1": 0.675, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,   "y2":  0.450,  "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.650, "x1": 0.900, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,   "y2":  0.675,  "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.825, "x1": 0.900, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,   "y2": _WTC_CY, "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.850, "x1": 0.900, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,   "y2": _WTC_CY, "theta2": -7 * np.pi / 12,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.875, "x1": 0.900, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,   "y2": _WTC_CY, "theta2": -2 * np.pi / 3,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.900, "x1": 0.900, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,   "y2": _WTC_CY, "theta2": -3 * np.pi / 4,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.925, "x1": 0.900, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,   "y2": _WTC_CY, "theta2": -5 * np.pi / 6,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.950, "x1": 0.900, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,   "y2": _WTC_CY, "theta2": -11 * np.pi / 12,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.975, "x1": 0.900, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.239, "y2": _WTC_CY, "theta2": np.pi,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 1.000, "x1": 0.975, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.314, "y2": _WTC_CY, "theta2": np.pi,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    ]
    N_GRID        = 60
    SMOOTH_WEIGHT = 30.0
    COLD_AT       = [0.325, 0.575, 0.650, 0.850, 0.875]
    OUT_FILE      = "wheel_omega_wtc.png"
    TITLE         = "wall_to_ceiling — all 90 frames"

    def on_A(w):   return w[0] <= WALL_X + R + TOL        # interior wall
    def on_B(w):   return w[1] >= _WTC_CY - R - TOL       # interior ceiling (from below)
    def formula_A(w, wp): return  (w[1] - wp[1]) / R      # wall:    up    = +CCW
    def formula_B(w, wp): return  (w[0] - wp[0]) / R      # ceiling: right = +CCW

    def t2f(t): return round(t * (N_FRAMES - 1)) + 1
    PHASES = [
        (t2f(0.150), "front arrives"),
        (t2f(0.175), "front rotates"),
        (t2f(0.325), "front on ceiling"),
        (t2f(0.350), "front slides"),
        (t2f(0.575), "sprint 75%"),
        (t2f(0.650), "back climbs"),
        (t2f(0.825), "back arrives"),
        (t2f(0.850), "back rotates"),
        (t2f(0.975), "back on ceiling"),
    ]

elif SCENARIO == "outside":
    _R   = R;  _BAR = wg.bar_len;  _THG = wg.thigh;  _CLF = wg.calf

    def _cj(phi):
        bwc_x = WALL_X - _R * np.sin(phi)
        bwc_y = CEILING_Y + _R * np.cos(phi)
        vx    = bwc_x - _BAR * np.cos(phi - np.pi / 4)
        vy    = bwc_y - _BAR * np.sin(phi - np.pi / 4)
        return (vx + _THG * np.cos(phi) - _CLF * np.sin(phi),
                vy + _THG * np.sin(phi) + _CLF * np.cos(phi))

    def _cj2(phi):
        rwc_x = WALL_X - _R * np.sin(phi)
        rwc_y = CEILING_Y + _R * np.cos(phi)
        vx    = rwc_x - _BAR * np.cos(phi - np.pi / 4)
        vy    = rwc_y - _BAR * np.sin(phi - np.pi / 4)
        return (vx - _THG * np.cos(phi) - _CLF * np.sin(phi),
                vy - _THG * np.sin(phi) + _CLF * np.cos(phi))

    _XSTP  = _cj(0)[0];           _Y0    = _cj(0)[1]
    _XSTP2 = _cj2(0)[0]
    _XW    = _cj(np.pi / 2)[0];   _YW    = _cj(np.pi / 2)[1]
    _YW2   = _cj2(np.pi / 2)[1]
    _CSE   = "outside_exact"

    def _y1d(t): return _YW - 1.75 * (t - 0.490)

    KEYFRAMES = [
        {"t": 0.000, "x1": 0.85,   "y1": _Y0, "theta1": 0.0,
                     "x2": 1.150,  "y2": _Y0, "theta2": 0.0,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.290, "x1": _XSTP,  "y1": _Y0, "theta1": 0.0,
                     "x2": 0.300,  "y2": _Y0, "theta2": 0.0,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.315, "x1": _XSTP,  "y1": _Y0, "theta1": 0.0,
                     "x2": 0.300,  "y2": _Y0, "theta2": 0.0,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.340, "x1": _cj(    np.pi/12)[0], "y1": _cj(    np.pi/12)[1],
                     "theta1":     np.pi/12,
                     "x2": 0.300, "y2": _Y0, "theta2": 0.0,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.365, "x1": _cj(    np.pi/ 6)[0], "y1": _cj(    np.pi/ 6)[1],
                     "theta1":     np.pi/ 6,
                     "x2": 0.300, "y2": _Y0, "theta2": 0.0,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.390, "x1": _cj(    np.pi/ 4)[0], "y1": _cj(    np.pi/ 4)[1],
                     "theta1":     np.pi/ 4,
                     "x2": 0.300, "y2": _Y0, "theta2": 0.0,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.415, "x1": _cj(    np.pi/ 3)[0], "y1": _cj(    np.pi/ 3)[1],
                     "theta1":     np.pi/ 3,
                     "x2": 0.300, "y2": _Y0, "theta2": 0.0,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.440, "x1": _cj(5 * np.pi/12)[0], "y1": _cj(5 * np.pi/12)[1],
                     "theta1": 5 * np.pi/12,
                     "x2": 0.300, "y2": _Y0, "theta2": 0.0,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.465, "x1": _XW, "y1": _YW,  "theta1": np.pi / 2,
                     "x2": 0.300, "y2": _Y0, "theta2": 0.0,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.490, "x1": _XW, "y1": _YW,  "theta1": np.pi / 2,
                     "x2": 0.300, "y2": _Y0, "theta2": 0.0,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.670, "x1": _XW, "y1": _y1d(0.670), "theta1": np.pi / 2,
                     "x2": _XSTP2, "y2": _Y0, "theta2": 0.0,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.695, "x1": _XW, "y1": _y1d(0.695), "theta1": np.pi / 2,
                     "x2": _XSTP2, "y2": _Y0, "theta2": 0.0,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.720, "x1": _XW, "y1": _y1d(0.720), "theta1": np.pi / 2,
                     "x2": _cj2(    np.pi/12)[0], "y2": _cj2(    np.pi/12)[1],
                     "theta2":     np.pi/12,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.745, "x1": _XW, "y1": _y1d(0.745), "theta1": np.pi / 2,
                     "x2": _cj2(    np.pi/ 6)[0], "y2": _cj2(    np.pi/ 6)[1],
                     "theta2":     np.pi/ 6,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.770, "x1": _XW, "y1": _y1d(0.770), "theta1": np.pi / 2,
                     "x2": _cj2(    np.pi/ 4)[0], "y2": _cj2(    np.pi/ 4)[1],
                     "theta2":     np.pi/ 4,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.795, "x1": _XW, "y1": _y1d(0.795), "theta1": np.pi / 2,
                     "x2": _cj2(    np.pi/ 3)[0], "y2": _cj2(    np.pi/ 3)[1],
                     "theta2":     np.pi/ 3,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.820, "x1": _XW, "y1": _y1d(0.820), "theta1": np.pi / 2,
                     "x2": _cj2(5 * np.pi/12)[0], "y2": _cj2(5 * np.pi/12)[1],
                     "theta2": 5 * np.pi/12,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.845, "x1": _XW, "y1": _y1d(0.845), "theta1": np.pi / 2,
                     "x2": _XW, "y2": _YW2, "theta2": np.pi / 2,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.870, "x1": _XW, "y1": _y1d(0.870), "theta1": np.pi / 2,
                     "x2": _XW, "y2": _YW2, "theta2": np.pi / 2,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 1.000, "x1": _XW, "y1": -0.350, "theta1": np.pi / 2,
                     "x2": _XW, "y2":  0.000, "theta2": np.pi / 2,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    ]
    N_GRID        = 60
    SMOOTH_WEIGHT = 200.0
    COLD_AT       = None
    OUT_FILE      = "wheel_omega_out.png"
    TITLE         = "outside corner — all 90 frames"

    # Ceiling surface ends at x=WALL_X=0 — wheel must still be over it.
    def on_A(w):   return w[1] >= CEILING_Y + R - TOL and w[0] >= WALL_X - TOL
    # Exterior wall contact requires wheel to be AT x≈WALL_X-R, not just anywhere left.
    def on_B(w):   return abs(w[0] - (WALL_X - R)) <= TOL
    def formula_A(w, wp): return -(w[0] - wp[0]) / R      # ceil-top: left  = +CCW
    def formula_B(w, wp): return -(w[1] - wp[1]) / R      # ext-wall: down  = +CCW

    def t2f(t): return round(t * (N_FRAMES - 1)) + 1
    PHASES = [
        (t2f(0.290), "front arrives"),
        (t2f(0.315), "front rotates"),
        (t2f(0.465), "front on wall"),
        (t2f(0.490), "back starts"),
        (t2f(0.670), "back arrives"),
        (t2f(0.695), "back rotates"),
        (t2f(0.845), "back on wall"),
        (t2f(0.870), "both descend"),
    ]

elif SCENARIO == "thin_edge":
    _R_TE   = R;  _BAR_TE = wg.bar_len

    def _cp_te(phi):
        dx0 = +_BAR_TE / np.sqrt(2.0)
        dy0 = _R_TE + _BAR_TE / np.sqrt(2.0)
        return (EDGE_X + dx0 * np.cos(phi) + dy0 * np.sin(phi),
                EDGE_Y - dx0 * np.sin(phi) + dy0 * np.cos(phi))

    def _cj_te(phi):
        vx, vy = _cp_te(phi)
        return (vx + wg.thigh * np.cos(phi) + wg.calf * np.sin(phi),
                vy - wg.thigh * np.sin(phi) + wg.calf * np.cos(phi))

    def _cj_te2(phi):
        vx, vy = _cp_te(phi)
        return (vx - wg.thigh * np.cos(phi) + wg.calf * np.sin(phi),
                vy + wg.thigh * np.sin(phi) + wg.calf * np.cos(phi))

    _XSTP_TE  = EDGE_X + _BAR_TE / np.sqrt(2.0) + wg.thigh
    _XSTP2_TE = EDGE_X - _BAR_TE / np.sqrt(2.0) + wg.thigh
    _Y0_TE    = EDGE_Y + _R_TE + _BAR_TE / np.sqrt(2.0) + wg.calf
    _XW_TE    = _cj_te(np.pi)[0];    _YW_TE  = _cj_te(np.pi)[1]
    _XSTP_red = EDGE_X - wg.thigh + _BAR_TE / np.sqrt(2.0)
    _XW_red   = _cj_te2(np.pi)[0];   _YW_red = _cj_te2(np.pi)[1]
    _CTE      = "thin_edge_exact"

    KEYFRAMES = [
        {"t": 0.00, "x1": -0.400,    "y1": _Y0_TE, "theta1": 0.0,
                    "x2": -0.088,    "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.20, "x1":  0.100,    "y1": _Y0_TE, "theta1": 0.0,
                    "x2": _XSTP_red, "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.24, "x1":  0.160,    "y1": _Y0_TE, "theta1": 0.0,
                    "x2": _cj_te2(  np.pi/6)[0], "y2": _cj_te2(  np.pi/6)[1],
                    "theta2": -np.pi / 6,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.27, "x1":  0.205,    "y1": _Y0_TE, "theta1": 0.0,
                    "x2": _cj_te2(  np.pi/3)[0], "y2": _cj_te2(  np.pi/3)[1],
                    "theta2": -np.pi / 3,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.30, "x1":  0.250,    "y1": _Y0_TE, "theta1": 0.0,
                    "x2": _cj_te2(  np.pi/2)[0], "y2": _cj_te2(  np.pi/2)[1],
                    "theta2": -np.pi / 2,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.33, "x1":  0.295,    "y1": _Y0_TE, "theta1": 0.0,
                    "x2": _cj_te2(2*np.pi/3)[0], "y2": _cj_te2(2*np.pi/3)[1],
                    "theta2": -2 * np.pi / 3,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.36, "x1": _XSTP2_TE, "y1": _Y0_TE, "theta1": 0.0,
                    "x2": _cj_te2(5*np.pi/6)[0], "y2": _cj_te2(5*np.pi/6)[1],
                    "theta2": -5 * np.pi / 6,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.39, "x1": _XSTP2_TE, "y1": _Y0_TE, "theta1": 0.0,
                    "x2": _XW_red,    "y2": _YW_red, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.42, "x1": _XSTP2_TE, "y1": _Y0_TE, "theta1": 0.0,
                    "x2": _XW_red,    "y2": _YW_red, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.50, "x1": _XSTP_TE,  "y1": _Y0_TE, "theta1": 0.0,
                    "x2": _XW_red,    "y2": _YW_red, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.62, "x1": _XSTP_TE,  "y1": _Y0_TE, "theta1": 0.0,
                    "x2":  0.30,      "y2": _YW_red, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.66, "x1": _cj_te(  np.pi/6)[0], "y1": _cj_te(  np.pi/6)[1],
                    "theta1": -np.pi / 6,
                    "x2":  0.300, "y2": _YW_red, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.70, "x1": _cj_te(  np.pi/3)[0], "y1": _cj_te(  np.pi/3)[1],
                    "theta1": -np.pi / 3,
                    "x2":  0.215, "y2": _YW_red, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.74, "x1": _cj_te(  np.pi/2)[0], "y1": _cj_te(  np.pi/2)[1],
                    "theta1": -np.pi / 2,
                    "x2":  0.130, "y2": _YW_red, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.78, "x1": _cj_te(2*np.pi/3)[0], "y1": _cj_te(2*np.pi/3)[1],
                    "theta1": -2 * np.pi / 3,
                    "x2":  0.045, "y2": _YW_red, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.82, "x1": _cj_te(5*np.pi/6)[0], "y1": _cj_te(5*np.pi/6)[1],
                    "theta1": -5 * np.pi / 6,
                    "x2": -0.040, "y2": _YW_red, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.86, "x1": _XW_TE,  "y1": _YW_TE, "theta1": -np.pi,
                    "x2": -0.125, "y2": _YW_red,  "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 1.00, "x1": -0.080,  "y1": _YW_TE, "theta1": -np.pi,
                    "x2": -0.350, "y2": _YW_red,  "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    ]
    N_GRID        = 60
    SMOOTH_WEIGHT = 30.0
    COLD_AT       = [0.42, 0.65]
    OUT_FILE      = "wheel_omega_te.png"
    TITLE         = "thin_edge — all 90 frames"

    _X_TOL = R + TOL   # x guard: wheel must be on the flat portion of the edge
    # Two-sided y check: wheel must be AT the surface height, not just anywhere
    # above/below it. A wheel sweeping through mid-pivot space (e.g. y=0.20 at
    # phi=π/2) must not be falsely detected as edge-bottom contact.
    def on_A(w):   return abs(w[1] - (EDGE_Y + R)) <= TOL and w[0] <= EDGE_X + _X_TOL
    def on_B(w):   return abs(w[1] - (EDGE_Y - R)) <= TOL and w[0] <= EDGE_X + _X_TOL
    def formula_A(w, wp): return -(w[0] - wp[0]) / R   # top:    right = -CW
    def formula_B(w, wp): return  (w[0] - wp[0]) / R   # bottom: left  = -CW

    def t2f(t): return round(t * (N_FRAMES - 1)) + 1
    PHASES = [
        (t2f(0.20), "back stops"),
        (t2f(0.24), "back pivots"),
        (t2f(0.39), "back below"),
        (t2f(0.42), "front rolls"),
        (t2f(0.50), "front stops"),
        (t2f(0.62), "front pauses"),
        (t2f(0.66), "front pivots"),
        (t2f(0.86), "front below"),
    ]

else:
    print(f"Unknown scenario '{SCENARIO}'. "
          "Choose: floor_to_wall, wall_to_ceiling, outside, thin_edge")
    sys.exit(1)

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
qs  = np.degrees(np.array([r[0] for r in results]))   # (N_FRAMES, 4)

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
        # Require previous frame also in contact: a landing frame (first touch)
        # has no prior rolling displacement to compute, so ω=0.
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
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1]})
fig.suptitle(TITLE, fontsize=13, y=0.98)

ax.plot(frames, omega_fl, label="Front-left",  color="#1f77b4", lw=1.8)
ax.plot(frames, omega_fr, label="Front-right", color="#1f77b4", lw=1.8,
        linestyle="--", dashes=(6, 3))
ax.plot(frames, omega_bl, label="Back-left",   color="#d62728", lw=1.8)
ax.plot(frames, omega_br, label="Back-right",  color="#d62728", lw=1.8,
        linestyle="--", dashes=(6, 3))
ax.axhline(0, color="black", lw=0.6, linestyle=":")
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
ax2.axhline( 135, color="red",  lw=0.8, linestyle="--", alpha=0.5, label="+135° limit")
ax2.axhline(-135, color="red",  lw=0.8, linestyle="--", alpha=0.5)
ax2.set_ylabel("Joint angle  (°)", fontsize=10)
ax2.set_xlabel("Frame", fontsize=10)
ax2.legend(loc="upper right", fontsize=9, ncol=3)
ax2.grid(axis="y", alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.97])

for f, label in PHASES:
    if 1 <= f <= N_FRAMES:
        for a in (ax, ax2):
            a.axvline(f, color="gray", lw=0.8, linestyle=":")
        yhi = ax.get_ylim()[1]
        ax.text(f + 0.4, yhi * 0.97, label,
                fontsize=7, color="gray", va="top", rotation=90)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_FILE)
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
