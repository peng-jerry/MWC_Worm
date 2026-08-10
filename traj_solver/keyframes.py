"""
keyframes.py — render every keyframe from each transition scenario.

Usage:
    python keyframes.py [scenario | all]     default: all

Produces one PNG per scenario in traj_solver/keyframes/:
    keyframes_floor_to_wall.png
    keyframes_wall_to_ceiling.png
    keyframes_outside.png
    keyframes_thin_edge.png

Applies the same surface-contact pre-corrections as solve_trajectory so
each panel shows the pose animate_transition actually solves (not the raw
hint coordinates).

Label "fr N / 90" is placed below each subplot so it never overlaps the robot.
Duplicate frame numbers (e.g. two KFs at fr 89) get an a/b suffix.
"""

import sys
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from animate_transition import (
    SCENARIO_CONFIG, l1, l2, l3,
    WALL_X, CEILING_Y,
    N_FRAMES, _draw_surfaces, _wheel_surface_adjust,
)
from solver import solve_ik
from visualize import plot_robot

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, "keyframes")
os.makedirs(OUT_DIR, exist_ok=True)

_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
if _arg == "all":
    scenarios = list(SCENARIO_CONFIG)
elif _arg in SCENARIO_CONFIG:
    scenarios = [_arg]
else:
    print(f"Unknown scenario '{_arg}'. Choose: {list(SCENARIO_CONFIG)} or 'all'")
    sys.exit(1)


def _make_labels(keyframes):
    """Return label strings; duplicate frame numbers get an a/b suffix."""
    raw = [round(kf["t"] * (N_FRAMES - 1)) for kf in keyframes]
    count = {fr: raw.count(fr) for fr in set(raw)}
    seen = {}
    labels = []
    for fr in raw:
        if count[fr] > 1:
            idx = seen.get(fr, 0)
            seen[fr] = idx + 1
            labels.append(f"fr {fr}{chr(ord('a') + idx)} / {N_FRAMES}")
        else:
            labels.append(f"fr {fr} / {N_FRAMES}")
    return labels


def _adjusted(kf, wg):
    """
    Return (x1, y1, x2, y2) after applying the same surface-contact
    pre-corrections that solve_trajectory applies before calling solve_ik.
    """
    x1, y1 = kf["x1"], kf["y1"]
    x2, y2 = kf["x2"], kf["y2"]
    th1, th2 = kf["theta1"], kf["theta2"]
    cs = kf["constraint_set"]
    wx = kf.get("wall_x",    WALL_X)
    cy = kf.get("ceiling_y", CEILING_Y)

    if cs == "wall":
        x1, y1 = _wheel_surface_adjust(x1, y1, th1, wg, side= 1, wall_x=wx, floor_y=0.0)
        x2, y2 = _wheel_surface_adjust(x2, y2, th2, wg, side=-1, wall_x=wx, floor_y=0.0)

    elif cs == "ceiling":
        x1, y1 = _wheel_surface_adjust(x1, y1, th1, wg, side= 1, wall_x=wx, floor_y=-0.4, ceiling_y=cy)
        x2, y2 = _wheel_surface_adjust(x2, y2, th2, wg, side=-1, wall_x=wx, floor_y=-0.4, ceiling_y=cy)

    elif cs == "ceiling_exact":
        if abs(th1 - np.pi) > 0.01:
            x2, y2 = _wheel_surface_adjust(x2, y2, th2, wg, side=-1, wall_x=wx, floor_y=-0.4, ceiling_y=cy)
        else:
            x1, y1 = _wheel_surface_adjust(x1, y1, th1, wg, side= 1, wall_x=wx, floor_y=-0.4, ceiling_y=cy)

    # wall_exact, outside, outside_exact, thin_edge_exact: no adjustment

    return x1, y1, x2, y2


for scenario in scenarios:
    cfg = SCENARIO_CONFIG[scenario]
    keyframes = cfg["keyframes"]
    xlim = cfg["xlim"]
    ylim = cfg["ylim"]
    n = len(keyframes)

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    cell_w = 3.0
    cell_h = cell_w * yspan / xspan

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * cell_w, nrows * (cell_h + 0.3)))
    axes_flat = np.array(axes).flatten()

    labels = _make_labels(keyframes)
    print(f"\n{scenario}: {n} keyframes  ({nrows}×{ncols} grid)")

    for i, (kf, label) in enumerate(zip(keyframes, labels)):
        ax = axes_flat[i]

        sc_wg     = cfg["wg"]
        cs        = kf["constraint_set"]
        wall_x    = kf.get("wall_x",    WALL_X)
        ceiling_y = kf.get("ceiling_y", CEILING_Y)
        th1, th2  = kf["theta1"], kf["theta2"]

        x1a, y1a, x2a, y2a = _adjusted(kf, sc_wg)

        q, success, info = solve_ik(
            x1a, y1a, th1,
            x2a, y2a, th2,
            l1, l2, l3,
            wg=sc_wg,
            constraint_set=cs,
            wall_x=wall_x,
            ceiling_y=ceiling_y,
            n_grid=cfg["n_grid"] * 6,
        )

        x1e, y1e = info["x1"], info["y1"]
        x2e, y2e = info["x2"], info["y2"]
        status = "✓" if success else "!"
        print(f"  {label:14s} {status}  "
              f"q=({np.degrees(q[0]):+6.1f} {np.degrees(q[1]):+6.1f} "
              f"{np.degrees(q[2]):+6.1f} {np.degrees(q[3]):+6.1f})°")

        _draw_surfaces(ax, cs, wall_x, ceiling_y, xlim, ylim,
                       marker_scale=cell_w / 10.0)
        plot_robot(
            q,
            x1e, y1e, th1,
            x2e, y2e, th2,
            l1, l2, l3,
            axle_half_length=sc_wg.bar_len,
            axle_a_len=sc_wg.bar_len_a,
            wheel_radius=sc_wg.wheel_r,
            calf=sc_wg.calf,
            arm_a1=sc_wg.arm_a1, arm_b1=sc_wg.arm_b1,
            arm_a2=sc_wg.arm_a2, arm_b2=sc_wg.arm_b2,
            ax=ax,
            title="",
            marker_scale=cell_w / 10.0,
            show_labels=False,
        )

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)
        # Label goes below each subplot — never overlaps the robot
        ax.set_xlabel(label, fontsize=7, labelpad=2)
        ax.set_ylabel("")

    for j in range(n, nrows * ncols):
        axes_flat[j].set_visible(False)

    fig.suptitle(scenario.replace("_", " "), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(OUT_DIR, f"keyframes_{scenario}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
