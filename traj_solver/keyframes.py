"""
keyframes.py — show each transition at its phase boundaries.

Usage:
    python keyframes.py [scenario | all]     default: all

For each transition, draws six panels taken from the solved trajectory: the
start, then the end of each of the five phases (approach, pivot, both move,
pivot, depart — the last panel is the finish).  Phase durations are whole multiples of
timing.SCHEDULE_STEP, so at 3 or 6 fps every boundary is an actual frame — the
panels show exactly what the animation and robot_timing.py contain.

Writes keyframes/keyframes_<scenario>.png.
"""

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from animate_transition import draw_frame, plot_limits
from trajectory import solve_trajectory
from kinematics import q_display
from scenarios import SCENARIOS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, "keyframes")


def plot(name, out_dir=OUT_DIR):
    scn = SCENARIOS[name]
    traj = solve_trajectory(scn)
    xlim, ylim = plot_limits(traj)

    starts = traj.phase_starts
    labels = ["start"] + [f"after {p.name}" for p in traj.phases]
    frames = [int(np.argmin(np.abs(traj.times - t))) for t in starts]   # boundary frames

    ncols = 3
    nrows = math.ceil(len(frames) / ncols)
    cell_w = 4.0
    cell_h = cell_w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_w, nrows * (cell_h + 0.4)))
    axes = np.array(axes).flatten()

    print(f"\n{name}:")
    for ax, k, label in zip(axes, frames, labels):
        draw_frame(ax, scn, traj.Q[k], traj.poses[k], xlim, ylim, title="",
                   marker_scale=cell_w / 10.0, show_labels=False)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.tick_params(labelsize=5)
        ax.set_xlabel(f"{label}  (t = {traj.times[k]:.1f} s)", fontsize=8, labelpad=2)
        qd = np.degrees(q_display(traj.Q[k]))
        print(f"  {label:20s} t={traj.times[k]:5.1f}s  q_disp=({qd[0]:+6.1f} {qd[1]:+6.1f} "
              f"{qd[2]:+6.1f} {qd[3]:+6.1f})°")
    for ax in axes[len(frames):]:
        ax.set_visible(False)

    fig.suptitle(scn.label, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"keyframes_{name}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg != "all" and arg not in SCENARIOS:
        print(f"Unknown scenario '{arg}'. Choose: {list(SCENARIOS)} or 'all'")
        sys.exit(1)
    for name in (SCENARIOS if arg == "all" else [arg]):
        plot(name)
