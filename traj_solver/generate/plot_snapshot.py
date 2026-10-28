"""
plot_snapshot.py — snapshot of one transition, once the leading assembly has rolled clear.

Usage:
    python generate/plot_snapshot.py [scenario ...]     default: all (-h for help)

Solves the transition exactly as animate.py does and draws the
frame at the end of phase 3, when the leading assembly has rolled clear of
the transition.  At an inside corner the trailing one has not yet rolled up
to it; at an outside corner or edge, where both roll together, it has.
Joint angles are printed in the calf-up display convention (q1/q4: 0° =
link straight up from the calf).

Writes outputs/snapshots/<scenario>.png.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# traj_solver/ (this script's parent folder) on the path, so `core` imports from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine.cli import output_dir, parse_args
from core.engine.drawing import draw_frame, plot_limits
from core.engine.kinematics import q_display
from core.engine.trajectory import solve_trajectory
from core.robot import l1, l2, l3
from core.scenarios import SCENARIOS

OUT_DIR = output_dir("snapshots")


def plot(name, out_dir=OUT_DIR):
    """Solve transition `name` (or load it from the cache), print the pose and
    joint angles at the end of phase 3 and save that frame to
    out_dir/<name>.png; returns the path."""
    scn = SCENARIOS[name]
    traj = solve_trajectory(scn)

    # The frame at the end of phase 3 (leading assembly rolled clear).
    k = int(np.argmin(np.abs(traj.times - traj.phase_starts[3])))
    q, info, p = traj.Q[k], traj.infos[k], traj.poses[k]
    label = f"{scn.label} — t = {traj.times[k]:.1f} s of {traj.times[-1]:.1f} s (after {traj.phases[2].name})"

    print("=" * 60)
    print(f"  Snapshot: {label}")
    print(f"  Links          : l1={l1}, l2={l2}, l3={l3}  (total = {l1 + l2 + l3:.2f} m)")
    print(f"  Front assembly : ({p['x1']:.4f}, {p['y1']:.4f}),  θ₁ = {np.degrees(p['theta1']):.1f}°")
    print(f"  Back assembly  : ({p['x2']:.4f}, {p['y2']:.4f}),  θ₂ = {np.degrees(p['theta2']):.1f}°")
    print(f"  Residual norm  : {info['residual_norm']:.2e}   "
          f"hard constraints {'satisfied' if info['feasible'] else 'violated: ' + str(info['violations'])}")
    print("-" * 60)
    for j, v in enumerate(q_display(q), 1):
        print(f"  q{j} = {np.degrees(v):+9.4f}°  ({v:+.6f} rad)")
    print("=" * 60)

    xlim, ylim = plot_limits(traj)
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_frame(ax, scn, q, p, xlim, ylim, title=label)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved → {out}")
    return out


def main():
    names, _ = parse_args(__doc__)
    for name in names:
        plot(name)


if __name__ == "__main__":
    main()
