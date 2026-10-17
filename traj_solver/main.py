"""
main.py — snapshot of one transition, between its two rolling phases.

Usage:
    python main.py [scenario]
    Scenarios: floor_to_wall (default), wall_to_ceiling, outside, thin_edge

Solves the transition exactly as animate_transition.py does and draws the
frame at the end of phase 3, when the leading assembly has rolled clear of
the transition and the trailing one has not yet rolled up to it.  Joint angles are printed in the calf-up display convention (q1/q4: 0° =
link straight up from the calf).

Writes robot_configs/robot_config_<scenario>.png.
"""

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
from scenarios import SCENARIOS, l1, l2, l3


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "floor_to_wall"
    if name not in SCENARIOS:
        print(f"Unknown scenario '{name}'. Choose: {list(SCENARIOS)}")
        sys.exit(1)
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

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_configs",
                       f"robot_config_{name}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
