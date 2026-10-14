"""
wheel_omega.py — wheel speed, joint angle and joint speed plots for one transition.

Usage:
    python wheel_omega.py [floor_to_wall|wall_to_ceiling|outside|thin_edge]
    (default floor_to_wall)

Solves the transition exactly as animate_transition.py does and plots, against
time, with the phase boundaries marked:
  1. wheel spin of all four wheels (rad/s, CCW positive; scenarios.wheel_spin)
     with the rolling-speed limit SLIDE_SPEED / wheel radius
  2. joint angles q1–q4 (display convention) with the soft and hard joint
     limits from constraints.py
  3. joint speeds (°/s) with the joint speed limit

Writes wheel_omega/wheel_omega_<short name>.png.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from animate_transition import solve_trajectory
from constraints import JOINT_HARD_LIMIT_DEG, JOINT_SOFT_LIMIT_DEG
from kinematics import q_display
from scenarios import SCENARIOS, wheel_spin
from timing import MAX_JOINT_SPEED, SLIDE_SPEED

OUT_NAMES = {"floor_to_wall": "wheel_omega_ftw.png", "wall_to_ceiling": "wheel_omega_wtc.png",
             "outside": "wheel_omega_out.png", "thin_edge": "wheel_omega_te.png"}


def plot(name, out_dir):
    scn = SCENARIOS[name]
    traj = solve_trajectory(scn)
    t = traj.times
    spin = wheel_spin(scn, traj.poses, t)
    qs = np.degrees(np.unwrap(np.array([q_display(q) for q in traj.Q]), axis=0))
    dqs = np.diff(qs, axis=0) / np.diff(t)[:, None]          # °/s over each step
    t_mid = 0.5 * (t[1:] + t[:-1])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"{scn.label} — {t[-1]:.1f} s, {len(traj)} frames at {traj.rate:g} Hz",
                 fontsize=13, y=0.98)

    # 1 wheel spin
    styles = [("Front arm A", "#1f77b4", "-"), ("Front arm B", "#1f77b4", "--"),
              ("Back arm A", "#d62728", "-"), ("Back arm B", "#d62728", "--")]
    for j, (label, color, ls) in enumerate(styles):
        ax1.plot(t, spin[:, j], label=label, color=color, lw=1.8, linestyle=ls)
    omega_lim = SLIDE_SPEED / scn.wg.wheel_r
    for sgn in (1, -1):
        ax1.axhline(sgn * omega_lim, color="orange", lw=0.8, linestyle="--", alpha=0.6,
                    label=f"rolling limit ±{omega_lim:.1f}" if sgn > 0 else None)
    ax1.axhline(0, color="black", lw=0.6, linestyle=":")
    ax1.set_ylabel("Wheel spin (rad/s)")
    ax1.legend(loc="upper right", fontsize=9, ncol=3)

    # 2 joint angles
    colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for j in range(4):
        ax2.plot(t, qs[:, j], label=f"q{j + 1}", color=colors[j], lw=1.8)
    for sgn in (1, -1):
        ax2.axhline(sgn * JOINT_SOFT_LIMIT_DEG, color="red", lw=0.8, linestyle="--", alpha=0.5,
                    label=f"±{JOINT_SOFT_LIMIT_DEG:g}° soft limit" if sgn > 0 else None)
        ax2.axhline(sgn * JOINT_HARD_LIMIT_DEG, color="red", lw=1.0, alpha=0.7,
                    label=f"±{JOINT_HARD_LIMIT_DEG:g}° hard limit" if sgn > 0 else None)
    ax2.axhline(0, color="black", lw=0.5, linestyle=":")
    ax2.set_ylabel("Joint angle (°)")
    ax2.legend(loc="upper right", fontsize=9, ncol=5)

    # 3 joint speeds
    for j in range(4):
        ax3.plot(t_mid, dqs[:, j], label=f"q{j + 1}", color=colors[j], lw=1.8)
    lim = np.degrees(MAX_JOINT_SPEED)
    for sgn in (1, -1):
        ax3.axhline(sgn * lim, color="red", lw=0.8, linestyle="--", alpha=0.5,
                    label=f"±{lim:.1f}°/s limit" if sgn > 0 else None)
    ax3.axhline(0, color="black", lw=0.5, linestyle=":")
    ax3.set_ylabel("Joint speed (°/s)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper right", fontsize=9, ncol=5)

    for ax in (ax1, ax2, ax3):
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(0, t[-1])
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # phase boundaries, labelled at the top of the first panel
    for phase, start in zip(traj.phases, traj.phase_starts):
        for ax in (ax1, ax2, ax3):
            ax.axvline(start, color="gray", lw=0.8, linestyle=":")
        ax1.text(start + 0.15, ax1.get_ylim()[1] * 0.97, phase.name,
                 fontsize=8, color="gray", va="top", rotation=90)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, OUT_NAMES[name])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "floor_to_wall"
    if name not in SCENARIOS:
        print(f"Unknown scenario '{name}'. Choose: {list(SCENARIOS)}")
        sys.exit(1)
    plot(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wheel_omega"))
