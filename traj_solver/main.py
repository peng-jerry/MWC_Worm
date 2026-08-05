"""
Snapshot viewer: solve and plot a single characteristic frame from each
animate_transition.py scenario.

Usage:
    python main.py [scenario]

Scenarios: floor_to_wall (default), wall_to_ceiling, outside, thin_edge

Geometry and keyframe poses are taken directly from animate_transition.py
so the output always matches the animation. Joint angles q1/q4 are shown
in the calf-up display convention (0° = link straight up from the calf).
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from animate_transition import (
    wg, l1, l2, l3,
    WALL_X, CEILING_Y, EDGE_X, EDGE_Y,
    SCENARIO_CONFIG,
)
from kinematics import q_display as _q_display
from solver import solve_ik
from visualize import plot_robot


def _draw_surfaces(ax, constraint_set, wall_x, ceiling_y, edge_x, edge_y, xlim, ylim):
    """Draw environment surfaces matching the scenario."""
    if constraint_set in ("floor", "wall", "wall_exact"):
        ax.plot([wall_x, xlim[1]], [0, 0],
                color="saddlebrown", lw=2, label="Floor")
        ax.plot([wall_x, wall_x], [0, ylim[1]],
                color="slategray", lw=2, label=f"Wall (x={wall_x})")

    elif constraint_set in ("ceiling", "ceiling_exact"):
        ax.plot([wall_x, wall_x], [ylim[0], ceiling_y],
                color="slategray", lw=2, label=f"Wall (x={wall_x})")
        ax.plot([wall_x, xlim[1]], [ceiling_y, ceiling_y],
                color="dimgray", lw=2, label=f"Ceiling (y={ceiling_y})")

    elif constraint_set in ("outside", "outside_exact"):
        block = Rectangle(
            (wall_x, ylim[0]), xlim[1] - wall_x, ceiling_y - ylim[0],
            facecolor="lightgray", alpha=0.45, zorder=0, label="Interior (solid)")
        ax.add_patch(block)
        ax.plot([wall_x, xlim[1]], [ceiling_y, ceiling_y],
                color="dimgray", lw=2, label=f"Ceiling top (y={ceiling_y})")
        ax.plot([wall_x, wall_x], [ylim[0], ceiling_y],
                color="slategray", lw=2, label=f"Wall exterior (x={wall_x})")

    elif constraint_set in ("thin_edge", "thin_edge_exact"):
        ax.plot([xlim[0], edge_x], [edge_y, edge_y],
                color="saddlebrown", lw=3, label=f"Thin edge (y={edge_y})")
        ax.plot(edge_x, edge_y, "D", color="saddlebrown", ms=8,
                label=f"Right terminus (x={edge_x})")


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "floor_to_wall"

    if scenario not in SCENARIO_CONFIG:
        print(f"Unknown scenario '{scenario}'. "
              "Choose: floor_to_wall, wall_to_ceiling, outside, thin_edge")
        sys.exit(1)

    cfg = SCENARIO_CONFIG[scenario]
    keyframes = cfg["keyframes"]
    xlim = cfg["xlim"]
    ylim = cfg["ylim"]

    # Pick the keyframe closest to t=0.45 as a representative mid-transition pose
    kf = min(keyframes, key=lambda k: abs(k["t"] - 0.45))

    x1, y1, theta1       = kf["x1"], kf["y1"], kf["theta1"]
    x2, y2, theta2       = kf["x2"], kf["y2"], kf["theta2"]
    constraint_set       = kf["constraint_set"]
    wall_x               = kf.get("wall_x",   WALL_X)
    ceiling_y_kf         = kf.get("ceiling_y", CEILING_Y)
    edge_x               = kf.get("wall_x",   EDGE_X)    # thin_edge repurposes wall_x
    edge_y               = kf.get("ceiling_y", EDGE_Y)   # thin_edge repurposes ceiling_y

    label = f"{scenario}  (t≈{kf['t']:.2f})"

    print("=" * 60)
    print(f"  Snapshot: {label}")
    print(f"  Constraint set : {constraint_set}")
    print(f"  Links          : l1={l1}, l2={l2}, l3={l3}  "
          f"(total = {l1+l2+l3:.2f} m)")
    print(f"  Front hint     : ({x1:.3f}, {y1:.3f}),  θ₁ = {np.degrees(theta1):.1f}°")
    print(f"  Back  hint     : ({x2:.3f}, {y2:.3f}),  θ₂ = {np.degrees(theta2):.1f}°")
    print("-" * 60)

    q, success, info = solve_ik(
        x1, y1, theta1,
        x2, y2, theta2,
        l1, l2, l3,
        wg=wg,
        constraint_set=constraint_set,
        wall_x=wall_x,
        ceiling_y=ceiling_y_kf,
        n_grid=360,
    )

    x1e, y1e = info["x1"], info["y1"]
    x2e, y2e = info["x2"], info["y2"]

    status = "CONVERGED" if success else "best-effort (did not fully converge)"
    print(f"  Solver         : {status}")
    print(f"  Residual norm  : {info['residual_norm']:.2e}")
    print(f"  Effective poses:")
    print(f"    Front: ({x1e:.4f}, {y1e:.4f})")
    print(f"    Back : ({x2e:.4f}, {y2e:.4f})")
    print("-" * 60)

    # Display q in calf-up convention (q1: 0°=up from front calf, q4: 0°=up from back calf)
    q_disp = _q_display(q)
    for i, qi in enumerate(q_disp, 1):
        print(f"  q{i} = {np.degrees(qi):+9.4f}°  ({qi:+.6f} rad)")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(9, 7))

    _draw_surfaces(ax, constraint_set, wall_x, ceiling_y_kf, edge_x, edge_y, xlim, ylim)

    plot_robot(
        q,
        x1e, y1e, theta1,
        x2e, y2e, theta2,
        l1, l2, l3,
        axle_half_length=wg.bar_len,
        axle_a_len=wg.bar_len_a,
        wheel_radius=wg.wheel_r,
        calf=wg.calf,
        arm_a1=wg.arm_a1, arm_b1=wg.arm_b1,
        arm_a2=wg.arm_a2, arm_b2=wg.arm_b2,
        ax=ax,
        title=label,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out = "robot_config.png"
    fig.savefig(out, dpi=150)
    print(f"\n  Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
