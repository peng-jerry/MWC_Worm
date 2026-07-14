"""
Snapshot viewer: solve and plot a single characteristic frame from each
animate_transition.py scenario.

Usage:
    python main.py [scenario]

Scenarios: floor_to_wall (default), wall_to_ceiling, outside, thin_edge

Geometry, constraint sets, and snapshot poses are taken directly from
the matching keyframes in animate_transition.py so the output matches
the animation.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from constraints import WheelGeometry
from solver import solve_ik
from visualize import plot_robot


def _draw_surfaces(ax, constraint_set, wall_x, ceiling_y, xlim, ylim):
    """Mirror of animate_transition._draw_surfaces so snapshots look identical."""
    if constraint_set == "wall":
        ax.plot([wall_x, xlim[1]], [0, 0],
                color="saddlebrown", lw=2, label="Floor")
        ax.plot([wall_x, wall_x], [0, ylim[1]],
                color="slategray", lw=2, label=f"Wall (x={wall_x})")

    elif constraint_set == "ceiling":
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
        ax.plot([xlim[0], wall_x], [ceiling_y, ceiling_y],
                color="saddlebrown", lw=3, label=f"Thin edge (y={ceiling_y})")
        ax.plot(wall_x, ceiling_y, "D", color="saddlebrown", ms=8,
                label=f"Right terminus (x={wall_x})")


def main():
    # ------------------------------------------------------------------ #
    #  Shared robot geometry — must match animate_transition.py exactly   #
    # ------------------------------------------------------------------ #
    l1, l2, l3 = 0.25, 0.50, 0.25
    wg = WheelGeometry(bar_len=0.200, wheel_r=0.050, spread=np.pi / 4,
                       calf=0.042, thigh=0.08951)

    WALL_X    = 0.0
    CEILING_Y = 0.55

    scenario = sys.argv[1] if len(sys.argv) > 1 else "floor_to_wall"

    # ------------------------------------------------------------------ #
    #  Per-scenario snapshot pose                                          #
    #                                                                      #
    #  Each pose is taken from a representative keyframe in               #
    #  animate_transition.py (approx t≈0.5) so it shows a mid-transition #
    #  state that matches the animation.                                   #
    # ------------------------------------------------------------------ #

    if scenario == "floor_to_wall":
        # t=0.57: front on wall mid-climb, back on floor approaching.
        # ceiling_y=2.0 matches _FTW_CY — keeps the ceiling penalty inactive.
        constraint_set = "wall"
        wall_x    = WALL_X
        ceiling_y = 2.0
        x1, y1, theta1 = 0.5,  0.0, 0.0
        x2, y2, theta2 = 1.2, 0.0,   0.0
        xlim = (-0.2, 1.8)
        ylim = (-0.2, 1.4)
        label = "Floor → Wall  (t≈0.46)"

    elif scenario == "wall_to_ceiling":
        # t=0.57: front on ceiling sliding right (x1=0.57→1.009 snap), back climbing
        # wall (y2=0.73). ceiling_y=1.2 matches _WTC_CY used throughout that scenario.
        constraint_set = "ceiling"
        wall_x    = WALL_X
        ceiling_y = 1.2
        x1, y1, theta1 = 0.57, 1.2, np.pi
        x2, y2, theta2 = 0.0,  0.73, -np.pi / 2
        xlim = (-0.5, 1.5)
        ylim = (-0.5, 1.5)
        label = "Wall → Ceiling  (t≈0.57)"

    elif scenario == "outside":
        # t≈0.48: front has completed its pivot onto the exterior wall (theta=π/2),
        # back still on ceiling at x2=0.300 waiting for front to finish (sequential design).
        # Front pause keyframe: t=0.465–0.490; back approach begins at t=0.490.
        # _XW=-0.191, _YW=0.541 (from _cj(π/2) with thigh=0.08951).
        # _Y0=0.741  (from _cj(0)[1], the ceiling-top y for the assembly centre).
        constraint_set = "outside_exact"
        wall_x    = WALL_X
        ceiling_y = CEILING_Y
        x1, y1, theta1 = -0.191, 0.541,  np.pi / 2
        x2, y2, theta2 =  0.300, 0.741,  0.0
        xlim = (-0.5, 1.5)
        ylim = (-0.5, 1.0)
        label = "Outside corner  (t≈0.48)"

    elif scenario == "thin_edge":
        # t=0.32: red (back) mid-pivot at phi=π/2 around right terminus;
        # green (front) at x1=0.280 approaching its stop.
        # EDGE_X=0.35, EDGE_Y=0.40.
        # _cp_te(π/2): vx=0.499, vy=0.301
        # _cj_te2(π/2): x2=0.541, y2=0.390  (thigh flips sign for side=-1)
        # _Y0_TE=0.591  (green assembly y on top of edge)
        constraint_set = "thin_edge_exact"
        wall_x    = 0.35   # right terminus x
        ceiling_y = 0.40   # edge y
        x1, y1, theta1 =  0.280, 0.591,  0.0
        x2, y2, theta2 =  0.541, 0.390, -np.pi / 2
        xlim = (-0.6, 1.0)
        ylim = (-0.1, 0.9)
        label = "Thin edge  (t≈0.32)"

    else:
        print(f"Unknown scenario '{scenario}'. "
              "Choose: floor_to_wall, wall_to_ceiling, outside, thin_edge")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  Solve IK                                                            #
    # ------------------------------------------------------------------ #
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
        ceiling_y=ceiling_y,
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
    for i, qi in enumerate(q, 1):
        print(f"  q{i} = {np.degrees(qi):+9.4f}°  ({qi:+.6f} rad)")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Plot                                                                #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 7))

    _draw_surfaces(ax, constraint_set, wall_x, ceiling_y, xlim, ylim)

    plot_robot(
        q,
        x1e, y1e, theta1,
        x2e, y2e, theta2,
        l1, l2, l3,
        axle_half_length=wg.bar_len,
        wheel_radius=wg.wheel_r,
        calf=wg.calf,
        thigh=wg.thigh,
        ax=ax,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(label, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out = "robot_config.png"
    fig.savefig(out, dpi=150)
    print(f"\n  Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
