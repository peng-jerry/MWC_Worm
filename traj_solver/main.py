"""
Example: solve IK for a two-axle robot linkage in 2-D.

Usage:
    python main.py

Edit CONSTRAINT_SET and the parameters below to explore configurations.
"""

import numpy as np
import matplotlib.pyplot as plt

from constraints import WheelGeometry
from solver import solve_ik
from visualize import plot_robot


def main():
    # ------------------------------------------------------------------ #
    #  Constraint set — toggle between:                                    #
    #    "none"  : purely kinematic IK, no physical constraints            #
    #    "floor" : wheels on floor (y=0); linkage stays above floor        #
    #    "wall"  : floor at y=0 + vertical wall at x=wall_x;              #
    #              wheels auto-placed on whichever surface they face       #
    # ------------------------------------------------------------------ #
    CONSTRAINT_SET = "outside"

    # ------------------------------------------------------------------ #
    #  Shared parameters                                                   #
    # ------------------------------------------------------------------ #
    l1, l2, l3 = 1.2, 1.0, 1.2
    wg = WheelGeometry(bar_len=0.25, wheel_r=0.12, spread=np.pi / 4)

    # ------------------------------------------------------------------ #
    #  Pose inputs                                                         #
    #                                                                      #
    #  "floor" / "none":  specify (x, y, theta) for both assemblies.      #
    #    y is overridden in floor mode to ensure wheel contact.            #
    #                                                                      #
    #  "wall":  x/y overridden based on which surface the wheels face.    #
    #    theta=0       → wheels face down  → floor contact (y adjusted)   #
    #    theta=-pi/2   → wheels face left  → wall contact  (x adjusted)   #
    #                                                                      #
    #  "ceiling":  x/y overridden based on which surface the wheels face. #
    #    theta=pi      → wheels face up    → ceiling contact (y adjusted) #
    #    theta=-pi/2   → wheels face left  → wall contact   (x adjusted)  #
    # ------------------------------------------------------------------ #
    if CONSTRAINT_SET == "outside":
        wall_x    = 0.0   # x-coordinate of the vertical wall
        ceiling_y = 3.0   # y-coordinate of the ceiling

        # Front assembly on TOP of ceiling (theta = 0 → V opens downward, wheels press up onto ceiling)
        # y is overridden so lowest wheel is tangent to ceiling from above; x stays as specified
        x1, y1, theta1 = 1.5, 0.0, 0.0

        # Back assembly on exterior face of wall (theta = pi/2 → V opens rightward, wheels press onto wall)
        # x is overridden so rightmost wheel is tangent to wall from the left; y stays as specified
        x2, y2, theta2 = 0.0, 1.5, np.pi / 2

    elif CONSTRAINT_SET == "ceiling":
        wall_x    = 0.0   # x-coordinate of the vertical wall
        ceiling_y = 3.0   # y-coordinate of the ceiling

        # Front assembly on the ceiling (theta = pi → V opens upward)
        # y is overridden to ceiling contact; x stays as specified
        x1, y1, theta1 = 2.5, 0.0, np.pi

        # Back assembly on the wall (theta = -pi/2 → V opens leftward)
        # x is overridden to wall contact; y stays as specified
        x2, y2, theta2 = 0.0, 1.0, -np.pi / 2

    elif CONSTRAINT_SET == "wall":
        wall_x    = 0.0
        ceiling_y = 3.0   # unused but kept for solve_ik signature

        # if theta is -np.pi / 2, x is overriden by wall contact, y is kept the same
        x1, y1, theta1 = 0.0, 2.0, -np.pi / 2

        # Back assembly on the floor (theta = 0 → V opens downward)
        x2, y2, theta2 = 2.5, 0.0, 0.0

    else:
        wall_x    = 0.0
        ceiling_y = 3.0
        #if constraint set = "floor", y is overridden to ensure wheel contact, but x and theta are kept as specified
        x1, y1, theta1 = 0.0, 0.0, 0.0
        x2, y2, theta2 = 1.5, 1.5, np.radians(45)

    # ------------------------------------------------------------------ #

    print("=" * 58)
    print(f"  Wheeled-robot IK  |  constraint set: '{CONSTRAINT_SET}'")
    print("=" * 58)
    if CONSTRAINT_SET in ("wall", "ceiling", "outside"):
        print(f"  Wall at x = {wall_x}")
    if CONSTRAINT_SET in ("ceiling", "outside"):
        print(f"  Ceiling at y = {ceiling_y}")
    print(f"  Front hint: ({x1:.3f}, {y1:.3f}), θ₁ = {np.degrees(theta1):.1f}°")
    print(f"  Back  hint: ({x2:.3f}, {y2:.3f}), θ₂ = {np.degrees(theta2):.1f}°")
    print(f"  Links: l1={l1}, l2={l2}, l3={l3}  (total = {l1+l2+l3:.3f})")
    print("-" * 58)

    q, success, info = solve_ik(
        x1, y1, theta1,
        x2, y2, theta2,
        l1, l2, l3,
        wg=wg,
        constraint_set=CONSTRAINT_SET,
        wall_x=wall_x,
        ceiling_y=ceiling_y,
        n_grid=360,
    )

    x1e, y1e = info["x1"], info["y1"]
    x2e, y2e = info["x2"], info["y2"]

    status = "CONVERGED" if success else "best-effort (did not fully converge)"
    print(f"  Solver status  : {status}")
    print(f"  Residual norm  : {info['residual_norm']:.2e}")
    print(f"  Position error : {info['position_error']:.2e}")
    print(f"  Angle error    : {np.degrees(info['orientation_error']):.4f}°")
    print(f"  Effective poses:")
    print(f"    Front: ({x1e:.4f}, {y1e:.4f}),  Back: ({x2e:.4f}, {y2e:.4f})")
    print("-" * 58)
    for i, qi in enumerate(q, 1):
        print(f"  q{i} = {np.degrees(qi):+9.4f}°  ({qi:+.6f} rad)")
    print("=" * 58)

    # ------------------------------------------------------------------ #
    #  Visualise                                                           #
    # ------------------------------------------------------------------ #
    fig, ax = plot_robot(q, x1e, y1e, theta1, x2e, y2e, theta2, l1, l2, l3,
                         axle_half_length=wg.bar_len,
                         wheel_radius=wg.wheel_r)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if CONSTRAINT_SET == "floor":
        ax.axhline(0, color="saddlebrown", linewidth=2, label="Floor")

    elif CONSTRAINT_SET == "wall":
        # Corner at (wall_x, 0): floor runs right from corner, wall runs up from corner
        ax.plot([wall_x, xlim[1]], [0, 0],
                color="saddlebrown", linewidth=2, label="Floor")
        ax.plot([wall_x, wall_x], [0, ylim[1]],
                color="slategray", linewidth=2, label=f"Wall (x={wall_x})")

    elif CONSTRAINT_SET in ("ceiling", "outside"):
        # Corner at (wall_x, ceiling_y): wall runs down from corner, ceiling runs right from corner
        ax.plot([wall_x, wall_x], [ylim[0], ceiling_y],
                color="slategray", linewidth=2, label=f"Wall (x={wall_x})")
        ax.plot([wall_x, xlim[1]], [ceiling_y, ceiling_y],
                color="dimgray", linewidth=2, label=f"Ceiling (y={ceiling_y})")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    out = "robot_config.png"
    fig.savefig(out, dpi=150)
    print(f"\n  Plot saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
