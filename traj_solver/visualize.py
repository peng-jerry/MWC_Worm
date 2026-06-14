"""
Matplotlib visualization of the robot configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from kinematics import forward_kinematics


def plot_robot(
    q,
    x1, y1, theta1,
    x2, y2, theta2,
    l1, l2, l3,
    axle_half_length=0.25,
    wheel_radius=0.12,
    ax=None,
    title=None,
):
    """
    Draw the full robot configuration.

    Parameters
    ----------
    q : array-like of shape (4,)
        Joint angles [q1, q2, q3, q4] in radians.
    x1, y1, theta1 : float
        Front assembly pose.
    x2, y2, theta2 : float
        Back assembly pose.
    l1, l2, l3 : float
        Link lengths.
    axle_half_length : float
        Length of the diagonal bar from the joint (q1/q4) to the far wheel.
    wheel_radius : float
        Radius drawn for each wheel.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 7))
    else:
        fig = ax.figure

    positions, theta_end = forward_kinematics(q, x1, y1, theta1, l1, l2, l3)

    # --- Linkage chain ---
    ax.plot(
        positions[:, 0], positions[:, 1],
        "-o", color="steelblue", linewidth=2.5, markersize=9,
        zorder=3, label="Linkage",
    )

    # Joint labels and angle annotations
    joint_names = ["q1", "q2", "q3", "q4"]
    for i, (pos, name) in enumerate(zip(positions, joint_names)):
        ax.annotate(
            f"{name}={np.degrees(q[i]):.1f}°",
            xy=pos, xytext=(8, 8), textcoords="offset points",
            fontsize=8, color="steelblue",
        )

    # Link length labels at midpoints
    link_pairs = [(0, 1, l1), (1, 2, l2), (2, 3, l3)]
    for i, j, length in link_pairs:
        mid = 0.5 * (positions[i] + positions[j])
        ax.annotate(
            f"l={length:.2f}",
            xy=mid, xytext=(5, -12), textcoords="offset points",
            fontsize=7, color="gray",
        )

    # --- Front and back assemblies (drawn at the actual q1 / q4 positions) ---
    _draw_assembly(ax, positions[0], theta1, axle_half_length, wheel_radius,
                   color="forestgreen", label="Front assembly")
    _draw_assembly(ax, positions[-1], theta_end, axle_half_length, wheel_radius,
                   color="crimson", label="Back assembly")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)

    q_str = ", ".join(f"{np.degrees(qi):.1f}°" for qi in q)
    if title is None:
        title = f"Robot IK solution\n[q1, q2, q3, q4] = [{q_str}]"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return fig, ax


def _draw_assembly(ax, joint_pos, theta, bar_len, wheel_r, color, label,
                   spread=np.pi / 4):
    """
    Draw one wheel assembly as a wishbone (V-shape).

    The joint (q1 or q4) is the apex of the V.  Two diagonal bars radiate from
    it symmetrically, one to each wheel.  `theta` is the forward direction of
    the assembly (apex points forward); wheels spread behind the joint.

    Parameters
    ----------
    spread : float
        Half-angle of the V opening, in radians (default π/4 = 45°).
    """
    joint = np.asarray(joint_pos, dtype=float)

    # Wheel positions: symmetric about the downward-perpendicular at theta=0.
    # At theta=0 the chain points right (+x), so wheels hang below (-y = -π/2).
    # The rearward centerline is theta - π/2, giving downward at zero degrees.
    center = theta - np.pi / 2
    left_wheel  = joint + bar_len * np.array([np.cos(center - spread),
                                               np.sin(center - spread)])
    right_wheel = joint + bar_len * np.array([np.cos(center + spread),
                                               np.sin(center + spread)])

    # Two diagonal bars from joint apex to each wheel
    ax.plot([joint[0], left_wheel[0]],  [joint[1], left_wheel[1]],
            "-", color=color, linewidth=2.5, zorder=2, label=label)
    ax.plot([joint[0], right_wheel[0]], [joint[1], right_wheel[1]],
            "-", color=color, linewidth=2.5, zorder=2)

    # Wheels
    for wpos in (left_wheel, right_wheel):
        circle = plt.Circle(wpos, wheel_r, color=color, alpha=0.4, zorder=2)
        ax.add_patch(circle)
        ax.plot(*wpos, "o", color=color, markersize=5, zorder=3)

    # Joint apex marker
    ax.plot(*joint, "s", color=color, markersize=8, zorder=5)
