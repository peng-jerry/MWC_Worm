"""
Matplotlib visualization of the robot configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
from kinematics import forward_kinematics


def plot_robot(
    q,
    x1, y1, theta1,
    x2, y2, theta2,
    l1, l2, l3,
    axle_half_length=0.140,
    axle_a_len=None,
    wheel_radius=0.050,
    calf=0.042,
    thigh=0.140,
    arm_a1=0.0, arm_b1=np.pi / 4,
    arm_a2=0.0, arm_b2=-np.pi / 4,
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

    _bar_a = axle_a_len if axle_a_len is not None else axle_half_length
    # --- Front and back assemblies (drawn at the actual q1 / q4 positions) ---
    _draw_assembly(ax, positions[0], theta1, axle_half_length, wheel_radius,
                   color="forestgreen", label="Front assembly",
                   calf=calf, thigh=thigh, side=1, arm_a=arm_a1, arm_b=arm_b1,
                   bar_len_a=_bar_a)
    _draw_assembly(ax, positions[-1], theta_end, axle_half_length, wheel_radius,
                   color="crimson", label="Back assembly",
                   calf=calf, thigh=thigh, side=-1, arm_a=arm_a2, arm_b=arm_b2,
                   bar_len_a=_bar_a)

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
                   arm_a=0.0, arm_b=np.pi / 4,
                   calf=0.042, thigh=0.08951, side=1, bar_len_a=None):
    """
    Draw one wheel assembly: L-bracket (thigh + calf) from chain joint to
    V-apex, then two arms from V-apex to each wheel.

    joint_pos : (x, y) of the chain joint (q1 or q4)
    theta     : assembly orientation (forward direction)
    arm_a/arm_b : offsets from centreline c = theta − π/2 for each arm
    side      : +1 for assembly 1 (q1), -1 for assembly 2 (q4)
    """
    joint = np.asarray(joint_pos, dtype=float)

    v_apex = joint + np.array([
        -side * thigh * np.cos(theta) + calf * np.sin(theta),
        -side * thigh * np.sin(theta) - calf * np.cos(theta),
    ])
    elbow = joint + np.array([
        -side * thigh * np.cos(theta),
        -side * thigh * np.sin(theta),
    ])

    center = theta - np.pi / 2
    _ba = bar_len_a if bar_len_a is not None else bar_len
    left_wheel  = v_apex + _ba     * np.array([np.cos(center + arm_a),
                                                np.sin(center + arm_a)])
    right_wheel = v_apex + bar_len * np.array([np.cos(center + arm_b),
                                                np.sin(center + arm_b)])

    # Thigh (joint → elbow) and calf (elbow → V-apex)
    ax.plot([joint[0], elbow[0]], [joint[1], elbow[1]],
            "-", color=color, linewidth=2.0, zorder=2, label=label)
    ax.plot([elbow[0], v_apex[0]], [elbow[1], v_apex[1]],
            "-", color=color, linewidth=2.0, zorder=2)

    # Two arms from V-apex to each wheel
    ax.plot([v_apex[0], left_wheel[0]],  [v_apex[1], left_wheel[1]],
            "-", color=color, linewidth=2.5, zorder=2)
    ax.plot([v_apex[0], right_wheel[0]], [v_apex[1], right_wheel[1]],
            "-", color=color, linewidth=2.5, zorder=2)

    # Wheels
    for wpos in (left_wheel, right_wheel):
        circle = plt.Circle(wpos, wheel_r, color=color, alpha=0.4, zorder=2)
        ax.add_patch(circle)
        ax.plot(*wpos, "o", color=color, markersize=5, zorder=3)

    # Joint marker
    ax.plot(*joint, "s", color=color, markersize=8, zorder=5)
