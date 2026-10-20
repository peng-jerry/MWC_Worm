"""
drawing.py — matplotlib drawing of a transition: the solid around it, the
robot's physical outline, and the chain with both wheel assemblies.

  plot_limits   axis limits that contain the whole solved transition
  draw_frame    one state of a transition: surfaces, outline and robot
  plot_robot    the chain and both wheel assemblies (also usable on its own)

Used by animate.py, plot_phases.py and plot_snapshot.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .constraints import JOINT_RADIUS, WheelGeometry, assembly_parts, wheel_centers
from .kinematics import forward_kinematics, q_display as _q_display
from .scenarios import l1, l2, l3

_WG = WheelGeometry()   # default (measured) assembly geometry


# ------------------------------------------------------------------ #
#  Transition                                                         #
# ------------------------------------------------------------------ #

def plot_limits(traj, margin=0.15):
    """
    Axis limits that contain the chain and wheels in every frame, plus a
    margin.  A side fixed by the scenario's xlim / ylim replaces the fitted one.
    """
    scn = traj.scenario
    pts = [np.asarray(scn.approach.origin, dtype=float), np.asarray(scn.depart.origin, dtype=float)]
    for q, _, p in traj:
        chain, _ = forward_kinematics(q, p["x1"], p["y1"], p["theta1"], l1, l2, l3)
        pts.extend(chain)
        pts.extend(wheel_centers(p["x1"], p["y1"], p["theta1"], scn.wg, side=1))
        pts.extend(wheel_centers(p["x2"], p["y2"], p["theta2"], scn.wg, side=-1))
    pts = np.array(pts)
    lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
    return tuple(tuple(fit if fixed is None else fixed for fit, fixed in zip((lo[i], hi[i]), lim))
                 for i, lim in enumerate((scn.xlim, scn.ylim)))


def draw_surfaces(ax, scn, xlim, ylim, marker_scale=1.0):
    """Draw the solid around the transition (scenarios.Scenario.env), filled,
    with its faces outlined."""
    env = scn.env
    for k, poly in enumerate(env.solid_polygons(xlim, ylim)):
        # Opaque, so the pieces of an inside corner's solid overlap invisibly.
        ax.add_patch(Polygon(np.array(poly), closed=True, facecolor="#e4e4e4", edgecolor="none",
                             zorder=0, label="Solid" if k == 0 else None))
    for k, (a, b) in enumerate(env.face_segments()):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="dimgray", lw=2, zorder=1,
                label="Surface" if k == 0 else None)


def _capsule_outline(a, b, r, n=24):
    """Points around a capsule (segment a–b thickened by r), for drawing."""
    d = b - a
    angle = np.arctan2(d[1], d[0]) if np.linalg.norm(d) > 0 else 0.0
    t1 = np.linspace(angle + np.pi / 2, angle + 3 * np.pi / 2, n)
    t2 = np.linspace(angle - np.pi / 2, angle + np.pi / 2, n)
    return np.vstack([a + r * np.stack([np.cos(t1), np.sin(t1)], axis=1),
                      b + r * np.stack([np.cos(t2), np.sin(t2)], axis=1)])


def draw_outline(ax, scn, q, p, marker_scale=1.0):
    """Draw the robot's physical outline (constraints.py) — the shapes the
    clearance rules are checked against — as thin outlines."""
    chain, _ = forward_kinematics(q, p["x1"], p["y1"], p["theta1"], l1, l2, l3)
    shapes = [(chain[i], chain[i + 1], JOINT_RADIUS, "steelblue") for i in range(3)]
    for x, y, theta, side, color in ((p["x1"], p["y1"], p["theta1"], +1, "forestgreen"),
                                     (p["x2"], p["y2"], p["theta2"], -1, "crimson")):
        shapes += [(a, b, r, color) for a, b, r in assembly_parts(x, y, theta, scn.wg, side).values()]
    for a, b, r, color in shapes:
        ax.add_patch(Polygon(_capsule_outline(np.asarray(a), np.asarray(b), r), closed=True,
                             fill=False, edgecolor=color, lw=0.6 * marker_scale, alpha=0.6, zorder=2))


def draw_frame(ax, scn, q, p, xlim, ylim, title=None, marker_scale=1.0, show_labels=True):
    """Draw one state of transition `scn` — joint angles q, assembly pose p:
    environment, chain and both assemblies, with their physical outline."""
    wg = scn.wg
    draw_surfaces(ax, scn, xlim, ylim, marker_scale)
    draw_outline(ax, scn, q, p, marker_scale)
    plot_robot(q, p["x1"], p["y1"], p["theta1"], p["x2"], p["y2"], p["theta2"], l1, l2, l3,
               axle_half_length=wg.bar_len, axle_a_len=wg.bar_len_a, wheel_radius=wg.wheel_r,
               calf=wg.calf, arm_a1=wg.arm_a1, arm_b1=wg.arm_b1, arm_a2=wg.arm_a2, arm_b2=wg.arm_b2,
               ax=ax, title=title, marker_scale=marker_scale, show_labels=show_labels)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")


# ------------------------------------------------------------------ #
#  Robot                                                              #
# ------------------------------------------------------------------ #

def plot_robot(
    q,
    x1, y1, theta1,
    x2, y2, theta2,
    l1, l2, l3,
    axle_half_length=_WG.bar_len,
    axle_a_len=_WG.bar_len_a,
    wheel_radius=_WG.wheel_r,
    calf=_WG.calf,
    arm_a1=_WG.arm_a1, arm_b1=_WG.arm_b1,
    arm_a2=_WG.arm_a2, arm_b2=_WG.arm_b2,
    ax=None,
    title=None,
    marker_scale=1.0,
    show_labels=True,
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

    ms = marker_scale

    # --- Linkage chain ---
    ax.plot(
        positions[:, 0], positions[:, 1],
        "-o", color="steelblue", linewidth=2.5 * ms, markersize=9 * ms,
        zorder=3, label="Linkage",
    )

    q_disp = _q_display(q)

    if show_labels:
        # Joint labels — q1/q4 displayed in calf-up convention (0° = link straight up from calf)
        joint_names = ["q1", "q2", "q3", "q4"]
        for i, (pos, name) in enumerate(zip(positions, joint_names)):
            ax.annotate(
                f"{name}={np.degrees(q_disp[i]):.1f}°",
                xy=pos, xytext=(8, 8), textcoords="offset points",
                fontsize=8, color="steelblue",
            )

        # Link length labels at midpoints
        link_pairs = [(0, 1, l1), (1, 2, l2), (2, 3, l3)]
        for i, j, length in link_pairs:
            mid = 0.5 * (positions[i] + positions[j])
            ax.annotate(
                f"l={length:.3f}",
                xy=mid, xytext=(5, -12), textcoords="offset points",
                fontsize=7, color="gray",
            )

    _bar_a = axle_a_len if axle_a_len is not None else axle_half_length
    # --- Front and back assemblies (drawn at the actual q1 / q4 positions) ---
    _draw_assembly(ax, positions[0], theta1, axle_half_length, wheel_radius,
                   color="forestgreen", label="Front assembly",
                   calf=calf, side=1, arm_a=arm_a1, arm_b=arm_b1,
                   bar_len_a=_bar_a, marker_scale=ms)
    _draw_assembly(ax, positions[-1], theta_end, axle_half_length, wheel_radius,
                   color="crimson", label="Back assembly",
                   calf=calf, side=-1, arm_a=arm_a2, arm_b=arm_b2,
                   bar_len_a=_bar_a, marker_scale=ms)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)

    q_str = ", ".join(f"{np.degrees(qi):.1f}°" for qi in q_disp)
    if title is None:
        title = f"Robot IK solution\n[q1, q2, q3, q4] = [{q_str}]"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return fig, ax


def _draw_assembly(ax, joint_pos, theta, bar_len, wheel_r, color, label,
                   arm_a=_WG.arm_a1, arm_b=_WG.arm_b1,
                   calf=_WG.calf, side=1, bar_len_a=None, marker_scale=1.0):
    """
    Draw one wheel assembly: calf strut from chain joint to V-apex,
    then two arms from V-apex to each wheel.

    joint_pos : (x, y) of the chain joint (q1 or q4)
    theta     : assembly orientation (forward direction)
    arm_a/arm_b : offsets from centreline c = theta − π/2 for each arm
    side      : +1 for assembly 1 (q1), -1 for assembly 2 (q4)
    """
    joint = np.asarray(joint_pos, dtype=float)

    v_apex = joint + np.array([
        calf * np.sin(theta),
        -calf * np.cos(theta),
    ])

    center = theta - np.pi / 2
    _ba = bar_len_a if bar_len_a is not None else bar_len
    left_wheel  = v_apex + _ba     * np.array([np.cos(center + arm_a),
                                                np.sin(center + arm_a)])
    right_wheel = v_apex + bar_len * np.array([np.cos(center + arm_b),
                                                np.sin(center + arm_b)])

    ms = marker_scale

    # Calf (joint → V-apex)
    ax.plot([joint[0], v_apex[0]], [joint[1], v_apex[1]],
            "-", color=color, linewidth=2.0 * ms, zorder=2, label=label)

    # Two arms from V-apex to each wheel
    ax.plot([v_apex[0], left_wheel[0]],  [v_apex[1], left_wheel[1]],
            "-", color=color, linewidth=2.5 * ms, zorder=2)
    ax.plot([v_apex[0], right_wheel[0]], [v_apex[1], right_wheel[1]],
            "-", color=color, linewidth=2.5 * ms, zorder=2)

    # Wheels
    for wpos in (left_wheel, right_wheel):
        circle = plt.Circle(wpos, wheel_r, color=color, alpha=0.4, zorder=2)
        ax.add_patch(circle)
        ax.plot(*wpos, "o", color=color, markersize=5 * ms, zorder=3)

    # Joint marker
    ax.plot(*joint, "s", color=color, markersize=8 * ms, zorder=5)
