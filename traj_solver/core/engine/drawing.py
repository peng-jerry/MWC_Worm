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

from .constraints import assembly_parts, wheel_centers
from .kinematics import forward_kinematics, q_display as _q_display
from ..robot import JOINT_RADIUS, WheelGeometry, l1, l2, l3

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
    """Draw the solid around the transition (planner.Scenario.env), filled,
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
    draw_surfaces(ax, scn, xlim, ylim, marker_scale)
    draw_outline(ax, scn, q, p, marker_scale)
    plot_robot(ax, q, p, scn.wg, title=title, marker_scale=marker_scale, show_labels=show_labels)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")


# ------------------------------------------------------------------ #
#  Robot                                                              #
# ------------------------------------------------------------------ #

def plot_robot(ax, q, pose, wg=_WG, title=None, marker_scale=1.0, show_labels=True):
    """
    Draw the chain and both wheel assemblies on `ax`.

    q             joint angles [q1, q2, q3, q4] (rad, solver convention)
    pose          places the front assembly: pose["x1"], pose["y1"],
                  pose["theta1"].  The back assembly is drawn where the
                  chain ends, so a q that does not close the chain shows it.
    wg            the wheel assembly geometry (constraints.WheelGeometry)
    title         None gives a default title listing the joint angles
    marker_scale  scales line widths and markers (small panels use < 1)
    show_labels   label each joint with its angle (display convention) and
                  each link with its length
    """
    positions, theta_end = forward_kinematics(q, pose["x1"], pose["y1"], pose["theta1"], l1, l2, l3)
    ms = marker_scale

    # Linkage chain
    ax.plot(positions[:, 0], positions[:, 1], "-o", color="steelblue",
            linewidth=2.5 * ms, markersize=9 * ms, zorder=3, label="Linkage")

    q_disp = _q_display(q)
    if show_labels:
        for i, pos in enumerate(positions):                 # joint angles at the joints
            ax.annotate(f"q{i + 1}={np.degrees(q_disp[i]):.1f}°", xy=pos, xytext=(8, 8),
                        textcoords="offset points", fontsize=8, color="steelblue")
        for i, length in enumerate((l1, l2, l3)):           # link lengths at their midpoints
            mid = 0.5 * (positions[i] + positions[i + 1])
            ax.annotate(f"l={length:.3f}", xy=mid, xytext=(5, -12),
                        textcoords="offset points", fontsize=7, color="gray")

    _draw_assembly(ax, positions[0], pose["theta1"], wg, +1, "forestgreen", "Front assembly", ms)
    _draw_assembly(ax, positions[-1], theta_end, wg, -1, "crimson", "Back assembly", ms)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    if title is None:
        q_str = ", ".join(f"{np.degrees(qi):.1f}°" for qi in q_disp)
        title = f"Robot IK solution\n[q1, q2, q3, q4] = [{q_str}]"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def _draw_assembly(ax, joint, theta, wg, side, color, label, marker_scale):
    """
    Draw one wheel assembly with its chain joint at `joint` and orientation
    theta: the calf from the chain joint to the V-apex, an arm to each wheel,
    and the wheels.  The wheels come from constraints.wheel_centers, so the
    drawing always matches the outline the rules check.  side: +1 front, −1 back.
    """
    joint = np.asarray(joint, dtype=float)
    v_apex = joint + np.array([wg.calf * np.sin(theta), -wg.calf * np.cos(theta)])
    wheels = wheel_centers(joint[0], joint[1], theta, wg, side)      # (arm A, arm B)
    ms = marker_scale

    ax.plot([joint[0], v_apex[0]], [joint[1], v_apex[1]],            # calf
            "-", color=color, linewidth=2.0 * ms, zorder=2, label=label)
    for w in wheels:                                                 # arms
        ax.plot([v_apex[0], w[0]], [v_apex[1], w[1]], "-", color=color, linewidth=2.5 * ms, zorder=2)
    for w in wheels:                                                 # wheels
        ax.add_patch(plt.Circle(w, wg.wheel_r, color=color, alpha=0.4, zorder=2))
        ax.plot(*w, "o", color=color, markersize=5 * ms, zorder=3)
    ax.plot(*joint, "s", color=color, markersize=8 * ms, zorder=5)   # chain joint
