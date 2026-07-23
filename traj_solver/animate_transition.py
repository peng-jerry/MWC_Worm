"""
animate_transition.py

Solve IK across a sequence of interpolated poses and render the result as a video.

Scenarios
---------
  "floor_to_wall"    Both assemblies transition from floor to interior wall
                     (inside bottom-left corner).
  "wall_to_ceiling"  Both assemblies transition from interior wall to ceiling
                     (inside top-left corner).
  "outside"          Both assemblies transition from ceiling top to exterior wall
                     (outside top-left corner).
  "thin_edge"        Both assemblies transition from top of a thin horizontal
                     edge to below it, wrapping around the right terminus.

Usage:
    python animate_transition.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constraints import WheelGeometry
from solver import solve_ik
from visualize import plot_robot


# ------------------------------------------------------------------ #
#  Keyframe interpolation                                              #
# ------------------------------------------------------------------ #

def _lerp_angle(a0: float, a1: float, t: float) -> float:
    """Shortest-arc linear interpolation between two angles."""
    diff = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    return a0 + t * diff


def eval_keyframes(keyframes: list, s: float) -> dict:
    """
    Evaluate the keyframe list at global parameter s in [0, 1].

    Each keyframe dict has keys:
        t              — parameter value in [0, 1]
        x1, y1, theta1 — front assembly pose hints (radians)
        x2, y2, theta2 — back assembly pose hints (radians)
        constraint_set — active constraint set for this segment
        wall_x         — x of vertical wall
        ceiling_y      — y of ceiling

    Positions are linearly interpolated; angles take the shortest arc.
    """
    times = [kf["t"] for kf in keyframes]

    if s <= times[0]:
        return {k: v for k, v in keyframes[0].items() if k != "t"}
    if s >= times[-1]:
        return {k: v for k, v in keyframes[-1].items() if k != "t"}

    for i in range(len(times) - 1):
        if times[i] <= s <= times[i + 1]:
            lo, hi = keyframes[i], keyframes[i + 1]
            t = (s - lo["t"]) / (hi["t"] - lo["t"])
            return {
                "x1":     lo["x1"]  + t * (hi["x1"]  - lo["x1"]),
                "y1":     lo["y1"]  + t * (hi["y1"]  - lo["y1"]),
                "theta1": _lerp_angle(lo["theta1"], hi["theta1"], t),
                "x2":     lo["x2"]  + t * (hi["x2"]  - lo["x2"]),
                "y2":     lo["y2"]  + t * (hi["y2"]  - lo["y2"]),
                "theta2": _lerp_angle(lo["theta2"], hi["theta2"], t),
                "constraint_set": lo["constraint_set"],
                "wall_x":    lo["wall_x"],
                "ceiling_y": lo["ceiling_y"],
            }


# ------------------------------------------------------------------ #
#  Surface contact helpers                                             #
# ------------------------------------------------------------------ #

def _wheel_surface_adjust(x, y, theta, wg, side=1,
                           wall_x=None, floor_y=None, ceiling_y=None):
    """
    Pre-correct (x, y) so every wheel clears each active INTERIOR surface.

    floor_y   — raise the assembly if any wheel dips below the floor
    ceiling_y — lower the assembly if any wheel pokes above the ceiling
    wall_x    — shift right if any wheel enters the (interior) wall

    Handles the dead-zone near +-45 deg where the binary |sin_c|/|cos_c|
    check in constraints.py doesn't fire for either surface.
    """
    from constraints import _v_apex
    vx, vy = _v_apex(x, y, theta, wg, side)
    c = theta - np.pi / 2
    arm_a, arm_b = (wg.arm_a1, wg.arm_b1) if side == 1 else (wg.arm_a2, wg.arm_b2)
    wheel_xs = [vx + wg.bar_len_a * np.cos(c + arm_a),
                vx + wg.bar_len   * np.cos(c + arm_b)]
    wheel_ys = [vy + wg.bar_len_a * np.sin(c + arm_a),
                vy + wg.bar_len   * np.sin(c + arm_b)]

    if floor_y is not None:
        min_wy = min(wheel_ys)
        if min_wy < floor_y + wg.wheel_r:
            delta = (floor_y + wg.wheel_r) - min_wy
            y += delta
            wheel_ys = [wy + delta for wy in wheel_ys]

    if ceiling_y is not None:
        max_wy = max(wheel_ys)
        if max_wy > ceiling_y - wg.wheel_r:
            y -= max_wy - (ceiling_y - wg.wheel_r)

    if wall_x is not None:
        min_wx = min(wheel_xs)
        if min_wx < wall_x + wg.wheel_r:
            x += (wall_x + wg.wheel_r) - min_wx

    return x, y


def _cj_pivot(theta, pivot_pos, arm_angle, arm_len, wg, side=1):
    """
    Chain-joint (x, y) while an assembly pivots about a fixed wheel centre.

    theta      : assembly orientation
    pivot_pos  : (x, y) of the fixed wheel centre
    arm_angle  : angle offset from centreline c=theta-π/2 to that wheel
    arm_len    : bar length from V-apex to that wheel (bar_len or bar_len_a)
    side       : +1 front assembly, -1 back assembly
    """
    c  = theta - np.pi / 2
    vx = pivot_pos[0] - arm_len * np.cos(c + arm_angle)
    vy = pivot_pos[1] - arm_len * np.sin(c + arm_angle)
    return (vx + side * wg.thigh * np.cos(theta) - wg.calf * np.sin(theta),
            vy + side * wg.thigh * np.sin(theta) + wg.calf * np.cos(theta))


# ------------------------------------------------------------------ #
#  IK trajectory solver                                                #
# ------------------------------------------------------------------ #

def solve_trajectory(keyframes, wg, l1, l2, l3, n_frames, n_grid=60,
                     smooth_weight=1.0, cold_at=None, seed_at=None):
    """
    Solve IK at n_frames evenly-spaced parameter values in [0, 1].

    Warm-starts each frame from the previous solution and scores candidates
    with an additional smooth_weight * ||q - q_prev||² term so the solver
    prefers configurations close to the previous frame's solution.
    Returns a list of (q, info, params) tuples.

    Parameters
    ----------
    cold_at : float, list of floats, or None
        Reset the warm-start (q_prev=None) each time s first reaches or
        exceeds one of the listed values.  Each value fires at most once.
        Use this to force a branch-switch at known keyframe boundaries.
    seed_at : dict {float: array-like(4)}, or None
        Provide a specific q_init the first time s reaches or exceeds each
        key.  Unlike cold_at (which drops smooth_weight), the seed is used
        as a warm-start with smooth_weight active — the solver finds the
        local minimum nearest the seed.  Use this to steer branch selection
        without a large discontinuity.  Each threshold fires at most once.
        cold_at overrides seed_at if both fire on the same frame.
    """
    results = []
    q_prev  = None
    # Normalise cold_at to a set of thresholds (each fires once).
    if cold_at is None:
        _cold_pending = set()
    elif isinstance(cold_at, (int, float)):
        _cold_pending = {float(cold_at)}
    else:
        _cold_pending = set(map(float, cold_at))
    # Normalise seed_at: dict {float threshold -> ndarray(4)}
    if seed_at is None:
        _seed_pending = {}
    else:
        _seed_pending = {float(k): np.asarray(v, dtype=float)
                         for k, v in seed_at.items()}

    for idx, s in enumerate(np.linspace(0, 1, n_frames)):
        p = eval_keyframes(keyframes, s)
        cs = p["constraint_set"]

        # Pre-correct hints before passing to solve_ik to handle the
        # dead-zone near +-45 deg where the binary surface selector in
        # constraints.py doesn't fire for either surface.
        if cs == "wall":
            # Inside floor-wall corner.
            p["x1"], p["y1"] = _wheel_surface_adjust(
                p["x1"], p["y1"], p["theta1"], wg, side=1,
                wall_x=p["wall_x"], floor_y=0.0)
            p["x2"], p["y2"] = _wheel_surface_adjust(
                p["x2"], p["y2"], p["theta2"], wg, side=-1,
                wall_x=p["wall_x"], floor_y=0.0)

        elif cs == "ceiling":
            # Inside wall-ceiling corner.  floor_y=-0.4 matches the wall bottom
            # so assemblies can start low and drive up to the ceiling corner.
            p["x1"], p["y1"] = _wheel_surface_adjust(
                p["x1"], p["y1"], p["theta1"], wg, side=1,
                wall_x=p["wall_x"], floor_y=-0.4, ceiling_y=p["ceiling_y"])
            p["x2"], p["y2"] = _wheel_surface_adjust(
                p["x2"], p["y2"], p["theta2"], wg, side=-1,
                wall_x=p["wall_x"], floor_y=-0.4, ceiling_y=p["ceiling_y"])

        elif cs == "wall_exact":
            # Exact positions from keyframes (floor/wall rotation transit);
            # no pre-adjustment — solve_ik skips contact snap for wall_exact.
            pass

        elif cs == "ceiling_exact":
            # Exact positions for the pivoting assembly; WSA for the static assembly.
            # Front rotation (theta1 ≠ π): assembly 1 is pivoting (exact hint),
            #   apply WSA to assembly 2 so it stays wall-snapped correctly.
            # Back rotation (theta1 = π): assembly 2 is pivoting (exact hint),
            #   apply WSA to assembly 1 (at theta=π, WSA gives same y as hint — safe).
            if abs(p["theta1"] - np.pi) > 0.01:
                # Front rotation phase
                p["x2"], p["y2"] = _wheel_surface_adjust(
                    p["x2"], p["y2"], p["theta2"], wg, side=-1,
                    wall_x=p["wall_x"], floor_y=-0.4, ceiling_y=p["ceiling_y"])
            else:
                # Back rotation phase
                p["x1"], p["y1"] = _wheel_surface_adjust(
                    p["x1"], p["y1"], p["theta1"], wg, side=1,
                    wall_x=p["wall_x"], floor_y=-0.4, ceiling_y=p["ceiling_y"])

        elif cs in ("outside", "outside_exact"):
            # "outside": apply_outside_constraint in solve_ik handles surface
            # placement for ceiling-top and wall-exterior orientations.
            # "outside_exact": exact pivot positions provided in keyframes;
            # no pre-adjustment needed (solve_ik skips contact adjustment).
            pass

        # Seeded warm-starts: replace q_prev with a specific seed (keeps smooth_weight).
        for _sv in sorted(_seed_pending.keys()):
            if s >= _sv:
                q_prev = _seed_pending.pop(_sv)
                break
        # Forced cold-starts: drop warm-start (smooth_weight=0 → global min search).
        # cold_at takes precedence over seed_at if both fire on the same frame.
        for _cv in sorted(_cold_pending):
            if s >= _cv:
                q_prev = None
                _cold_pending.discard(_cv)
                break

        try:
            # The outside penalty landscape is complex enough that the default
            # grid rarely finds seeds in the corner-wrapping region.  Use 6x
            # the normal grid density to match main.py's n_grid=360 baseline.
            ik_grid = n_grid * 6
            q, ok, info = solve_ik(
                p["x1"], p["y1"], p["theta1"],
                p["x2"], p["y2"], p["theta2"],
                l1, l2, l3,
                wg=wg,
                constraint_set=cs,
                wall_x=p["wall_x"],
                ceiling_y=p["ceiling_y"],
                q_init=q_prev,
                smooth_weight=smooth_weight,
                n_grid=ik_grid,
            )
            q_prev = q.copy()

        except Exception as exc:
            print(f"  [frame {idx + 1:3d}] WARNING: {exc}")
            q    = q_prev.copy() if q_prev is not None else np.zeros(4)
            info = {
                "x1": p["x1"], "y1": p["y1"],
                "x2": p["x2"], "y2": p["y2"],
                "residual_norm": float("inf"),
            }

        results.append((q, info, p))
        res = info["residual_norm"]
        tag = "ok" if res < 1e-4 else f"WARN res={res:.2e}"
        print(f"  frame {idx + 1:3d}/{n_frames}  s={s:.3f}  {tag}")

    return results


# ------------------------------------------------------------------ #
#  Rendering                                                           #
# ------------------------------------------------------------------ #

def _draw_surfaces(ax, constraint_set, wall_x, ceiling_y, xlim, ylim):
    """Overlay constraint-set surface geometry on ax."""
    if constraint_set == "floor":
        ax.axhline(0, color="saddlebrown", lw=2, label="Floor")

    elif constraint_set == "wall":
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
        # Filled interior block so it's clear the robot wraps around the outside.
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


def render_video(results, wg, l1, l2, l3, xlim, ylim,
                 output_file="transition.mp4", fps=20):
    """Render solved frames to a video file (MP4 or GIF)."""

    fig, ax = plt.subplots(figsize=(10, 7))

    def draw_frame(idx):
        ax.cla()
        q, info, p = results[idx]

        _draw_surfaces(ax, p["constraint_set"], p["wall_x"], p["ceiling_y"],
                       xlim, ylim)

        plot_robot(
            q,
            info["x1"], info["y1"], p["theta1"],
            info["x2"], info["y2"], p["theta2"],
            l1, l2, l3,
            axle_half_length=wg.bar_len,
            axle_a_len=wg.bar_len_a,
            wheel_radius=wg.wheel_r,
            calf=wg.calf,
            thigh=wg.thigh,
            arm_a1=wg.arm_a1, arm_b1=wg.arm_b1,
            arm_a2=wg.arm_a2, arm_b2=wg.arm_b2,
            ax=ax,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(
            f"Frame {idx + 1}/{len(results)}  —  "
            + "  ".join(f"q{i+1}={np.degrees(qi):+.1f}°" for i, qi in enumerate(q)),
            fontsize=9,
        )

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=len(results), interval=1000 / fps
    )

    saved = False
    if output_file.endswith(".mp4"):
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(output_file, writer=writer, dpi=120)
            print(f"\nSaved -> {output_file}")
            saved = True
        except (OSError, Exception):
            pass

    if not saved:
        gif_file = os.path.splitext(output_file)[0] + ".gif"
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_file, writer=writer, dpi=100)
        print(f"\nSaved -> {gif_file}  (install ffmpeg for MP4)")

    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────── #
#  Scenario data — defined at module level so wheel_omega.py can import.     #
# ────────────────────────────────────────────────────────────────────────── #

l1, l2, l3 = 0.25, 0.50, 0.25
wg = WheelGeometry(bar_len=0.200, wheel_r=0.050,
                   arm_a1=0.0, arm_b1=np.pi / 4,
                   arm_a2=0.0, arm_b2=-np.pi / 4,
                   calf=0.042, thigh=0.08951)

WALL_X    = 0.0
CEILING_Y = 0.55

EDGE_X = 0.44   # right terminus x for "thin_edge"  (0.35 × 1.25)
EDGE_Y = 0.50   # edge height for "thin_edge"      (0.40 × 1.25)

# ── Scenario 1: floor -> wall  (inside bottom-left corner) ──────────────────
#
#  Both assemblies start on the floor (theta=0) and end on the left wall
#  (theta=-pi/2).  Front slides to the corner, rotates 90 deg, climbs the
#  wall; then the back repeats the same steps.
#
#  Chain l1+l2+l3=1.00 m.  Start/end in a near-flat (≥94%) configuration.
#  Minimum assembly separation ≥0.55 m throughout so q1 and q4 stay well
#  below 135°.
#
#  _FTW_CY = 2.0: tall room, ceiling never interferes.

_FTW_CY = 2.0

# Inside corner pivot: arm_a1 (front) and arm_b2 (back) each touch
# this point when their assembly first reaches the corner.
_FTW_CORNER = (WALL_X + wg.wheel_r, wg.wheel_r)

# Pivot helpers — compute chain-joint position at each rotation step.
# Front: arm_a1 (angle=0, length=bar_len_a) fixed at corner.
# Back:  arm_b2 (angle=arm_b2, length=bar_len) fixed at corner.
def _ftw1(th):
    return _cj_pivot(th, _FTW_CORNER, wg.arm_a1,  wg.bar_len_a, wg, side= 1)
def _ftw2(th):
    return _cj_pivot(th, _FTW_CORNER, wg.arm_b2,  wg.bar_len,   wg, side=-1)

KEYFRAMES_FLOOR_TO_WALL = [
    # ---- Approach at 3.000/unit (ω=0.674 rad/frame ✓), sep=0.990m constant ----
    {"t": 0.000, "x1": 0.4, "y1": 0.0, "theta1": 0.0,
                 "x2": 1.0, "y2": 0.0, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # Front arm_a1 reaches corner; back holds. WSA corrects x1→_ftw1(0)[0]≈0.140,
    # y1→_ftw1(0)[1]≈0.233 via floor-snap.  Back y2=0.233 matches floor-snap.
    {"t": 0.15, "x1": 0.0, "y1": 0.0, "theta1": 0.0,
                 "x2": 1.00, "y2": 0.233, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # ---- Front rotation: arm_a1 pivots at corner, theta: 0 → -π/2 ----
    # Assembly rotates like a rigid line between the two wheels; x1,y1 computed
    # from the pivot formula so hints match the geometry exactly.
    {"t": 0.180, "x1": _ftw1(-np.pi / 12)[0], "y1": _ftw1(-np.pi / 12)[1],
                 "theta1": -np.pi / 12,
                 "x2": 0.980, "y2": 0.233, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.205, "x1": _ftw1(-np.pi / 6)[0],  "y1": _ftw1(-np.pi / 6)[1],
                 "theta1": -np.pi / 6,
                 "x2": 0.980, "y2": 0.233, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.228, "x1": _ftw1(-np.pi / 4)[0],  "y1": _ftw1(-np.pi / 4)[1],
                 "theta1": -np.pi / 4,
                 "x2": 0.960, "y2": 0.233, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.248, "x1": _ftw1(-np.pi / 3)[0],  "y1": _ftw1(-np.pi / 3)[1],
                 "theta1": -np.pi / 3,
                 "x2": 0.960, "y2": 0.233, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.298, "x1": _ftw1(-5 * np.pi / 12)[0], "y1": _ftw1(-5 * np.pi / 12)[1],
                 "theta1": -5 * np.pi / 12,
                 "x2": 0.958, "y2": 0.233, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # theta=-90°: front locks on wall.  y1=floor_y_for_assembly(-π/2)=0.102 so
    # the hint interpolation to y1=0.42 at t=0.44 stays within ω=0.70.
    {"t": 0.328, "x1": _ftw1(-np.pi / 2)[0],  "y1": 0.102,
                 "theta1": -np.pi / 2,
                 "x2": 0.958, "y2": 0.233, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # ---- Both moving: front climbs, back slowly approaches ----
    {"t": 0.44, "x1": 0.0, "y1": 0.42, "theta1": -np.pi / 2,
                 "x2": 0.929, "y2": 0.0, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.51, "x1": 0.0, "y1": 0.62, "theta1": -np.pi / 2,
                 "x2": 0.711, "y2": 0.0, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.58, "x1": 0.0, "y1": 0.82, "theta1": -np.pi / 2,
                 "x2": 0.493, "y2": 0.0, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # Back arm_b2 reaches corner: x2,y2 from pivot formula at theta=0.
    {"t": 0.735, "x1": 0.0, "y1": 1.0, "theta1": -np.pi / 2,
                 "x2": _ftw2(0.0)[0], "y2": _ftw2(0.0)[1], "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # ---- Back rotation: arm_b2 pivots at corner, theta: 0 → -π/2 ----
    {"t": 0.76, "x1": 0.0, "y1": 1.0, "theta1": -np.pi / 2,
                 "x2": _ftw2(-np.pi / 12)[0], "y2": _ftw2(-np.pi / 12)[1],
                 "theta2": -np.pi / 12,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.785, "x1": 0.0, "y1": 1.0, "theta1": -np.pi / 2,
                 "x2": _ftw2(-np.pi / 6)[0],  "y2": _ftw2(-np.pi / 6)[1],
                 "theta2": -np.pi / 6,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.810, "x1": 0.0, "y1": 1.0, "theta1": -np.pi / 2,
                 "x2": _ftw2(-np.pi / 4)[0],  "y2": _ftw2(-np.pi / 4)[1],
                 "theta2": -np.pi / 4,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.835, "x1": 0.0, "y1": 1.0, "theta1": -np.pi / 2,
                 "x2": _ftw2(-np.pi / 3)[0],  "y2": _ftw2(-np.pi / 3)[1],
                 "theta2": -np.pi / 3,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.86, "x1": 0.0, "y1": 1.0, "theta1": -np.pi / 2,
                 "x2": _ftw2(-5 * np.pi / 12)[0], "y2": _ftw2(-5 * np.pi / 12)[1],
                 "theta2": -5 * np.pi / 12,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.885, "x1": 0.0, "y1": 1.0, "theta1": -np.pi / 2,
                 "x2": _ftw2(-np.pi / 2)[0],  "y2": _ftw2(-np.pi / 2)[1],
                 "theta2": -np.pi / 2,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 1.000, "x1": 0.0, "y1": 1.2, "theta1": -np.pi / 2,
                 "x2": 0.0,  "y2": 0.3,  "theta2": -np.pi / 2,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
]

# ── Scenario 2: wall -> ceiling  (inside top-left corner) ───────────────────
#
#  Both assemblies start on the left wall (theta=-pi/2) and end on the ceiling
#  (theta=pi, shortest arc through -pi).
#
#  Near-flat start and end: chain fully extended along the thigh direction
#  (q≈0) at both ends.

_WTC_CY = 1.5   # local ceiling height for this scenario

KEYFRAMES_WALL_TO_CEILING = [
    # Both on wall, near-flat vertical chain (~89% of 1.00 m chain).
    # Front (side=+1) above back (side=-1): keeps them from crossing.
    {"t": 0.00, "x1": 0.0, "y1": 1.19, "theta1": -np.pi / 2,
                "x2": 0.0, "y2": 0.40,  "theta2": -np.pi / 2,
                "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # Front slides to corner.
    {"t": 0.10, "x1": 0.0, "y1": _WTC_CY, "theta1": -np.pi / 2,
                "x2": 0.0, "y2": 0.40,     "theta2": -np.pi / 2,
                "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # Front rotates at corner (-pi/2 → pi via -pi, CW 90°) in 15° steps.
    # y2 held at 0.40 during rotation to keep chain within reach (dist≤1.0m);
    # ramps to 0.30 by t=0.250 so back assembly lags at the start of the slide.
    {"t": 0.125, "x1": 0.0, "y1": _WTC_CY, "theta1": -7 * np.pi / 12,
                 "x2": 0.0, "y2": 0.40,     "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.150, "x1": 0.0, "y1": _WTC_CY, "theta1": -2 * np.pi / 3,
                 "x2": 0.0, "y2": 0.37,     "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.175, "x1": 0.0, "y1": _WTC_CY, "theta1": -3 * np.pi / 4,
                 "x2": 0.0, "y2": 0.30,     "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.200, "x1": 0.0, "y1": _WTC_CY, "theta1": -5 * np.pi / 6,
                 "x2": 0.0, "y2": 0.30,     "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.225, "x1": 0.0, "y1": _WTC_CY, "theta1": -11 * np.pi / 12,
                 "x2": 0.0, "y2": 0.30,     "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.250, "x1": 0.0, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0, "y2": 0.30,     "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # Front slides right on ceiling; back climbs wall simultaneously.
    # Same x1/y2 profile as e57e82a; Δt stretched so ω ≤ 0.69 (wheel_r=0.050).
    # y2 lags x1 early to keep chain extension ≥75% in the first step.
    {"t": 0.331, "x1": 0.25, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0,  "y2": 0.44,     "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.412, "x1": 0.50, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0,  "y2": 0.69,     "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.484, "x1": 0.71, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0,  "y2": 0.91,     "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.556, "x1": 0.88, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0,  "y2": 1.13,     "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.616, "x1": 1.00, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0,  "y2": 1.31,     "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # Back arrives at corner; front holds at x1=1.06.
    {"t": 0.679, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0,  "y2": _WTC_CY,  "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # Back rotates at corner (-pi/2 → pi via -pi, CW 90°) in 15° steps.
    {"t": 0.704, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0,  "y2": _WTC_CY,  "theta2": -7 * np.pi / 12,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.729, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0,  "y2": _WTC_CY,  "theta2": -2 * np.pi / 3,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.754, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0,  "y2": _WTC_CY,  "theta2": -3 * np.pi / 4,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.779, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0,  "y2": _WTC_CY,  "theta2": -5 * np.pi / 6,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.804, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0,  "y2": _WTC_CY,  "theta2": -11 * np.pi / 12,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": 0.829, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.0,  "y2": _WTC_CY,  "theta2": np.pi,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # Both on ceiling; spread to near-flat end (~90% of 1.00 m chain).
    {"t": 1.00,  "x1": 1.20, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": 0.30, "y2": _WTC_CY,  "theta2": np.pi,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
]

# ── Scenario 3: ceiling-top -> exterior wall  (outside top-left) ─────────────
#
#  Each assembly pivots about its arm_a (leading) wheel at the corner.
#  Assembly 1 (side=+1): arm_a1 wheel is the pivot.
#  Assembly 2 (side=-1): arm_a2 wheel is the pivot.
#
#  All keyframes use constraint_set="outside_exact" to pass exact chain-joint
#  positions without surface-contact clamping.

_R    = wg.wheel_r    # 0.050
_BARA = wg.bar_len_a  # ≈ 0.141  (arm_a lever length)
_THG  = wg.thigh      # 0.08951
_CLF  = wg.calf       # 0.042


def _cj(phi):
    """Chain-joint for assembly 1 (side=+1): pivots about arm_a1 wheel."""
    awc_x = WALL_X - _R * np.sin(phi)
    awc_y = CEILING_Y + _R * np.cos(phi)
    arm_dir = phi - np.pi / 2 + wg.arm_a1   # arm_a1 direction at theta=phi
    vx    = awc_x - _BARA * np.cos(arm_dir)
    vy    = awc_y - _BARA * np.sin(arm_dir)
    return (vx + _THG * np.cos(phi) - _CLF * np.sin(phi),
            vy + _THG * np.sin(phi) + _CLF * np.cos(phi))


def _cj2(phi):
    """Chain-joint for assembly 2 (side=-1): pivots about arm_a2 wheel."""
    awc_x = WALL_X - _R * np.sin(phi)
    awc_y = CEILING_Y + _R * np.cos(phi)
    arm_dir = phi - np.pi / 2 + wg.arm_a2   # arm_a2 direction at theta=phi
    vx    = awc_x - _BARA * np.cos(arm_dir)
    vy    = awc_y - _BARA * np.sin(arm_dir)
    return (vx - _THG * np.cos(phi) - _CLF * np.sin(phi),
            vy - _THG * np.sin(phi) + _CLF * np.cos(phi))


# Assembly 1 stop: right-wheel centre at (WALL_X, CEILING_Y+R).
_XSTP  = _cj(0)[0]           # chain-joint x when phi=0
_Y0    = _cj(0)[1]           # chain-joint y when phi=0 (ceiling-contact y)
# Assembly 2 stop: right-wheel centre at (WALL_X, CEILING_Y+R).
_XSTP2 = _cj2(0)[0]
# End of pivot for each assembly (phi=π/2).
_XW    = _cj(np.pi / 2)[0]
_YW    = _cj(np.pi / 2)[1]
_YW2   = _cj2(np.pi / 2)[1]

_CSE   = "outside_exact"  # exact positions; outside penalties still on


def _y1_descent(t):
    """Front y during linear descent from _YW at t=0.490 at 1.75/unit."""
    return _YW - 1.75 * (t - 0.490)


KEYFRAMES_OUTSIDE = [
    # Phase 0 — both approach on ceiling (theta=0), moving left at ≤3.0/unit.
    # front: 0.82→_XSTP, back: 1.15→0.300
    {"t": 0.000, "x1": 0.76,   "y1": _Y0, "theta1": 0.0,
                 "x2": 1.150,  "y2": _Y0, "theta2": 0.0,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Front arrives at its stop; back at x2=0.30.
    {"t": 0.290, "x1": _XSTP,  "y1": _Y0, "theta1": 0.0,
                 "x2": 0.300,  "y2": _Y0, "theta2": 0.0,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Front stationary at corner for 0.025 before rotation; back holds at x2=0.300.
    {"t": 0.315, "x1": _XSTP, "y1": _Y0, "theta1": 0.0,
                 "x2": 0.300, "y2": _Y0, "theta2": 0.0,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Front rotates (phi: 0 → π/2): 6 steps × 15° × Δt=0.025 = 0.150 total.
    # Back holds stationary at x2=0.300 while front completes its rotation.
    {"t": 0.340, "x1": _cj(    np.pi / 12)[0], "y1": _cj(    np.pi / 12)[1],
                 "theta1":     np.pi / 12,
                 "x2": 0.300,                   "y2": _Y0, "theta2": 0.0,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": 0.365, "x1": _cj(    np.pi /  6)[0], "y1": _cj(    np.pi /  6)[1],
                 "theta1":     np.pi /  6,
                 "x2": 0.300,                   "y2": _Y0, "theta2": 0.0,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": 0.390, "x1": _cj(    np.pi /  4)[0], "y1": _cj(    np.pi /  4)[1],
                 "theta1":     np.pi /  4,
                 "x2": 0.300,                   "y2": _Y0, "theta2": 0.0,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": 0.415, "x1": _cj(    np.pi /  3)[0], "y1": _cj(    np.pi /  3)[1],
                 "theta1":     np.pi /  3,
                 "x2": 0.300,                   "y2": _Y0, "theta2": 0.0,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": 0.440, "x1": _cj(5 * np.pi / 12)[0], "y1": _cj(5 * np.pi / 12)[1],
                 "theta1": 5 * np.pi / 12,
                 "x2": 0.300,                    "y2": _Y0, "theta2": 0.0,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Front rotation done — fully on exterior wall; back still on ceiling at x2=0.300.
    {"t": 0.465, "x1": _XW, "y1": _YW, "theta1": np.pi / 2,
                 "x2": 0.300,            "y2": _Y0, "theta2": 0.0,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Both pause for 0.025; then back starts approaching while front descends.
    {"t": 0.490, "x1": _XW, "y1": _YW, "theta1": np.pi / 2,
                 "x2": 0.300,            "y2": _Y0, "theta2": 0.0,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Back approaches at ≈2.72/unit, arriving at _XSTP2 at t=0.670.
    # Front descends at 1.75/unit from _YW throughout.
    {"t": 0.670, "x1": _XW, "y1": _y1_descent(0.670), "theta1": np.pi / 2,
                 "x2": _XSTP2,               "y2": _Y0, "theta2": 0.0,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Back stationary at corner for 0.025 before rotation.
    {"t": 0.695, "x1": _XW, "y1": _y1_descent(0.695), "theta1": np.pi / 2,
                 "x2": _XSTP2,               "y2": _Y0, "theta2": 0.0,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Back rotates (phi: 0 → π/2): 6 steps × 15° × Δt=0.025 = 0.150 total.
    # Front descends at 1.75/unit throughout.
    {"t": 0.720, "x1": _XW, "y1": _y1_descent(0.720),
                 "theta1": np.pi / 2,
                 "x2": _cj2(    np.pi / 12)[0], "y2": _cj2(    np.pi / 12)[1],
                 "theta2":     np.pi / 12,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": 0.745, "x1": _XW, "y1": _y1_descent(0.745),
                 "theta1": np.pi / 2,
                 "x2": _cj2(    np.pi /  6)[0], "y2": _cj2(    np.pi /  6)[1],
                 "theta2":     np.pi /  6,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": 0.770, "x1": _XW, "y1": _y1_descent(0.770),
                 "theta1": np.pi / 2,
                 "x2": _cj2(    np.pi /  4)[0], "y2": _cj2(    np.pi /  4)[1],
                 "theta2":     np.pi /  4,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": 0.795, "x1": _XW, "y1": _y1_descent(0.795),
                 "theta1": np.pi / 2,
                 "x2": _cj2(    np.pi /  3)[0], "y2": _cj2(    np.pi /  3)[1],
                 "theta2":     np.pi /  3,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": 0.820, "x1": _XW, "y1": _y1_descent(0.820),
                 "theta1": np.pi / 2,
                 "x2": _cj2(5 * np.pi / 12)[0], "y2": _cj2(5 * np.pi / 12)[1],
                 "theta2": 5 * np.pi / 12,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Back rotation done — fully on exterior wall.
    {"t": 0.845, "x1": _XW, "y1": _y1_descent(0.845),
                 "theta1": np.pi / 2,
                 "x2": _XW,  "y2": _YW2, "theta2": np.pi / 2,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Both descend at ≤3.0/unit.
    {"t": 1.000, "x1": _XW, "y1": -0.350, "theta1": np.pi / 2,
                 "x2": _XW,  "y2":  0.125, "theta2": np.pi / 2,
                 "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
]

# ── Scenario 4: top of thin edge -> below thin edge ──────────────────────────
#
#  Both assemblies start ON TOP of a thin horizontal edge and end BELOW it,
#  wrapping clockwise (theta: 0 → −π) around the right terminus at
#  (EDGE_X, EDGE_Y).
#
#  constraint_set="thin_edge_exact" is used during pivot phases.

_R_TE   = wg.wheel_r    # 0.050
_BAR_TE = wg.bar_len_a  # arm_a1 pivot wheel length (≈ bar_len/√2 for arm_b1=π/4)


def _cp_te(phi):
    """Body center during CW pivot: arm_a wheel orbits terminus (EDGE_X, EDGE_Y) at radius wheel_r. theta=−phi."""
    _orbit = _R_TE + _BAR_TE
    return (EDGE_X + _orbit * np.sin(phi),
            EDGE_Y + _orbit * np.cos(phi))


def _cj_te(phi):
    """Chain-joint for side=+1 assembly during CW pivot. theta=−phi."""
    vx, vy = _cp_te(phi)
    return (vx + wg.thigh * np.cos(phi) + wg.calf * np.sin(phi),
            vy - wg.thigh * np.sin(phi) + wg.calf * np.cos(phi))


def _cj_te2(phi):
    """Chain-joint for side=-1 assembly during CW pivot. theta=−phi."""
    vx, vy = _cp_te(phi)
    return (vx - wg.thigh * np.cos(phi) + wg.calf * np.sin(phi),
            vy + wg.thigh * np.sin(phi) + wg.calf * np.cos(phi))


_XSTP_TE  = _cj_te(0)[0]    # green stop: arm_a1 wheel at terminus
_Y0_TE    = _cj_te(0)[1]    # chain-joint y when assembly sits on edge top at theta=0
_XSTP_red = _cj_te2(0)[0]   # red stop: arm_a2 wheel at terminus
# Green intermediate stop: arm_b1 wheel at terminus (arm_b1 uses bar_len, not bar_len_a).
_arm_b1_dir_at_0 = -np.pi / 2 + wg.arm_b1
_XSTP2_TE = EDGE_X - wg.bar_len * np.cos(_arm_b1_dir_at_0) + wg.thigh
_XW_TE    = _cj_te(np.pi)[0]
_YW_TE    = _cj_te(np.pi)[1]
_XW_red   = _cj_te2(np.pi)[0]
_YW_red   = _cj_te2(np.pi)[1]

_CTE = "thin_edge_exact"

KEYFRAMES_THIN_EDGE = [
    # Phase 0 — both approach on edge top (theta=0), moving right.
    # Red (x2, assembly 2) leads; green (x1, assembly 1) trails.
    {"t": 0.00, "x1": -0.50, "y1": _Y0_TE, "theta1": 0.0,
                "x2": -0.108, "y2": _Y0_TE, "theta2": 0.0,
                "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    # Phase 1 — red stops (left wheel above terminus); green still approaching.
    {"t": 0.20, "x1":  0.125,    "y1": _Y0_TE, "theta1": 0.0,
                "x2": _XSTP_red, "y2": _Y0_TE, "theta2": 0.0,
                "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    # Phase 2 — red pivot (CW: theta2: 0 → −π), 12 steps × 15° × Δt=0.025.
    {"t": 0.225, "x1":  0.178,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2( np.pi/12)[0], "y2": _cj_te2( np.pi/12)[1],
                 "theta2": -np.pi / 12,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.250, "x1":  0.231,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(  np.pi/6)[0], "y2": _cj_te2(  np.pi/6)[1],
                 "theta2": -np.pi / 6,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.275, "x1":  0.25,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(  np.pi/4)[0], "y2": _cj_te2(  np.pi/4)[1],
                 "theta2": -np.pi / 4,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.300, "x1":  0.25,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(  np.pi/3)[0], "y2": _cj_te2(  np.pi/3)[1],
                 "theta2": -np.pi / 3,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.325, "x1": 0.275, "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(5*np.pi/12)[0], "y2": _cj_te2(5*np.pi/12)[1],
                 "theta2": -5 * np.pi / 12,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.350, "x1": 0.3, "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(  np.pi/2)[0], "y2": _cj_te2(  np.pi/2)[1],
                 "theta2": -np.pi / 2,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.375, "x1": 0.3, "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(7*np.pi/12)[0], "y2": _cj_te2(7*np.pi/12)[1],
                 "theta2": -7 * np.pi / 12,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.400, "x1": 0.3, "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(2*np.pi/3)[0], "y2": _cj_te2(2*np.pi/3)[1],
                 "theta2": -2 * np.pi / 3,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.425, "x1": 0.3, "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(3*np.pi/4)[0], "y2": _cj_te2(3*np.pi/4)[1],
                 "theta2": -3 * np.pi / 4,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.450, "x1": 0.33, "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(5*np.pi/6)[0], "y2": _cj_te2(5*np.pi/6)[1],
                 "theta2": -5 * np.pi / 6,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.475, "x1": _XSTP2_TE, "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(11*np.pi/12)[0], "y2": _cj_te2(11*np.pi/12)[1],
                 "theta2": -11 * np.pi / 12,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    # Red pivot done — cold_at=0.50 fires here.
    {"t": 0.500, "x1": _XSTP2_TE, "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _XW_red,    "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    # Phase 3 — stagger: green rolls to pivot start.
    # cold_at=0.61 fires between t=0.595 and t=0.620.
    {"t": 0.595, "x1": _XSTP_TE,  "y1": _Y0_TE, "theta1": 0.0,
                 "x2":  0.375,     "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    # Phase 4 — green pivot (CW: theta1: 0 → −π), 12 steps × 15° × Δt=0.025.
    # Red retreats from x2=0.375 → −0.156 over 11 steps.
    {"t": 0.620, "x1": _cj_te( np.pi/12)[0], "y1": _cj_te( np.pi/12)[1],
                 "theta1": -np.pi / 12,
                 "x2":  0.375,   "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.645, "x1": _cj_te(  np.pi/6)[0], "y1": _cj_te(  np.pi/6)[1],
                 "theta1": -np.pi / 6,
                 "x2":  0.327,   "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.670, "x1": _cj_te(  np.pi/4)[0], "y1": _cj_te(  np.pi/4)[1],
                 "theta1": -np.pi / 4,
                 "x2":  0.279,   "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.695, "x1": _cj_te(  np.pi/3)[0], "y1": _cj_te(  np.pi/3)[1],
                 "theta1": -np.pi / 3,
                 "x2":  0.231,   "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.720, "x1": _cj_te(5*np.pi/12)[0], "y1": _cj_te(5*np.pi/12)[1],
                 "theta1": -5 * np.pi / 12,
                 "x2":  0.183,   "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.745, "x1": _cj_te(  np.pi/2)[0], "y1": _cj_te(  np.pi/2)[1],
                 "theta1": -np.pi / 2,
                 "x2":  0.135,   "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.770, "x1": _cj_te(7*np.pi/12)[0], "y1": _cj_te(7*np.pi/12)[1],
                 "theta1": -7 * np.pi / 12,
                 "x2":  0.087,   "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.795, "x1": _cj_te(2*np.pi/3)[0], "y1": _cj_te(2*np.pi/3)[1],
                 "theta1": -2 * np.pi / 3,
                 "x2":  0.039,   "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.820, "x1": _cj_te(3*np.pi/4)[0], "y1": _cj_te(3*np.pi/4)[1],
                 "theta1": -3 * np.pi / 4,
                 "x2": -0.009,   "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.845, "x1": _cj_te(5*np.pi/6)[0], "y1": _cj_te(5*np.pi/6)[1],
                 "theta1": -5 * np.pi / 6,
                 "x2": -0.057,   "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.870, "x1": _cj_te(11*np.pi/12)[0], "y1": _cj_te(11*np.pi/12)[1],
                 "theta1": -11 * np.pi / 12,
                 "x2": -0.105,   "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    # Green pivot done; red retreat done — both below edge.
    {"t": 0.895, "x1": _XW_TE,  "y1": _YW_TE, "theta1": -np.pi,
                 "x2": -0.156,   "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    # Phase 5 — both below edge, moving left.
    {"t": 1.00, "x1":  0.040,   "y1": _YW_TE, "theta1": -np.pi,
                "x2": -0.440,   "y2": _YW_red, "theta2": -np.pi,
                "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
]

# ── Solver config and render metadata per scenario ───────────────────────────
SCENARIO_CONFIG = {
    "floor_to_wall": {
        "keyframes":     KEYFRAMES_FLOOR_TO_WALL,
        "n_grid":        120,
        "smooth_weight": 200.0,
        "cold_at":       None,
        "xlim":          (-0.2, 2.2),
        "ylim":          (-0.2, 1.5),
        "output":        "floor_to_wall.mp4",
        "label":         "Floor-to-wall",
    },
    "wall_to_ceiling": {
        "keyframes":     KEYFRAMES_WALL_TO_CEILING,
        "n_grid":        60,
        "smooth_weight": 60.0,
        "cold_at":       None,
        "xlim":          (-0.5, 1.75),
        "ylim":          (-0.5, 2.0),
        "output":        "wall_to_ceiling.mp4",
        "label":         "Wall-to-ceiling",
    },
    "outside": {
        "keyframes":     KEYFRAMES_OUTSIDE,
        "n_grid":        60,
        "smooth_weight": 200.0,
        "cold_at":       None,
        "xlim":          (-0.75, 1.5),
        "ylim":          (-0.75, 1.25),
        "output":        "outside.mp4",
        "label":         "Outside corner (ceiling-top -> exterior wall)",
    },
    "thin_edge": {
        "keyframes":     KEYFRAMES_THIN_EDGE,
        "n_grid":        60,
        "smooth_weight": 30.0,
        "cold_at":       [0.50, 0.61],
        "xlim":          (-0.7, 1.2),
        "ylim":          (-0.35, 1.0),
        "output":        "thin_edge.mp4",
        "label":         "Thin edge (top -> bottom)",
    },
}


if __name__ == "__main__":
    SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "wall_to_ceiling"

    if SCENARIO not in SCENARIO_CONFIG:
        print(f"Unknown scenario '{SCENARIO}'. Valid: {list(SCENARIO_CONFIG)}")
        sys.exit(1)

    cfg           = SCENARIO_CONFIG[SCENARIO]
    keyframes     = cfg["keyframes"]
    xlim          = cfg["xlim"]
    ylim          = cfg["ylim"]
    output        = cfg["output"]
    label         = cfg["label"]
    N_FRAMES      = 90
    FPS           = 15
    N_GRID        = cfg["n_grid"]
    SMOOTH_WEIGHT = cfg["smooth_weight"]

    print(f"{label}  ({N_FRAMES} frames @ {FPS} fps)")
    print("-" * 60)
    frames = solve_trajectory(keyframes, wg, l1, l2, l3, N_FRAMES, N_GRID,
                              smooth_weight=SMOOTH_WEIGHT, cold_at=cfg["cold_at"],
                              seed_at=cfg.get("seed_at"))
    print("-" * 60)
    # ---- joint-angle limit + jump + sign + ω check ------------------- #
    _JUMP_THRESH  = np.radians(10.0)   # per-frame limit (all 4 joints)
    _LIMIT        = np.radians(135.0)
    _OMEGA_MAX    = 0.70               # rad/frame
    _WHEEL_R      = wg.wheel_r
    _issues = []
    _names  = ["q1", "q2", "q3", "q4"]
    # Limit + sign checks (per frame)
    for _i, (_q, _, _p) in enumerate(frames):
        for _jidx in (0, 3):           # |q1|, |q4| >= 135°
            if abs(_q[_jidx]) >= _LIMIT:
                _issues.append(f"  LIMIT   fr{_i:02d}  {_names[_jidx]}={np.degrees(_q[_jidx]):+.1f}°")
        for _jidx in (1, 2):           # q2, q3 must be <= 0
            if float(_q[_jidx]) > 0.0:
                _issues.append(f"  SIGN    fr{_i:02d}  {_names[_jidx]}={np.degrees(float(_q[_jidx])):+.2f}°")
    # Jump check (frame-to-frame)
    for _i in range(1, len(frames)):
        _qp, _qc = frames[_i-1][0], frames[_i][0]
        for _j in range(4):
            _d = (_qc[_j] - _qp[_j] + np.pi) % (2*np.pi) - np.pi
            if abs(_d) >= _JUMP_THRESH:
                _issues.append(
                    f"  JUMP    fr{_i-1:02d}→{_i:02d}  {_names[_j]}"
                    f"  {np.degrees(_qp[_j]):+.1f}°→{np.degrees(_qc[_j]):+.1f}°"
                    f"  Δ={np.degrees(abs(_d)):.1f}°")
    # ω check (both-wheels-rolling phases: theta ~constant between frames)
    for _i in range(1, len(frames)):
        _pp, _pc = frames[_i-1][2], frames[_i][2]
        for (_xk, _yk, _tk) in [("x1","y1","theta1"), ("x2","y2","theta2")]:
            _dth = abs(float(_pc[_tk]) - float(_pp[_tk]))
            _dth = (_dth + np.pi) % (2*np.pi) - np.pi
            if abs(_dth) > 0.05:   # pivoting frame — skip ω check
                continue
            _dx = abs(float(_pc[_xk]) - float(_pp[_xk]))
            _dy = abs(float(_pc[_yk]) - float(_pp[_yk]))
            _om = max(_dx, _dy) / _WHEEL_R   # rad/frame
            if _om > _OMEGA_MAX:
                _issues.append(
                    f"  OMEGA   fr{_i-1:02d}→{_i:02d}  {_tk[:2]}  ω={_om:.3f} rad/fr")
    if not _issues:
        print("Check: CLEAN (no limits, signs, jumps≥10°, or ω violations)")
    else:
        for _msg in _issues:
            print(_msg)
        print(f"  → {len(_issues)} issue(s) total")
    # ------------------------------------------------------------------ #
    if "--no-render" not in sys.argv:
        print("Rendering...")
        render_video(frames, wg, l1, l2, l3, xlim, ylim, output, FPS)
