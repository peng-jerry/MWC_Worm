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

Edit SCENARIO at the bottom to choose which scenario to render.
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

def _wheel_surface_adjust(x, y, theta, wg,
                           wall_x=None, floor_y=None, ceiling_y=None):
    """
    Pre-correct (x, y) so every wheel clears each active INTERIOR surface.

    floor_y   — raise the assembly if any wheel dips below the floor
    ceiling_y — lower the assembly if any wheel pokes above the ceiling
    wall_x    — shift right if any wheel enters the (interior) wall

    Handles the dead-zone near +-45 deg where the binary |sin_c|/|cos_c|
    check in constraints.py doesn't fire for either surface.
    """
    c = theta - np.pi / 2
    dirs = [c - wg.spread, c + wg.spread]
    wheel_xs = [x + wg.bar_len * np.cos(d) for d in dirs]
    wheel_ys = [y + wg.bar_len * np.sin(d) for d in dirs]

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


def _outside_corner_adjust(x, y, theta, wg, wall_x=0.0, ceiling_y=3.0):
    """
    Dead-zone correction for the EXTERIOR top-left corner.

    Raises the assembly if any wheel penetrates the ceiling TOP surface
    (wheels must stay at y >= ceiling_y + wheel_r), then shifts it LEFT
    if any wheel penetrates the exterior wall face (max wheel x must be
    <= wall_x - wheel_r).

    Only call when the hint is within ~0.05 m of (wall_x, ceiling_y).
    Calling it for ceiling-top sliding frames (large x) would incorrectly
    pull those assemblies leftward to the exterior wall position.
    """
    c = theta - np.pi / 2
    dirs = [c - wg.spread, c + wg.spread]
    wheel_xs = [x + wg.bar_len * np.cos(d) for d in dirs]
    wheel_ys = [y + wg.bar_len * np.sin(d) for d in dirs]

    min_wy = min(wheel_ys)
    if min_wy < ceiling_y + wg.wheel_r:
        delta = (ceiling_y + wg.wheel_r) - min_wy
        y += delta
        wheel_ys = [wy + delta for wy in wheel_ys]

    max_wx = max(wheel_xs)
    if max_wx > wall_x - wg.wheel_r:
        x -= max_wx - (wall_x - wg.wheel_r)

    return x, y


# ------------------------------------------------------------------ #
#  IK trajectory solver                                                #
# ------------------------------------------------------------------ #

def solve_trajectory(keyframes, wg, l1, l2, l3, n_frames, n_grid=60,
                     smooth_weight=1.0, cold_at=None):
    """
    Solve IK at n_frames evenly-spaced parameter values in [0, 1].

    Warm-starts each frame from the previous solution and scores candidates
    with an additional smooth_weight * ||q - q_prev||² term so the solver
    prefers configurations close to the previous frame's solution.
    Returns a list of (q, info, params) tuples.

    Parameters
    ----------
    cold_at : float or None
        If provided, resets the warm-start (q_prev=None) the first time the
        trajectory parameter s reaches or exceeds this value.  Use this to
        force a branch switch at a known keyframe boundary (e.g. 0.43 for the
        outside corner when the back assembly completes its rotation to the wall).
    """
    results = []
    q_prev  = None
    cold_fired = False

    for idx, s in enumerate(np.linspace(0, 1, n_frames)):
        p = eval_keyframes(keyframes, s)
        cs = p["constraint_set"]

        # Pre-correct hints before passing to solve_ik to handle the
        # dead-zone near +-45 deg where the binary surface selector in
        # constraints.py doesn't fire for either surface.
        if cs == "wall":
            # Inside floor-wall corner.
            p["x1"], p["y1"] = _wheel_surface_adjust(
                p["x1"], p["y1"], p["theta1"], wg,
                wall_x=p["wall_x"], floor_y=0.0)
            p["x2"], p["y2"] = _wheel_surface_adjust(
                p["x2"], p["y2"], p["theta2"], wg,
                wall_x=p["wall_x"], floor_y=0.0)

        elif cs == "ceiling":
            # Inside wall-ceiling corner.  Include floor_y=0 to prevent
            # the back assembly from dipping below y=0 at the start.
            p["x1"], p["y1"] = _wheel_surface_adjust(
                p["x1"], p["y1"], p["theta1"], wg,
                wall_x=p["wall_x"], floor_y=0.0, ceiling_y=p["ceiling_y"])
            p["x2"], p["y2"] = _wheel_surface_adjust(
                p["x2"], p["y2"], p["theta2"], wg,
                wall_x=p["wall_x"], floor_y=0.0, ceiling_y=p["ceiling_y"])

        elif cs in ("outside", "outside_exact"):
            # "outside": apply_outside_constraint in solve_ik handles surface
            # placement for ceiling-top and wall-exterior orientations.
            # "outside_exact": exact pivot positions provided in keyframes;
            # no pre-adjustment needed (solve_ik skips contact adjustment).
            pass

        # Forced cold-start: drop warm-start exactly once when s crosses cold_at.
        # This allows a deliberate branch switch at a known trajectory boundary
        # without affecting any other frame.
        if cold_at is not None and not cold_fired and s >= cold_at:
            q_prev = None
            cold_fired = True

        try:
            # The outside penalty landscape is complex enough that the default
            # grid rarely finds seeds in the corner-wrapping region.  Use 6x
            # the normal grid density to match main.py's n_grid=360 baseline.
            ik_grid = n_grid * 6 if cs in ("outside", "outside_exact", "thin_edge", "thin_edge_exact") else n_grid
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

    elif constraint_set == "ceiling":
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
            wheel_radius=wg.wheel_r,
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


# ------------------------------------------------------------------ #
#  Scenarios                                                           #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    l1, l2, l3 = 1.2, 1.0, 1.2
    wg = WheelGeometry(bar_len=0.25, wheel_r=0.12, spread=np.pi / 4)

    WALL_X    = 0.0
    CEILING_Y = 3.0

    EDGE_X = 1.0    # right terminus x for "thin_edge"
    EDGE_Y = 1.5    # edge height for "thin_edge"

    # ---- select scenario ------------------------------------------ #
    import sys as _sys
    SCENARIO = _sys.argv[1] if len(_sys.argv) > 1 else "outside"
    # --------------------------------------------------------------- #

    # ---------------------------------------------------------------- #
    #  Scenario 1: floor -> wall  (inside bottom-left corner)          #
    #                                                                  #
    #  Both assemblies start on the floor (theta=0) and end on the     #
    #  left wall (theta=-pi/2).  Front slides to the corner, rotates   #
    #  90 deg, climbs the wall; then the back repeats the same steps.  #
    # ---------------------------------------------------------------- #
    KEYFRAMES_FLOOR_TO_WALL = [
        {"t": 0.00, "x1": 2.5, "y1": 0.0, "theta1": 0.0,
                    "x2": 4.5, "y2": 0.0, "theta2": 0.0,
                    "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.25, "x1": 0.5, "y1": 0.0, "theta1": 0.0,
                    "x2": 2.5, "y2": 0.0, "theta2": 0.0,
                    "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Front slides to corner — theta1 stays 0.
        {"t": 0.33, "x1": 0.0, "y1": 0.0, "theta1": 0.0,
                    "x2": 2.3, "y2": 0.0, "theta2": 0.0,
                    "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Front rotates at corner (0 -> -pi/2). y1=0 so correction fires at every angle.
        {"t": 0.45, "x1": 0.0, "y1": 0.0, "theta1": -np.pi / 2,
                    "x2": 2.0, "y2": 0.0, "theta2": 0.0,
                    "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.65, "x1": 0.0, "y1": 2.0, "theta1": -np.pi / 2,
                    "x2": 1.0, "y2": 0.0, "theta2": 0.0,
                    "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Back slides to corner — theta2 stays 0.
        {"t": 0.74, "x1": 0.0, "y1": 2.4, "theta1": -np.pi / 2,
                    "x2": 0.0, "y2": 0.0, "theta2": 0.0,
                    "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Back rotates at corner (0 -> -pi/2).
        {"t": 0.84, "x1": 0.0, "y1": 2.8, "theta1": -np.pi / 2,
                    "x2": 0.0, "y2": 0.0, "theta2": -np.pi / 2,
                    "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 1.00, "x1": 0.0, "y1": 3.5, "theta1": -np.pi / 2,
                    "x2": 0.0, "y2": 1.5, "theta2": -np.pi / 2,
                    "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    ]

    # ---------------------------------------------------------------- #
    #  Scenario 2: wall -> ceiling  (inside top-left corner)           #
    #                                                                  #
    #  Both assemblies start on the left wall (theta=-pi/2) and end    #
    #  on the ceiling (theta=pi, shortest arc through -pi).  Mirrors   #
    #  scenario 1 with x/y roles swapped.                              #
    # ---------------------------------------------------------------- #
    KEYFRAMES_WALL_TO_CEILING = [
        {"t": 0.00, "x1": 0.0, "y1": 1.5, "theta1": -np.pi / 2,
                    "x2": 0.0, "y2": 0.0, "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.25, "x1": 0.0, "y1": 2.5, "theta1": -np.pi / 2,
                    "x2": 0.0, "y2": 1.0, "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Front slides to corner — theta1 stays -pi/2.
        {"t": 0.33, "x1": 0.0, "y1": CEILING_Y, "theta1": -np.pi / 2,
                    "x2": 0.0, "y2": 1.5, "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Front rotates at corner (-pi/2 -> pi via -pi). y1=CEILING_Y so correction fires.
        {"t": 0.45, "x1": 0.0, "y1": CEILING_Y, "theta1": np.pi,
                    "x2": 0.0, "y2": 1.5, "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.65, "x1": 2.0, "y1": CEILING_Y, "theta1": np.pi,
                    "x2": 0.0, "y2": 2.8, "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Back slides to corner — theta2 stays -pi/2.
        {"t": 0.74, "x1": 2.4, "y1": CEILING_Y, "theta1": np.pi,
                    "x2": 0.0, "y2": CEILING_Y, "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Back rotates at corner (-pi/2 -> pi).
        {"t": 0.84, "x1": 2.8, "y1": CEILING_Y, "theta1": np.pi,
                    "x2": 0.0, "y2": CEILING_Y, "theta2": np.pi,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 1.00, "x1": 3.5, "y1": CEILING_Y, "theta1": np.pi,
                    "x2": 1.5, "y2": CEILING_Y, "theta2": np.pi,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    ]

    # ---------------------------------------------------------------- #
    #  Scenario 3: ceiling-top -> exterior wall  (outside top-left)    #
    #                                                                  #
    #  Each assembly goes through two pivot phases at the corner:      #
    #    Phase A — back-wheel pivot: rotate about the back (right)     #
    #              wheel, which stays fixed on the ceiling top.        #
    #    Phase B — front-wheel pivot: once the front (left) wheel      #
    #              touches the exterior wall face, it becomes the new  #
    #              pivot until the back wheel is directly above it.    #
    #                                                                  #
    #  constraint_set="outside_exact" is used during phase B (and the  #
    #  brief post-rotation frame where y > CEILING_Y) so that the      #
    #  exact pivot-computed positions are used without being overridden #
    #  by the ceiling-contact formula.  Outside penalties still apply. #
    # ---------------------------------------------------------------- #

    # --- Corner geometry constants ---
    _R    = wg.wheel_r    # 0.12
    _BAR  = wg.bar_len    # 0.25
    # Corner point (top-left exterior corner).
    _CX   = WALL_X        # 0.0
    _CY   = CEILING_Y     # 3.0
    # Assembly stops when back-wheel CENTER is directly above the corner:
    #   back-wheel center = (WALL_X, CEILING_Y + R) = (0, 3.12)
    #   assembly center   = (WALL_X - BAR·cos(-π/4), CEILING_Y + R - BAR·sin(-π/4))
    _XSTP = _CX - _BAR * np.cos(-np.pi / 4)            # -0.177
    _Y0   = _CY + _R - _BAR * np.sin(-np.pi / 4)       # 3.297

    def _cp(phi):
        """
        Corner pivot: assembly center and orientation when the back wheel has
        rolled φ radians (0 → π/2) around the corner.

        The back-wheel center traces a quarter-circle of radius R centered at
        the corner (CX, CY).  The whole assembly rotates about the corner as a
        rigid body, so theta = phi.  At phi=0 the assembly is flat on the
        ceiling; at phi=π/2 both wheels are on the exterior wall face.
        """
        bwc_x = _CX - _R * np.sin(phi)
        bwc_y = _CY + _R * np.cos(phi)
        x = bwc_x - _BAR * np.cos(phi - np.pi / 4)
        y = bwc_y - _BAR * np.sin(phi - np.pi / 4)
        return x, y   # theta = phi

    # Assembly center at end of pivot (phi = π/2):
    #   back wheel at (-R, CY) = (-0.12, 3.0) — corner of ceiling and wall
    #   front wheel at (-R, CY - BAR√2·sin(π/4)) = (-0.12, 2.646) — on wall
    _XW   = _cp(np.pi / 2)[0]   # -0.297
    _YW   = _cp(np.pi / 2)[1]   # 2.823

    _CS   = "outside"        # normal: ceiling/wall contact auto-set
    _CSE  = "outside_exact"  # exact positions, outside penalties still on

    KEYFRAMES_OUTSIDE = [
        # Phase 0 — both approach on ceiling (theta=0), moving left.
        {"t": 0.00, "x1":  2.0,         "y1": _Y0, "theta1": 0.0,
                    "x2":  3.5,         "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},

        # Phase 1 — front assembly stops when back-wheel CENTER is directly
        # above the corner (x_bwc = WALL_X = 0, one wheel-radius further left
        # than the old tangent-at-wall stop).  Switch to outside_exact: at
        # x<0, y>ceiling_y the contact formula applies a spurious x-clamp.
        {"t": 0.24, "x1": _XSTP,        "y1": _Y0, "theta1": 0.0,
                    "x2": _XSTP + 1.5,  "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},

        # Phase 2 — front corner pivot (phi: 0 → π/2).
        # The back wheel rolls around the corner on an arc of radius R; the
        # whole assembly rotates about the corner point (CX, CY).  Use π/8
        # steps so no frame exceeds ~10° of joint-angle change.
        {"t": 0.29, "x1": _cp(np.pi / 8)[0], "y1": _cp(np.pi / 8)[1],
                    "theta1": np.pi / 8,
                    "x2": _XSTP + 1.5,  "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.34, "x1": _cp(np.pi / 4)[0], "y1": _cp(np.pi / 4)[1],
                    "theta1": np.pi / 4,
                    "x2": _XSTP + 1.5,  "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.39, "x1": _cp(3 * np.pi / 8)[0], "y1": _cp(3 * np.pi / 8)[1],
                    "theta1": 3 * np.pi / 8,
                    "x2": _XSTP + 1.5,  "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Phase 2 done — front at theta=π/2; back wheel at corner (-R, CY),
        # front wheel on wall at (-R, 2.646).
        {"t": 0.44, "x1": _XW,           "y1": _YW, "theta1": np.pi / 2,
                    "x2": _XSTP + 1.5,  "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},

        # Phase 3 — joint forward drive: front descends wall while back rolls
        # toward the corner stop.  Use fine steps as x2 crosses zero — the chain
        # routing transitions there and coarse steps cause IK branch switches.
        {"t": 0.48, "x1": _XW, "y1": 2.7, "theta1": np.pi / 2,
                    "x2": 1.0,            "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CS, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.52, "x1": _XW, "y1": 2.5, "theta1": np.pi / 2,
                    "x2": 0.6,            "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CS, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.56, "x1": _XW, "y1": 2.35,"theta1": np.pi / 2,
                    "x2": 0.2,            "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CS, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # x2 approaching zero — switch to exact, then step past zero slowly.
        {"t": 0.59, "x1": _XW, "y1": 2.25,"theta1": np.pi / 2,
                    "x2": 0.05,           "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.61, "x1": _XW, "y1": 2.2, "theta1": np.pi / 2,
                    "x2": -0.05,          "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.63, "x1": _XW, "y1": 2.15,"theta1": np.pi / 2,
                    "x2": -0.12,          "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Back arrives at stop; begin back pivot almost immediately so the IK
        # has two moving assemblies as the front passes through the near-degenerate
        # zone (y1≈2.0 where P1 crowds P3 and q2–q4 become hyper-sensitive).
        {"t": 0.66, "x1": _XW, "y1": 2.05,"theta1": np.pi / 2,
                    "x2": _XSTP,          "y2": _Y0, "theta2": 0.0,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},

        # Phase 4 — back corner pivot (phi: 0 → π/2), identical geometry to phase 2.
        # Overlap with front descent so the IK landscape evolves smoothly.
        {"t": 0.70, "x1": _XW, "y1": 1.8, "theta1": np.pi / 2,
                    "x2": _cp(np.pi / 8)[0], "y2": _cp(np.pi / 8)[1],
                    "theta2": np.pi / 8,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.76, "x1": _XW, "y1": 1.4, "theta1": np.pi / 2,
                    "x2": _cp(np.pi / 4)[0], "y2": _cp(np.pi / 4)[1],
                    "theta2": np.pi / 4,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.82, "x1": _XW, "y1": 1.0, "theta1": np.pi / 2,
                    "x2": _cp(3 * np.pi / 8)[0], "y2": _cp(3 * np.pi / 8)[1],
                    "theta2": 3 * np.pi / 8,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Phase 4 done — back at theta=π/2.
        {"t": 0.88, "x1": _XW, "y1": 0.8, "theta1": np.pi / 2,
                    "x2": _XW,              "y2": _YW, "theta2": np.pi / 2,
                    "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},

        # Phase 5 — both on exterior wall, descending.
        {"t": 0.96, "x1": _XW, "y1": 0.5, "theta1": np.pi / 2,
                    "x2": _XW,  "y2": 2.0, "theta2": np.pi / 2,
                    "constraint_set": _CS, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 1.00, "x1": _XW, "y1": 0.0, "theta1": np.pi / 2,
                    "x2": _XW,  "y2": 1.5, "theta2": np.pi / 2,
                    "constraint_set": _CS, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    ]

    # ---------------------------------------------------------------- #
    #  Scenario 4: top of thin edge -> below thin edge               #
    #                                                                  #
    #  Both assemblies start ON TOP of a thin horizontal edge and end  #
    #  BELOW it, wrapping clockwise (theta: 0 → −π) around the right  #
    #  terminus at (EDGE_X, EDGE_Y).                                   #
    #                                                                  #
    #  At theta=0 the assembly moves rightward; the RIGHT wheel leads. #
    #  Stop: BACK (left/trailing) wheel center directly above terminus.#
    #  That wheel center then traces a semicircle of radius wheel_r    #
    #  about the terminus (CW, 180°), with the assembly rotating as a  #
    #  rigid body about the terminus.  At completion (theta=−π) the    #
    #  front (right) wheel arrives at the underside exactly as the     #
    #  back wheel finishes — same timing rule as the outside scenario. #
    #                                                                  #
    #  constraint_set="thin_edge_exact" is used during pivot phases.   #
    # ---------------------------------------------------------------- #

    # --- Thin-edge geometry constants ---
    _R_TE   = wg.wheel_r    # 0.12
    _BAR_TE = wg.bar_len    # 0.25

    # Stop: back (left at theta=0) wheel center directly above terminus
    _XSTP_TE = EDGE_X + _BAR_TE / np.sqrt(2.0)            # 1.177
    _Y0_TE   = EDGE_Y + _R_TE + _BAR_TE / np.sqrt(2.0)    # 1.797

    def _cp_te(phi):
        """Assembly center during CW semicircle pivot about terminus (phi: 0→π). theta = −phi."""
        dx0 = +_BAR_TE / np.sqrt(2.0)          # +0.177  (back-wheel x offset from terminus)
        dy0 = _R_TE + _BAR_TE / np.sqrt(2.0)   # +0.297  (back-wheel y offset from terminus)
        return (EDGE_X + dx0 * np.cos(phi) + dy0 * np.sin(phi),
                EDGE_Y - dx0 * np.sin(phi) + dy0 * np.cos(phi))

    # _cp_te key values (phi → (x, y), theta):
    #   0     → (1.177, 1.797),  0
    #   π/6   → (1.302, 1.669), −π/6   ← back wheel crosses EDGE_Y (x≈1.30 ≥ EDGE_X ✓)
    #   π/3   → (1.346, 1.496), −π/3   ← assembly just below EDGE_Y; chain clear
    #   π/2   → (1.297, 1.323), −π/2
    #   2π/3  → (1.169, 1.198), −2π/3
    #   5π/6  → (0.996, 1.154), −5π/6
    #   π     → (0.823, 1.203), −π    ← both wheels at y=EDGE_Y−R (touching underside)
    _XW_TE = _cp_te(np.pi)[0]   # 0.823
    _YW_TE = _cp_te(np.pi)[1]   # 1.203

    _CTE = "thin_edge_exact"

    # _XSTP2_TE: stop for back assembly — front (right) wheel tangent to top of terminus
    _XSTP2_TE = EDGE_X - _BAR_TE / np.sqrt(2.0)   # 0.823

    KEYFRAMES_THIN_EDGE = [
        # Phase 0 — both approach on edge top (theta=0), moving right.
        # Maintain ~1.5 m separation; back rolls at constant speed and reaches its
        # tangent stop (x=0.823) only after the front has rotated 120° (phi=2π/3).
        {"t": 0.00, "x1": -0.5, "y1": _Y0_TE, "theta1": 0.0,
                    "x2": -2.0, "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Phase 1 — front stops (back/left wheel above terminus); back still 1.5 m behind.
        {"t": 0.25, "x1": _XSTP_TE, "y1": _Y0_TE, "theta1": 0.0,
                    "x2": -0.323,   "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},

        # Phase 2 — front semicircle pivot (CW: theta1: 0 → −π), 6 steps of 30°.
        # Back rolls at a uniform speed throughout, arriving at _XSTP2_TE=0.823
        # (front/right wheel tangent to terminus top) only at phi=2π/3 (t=0.41).
        # Delaying arrival until phi=2π/3 prevents wheel overlap: at phi=π/3 and π/2
        # the front assembly's trailing wheel swings near the back assembly's leading
        # wheel — keeping the back at x≤0.55 through those steps gives ≥0.24 m clearance.
        # At phi≥2π/3 the straight-line chain crossing falls left of EDGE_X, so the
        # IK routes the chain in a right-loop over the terminus — forced by the
        # crossing penalty (weight 1e4) which dwarfs the smooth term.
        {"t": 0.29, "x1": _cp_te(  np.pi/6)[0], "y1": _cp_te(  np.pi/6)[1],
                    "theta1": -np.pi / 6,
                    "x2":  0.00,     "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.33, "x1": _cp_te(  np.pi/3)[0], "y1": _cp_te(  np.pi/3)[1],
                    "theta1": -np.pi / 3,
                    "x2":  0.27,     "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.37, "x1": _cp_te(  np.pi/2)[0], "y1": _cp_te(  np.pi/2)[1],
                    "theta1": -np.pi / 2,
                    "x2":  0.55,     "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.41, "x1": _cp_te(2*np.pi/3)[0], "y1": _cp_te(2*np.pi/3)[1],
                    "theta1": -2 * np.pi / 3,
                    "x2": _XSTP2_TE, "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.44, "x1": _cp_te(5*np.pi/6)[0], "y1": _cp_te(5*np.pi/6)[1],
                    "theta1": -5 * np.pi / 6,
                    "x2": _XSTP2_TE, "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Front pivot done — back holds at _XSTP2_TE while front drives away.
        {"t": 0.47, "x1": _XW_TE,   "y1": _YW_TE, "theta1": -np.pi,
                    "x2": _XSTP2_TE, "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},

        # Front drives left (away from terminus) while back holds.
        # Sequential: front clears first, then back rolls off.
        {"t": 0.54, "x1": 0.5,       "y1": _YW_TE, "theta1": -np.pi,
                    "x2": _XSTP2_TE, "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Roll-off — back front-wheel rolls past terminus to pivot stop; front holds.
        {"t": 0.58, "x1": 0.5,       "y1": _YW_TE, "theta1": -np.pi,
                    "x2": _XSTP_TE,  "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Delay — back holds at pivot start.
        {"t": 0.62, "x1": 0.5,       "y1": _YW_TE, "theta1": -np.pi,
                    "x2": _XSTP_TE,  "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},

        # Phase 3 — back semicircle pivot (CW: theta2: 0 → −π), 6 steps of 30°.
        # Front retreats at ~5 m/unit throughout (slower than approach speed of 6.7 m/unit).
        {"t": 0.66, "x1":  0.30, "y1": _YW_TE, "theta1": -np.pi,
                    "x2": _cp_te(  np.pi/6)[0], "y2": _cp_te(  np.pi/6)[1],
                    "theta2": -np.pi / 6,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.70, "x1":  0.10, "y1": _YW_TE, "theta1": -np.pi,
                    "x2": _cp_te(  np.pi/3)[0], "y2": _cp_te(  np.pi/3)[1],
                    "theta2": -np.pi / 3,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.74, "x1": -0.10, "y1": _YW_TE, "theta1": -np.pi,
                    "x2": _cp_te(  np.pi/2)[0], "y2": _cp_te(  np.pi/2)[1],
                    "theta2": -np.pi / 2,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.78, "x1": -0.30, "y1": _YW_TE, "theta1": -np.pi,
                    "x2": _cp_te(2*np.pi/3)[0], "y2": _cp_te(2*np.pi/3)[1],
                    "theta2": -2 * np.pi / 3,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.82, "x1": -0.50, "y1": _YW_TE, "theta1": -np.pi,
                    "x2": _cp_te(5*np.pi/6)[0], "y2": _cp_te(5*np.pi/6)[1],
                    "theta2": -5 * np.pi / 6,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Back pivot done — both at theta=−π below edge.
        {"t": 0.86, "x1": -0.70, "y1": _YW_TE, "theta1": -np.pi,
                    "x2": _XW_TE,  "y2": _YW_TE, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},

        # Phase 4 — both below edge, moving left.
        {"t": 0.93, "x1": -1.05, "y1": _YW_TE, "theta1": -np.pi,
                    "x2":  0.40,  "y2": _YW_TE, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 1.00, "x1": -1.45, "y1": _YW_TE, "theta1": -np.pi,
                    "x2": -0.20,  "y2": _YW_TE, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    ]

    if SCENARIO == "floor_to_wall":
        keyframes = KEYFRAMES_FLOOR_TO_WALL
        xlim = (-0.5, 6.5)
        ylim = (-0.5, 5.5)
        output = "floor_to_wall.mp4"
        label  = "Floor-to-wall"
    elif SCENARIO == "wall_to_ceiling":
        keyframes = KEYFRAMES_WALL_TO_CEILING
        xlim = (-0.5, 4.5)
        ylim = (-0.5, 3.5)
        output = "wall_to_ceiling.mp4"
        label  = "Wall-to-ceiling"
    elif SCENARIO == "outside":
        keyframes = KEYFRAMES_OUTSIDE
        xlim = (-1.5, 5.5)
        ylim = (-0.5, 4.5)
        output = "outside.mp4"
        label  = "Outside corner (ceiling-top -> exterior wall)"
    else:  # "thin_edge"
        keyframes = KEYFRAMES_THIN_EDGE
        xlim = (-3.0, 2.5)
        ylim = (-0.5, 3.5)
        output = "thin_edge.mp4"
        label  = "Thin edge (top -> bottom)"

    N_FRAMES       = 90
    FPS            = 15
    N_GRID         = 60    # solve_trajectory already applies ×6 for outside/thin_edge
    # Higher smooth weight for outside corner: prevents IK branch switches from
    # outcompeting the continuity cost in the complex exterior-corner penalty landscape.
    SMOOTH_WEIGHT  = 30.0 if SCENARIO == "thin_edge" else (20.0 if SCENARIO == "outside" else 10.0)

    print(f"{label}  ({N_FRAMES} frames @ {FPS} fps)")
    print("-" * 60)
    frames = solve_trajectory(keyframes, wg, l1, l2, l3, N_FRAMES, N_GRID,
                              smooth_weight=SMOOTH_WEIGHT, cold_at=None)
    print("-" * 60)
    print("Rendering...")
    render_video(frames, wg, l1, l2, l3, xlim, ylim, output, FPS)
