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
                     smooth_weight=1.0):
    """
    Solve IK at n_frames evenly-spaced parameter values in [0, 1].

    Warm-starts each frame from the previous solution and scores candidates
    with an additional smooth_weight * ||q - q_prev||² term so the solver
    prefers configurations close to the previous frame's solution.
    Returns a list of (q, info, params) tuples.
    """
    results = []
    q_prev  = None

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

        elif cs == "outside":
            # apply_outside_constraint in solve_ik handles surface placement
            # for all assembly orientations including the corner transition.
            # _outside_corner_adjust is NOT called here: at intermediate rotation
            # angles it places the assembly in the solid block (x<0, y>3), which
            # creates large, unavoidable crossing penalties.  The solver's built-in
            # apply_outside_constraint positions the assembly at the corner point
            # (wall_x, ceiling_y) for dead-zone angles — zero crossing penalty.
            pass

        try:
            # The outside penalty landscape is complex enough that the default
            # grid rarely finds seeds in the corner-wrapping region.  Use 6x
            # the normal grid density to match main.py's n_grid=360 baseline.
            ik_grid = n_grid * 6 if cs in ("outside", "thin_edge") else n_grid
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

    elif constraint_set == "outside":
        # Filled interior block so it's clear the robot wraps around the outside.
        block = Rectangle(
            (wall_x, ylim[0]), xlim[1] - wall_x, ceiling_y - ylim[0],
            facecolor="lightgray", alpha=0.45, zorder=0, label="Interior (solid)")
        ax.add_patch(block)
        ax.plot([wall_x, xlim[1]], [ceiling_y, ceiling_y],
                color="dimgray", lw=2, label=f"Ceiling top (y={ceiling_y})")
        ax.plot([wall_x, wall_x], [ylim[0], ceiling_y],
                color="slategray", lw=2, label=f"Wall exterior (x={wall_x})")

    elif constraint_set == "thin_edge":
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
    #  Both assemblies start ON TOP of the ceiling (theta=0, robot     #
    #  body above y=CEILING_Y) and end on the exterior face of the     #
    #  left wall (theta=pi/2, robot body left of x=WALL_X).           #
    #                                                                  #
    #  _outside_corner_adjust is only applied when the hint is within  #
    #  0.05 m of (WALL_X, CEILING_Y); away from the corner the        #
    #  standard apply_outside_constraint handles surface placement.    #
    #                                                                  #
    #  When the two assemblies are on opposite sides of the corner,    #
    #  _outside_surface_crossing_penalty in constraints.py drives the  #
    #  linkage through the corner point (WALL_X, CEILING_Y).          #
    # ---------------------------------------------------------------- #
    KEYFRAMES_OUTSIDE = [
        # Both on ceiling-top. Back (x2) leads — starts 2 m closer to corner.
        {"t": 0.00, "x1": 4.5, "y1": CEILING_Y, "theta1": 0.0,
                    "x2": 3.0, "y2": CEILING_Y, "theta2": 0.0,
                    "constraint_set": "outside", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        {"t": 0.20, "x1": 2.5, "y1": CEILING_Y, "theta1": 0.0,
                    "x2": 0.5, "y2": CEILING_Y, "theta2": 0.0,
                    "constraint_set": "outside", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Back slides to corner — theta2 stays 0.
        {"t": 0.28, "x1": 2.0, "y1": CEILING_Y, "theta1": 0.0,
                    "x2": 0.0, "y2": CEILING_Y, "theta2": 0.0,
                    "constraint_set": "outside", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Back rotation first half (0 -> pi/4): still in ceiling-top contact.
        # At theta2=pi/4 the ceiling-top mode ends; y2 is still at CEILING_Y.
        {"t": 0.33, "x1": 1.75, "y1": CEILING_Y, "theta1": 0.0,
                    "x2": 0.0,  "y2": CEILING_Y, "theta2": np.pi / 4,
                    "constraint_set": "outside", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Back rotation second half (pi/4 -> pi/2): wall mode, no ceiling support.
        # Gravity pulls the assembly 0.3 m down during this ~0.22 s window.
        {"t": 0.38, "x1": 1.5, "y1": CEILING_Y, "theta1": 0.0,
                    "x2": 0.0, "y2": CEILING_Y - 0.3, "theta2": np.pi / 2,
                    "constraint_set": "outside", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Back descends quickly through the wheel-blocking zone (y in [1.99, 2.91]);
        # front slides toward the corner to keep the chain taut.
        {"t": 0.55, "x1": 0.8, "y1": CEILING_Y, "theta1": 0.0,
                    "x2": 0.0, "y2": 1.5, "theta2": np.pi / 2,
                    "constraint_set": "outside", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Front slides to corner — theta1 stays 0.
        {"t": 0.65, "x1": 0.0, "y1": CEILING_Y, "theta1": 0.0,
                    "x2": 0.0, "y2": 0.8, "theta2": np.pi / 2,
                    "constraint_set": "outside", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Front rotation first half (0 -> pi/4): ceiling-top contact, y unchanged.
        {"t": 0.70, "x1": 0.0, "y1": CEILING_Y, "theta1": np.pi / 4,
                    "x2": 0.0, "y2": 0.65, "theta2": np.pi / 2,
                    "constraint_set": "outside", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Front rotation second half (pi/4 -> pi/2): wall mode, gravity pulls down 0.3 m.
        {"t": 0.75, "x1": 0.0, "y1": CEILING_Y - 0.3, "theta1": np.pi / 2,
                    "x2": 0.0, "y2": 0.5, "theta2": np.pi / 2,
                    "constraint_set": "outside", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
        # Both on exterior wall, descending.
        {"t": 1.00, "x1": 0.0, "y1": 2.3, "theta1": np.pi / 2,
                    "x2": 0.0, "y2": 0.3, "theta2": np.pi / 2,
                    "constraint_set": "outside", "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    ]

    # ---------------------------------------------------------------- #
    #  Scenario 4: top of thin edge -> below thin edge               #
    #                                                                  #
    #  Both assemblies start ON TOP of a thin horizontal edge          #
    #  (theta=0, wheels face down) and end BELOW it (theta=pi, wheels  #
    #  face up), wrapping around the right terminus at (EDGE_X, EDGE_Y)#
    #                                                                  #
    #  Rotation intermediate keyframe at theta=pi/2 forces _lerp_angle #
    #  through +pi/2 (rightward, around the terminus) rather than the  #
    #  opposite shortest-arc direction (-pi/2, wrong way).             #
    # ---------------------------------------------------------------- #
    KEYFRAMES_THIN_EDGE = [
        # Both on top; front (x1) leads toward terminus.
        {"t": 0.00, "x1": -0.5, "y1": EDGE_Y, "theta1": 0.0,
                    "x2": -2.5, "y2": EDGE_Y, "theta2": 0.0,
                    "constraint_set": "thin_edge", "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.22, "x1":  0.5, "y1": EDGE_Y, "theta1": 0.0,
                    "x2": -0.5, "y2": EDGE_Y, "theta2": 0.0,
                    "constraint_set": "thin_edge", "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Front slides to right terminus.
        {"t": 0.30, "x1":  EDGE_X, "y1": EDGE_Y, "theta1": 0.0,
                    "x2": -0.7, "y2": EDGE_Y, "theta2": 0.0,
                    "constraint_set": "thin_edge", "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Front rotates at terminus, first half (0 -> -pi/2, CW through left side).
        {"t": 0.38, "x1":  EDGE_X, "y1": EDGE_Y, "theta1": -np.pi / 2,
                    "x2": -0.5, "y2": EDGE_Y, "theta2": 0.0,
                    "constraint_set": "thin_edge", "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Front rotation complete (pi/2 -> pi); now below edge.
        {"t": 0.46, "x1":  EDGE_X, "y1": EDGE_Y, "theta1": np.pi,
                    "x2": -0.5, "y2": EDGE_Y, "theta2": 0.0,
                    "constraint_set": "thin_edge", "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Front slides left under edge; back approaches terminus.
        {"t": 0.60, "x1": -0.2, "y1": EDGE_Y, "theta1": np.pi,
                    "x2":  0.5, "y2": EDGE_Y, "theta2": 0.0,
                    "constraint_set": "thin_edge", "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Back slides to right terminus.
        {"t": 0.68, "x1": -0.5, "y1": EDGE_Y, "theta1": np.pi,
                    "x2":  EDGE_X, "y2": EDGE_Y, "theta2": 0.0,
                    "constraint_set": "thin_edge", "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Back rotates at terminus, first half (0 -> -pi/2, CW).
        {"t": 0.76, "x1": -0.8, "y1": EDGE_Y, "theta1": np.pi,
                    "x2":  EDGE_X, "y2": EDGE_Y, "theta2": -np.pi / 2,
                    "constraint_set": "thin_edge", "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Back rotation complete (pi/2 -> pi); now below edge.
        {"t": 0.84, "x1": -1.0, "y1": EDGE_Y, "theta1": np.pi,
                    "x2":  EDGE_X, "y2": EDGE_Y, "theta2": np.pi,
                    "constraint_set": "thin_edge", "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Both under edge, sliding left.
        {"t": 1.00, "x1": -1.5, "y1": EDGE_Y, "theta1": np.pi,
                    "x2": -0.5, "y2": EDGE_Y, "theta2": np.pi,
                    "constraint_set": "thin_edge", "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
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
        ylim = (0.5, 2.5)
        output = "thin_edge.mp4"
        label  = "Thin edge (top -> bottom)"

    N_FRAMES       = 90
    FPS            = 15
    N_GRID         = 60
    SMOOTH_WEIGHT  = 1.0   # weight on ||q - q_prev||² in candidate scoring

    print(f"{label}  ({N_FRAMES} frames @ {FPS} fps)")
    print("-" * 60)
    frames = solve_trajectory(keyframes, wg, l1, l2, l3, N_FRAMES, N_GRID,
                              smooth_weight=SMOOTH_WEIGHT)
    print("-" * 60)
    print("Rendering...")
    render_video(frames, wg, l1, l2, l3, xlim, ylim, output, FPS)
