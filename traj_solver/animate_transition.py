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
    dirs = [c - wg.spread, c + wg.spread]
    wheel_xs = [vx + wg.bar_len * np.cos(d) for d in dirs]
    wheel_ys = [vy + wg.bar_len * np.sin(d) for d in dirs]

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
            calf=wg.calf,
            thigh=wg.thigh,
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
    l1, l2, l3 = 0.25, 0.50, 0.25
    wg = WheelGeometry(bar_len=0.200, wheel_r=0.050, spread=np.pi / 4,
                       calf=0.042, thigh=0.08951)

    WALL_X    = 0.0
    CEILING_Y = 0.55

    EDGE_X = 0.44   # right terminus x for "thin_edge"  (0.35 × 1.25)
    EDGE_Y = 0.50   # edge height for "thin_edge"      (0.40 × 1.25)

    # ---- select scenario ------------------------------------------ #
    import sys as _sys
    SCENARIO = _sys.argv[1] if len(_sys.argv) > 1 else "wall_to_ceiling"
    # --------------------------------------------------------------- #

    # ---------------------------------------------------------------- #
    #  Scenario 1: floor -> wall  (inside bottom-left corner)          #
    #                                                                  #
    #  Both assemblies start on the floor (theta=0) and end on the     #
    #  left wall (theta=-pi/2).  Front slides to the corner, rotates   #
    #  90 deg, climbs the wall; then the back repeats the same steps.  #
    #                                                                  #
    #  Chain l1+l2+l3=0.80 m.  Start/end in a near-flat (≥94%)        #
    #  configuration.  Minimum assembly separation ≥0.55 m throughout  #
    #  so q1 and q4 stay well below 135°.                              #
    #                                                                  #
    #  Contact geometry (theta=0 floor, theta=-pi/2 wall):             #
    #    floor_y = 0.191  (joint y on floor, side=±1)                  #
    #    wall_x  = 0.191  (joint x on wall, side=±1)                   #
    #    back (side=-1) min y on wall ≈ 0.24 m (floor clears wheels)   #
    #  _FTW_CY = 2.0: tall room, ceiling never interferes.             #
    # ---------------------------------------------------------------- #
    _FTW_CY = 2.0

    # Explicit contact positions (bar_len=0.200, wheel_r=0.050, spread=pi/4,
    # calf=0.042, thigh=0.08951, side=+1):
    #   floor_y(0°)=-15°/-30°/-45° = 0.233/0.241/0.235/0.216
    #   wall_x(-45°/-60°/-75°/-90°) = 0.343/0.324/0.287/0.233
    # "wall_exact" skips the binary floor/wall snap so the corner transit
    # (x1: 0→0.343, y1: 0.216→0 at theta=-45°) is smooth across ~2 frames.
    KEYFRAMES_FLOOR_TO_WALL = [
        # ---- Approach at 3.000/unit (ω=0.674 rad/frame ✓), sep=0.990m constant ----
        {"t": 0.000, "x1": 0.4, "y1": 0.0, "theta1": 0.0,
                     "x2": 1.0, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        # Front reaches corner; back holds at x2=0.990.
        # Front left wheel just touches wall at theta=0 when x1=wall_x_for_assembly(0)=0.281.
        # Set x1=0.281 explicitly so WSA never snaps x1 during approach — the snap would
        # hold x1 fixed while x2 keeps moving, creating a sudden extension drop that breaks
        # the warm chain.  With x1=0.281 the approach sep decreases smoothly 99%→71%
        # over ~14 frames (≈2°/frame in q1), well within warm-chain tracking range.
        {"t": 0.15, "x1": 0.0, "y1": 0.0, "theta1": 0.0,
                     "x2": 1.00, "y2": 0.233, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        # ---- Front rotation: floor phase (theta 0→-45°, back holds at x2=0.990) ----
        {"t": 0.175, "x1": 0.000, "y1": 0.241, "theta1": -np.pi / 12,
                     "x2": 0.980, "y2": 0.233, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.20, "x1": 0.000, "y1": 0.235, "theta1": -np.pi / 6,
                     "x2": 0.980, "y2": 0.233, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.225, "x1": 0.000, "y1": 0.216, "theta1": -np.pi / 4,
                     "x2": 0.960, "y2": 0.233, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        # ---- Front rotation: wall phase (theta -45°→-90°, wall snap handles x1) ----
        # No separate transit keyframe: theta always changes → OMEGA check skipped.
        {"t": 0.245, "x1": 0.324, "y1": 0.000, "theta1": -np.pi / 3,
                     "x2": 0.960, "y2": 0.233, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.275, "x1": 0.287, "y1": 0.000, "theta1": -5 * np.pi / 12,
                     "x2": 0.958, "y2": 0.233, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        # theta=-90°: front locks on wall; y2=0.233 matches floor-snap value → no jump.
        {"t": 0.305, "x1": 0.000, "y1": 0.000, "theta1": -np.pi / 2,
                     "x2": 0.958, "y2": 0.233, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        # ---- Both moving at 2.870/unit (ω=0.645 rad/frame ✓) ----
        # Front climbs wall (theta=-pi/2); back approaches corner on floor (theta=0).
        {"t": 0.44, "x1": 0.0, "y1": 0.42, "theta1": -np.pi / 2,
                     "x2": 0.79, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.51, "x1": 0.0, "y1": 0.62, "theta1": -np.pi / 2,
                     "x2": 0.63, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.58, "x1": 0.0, "y1": 0.82, "theta1": -np.pi / 2,
                     "x2": 0.48, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        # Back reaches corner; front at y1=0.990.
        {"t": 0.735, "x1": 0.0, "y1": 1.0, "theta1": -np.pi / 2,
                     "x2": 0.000, "y2": 0.0, "theta2": 0.0,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        # ---- Back rotates 0→-pi/2 in 6×Δt=0.025; front climbs to y1=1.125 ----
        {"t": 0.76, "x1": 0.0, "y1": 1.1, "theta1": -np.pi / 2,
                     "x2": 0.0,  "y2": 0.0,  "theta2": -np.pi / 12,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.785, "x1": 0.0, "y1": 1.1, "theta1": -np.pi / 2,
                     "x2": 0.0,  "y2": 0.0,  "theta2": -np.pi / 6,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.810, "x1": 0.0, "y1": 1.1, "theta1": -np.pi / 2,
                     "x2": 0.0,  "y2": 0.0,  "theta2": -np.pi / 4,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.835, "x1": 0.0, "y1": 1.1, "theta1": -np.pi / 2,
                     "x2": 0.0,  "y2": 0.0,  "theta2": -np.pi / 3,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.86, "x1": 0.0, "y1": 1.1, "theta1": -np.pi / 2,
                     "x2": 0.0,  "y2": 0.0,  "theta2": -5 * np.pi / 12,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 0.885, "x1": 0.0, "y1": 1.1, "theta1": -np.pi / 2,
                     "x2": 0.0,  "y2": 0.0,  "theta2": -np.pi / 2,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
        {"t": 1.000, "x1": 0.0, "y1": 1.2, "theta1": -np.pi / 2,
                     "x2": 0.0,  "y2": 0.3,  "theta2": -np.pi / 2,
                     "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    
    ]

    # ---------------------------------------------------------------- #
    #  Scenario 2: wall -> ceiling  (inside top-left corner)           #
    #                                                                  #
    #  Both assemblies start on the left wall (theta=-pi/2) and end    #
    #  on the ceiling (theta=pi, shortest arc through -pi).            #
    #                                                                  #
    #  Near-flat start and end: chain fully extended along the thigh   #
    #  direction (q≈0) at both ends.                                   #
    #                                                                  #
    #  Effective joint positions (from _wheel_surface_adjust):         #
    #    Wall (theta=-pi/2): x≈0.149 (wall snap)                      #
    #      front (side=+1): y ∈ [0.060, 0.962]  (ceil snap at 0.962) #
    #      back  (side=-1): y ∈ [0.239, 1.141]  (ceil snap at 1.141) #
    #    Ceiling (theta=pi): y≈1.009 (ceil snap)                      #
    #      front (side=+1): x_min ≈ 0.060                             #
    #      back  (side=-1): x_min ≈ 0.239                             #
    #                                                                  #
    #  Start (wall, near-flat): y1=0.95, y2=0.24, sep=0.71m (89%)    #
    #  End (ceiling, near-flat): x1=0.96, x2=0.24, sep=0.72m (90%)   #
    #  Back holds at y2=0.24 throughout front rotation → sep≥0.71m.   #
    #  Front holds at x1=0.85 throughout back rotation → sep≥0.67m.   #
    # ---------------------------------------------------------------- #
    _WTC_CY = 1.5   # local ceiling height for this scenario (1.2 × 1.25)

    KEYFRAMES_WALL_TO_CEILING = [
        # Both on wall, near-flat vertical chain (~89% of 1.00 m chain).
        # Front (side=+1) above back (side=-1): keeps them from crossing.
        {"t": 0.00, "x1": 0.0, "y1": 1.19, "theta1": -np.pi / 2,
                    "x2": 0.0, "y2": 0.30,  "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        # Front slides to corner.
        {"t": 0.10, "x1": 0.0, "y1": _WTC_CY, "theta1": -np.pi / 2,
                    "x2": 0.0, "y2": 0.30,     "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        # Front rotates at corner (-pi/2 → pi via -pi) in 15° steps × Δt=0.025.
        {"t": 0.125, "x1": 0.0, "y1": _WTC_CY, "theta1": -7 * np.pi / 12,
                     "x2": 0.0, "y2": 0.30,     "theta2": -np.pi / 2,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.150, "x1": 0.0, "y1": _WTC_CY, "theta1": -2 * np.pi / 3,
                     "x2": 0.0, "y2": 0.30,     "theta2": -np.pi / 2,
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
        {"t": 0.30, "x1": 0.25, "y1": _WTC_CY, "theta1": np.pi,
                    "x2": 0.0,  "y2": 0.44,     "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.385, "x1": 0.50, "y1": _WTC_CY, "theta1": np.pi,
                    "x2": 0.0,  "y2": 0.69,     "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.46, "x1": 0.71, "y1": _WTC_CY, "theta1": np.pi,
                    "x2": 0.0,  "y2": 0.91,     "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.525, "x1": 0.88, "y1": _WTC_CY, "theta1": np.pi,
                    "x2": 0.0,  "y2": 1.13,     "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.58, "x1": 1.00, "y1": _WTC_CY, "theta1": np.pi,
                    "x2": 0.0,  "y2": 1.31,     "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        # Back arrives at corner; front holds at x1=1.06.
        {"t": 0.64, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                    "x2": 0.0,  "y2": _WTC_CY,  "theta2": -np.pi / 2,
                    "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        # Back rotates at corner (-pi/2 → pi via -pi) in 15° steps × Δt=0.025.
        {"t": 0.665, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,  "y2": _WTC_CY,  "theta2": -7 * np.pi / 12,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.690, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,  "y2": _WTC_CY,  "theta2": -2 * np.pi / 3,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.715, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,  "y2": _WTC_CY,  "theta2": -3 * np.pi / 4,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.740, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,  "y2": _WTC_CY,  "theta2": -5 * np.pi / 6,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.765, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,  "y2": _WTC_CY,  "theta2": -11 * np.pi / 12,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        {"t": 0.790, "x1": 1.06, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.0,  "y2": _WTC_CY,  "theta2": np.pi,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
        # Both on ceiling; spread to near-flat end (~90% of 1.00 m chain).
        {"t": 1.00,  "x1": 1.20, "y1": _WTC_CY, "theta1": np.pi,
                     "x2": 0.30, "y2": _WTC_CY,  "theta2": np.pi,
                     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    ]

    # ---------------------------------------------------------------- #
    #  Scenario 3: ceiling-top -> exterior wall  (outside top-left)    #
    #                                                                  #
    #  Each assembly pivots about its inner (back) wheel at the corner.#
    #  Assembly 1 (side=+1): right wheel is the pivot.                 #
    #  Assembly 2 (side=-1): LEFT wheel is the pivot (different wheel  #
    #    due to side=-1 L-bracket orientation).                        #
    #                                                                  #
    #  All keyframes use constraint_set="outside_exact" to pass exact  #
    #  chain-joint positions without surface-contact clamping.         #
    # ---------------------------------------------------------------- #

    # --- Corner geometry constants ---
    _R    = wg.wheel_r    # 0.050
    _BAR  = wg.bar_len    # 0.140
    _THG  = wg.thigh      # 0.140
    _CLF  = wg.calf       # 0.042
    _CX   = WALL_X        # 0.0
    _CY   = CEILING_Y     # 0.55

    def _cj(phi):
        """Chain-joint for assembly 1 (side=+1): pivots about RIGHT wheel."""
        bwc_x = _CX - _R * np.sin(phi)
        bwc_y = _CY + _R * np.cos(phi)
        vx    = bwc_x - _BAR * np.cos(phi - np.pi / 4)
        vy    = bwc_y - _BAR * np.sin(phi - np.pi / 4)
        return (vx + _THG * np.cos(phi) - _CLF * np.sin(phi),
                vy + _THG * np.sin(phi) + _CLF * np.cos(phi))

    def _cj2(phi):
        """Chain-joint for assembly 2 (side=-1): pivots about RIGHT wheel."""
        rwc_x = _CX - _R * np.sin(phi)
        rwc_y = _CY + _R * np.cos(phi)
        vx    = rwc_x - _BAR * np.cos(phi - np.pi / 4)
        vy    = rwc_y - _BAR * np.sin(phi - np.pi / 4)
        return (vx - _THG * np.cos(phi) - _CLF * np.sin(phi),
                vy - _THG * np.sin(phi) + _CLF * np.cos(phi))

    # Assembly 1 stop: right-wheel centre at (WALL_X, CEILING_Y+R).
    _XSTP  = _cj(0)[0]           # -0.010  (thigh=0.08951)
    _Y0    = _cj(0)[1]           # 0.741
    # Assembly 2 stop: right-wheel centre at (WALL_X, CEILING_Y+R).
    _XSTP2 = _cj2(0)[0]          # -0.189  (thigh=0.08951)
    # End of pivot for each assembly (phi=π/2).
    _XW    = _cj(np.pi / 2)[0]   # -0.191
    _YW    = _cj(np.pi / 2)[1]   # 0.541  (thigh=0.08951)
    _YW2   = _cj2(np.pi / 2)[1]  # 0.362  (thigh=0.08951)

    _CSE   = "outside_exact"  # exact positions; outside penalties still on

    # Back waits until front rotation is complete, then approaches.
    # x goes from 0.300 at t=0.490 to _XSTP2 at t=0.670 (Δt=0.180 → 2.72/unit ✓).
    # Front descends from _YW=0.541 starting at t=0.490, ending at t=1.000 (Δt=0.510).
    # Rate = 0.891/0.510 = 1.75/unit ✓.
    def _y1_descent(t):
        """Front y during linear descent from _YW at t=0.490 at 1.75/unit."""
        return _YW - 1.75 * (t - 0.490)

    KEYFRAMES_OUTSIDE = [
        # Phase 0 — both approach on ceiling (theta=0), moving left at ≤3.0/unit.
        # front: 0.82→_XSTP≈-0.051, Δ=0.871 in 0.290 → 3.00/unit ✓
        # back:  1.15→0.300,         Δ=0.850 in 0.290 → 2.93/unit ✓
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
        # Back approaches at _BACK_RATE2 ≈ 2.72/unit, arriving at _XSTP2 at t=0.670.
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
        # Both descend at ≤3.0/unit (back Δt=0.155 remaining → 1.26/unit ✓).
        # front: _YW→-0.350, Δ≈0.849 over Δt=0.510 → 1.66/unit ✓
        # back:  _YW2→0.125, Δ=0.195 over Δt=0.155 → 1.26/unit ✓
        {"t": 1.000, "x1": _XW, "y1": -0.350, "theta1": np.pi / 2,
                     "x2": _XW,  "y2":  0.125, "theta2": np.pi / 2,
                     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
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
    _R_TE   = wg.wheel_r    # 0.050
    _BAR_TE = wg.bar_len    # 0.140

    # V-apex pivot function: CW semicircle about terminus (phi: 0→π), theta = −phi.
    def _cp_te(phi):
        """V-apex position during CW semicircle pivot about terminus. theta = −phi."""
        dx0 = +_BAR_TE / np.sqrt(2.0)          # +0.099  (back-wheel x offset from terminus)
        dy0 = _R_TE + _BAR_TE / np.sqrt(2.0)   # +0.149  (back-wheel y offset from terminus)
        return (EDGE_X + dx0 * np.cos(phi) + dy0 * np.sin(phi),
                EDGE_Y - dx0 * np.sin(phi) + dy0 * np.cos(phi))

    def _cj_te(phi):
        """Chain-joint for side=+1 assembly during CW pivot: left wheel fixed at (EDGE_X, EDGE_Y+R). theta=−phi."""
        vx, vy = _cp_te(phi)
        return (vx + wg.thigh * np.cos(phi) + wg.calf * np.sin(phi),
                vy - wg.thigh * np.sin(phi) + wg.calf * np.cos(phi))

    def _cj_te2(phi):
        """Chain-joint for side=-1 assembly during CW pivot. Same V-apex as _cp_te (left wheel
        rolls around (EDGE_X, EDGE_Y) at radius R), but thigh sign flips for side=-1."""
        vx, vy = _cp_te(phi)
        return (vx - wg.thigh * np.cos(phi) + wg.calf * np.sin(phi),
                vy + wg.thigh * np.sin(phi) + wg.calf * np.cos(phi))

    # --- side=+1 (green, assembly 1) constants: pivots second (trailing on top, leads below) ---
    # Left-wheel stop: cx - thigh - BAR/√2 = EDGE_X → cx = EDGE_X + thigh + BAR/√2
    _XSTP_TE  = EDGE_X + _BAR_TE / np.sqrt(2.0) + wg.thigh   # 0.539
    # Front-wheel arrival (right wheel at EDGE_X): cx - thigh + BAR/√2 = EDGE_X → cx = EDGE_X + thigh - BAR/√2
    _XSTP2_TE = EDGE_X - _BAR_TE / np.sqrt(2.0) + wg.thigh   # 0.340
    _Y0_TE    = EDGE_Y + _R_TE + _BAR_TE / np.sqrt(2.0) + wg.calf  # 0.591
    _XW_TE    = _cj_te(np.pi)[0]    # 0.162
    _YW_TE    = _cj_te(np.pi)[1]    # 0.209

    # --- side=-1 (red, assembly 2) constants: pivots first (leads on top, trails below) ---
    # Left-wheel stop: cx + thigh - BAR/√2 = EDGE_X → cx = EDGE_X - thigh + BAR/√2
    _XSTP_red = EDGE_X - wg.thigh + _BAR_TE / np.sqrt(2.0)    # 0.360
    _XW_red   = _cj_te2(np.pi)[0]   # 0.340
    _YW_red   = _cj_te2(np.pi)[1]   # 0.209 (= _YW_TE; both sides end at same y)

    _CTE = "thin_edge_exact"

    KEYFRAMES_THIN_EDGE = [
        # Phase 0 — both approach on edge top (theta=0), moving right.
        # Red (x2, assembly 2) leads; green (x1, assembly 1) trails.
        # x1: -0.475→0.125 in 0.20 → 3.00/unit ✓   x2: -0.108→_XSTP_red in 0.20 → 3.00/unit ✓
        {"t": 0.00, "x1": -0.475, "y1": _Y0_TE, "theta1": 0.0,
                    "x2": -0.108, "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},

        # Phase 1 — red stops (left wheel above terminus); green still approaching.
        {"t": 0.20, "x1":  0.125,    "y1": _Y0_TE, "theta1": 0.0,
                    "x2": _XSTP_red, "y2": _Y0_TE, "theta2": 0.0,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},

        # Phase 2 — red pivot (CW: theta2: 0 → −π), 12 steps × 15° × Δt=0.025 = 0.300 total.
        # 15°/2.225 fr = 6.74°/fr ✓ (30° steps gave 13.5°/fr → JUMP violations).
        # Green approaches from x1=0.125: 0.053/step → 2.12/unit → ω=0.476 ✓; arrives at _XSTP2_TE step 5.
        {"t": 0.225, "x1":  0.178,    "y1": _Y0_TE, "theta1": 0.0,
                     "x2": _cj_te2( np.pi/12)[0], "y2": _cj_te2( np.pi/12)[1],
                     "theta2": -np.pi / 12,
                     "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.250, "x1":  0.231,    "y1": _Y0_TE, "theta1": 0.0,
                     "x2": _cj_te2(  np.pi/6)[0], "y2": _cj_te2(  np.pi/6)[1],
                     "theta2": -np.pi / 6,
                     "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.275, "x1":  0.284,    "y1": _Y0_TE, "theta1": 0.0,
                     "x2": _cj_te2(  np.pi/4)[0], "y2": _cj_te2(  np.pi/4)[1],
                     "theta2": -np.pi / 4,
                     "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.300, "x1":  0.3,    "y1": _Y0_TE, "theta1": 0.0,
                     "x2": _cj_te2(  np.pi/3)[0], "y2": _cj_te2(  np.pi/3)[1],
                     "theta2": -np.pi / 3,
                     "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        # Green arrives at front-wheel stop (_XSTP2_TE) and holds while red continues pivot.
        {"t": 0.325, "x1": 0.3, "y1": _Y0_TE, "theta1": 0.0,
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
        {"t": 0.400, "x1": 0.337, "y1": _Y0_TE, "theta1": 0.0,
                     "x2": _cj_te2(2*np.pi/3)[0], "y2": _cj_te2(2*np.pi/3)[1],
                     "theta2": -2 * np.pi / 3,
                     "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.425, "x1": _XSTP2_TE, "y1": _Y0_TE, "theta1": 0.0,
                     "x2": _cj_te2(3*np.pi/4)[0], "y2": _cj_te2(3*np.pi/4)[1],
                     "theta2": -3 * np.pi / 4,
                     "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
        {"t": 0.450, "x1": _XSTP2_TE, "y1": _Y0_TE, "theta1": 0.0,
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

        # Phase 3 — stagger: green rolls to pivot start in 0.095 t-units → 2.98/unit → ω=0.670 ✓
        # cold_at=0.61 fires between stagger end (t=0.595) and Phase 4 step 1 (t=0.620).
        {"t": 0.595, "x1": _XSTP_TE, "y1": _Y0_TE, "theta1": 0.0,
                     "x2":  0.375,    "y2": _YW_red, "theta2": -np.pi,
                     "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},

        # Phase 4 — green pivot (CW: theta1: 0 → −π), 12 steps × 15° × Δt=0.025 = 0.300 total.
        # Red retreats from x2=0.375 → −0.156 over 11 steps: 0.048/step → 1.93/unit → ω=0.434 ✓
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

        # Phase 5 — both below edge, moving left (0.105 t-units).
        # green: 0.309/0.105=2.94/unit ω=0.661 ✓   red: 0.284/0.105=2.71/unit ω=0.608 ✓
        {"t": 1.00, "x1": -0.100,   "y1": _YW_TE, "theta1": -np.pi,
                    "x2": -0.440,   "y2": _YW_red, "theta2": -np.pi,
                    "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    ]

    if SCENARIO == "floor_to_wall":
        keyframes = KEYFRAMES_FLOOR_TO_WALL
        xlim = (-0.2, 2.2)
        ylim = (-0.2, 1.5)
        output = "floor_to_wall.mp4"
        label  = "Floor-to-wall"
    elif SCENARIO == "wall_to_ceiling":
        keyframes = KEYFRAMES_WALL_TO_CEILING
        xlim = (-0.5, 1.75)
        ylim = (-0.5, 2.0)
        output = "wall_to_ceiling.mp4"
        label  = "Wall-to-ceiling"
    elif SCENARIO == "outside":
        keyframes = KEYFRAMES_OUTSIDE
        xlim = (-0.75, 1.5)
        ylim = (-0.75, 1.25)
        output = "outside.mp4"
        label  = "Outside corner (ceiling-top -> exterior wall)"
    else:  # "thin_edge"
        keyframes = KEYFRAMES_THIN_EDGE
        xlim = (-0.7, 1.2)
        ylim = (-0.35, 1.0)
        output = "thin_edge.mp4"
        label  = "Thin edge (top -> bottom)"

    N_FRAMES       = 90
    FPS            = 15
    N_GRID         = 60
    if SCENARIO == "thin_edge":
        SMOOTH_WEIGHT = 30.0
    elif SCENARIO == "wall_to_ceiling":
        SMOOTH_WEIGHT = 60.0
    elif SCENARIO == "outside":
        SMOOTH_WEIGHT = 200.0
    elif SCENARIO == "floor_to_wall":
        SMOOTH_WEIGHT = 200.0
    else:
        SMOOTH_WEIGHT = 20.0

    print(f"{label}  ({N_FRAMES} frames @ {FPS} fps)")
    print("-" * 60)
    if SCENARIO == "thin_edge":
        _cold = [0.50, 0.61]
    elif SCENARIO == "wall_to_ceiling":
        _cold = [0.29]
    elif SCENARIO == "floor_to_wall":
        _cold = None
    else:
        _cold = None
    frames = solve_trajectory(keyframes, wg, l1, l2, l3, N_FRAMES, N_GRID,
                              smooth_weight=SMOOTH_WEIGHT, cold_at=_cold)
    print("-" * 60)
    # ---- joint-angle limit + jump + sign + ω check ----------------- #
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
            # _pc[xk] is already the interpolated position at this frame;
            # dx is change per frame in metres (no further scaling needed)
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
    # ---------------------------------------------------------------- #
    if "--no-render" not in sys.argv:
        print("Rendering...")
        render_video(frames, wg, l1, l2, l3, xlim, ylim, output, FPS)
