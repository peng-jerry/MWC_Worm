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
import math
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constraints import WheelGeometry, floor_y_for_assembly as _floor_y
from solver import solve_ik
from visualize import plot_robot
from kinematics import q_display as _q_display


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
    return (vx - wg.calf * np.sin(theta),
            vy + wg.calf * np.cos(theta))


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

def _draw_surfaces(ax, constraint_set, wall_x, ceiling_y, xlim, ylim,
                   marker_scale=1.0):
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
        ax.plot(wall_x, ceiling_y, "D", color="saddlebrown", ms=8 * marker_scale,
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
        q_disp = _q_display(q)
        ax.set_title(
            f"Frame {idx + 1}/{len(results)}  —  "
            + "  ".join(f"q{i+1}={np.degrees(qi):+.1f}°" for i, qi in enumerate(q_disp)),
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
# arm_b is the hypotenuse of a right triangle: base=wheelbase, height=bar_len_a.
# Ensures arm_a and arm_b wheels are at equal depth at theta=0 (simultaneous floor contact).
_WHEELBASE = 0.1263   # centre-to-centre distance between arm_a and arm_b wheels (m)
wg = WheelGeometry(
    bar_len=np.hypot(_WHEELBASE, 0.125),       # ≈ 0.1777 m
    wheel_r=0.050,
    arm_a1=0.0,  arm_b1= np.arctan(_WHEELBASE / 0.125),    # ≈ +45.3°  (ftw / wtc)
    arm_a2=0.0,  arm_b2=-np.arctan(_WHEELBASE / 0.125),    # ≈ −45.3°  (ftw / wtc)
    calf=0.0921, bar_len_a=0.125,
)

_ARM_B = wg.arm_b1    # magnitude ≈ 45.3° — reused by outside / thin_edge variants

wg_outside = WheelGeometry(
    bar_len=wg.bar_len, wheel_r=wg.wheel_r,
    arm_a1=0.0, arm_b1=-_ARM_B,               # both negative (arm_b trails)
    arm_a2=0.0, arm_b2=-_ARM_B,
    calf=wg.calf, bar_len_a=wg.bar_len_a,
)
wg_thin_edge = WheelGeometry(
    bar_len=wg.bar_len, wheel_r=wg.wheel_r,
    arm_a1=0.0, arm_b1= _ARM_B,               # both positive (arm_b leads)
    arm_a2=0.0, arm_b2= _ARM_B,
    calf=wg.calf, bar_len_a=wg.bar_len_a,
)

WALL_X    = 0.0
CEILING_Y = 0.55

EDGE_X = 0.44   # right terminus x for "thin_edge"  (0.35 × 1.25)
EDGE_Y = 0.50   # edge height for "thin_edge"      (0.40 × 1.25)

# ── Frame count and ω limit — shared with wheel_omega.py ────────────────────
N_FRAMES   = 90        # frames per trajectory
_OMEGA_LIM   = 30 * np.radians(7.2)   # hard-ceiling for ω check (≈3.770 rad/fr) — rarely reached in practice
_OMEGA_SLIDE = 0.70                    # operational target for _dt_slide keyframe spacing (0.035 m/fr at wheel_r=0.050)

def _dt_slide(*disps):
    """Min Δt between two keyframes so max wheel-surface speed ≤ _OMEGA_SLIDE.

    Uses integer-frame ceiling so the ω bound holds for every discrete frame.
    """
    n = math.ceil(max(abs(d) for d in disps) / (_OMEGA_SLIDE * wg.wheel_r))
    return n / (N_FRAMES - 1)

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

# Wheel assembly separation: 80 % of the total chain length.
# Drives approach start position (x1) and departure end position (y1).
_SEP        = 0.8 * (l1 + l2 + l3)   # 0.80 m with current geometry
_FTW_END_Y2 = 0.1                     # departure: back assembly y at t=1.0
# Geometric body positions derived from wheel assembly at surface contact.
# At theta=-π/2 on floor (front assembly on wall): body y = floor_y_for_assembly(-π/2, wg, 1).
_FTW_Y1_ROT_END = wg.wheel_r + wg.bar_len * np.sin(wg.arm_b1)    # ≈ 0.172
# At theta=0 on floor (back assembly flat): body y = floor_y_for_assembly(0, wg, 1).
_FTW_Y2_FLOOR   = wg.wheel_r + wg.bar_len * np.cos(wg.arm_b1) + wg.calf  # ≈ 0.301
# x2 at end of front rotation (start of "both moving" phase).
_FTW_X2_SLIDE_S = 0.958
# Both assemblies advance by _FTW_APP_DELTA together so joint angles stay
# constant during approach (constant separation = _FTW_X2_SLIDE_S − _ftw1(0)[0]).
# ω limit: 0.30 m / (9 fr × 0.05 m) = 0.667 rad/fr ✓
_FTW_APP_DELTA  = 0.30
# y-separation when the back rotation finishes: used for departure.
_FTW_DEPART_SEP = (l1 + l2 + l3) - _floor_y(-np.pi / 2, wg, -1)

KEYFRAMES_FLOOR_TO_WALL = [
    # ---- Approach: both assemblies advance together (constant separation). ----
    # Separation = _FTW_X2_SLIDE_S − _ftw1(0)[0] ≈ 0.908 throughout → joints fixed.
    # Both travel _FTW_APP_DELTA = 0.45 m in Δt=0.15 → ω=0.692 rad/frame ✓
    {"t": 0.000, "x1": _ftw1(0.0)[0] + _FTW_APP_DELTA, "y1": 0.0, "theta1": 0.0,
                 "x2": _FTW_X2_SLIDE_S + _FTW_APP_DELTA, "y2": 0.0, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # Front arm_a1 reaches corner; back holds at x2=0.958 (monotone — no reversal).
    # x1 from WSA-corrected pivot; chain extension at stop ≈ 91% (stable with correct x1).
    {"t": 0.15, "x1": _ftw1(0.0)[0], "y1": _ftw1(0.0)[1], "theta1": 0.0,
                 "x2": 0.958, "y2": _FTW_Y2_FLOOR, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # ---- Front rotation: arm_a1 pivots at corner, theta: 0 → -π/2 ----
    # x1 from _ftw1 pivot (arm_a1 exactly at wall); y1 from floor_y_for_assembly
    # (arm_b1 deepens without thigh → pivot formula gives y<floor at large theta).
    # x2=0.958 held constant through rotation — back never reverses direction.
    {"t": 0.183, "x1": _ftw1(-np.pi / 12)[0], "y1": _floor_y(-np.pi / 12, wg, 1),
                 "theta1": -np.pi / 12,
                 "x2": 0.958, "y2": _FTW_Y2_FLOOR, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.216, "x1": _ftw1(-np.pi / 6)[0],  "y1": _floor_y(-np.pi / 6,  wg, 1),
                 "theta1": -np.pi / 6,
                 "x2": 0.958, "y2": _FTW_Y2_FLOOR, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.249, "x1": _ftw1(-np.pi / 4)[0],  "y1": _floor_y(-np.pi / 4,  wg, 1),
                 "theta1": -np.pi / 4,
                 "x2": 0.958, "y2": _FTW_Y2_FLOOR, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.282, "x1": _ftw1(-np.pi / 3)[0],  "y1": _floor_y(-np.pi / 3,  wg, 1),
                 "theta1": -np.pi / 3,
                 "x2": 0.958, "y2": _FTW_Y2_FLOOR, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.315, "x1": _ftw1(-5 * np.pi / 12)[0], "y1": _floor_y(-5 * np.pi / 12, wg, 1),
                 "theta1": -5 * np.pi / 12,
                 "x2": 0.958, "y2": _FTW_Y2_FLOOR, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # theta=-90°: front locks on wall.  Δt=0.045 (not 0.033) so q2 changes
    # ≤7.5°/frame through the -75°→-90° branch-flip zone.
    # y1=_FTW_Y1_ROT_END so the hint interpolation to y1=0.42 at t=0.48 is ω-safe.
    {"t": 0.360, "x1": _ftw1(-np.pi / 2)[0],  "y1": _FTW_Y1_ROT_END,
                 "theta1": -np.pi / 2,
                 "x2": _FTW_X2_SLIDE_S, "y2": _FTW_Y2_FLOOR, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # ---- Both moving: front climbs, back slowly approaches ----
    # y1 scaled as fraction of total chain so hints stay proportional if links change.
    # x2 held at 0.958 through t=0.44 so chain extension stays ≥66% at fr41-42
    # (new geometry has x1_eff=0.268 vs old 0.233, requiring a higher x2 offset to
    # prevent the branch-flip that fires at ~63% extension with the old x2=0.929 start).
    # ω budget: Δx2=0.218 per step (same as old) spread over equal Δt=0.07 intervals.
    # y2=_FTW_Y2_FLOOR (not 0.0) so the ω hint matches the physical floor contact height.
    {"t": 0.480, "x1": 0.0, "y1": 0.42 * (l1 + l2 + l3), "theta1": -np.pi / 2,
                 "x2": 0.958, "y2": _FTW_Y2_FLOOR, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # x2 decreases at constant rate from 0.958 (t=0.480) to ~0.550 (t=0.620) to avoid
    # a rate-change kink that causes q2 to jump branches at ~fr47.
    # Rate: (0.958-0.550)/(0.620-0.480)/89 ≈ 0.033 m/frame → ω=0.66 < 0.70 ✓
    {"t": 0.550, "x1": 0.0, "y1": 0.62 * (l1 + l2 + l3), "theta1": -np.pi / 2,
                 "x2": 0.754, "y2": _FTW_Y2_FLOOR, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.620, "x1": 0.0, "y1": 0.82 * (l1 + l2 + l3), "theta1": -np.pi / 2,
                 "x2": 0.550, "y2": _FTW_Y2_FLOOR, "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # Back arm_b2 reaches corner: x2,y2 from pivot formula at theta=0.
    # y1 = l1+l2+l3: at back's theta=-π/2 the chain spans purely vertically from
    # y2≈0 (floor) to y1, so y1 must equal the total chain length.
    {"t": 0.775, "x1": 0.0, "y1": l1 + l2 + l3, "theta1": -np.pi / 2,
                 "x2": _ftw2(0.0)[0], "y2": _ftw2(0.0)[1], "theta2": 0.0,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # ---- Back rotation: arm_b2 slides up wall, arm_a2 slides inward on floor ----
    # x2 from _ftw2 (= wall_x_for_assembly: arm_b2 on wall at x=WALL_X+wheel_r).
    # y2 = floor_y_for_assembly (arm_a2 stays on floor throughout).
    # At theta=-π/2, floor_y gives y2=0.050 and arm_a2 arrives at the corner (0.050,0.050).
    # Clamping y2 to _FTW_Y1_ROT_END lifts arm_a2 off the floor → assembly appears to float.
    {"t": 0.805, "x1": 0.0, "y1": l1 + l2 + l3, "theta1": -np.pi / 2,
                 "x2": _ftw2(-np.pi / 12)[0], "y2": _floor_y(-np.pi / 12, wg, -1),
                 "theta2": -np.pi / 12,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.835, "x1": 0.0, "y1": l1 + l2 + l3, "theta1": -np.pi / 2,
                 "x2": _ftw2(-np.pi / 6)[0],  "y2": _floor_y(-np.pi / 6,  wg, -1),
                 "theta2": -np.pi / 6,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.865, "x1": 0.0, "y1": l1 + l2 + l3, "theta1": -np.pi / 2,
                 "x2": _ftw2(-np.pi / 4)[0],  "y2": _floor_y(-np.pi / 4,  wg, -1),
                 "theta2": -np.pi / 4,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    {"t": 0.895, "x1": 0.0, "y1": l1 + l2 + l3, "theta1": -np.pi / 2,
                 "x2": _ftw2(-np.pi / 3)[0],  "y2": _floor_y(-np.pi / 3,  wg, -1),
                 "theta2": -np.pi / 3,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # Δt=0.040 for last two steps (steps 1-4 at 0.030): near-singularity at high chain
    # extension amplifies theta2 rate → q4 jumps. 4.21°/frame × 1.7 ≈ 7.2°/frame < 7.5° ✓
    {"t": 0.935, "x1": 0.0, "y1": l1 + l2 + l3, "theta1": -np.pi / 2,
                 "x2": _ftw2(-5 * np.pi / 12)[0], "y2": _floor_y(-5 * np.pi / 12, wg, -1),
                 "theta2": -5 * np.pi / 12,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # theta=-π/2: arm_a2 arrives at corner (0.050, 0.050); y2=floor_y=0.050.
    # Departure Δy = 0.250 in 2.2 fr → ω=2.25 rad/fr < 3.927 ✓
    {"t": 0.975, "x1": 0.0, "y1": l1 + l2 + l3, "theta1": -np.pi / 2,
                 "x2": _ftw2(-np.pi / 2)[0],  "y2": _floor_y(-np.pi / 2, wg, -1),
                 "theta2": -np.pi / 2,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
    # ---- Departure: both climb wall together (constant separation = _FTW_DEPART_SEP). ----
    {"t": 1.000, "x1": 0.0, "y1": _FTW_END_Y2 + _FTW_DEPART_SEP, "theta1": -np.pi / 2,
                 "x2": 0.0, "y2": _FTW_END_Y2,                     "theta2": -np.pi / 2,
                 "constraint_set": "wall", "wall_x": WALL_X, "ceiling_y": _FTW_CY},
]

# ── Scenario 2: wall -> ceiling  (inside top-left corner) ───────────────────
#
#  Both assemblies start on the left wall (theta=-pi/2) and end on the ceiling
#  (theta=pi, shortest arc through -pi).
#
#  Near-flat start and end: chain fully extended along the thigh direction
#  (q≈0) at both ends.

_WTC_CY    = 1.5   # local ceiling height for this scenario
# Back assembly y2 at t=0 (start of approach).
_WTC_Y2_ROT = _WTC_CY - (l1 + l2 + l3) + 0.01            # ≈ 0.51
# Approach-end y2: both assemblies advance by the same Δy = _WTC_CY − (_WTC_Y2_ROT+_SEP)
# during approach so separation stays _SEP → q frozen → no IK branch switches.
_WTC_Y2_APP_END = _WTC_CY - _SEP                          # ≈ 0.70 — symmetric approach: both advance same Δy, q frozen
# At theta=π: calf is perpendicular to theta (horizontal) → no y-offset from calf.
# arm_a2 x-offset = bar_len_a * cos(π/2 + 0) = 0; arm_b2 x-offset > 0 → min = 0.
# x_joint = wall_x + wheel_r (= wall_x_for_assembly(π, wg, WALL_X, side=-1)).
_WTC_X2_EFF  = WALL_X + wg.wheel_r                      # arm_a2 at wall-ceiling corner (0.05, 1.45) after back rotation
# Full-extension reference: same logic as FTW y1 = l1+l2+l3.
# At theta2=π: chain = x1_eff − x2_eff = x1 − _WTC_X2_EFF → x1 = _WTC_X2_EFF + chain_length.
# This is the front-assembly x at which chain would be 100 % with back at the corner.
# IK branch instability near 100 % extension prevents reaching it cleanly in trajectory.
_WTC_X1_HOLD = _WTC_X2_EFF + (l1 + l2 + l3)            # ≈ 1.140

# Wall-ceiling corner pivot (same structure as FTW floor-wall corner):
#   front (side=+1): arm_a1 (angle=0, bar_len_a) is at the corner when theta=−π/2
#   back  (side=−1): arm_b2 (angle=arm_b2, bar_len) is at the corner when theta=−π/2
# Identical arm choices to FTW; only the corner coordinates differ.
_WTC_CORNER = (WALL_X + wg.wheel_r, _WTC_CY - wg.wheel_r)   # (0.05, 1.45)

def _wtc1(th):
    return _cj_pivot(th, _WTC_CORNER, wg.arm_a1, wg.bar_len_a, wg, side=+1)
def _wtc2(th):
    return _cj_pivot(th, _WTC_CORNER, wg.arm_b2, wg.bar_len,   wg, side=-1)

# Glide: x1 advances along ceiling to _WTC_X1_GLIDE_E while y2 holds.
# Phase A' (4 steps): x1 and y2 both advance 0.10 m/step; front moves right 0.40 m total
#   (13 frames) before the back reaches the corner.
# Phase B: x1 holds at Phase A' end (0.90); y2 advances 1.10→1.50 in 2 large strides (0.20 m each,
#   6/89 fr each). Rate = 0.667 ω — same as Phase A', so no abrupt change at the boundary.
#   Fewer, larger strides give a purposeful feel vs. four slow creep-steps.
# Back rotation: back rotates 90° CW; x1 stays at 0.90 (chain 80-90% extension → clean IK).
#   Starts immediately after Phase B (no pre-rotation hold, matching front assembly sequence).
#   At rotation end (theta2=π), arm_a2 is at wall-ceiling corner (0.05, 1.45); arm_b2 on ceiling.
# End: 4-frame hold at corner, then 1 departure frame (ω=0.70 at limit).
_WTC_X1_ROT_END    = WALL_X + wg.wheel_r + wg.bar_len * np.sin(wg.arm_b1)  # x at theta=π (≈ −0.076)
_WTC_X1_GLIDE_E    = 0.40                                              # glide end; Phase A' starts here (14/89 from rot-end)
_WTC_GLIDE_DT      = _dt_slide(_WTC_X1_GLIDE_E - _WTC_X1_ROT_END)    # 14/89
_WTC_SLIDE_N       = 4                                                 # Phase A' steps (x1: 0.40→0.80, y2: 0.70→1.10)
_WTC_SLIDE_A_DX    = 0.10                                              # Δx1 = Δy2 per Phase A' step
_WTC_SLIDE_A_DT    = _dt_slide(_WTC_SLIDE_A_DX)                       # 3/89 per step
_WTC_SLIDE_1_DT    = 4 / (N_FRAMES - 1)                               # first step 4/89 smooths q4 jump at glide→A' boundary
_WTC_SLIDE_A_END   = _WTC_X1_GLIDE_E + _WTC_SLIDE_N * _WTC_SLIDE_A_DX  # x1 after Phase A' = 0.80
_WTC_Y2_SLIDE_S    = _WTC_Y2_APP_END                                   # Phase A' start y2 (= 0.70)
_WTC_Y2_SLIDE_E    = _WTC_Y2_SLIDE_S + _WTC_SLIDE_N * _WTC_SLIDE_A_DX  # y2 after Phase A' = 1.10
_WTC_T_APPROACH    = _dt_slide(_WTC_CY - _WTC_Y2_ROT - _SEP)                       # 6/89 (q frozen during approach → no jump risk)
_WTC_FRONT_ROT_DT  = 0.040                                             # front rotation: 0.040 → ~7°/fr on q1 (< 7.2° threshold)
_WTC_ROT_DT        = 0.034                                             # back rotation: DT=0.034 (q4 ≈ 6.8°/fr; 0.030 was too fast)
_WTC_T_ROT_END     = _WTC_T_APPROACH + 6 * _WTC_FRONT_ROT_DT          # 6 × 15° steps
_WTC_T_GLIDE_END   = _WTC_T_ROT_END + _WTC_GLIDE_DT                  # end of x1-only glide
_WTC_T_SLIDE_A     = _WTC_T_GLIDE_END + _WTC_SLIDE_1_DT + (_WTC_SLIDE_N - 1) * _WTC_SLIDE_A_DT  # Phase A' end
_WTC_PHASE_B_DX    = 0.20                                              # y2 step size in Phase B (2 large strides)
_WTC_PHASE_B_N     = round((_WTC_CY - _WTC_Y2_SLIDE_E) / _WTC_PHASE_B_DX)  # 2 steps (y2: 1.10→1.50)
_WTC_PHASE_B_DT    = _dt_slide(_WTC_PHASE_B_DX)                       # 6/89 per step (ω=0.667, same rate as Phase A')
_WTC_T_SLIDE_END   = _WTC_T_SLIDE_A + _WTC_PHASE_B_N * _WTC_PHASE_B_DT  # Phase B end = back arrives at corner
_WTC_POST_HOLD_DT  = 4 / (N_FRAMES - 1)                                   # 4-frame hold at corner after back rotation
_WTC_DEPART_DX     = 0.035                                                  # 1 departure frame at ω limit (Δx=0.035 m → ω=0.70 rad/fr)

KEYFRAMES_WALL_TO_CEILING = [
    # Both on wall; separation along y = _SEP ✓.  Back holds at y2=_WTC_Y2_ROT;
    # front starts _SEP above it.
    {"t": 0.00, "x1": 0.0, "y1": _WTC_Y2_ROT + _SEP, "theta1": -np.pi / 2,
                "x2": 0.0, "y2": _WTC_Y2_ROT,        "theta2": -np.pi / 2,
                "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # Front slides to corner; back holds at y2=_WTC_Y2_APP_END.
    {"t": _WTC_T_APPROACH, "x1": 0.0, "y1": _WTC_CY, "theta1": -np.pi / 2,
                "x2": 0.0, "y2": _WTC_Y2_APP_END, "theta2": -np.pi / 2,
                "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # Front rotates at corner (-pi/2 → pi via -pi, CW 90°) in 15° steps.
    # x1, y1 from pivot formula (arm_a1 fixed at _WTC_CORNER) — same logic as FTW _ftw1.
    # y2 held at _WTC_Y2_APP_END during rotation to keep chain within reach (dist≤1.0m).
    {"t": _WTC_T_APPROACH + 1*_WTC_FRONT_ROT_DT, "x1": _wtc1(-7 * np.pi / 12)[0], "y1": _wtc1(-7 * np.pi / 12)[1],
                 "theta1": -7 * np.pi / 12,
                 "x2": 0.0, "y2": _WTC_Y2_APP_END, "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": _WTC_T_APPROACH + 2*_WTC_FRONT_ROT_DT, "x1": _wtc1(-2 * np.pi / 3)[0], "y1": _wtc1(-2 * np.pi / 3)[1],
                 "theta1": -2 * np.pi / 3,
                 "x2": 0.0, "y2": _WTC_Y2_APP_END, "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": _WTC_T_APPROACH + 3*_WTC_FRONT_ROT_DT, "x1": _wtc1(-3 * np.pi / 4)[0], "y1": _wtc1(-3 * np.pi / 4)[1],
                 "theta1": -3 * np.pi / 4,
                 "x2": 0.0, "y2": _WTC_Y2_APP_END, "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": _WTC_T_APPROACH + 4*_WTC_FRONT_ROT_DT, "x1": _wtc1(-5 * np.pi / 6)[0], "y1": _wtc1(-5 * np.pi / 6)[1],
                 "theta1": -5 * np.pi / 6,
                 "x2": 0.0, "y2": _WTC_Y2_APP_END, "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": _WTC_T_APPROACH + 5*_WTC_FRONT_ROT_DT, "x1": _wtc1(-11 * np.pi / 12)[0], "y1": _wtc1(-11 * np.pi / 12)[1],
                 "theta1": -11 * np.pi / 12,
                 "x2": 0.0, "y2": _WTC_Y2_APP_END, "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # theta=π: arm_a1 pivot puts body inside wall; use wall_x_for_assembly (arm_b1 limit)
    # so the slope from here to x1=l1 at t=0.331 stays within ω=0.70.  Same override
    # logic as FTW using 0.102 instead of _ftw1(-pi/2)[1]=-0.040.
    {"t": _WTC_T_ROT_END, "x1": _WTC_X1_ROT_END,
                 "y1": _wtc1(np.pi)[1],
                 "theta1": np.pi,
                 "x2": 0.0, "y2": _WTC_Y2_APP_END, "theta2": -np.pi / 2,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # Sliding phase — FTW-like: x1 glides ahead while y2 holds, then both advance.
    # Phase Glide: x1 from rotation-end to _WTC_X1_GLIDE_E; y2 stays at _WTC_Y2_SLIDE_S.
    {"t": _WTC_T_GLIDE_END,
     "x1": _WTC_X1_GLIDE_E, "y1": _WTC_CY, "theta1": np.pi,
     "x2": 0.0, "y2": _WTC_Y2_SLIDE_S, "theta2": -np.pi / 2,
     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # Phase A' (4 steps): x1 and y2 both advance 0.10 m per step.
    # Step 1 uses 4/89 interval to smooth the q4 jump at the glide→A' boundary.
    # Steps 2-4 each use 3/89.  Front moves right 0.40 m total (13 frames).
    {"t": _WTC_T_GLIDE_END + _WTC_SLIDE_1_DT,
     "x1": _WTC_X1_GLIDE_E + 1*_WTC_SLIDE_A_DX, "y1": _WTC_CY, "theta1": np.pi,
     "x2": 0.0, "y2": _WTC_Y2_SLIDE_S + 1*_WTC_SLIDE_A_DX, "theta2": -np.pi / 2,
     "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    *[{"t": _WTC_T_GLIDE_END + _WTC_SLIDE_1_DT + k * _WTC_SLIDE_A_DT,
       "x1": _WTC_X1_GLIDE_E + (k + 1) * _WTC_SLIDE_A_DX, "y1": _WTC_CY, "theta1": np.pi,
       "x2": 0.0, "y2": _WTC_Y2_SLIDE_S + (k + 1) * _WTC_SLIDE_A_DX, "theta2": -np.pi / 2,
       "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY}
      for k in range(1, _WTC_SLIDE_N)],
    # Phase B: x1 holds at _WTC_SLIDE_A_END (0.80); y2 advances 1.10→1.50 in 2 large strides.
    # Front stays stationary while back slides to the corner.
    *[{"t": _WTC_T_SLIDE_A + k * _WTC_PHASE_B_DT,
       "x1": _WTC_SLIDE_A_END, "y1": _WTC_CY, "theta1": np.pi,
       "x2": 0.0,
       "y2": _WTC_Y2_SLIDE_E + k * _WTC_PHASE_B_DX, "theta2": -np.pi / 2,
       "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY}
      for k in range(1, _WTC_PHASE_B_N + 1)],
    # Back rotates at corner (-pi/2 → pi via -pi, CW 90°) in 15° steps.
    # x2, y2 from pivot formula (arm_b2 fixed at _WTC_CORNER) — same logic as FTW _ftw2.
    # x1 holds at _WTC_SLIDE_A_END (0.80); chain stays ~80% extended → smooth IK.
    {"t": _WTC_T_SLIDE_END + 1*_WTC_ROT_DT, "x1": _WTC_SLIDE_A_END, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": _wtc2(-7 * np.pi / 12)[0], "y2": _wtc2(-7 * np.pi / 12)[1],
                 "theta2": -7 * np.pi / 12,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": _WTC_T_SLIDE_END + 2*_WTC_ROT_DT, "x1": _WTC_SLIDE_A_END, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": _wtc2(-2 * np.pi / 3)[0], "y2": _wtc2(-2 * np.pi / 3)[1],
                 "theta2": -2 * np.pi / 3,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": _WTC_T_SLIDE_END + 3*_WTC_ROT_DT, "x1": _WTC_SLIDE_A_END, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": _wtc2(-3 * np.pi / 4)[0], "y2": _wtc2(-3 * np.pi / 4)[1],
                 "theta2": -3 * np.pi / 4,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": _WTC_T_SLIDE_END + 4*_WTC_ROT_DT, "x1": _WTC_SLIDE_A_END, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": _wtc2(-5 * np.pi / 6)[0], "y2": _wtc2(-5 * np.pi / 6)[1],
                 "theta2": -5 * np.pi / 6,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    {"t": _WTC_T_SLIDE_END + 5*_WTC_ROT_DT, "x1": _WTC_SLIDE_A_END, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": _wtc2(-11 * np.pi / 12)[0], "y2": _wtc2(-11 * np.pi / 12)[1],
                 "theta2": -11 * np.pi / 12,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # theta=π: arm_a2 at wall-ceiling corner (0.05, 1.45); x2=_WTC_X2_EFF=wall+R.
    {"t": _WTC_T_SLIDE_END + 6*_WTC_ROT_DT, "x1": _WTC_SLIDE_A_END, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": _WTC_X2_EFF, "y2": _wtc2(np.pi)[1],
                 "theta2": np.pi,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # 4-frame hold at corner (arm_a2 touching wall, arm_a2 at ceiling).
    {"t": _WTC_T_SLIDE_END + 6*_WTC_ROT_DT + _WTC_POST_HOLD_DT,
                 "x1": _WTC_SLIDE_A_END, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": _WTC_X2_EFF, "y2": _wtc2(np.pi)[1], "theta2": np.pi,
                 "constraint_set": "ceiling", "wall_x": WALL_X, "ceiling_y": _WTC_CY},
    # Departure: 1 frame at ω limit; both assemblies advance Δx=0.035 m along ceiling.
    {"t": 1.00,  "x1": _WTC_SLIDE_A_END + _WTC_DEPART_DX, "y1": _WTC_CY, "theta1": np.pi,
                 "x2": _WTC_X2_EFF + _WTC_DEPART_DX, "y2": _wtc2(np.pi)[1], "theta2": np.pi,
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
_BARA = wg.bar_len_a  # 0.125  (arm_a lever length)
_CLF  = wg.calf       # 0.0921


def _cj(phi):
    """Chain-joint for assembly 1 (side=+1): pivots about arm_a1 wheel."""
    awc_x = WALL_X - _R * np.sin(phi)
    awc_y = CEILING_Y + _R * np.cos(phi)
    arm_dir = phi - np.pi / 2 + wg.arm_a1   # arm_a1 direction at theta=phi
    vx    = awc_x - _BARA * np.cos(arm_dir)
    vy    = awc_y - _BARA * np.sin(arm_dir)
    return (vx - _CLF * np.sin(phi),
            vy + _CLF * np.cos(phi))


def _cj2(phi):
    """Chain-joint for assembly 2 (side=-1): pivots about arm_a2 wheel."""
    awc_x = WALL_X - _R * np.sin(phi)
    awc_y = CEILING_Y + _R * np.cos(phi)
    arm_dir = phi - np.pi / 2 + wg.arm_a2   # arm_a2 direction at theta=phi
    vx    = awc_x - _BARA * np.cos(arm_dir)
    vy    = awc_y - _BARA * np.sin(arm_dir)
    return (vx - _CLF * np.sin(phi),
            vy + _CLF * np.cos(phi))


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

# Separation parameters — both used in approach and departure.
_SEP_FAR  = 0.95 * (l1 + l2 + l3)   # assembly separation at animation start/end (≈ 0.950 m)
_SEP_NEAR = l2                         # assembly separation when near the corner  (= 0.500 m)

# Approach: front starts _SEP_NEAR past the corner; back starts _SEP_FAR behind front.
_OUT_X2_NEAR  = 0.50                              # back hold x while front rotates
_OUT_X1_START = _XSTP  + _SEP_NEAR               # front start x (= _OUT_X2_NEAR)
_OUT_X2_START = _OUT_X1_START + _SEP_NEAR    # back start x  = _OUT_X1_START + 0.95*L
# Back moves toward corner during front rotation to give the front more reach.
_OUT_X2_ROT_END  = 0.300                          # back x at end of front rotation
_OUT_X2_ROT_STEP = (_OUT_X2_ROT_END - _OUT_X2_NEAR) / 6  # per-step decrement (= −0.03333 m)

# Phase timing (all derived from ω constraint and sep constants).
_OUT_T_FRONT_STOP = _dt_slide(_SEP_FAR, _SEP_NEAR)          # approach: back moves _SEP_FAR
_OUT_ROT_DT       = 0.035                                    # 0.025 causes 10°/fr q1 jump during rotation
_OUT_T_ROT1_S     = _OUT_T_FRONT_STOP + 0.025               # pause → front rotation start
_OUT_T_ROT1_E     = _OUT_T_ROT1_S     + 6 * _OUT_ROT_DT    # front rotation end (6 × 15°)
_OUT_T_BOTH_S     = _OUT_T_ROT1_E     + 0.025               # pause → both-moving start
_OUT_T_BACK_ARR   = _OUT_T_BOTH_S     + _dt_slide(_OUT_X2_ROT_END - _XSTP2)  # back travels from ROT_END
_OUT_T_ROT2_E     = _OUT_T_BACK_ARR   + 6 * _OUT_ROT_DT    # back rotation end (no pause3)

# Departure: both descend together maintaining constant separation.
_OUT_Y1_FINAL = _YW2 - _SEP_FAR   # front end y  (= _YW2 − 0.95L ≈ −0.400)


def _y1_descent(t):
    """Front chain-joint y: linear descent from _YW at _OUT_T_BOTH_S to _OUT_Y1_FINAL at t=1.0."""
    frac = (t - _OUT_T_BOTH_S) / (1.0 - _OUT_T_BOTH_S)
    return _YW + frac * (_OUT_Y1_FINAL - _YW)


_SEP_ROT2E    = _YW2 - _y1_descent(_OUT_T_ROT2_E)   # actual y-separation when both land on wall
_OUT_Y2_FINAL = _OUT_Y1_FINAL + _SEP_ROT2E           # back descends same Δy as front


KEYFRAMES_OUTSIDE = [
    # Approach — both flat (theta=0), sep closes from _SEP_FAR to _SEP_NEAR.
    # Front travels _SEP_NEAR; back travels _SEP_FAR (faster, closes gap).
    {"t": 0.000,
     "x1": _OUT_X1_START, "y1": _Y0, "theta1": 0.0,
     "x2": _OUT_X2_START, "y2": _Y0, "theta2": 0.0,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": _OUT_T_FRONT_STOP,
     "x1": _XSTP,         "y1": _Y0, "theta1": 0.0,
     "x2": _OUT_X2_NEAR,  "y2": _Y0, "theta2": 0.0,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Pause before front rotation.
    {"t": _OUT_T_ROT1_S,
     "x1": _XSTP,         "y1": _Y0, "theta1": 0.0,
     "x2": _OUT_X2_NEAR,  "y2": _Y0, "theta2": 0.0,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Front rotation: phi 0 → π/2 (6 × 15° × Δt=_OUT_ROT_DT).  Back inches from _OUT_X2_NEAR to _OUT_X2_ROT_END.
    {"t": _OUT_T_ROT1_S + 1*_OUT_ROT_DT,
     "x1": _cj(    np.pi / 12)[0], "y1": _cj(    np.pi / 12)[1], "theta1":     np.pi / 12,
     "x2": _OUT_X2_NEAR + 1*_OUT_X2_ROT_STEP, "y2": _Y0, "theta2": 0.0,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": _OUT_T_ROT1_S + 2*_OUT_ROT_DT,
     "x1": _cj(    np.pi /  6)[0], "y1": _cj(    np.pi /  6)[1], "theta1":     np.pi /  6,
     "x2": _OUT_X2_NEAR + 2*_OUT_X2_ROT_STEP, "y2": _Y0, "theta2": 0.0,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": _OUT_T_ROT1_S + 3*_OUT_ROT_DT,
     "x1": _cj(    np.pi /  4)[0], "y1": _cj(    np.pi /  4)[1], "theta1":     np.pi /  4,
     "x2": _OUT_X2_NEAR + 3*_OUT_X2_ROT_STEP, "y2": _Y0, "theta2": 0.0,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": _OUT_T_ROT1_S + 4*_OUT_ROT_DT,
     "x1": _cj(    np.pi /  3)[0], "y1": _cj(    np.pi /  3)[1], "theta1":     np.pi /  3,
     "x2": _OUT_X2_NEAR + 4*_OUT_X2_ROT_STEP, "y2": _Y0, "theta2": 0.0,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": _OUT_T_ROT1_S + 5*_OUT_ROT_DT,
     "x1": _cj(5 * np.pi / 12)[0], "y1": _cj(5 * np.pi / 12)[1], "theta1": 5 * np.pi / 12,
     "x2": _OUT_X2_NEAR + 5*_OUT_X2_ROT_STEP, "y2": _Y0, "theta2": 0.0,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Front rotation done (phi=π/2).  Pause before both-moving.
    {"t": _OUT_T_ROT1_E,
     "x1": _XW, "y1": _YW, "theta1": np.pi / 2,
     "x2": _OUT_X2_ROT_END, "y2": _Y0, "theta2": 0.0,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": _OUT_T_BOTH_S,
     "x1": _XW, "y1": _YW, "theta1": np.pi / 2,
     "x2": _OUT_X2_ROT_END, "y2": _Y0, "theta2": 0.0,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Both moving: front descends linearly; back approaches _XSTP2 (travels _SEP_NEAR).
    {"t": _OUT_T_BACK_ARR,
     "x1": _XW, "y1": _y1_descent(_OUT_T_BACK_ARR), "theta1": np.pi / 2,
     "x2": _XSTP2, "y2": _Y0, "theta2": 0.0,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Back rotation: phi 0 → π/2 (6 × 15° × Δt=_OUT_ROT_DT).  No pause — rotation starts immediately.
    {"t": _OUT_T_BACK_ARR + 1*_OUT_ROT_DT,
     "x1": _XW, "y1": _y1_descent(_OUT_T_BACK_ARR + 1*_OUT_ROT_DT), "theta1": np.pi / 2,
     "x2": _cj2(    np.pi / 12)[0], "y2": _cj2(    np.pi / 12)[1], "theta2":     np.pi / 12,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": _OUT_T_BACK_ARR + 2*_OUT_ROT_DT,
     "x1": _XW, "y1": _y1_descent(_OUT_T_BACK_ARR + 2*_OUT_ROT_DT), "theta1": np.pi / 2,
     "x2": _cj2(    np.pi /  6)[0], "y2": _cj2(    np.pi /  6)[1], "theta2":     np.pi /  6,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": _OUT_T_BACK_ARR + 3*_OUT_ROT_DT,
     "x1": _XW, "y1": _y1_descent(_OUT_T_BACK_ARR + 3*_OUT_ROT_DT), "theta1": np.pi / 2,
     "x2": _cj2(    np.pi /  4)[0], "y2": _cj2(    np.pi /  4)[1], "theta2":     np.pi /  4,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": _OUT_T_BACK_ARR + 4*_OUT_ROT_DT,
     "x1": _XW, "y1": _y1_descent(_OUT_T_BACK_ARR + 4*_OUT_ROT_DT), "theta1": np.pi / 2,
     "x2": _cj2(    np.pi /  3)[0], "y2": _cj2(    np.pi /  3)[1], "theta2":     np.pi /  3,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    {"t": _OUT_T_BACK_ARR + 5*_OUT_ROT_DT,
     "x1": _XW, "y1": _y1_descent(_OUT_T_BACK_ARR + 5*_OUT_ROT_DT), "theta1": np.pi / 2,
     "x2": _cj2(5 * np.pi / 12)[0], "y2": _cj2(5 * np.pi / 12)[1], "theta2": 5 * np.pi / 12,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Back rotation done — fully on wall.  sep = y2 − y1 ≈ _SEP_NEAR here.
    {"t": _OUT_T_ROT2_E,
     "x1": _XW, "y1": _y1_descent(_OUT_T_ROT2_E), "theta1": np.pi / 2,
     "x2": _XW, "y2": _YW2, "theta2": np.pi / 2,
     "constraint_set": _CSE, "wall_x": WALL_X, "ceiling_y": CEILING_Y},
    # Departure: both descend together at constant separation _SEP_ROT2E.
    {"t": 1.000,
     "x1": _XW, "y1": _OUT_Y1_FINAL, "theta1": np.pi / 2,
     "x2": _XW, "y2": _OUT_Y2_FINAL, "theta2": np.pi / 2,
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
    return (vx + wg.calf * np.sin(phi),
            vy + wg.calf * np.cos(phi))


def _cj_te2(phi):
    """Chain-joint for side=-1 assembly during CW pivot. theta=−phi."""
    vx, vy = _cp_te(phi)
    return (vx + wg.calf * np.sin(phi),
            vy + wg.calf * np.cos(phi))


_XSTP_TE  = _cj_te(0)[0]    # green stop: arm_a1 wheel at terminus
_Y0_TE    = _cj_te(0)[1]    # chain-joint y when assembly sits on edge top at theta=0
_XSTP_red = _cj_te2(0)[0]   # red stop: arm_a2 wheel at terminus
# Green intermediate stop: arm_b1 wheel at terminus (arm_b1 uses bar_len, not bar_len_a).
_XW_TE    = _cj_te(np.pi)[0]
_YW_TE    = _cj_te(np.pi)[1]
_XW_red   = _cj_te2(np.pi)[0]
_YW_red   = _cj_te2(np.pi)[1]

_CTE = "thin_edge_exact"

# Red retreat bounds during green rotation (Phase 4).
_TE_X2_RETREAT_S = 0.375   # red x2 when green rotation begins
_TE_X2_RETREAT_E = -0.156  # red x2 when green rotation ends (11 steps later)

KEYFRAMES_THIN_EDGE = [
    # Phase 0 — both approach on edge top (theta=0), moving right.
    # Red (x2, assembly 2) leads; green (x1, assembly 1) trails.
    {"t": 0.00, "x1": -0.575, "y1": _Y0_TE, "theta1": 0.0,
                "x2": -0.075, "y2": _Y0_TE, "theta2": 0.0,
                "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    # Phase 1 — red stops (left wheel above terminus); green still approaching.
    {"t": 0.20, "x1":  -0.06,    "y1": _Y0_TE, "theta1": 0.0,
                "x2": _XSTP_red, "y2": _Y0_TE, "theta2": 0.0,
                "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    # Phase 2 — red pivot (CW: theta2: 0 → −π), 12 steps × 15° × Δt=0.025.
    # Green (x1) holds back to prevent wheel overlap with red's rotating body.
    # Red's arm_a2 pivot wheel is fixed at (EDGE_X, EDGE_Y+wheel_r)=(0.44,0.55)
    # throughout its rotation; green's arm_b1 wheel must stay ≥0.10m away (2×wheel_r).
    # Constraint: x1 ≤ 0.250 (arm_b1 wheel at x=0.302, dist=0.138 to pivot).
    # Early frames also need x1 reduced to clear red's swinging arm_b2 wheel.
    {"t": 0.232, "x1":  -0.06,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2( np.pi/12)[0], "y2": _cj_te2( np.pi/12)[1],
                 "theta2": -np.pi / 12,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.264, "x1":  -0.06,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(  np.pi/6)[0], "y2": _cj_te2(  np.pi/6)[1],
                 "theta2": -np.pi / 6,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.296, "x1":  -0.06,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(  np.pi/4)[0], "y2": _cj_te2(  np.pi/4)[1],
                 "theta2": -np.pi / 4,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.321, "x1":  0,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(  np.pi/3)[0], "y2": _cj_te2(  np.pi/3)[1],
                 "theta2": -np.pi / 3,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.346, "x1":  0.05,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(5*np.pi/12)[0], "y2": _cj_te2(5*np.pi/12)[1],
                 "theta2": -5 * np.pi / 12,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.371, "x1":  0.1,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(  np.pi/2)[0], "y2": _cj_te2(  np.pi/2)[1],
                 "theta2": -np.pi / 2,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.396, "x1":  0.15,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(7*np.pi/12)[0], "y2": _cj_te2(7*np.pi/12)[1],
                 "theta2": -7 * np.pi / 12,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.421, "x1":  0.2,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(2*np.pi/3)[0], "y2": _cj_te2(2*np.pi/3)[1],
                 "theta2": -2 * np.pi / 3,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.446, "x1":  0.25,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(3*np.pi/4)[0], "y2": _cj_te2(3*np.pi/4)[1],
                 "theta2": -3 * np.pi / 4,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.471, "x1":  0.3,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(5*np.pi/6)[0], "y2": _cj_te2(5*np.pi/6)[1],
                 "theta2": -5 * np.pi / 6,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    {"t": 0.496, "x1":  0.35,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _cj_te2(11*np.pi/12)[0], "y2": _cj_te2(11*np.pi/12)[1],
                 "theta2": -11 * np.pi / 12,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    # Red pivot done; green waits at x1=0.25 until safe to advance.
    {"t": 0.521, "x1":  0.4,    "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _XW_red,    "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    # Phase 3 — stagger: green rolls to pivot start.
    # cold_at=0.61 fires between t=0.616 and t=0.641.
    {"t": 0.616, "x1": _XSTP_TE,  "y1": _Y0_TE, "theta1": 0.0,
                 "x2": _TE_X2_RETREAT_S, "y2": _YW_red, "theta2": -np.pi,
                 "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
    # Phase 4 — green pivot (CW: theta1: 0 → −π), 12 steps × 15° × Δt=0.025.
    # Red retreats from _TE_X2_RETREAT_S to _TE_X2_RETREAT_E in 11 equal steps.
    *[{"t": 0.641 + k * 0.025,
       "x1": _cj_te((k + 1) * np.pi / 12)[0], "y1": _cj_te((k + 1) * np.pi / 12)[1],
       "theta1": -(k + 1) * np.pi / 12,
       "x2": _TE_X2_RETREAT_S + k * (_TE_X2_RETREAT_E - _TE_X2_RETREAT_S) / 11,
       "y2": _YW_red, "theta2": -np.pi,
       "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y}
      for k in range(12)],
    # Phase 5 — both below edge, moving left.
    {"t": 1.00, "x1":  0.20,   "y1": _YW_TE, "theta1": -np.pi,
                "x2": -0.40,   "y2": _YW_red, "theta2": -np.pi,
                "constraint_set": _CTE, "wall_x": EDGE_X, "ceiling_y": EDGE_Y},
]

# ── Solver config and render metadata per scenario ───────────────────────────
SCENARIO_CONFIG = {
    "floor_to_wall": {
        "keyframes":     KEYFRAMES_FLOOR_TO_WALL,
        "wg":            wg,
        "n_grid":        120,
        "smooth_weight": 20.0,
        "cold_at":       None,
        "xlim":          (-0.2, 2.2),
        "ylim":          (-0.2, 1.5),
        "output":        "floor_to_wall.mp4",
        "label":         "Floor-to-wall",
    },
    "wall_to_ceiling": {
        "keyframes":     KEYFRAMES_WALL_TO_CEILING,
        "wg":            wg,
        "n_grid":        60,
        "smooth_weight": 100.0,
        "cold_at":       None,
        "xlim":          (-0.5, 1.75),
        "ylim":          (-0.5, 2.0),
        "output":        "wall_to_ceiling.mp4",
        "label":         "Wall-to-ceiling",
    },
    "outside": {
        "keyframes":     KEYFRAMES_OUTSIDE,
        "wg":            wg_outside,
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
        "wg":            wg_thin_edge,
        "n_grid":        60,
        "smooth_weight": 30.0,
        "cold_at":       None,
        "xlim":          (-0.7, 1.2),
        "ylim":          (-0.35, 1.3),
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
    _SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    output        = os.path.join(_SCRIPT_DIR, "gifs", cfg["output"])
    os.makedirs(os.path.dirname(output), exist_ok=True)
    label         = cfg["label"]
    FPS           = 15
    N_GRID        = cfg["n_grid"]
    SMOOTH_WEIGHT = cfg["smooth_weight"]

    print(f"{label}  ({N_FRAMES} frames @ {FPS} fps)")
    print("-" * 60)
    frames = solve_trajectory(keyframes, cfg["wg"], l1, l2, l3, N_FRAMES, N_GRID,
                              smooth_weight=SMOOTH_WEIGHT, cold_at=cfg["cold_at"],
                              seed_at=cfg.get("seed_at"))
    print("-" * 60)
    # ---- joint-angle limit + jump + sign + ω check ------------------- #
    _JUMP_THRESH  = np.radians(7.2)
    _SOFT_DISP    = np.radians(95.0)   # ±95° display soft limit
    _LIMIT_DISP   = np.radians(105.0)  # ±105° display hard limit
    _OMEGA_MAX    = _OMEGA_LIM
    _WHEEL_R      = cfg["wg"].wheel_r
    _issues = []
    _names  = ["q1", "q2", "q3", "q4"]
    # Limit + sign checks (per frame)
    for _i, (_q, _, _p) in enumerate(frames):
        _qd = _q_display(_q)
        for _jidx in range(4):
            _qd_abs = abs(float(_qd[_jidx]))
            if _qd_abs > _LIMIT_DISP:
                _issues.append(f"  LIMIT   fr{_i:02d}  {_names[_jidx]}={np.degrees(float(_qd[_jidx])):+.1f}°")
            elif _qd_abs > _SOFT_DISP:
                _issues.append(f"  SOFT    fr{_i:02d}  {_names[_jidx]}={np.degrees(float(_qd[_jidx])):+.1f}°")
        for _jidx in (1, 2):           # q2, q3 must be <= 0 (internal = display)
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
        print("Check: CLEAN (no limits, signs, jumps≥7.2°, or ω violations)")
    else:
        for _msg in _issues:
            print(_msg)
        print(f"  → {len(_issues)} issue(s) total")
    # ------------------------------------------------------------------ #
    if "--no-render" not in sys.argv:
        print("Rendering...")
        render_video(frames, cfg["wg"], l1, l2, l3, xlim, ylim, output, FPS)
