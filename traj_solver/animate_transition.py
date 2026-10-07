"""
animate_transition.py — solve a transition's joint angles, check them, render them.

Pipeline
--------
  scenarios.py       geometry and the five phases of each transition
                     (approach, pivot, both move, pivot, depart)
  solve_trajectory   samples the phases at timing.SAMPLE_RATE, fixes both ends
                     to the cruise posture, and lets trajectory_ik choose every
                     frame's joint angles together.  If a joint would have to
                     move faster than the limit, that phase is slowed down and
                     the trajectory is solved again.
  render_video       GIF/MP4 of the solved transition
  __main__           solve, run the built-in check (joint limits, bend
                     direction, joint speed, wheel speed) and render

Usage:
    python animate_transition.py [scenario] [--no-render]
        scenario      floor_to_wall | wall_to_ceiling | outside | thin_edge
                      (default wall_to_ceiling)
        --no-render   solve and check only
"""

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constraints import wheel_centers
from kinematics import forward_kinematics, q_display
from scenarios import (SCENARIOS, Scenario, l1, l2, l3, plan_phases, phase_starts,
                       sample_poses, cruise_posture, wheel_spin)
from timing import SAMPLE_RATE, REFERENCE_RATE, PLAN_JOINT_SPEED, MAX_JOINT_SPEED, SLIDE_SPEED
from trajectory_ik import solve_trajectory_dp
from visualize import plot_robot


# Weight on joint motion between consecutive frames, per frame at
# timing.REFERENCE_RATE (rescaled to the actual rate).  The same for every
# transition.
SMOOTH_WEIGHT = 50.0

# A phase is slowed down when a joint exceeds PLAN_JOINT_SPEED by more than
# RETIME_TOLERANCE in it; the trajectory is then solved again, at most
# MAX_RETIMES times.
RETIME_TOLERANCE = 0.005
MAX_RETIMES = 5


# ------------------------------------------------------------------ #
#  Solving                                                            #
# ------------------------------------------------------------------ #

@dataclass
class Trajectory:
    """
    A solved transition, one entry per frame.

    times    (N,) seconds
    poses    assembly poses per frame (dicts, see scenarios.sample_poses)
    Q        (N, 4) joint angles (rad)
    infos    per-frame diagnostics (trajectory_ik.solve_trajectory_dp)
    phases   the five phases with their final durations
    retimes  how many times phases were slowed down for the joint speed limit

    Iterating yields (q, info, pose) for each frame.
    """
    scenario: Scenario
    phases: list
    times: np.ndarray
    poses: list
    Q: np.ndarray
    infos: list
    retimes: int

    @property
    def rate(self):
        """Frames per second."""
        return (len(self.times) - 1) / self.times[-1]

    @property
    def phase_starts(self):
        """Start time of each phase plus the end time."""
        return phase_starts(self.phases)

    def phase_at(self, t):
        """Index of the phase running at time t."""
        return min(int(np.searchsorted(self.phase_starts, t, side="right")) - 1,
                   len(self.phases) - 1)

    def __len__(self):
        return len(self.times)

    def __iter__(self):
        return iter(zip(self.Q, self.infos, self.poses))


def _phase_peak_speeds(times, Q, phases):
    """Fastest joint speed (rad/s) within each phase.  Each step between two
    frames counts toward the phase that contains its midpoint."""
    D = np.unwrap(np.array([q_display(q) for q in Q]), axis=0)
    speed = np.abs(np.diff(D, axis=0)).max(axis=1) / np.diff(times)
    mid = 0.5 * (times[1:] + times[:-1])
    starts = phase_starts(phases)
    return np.array([speed[(mid > starts[i]) & (mid < starts[i + 1])].max(initial=0.0)
                     for i in range(len(phases))])


def solve_trajectory(scn, rate=SAMPLE_RATE):
    """
    Plan the transition `scn`, sample it at `rate` frames per second and solve
    every frame's joint angles together, with both ends fixed to the cruise
    posture.  Returns a Trajectory.

    The phases are first timed from the wheel and pivot speeds alone
    (scenarios.plan_phases).  If the solved joints must move faster than
    timing.PLAN_JOINT_SPEED in some phase, that phase is lengthened in
    proportion and the trajectory is solved again.
    """
    min_durations = None
    for retimes in range(MAX_RETIMES + 1):
        phases = plan_phases(scn, min_durations)
        times, poses = sample_poses(scn, phases, rate)
        frame_rate = (len(times) - 1) / times[-1]
        Q, infos = solve_trajectory_dp(
            poses, scn.wg, l1, l2, l3,
            max_step=PLAN_JOINT_SPEED / frame_rate,
            # Joint steps shrink in proportion to the frame interval, so the
            # per-frame weight scales with rate² to keep the same balance.
            smooth_weight=SMOOTH_WEIGHT * (frame_rate / REFERENCE_RATE) ** 2,
            q_start=cruise_posture(scn, poses[0]),
            q_end=cruise_posture(scn, poses[-1]),
        )
        peaks = _phase_peak_speeds(times, Q, phases)
        too_fast = peaks > PLAN_JOINT_SPEED * (1 + RETIME_TOLERANCE)
        if not too_fast.any() or retimes == MAX_RETIMES:
            break
        min_durations = [p.duration * (pk / PLAN_JOINT_SPEED if fast else 1.0)
                         for p, pk, fast in zip(phases, peaks, too_fast)]
    return Trajectory(scn, phases, times, poses, Q, infos, retimes)


# ------------------------------------------------------------------ #
#  Drawing                                                            #
# ------------------------------------------------------------------ #

# Animation playback: GIF_SPEEDUP × real time, drawing every k-th frame so the
# animation stays at or below GIF_MAX_FPS.
GIF_SPEEDUP = 5.0
GIF_MAX_FPS = 15.0


def plot_limits(traj, margin=0.15):
    """Axis limits that contain the chain and wheels in every frame, plus a margin."""
    scn = traj.scenario
    pts = [np.asarray(scn.point, dtype=float)]
    for q, _, p in traj:
        chain, _ = forward_kinematics(q, p["x1"], p["y1"], p["theta1"], l1, l2, l3)
        pts.extend(chain)
        pts.extend(wheel_centers(p["x1"], p["y1"], p["theta1"], scn.wg, side=1))
        pts.extend(wheel_centers(p["x2"], p["y2"], p["theta2"], scn.wg, side=-1))
    pts = np.array(pts)
    lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
    return (lo[0], hi[0]), (lo[1], hi[1])


def draw_surfaces(ax, scn, xlim, ylim, marker_scale=1.0):
    """Draw the transition's environment (the surfaces of its constraint set)."""
    cs, wx, cy = scn.constraint_set, scn.wall_x, scn.ceiling_y
    if cs == "wall":                                   # floor + wall
        ax.plot([wx, xlim[1]], [0, 0], color="saddlebrown", lw=2, label="Floor")
        ax.plot([wx, wx], [0, ylim[1]], color="slategray", lw=2, label=f"Wall (x={wx})")
    elif cs == "ceiling":                              # wall + ceiling
        ax.plot([wx, wx], [ylim[0], cy], color="slategray", lw=2, label=f"Wall (x={wx})")
        ax.plot([wx, xlim[1]], [cy, cy], color="dimgray", lw=2, label=f"Ceiling (y={cy})")
    elif cs == "outside":                              # solid block: top + exterior wall
        ax.add_patch(Rectangle((wx, ylim[0]), xlim[1] - wx, cy - ylim[0],
                               facecolor="lightgray", alpha=0.45, zorder=0, label="Solid block"))
        ax.plot([wx, xlim[1]], [cy, cy], color="dimgray", lw=2, label=f"Top (y={cy})")
        ax.plot([wx, wx], [ylim[0], cy], color="slategray", lw=2, label=f"Exterior wall (x={wx})")
    elif cs == "thin_edge":                            # edge ending at (wall_x, ceiling_y)
        ax.plot([xlim[0], wx], [cy, cy], color="saddlebrown", lw=3, label=f"Thin edge (y={cy})")
        ax.plot(wx, cy, "D", color="saddlebrown", ms=8 * marker_scale, label=f"Edge end (x={wx})")


def draw_frame(ax, traj, k, xlim, ylim, title=None, marker_scale=1.0, show_labels=True):
    """Draw frame k of a trajectory: environment, chain and both assemblies."""
    wg = traj.scenario.wg
    q, p = traj.Q[k], traj.poses[k]
    draw_surfaces(ax, traj.scenario, xlim, ylim, marker_scale)
    plot_robot(q, p["x1"], p["y1"], p["theta1"], p["x2"], p["y2"], p["theta2"], l1, l2, l3,
               axle_half_length=wg.bar_len, axle_a_len=wg.bar_len_a, wheel_radius=wg.wheel_r,
               calf=wg.calf, arm_a1=wg.arm_a1, arm_b1=wg.arm_b1, arm_a2=wg.arm_a2, arm_b2=wg.arm_b2,
               ax=ax, title=title, marker_scale=marker_scale, show_labels=show_labels)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")


def render_video(traj, output_file="transition.mp4", speedup=GIF_SPEEDUP, max_fps=GIF_MAX_FPS):
    """
    Render a trajectory to MP4 (if ffmpeg is available) or GIF.

    Plays at `speedup` × real time; every k-th frame is drawn so playback stays
    at or below max_fps, and the final frame is always drawn.
    """
    stride = max(1, math.ceil(traj.rate * speedup / max_fps - 1e-9))
    shown = list(range(0, len(traj), stride))
    if shown[-1] != len(traj) - 1:
        shown.append(len(traj) - 1)
    fps = traj.rate * speedup / stride
    xlim, ylim = plot_limits(traj)

    fig, ax = plt.subplots(figsize=(10, 7))

    def draw(i):
        ax.cla()
        k = shown[i]
        t = traj.times[k]
        qd = np.degrees(q_display(traj.Q[k]))
        draw_frame(ax, traj, k, xlim, ylim)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{traj.scenario.label} — t = {t:4.1f} s ({traj.phases[traj.phase_at(t)].name})\n"
                     + "  ".join(f"q{j + 1}={v:+.1f}°" for j, v in enumerate(qd)), fontsize=9)

    anim = animation.FuncAnimation(fig, draw, frames=len(shown), interval=1000 / fps)
    saved = False
    if output_file.endswith(".mp4"):
        try:
            anim.save(output_file, writer=animation.FFMpegWriter(fps=fps, bitrate=2000), dpi=120)
            print(f"\nSaved -> {output_file}")
            saved = True
        except Exception:
            pass
    if not saved:
        gif_file = os.path.splitext(output_file)[0] + ".gif"
        anim.save(gif_file, writer=animation.PillowWriter(fps=fps), dpi=100)
        print(f"\nSaved -> {gif_file}  (install ffmpeg for MP4)")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Built-in check                                                     #
# ------------------------------------------------------------------ #

SOFT_DEG  = 95.0    # display-convention joint angle that triggers a warning
LIMIT_DEG = 105.0   # display-convention hard joint limit


def check(traj):
    """
    Problems a playback on the robot would hit:
      LIMIT  joint beyond ±105°          SOFT   joint beyond ±95° (warning)
      SIGN   q2 or q3 bending the wrong way (> 0)
      JUMP   joint step at or above the joint speed limit
      OMEGA  rolling wheel faster than SLIDE_SPEED
    """
    issues = []
    names = ["q1", "q2", "q3", "q4"]
    jump = MAX_JOINT_SPEED / traj.rate                       # rad per frame
    omega_max = SLIDE_SPEED / traj.scenario.wg.wheel_r * 1.01
    for k, (q, _, _) in enumerate(traj):
        t = traj.times[k]
        for j, v in enumerate(np.degrees(q_display(q))):
            if abs(v) > LIMIT_DEG:
                issues.append(f"  LIMIT   t={t:5.2f}s  {names[j]}={v:+.1f}°")
            elif abs(v) > SOFT_DEG:
                issues.append(f"  SOFT    t={t:5.2f}s  {names[j]}={v:+.1f}°")
        for j in (1, 2):
            if q[j] > 0.0:
                issues.append(f"  SIGN    t={t:5.2f}s  {names[j]}={np.degrees(q[j]):+.2f}°")
    for k in range(1, len(traj)):
        d = (traj.Q[k] - traj.Q[k - 1] + np.pi) % (2 * np.pi) - np.pi
        for j in range(4):
            if abs(d[j]) >= jump:
                issues.append(f"  JUMP    t={traj.times[k]:5.2f}s  {names[j]}  Δ={np.degrees(abs(d[j])):.2f}°"
                              f" ({np.degrees(abs(d[j])) * traj.rate:.1f}°/s)")
    spin = wheel_spin(traj.scenario, traj.poses, traj.times)
    for k, j in zip(*np.nonzero(np.abs(spin) > omega_max)):
        issues.append(f"  OMEGA   t={traj.times[k]:5.2f}s  wheel {j}  ω={spin[k, j]:+.2f} rad/s")
    return issues


if __name__ == "__main__":
    name = next((a for a in sys.argv[1:] if not a.startswith("--")), "wall_to_ceiling")
    if name not in SCENARIOS:
        print(f"Unknown scenario '{name}'. Valid: {list(SCENARIOS)}")
        sys.exit(1)
    scn = SCENARIOS[name]

    traj = solve_trajectory(scn)
    print(f"{scn.label}: {traj.times[-1]:.1f} s, {len(traj)} frames at {traj.rate:g} Hz"
          f" (animation at {GIF_SPEEDUP:g}× real time)")
    for phase, start in zip(traj.phases, traj.phase_starts):
        print(f"  {start:5.1f} s  {phase.name:13s} {phase.duration:5.2f} s")
    if traj.retimes:
        print(f"  ({traj.retimes} phase slow-down(s) to keep the joints under "
              f"{np.degrees(PLAN_JOINT_SPEED):.1f}°/s)")
    print("-" * 60)

    issues = check(traj)
    if not issues:
        print(f"Check: CLEAN (no limits, signs, jumps≥{np.degrees(MAX_JOINT_SPEED / traj.rate):.1f}°,"
              f" or ω violations)")
    else:
        for msg in issues:
            print(msg)
        print(f"  → {len(issues)} issue(s) total")

    if "--no-render" not in sys.argv:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gifs")
        os.makedirs(out_dir, exist_ok=True)
        print("Rendering...")
        render_video(traj, os.path.join(out_dir, scn.output))
