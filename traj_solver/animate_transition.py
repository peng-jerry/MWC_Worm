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
  render_video       GIF/MP4 of the solved transition, with every part drawn
                     at its real size (constraints.py outline)
  __main__           solve, run the built-in check (hard rules, joint
                     limits, bend direction, joint speed, wheel speed) and
                     render

Usage:
    python animate_transition.py [scenario] [--no-render]
        scenario      floor_to_wall | wall_to_ceiling | outside | thin_edge
                      (default wall_to_ceiling)
        --no-render   solve and check only
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constraints import (JOINT_HARD_LIMIT_DEG, JOINT_SOFT_LIMIT_DEG, JOINT_RADIUS,
                         assembly_parts, wheel_centers)
from kinematics import forward_kinematics, q_display
from scenarios import (SCENARIOS, Scenario, CHAIN_LENGTH, l1, l2, l3, plan_phases, phase_starts,
                       sample_poses, cruise_posture, follow_gap, wheel_spin)
from timing import SAMPLE_RATE, PLAN_JOINT_SPEED, MAX_JOINT_SPEED, SLIDE_SPEED
from trajectory_ik import solve_trajectory_dp
from visualize import plot_robot


# A phase is slowed down when a joint exceeds PLAN_JOINT_SPEED by more than
# RETIME_TOLERANCE in it; the trajectory is then solved again, at most
# MAX_RETIMES times.
RETIME_TOLERANCE = 0.005
MAX_RETIMES = 5


def _wrap(a):
    """Wrap angles to [-π, π)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


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
    gap      gap (m) the partner keeps while following a pivot, or None when
             it parks (scenarios.follow_gap)

    Iterating yields (q, info, pose) for each frame.
    """
    scenario: Scenario
    phases: list
    times: np.ndarray
    poses: list
    Q: np.ndarray
    infos: list
    retimes: int
    gap: Optional[float]

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

    At an outside corner or edge the follow gap is chosen first
    (scenarios.follow_gap).  The phases are then timed from the wheel and
    pivot speeds alone (scenarios.plan_phases).  If the solved joints must
    move faster than timing.PLAN_JOINT_SPEED in some phase, that phase is
    lengthened in proportion and the trajectory is solved again.
    """
    gap = follow_gap(scn, rate) if scn.partner_follows else None
    min_durations = None
    for retimes in range(MAX_RETIMES + 1):
        phases = plan_phases(scn, min_durations, gap)
        times, poses = sample_poses(scn, phases, rate)
        frame_rate = (len(times) - 1) / times[-1]
        Q, infos = solve_trajectory_dp(
            poses, scn.env, scn.wg, l1, l2, l3,
            dt=1.0 / frame_rate,
            max_step=PLAN_JOINT_SPEED / frame_rate,
            q_start=cruise_posture(scn, poses[0]),
            q_end=cruise_posture(scn, poses[-1]),
        )
        peaks = _phase_peak_speeds(times, Q, phases)
        too_fast = peaks > PLAN_JOINT_SPEED * (1 + RETIME_TOLERANCE)
        if not too_fast.any() or retimes == MAX_RETIMES:
            break
        min_durations = [p.duration * (pk / PLAN_JOINT_SPEED if fast else 1.0)
                         for p, pk, fast in zip(phases, peaks, too_fast)]
    return Trajectory(scn, phases, times, poses, Q, infos, retimes, gap)


# ------------------------------------------------------------------ #
#  Drawing                                                            #
# ------------------------------------------------------------------ #

# Animation playback: every GIF_FRAME_STEP-th solved frame is drawn, one every
# GIF_FRAME_MS.  Speed = GIF_FPS × GIF_FRAME_STEP / sample rate; at 6 fps that
# is 20 × 1 / 6 ≈ 3.3× real time.  GIF frame delays are stored in whole
# hundredths of a second, so GIF_FRAME_MS must be a multiple of 10.
GIF_FRAME_MS = 50
GIF_FPS = 1000 / GIF_FRAME_MS
GIF_FRAME_STEP = 1


class _GifWriter(animation.PillowWriter):
    """
    PillowWriter with an exact frame delay.  The stock writer passes
    int(1000 / fps) ms, which Pillow truncates to whole hundredths of a
    second, so e.g. 1000/60 fps (59.999… ms) would be saved as 50 ms.  This
    rounds the delay to the nearest hundredth instead.
    """
    def finish(self):
        self._frames[0].save(self.outfile, save_all=True, append_images=self._frames[1:],
                             duration=10 * round(100 / self.fps), loop=0)


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


def playback_speed(traj, fps=GIF_FPS, frame_step=GIF_FRAME_STEP):
    """How many times faster than real time the animation plays."""
    return fps * frame_step / traj.rate


def render_video(traj, output_file="transition.mp4", fps=GIF_FPS, frame_step=GIF_FRAME_STEP):
    """
    Render a trajectory to MP4 (if ffmpeg is available) or GIF.

    Draws every `frame_step`-th solved frame (always including the last) and
    shows them at `fps` frames per second.  Only solved frames are drawn,
    never states in between.
    """
    shown = list(range(0, len(traj), frame_step))
    if shown[-1] != len(traj) - 1:
        shown.append(len(traj) - 1)
    xlim, ylim = plot_limits(traj)
    speed = playback_speed(traj, fps, frame_step)

    fig, ax = plt.subplots(figsize=(10, 7))

    def draw(i):
        ax.cla()
        k = shown[i]
        t = traj.times[k]
        q = traj.Q[k]
        qd = np.degrees(q_display(q))
        draw_frame(ax, traj.scenario, q, traj.poses[k], xlim, ylim)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{traj.scenario.label} — t = {t:4.1f} s ({traj.phases[traj.phase_at(t)].name})"
                     f"  ({speed:.1f}× real time)\n"
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
        anim.save(gif_file, writer=_GifWriter(fps=fps), dpi=100)
        print(f"\nSaved -> {gif_file}  (install ffmpeg for MP4)")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Built-in check                                                     #
# ------------------------------------------------------------------ #

def check(traj):
    """
    Problems a playback on the robot would hit (limits from constraints.py):
      HARD   a hard rule broken (clearance, joint limit or bend direction)
      LIMIT  joint beyond the hard limit (±110°)
      SOFT   joint beyond the soft limit (±90°, warning)
      SIGN   q2 or q3 bending the wrong way (> 0)
      JUMP   joint step at or above the joint speed limit
      OMEGA  rolling wheel faster than SLIDE_SPEED
    """
    issues = []
    names = ["q1", "q2", "q3", "q4"]
    jump = MAX_JOINT_SPEED / traj.rate                       # rad per frame
    omega_max = SLIDE_SPEED / traj.scenario.wg.wheel_r * 1.01
    for k, (q, info, _) in enumerate(traj):
        t = traj.times[k]
        for rule, amount in info["violations"].items():
            unit = "°" if rule in ("joint limit", "bend") else " mm"
            value = np.degrees(amount) if unit == "°" else amount * 1000
            issues.append(f"  HARD    t={t:5.2f}s  {rule} by {value:.1f}{unit}")
        for j, v in enumerate(np.degrees(q_display(q))):
            if abs(v) > JOINT_HARD_LIMIT_DEG:
                issues.append(f"  LIMIT   t={t:5.2f}s  {names[j]}={v:+.1f}°")
            elif abs(v) > JOINT_SOFT_LIMIT_DEG:
                issues.append(f"  SOFT    t={t:5.2f}s  {names[j]}={v:+.1f}°")
        for j in (1, 2):
            if q[j] > 0.0:
                issues.append(f"  SIGN    t={t:5.2f}s  {names[j]}={np.degrees(q[j]):+.2f}°")
    for k in range(1, len(traj)):
        d = _wrap(traj.Q[k] - traj.Q[k - 1])
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
          f" (animation at {GIF_FPS:g} fps, {playback_speed(traj):.2f}× real time)")
    if traj.gap is None:
        print("  partner parks during each pivot")
    else:
        print(f"  partner follows each pivot at a {traj.gap:.3f} m gap "
              f"({traj.gap / CHAIN_LENGTH:.3f} L, scenarios.follow_gap)")
    for phase, start in zip(traj.phases, traj.phase_starts):
        print(f"  {start:5.1f} s  {phase.name:13s} {phase.duration:5.2f} s")
    if traj.retimes:
        print(f"  ({traj.retimes} phase slow-down(s) to keep the joints under "
              f"{np.degrees(PLAN_JOINT_SPEED):.1f}°/s)")
    print("-" * 60)

    issues = check(traj)
    if not issues:
        print(f"Check: CLEAN (no hard rules broken; no limits, signs, jumps≥"
              f"{np.degrees(MAX_JOINT_SPEED / traj.rate):.1f}° or ω violations)")
    else:
        for msg in issues:
            print(msg)
        print(f"  → {len(issues)} issue(s) total")

    if "--no-render" not in sys.argv:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gifs")
        os.makedirs(out_dir, exist_ok=True)
        print("Rendering...")
        render_video(traj, os.path.join(out_dir, scn.output))
