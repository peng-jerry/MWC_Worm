"""
animate.py — solve a transition's joint angles, check them, render them.

Pipeline
--------
  core/scenarios.py  geometry and the phases of each transition
                     (approach, pivot, roll clear, roll up, pivot, depart;
                     roll clear and roll up are one phase at an outside
                     corner or edge), each easing in and out of rest
  core/trajectory.py solve_trajectory: samples the phases at
                     params.SAMPLE_RATE and lets trajectory_ik choose every
                     frame's joint angles together (cached between runs)
  render_video       GIF/MP4 of the solved transition, with every part drawn
                     at its real size (core/drawing.py, constraints.py outline)
  __main__           solve, run the built-in check (hard rules, joint
                     limits, bend direction, joint speed, wheel speed) and
                     render

Usage:
    python animate.py [scenario] [--no-render]
        scenario      floor_to_wall | wall_to_ceiling | outside | thin_edge
                      (default wall_to_ceiling)
        --no-render   solve and check only

Writes outputs/gifs/<scenario>.mp4 (.gif if ffmpeg is not installed).
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.constraints import JOINT_HARD_LIMIT_DEG, JOINT_SOFT_LIMIT_DEG
from core.drawing import draw_frame, plot_limits
from core.kinematics import q_display
from core.params import PLAN_JOINT_SPEED, MAX_JOINT_SPEED, SLIDE_SPEED
from core.scenarios import SCENARIOS, WHEELS, CHAIN_LENGTH, wheel_speeds
from core.trajectory import solve_trajectory


def _wrap(a):
    """Wrap angles to [-π, π)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ #
#  Animation                                                          #
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
    spin, _ = wheel_speeds(traj.scenario, traj.phases, traj.times)
    for k, j in zip(*np.nonzero(np.abs(spin) > omega_max)):
        issues.append(f"  OMEGA   t={traj.times[k]:5.2f}s  {WHEELS[j]}  ω={spin[k, j]:+.2f} rad/s")
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
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "gifs")
        os.makedirs(out_dir, exist_ok=True)
        print("Rendering...")
        render_video(traj, os.path.join(out_dir, scn.output))
