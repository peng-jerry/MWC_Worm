"""
animate.py — solve a transition's joint angles, check them, render them.

Pipeline
--------
  core/transitions.py          the transition: its two surfaces and how it turns
  core/engine/planner.py     its phases, from geometry (approach, pivot, roll
                             clear, roll up, pivot, depart; roll clear and roll
                             up are one phase at an outside corner or edge),
                             each easing in and out of rest
  core/engine/trajectory.py  solve_trajectory: samples the phases at
                             params.SAMPLE_RATE and lets trajectory_ik choose
                             every frame's joint angles together (cached)
  render_video               GIF/MP4 of the solved transition, with every part
                             drawn at its real size (core/engine/drawing.py)
  core/engine/solve_check.py the check: hard rules, joint limits, bend
                             direction, joint speed, wheel speed, drive

Usage:
    python generate/animate.py [transition ...] [--no-render]     default: all (-h for help)
        --no-render   solve and check only

Writes outputs/gifs/<transition>.mp4 (.gif if ffmpeg is not installed).  Exits
non-zero if the check finds a problem in any transition, or one cannot be
planned at all (so build_all.py reports it); joints past the soft limit are
only warnings.
"""

import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# traj_solver/ (this script's parent folder) on the path, so `core` imports from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine.cli import run_each, output_dir, parse_args
from core.engine.drawing import draw_frame, plot_limits
from core.engine.kinematics import q_display
from core.engine.solve_check import check
from core.engine.trajectory import solve_trajectory
from core.motor_limits import MAX_JOINT_SPEED
from core.params import PLAN_JOINT_SPEED
from core.robot import CHAIN_LENGTH
from core.transitions import TRANSITIONS

OUT_DIR = output_dir("gifs")


# ------------------------------------------------------------------ #
#  Animation                                                          #
# ------------------------------------------------------------------ #

# Animation playback: every GIF_FRAME_STEP-th solved frame is drawn, one every
# GIF_FRAME_MS.  Speed = GIF_FPS × GIF_FRAME_STEP / sample rate; at 6 waypoints
# per second that is 20 × 1 / 6 ≈ 3.3× real time.  GIF frame delays are stored in whole
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
        """Write the collected frames as one looping GIF."""
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
        """Draw animation frame i: solved frame shown[i], titled with its time, phase and joints."""
        ax.cla()
        k = shown[i]
        t = traj.times[k]
        q = traj.Q[k]
        qd = np.degrees(q_display(q))
        draw_frame(ax, traj.transition, q, traj.poses[k], xlim, ylim)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{traj.transition.label} — t = {t:4.1f} s ({traj.phases[traj.phase_at(t)].name})"
                     f"  ({speed:.1f}× real time)\n"
                     + "  ".join(f"q{j + 1}={v:+.1f}°" for j, v in enumerate(qd)), fontsize=9)

    anim = animation.FuncAnimation(fig, draw, frames=len(shown), interval=1000 / fps)
    ffmpeg = animation.FFMpegWriter.isAvailable()
    saved = False
    if output_file.endswith(".mp4") and ffmpeg:
        try:
            anim.save(output_file, writer=animation.FFMpegWriter(fps=fps, bitrate=2000), dpi=120)
            print(f"\nSaved -> {output_file}")
            saved = True
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"\nffmpeg failed ({e}); writing a GIF instead")
    if not saved:
        gif_file = os.path.splitext(output_file)[0] + ".gif"
        anim.save(gif_file, writer=_GifWriter(fps=fps), dpi=100)
        print(f"\nSaved -> {gif_file}" + ("" if ffmpeg else "  (install ffmpeg for MP4)"))
    plt.close(fig)


def run(name, render=True):
    """Solve one transition, print its timing and check, and render it.
    Returns True when the check found no problem."""
    trn = TRANSITIONS[name]
    traj = solve_trajectory(trn)
    print(f"{trn.label}: {traj.times[-1]:.1f} s, {len(traj)} frames at {traj.rate:g} Hz"
          f" (animation at {GIF_FPS:g} fps, {playback_speed(traj):.2f}× real time)")
    if traj.gap is None:
        print("  partner parks during each pivot")
    else:
        print(f"  partner follows each pivot at a {traj.gap:.3f} m gap "
              f"({traj.gap / CHAIN_LENGTH:.3f} L, follow_gap.py)")
    for phase, start in zip(traj.phases, traj.phase_starts):
        print(f"  {start:5.1f} s  {phase.name:13s} {phase.duration:5.2f} s")
    if traj.retimes:
        print(f"  ({traj.retimes} phase slow-down(s) to keep the joints under "
              f"{np.degrees(PLAN_JOINT_SPEED):.1f}°/s)")
    print("-" * 60)

    problems, warnings = check(traj)
    for msg in problems + warnings:
        print(msg)
    if problems:
        print(f"  → {len(problems)} problem(s)")
    else:
        print(f"Check: CLEAN (no hard rules broken; no limits, signs, jumps≥"
              f"{np.degrees(MAX_JOINT_SPEED / traj.rate):.1f}°, ω or drive violations)"
              + (f"; {len(warnings)} joint(s) past the soft limit" if warnings else ""))

    if render:
        os.makedirs(OUT_DIR, exist_ok=True)
        print("Rendering...")
        render_video(traj, os.path.join(OUT_DIR, trn.output))
    return not problems


def main():
    names, args = parse_args(__doc__, [(("--no-render",), dict(
        action="store_true", help="solve and check only; write no animation"))])

    def animate(name):
        if name != names[0]:
            print()
        return run(name, render=not args.no_render)

    sys.exit(1 if run_each(names, animate) else 0)


if __name__ == "__main__":
    main()
