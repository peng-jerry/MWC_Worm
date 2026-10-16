"""
check_ik.py — joint-angle quality report for the transition scenarios.

Solves each scenario exactly as animate_transition.py does and reports the
metrics that matter when the joint angles are played back on the motors:
speed against the joint limit, smoothness, joint-limit and clearance
margins, and whether every hard rule (constraints.py) holds.

Usage:
    python check_ik.py [scenario ...]                  default: all scenarios
    python check_ik.py [scenario ...] --save-baseline  also store the joint
                                                       angles in ik_baseline.npz

Without --save-baseline, each scenario is also compared with the stored
baseline on the time samples both runs share (so a 6 fps run can be
compared with a 3 fps baseline).  Save a new baseline whenever you accept
a change; keyframe edits invalidate old comparisons.

The sample rate comes from timing.py; override it per run with e.g.
    TRAJ_SAMPLE_RATE=3 python check_ik.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory import solve_trajectory
from constraints import (CLEARANCE_GROUPS, HARD_CLEARANCE, JOINT_HARD_LIMIT_DEG,
                         JOINT_SOFT_LIMIT_DEG, SOFT_CLEARANCE)
from kinematics import q_display
from scenarios import SCENARIOS
from timing import MAX_JOINT_SPEED

_DIR          = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(_DIR, "ik_baseline.npz")

SOFT_DEG = JOINT_SOFT_LIMIT_DEG   # display-convention joint limits (constraints.py)
HARD_DEG = JOINT_HARD_LIMIT_DEG
MOVE_DEG = 0.05    # steps smaller than this count as stationary when counting reversals


def solve(scenario):
    """Solve one scenario; returns (Trajectory, seconds)."""
    t0 = time.time()
    traj = solve_trajectory(SCENARIOS[scenario])
    return traj, time.time() - t0


def metrics(traj):
    """Quality metrics for one solved trajectory."""
    Q   = traj.Q
    jump_deg = np.degrees(MAX_JOINT_SPEED / traj.rate)       # motor speed limit per frame
    Dw  = np.degrees(np.array([q_display(q) for q in Q]))   # wrapped, for limit checks
    D   = np.degrees(np.unwrap(np.radians(Dw), axis=0))     # unwrapped, for rates
    dq  = np.diff(D, axis=0)                                # °/frame
    ddq = np.diff(dq, axis=0)                               # °/frame²

    # Direction changes of each joint's velocity, ignoring near-stationary steps.
    reversals = 0
    for j in range(4):
        s = np.sign(dq[np.abs(dq[:, j]) > MOVE_DEG, j])
        reversals = max(reversals, int(np.sum(s[1:] != s[:-1])))

    infos = traj.infos
    margin = np.array([info["margin_score"] for info in infos])
    return {
        "Q":         Q,
        "residual":  max(info["residual_norm"] for info in infos),
        "max_dq":    float(np.abs(dq).max()),
        "jump_deg":  jump_deg,
        "jumps":     int(np.sum(np.abs(dq) >= jump_deg)),
        "max_ddq":   float(np.abs(ddq).max()),
        "reversals": reversals,
        "travel":    float(np.abs(dq).sum()),
        "max_q":     float(np.abs(Dw).max()),
        "soft":      int(np.sum(np.any(np.abs(Dw) > SOFT_DEG, axis=1))),
        "hard":      int(np.sum(np.any(np.abs(Dw) > HARD_DEG, axis=1))),
        "sign":      int(np.sum(np.any(Q[:, 1:3] > 0.0, axis=1))),
        "margin":    float(margin.sum() / traj.rate),     # soft score integrated over time (s)
        "margin_max": float(margin.max()),
        "clearance": {g: min(info["clearance"][g] for info in infos) for g in CLEARANCE_GROUPS},
        # Hard constraints broken by the chosen configuration, and frames where
        # no configuration satisfies them (the pose itself is infeasible).
        "violated":  {k: info["violations"] for k, info in enumerate(infos) if not info["feasible"]},
        "blocked":   {k: info["blocking"] for k, info in enumerate(infos) if info["blocking"]},
    }


def _term_summary(per_frame):
    """'  — term: fr a, b, …' summary of {frame: {term: value}}."""
    if not per_frame:
        return ""
    by_term = {}
    for k, terms in per_frame.items():
        for name in terms:
            by_term.setdefault(name, []).append(k)
    parts = []
    for name, ks in by_term.items():
        shown = ", ".join(map(str, ks[:8])) + (", …" if len(ks) > 8 else "")
        parts.append(f"{name}: fr {shown}")
    return "  — " + "; ".join(parts)


def _shared_samples(cur, base):
    """Rows of cur and base at the time samples both share, or None if none are shared."""
    nc, nb = len(cur) - 1, len(base) - 1
    if nc % nb == 0:
        return cur[::nc // nb], base
    if nb % nc == 0:
        return cur, base[::nb // nc]
    return None


def report(scenario, traj, m, seconds, baseline):
    n = len(traj)
    phases = ", ".join(f"{p.name} {p.duration:.1f}" for p in traj.phases)
    print(f"\n=== {scenario}   ({traj.times[-1]:.1f} s, {n} frames at {traj.rate:g} Hz, "
          + (f"loaded from cache in {seconds:.1f} s)" if traj.cached else f"solved in {seconds:.1f} s)"))
    print(f"  phases (s)          {phases}"
          + (f"   [{traj.retimes} slow-down(s)]" if traj.retimes else ""))
    print(f"  max residual        {m['residual']:.1e}")
    print(f"  max |Δq|            {m['max_dq']:5.2f} °/frame ({m['max_dq'] * traj.rate:.1f} °/s)     "
          f"steps ≥ {m['jump_deg']:.2f}° (motor limit): {m['jumps']}")
    print(f"  max |Δ²q|           {m['max_ddq']:5.2f} °/frame²")
    print(f"  velocity reversals  {m['reversals']} (worst joint)")
    print(f"  total joint travel  {m['travel']:.1f}°")
    print(f"  max |q| (display)   {m['max_q']:.1f}°   frames > {SOFT_DEG:.0f}°: {m['soft']}   "
          f"> {HARD_DEG:.0f}°: {m['hard']}   q2/q3 > 0: {m['sign']}")
    print(f"  margin score        {m['margin']:.3g} over the run, max {m['margin_max']:.3g} per frame "
          f"(clearance and joint margins; 1 = one at its hard limit)")
    print(f"  smallest clearance  " + ", ".join(f"{g} {v * 1000:.0f} mm" for g, v in m["clearance"].items())
          + f"   (hard {HARD_CLEARANCE * 1000:.0f}, soft {SOFT_CLEARANCE * 1000:.0f} mm)")
    print(f"  hard violations     {len(m['violated'])}/{n} frames{_term_summary(m['violated'])}")
    print(f"  pose infeasible     {len(m['blocked'])}/{n} frames (no configuration meets "
          f"every hard rule){_term_summary(m['blocked'])}")
    if scenario in baseline:
        shared = _shared_samples(m["Q"], baseline[scenario])
        if shared is None:
            print(f"  vs baseline         no shared time samples ({n} vs {len(baseline[scenario])} frames)")
        else:
            cur, base = shared
            d = np.degrees(np.abs((cur - base + np.pi) % (2 * np.pi) - np.pi))
            print(f"  vs baseline         max |Δq| {d.max():.2f}°   frames differing > 1°: "
                  f"{int(np.sum(d.max(axis=1) > 1.0))}/{len(d)} shared samples")


def main():
    save      = "--save-baseline" in sys.argv
    scenarios = [a for a in sys.argv[1:] if not a.startswith("--")] or list(SCENARIOS)
    unknown   = [s for s in scenarios if s not in SCENARIOS]
    if unknown:
        print(f"Unknown scenario(s) {unknown}. Choose from {list(SCENARIOS)}")
        sys.exit(1)

    stored   = dict(np.load(BASELINE_PATH)) if os.path.exists(BASELINE_PATH) else {}
    baseline = {} if save else stored

    for sc in scenarios:
        traj, seconds = solve(sc)
        m = metrics(traj)
        report(sc, traj, m, seconds, baseline)
        stored[sc] = m["Q"]

    if save:
        np.savez(BASELINE_PATH, **stored)
        print(f"\nSaved baseline → {BASELINE_PATH}")


if __name__ == "__main__":
    main()
