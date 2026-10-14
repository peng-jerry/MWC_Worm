"""
test_constraints.py — checks of the geometry, environment and robot rules
against hand-worked cases.

Each check builds a situation whose answer is known (a point a set distance
from a wall, a link pushed through a plate, a joint past its limit, ...) and
compares.  The plan of every transition is also sampled, which runs the
plan's own checks (reach, assemblies vs surfaces and each other).

Usage:
    python test_constraints.py         (about 20 s)
Exits non-zero if any check fails.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenarios as S
from constraints import (CLEARANCE_GROUPS, FEASIBILITY_TOL, HARD_CLEARANCE, JOINT_HARD_LIMIT_DEG,
                         JOINT_SOFT_LIMIT_DEG, SOFT_CLEARANCE, Evaluation, evaluate)
from environment import Environment, Surface
from geometry import point_segment_distance, segment_distance, segments_cross

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


# ── geometry ──────────────────────────────────────────────────────────────────

def test_geometry():
    print("geometry")
    p = lambda *v: np.array(v, dtype=float)
    check("parallel segments 0.1 apart", close(segment_distance(p(0, 0), p(1, 0), p(0.5, 0.1), p(2, 0.1)), 0.1))
    check("crossing segments touch", close(segment_distance(p(0, 0), p(1, 1), p(0, 1), p(1, 0)), 0.0))
    check("point to segment", close(segment_distance(p(0.5, 0.3), p(0.5, 0.3), p(0, 0), p(1, 0)), 0.3))
    check("point to point", close(segment_distance(p(0, 0), p(0, 0), p(0.3, 0.4), p(0.3, 0.4)), 0.5))
    # Against the direct formula (nearest of the four end-to-segment
    # distances, 0 when the segments cross) on random segments.
    rng = np.random.default_rng(1)
    a, b, c, d = (rng.uniform(-1, 1, (2000, 2)) for _ in range(4))
    ref = np.minimum(np.minimum(point_segment_distance(a, c, d), point_segment_distance(b, c, d)),
                     np.minimum(point_segment_distance(c, a, b), point_segment_distance(d, a, b)))
    ref = np.where(segments_cross(a, b, c, d), 0.0, ref)
    err = np.abs(segment_distance(a, b, c, d) - ref).max()
    check("2000 random segment pairs match the direct formula", err < 1e-12, f"worst {err:.1e}")


# ── environment ───────────────────────────────────────────────────────────────

def test_environment():
    print("environment")
    p = lambda *v: np.array(v, dtype=float)
    corner = Environment(Surface((0, 0), (1, 0), (0, 1)), Surface((0, 0), (0, 1), (1, 0)), "inside")
    check("inside corner: point 0.1 from the wall", close(corner.clearance(p(0.1, 0.5), p(0.1, 0.5), 0), 0.1))
    check("inside corner: capsule of radius 0.02, 0.1 above the floor",
          close(corner.clearance(p(0.2, 0.1), p(0.6, 0.1), 0.02), 0.08))
    check("inside corner: point behind the wall is inside", corner.clearance(p(-0.01, 0.5), p(-0.01, 0.5), 0) < 0)

    block = Environment(Surface((0, 0.55), (1, 0), (0, 1)), Surface((0, 0.55), (0, -1), (-1, 0)), "outside")
    check("outside corner: point beyond the corner, 0.1 / 0.05 off",
          close(block.clearance(p(-0.1, 0.6), p(-0.1, 0.6), 0), np.hypot(0.1, 0.05)))
    check("outside corner: point inside the block", block.clearance(p(0.1, 0.5), p(0.1, 0.5), 0) < 0)
    check("outside corner: link cut through the block's corner (both ends outside)",
          block.clearance(p(0.1, 0.56), p(-0.1, 0.40), 0) < 0)
    check("outside corner: link passing over the corner 0.1 / √2 away",
          close(block.clearance(p(0.05, 0.7), p(-0.2, 0.45), 0), 0.1 / np.sqrt(2)))
    check("outside corner: link passing the corner 0.05 away",
          close(block.clearance(p(-0.05, 0.8), p(-0.05, 0.3), 0), 0.05))

    t = 0.0034
    plate = Environment(Surface((0.5, 0.5), (-1, 0), (0, 1)), Surface((0.5, 0.5 - t), (-1, 0), (0, -1)), "outside")
    check("plate: link straight through it", plate.clearance(p(0.3, 0.6), p(0.3, 0.4), 0) < 0)
    check("plate: link just past its end", close(plate.clearance(p(0.6, 0.6), p(0.6, 0.4), 0), 0.1))
    check("plate: point inside its thickness", plate.clearance(p(0.3, 0.4990), p(0.3, 0.4990), 0) < 0)
    straight = 0.02
    around = plate.separation(p(0.3, 0.51), p(0.3, 0.51), p(0.3, 0.49), p(0.3, 0.49))
    check("plate: points on opposite faces are measured around its end",
          around > 0.39, f"{straight * 1000:.0f} mm straight, {float(around) * 1000:.0f} mm around")
    check("plate: points on the same face are measured straight",
          close(plate.separation(p(0.3, 0.51), p(0.3, 0.51), p(0.2, 0.51), p(0.2, 0.51)), 0.1))


# ── robot rules ───────────────────────────────────────────────────────────────

def test_rules():
    print("robot rules")
    scn = S.SCENARIOS["floor_to_wall"]
    env, wg = scn.env, scn.wg
    phases = S.plan_phases(scn)
    _, poses = S.sample_poses(scn, phases, 1.0)
    pose = poses[0]
    q = S.cruise_posture(scn, pose)
    ev = evaluate(q, pose, env, wg, S.l1, S.l2, S.l3)
    check("cruise posture at the start meets every hard rule", bool(ev.feasible[0]), str(ev.violations(0)))
    check("cruise posture: chain ≥ hard clearance from the floor",
          ev.clearance["surface"][0] >= HARD_CLEARANCE, f"{ev.clearance['surface'][0] * 1000:.0f} mm")

    def with_joint(j, display_deg):
        """q with joint j set to display_deg in the display convention."""
        out = q.copy()
        out[j] = np.radians(display_deg) + (np.pi / 2 if j in (0, 3) else 0.0)
        return out

    over = with_joint(0, JOINT_HARD_LIMIT_DEG + 5)
    check("q1 5° past the hard limit breaks 'joint limit'",
          "joint limit" in evaluate(over, pose, env, wg, S.l1, S.l2, S.l3).violations(0))
    # Soft score: one joint halfway from the soft to the hard limit scores
    # 0.5² = 0.25; one clearance halfway from soft to hard does the same.
    halfway_deg = 0.5 * (JOINT_SOFT_LIMIT_DEG + JOINT_HARD_LIMIT_DEG)
    halfway_gap = 0.5 * (SOFT_CLEARANCE + HARD_CLEARANCE)
    far = {g: np.array([1.0]) for g in CLEARANCE_GROUPS}
    one_joint = Evaluation(far, np.radians([[0.0, halfway_deg, 0.0, 0.0]]), np.array([-0.1]))
    one_gap = Evaluation(dict(far, surface=np.array([halfway_gap])), np.zeros((1, 4)), np.array([-0.1]))
    check("a joint halfway from the soft to the hard limit scores 0.25", close(one_joint.soft[0], 0.25))
    check("a clearance halfway from the soft to the hard value scores 0.25", close(one_gap.soft[0], 0.25))
    bent = q.copy()
    bent[1] = abs(bent[1])
    check("q2 bent the wrong way breaks 'bend'",
          "bend" in evaluate(bent, pose, env, wg, S.l1, S.l2, S.l3).violations(0))
    down = q.copy()
    down[0] = np.pi / 2 + np.radians(-100)          # l1 swung down toward the floor
    ev_down = evaluate(down, pose, env, wg, S.l1, S.l2, S.l3)
    check("l1 swung down into the floor breaks a clearance",
          any(k.startswith("clearance") for k in ev_down.violations(0)), str(ev_down.violations(0)))

    # Assemblies: both at one pose overlap; the plan check must say so.
    stacked = dict(pose, x2=pose["x1"], y2=pose["y1"], theta2=pose["theta1"])
    check("two assemblies on top of each other are reported",
          "front and back assemblies too close" in S.check_assemblies(scn, stacked))
    wheels_in = dict(pose, y1=pose["y1"] - 0.01)
    check("an assembly pushed 10 mm into the floor is reported",
          any("in the surface" in k for k in S.check_assemblies(scn, wheels_in)))


# ── plans ─────────────────────────────────────────────────────────────────────

def test_plans():
    print("plans (every transition sampled at 6 fps: reach, assemblies vs surfaces and each other)")
    for name, scn in S.SCENARIOS.items():
        gap = S.follow_gap(scn) if scn.partner_follows else None
        try:
            phases = S.plan_phases(scn, gap=gap)
            _, poses = S.sample_poses(scn, phases, 6.0)
            ok, detail = True, f"{len(poses)} poses" + (f", follow gap {gap:.3f} m" if gap else "")
        except ValueError as e:
            ok, detail = False, str(e)
        check(f"{name} plan is valid", ok, detail)
    # The pivot around the plate's end keeps the trailing wheel on the plate.
    scn = S.SCENARIOS["thin_edge"]
    env, r = scn.env, scn.wg.wheel_r
    side = S.SIDE[scn.lead]
    trailing = int(np.argmax(S.wheel_offsets(S.surface_orientation(scn.approach), scn.wg, side)
                             @ np.array(scn.approach.tangent, float)))
    worst = 0.0
    for f in np.linspace(0, 1, 401):
        J, th = S.pivot_pose(scn, side, f)
        w = J + S.wheel_offsets(th, scn.wg, side)[trailing]
        worst = max(worst, abs(float(env.clearance(w, w, r))))
    check("thin edge: the pivoting wheel stays on the plate (around both corners and the end face)",
          worst < FEASIBILITY_TOL, f"worst {worst * 1000:.3f} mm")


if __name__ == "__main__":
    test_geometry()
    test_environment()
    test_rules()
    test_plans()
    if FAILURES:
        print(f"\nFAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("\nPASS")
