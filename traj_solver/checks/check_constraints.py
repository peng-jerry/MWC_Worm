"""
check_constraints.py — checks of the geometry, environment and robot rules
against hand-worked cases.

Each check builds a situation whose answer is known (a point a set distance
from a wall, a link pushed through a plate, a joint past its limit, ...) and
compares.  Every transition definition is also checked, and every plan
sampled, which runs the plan's own checks (reach, assemblies vs surfaces and
each other) — so a new transition is covered as soon as it is added.

Usage:
    python checks/check_constraints.py        (about 20 s; always checks every transition)
Exits non-zero if any check fails.
"""

import os
import sys

import numpy as np

# traj_solver/ (this script's parent folder) on the path, so `core` imports from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine import planner as P
from core.engine.constraints import CLEARANCE_GROUPS, Evaluation, evaluate
from core.engine.environment import Environment, Surface
from core.engine.geometry import point_segment_distance, segment_distance, segments_cross
from core.params import (FEASIBILITY_TOL, HARD_CLEARANCE, JOINT_HARD_LIMIT_DEG, JOINT_SOFT_LIMIT_DEG,
                         SOFT_CLEARANCE)
from core.robot import MOTOR_DEPTH, MOTOR_HEIGHT, l1, l2, l3
from core.scenarios import SCENARIOS

FAILURES = []


def check(name, ok, detail=""):
    """Print one check's result (and `detail`); record it if it failed."""
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def close(a, b, tol=1e-9):
    """True when a and b agree to within tol."""
    return abs(float(a) - float(b)) <= tol


# ── geometry ──────────────────────────────────────────────────────────────────

def check_geometry():
    """Segment distances against cases worked by hand and against a direct formula."""
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

def check_environment():
    """Clearance and separation at an inside corner, an outside corner and a thin plate."""
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

def check_rules():
    """The hard and soft rules on configurations that keep or break them on purpose."""
    print("robot rules")
    scn = next(iter(SCENARIOS.values()))           # any transition: only its start pose is used
    env, wg = scn.env, scn.wg
    phases = P.plan_phases(scn)
    _, poses = P.sample_poses(scn, phases, 1.0)
    pose = poses[0]
    q = P.cruise_posture(scn, pose)
    ev = evaluate(q, pose, env, wg, l1, l2, l3)
    check("cruise posture at the start meets every hard rule", bool(ev.feasible[0]), str(ev.violations(0)))
    check("cruise posture: chain ≥ hard clearance from the surface",
          ev.clearance["surface"][0] >= HARD_CLEARANCE, f"{ev.clearance['surface'][0] * 1000:.0f} mm")

    def with_joint(j, display_deg):
        """q with joint j set to display_deg in the display convention."""
        out = q.copy()
        out[j] = np.radians(display_deg) + (np.pi / 2 if j in (0, 3) else 0.0)
        return out

    over = with_joint(0, JOINT_HARD_LIMIT_DEG + 5)
    check("q1 5° past the hard limit breaks 'joint limit'",
          "joint limit" in evaluate(over, pose, env, wg, l1, l2, l3).violations(0))
    # Soft score: one joint halfway from the soft to the hard limit scores
    # 0.5² = 0.25; one clearance halfway from soft to hard does the same.
    halfway_deg = 0.5 * (JOINT_SOFT_LIMIT_DEG + JOINT_HARD_LIMIT_DEG)
    halfway_gap = 0.5 * (SOFT_CLEARANCE + HARD_CLEARANCE)
    far = {g: np.array([1.0]) for g in CLEARANCE_GROUPS}
    one_joint = Evaluation(far, np.radians([[0.0, halfway_deg, 0.0, 0.0]]), np.array([-0.1]))
    one_gap = Evaluation(dict(far, surface=np.array([halfway_gap])), np.zeros((1, 4)), np.array([-0.1]))
    check("a joint halfway from the soft to the hard limit scores 0.25", close(one_joint.soft[0], 0.25))
    check("a clearance halfway from the soft to the hard value scores 0.25", close(one_gap.soft[0], 0.25))
    # The chain standing straight up, the back assembly upside down at its top:
    # measured from the edge of its joint housing (JOINT_RADIUS up the link),
    # each end link clears its own motor by
    #   (MOTOR_DEPTH + JOINT_RADIUS) − JOINT_RADIUS − MOTOR_HEIGHT / 2
    # = MOTOR_DEPTH − MOTOR_HEIGHT / 2 (44 mm at 61 and 34 mm).  Measured from
    # the joint centre it would be JOINT_RADIUS less, whatever the link's angle.
    upright = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2])
    tower = {"x1": 0.5, "y1": 0.5, "theta1": 0.0,
             "x2": 0.5, "y2": 0.5 + l1 + l2 + l3, "theta2": np.pi}
    ev_up = evaluate(upright, tower, env, wg, l1, l2, l3)
    expected = MOTOR_DEPTH - MOTOR_HEIGHT / 2
    for g in ("front assembly", "back assembly"):
        check(f"upright end link clears its own motor by {expected * 1000:.0f} mm ({g})",
              close(ev_up.clearance[g][0], expected), f"{ev_up.clearance[g][0] * 1000:.1f} mm")
    bent = q.copy()
    bent[1] = abs(bent[1])
    check("q2 bent the wrong way breaks 'bend'",
          "bend" in evaluate(bent, pose, env, wg, l1, l2, l3).violations(0))
    down = q.copy()
    down[0] = np.pi / 2 + np.radians(-100)          # l1 swung down toward the surface
    ev_down = evaluate(down, pose, env, wg, l1, l2, l3)
    check("l1 swung down into the surface breaks a clearance",
          any(k.startswith("clearance") for k in ev_down.violations(0)), str(ev_down.violations(0)))

    # Assemblies: both at one pose overlap; the plan check must say so.
    stacked = dict(pose, x2=pose["x1"], y2=pose["y1"], theta2=pose["theta1"])
    check("two assemblies on top of each other are reported",
          "front and back assemblies too close" in P.check_assemblies(scn, stacked))
    n = scn.approach.normal                           # the front assembly rests on the approach surface
    wheels_in = dict(pose, x1=pose["x1"] - 0.01 * n[0], y1=pose["y1"] - 0.01 * n[1])
    check("an assembly pushed 10 mm into the surface is reported",
          any("in the surface" in k for k in P.check_assemblies(scn, wheels_in)))


# ── plans ─────────────────────────────────────────────────────────────────────

def check_definitions():
    """A transition defined wrongly is refused when it is read, with a message naming the
    mistake (core/scenarios.py, Scenario.__post_init__); a correct new one is accepted."""
    print("transition definitions")
    floor = Surface((0.0, 0.0), (1, 0), (0, 1))
    wall = Surface((0.0, 0.0), (0, 1), (1, 0))
    wg = next(iter(SCENARIOS.values())).wg

    def refused(why, expect, **fields):
        args = dict(name="test", label="test", approach=floor, depart=wall, turn=-np.pi / 2,
                    corner="inside", wg=wg)
        args.update(fields)
        try:
            P.Scenario(**args)
            check(f"refused: {why}", False, "accepted")
        except ValueError as e:
            check(f"refused: {why}", expect in str(e), str(e)[:90])

    refused("a turn that does not reach the other surface", "does not take", turn=+np.pi / 2 + 0.1)
    refused("the long way round a corner", "short way round", turn=+3 * np.pi / 2)
    refused("a tangent pointing toward the transition", "cannot arch away",
            depart=Surface((0.0, 0.0), (0, -1), (1, 0)), turn=-np.pi / 2)
    refused("an inside corner whose surfaces do not meet", "must start at the same point",
            depart=Surface((0.0, 0.1), (0, 1), (1, 0)))
    refused("a corner that is neither inside nor outside", "corner must be", corner="sideways")
    try:
        Surface((0.0, 0.0), (1, 0), (0, 2))
        check("refused: a normal that is not a unit vector", False, "accepted")
    except ValueError as e:
        check("refused: a normal that is not a unit vector", True, str(e)[:90])
    # The template in core/scenarios.py (the reverse of floor_to_wall) is accepted, and the
    # leading assembly comes out of its surfaces.
    reverse = P.Scenario(name="wall_to_floor", label="test", approach=wall, depart=floor,
                         turn=+np.pi / 2, corner="inside", wg=wg)
    check("accepted: wall to floor, led by the back assembly", reverse.lead == "back", reverse.lead)


def check_plans():
    """Every transition's plan samples cleanly, and every pivot keeps its assembly on the solid."""
    print("plans (every transition sampled at 6 fps: reach, assemblies vs surfaces and each other)")
    for name, scn in SCENARIOS.items():
        gap = P.follow_gap(scn) if scn.partner_follows else None
        try:
            phases = P.plan_phases(scn, gap=gap)
            _, poses = P.sample_poses(scn, phases, 6.0)
            ok, detail = True, f"{len(poses)} poses" + (f", follow gap {gap:.3f} m" if gap else "")
        except ValueError as e:
            ok, detail = False, str(e)
        check(f"{name} plan is valid", ok, detail)
    # Both assemblies' pivots, in every transition: some wheel touches the solid
    # throughout and none goes into it — pressed into an inside corner, or
    # rolling round an outside corner and along any end face.
    for name, scn in SCENARIOS.items():
        env, r = scn.env, scn.wg.wheel_r
        worst = 0.0
        for side in (+1, -1):
            for f in np.linspace(0, 1, 401):
                J, th = P.pivot_pose(scn, side, f)
                nearest = min(float(env.clearance(w, w, r)) for w in J + P.wheel_offsets(th, scn.wg, side))
                worst = max(worst, abs(nearest))
        check(f"{name}: each pivoting assembly keeps a wheel on the solid and none in it",
              worst < FEASIBILITY_TOL, f"worst {worst * 1000:.3f} mm")


def main():
    check_geometry()
    check_environment()
    check_rules()
    check_definitions()
    check_plans()
    if FAILURES:
        print(f"\nFAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
