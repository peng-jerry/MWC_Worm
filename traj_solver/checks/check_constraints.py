"""
check_constraints.py — checks of the geometry, environment and robot rules
against hand-worked cases.

Each check builds a situation whose answer is known (a point a set distance
from a wall, a link pushed through a plate, a joint past its limit, ...) and
compares.  Every transition definition is also checked, and every plan
sampled, which runs the plan's own checks (reach, assemblies vs surfaces and
each other) — so a new transition is covered as soon as it is added.

Usage:
    python checks/check_constraints.py        (about a minute, most of it searching the
                                              follow gaps; always checks every transition)
Exits non-zero if any check fails.
"""

import os
import sys
import tempfile
from dataclasses import replace
from functools import cache

import numpy as np

# traj_solver/ (this script's parent folder) on the path, so `core` imports from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine import placement, planner
from core.engine.constraints import CLEARANCE_GROUPS, Evaluation, evaluate, wheel_centers
from core.engine.environment import Environment, Surface
from core.engine.follow_gap import follow_gap
from core.engine.geometry import point_segment_distance, segment_distance, segments_cross
from core.engine.kinematics import q_display, q_from_display
from core.engine.timing_file import read_current, read_timing, waypoints, write_timing
from core.engine.trajectory import Trajectory
from core.engine.transition import Transition
from core.engine.wheels import WHEELS, wheel_speeds, wheels_touching, without_drive
from core.motor_calibration import JOINTS, Motor
from core.params import (FEASIBILITY_TOL, HARD_CLEARANCE, JOINT_HARD_LIMIT_DEG, JOINT_SOFT_LIMIT_DEG,
                         RAMP_TIME, SAMPLE_RATE, SOFT_CLEARANCE)
from core.robot import MOTOR_DEPTH, MOTOR_HEIGHT, l1, l2, l3, wg_inside
from core.transitions import TRANSITIONS

FAILURES = []


def check(name, ok, detail=""):
    """Print one check's result (and `detail`); record it if it failed."""
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def close(a, b, tol=1e-9):
    """True when a and b agree to within tol."""
    return abs(float(a) - float(b)) <= tol


@cache
def gap_of(name):
    """The follow gap of transition `name` (None where the partner parks), searched once."""
    t = TRANSITIONS[name]
    return follow_gap(t) if t.partner_follows else None


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
    trn = next(iter(TRANSITIONS.values()))           # any transition: only its start pose is used
    env, wg = trn.env, trn.wg
    phases = planner.plan_phases(trn)
    _, poses = planner.sample_poses(trn, phases, 1.0)
    pose = poses[0]
    q = planner.cruise_posture(trn, pose)
    ev = evaluate(q, pose, env, wg, l1, l2, l3)
    check("cruise posture at the start meets every hard rule", bool(ev.feasible[0]), str(ev.violations(0)))
    check("cruise posture: chain ≥ hard clearance from the surface",
          ev.clearance["surface"][0] >= HARD_CLEARANCE, f"{ev.clearance['surface'][0] * 1000:.0f} mm")

    def with_joint(j, display_deg):
        """q with joint j set to display_deg in the display convention."""
        out = q_display(q)
        out[j] = np.radians(display_deg)
        return q_from_display(out)

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
    upright = q_from_display(np.zeros(4))
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
    down = with_joint(0, -100)                      # l1 swung down toward the surface
    ev_down = evaluate(down, pose, env, wg, l1, l2, l3)
    check("l1 swung down into the surface breaks a clearance",
          any(k.startswith("clearance") for k in ev_down.violations(0)), str(ev_down.violations(0)))

    # Assemblies: both at one pose overlap; the plan check must say so.
    stacked = dict(pose, x2=pose["x1"], y2=pose["y1"], theta2=pose["theta1"])
    check("two assemblies on top of each other are reported",
          "front and back assemblies too close" in placement.check_assemblies(trn, stacked))
    n = trn.approach.normal                           # the front assembly rests on the approach surface
    wheels_in = dict(pose, x1=pose["x1"] - 0.01 * n[0], y1=pose["y1"] - 0.01 * n[1])
    check("an assembly pushed 10 mm into the surface is reported",
          any("in the surface" in k for k in placement.check_assemblies(trn, wheels_in)))


# ── plans ─────────────────────────────────────────────────────────────────────

def check_definitions():
    """A transition defined wrongly is refused when it is read, with a message naming the
    mistake (core/transitions.py, Transition.__post_init__); a correct new one is accepted."""
    print("transition definitions")
    floor = Surface((0.0, 0.0), (1, 0), (0, 1))
    wall = Surface((0.0, 0.0), (0, 1), (1, 0))
    wg = next(iter(TRANSITIONS.values())).wg

    def refused(why, expect, **fields):
        args = dict(name="test", label="test", approach=floor, depart=wall, turn=-np.pi / 2,
                    corner="inside", wg=wg)
        args.update(fields)
        try:
            Transition(**args)
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
    # The template in core/transitions.py (the reverse of floor_to_wall) is accepted, and the
    # leading assembly comes out of its surfaces.
    reverse = Transition(name="wall_to_floor", label="test", approach=wall, depart=floor,
                         turn=+np.pi / 2, corner="inside", wg=wg)
    check("accepted: wall to floor, led by the back assembly", reverse.lead == "back", reverse.lead)


def check_plans():
    """Every transition's plan samples cleanly, and every pivot keeps its assembly on the solid."""
    print("plans (every transition sampled at 6 per second: reach, assemblies vs surfaces and each other)")
    for name, trn in TRANSITIONS.items():
        gap = gap_of(name)
        try:
            phases = planner.plan_phases(trn, gap=gap)
            _, poses = planner.sample_poses(trn, phases, 6.0)
            ok, detail = True, f"{len(poses)} poses" + (f", follow gap {gap:.3f} m" if gap else "")
        except ValueError as e:
            ok, detail = False, str(e)
        check(f"{name} plan is valid", ok, detail)
    # Both assemblies' pivots, in every transition: some wheel touches the solid
    # throughout and none goes into it — pressed into an inside corner, or
    # rolling round an outside corner and along any end face.
    for name, trn in TRANSITIONS.items():
        env, r = trn.env, trn.wg.wheel_r
        worst = 0.0
        for side in (+1, -1):
            for f in np.linspace(0, 1, 401):
                J, th = placement.pivot_pose(trn, side, f)
                nearest = min(float(env.clearance(w, w, r)) for w in J + placement.wheel_offsets(th, trn.wg, side))
                worst = max(worst, abs(nearest))
        check(f"{name}: each pivoting assembly keeps a wheel on the solid and none in it",
              worst < FEASIBILITY_TOL, f"worst {worst * 1000:.3f} mm")


def check_drive():
    """Driven wheels stay on the solid whenever their assembly moves, in every plan,
    sampled every 0.05 s; and a plan that lifts one is caught."""
    print("drive (driven wheels on the solid while their assembly moves, every 0.05 s)")

    def lifted(trn, gap):
        phases = planner.plan_phases(trn, gap=gap)
        times = np.arange(0.0, planner.phase_starts(phases)[-1], 0.05)
        off = without_drive(trn, phases, times)
        return ", ".join(f"{WHEELS[j]} {int(off[:, j].sum())}" for j in range(4) if off[:, j].any())

    for name, trn in TRANSITIONS.items():
        gap = gap_of(name)
        off = lifted(trn, gap)
        check(f"{name}: driven wheels stay on the solid", not off, off and f"samples off: {off}")
    # The outside corner with the inside-corner arm layout lifts the front
    # driven wheel off the block while that assembly moves.
    wrong = replace(TRANSITIONS["outside"], name="outside_inside_arms", wg=wg_inside)
    off = lifted(wrong, 0.635)
    check("an outside corner with the inside-corner arms is caught", "front arm A" in off, off)


def check_wheels():
    """Wheel speeds (core/engine/wheels.py) against cases worked by hand, and against
    rolling without slipping on every plan."""
    print("wheel speeds")
    # Halfway through an approach both assemblies roll at their peak speed,
    # distance / (duration − RAMP_TIME) (planner.ease), so every wheel turns at that
    # over its radius.  Rolling toward −x on a floor turns a wheel counter-clockwise
    # (+), and so does rolling up a wall on its left (the same picture turned 90°).
    for name, axis, where in (("floor_to_wall", "x1", "along the floor"), ("wall_to_ceiling", "y1", "up the wall")):
        trn = TRANSITIONS[name]
        phases = planner.plan_phases(trn)
        starts = planner.phase_starts(phases)
        d = abs(planner.plan_pose(trn, phases, starts[1])[axis] - planner.plan_pose(trn, phases, starts[0])[axis])
        want = d / ((phases[0].duration - RAMP_TIME) * trn.wg.wheel_r)
        _, motor = wheel_speeds(trn, phases, np.array([0.5 * (starts[0] + starts[1])]))
        check(f"{name}: rolling {where}, every wheel turns at +{want:.4f} rad/s",
              np.allclose(motor[0], want, rtol=0.0, atol=1e-6), str(np.round(motor[0], 4)))

    # Over an outside corner a wheel hinges with its assembly: halfway through
    # outside's front pivot, front arm A turns with the assembly at
    # turn / (duration − RAMP_TIME), and its motor, which turns with the
    # assembly too, stands still.
    trn = TRANSITIONS["outside"]
    phases = planner.plan_phases(trn, gap=gap_of("outside"))
    starts = planner.phase_starts(phases)
    rolling, motor = wheel_speeds(trn, phases, np.array([0.5 * (starts[1] + starts[2])]))
    want = trn.turn / (phases[1].duration - RAMP_TIME)
    check(f"outside: front arm A hinges over the corner at {want:.4f} rad/s, its motor still",
          close(rolling[0, 0], want, 1e-6) and close(motor[0, 0], 0.0, 1e-6),
          f"rolling {rolling[0, 0]:.4f} rad/s, motor {motor[0, 0]:.1e} rad/s")

    # Rolling without slipping: a wheel on the solid turns at its centre's
    # speed / r, the centre's speed taken straight from the plan's positions.
    h = 1e-4
    for name, trn in TRANSITIONS.items():
        phases = planner.plan_phases(trn, gap=gap_of(name))
        end = planner.phase_starts(phases)[-1]
        times = np.arange(0.0, end, 0.05)
        rolling, _ = wheel_speeds(trn, phases, times)
        on = wheels_touching(trn, phases, times)

        def centres(t):
            p = planner.plan_pose(trn, phases, t)
            return np.array([*wheel_centers(p["x1"], p["y1"], p["theta1"], trn.wg, side=1),
                             *wheel_centers(p["x2"], p["y2"], p["theta2"], trn.wg, side=-1)])

        speed = np.array([np.linalg.norm(centres(min(t + h, end)) - centres(max(t - h, 0.0)), axis=1)
                          / (min(t + h, end) - max(t - h, 0.0)) for t in times])
        worst = np.abs(np.abs(rolling) * trn.wg.wheel_r - speed)[on].max()
        check(f"{name}: wheels on the solid roll without slipping", worst < 1e-6,
              f"worst {worst * 1000:.4f} mm/s over {int(on.sum())} samples, every 0.05 s")


def check_timing_file():
    """The motor calibration's conversions, and a timing file written and read back
    (core/engine/timing_file.py) for a made-up trajectory."""
    print("timing files")
    m = Motor(offset=1.3, sign=-1, ratio=30.0)
    check("calibration: 10° at the joint reads 1.3 − 30 × 10° at the motor",
          close(m.position(np.radians(10.0)), 1.3 - 30.0 * np.radians(10.0)))
    x = np.random.default_rng(2).uniform(-2.0, 2.0, 100)
    check("calibration: angle() and rate() undo position() and speed()",
          np.allclose(m.angle(m.position(x)), x, rtol=0.0, atol=1e-12)
          and np.allclose(m.rate(m.speed(x)), x, rtol=0.0, atol=1e-12))

    # floor_to_wall's plan with the joints swinging in a known way (the angles
    # need not close the chain: only the file is tested).
    trn = TRANSITIONS["floor_to_wall"]
    phases = planner.plan_phases(trn)
    times, poses = planner.sample_poses(trn, phases, SAMPLE_RATE)

    def made_up(display):
        return Trajectory(trn, phases, times, poses, q_from_display(display), [{}] * len(times), 0, None,
                          key="check_constraints")

    display = np.radians([-36.0, -54.0, -54.0, -36.0]) + 0.3 * np.sin(times / 5.0)[:, None] * np.array([1, -1, 1, -1])
    traj = made_up(display)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "floor_to_wall.csv")
        write_timing(traj, path)
        with open(path) as fh:
            first = fh.readline()
        cols, expected = read_timing(path), waypoints(traj)
        current = read_current(traj, path) is not None
        nudged = display.copy()
        nudged[10, 0] += 1e-4
        stale = read_current(made_up(nudged), path) is None
    check("timing file: '#' header lines, then the columns in order",
          first.startswith("#") and list(cols) == list(expected), ", ".join(cols))
    worst = max(np.abs(cols[c] - expected[c]).max() for c in cols)
    check("timing file: read back as written, to its 6 decimals", worst <= 5e-7, f"worst {worst:.1e}")
    check("timing file: the time column is the waypoints' times", np.allclose(cols["time_s"], times, rtol=0.0, atol=5e-7))
    check("timing file: joint columns are the joint angles in motor units",
          all(np.allclose(cols[f"{n}_rad"], m.position(display[:, j]), rtol=0.0, atol=5e-7)
              for j, (n, m) in enumerate(JOINTS.items())))
    speeds = [c for c in cols if c.endswith("_rad_s")]
    check("timing file: the robot is at rest at the first and last waypoint",
          all(cols[c][0] == 0.0 and cols[c][-1] == 0.0 for c in speeds))
    check("timing file: a file is current for its own solve, and stale for any other", current and stale)


def main():
    check_geometry()
    check_environment()
    check_rules()
    check_definitions()
    check_plans()
    check_drive()
    check_wheels()
    check_timing_file()
    if FAILURES:
        print(f"\nFAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
