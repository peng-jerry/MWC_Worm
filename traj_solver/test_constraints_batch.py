"""
test_constraints_batch.py — check constraints_batch.evaluate_batch against the
scalar reference in constraints.py, term by term, for every constraint set.

Poses are sampled through every phase of every transition (scenarios.py,
POSE_RATE per second).  At each pose the candidates are the closed-chain q1
grid the solver actually scores, plus random joint vectors that exercise the
sign and joint-limit terms.  Each pose is tested under all constraint sets.

Usage:
    python test_constraints_batch.py          # ~1.5 min: 180-point grid + 100 random per pose
    python test_constraints_batch.py --full   # ~7 min:   720-point grid + 400 random per pose
Exits non-zero if any value differs by more than the tolerance.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constraints as C
from scenarios import SCENARIOS, l1, l2, l3, plan_phases, sample_poses
from constraints_batch import evaluate_batch, active_terms
from kinematics import forward_kinematics
from solver import solve_branch_batch

ATOL = 1e-9      # absolute tolerance on raw terms (O(1) values)
RTOL = 1e-12     # relative tolerance on weighted totals (up to ~1e5)
FULL = "--full" in sys.argv
N_GRID = 720 if FULL else 180
N_RANDOM = 400 if FULL else 100
POSE_RATE = 0.75     # test poses per second of each transition


def scalar_terms(q, pose, wg, cs, wall_x, ceiling_y):
    """Reference values from the scalar functions, grouped where the batch
    evaluator splits one scalar function into two named terms."""
    x1, y1, th1 = pose[:3]
    P, th_end = forward_kinematics(q, x1, y1, th1, l1, l2, l3)
    wraps = cs in ("outside", "outside_exact", "thin_edge", "thin_edge_exact")
    ref = {
        "collision": C._collision_penalty(P, th_end, x1, y1, th1, wg, 0.001 if wraps else 1.0),
        "overlap":   C._no_overlap_penalty(P, th_end, x1, y1, th1, wg),
        "joint_soft_limit": (8.0 if cs in ("wall", "wall_exact", "ceiling", "ceiling_exact")
                             else 1.0) * C._joint_limit_penalty(q),
        "joint_hard_limit": C._joint_hard_limit_penalty(q),
    }
    if cs in ("wall", "wall_exact", "ceiling", "ceiling_exact", "thin_edge"):
        ref["bend_sign"] = C._joint_sign_penalty(q, require_negative=True)
    elif cs == "thin_edge_exact":
        ref["bend_sign"] = C._joint_sign_penalty(q, require_negative=True, constant=0.0)
    elif cs in ("outside", "outside_exact"):
        q3 = float(q[2])
        ref["bend_sign"] = (q3 * q3 if cs == "outside_exact" else 1.0 + q3 * q3) if q3 > 0 else 0.0
    if not wraps:
        ref["v_cone"] = C._v_cone_entry_penalty(P, th_end, th1, wg)
    if cs in ("floor", "wall", "wall_exact"):
        ref["floor"] = C._floor_penalty(P)
    if cs in ("wall", "wall_exact", "ceiling"):
        ref["wall"] = C._wall_penalty(P, th_end, x1, y1, th1, wg, wall_x)
    if cs in ("ceiling", "ceiling_exact"):
        ref["ceiling"] = C._ceiling_penalty(P, th_end, x1, y1, th1, wg, ceiling_y)
    if cs in ("outside", "outside_exact"):
        ref["outside_corner"] = C._outside_corner_penalty(P, wall_x, ceiling_y)
        ref["outside_surface_crossing"] = C._outside_surface_crossing_penalty(P, wall_x, ceiling_y)
        ref["outside_clearance"] = C._outside_clearance_penalty(P, wall_x, ceiling_y)
    if wraps and cs.startswith("thin"):
        ref["thin_edge_crossing"] = C._thin_edge_crossing_penalty(P, wall_x, ceiling_y)
    return ref


def batch_grouped(pb):
    """Batch terms regrouped to match scalar_terms."""
    t = dict(pb.terms)
    if t:
        t["collision"] = t.pop("wheel_penetration") + t.pop("bar_crossing")
        t["overlap"] = t.pop("wheel_overlap") + t.pop("link_self_crossing")
    return t


def main():
    rng = np.random.default_rng(0)
    worst_term, worst_total = 0.0, 0.0
    n_values, failures = 0, []

    for sc, scn in SCENARIOS.items():
        wg = scn.wg
        _, poses = sample_poses(scn, plan_phases(scn), POSE_RATE)
        for k, p in enumerate(poses):
            pose = (p["x1"], p["y1"], p["theta1"], p["x2"], p["y2"], p["theta2"])
            wall_x, ceiling_y = p["wall_x"], p["ceiling_y"]
            q1 = np.linspace(-np.pi, np.pi, N_GRID, endpoint=False)
            grid = np.concatenate([solve_branch_batch(q1, s, *pose, l1, l2, l3)
                                   for s in (+1.0, -1.0)])
            grid = grid[~np.isnan(grid[:, 0])]
            Q = np.concatenate([grid, rng.uniform(-np.pi, np.pi, (N_RANDOM, 4))])

            for cs in C.CONSTRAINT_SETS:
                pb = evaluate_batch(Q, *pose, l1, l2, l3, wg, cs,
                                    wall_x=wall_x, ceiling_y=ceiling_y)
                assert list(pb.kinds) == list(active_terms(cs))
                grouped = batch_grouped(pb)
                for i, q in enumerate(Q):
                    ref_total = C.total_penalty(q, *pose, l1, l2, l3, wg, cs,
                                                wall_x=wall_x, ceiling_y=ceiling_y)
                    d_total = abs(pb.total[i] - ref_total)
                    worst_total = max(worst_total, d_total / max(1.0, abs(ref_total)))
                    if d_total > ATOL + RTOL * abs(ref_total):
                        failures.append((sc, k, cs, i, "total", pb.total[i], ref_total))
                    if cs == "none":
                        n_values += 1
                        continue
                    ref = scalar_terms(q, pose, wg, cs, wall_x, ceiling_y)
                    if set(ref) != set(grouped):
                        failures.append((sc, k, cs, i, "term set",
                                         sorted(grouped), sorted(ref)))
                        continue
                    for name, rv in ref.items():
                        d = abs(grouped[name][i] - rv)
                        worst_term = max(worst_term, d)
                        if d > ATOL:
                            failures.append((sc, k, cs, i, name, grouped[name][i], rv))
                    n_values += 1 + len(ref)

    print(f"compared {n_values} values across {len(C.CONSTRAINT_SETS)} constraint sets")
    print(f"worst term difference      {worst_term:.2e}  (tolerance {ATOL:.0e})")
    print(f"worst total difference     {worst_total:.2e}  (relative, floor 1)")
    if failures:
        print(f"FAILED: {len(failures)} mismatches; first 10:")
        for f in failures[:10]:
            print("  ", f)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
