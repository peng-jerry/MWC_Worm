"""
solver.py — closed-chain inverse kinematics for one pair of assembly poses.

The chain
---------
The front assembly sits at (x1, y1, θ1) and the back assembly at (x2, y2, θ2).
Four joints connect them through three links:

    P0 --l1-- P1 --l2-- P2 --l3-- P3
    q1 at P0 (relative to θ1), q2 at P1, q3 at P2, q4 at P3 (sets the back
    assembly's orientation)

Closing the chain fixes three things — P3's x and y, and the back assembly's
orientation θ2 — while there are four joints, so one degree of freedom is
left free.  This module parametrises that free DOF by q1:

  * Fix q1.  P1 is then known, and links l2 and l3 must reach from P1 to P3:
    a two-link problem solved exactly by the law of cosines.  It has two
    solutions, the elbow branches (sign = +1 gives q3 <= 0, the inward bend
    every scenario uses).
  * q4 then follows from the orientation, so every (q1, branch) candidate
    closes the chain exactly — there is no kinematic residual to minimise.

Choosing q1 is therefore a 1-D search, which trajectory_ik.py does for every
frame of a transition at once.

    solve_branch_batch   closed-form q for many q1 values on one branch
    posture_cost         distance from the neutral posture (tie-breaker)
"""

import numpy as np

from kinematics import q_display


def _wrap(a):
    """Wrap angles to [-π, π)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def solve_branch_batch(q1, sign, x1, y1, theta1, x2, y2, theta2, l1, l2, l3):
    """
    Joint angles for an array of q1 values on one elbow branch.

    Parameters
    ----------
    q1 : array-like (K,)
        Values of the free joint q1 (radians).
    sign : +1.0 or -1.0
        Elbow branch of the l2/l3 sub-chain; +1 gives q3 <= 0.
    x1, y1, theta1, x2, y2, theta2 : float
        Assembly poses (chain joints and orientations).
    l1, l2, l3 : float
        Link lengths.

    Returns
    -------
    Q : ndarray (K, 4)
        [q1, q2, q3, q4] per row, wrapped to [-π, π).  Rows are NaN where the
        sub-chain cannot reach P3 from that q1.  Every finite row closes the
        chain exactly.
    """
    q1 = np.asarray(q1, dtype=float)
    phi1 = theta1 + q1                       # absolute direction of link l1
    # Vector from P1 to P3: the target for the two-link sub-chain (l2, l3).
    px = x2 - (x1 + l1 * np.cos(phi1))
    py = y2 - (y1 + l1 * np.sin(phi1))
    d2 = px ** 2 + py ** 2
    d = np.sqrt(d2)

    # Law of cosines for the elbow angle: cos(q3) = (d² - l2² - l3²) / (2 l2 l3).
    # |cos q3| > 1 means P3 is out of the sub-chain's reach for this q1.
    cos_q3 = (d2 - l2 ** 2 - l3 ** 2) / (2.0 * l2 * l3)
    reachable = np.abs(cos_q3) <= 1.0

    # Law of cosines for the angle beta between P1→P3 and link l2:
    # cos(beta) = (d² + l2² - l3²) / (2 d l2).  Clipped for rounding at full
    # extension; d = 0 only happens for unreachable rows, which are masked.
    alpha = np.arctan2(py, px)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_beta = np.clip((d2 + l2 ** 2 - l3 ** 2) / (2.0 * d * l2), -1.0, 1.0)
    beta = np.arccos(cos_beta)

    # Branch: l2 turns by +beta (sign +1) or -beta from P1→P3, and the elbow
    # bends back by the opposite sign so l3 ends on P3.
    phi2 = alpha + sign * beta
    q3 = -sign * np.arccos(np.clip(cos_q3, -1.0, 1.0))
    phi3 = phi2 + q3

    # Relative joint angles; q4 makes the back assembly's orientation theta2.
    Q = _wrap(np.stack([q1, phi2 - phi1, phi3 - phi2, theta2 - phi3], axis=-1))
    Q[~reachable] = np.nan
    return Q


def posture_cost(Q):
    """
    ||q_display||² for candidate rows Q (K, 4): how far each is from the
    neutral posture (q1/q4 straight up from the calf, q2/q3 straight), the
    centre of the joint range.  Used as a tie-breaker.
    """
    qd = q_display(Q)
    return np.sum(qd * qd, axis=-1)
