"""
Forward kinematics for the 4-joint, 3-link planar linkage.

Chain topology:
  front assembly (x1,y1,theta1)
    -> q1 (relative to theta1)
    -> link l1
    -> q2 (relative to previous direction)
    -> link l2
    -> q3 (relative to previous direction)
    -> link l3
    -> q4 (relative to previous direction)
    -> back assembly (x2,y2,theta2)

All angles in radians.

Display convention (wheel_omega.py, visualize.py, main.py):
  q1_display = q1 - π/2   (0° = link straight up from front calf)
  q4_display = q4 - π/2   (0° = last link directly above back calf, i.e. P2 above P3)
  q2, q3 are displayed as-is.
"""

import numpy as np


def forward_kinematics(q, x1, y1, theta1, l1, l2, l3):
    """
    Compute joint positions and final orientation from joint angles.

    Parameters
    ----------
    q : array-like of shape (4,)
        Joint angles [q1, q2, q3, q4] in radians (each relative to prior segment).
    x1, y1 : float
        Position of the front assembly attachment point.
    theta1 : float
        Orientation of the front assembly in radians.
    l1, l2, l3 : float
        Lengths of the three links.

    Returns
    -------
    positions : ndarray of shape (4, 2)
        Positions of the four joint nodes [P0, P1, P2, P3].
        P0 is the front attachment; P3 is the back attachment.
    theta_end : float
        Orientation of the back assembly implied by the chain.
    """
    q1, q2, q3, q4 = q

    # Cumulative absolute link directions
    phi1 = theta1 + q1
    phi2 = phi1 + q2
    phi3 = phi2 + q3

    # Joint positions along the chain
    p0 = np.array([x1, y1])
    p1 = p0 + l1 * np.array([np.cos(phi1), np.sin(phi1)])
    p2 = p1 + l2 * np.array([np.cos(phi2), np.sin(phi2)])
    p3 = p2 + l3 * np.array([np.cos(phi3), np.sin(phi3)])

    # Orientation of the back assembly
    theta_end = phi3 + q4

    return np.stack([p0, p1, p2, p3]), theta_end


def q_display(q):
    """
    Convert internal solver q to display convention:
      q1_display = q1 - π/2  (0° when link is straight up from front calf)
      q4_display = q4 - π/2  (0° when last link directly above back calf, P2 above P3)
      q2, q3 unchanged.
    Result is wrapped to [-π, π].
    """
    qd = np.asarray(q, dtype=float).copy()
    qd[0] = (qd[0] - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    qd[3] = (qd[3] - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    return qd


def reachability_check(x1, y1, x2, y2, l1, l2, l3):
    """
    Return True if (x2,y2) is within the total reach of the chain from (x1,y1).
    This is a necessary but not sufficient condition for a solution to exist.
    """
    dist = np.hypot(x2 - x1, y2 - y1)
    total_len = l1 + l2 + l3
    return dist <= total_len
