"""
hermite.py — the cubic Hermite curve the robot plays between two waypoints.

The solver (trajectory.py), the playback check (checks/check_playback.py) and
the robot itself (Motor/playback.py) all use this one curve, so this module
imports nothing and works on plain numbers and numpy arrays alike.

From a waypoint at position q0 and speed v0 to the next at q1 and v1, T seconds
later, at fraction a ∈ [0, 1] of the way:

    q(a) = h00(a)·q0 + h10(a)·T·v0 + h01(a)·q1 + h11(a)·T·v1
"""


def hermite(q0, v0, q1, v1, T, a):
    """Position on the curve at fraction a."""
    a2, a3 = a * a, a * a * a
    h00, h10 = 2 * a3 - 3 * a2 + 1, a3 - 2 * a2 + a
    h01, h11 = -2 * a3 + 3 * a2, a3 - a2
    return h00 * q0 + h10 * T * v0 + h01 * q1 + h11 * T * v1


def hermite_speed(q0, v0, q1, v1, T, a):
    """Speed along the curve at fraction a."""
    return (6 * a * a - 6 * a) * (q0 - q1) / T + (3 * a * a - 4 * a + 1) * v0 + (3 * a * a - 2 * a) * v1


def hermite_acceleration(q0, v0, q1, v1, T, a):
    """Acceleration along the curve at fraction a."""
    return ((12 * a - 6) * (q0 - q1) / T + (6 * a - 4) * v0 + (6 * a - 2) * v1) / T
