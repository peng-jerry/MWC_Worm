"""
geometry.py — vectorised 2-D distance tests.

Shared by the robot model (constraints.py) and the environment
(environment.py).  Points are arrays whose last axis has length 2; leading
axes broadcast, so one call tests every candidate configuration at once
(e.g. a (K, 2) array of link ends against a single (2,) surface point).

Every collision test in the project reduces to the distance between two
segments: a capsule (a segment with a radius: a link, an arm, a wheel when
the segment has zero length) clears another capsule by
segment_distance − r1 − r2.
"""

import numpy as np


def cross(u, v):
    """z-component of the 2-D cross product u × v."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def dot(u, v):
    """Dot product u · v over the last axis."""
    return u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1]


def point_segment_distance(p, a, b):
    """Distance from points p to segments a–b (a zero-length segment is a point)."""
    ab = b - a
    length2 = dot(ab, ab)
    t = np.clip(dot(p - a, ab) / np.maximum(length2, 1e-18), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def segments_cross(a, b, c, d):
    """True where segments a–b and c–d cross at a point inside both
    (segments that merely touch do not count)."""
    ab, cd = b - a, d - c
    return ((cross(ab, c - a) * cross(ab, d - a) < 0.0)
            & (cross(cd, a - c) * cross(cd, b - c) < 0.0))


def closest_points(a, b, c, d):
    """
    The nearest pair of points (p on segment a–b, q on segment c–d).
    Zero-length segments (points) are allowed; where the segments cross,
    p = q.  (Ericson, Real-Time Collision Detection, §5.1.9.)
    """
    d1, d2, w = b - a, d - c, a - c
    A, E = dot(d1, d1), dot(d2, d2)
    B, C, F = dot(d1, d2), dot(d1, w), dot(d2, w)
    eps = 1e-18
    safe_A, safe_E = np.where(A > eps, A, 1.0), np.where(E > eps, E, 1.0)
    denom = A * E - B * B
    # Both segments proper: nearest points of the two lines, clamped.
    s = np.where(denom > eps, np.clip((B * F - C * E) / np.where(denom > eps, denom, 1.0), 0.0, 1.0), 0.0)
    t = (B * s + F) / safe_E
    s = np.where(t < 0.0, np.clip(-C / safe_A, 0.0, 1.0),
                 np.where(t > 1.0, np.clip((B - C) / safe_A, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    # a–b is a point: nearest point of c–d to it.
    s = np.where(A <= eps, 0.0, s)
    t = np.where(A <= eps, np.clip(F / safe_E, 0.0, 1.0), t)
    # c–d is a point: nearest point of a–b to it.
    s = np.where((E <= eps) & (A > eps), np.clip(-C / safe_A, 0.0, 1.0), s)
    t = np.where(E <= eps, 0.0, t)
    return a + s[..., None] * d1, c + t[..., None] * d2


def segment_distance(a, b, c, d):
    """Shortest distance between segments a–b and c–d (0 where they meet)."""
    p, q = closest_points(a, b, c, d)
    return np.linalg.norm(p - q, axis=-1)


def clip_polygon(polygon, origin, normal):
    """
    The part of a convex polygon (list of (x, y)) on the side n·(x − o) <= 0
    of the line through `origin` with normal `normal` (Sutherland–Hodgman).
    Used for drawing the environment.
    """
    o, n = np.asarray(origin, float), np.asarray(normal, float)
    out = []
    for i in range(len(polygon)):
        p, q = np.asarray(polygon[i], float), np.asarray(polygon[(i + 1) % len(polygon)], float)
        hp, hq = n @ (p - o), n @ (q - o)
        if hp <= 0:
            out.append(p)
        if hp * hq < 0:
            out.append(p + hp / (hp - hq) * (q - p))
    return out
