"""
environment.py — the solid around a transition, built from its two surfaces.

A transition is described by two flat surfaces (Surface): the one the robot
arrives on (approach) and the one it leaves on (depart).  Each surface is a
half-line that starts at its `origin` — where it ends at the transition —
and runs along `tangent`, with `normal` pointing out into free space.

  inside corner    floor → wall, wall → ceiling.  Both surfaces start at the
                   same corner.  Free space is the wedge in front of both
                   surfaces; everything else is solid.
  outside corner   top of a block → its side wall, top → underside of a
  or edge          plate.  The solid is the wedge behind both surfaces.  If
                   the two origins differ (a plate with thickness), the
                   segment between them is the solid's end face.

Environment answers what the planner, the solver and the drawings need:
  clearance(a, b, r)   distance from capsules (segments a–b, radius r) to the
                       solid; negative when they reach into it
  separation(…)        distance between two parts through free space (parts
                       on opposite faces of a plate are measured around it)
  faces                the flat faces a wheel can touch: both surfaces, plus
                       the end face if there is one
  solid_polygons       the solid cut to a view box, for drawing
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from geometry import (clip_polygon, closest_points, dot, point_segment_distance,
                      segment_distance, segments_cross)

FAR = 100.0     # m; the half-line surfaces are handled as segments this long


@dataclass(frozen=True)
class Surface:
    """
    A flat surface that ends at the transition.

    origin   where the surface ends (the corner, or the edge of a plate)
    tangent  unit vector along the surface, pointing away from the origin
             into the part the robot drives on
    normal   unit vector out of the surface, into free space
    """
    origin: Tuple[float, float]
    tangent: Tuple[float, float]
    normal: Tuple[float, float]

    def __post_init__(self):
        t, n = np.asarray(self.tangent, float), np.asarray(self.normal, float)
        if abs(np.linalg.norm(t) - 1) > 1e-9 or abs(np.linalg.norm(n) - 1) > 1e-9 or abs(t @ n) > 1e-9:
            raise ValueError("surface tangent and normal must be perpendicular unit vectors")


@dataclass(frozen=True)
class Face:
    """A flat face of the solid: `length` metres from `origin` along `tangent`
    (inf for the two surfaces), with `normal` pointing into free space."""
    origin: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    length: float


class Environment:
    """The solid around one transition (see the module docstring)."""

    def __init__(self, approach: Surface, depart: Surface, corner: str):
        if corner not in ("inside", "outside"):
            raise ValueError("corner must be 'inside' or 'outside'")
        self.corner = corner
        oA, oB = np.asarray(approach.origin, float), np.asarray(depart.origin, float)
        faces = [Face(oA, np.asarray(approach.tangent, float), np.asarray(approach.normal, float), math.inf)]
        if np.linalg.norm(oB - oA) > 1e-12:
            if corner == "inside":
                raise ValueError("the two surfaces of an inside corner must start at the same point")
            # End face from oA to oB.  Its normal is the perpendicular that
            # points away from the solid, which lies along the approach tangent.
            tE = (oB - oA) / np.linalg.norm(oB - oA)
            nE = np.array([-tE[1], tE[0]])
            if nE @ faces[0].tangent > 0:
                nE = -nE
            faces.append(Face(oA, tE, nE, float(np.linalg.norm(oB - oA))))
        faces.append(Face(oB, np.asarray(depart.tangent, float), np.asarray(depart.normal, float), math.inf))
        self.faces = faces
        self.corners = [oA] if len(faces) == 2 else [oA, oB]     # the solid's corner points
        self._boundary = [(f.origin, f.origin + min(f.length, FAR) * f.tangent) for f in faces]

    # ── Clearance ─────────────────────────────────────────────────────────────

    def clearance(self, a, b, r):
        """
        Distance (m) from capsules — segments a–b thickened by radius r — to
        the solid.  Negative when a capsule reaches into it; the more negative,
        the deeper (only the sign and the ranking matter for a capsule whose
        centre line is inside).  a, b broadcast against each other: (K, 2)
        and (2,) give (K,).
        """
        a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
        dist = np.full(a.shape[:-1], np.inf)
        for c, d in self._boundary:
            dist = np.minimum(dist, segment_distance(a, b, c, d))
        depth = self._depth(a, b)
        return np.where(depth > 0.0, -depth, dist) - r

    def separation(self, a, b, c, d, within=np.inf):
        """
        Distance between segments a–b and c–d (two parts' centre lines)
        through free space.

        Normally the straight-line distance.  But two parts on opposite faces
        of a plate cannot touch through it: when the straight line between
        their nearest points passes through the solid, the distance is taken
        around the solid instead, past its nearest corner (a little short
        of the true path when it must pass two corners — on the safe side).
        Only distances below `within` are checked for this, since the
        detour can only make a distance longer.
        """
        p, q = closest_points(a, b, c, d)
        dist = np.linalg.norm(p - q, axis=-1)
        if self.corner == "inside":
            return dist           # free space is convex: every straight line stays in it
        near = dist < within
        if not np.any(near):
            return dist
        # Work on flat (n, 2) arrays so single pairs and batches go the same way.
        shape = dist.shape
        a, b, c, d, p, q = (np.broadcast_to(v, p.shape).reshape(-1, 2) for v in (a, b, c, d, p, q))
        dist, near = dist.reshape(-1).copy(), near.reshape(-1)
        hidden = np.zeros(len(dist), dtype=bool)
        hidden[near] = self._depth(p[near], q[near]) > 0.0
        if np.any(hidden):
            dist[hidden] = np.minimum.reduce([point_segment_distance(v, a[hidden], b[hidden])
                                              + point_segment_distance(v, c[hidden], d[hidden])
                                              for v in self.corners])
        return dist.reshape(shape)

    def _depth(self, a, b):
        """How far the segment a–b reaches into the solid (0 if it stays out)."""
        if self.corner == "inside":
            # Free space is the wedge in front of both surfaces, and it is
            # convex, so the segment stays free exactly when both ends do.
            h = [dot(p - f.origin, f.normal) for f in self.faces for p in (a, b)]
            return np.maximum(0.0, -np.minimum.reduce(h))
        # Outside corner or edge: the solid is the convex region behind every
        # face.  Clip the segment to it (Cyrus–Beck): the part inside has
        # parameter range [lo, hi] along a→b.
        lo, hi = np.zeros(a.shape[:-1]), np.ones(a.shape[:-1])
        empty = np.zeros(a.shape[:-1], dtype=bool)
        # How far each end lies behind its nearest face (negative when outside).
        da, db = np.full(a.shape[:-1], np.inf), np.full(a.shape[:-1], np.inf)
        for f in self.faces:
            ha, hb = dot(a - f.origin, f.normal), dot(b - f.origin, f.normal)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ha / (ha - hb)
            empty |= (ha > 0.0) & (hb > 0.0)
            lo = np.where((ha > 0.0) & (hb <= 0.0), np.maximum(lo, t), lo)
            hi = np.where((ha <= 0.0) & (hb > 0.0), np.minimum(hi, t), hi)
            da, db = np.minimum(da, -ha), np.minimum(db, -hb)
        inside = np.where(empty, 0.0, np.maximum(0.0, hi - lo)) * np.linalg.norm(b - a, axis=-1)
        depth = np.maximum(inside, np.maximum(np.maximum(da, db), 0.0))
        # A solid of zero thickness has no inside to clip to; crossing one of
        # its faces still counts.
        for c, d in self._boundary:
            depth = np.where(segments_cross(a, b, c, d), np.maximum(depth, 1e-9), depth)
        return depth

    # ── Drawing ───────────────────────────────────────────────────────────────

    def solid_polygons(self, xlim, ylim):
        """The solid cut to the box xlim × ylim, as a list of polygons."""
        box = [(xlim[0], ylim[0]), (xlim[1], ylim[0]), (xlim[1], ylim[1]), (xlim[0], ylim[1])]
        if self.corner == "inside":
            # Solid = everything behind either surface: one piece per surface.
            return [clip_polygon(box, f.origin, f.normal) for f in self.faces]
        poly = box
        for f in self.faces:
            if poly:
                poly = clip_polygon(poly, f.origin, f.normal)
        return [poly] if poly else []

    def face_segments(self):
        """Each face as a drawable segment (the surfaces cut at FAR)."""
        return list(self._boundary)
