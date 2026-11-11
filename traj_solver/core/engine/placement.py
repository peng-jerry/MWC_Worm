"""
placement.py — where an assembly goes, from geometry alone: resting on a
surface, pivoting across the transition, or at a set gap from the other one;
and whether a pose of the two assemblies is allowed.

  rest_point         chain joint of an assembly resting on a surface
  pivot_pose         chain joint and orientation partway through a pivot
  roll_to_gap        chain joint on a surface, a set gap from the other one
  along              how far a chain joint is along a surface
  check_assemblies   what is wrong with a pose of the two assemblies, if anything
"""

import math

import numpy as np

from .constraints import assembly_parts, wheel_centers
from .geometry import vec
from .transition import PlanError, outline, surface_orientation
from ..params import FEASIBILITY_TOL, HARD_CLEARANCE


def _up(theta):
    """The 'up' direction of an assembly at orientation theta (away from its wheels)."""
    return np.array([-math.sin(theta), math.cos(theta)])


def wheel_offsets(theta, wg, side):
    """Wheel centres relative to the chain joint, shape (2, 2): arm A, arm B."""
    return np.array(wheel_centers(0.0, 0.0, theta, wg, side))


def rest_point(trn, surface, u, side):
    """
    Chain joint of an assembly resting on `surface`, u metres from the
    surface's end along it.  The lowest wheel just touches it.
    """
    n, t = vec(surface.normal), vec(surface.tangent)
    W = wheel_offsets(surface_orientation(surface), trn.wg, side)
    height = trn.wg.wheel_r - np.min(W @ n)        # chain joint height above the surface
    return vec(surface.origin) + u * t + height * n


def along(surface, joint):
    """How far a chain joint is from the end of `surface`, measured along it."""
    return float((joint - vec(surface.origin)) @ vec(surface.tangent))


def pivot_pose(trn, side, f):
    """
    Chain joint and orientation of an assembly pivoting across the transition,
    f ∈ [0, 1] through the pivot.

    Inside corner: the orientation turns at an even rate and the assembly
    stays pressed into both surfaces — for each surface the wheel nearest to
    it just touches it.  That fixes the joint's distance from both surfaces.

    Outside corner or edge: the trailing wheel (the last to leave the
    approach surface) rolls around the solid's outline (outline): its centre
    stays one wheel radius from the solid, f running evenly along that path,
    while the orientation turns at an even rate.  Round a single corner the
    two agree: the assembly stands square on a surface that turns with it.
    Where the solid has an end face, the wheel rolls along it while the
    assembly keeps turning, instead of the turn pausing there; the assembly
    leans a little off square near the corners (under 4° at the thin edge).
    """
    thA = surface_orientation(trn.approach)
    r = trn.wg.wheel_r
    if trn.corner == "inside":
        theta = thA + f * trn.turn
        W = wheel_offsets(theta, trn.wg, side)
        O = vec(trn.approach.origin)
        nA, nB = vec(trn.approach.normal), vec(trn.depart.normal)
        rhs = [nA @ O + r - np.min(W @ nA), nB @ O + r - np.min(W @ nB)]
        return np.linalg.solve(np.array([nA, nB]), rhs), theta

    trailing = int(np.argmax(wheel_offsets(thA, trn.wg, side) @ vec(trn.approach.tangent)))
    pieces = outline(trn)
    s = f * sum(length for _, length in pieces)
    for k, (piece, length) in enumerate(pieces):
        if s <= length or k == len(pieces) - 1:
            g = min(s / length, 1.0) if length > 0 else 1.0
            break
        s -= length
    if piece[0] == "turn":
        _, corner, th0, th1 = piece
        centre = corner + r * _up(th0 + g * (th1 - th0))
    else:
        _, start, end, th = piece
        centre = start + g * (end - start) + r * _up(th)
    theta = thA + f * trn.turn
    return centre - wheel_offsets(theta, trn.wg, side)[trailing], theta


def roll_to_gap(trn, surface, side, other, gap):
    """
    Chain joint of an assembly resting on `surface` exactly `gap` from
    the chain joint `other`, choosing the position farther from the
    surface's end.
    """
    base = rest_point(trn, surface, 0.0, side)
    t = vec(surface.tangent)
    # |base + u·t - other| = gap  →  u² + 2bu + c = 0
    d = base - other
    b = t @ d
    c = d @ d - gap ** 2
    disc = b * b - c
    if disc < 0:
        raise PlanError(f"{trn.name}: no position on the surface is {gap:.3f} m from the "
                         f"other chain joint")
    return base + (-b + math.sqrt(disc)) * t


def check_assemblies(trn, pose):
    """
    Problems with where a pose puts the two wheel assemblies, as
    {description: gap (m)}; empty when there are none.

      * no part of either assembly may enter the solid (wheels touch it,
        so a gap down to -FEASIBILITY_TOL is allowed)
      * the two assemblies stay HARD_CLEARANCE apart, measured through free
        space (on opposite faces of a plate, around its end)
    """
    env = trn.env
    front = assembly_parts(pose["x1"], pose["y1"], pose["theta1"], trn.wg, +1)
    back = assembly_parts(pose["x2"], pose["y2"], pose["theta2"], trn.wg, -1)
    names = [f"front {n}" for n in front] + [f"back {n}" for n in back]
    a, b, r = (np.array([part[i] for part in [*front.values(), *back.values()]]) for i in range(3))
    problems = {}
    for name, gap in zip(names, env.clearance(a, b, r)):
        if gap < -FEASIBILITY_TOL:
            problems[f"{name} in the surface"] = float(gap)
    # Every front part against every back part: (7, 1) against (1, 7).
    n = len(front)
    gap = (env.separation(a[:n, None], b[:n, None], a[None, n:], b[None, n:],
                          within=HARD_CLEARANCE + r[:n, None] + r[None, n:])
           - r[:n, None] - r[None, n:]).min()
    if gap < HARD_CLEARANCE - FEASIBILITY_TOL:
        problems["front and back assemblies too close"] = float(gap)
    return problems
