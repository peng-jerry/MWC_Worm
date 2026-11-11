"""
transition.py — what defines a transition: its two surfaces, how the robot
turns between them, and the checks on that definition.

The transitions themselves are the entries of core/transitions.py.

  PlanError             a transition that cannot be planned or solved
  Transition            one definition, and what follows from it: which
                        assembly leads, and whether its partner follows
  surface_orientation   an assembly's orientation resting on a surface
  outline               the path of a wheel around an outside corner or edge
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple

import numpy as np

from .environment import Environment, Surface
from .geometry import vec
from .kinematics import wrap
from ..robot import WheelGeometry


class PlanError(ValueError):
    """
    A transition that cannot be planned or solved, with a message naming it
    and saying why (out of the chain's reach, an assembly would hit something,
    no valid joint angles, …).  The scripts report it and go on to the next
    transition.  A mistake in a transition's definition is a plain ValueError
    instead, raised as soon as core/transitions.py is read.
    """


@dataclass(frozen=True)
class Transition:
    """
    One transition: the two surfaces and how the robot turns between them.
    How to define one is explained in core/transitions.py.

    name, label       the key in TRANSITIONS (and the output files' name), and
                      a title for plots and timing files
    approach, depart  the surface the robot arrives on and leaves on
                      (environment.Surface); their origins are where they
                      end at the transition
    turn              each assembly's rotation during its pivot (rad, + =
                      CCW); takes the approach orientation to the departure
                      orientation
    corner            "inside" (concave corner) or "outside" (convex corner
                      or the end of a plate); decides how an assembly pivots
                      and whether its partner parks or follows
    wg                the arm layout (robot.WheelGeometry)
    xlim, ylim        drawing only: fixed (low, high) plot limits; None on
                      either side fits that side to the robot

    Derived: lead (which assembly crosses first), output (the animation's
    file name), env (the solid), partner_follows, rolls_together.
    """
    name: str
    label: str
    approach: Surface
    depart: Surface
    turn: float
    corner: str
    wg: WheelGeometry
    xlim: Tuple[Optional[float], Optional[float]] = (None, None)
    ylim: Tuple[Optional[float], Optional[float]] = (None, None)

    @cached_property
    def env(self) -> Environment:
        """The solid around the transition (environment.Environment).  Built
        once: the planner asks for it tens of thousands of times per solve."""
        return Environment(self.approach, self.depart, self.corner)

    @property
    def lead(self):
        """
        "front" or "back": the assembly that crosses first.  It is not a
        choice.  The elbows bend one way only (q2, q3 ≤ 0), so the chain
        arches to the left of the direction from the front assembly to the
        back one, and it must arch away from the surface: on the approach
        surface, that decides which assembly is nearer the transition.
        __post_init__ checks that the departure surface agrees.
        """
        return "front" if _free_side(self.approach) > 0 else "back"

    @property
    def output(self):
        """The animation's file name (animate.py writes it to outputs/gifs/)."""
        return f"{self.name}.mp4"

    @property
    def partner_follows(self):
        """
        Whether the partner of a pivoting assembly rolls after it (True) or
        stays parked (False).  At an inside corner the pivoting assembly stays
        tucked into the corner, so its chain joint barely moves and the
        partner can stay where it is.  At an outside corner or edge it swings
        around the outside of the transition, carrying its chain joint
        away — beyond the chain's reach if the partner stayed put.
        """
        return self.corner == "outside"

    @property
    def rolls_together(self):
        """
        Whether both assemblies always roll at the same time between the two
        pivots (True: one phase) or, by preference, one after the other
        (False: the leading one rolls clear, then the trailing one rolls up).
        At an inside corner, rolling both at once would bring them toward the
        corner together and fold the chain up between the two surfaces — but
        at a corner wider than a right angle, rolling them in turn can stretch
        the chain beyond its reach, and then plan_phases has them roll
        together after all.  At an outside corner or
        edge, rolling both at once keeps the gap at or below the follow gap
        all the way (each moves in a straight line, so the gap is largest at
        the ends, where it is the follow gap); rolling them in turn would
        stretch the chain over the transition instead.
        """
        return self.corner == "outside"

    def __post_init__(self):
        """Check the definition as soon as it is read, with a message naming the transition."""
        if self.corner not in ("inside", "outside"):
            raise ValueError(f"{self.name}: corner must be 'inside' or 'outside', not {self.corner!r}")
        try:
            self.env                                # validates the surfaces (environment.Environment)
        except ValueError as e:
            raise ValueError(f"{self.name}: {e}") from None
        if abs(wrap(surface_orientation(self.approach) + self.turn
                     - surface_orientation(self.depart))) > 1e-9:
            raise ValueError(f"{self.name}: turn {np.degrees(self.turn):.1f}° does not take the "
                             f"approach orientation to the departure orientation")
        parallel = abs(abs(vec(self.approach.normal) @ vec(self.depart.normal)) - 1.0) < 1e-9
        if not parallel and abs(self.turn) >= math.pi:
            raise ValueError(f"{self.name}: at a corner the turn is the short way round "
                             f"(|turn| < 180°), not {np.degrees(self.turn):.1f}°")
        # Leaving, the leading assembly is the one farther from the transition,
        # so the departure surface must have its free side the other way round.
        if (_free_side(self.depart) < 0) != (self.lead == "front"):
            raise ValueError(f"{self.name}: the chain cannot arch away from both surfaces — check "
                             f"that each tangent points away from the transition, along the part "
                             f"the robot drives on, and each normal into free space")
        if self.corner == "outside":
            outline(self)                          # an end face must lie between the two surfaces


def _free_side(surface):
    """
    +1 if the free space above `surface` is on the left of its tangent (the
    direction away from the transition), −1 if on the right — the z-component
    of tangent × normal.  Decides which assembly must lead (Transition.lead).
    """
    t, n = surface.tangent, surface.normal
    return float(np.sign(t[0] * n[1] - t[1] * n[0]))


def surface_orientation(surface):
    """Assembly orientation θ for resting on `surface` (its wheels point into it)."""
    n = surface.normal
    return math.atan2(n[1], n[0]) - math.pi / 2


def outline(trn):
    """
    The path of the trailing wheel's centre around an outside corner or edge,
    as pieces ("turn", corner point, θ from, θ to) and ("roll", from, to, θ),
    each with its length.  The wheel turns round the approach surface's end,
    rolls along the end face if the solid has one, and turns round the
    depart surface's end.
    """
    r = trn.wg.wheel_r
    thA = surface_orientation(trn.approach)
    oA, oB = vec(trn.approach.origin), vec(trn.depart.origin)
    end_faces = [f for f in trn.env.faces if np.isfinite(f.length)]
    if not end_faces:
        return [(("turn", oA, thA, thA + trn.turn), r * abs(trn.turn))]
    nE = end_faces[0].normal
    # The end face's orientation, reached from the approach orientation by
    # turning the same way as the whole pivot.
    dE = wrap(math.atan2(nE[1], nE[0]) - math.pi / 2 - thA)
    if dE * trn.turn < 0 or abs(dE) > abs(trn.turn) + 1e-9:
        raise ValueError(f"{trn.name}: the end face does not lie between the two surfaces")
    thE = thA + dE
    return [(("turn", oA, thA, thE), r * abs(dE)),
            (("roll", oA, oB, thE), float(np.linalg.norm(oB - oA))),
            (("turn", oB, thE, thA + trn.turn), r * abs(trn.turn - dE))]
