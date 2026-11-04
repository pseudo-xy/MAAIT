"""Manual interception.

The operator picks a boat to intercept with a MANUAL USV. The mockup draws a
line from the USV to a point on the boat's travel path; this module answers the
question behind that line: **at the USV's current (operator-set) speed, can it
reach a point on the boat's path at the same time the boat gets there?**

Per the design, the operator sets the speed and the sim only reports feasibility
— it does not auto-solve the speed. So the key entry point, `evaluate_intercept`,
returns whether the current speed makes the intercept, the intercept point, and
the small time margin (how much sooner/later the USV arrives than the boat).

Geometry (all in grid units, straight-line boat motion toward its goal):
  * The boat travels from its current position toward `end_goal` at boat speed.
  * We solve for the earliest time t at which the USV, starting from its position
    and moving at USV speed, can reach the boat's position-at-time-t.
  * That's the standard moving-target interception equation (a quadratic in t).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .behaviors import UNITS_PER_KNOT_MIN
from .models import USV, Boat, World
from .types import Vec2


@dataclass
class InterceptResult:
    feasible: bool          # can the USV make the intercept at its current speed?
    point: Vec2 | None      # where the intercept happens (None if not feasible)
    time_min: float | None  # minutes until intercept (None if not feasible)
    usv_speed_knots: float  # the speed this evaluation used
    boat_speed_knots: float
    reason: str             # short human-readable explanation


def _boat_velocity(boat: Boat) -> Vec2:
    """Boat's velocity vector in grid units per minute (toward its goal). If the
    boat has no goal or isn't moving, returns a zero vector."""
    if boat.end_goal is None:
        return Vec2(x=0.0, y=0.0)
    direction = (boat.end_goal - boat.position).normalized()
    speed_units = boat.speed_knots() * UNITS_PER_KNOT_MIN  # units per minute
    return direction * speed_units


def _solve_intercept_time(
    usv_pos: Vec2, usv_speed_units: float, boat_pos: Vec2, boat_vel: Vec2
) -> float | None:
    """Earliest time t >= 0 at which a pursuer starting at `usv_pos` moving at
    `usv_speed_units` (units/min, any direction) can reach a target starting at
    `boat_pos` moving with constant velocity `boat_vel` (units/min).

    Solves |(boat_pos + boat_vel*t) - usv_pos| = usv_speed_units * t, i.e. a
    quadratic a*t^2 + b*t + c = 0. Returns the smallest non-negative root, or
    None if there is no real non-negative solution (can't catch it)."""
    rel = boat_pos - usv_pos  # vector from USV to boat now
    a = boat_vel.x**2 + boat_vel.y**2 - usv_speed_units**2
    b = 2 * (rel.x * boat_vel.x + rel.y * boat_vel.y)
    c = rel.x**2 + rel.y**2

    # If a ~ 0, the speeds match: linear equation b*t + c = 0.
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return 0.0 if abs(c) < 1e-12 else None
        t = -c / b
        return t if t >= 0 else None

    disc = b * b - 4 * a * c
    if disc < 0:
        return None  # no real solution -> cannot intercept
    sqrt_disc = math.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    candidates = sorted(t for t in (t1, t2) if t >= -1e-9)
    if not candidates:
        return None
    return max(candidates[0], 0.0)


def evaluate_intercept(
    usv: USV, boat: Boat, world: World, speed_knots: float | None = None
) -> InterceptResult:
    """Can `usv` intercept `boat` at the given speed (defaults to the USV's
    current effective speed)? Reports feasibility, the intercept point, and time.

    'Feasible' means: there is a non-negative intercept time, the intercept point
    lies inside the grid, and the boat reaches that point before finishing its
    current leg to its goal (i.e. we intercept it on the path it's actually on).
    """
    usv_speed_knots = (
        speed_knots if speed_knots is not None else usv.effective_speed_knots()
    )
    boat_speed_knots = boat.speed_knots()
    usv_speed_units = usv_speed_knots * UNITS_PER_KNOT_MIN

    boat_vel = _boat_velocity(boat)

    # A stationary-goal-less boat: intercept is just reaching its current spot.
    if boat_vel.length() == 0:
        t = usv.position.distance_to(boat.position) / usv_speed_units if usv_speed_units > 0 else None
        if t is None:
            return InterceptResult(False, None, None, usv_speed_knots,
                                   boat_speed_knots, "USV has no speed")
        return InterceptResult(True, boat.position, t, usv_speed_knots,
                               boat_speed_knots, "boat is stationary")

    t = _solve_intercept_time(usv.position, usv_speed_units, boat.position, boat_vel)
    if t is None:
        return InterceptResult(
            False, None, None, usv_speed_knots, boat_speed_knots,
            "too slow to catch the boat at this speed",
        )

    point = boat.position + boat_vel * t

    # intercept must be inside the grid
    if not world.grid.contains(point):
        return InterceptResult(
            False, point, t, usv_speed_knots, boat_speed_knots,
            "intercept point is outside the grid",
        )

    # intercept must happen before the boat reaches its current goal (still on
    # this leg of its path)
    time_to_goal = boat.position.distance_to(boat.end_goal) / (
        boat.speed_knots() * UNITS_PER_KNOT_MIN
    )
    if t > time_to_goal + 1e-9:
        return InterceptResult(
            False, point, t, usv_speed_knots, boat_speed_knots,
            "boat reaches its goal before the USV can intercept",
        )

    return InterceptResult(
        True, point, t, usv_speed_knots, boat_speed_knots,
        "intercept feasible at this speed",
    )


def minimum_intercept_speed_knots(
    usv: USV, boat: Boat, world: World, hi_knots: float = 100.0
) -> float | None:
    """The lowest USV speed (knots) that makes the intercept feasible, found by
    binary search. Useful to TELL the operator what speed they'd need — the sim
    still lets them set the speed themselves; this is advisory only. Returns None
    if even `hi_knots` can't intercept."""
    lo, hi = 0.0, hi_knots
    if not evaluate_intercept(usv, boat, world, speed_knots=hi).feasible:
        return None
    for _ in range(40):  # ~1e-12 precision over the range
        mid = (lo + hi) / 2
        if evaluate_intercept(usv, boat, world, speed_knots=mid).feasible:
            hi = mid
        else:
            lo = mid
    return hi


def commit_intercept(
    world: World, usv_id: str, boat_id: str, speed_knots: float
) -> InterceptResult:
    """Operator action: point a MANUAL USV at a boat to intercept, at a chosen
    speed. Sets the USV's speed override and intercept target so it steers to the
    boat and stops at identification range. Returns the feasibility evaluation so
    the interface can show the operator whether they'll make it.

    Committing does not require feasibility — the operator may commit a course
    that won't quite work; the returned result tells them so (and they can raise
    the speed). No-op for agent USVs."""
    usv = world.usv(usv_id)
    boat = world.boat(boat_id)
    if usv is None or boat is None or usv.mode.value != "manual":
        return InterceptResult(False, None, None, speed_knots, 0.0,
                               "invalid USV or boat")
    usv.set_operator_speed(speed_knots)
    usv.intercept_target_id = boat_id
    return evaluate_intercept(usv, boat, world, speed_knots=speed_knots)


def cancel_intercept(world: World, usv_id: str) -> None:
    """Operator action: clear an interception, returning the USV to straight-line
    travel at base speed."""
    usv = world.usv(usv_id)
    if usv is None:
        return
    usv.intercept_target_id = None
    usv.set_operator_speed(None)
