"""Operator follow-up after a MANUAL identification.

The operator handles what the manual USV identifies, with the normal manual
controls (double-click the USV, click where to go):

  * CIVILIAN  -> left alone.
  * FRIENDLY  -> it carries on to its exit (friendlies always transit to their
                 exit edge); the operator follows it out manually. The escort
                 start is logged at the ID, the end when it leaves the map.
  * ENEMY     -> the operator circles it manually. Once the manual USV has spent
                 CIRCLE_DURATION_MIN (the same 1 minute an AI USV circles) within
                 identification range of it, the enemy is eliminated, exactly as
                 when an AI USV circles it. Time outside that range does not count
                 (the clock pauses; it does not reset).

The contact is claimed by the manual USV (`Boat.manual_handler`) so AI USVs do
not come and circle / escort it instead. Only if the manual USV is destroyed or
returns to base before finishing is the claim released, so the AI team may take
over. If the USV's settings timeline switches it to MDP / LLM mid-job, that same
USV keeps the contact and its AI finishes it (an enemy's circling time carries
over; the friendly is escorted out) -- see keep_on_mode_switch().
"""

from __future__ import annotations

from .identification import identify_range
from .models import USV, Boat, World
from .types import Perception, USVMode
from .usv_movement import CIRCLE_DURATION_MIN


def claim(world: World, usv: USV, boat: Boat, now_min: float | None = None) -> None:
    """Called right after a manual identification."""
    if boat.perceived_class in (Perception.ENEMY, Perception.FRIENDLY):
        boat.manual_handler = usv.id
        boat.claimed_by = usv.id
        boat.manual_circle_min = 0.0
        if boat.perceived_class is Perception.FRIENDLY and boat.escort_start_min is None:
            boat.escort_start_min = None if now_min is None else round(now_min, 3)
        # any AI USV working this contact lets it go (it will be re-assigned)
        from .usv_movement import _release
        for u in world.usvs:
            if u.mode.is_agent() and u.assigned_target_id == boat.id:
                _release(u, boat)
    else:
        boat.manual_handler = None          # civilian: nothing to follow up


def keep_on_mode_switch(world: World, usv: USV, now_min: float) -> None:
    """The USV just switched Manual -> MDP/LLM (its timeline). Hand the contact
    it was following up to its own AI controller, carrying the circling time."""
    mine = [b for b in world.boats if b.manual_handler == usv.id and not b.left_grid]
    for i, b in enumerate(mine):
        b.manual_handler = None
        if i == 0:                           # a USV works one contact at a time
            usv.assigned_target_id = b.id
            usv.assignment_pinned = True
            if b.perceived_class is Perception.ENEMY:
                usv.circling_since_min = now_min - b.manual_circle_min


def circle_progress(boat: Boat) -> float:
    """Fraction of the required circling time completed (0..1)."""
    return min(1.0, boat.manual_circle_min / CIRCLE_DURATION_MIN)


def update_manual_engagements(world: World, now_min: float, dt_min: float) -> list[str]:
    """Advance operator-owned circling by one tick. Returns enemies eliminated."""
    done: list[str] = []
    rng = identify_range(world.grid)
    for boat in world.boats:
        hid = boat.manual_handler
        if hid is None or boat.left_grid:
            continue
        usv = world.usv(hid)
        if (usv is None or usv.eliminated or usv.left_grid or usv.rtb
                or not boat.identified
                or boat.perceived_class not in (Perception.ENEMY, Perception.FRIENDLY)):
            boat.manual_handler = None       # hand the contact back to the team
            continue
        if usv.mode is not USVMode.MANUAL or boat.perceived_class is not Perception.ENEMY:
            continue                         # friendly: the operator just follows it out
        if usv.position.distance_to(boat.position) > rng:
            continue                         # not close enough: the clock pauses
        if boat.manual_circle_min == 0.0:
            boat.manual_circle_start_min = now_min
        boat.manual_circle_min += dt_min
        if boat.manual_circle_min >= CIRCLE_DURATION_MIN - 1e-9:
            usv.circle_events.append((boat.id, round(boat.manual_circle_start_min, 3),
                                      round(now_min, 3)))
            boat.left_grid = True            # eliminated -> disappears
            boat.eliminated = True
            boat.manual_handler = None
            done.append(boat.id)
    return done
