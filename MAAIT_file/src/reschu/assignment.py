"""Assignment of agent USVs to boats.

When the simulation starts, each AGENT USV is linked to exactly one boat (the
mockup draws a line from the USV to that boat). The rules:

  * Selection: the CLOSEST available boat wins. The planner does NOT use any
    knowledge of a boat's true nature (no aggressiveness/threat) — it can't see
    ground truth, so selection is purely by distance.
  * Exclusive: two agent USVs may not be linked to the same boat.
  * Operator re-pick: double-clicking a USV releases its boat and auto-assigns
    another available one.
  * More USVs than boats: a USV with no boat left to take stays idle (holds).

Aggressiveness is engine-side knowledge (set in config); the operator only infers
it from on-screen behavior. Using it as a tie-break here is therefore fine.

A USV's link is stored in `usv.assigned_target_id`. This module only decides the
links; movement toward the linked boat is handled in usv_movement / the engine.
"""

from __future__ import annotations

from .models import USV, World
from .types import USVMode


def _engine_assigns(usv: USV, world: World) -> bool:
    """Does the engine pick this USV's contacts? Not for MDP USVs when the world
    uses the MDP team assignment (they choose their own, in policies.MDPPolicy)."""
    if not usv.mode.is_agent():
        return False
    return not (usv.mode is USVMode.MDP and world.mdp_assignment in ("value", "rate"))


def _assigned_boat_ids(world: World) -> set[str]:
    """IDs of boats currently linked to some agent USV."""
    return {
        u.assigned_target_id
        for u in world.usvs
        if u.mode.is_agent() and u.assigned_target_id is not None
    }


def _pick_boat_for(usv: USV, world: World, taken: set[str]):
    """Choose the best available boat for one USV: the closest one that still
    needs work and isn't taken by another agent. 'Needs work' is truth-free and
    means still on the grid — including an identified enemy that hasn't been
    circled/eliminated yet, and a friendly/civilian not yet escorted off. Purely
    distance-based selection; no use of a boat's true nature."""
    available = [
        b for b in world.boats
        if b.id not in taken and b.needs_work()
    ]
    if not available:
        return None
    return min(
        available,
        key=lambda b: usv.position.distance_to(b.position),
    )


def assign_all(world: World) -> None:
    """Assign every agent USV to a boat at simulation start, exclusively.

    USVs are processed in order; each takes the best boat not already taken.
    Manual USVs are left unassigned (they travel in a straight line instead).
    Idempotent enough to call once at setup; clears agent links first so a
    re-run reassigns cleanly."""
    for u in world.usvs:
        if _engine_assigns(u, world):
            u.assigned_target_id = None

    taken = _assigned_boat_ids(world)
    for u in world.usvs:
        if not _engine_assigns(u, world):
            continue
        boat = _pick_boat_for(u, world, taken)
        if boat is not None:
            u.assigned_target_id = boat.id
            taken.add(boat.id)
        # else: no boat available -> this USV stays idle (holds)


def reassign_free_agents(world: World) -> None:
    """Give a new boat to every agent USV that currently has no assignment
    (e.g. it just finished circling or its escorted boat left the grid).

    Priority mirrors the initial assignment — closest wins (distance only) — but we prefer boats that still NEED work: not yet correctly identified,
    still on the grid, and not already assigned to another USV. If no such boat
    exists, the USV stays idle."""
    taken = _assigned_boat_ids(world)
    for u in world.usvs:
        if not _engine_assigns(u, world) or u.assigned_target_id is not None:
            continue
        # a USV returning to base (or already gone/eliminated) must NOT be given a
        # new boat — it has disengaged and is heading for the edge to exit.
        if u.rtb or u.left_grid or u.eliminated:
            continue
        boat = _pick_unworked_boat(u, world, taken)
        if boat is not None:
            u.assigned_target_id = boat.id
            taken.add(boat.id)


def _pick_unworked_boat(usv: USV, world: World, taken: set[str]):
    """Best boat that still NEEDS WORK, on the grid, and not taken — closest by
    distance. 'Needs work' is truth-free (Boat.needs_work): anything still on the
    grid, which includes a correctly-identified ENEMY that hasn't been circled/
    eliminated yet (e.g. a pre-known enemy at start) and a friendly/civilian not
    yet escorted off. Falls back to None (idle) if every boat is done."""
    available = [
        b for b in world.boats
        if b.id not in taken
        and b.needs_work()
    ]
    if not available:
        return None
    return min(
        available,
        key=lambda b: usv.position.distance_to(b.position),
    )


def retarget_usv(world: World, usv_id: str, boat_id: str,
                 now_min: float | None = None) -> str | None:
    """Operator picked a NEW TARGET for an agent USV from the change-target list.
    Any boat still on the map may be picked, engaged or not:
      * this USV releases its current boat;
      * another agent USV working the picked boat lets it go (its algorithm then
        gives it something else);
      * a boat the manual USV was following up is handed to this agent.
    The pick is pinned: the team assignment keeps it until that contact resolves.
    Returns the new boat id, or None if the request was not valid."""
    from .usv_movement import _release
    usv = world.usv(usv_id)
    boat = world.boat(boat_id) if boat_id else None
    if (usv is None or not usv.mode.is_agent() or usv.eliminated or usv.left_grid
            or boat is None or boat.left_grid):
        return None
    if usv.assigned_target_id != boat.id:
        cur = world.boat(usv.assigned_target_id) if usv.assigned_target_id else None
        _release(usv, cur, now_min)
        for other in world.usvs:
            if other is not usv and other.assigned_target_id == boat.id:
                _release(other, boat, now_min)
                other.assignment_pinned = False
        boat.manual_handler = None
        usv.assigned_target_id = boat.id
        usv.pending_declaration = None
        usv.intercept_target_id = None
    usv.assignment_pinned = True
    return boat.id


def reassign_usv(world: World, usv_id: str) -> str | None:
    """Operator double-click action: release this USV's current boat and give it
    another AVAILABLE boat (not linked to any other agent USV). Returns the new
    boat id, or None if nothing else is available (USV goes idle).

    The just-released boat is allowed to be re-picked only if no other boat is
    available — normally the point is to switch to a different one, so we exclude
    it unless it's the only option."""
    usv = world.usv(usv_id)
    if usv is None or not usv.mode.is_agent():
        return None

    previous = usv.assigned_target_id
    usv.assigned_target_id = None  # release first

    taken = _assigned_boat_ids(world)
    # exclude the previously-linked boat so we actually switch away from it
    if previous is not None:
        taken = taken | {previous}

    boat = _pick_boat_for(usv, world, taken)
    if boat is None and previous is not None:
        # nothing else available: fall back to re-taking the previous boat
        # only if it isn't held by someone else
        if previous not in _assigned_boat_ids(world):
            boat = world.boat(previous)

    usv.assigned_target_id = boat.id if boat is not None else None
    # an operator pick is respected by the MDP team assignment until it resolves
    usv.assignment_pinned = usv.assigned_target_id is not None
    return usv.assigned_target_id
