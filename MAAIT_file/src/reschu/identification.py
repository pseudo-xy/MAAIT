"""Identification rules — the bridge from a boat's true type to what the
operator perceives.

Rules:
  * A USV must be within IDENTIFY_RANGE of a boat to identify it at all.
  * LLM-mode USVs identify eligible nearby boats automatically each tick
    (auto_identify), always correctly.
  * MDP-mode USVs are NOT auto-identified. Identification is one of the MDP's
    own decisions: its policy sets `usv.pending_declaration = (boat_id, class)`
    and the engine applies it via apply_declarations(), only while the boat is
    inside ID range. The declared class may be wrong (the MDP's own mistake).
  * GREEDY-mode USVs identify in-range boats automatically, like LLM ones.
  * WRONG-ID SETTING: World.agent_wrong_ids = the number of wrong IDs wanted
    from ALL agent USVs together (MDP, Greedy, LLM) in one mission. Correct
    agent IDs are turned wrong at planned points until that number is reached;
    the MDP's own mistakes count toward it. After it is reached, nothing is
    forced, and further MDP mistakes are kept and counted as EXTRA. See
    plan_agent_errors / _agent_label. Every agent label sticks: other agent
    USVs never re-check it; only the operator can relabel the boat.
  * MANUAL-mode USVs identify only on an explicit operator action; the operator
    can pick a wrong class.

"Correctly identified" boats are left alone. A boat that was *mis*identified is
still eligible for another attempt — by the same USV coming back, or any other —
so the label can be corrected before the mission ends. Every attempt is counted
on the boat (`id_attempts`), which is the research measure of interest.
"""

from __future__ import annotations

import random

from .models import USV, World
from .types import Perception, USVMode, Vec2

# Base (Small-map) identification range in grid cells. The ACTUAL range on a map
# is this times grid.range_scale (Small x1 = 2.0, Medium x2 = 4.0, Large x3 = 6.0),
# so it stays the same fraction of each map. Use identify_range(grid) to get the
# scaled value rather than this constant directly.
IDENTIFY_RANGE = 2.0


def identify_range(grid) -> float:
    """Scaled identification range for the given grid (Small x1, Med x2, Lg x3)."""
    return IDENTIFY_RANGE * grid.range_scale


def _in_range(usv: USV, boat_pos: Vec2, grid) -> bool:
    return usv.position.distance_to(boat_pos) <= identify_range(grid)


def _needs_identification(boat) -> bool:
    """Eligible for an ID attempt if it has never been correctly identified.
    (Unidentified, or previously MISidentified, are both still eligible.)"""
    return not boat.is_correctly_identified()


_OTHER_LABELS = {
    Perception.ENEMY: (Perception.FRIENDLY, Perception.CIVILIAN),
    Perception.FRIENDLY: (Perception.ENEMY, Perception.CIVILIAN),
    Perception.CIVILIAN: (Perception.ENEMY, Perception.FRIENDLY),
}


def _uses_agents(world: World) -> bool:
    """True if any USV is agent-controlled at some point of the mission."""
    return any(u.mode.is_agent() or any(getattr(blk["mode"], "is_agent", lambda: False)()
                                        for blk in u.mode_schedule)
               for u in world.usvs)


def plan_agent_errors(world: World, rng: random.Random) -> None:
    """Decide at mission start WHICH agent identifications will be made wrong.

    World.agent_wrong_ids = N is the number of wrong identifications wanted from
    ALL agent-controlled USVs together (MDP, Greedy, LLM) in one mission. Agent
    IDs are numbered as they happen (1st, 2nd, ... over all agent USVs). N
    positions are drawn at random from 1..M (M = contacts not known at the
    start); an agent ID at one of those positions is turned wrong if it was
    right. An MDP's OWN mistakes count toward N as they happen, so they replace
    forced ones. If contacts are running out before N is reached, the next
    correct agent IDs are turned wrong (catch-up). Once N is reached, nothing is
    forced any more; an MDP mistake after that is still applied, and counted
    separately as an EXTRA wrong ID (see _agent_label).
    """
    world.agent_ids_made = 0
    world.agent_wrong_counted = world.agent_wrong_forced = 0
    world.agent_wrong_own = world.agent_wrong_extra = 0
    n = max(0, int(world.agent_wrong_ids))
    if n == 0 or not _uses_agents(world):
        world.agent_error_plan = []
        return
    m = max(n, sum(1 for b in world.boats if not b.identified))
    world.agent_error_plan = sorted(rng.sample(range(1, m + 1), n))


# kept for older code and tests
plan_greedy_errors = plan_agent_errors


def _agent_label(world: World, boat, proposed: Perception,
                 rng: random.Random) -> tuple[Perception, str | None]:
    """The label an agent identification ends up with, and what kind of error
    it is: None (correct), "forced" (made wrong to reach the set number),
    "own" (the agent's own mistake, counted toward the set number) or
    "extra" (the agent's own mistake after the set number was reached)."""
    world.agent_ids_made += 1
    truth = Perception.from_classification(boat.true_class)
    left = world.agent_wrong_ids - world.agent_wrong_counted
    if proposed is not truth:                     # the agent got it wrong itself
        if left > 0:
            world.agent_wrong_counted += 1
            world.agent_wrong_own += 1
            return proposed, "own"
        world.agent_wrong_extra += 1
        return proposed, "extra"
    if left <= 0:
        return proposed, None
    # Catch-up: make this one wrong anyway if the wrong IDs still owed are as
    # many as the boats left to identify (this one included), or if the mission
    # is in its last LATE_CATCH_UP share of time, so the set number is reached
    # before time runs out.
    remaining = sum(1 for b in world.boats if not b.left_grid and not b.identified)
    late = (world.mission_minutes is not None and
            world.mission_time_min >= (1 - LATE_CATCH_UP) * world.mission_minutes)
    if world.agent_ids_made not in world.agent_error_plan and left < remaining and not late:
        return proposed, None
    world.agent_wrong_counted += 1
    world.agent_wrong_forced += 1
    return rng.choice(_OTHER_LABELS[truth]), "forced"


_AGENT_LABELLERS = ("greedy", "llm", "mdp")
LATE_CATCH_UP = 0.3      # last 30% of the mission: owed wrong IDs are made on the next correct agent IDs


def auto_identify(world: World, rng: random.Random | None = None,
                  agent_rng: random.Random | None = None) -> list[str]:
    """Agent-mode USVs attempt to identify boats in range that are not yet
    correctly identified. Returns the list of boat ids attempted this call.

    A USV does not re-attempt a boat it has already tried (tracked on the USV),
    so it "moves on" after an attempt rather than re-rolling every tick. A
    misidentified boat can still be corrected by a DIFFERENT USV that comes in
    range. Identification is deterministic: an in-range boat is identified on
    the first sweep. (`rng` is accepted for call compatibility and is unused.)
    """

    attempted: list[str] = []
    for usv in world.usvs:
        if not usv.mode.is_agent():
            continue
        if usv.mode is USVMode.MDP:
            continue  # the MDP declares classes itself (see apply_declarations)
        for boat in world.boats:
            if not _needs_identification(boat):
                continue
            if boat.identified and boat.identified_by_mode in _AGENT_LABELLERS:
                continue  # an agent's label sticks (only the operator relabels)
            if boat.id in usv.attempted_boat_ids:
                continue  # this USV already tried this boat; don't re-roll
            if not _in_range(usv, boat.position, world.grid):
                continue
            usv.attempted_boat_ids.append(boat.id)
            # Greedy / LLM-stub USVs see the true class; the shared wrong-ID
            # setting may turn it wrong
            label, kind = _agent_label(world, boat, Perception.from_classification(boat.true_class),
                                       agent_rng or random.Random())
            boat.identify(as_class=label)
            boat.id_error = kind
            boat.identified_by_mode = usv.mode.value
            attempted.append(boat.id)
    return attempted


def apply_declarations(world: World, agent_rng: random.Random | None = None) -> list[str]:
    """Apply the class declarations requested by agent policies (the MDP).

    For each USV with a `pending_declaration = (boat_id, declared_class)`:
      * the boat must exist, still be on the grid, and be inside this USV's
        identification range — otherwise the request is dropped (the policy can
        simply request again on a later tick);
      * the boat is labelled with the DECLARED class, which may be wrong. A wrong
        declaration is recorded and counted like any misidentification, and the
        boat keeps its true behavior;
      * declaring UNKNOWN is a no-op.
    The request is always cleared after processing. Returns the ids of boats
    that received a declaration this call.
    """
    declared: list[str] = []
    for usv in world.usvs:
        req = usv.pending_declaration
        if req is None:
            continue
        usv.pending_declaration = None
        boat_id, as_class = req
        boat = world.boat(boat_id)
        if boat is None or boat.left_grid:
            continue
        if as_class is Perception.UNKNOWN:
            continue
        if not _in_range(usv, boat.position, world.grid):
            continue
        if boat.identified and boat.identified_by_mode in _AGENT_LABELLERS:
            continue  # an agent's label sticks (only the operator relabels)
        # the shared wrong-ID setting may turn a correct declaration wrong; an
        # MDP mistake is kept and counted (toward the set number, or as extra)
        label, kind = _agent_label(world, boat, as_class, agent_rng or random.Random())
        boat.identify(as_class=label)
        boat.id_error = kind
        boat.identified_by_mode = "mdp"
        if boat.id not in usv.attempted_boat_ids:
            usv.attempted_boat_ids.append(boat.id)
        declared.append(boat.id)
    return declared


def manual_identify(
    world: World, usv_id: str, boat_id: str, as_class: Perception | None = None
) -> bool:
    """Operator action: identify `boat_id` using manual USV `usv_id`.

    Succeeds only if the USV is manual and within range. `as_class` is the class
    the operator picked from the classify menu (Unknown is treated as no-op /
    leaves it unidentified); if None, the identification is recorded as correct
    (the boat's true class). The operator CAN pick the wrong class — the boat
    keeps its true behavior, and the mistake is counted like any misID. Returns
    True on success."""
    usv = world.usv(usv_id)
    boat = world.boat(boat_id)
    if usv is None or boat is None:
        return False
    if usv.mode is not USVMode.MANUAL:
        return False
    if not _in_range(usv, boat.position, world.grid):
        return False
    if as_class is Perception.UNKNOWN:
        return False  # "Unknown" = operator declines to classify -> no-op
    boat.identify(as_class=as_class)
    boat.identified_by_mode = "manual"
    # the operator now handles it manually: follow a friendly out / circle an enemy
    from .manual_engagement import claim
    claim(world, usv, boat, world.mission_time_min)
    return True
