"""Identification rules — the bridge from a boat's true type to what the
operator perceives.

Rules:
  * A USV must be within IDENTIFY_RANGE of a boat to identify it at all.
  * LLM-mode USVs identify eligible nearby boats automatically each tick
    (auto_identify), always correctly.
  * MDP-mode USVs are NOT auto-identified. Identification is one of the MDP's
    own decisions: its policy sets `usv.pending_declaration = (boat_id, class)`
    and the engine applies it via apply_declarations(), only while the boat is
    inside ID range. The declared class may be wrong — that error is the MDP's
    own, never injected noise.
  * GREEDY-mode USVs identify in-range boats automatically like LLM ones, but a
    set number of their identifications per mission are WRONG
    (World.greedy_wrong_ids, shared by all greedy USVs; see plan_greedy_errors).
    A label a greedy USV gives sticks: other greedy/LLM USVs do not re-check it,
    so a wrong label stays until the operator relabels the boat.
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


def plan_greedy_errors(world: World, rng: random.Random) -> None:
    """Decide at mission start WHICH greedy identifications will be wrong.

    N = world.greedy_wrong_ids wrong labels are spread at random over the greedy
    USVs' identifications: N distinct positions are drawn from 1..M, where M is
    the number of contacts not known at the start, and the greedy ID with that
    running number (1st, 2nd, ...) is the wrong one. If the mission is running
    out of contacts before the plan is used up, the remaining wrong IDs are
    made on the next greedy identifications (catch-up), so the set number is
    reached whenever the greedy USVs make at least that many identifications.
    """
    world.greedy_ids_made = 0
    world.greedy_wrong_made = 0
    n = max(0, int(world.greedy_wrong_ids))
    if n == 0 or not any(u.mode is USVMode.GREEDY or USVMode.GREEDY in
                         [b["mode"] for b in u.mode_schedule] for u in world.usvs):
        world.greedy_error_plan = []
        return
    m = max(n, sum(1 for b in world.boats if not b.identified))
    world.greedy_error_plan = sorted(rng.sample(range(1, m + 1), n))


def _greedy_label(world: World, boat, rng: random.Random) -> Perception | None:
    """Label for this greedy identification: None = correct, else a wrong class."""
    world.greedy_ids_made += 1
    left = world.greedy_wrong_ids - world.greedy_wrong_made
    if left <= 0:
        return None
    # boats still to be identified, this one included: if no more than the wrong
    # IDs still owed, this one must be wrong (catch-up)
    remaining = sum(1 for b in world.boats if not b.left_grid and not b.identified)
    if world.greedy_ids_made not in world.greedy_error_plan and left < remaining:
        return None
    world.greedy_wrong_made += 1
    truth = Perception.from_classification(boat.true_class)
    return rng.choice(_OTHER_LABELS[truth])


def auto_identify(world: World, rng: random.Random | None = None,
                  greedy_rng: random.Random | None = None) -> list[str]:
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
        greedy = usv.mode is USVMode.GREEDY
        for boat in world.boats:
            if not _needs_identification(boat):
                continue
            if boat.identified and boat.identified_by_mode in ("greedy", "llm"):
                continue  # a greedy/LLM label sticks (only the operator relabels)
            if boat.id in usv.attempted_boat_ids:
                continue  # this USV already tried this boat; don't re-roll
            if not _in_range(usv, boat.position, world.grid):
                continue
            usv.attempted_boat_ids.append(boat.id)
            if greedy:
                wrong = _greedy_label(world, boat, greedy_rng or random.Random())
                boat.identify(as_class=wrong)
                boat.greedy_wrong = wrong is not None
            else:
                boat.identify()  # LLM stub: always correct
            boat.identified_by_mode = usv.mode.value
            attempted.append(boat.id)
    return attempted


def apply_declarations(world: World) -> list[str]:
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
        boat.identify(as_class=as_class)
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
