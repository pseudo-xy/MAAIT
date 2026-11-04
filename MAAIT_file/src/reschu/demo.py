"""Runnable headless demo: build the demo world, run the simulation, print a
mission log, and save the full run to a CSV.

    python -m reschu.demo

Console output (once per simulated minute):
  * a summary line (time, how many boats identified, each USV's mode + comms)
  * a per-boat line: position, nearest AGENT USV, distance, id status.

CSV output (every tick, higher resolution than the console):
  written to  results/demo_run.csv  (overwritten each run)
  one row per boat per timestep, columns:
    time_min, boat_id, x, y, true_class, behavior,
    nearest_agent, distance, within_range, identified, perceived_class
"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

from .engine import SimulationEngine
from .identification import IDENTIFY_RANGE, identify_range
from .models import World
from .scenario import demo_world, simple_world
from .types import Classification, Perception, USVMode, Vec2, threat_for_behavior

# Fixed output location, overwritten every run.
CSV_PATH = Path("results") / "demo_run.csv"
USV_CSV_PATH = Path("results") / "demo_run_usvs.csv"
COMBINED_CSV_PATH = Path("results") / "demo_run_combined.csv"

# Combined file: every row is one entity (a boat OR a USV) at one timestep,
# distinguished by row_type. Shared columns come first; then boat-only columns,
# then USV-only columns. Columns that don't apply to a row are left blank.
COMBINED_COLUMNS = [
    # --- shared ---
    "time_min",
    "row_type",        # "boat" or "usv"
    "entity_id",       # boat_id or usv_id
    "x",
    "y",
    "start_x",
    "start_y",
    "goal_x",
    "goal_y",
    "speed_knots",     # shared: boats and USVs both have a speed
    # --- boat-only ---
    "true_class",
    "behavior",
    "nearest_agent",
    "distance",
    "within_range",
    "identified",
    "perceived_class",
    "id_attempts",
    "correctly_identified",
    "escort_start_min",
    "escort_end_min",
    "left_grid",
    # --- usv-only ---
    "mode",
    "heading_x",
    "heading_y",
    "comms_state",
    "activity",
    "target_boat",
    "target_distance",
    "circling_since_min",
]

CSV_COLUMNS = [
    "time_min",
    "boat_id",
    "x",
    "y",
    "start_x",
    "start_y",
    "goal_x",
    "goal_y",
    "true_class",
    "behavior",
    "speed_knots",
    "nearest_agent",
    "distance",
    "within_range",
    "identified",
    "perceived_class",
    "id_attempts",
    "correctly_identified",
    "escort_start_min",
    "escort_end_min",
    "left_grid",
]

USV_CSV_COLUMNS = [
    "time_min",
    "usv_id",
    "mode",
    "x",
    "y",
    "start_x",
    "start_y",
    "goal_x",
    "goal_y",
    "heading_x",
    "heading_y",
    "speed_knots",
    "comms_state",
    "activity",          # seeking / circling / escorting / straight-line / intercepting / idle
    "target_boat",       # boat this USV is engaged with (agent link or intercept)
    "target_distance",   # distance to that boat (blank if none)
    "circling_since_min",  # when the current circle began (none if not circling)
]


def _nearest_agent(world: World, pos: Vec2):
    """Return (usv, distance) for the closest AGENT-mode USV, or (None, inf)."""
    agents = [u for u in world.usvs if u.mode.is_agent()]
    if not agents:
        return None, float("inf")
    best = min(agents, key=lambda u: u.position.distance_to(pos))
    return best, best.position.distance_to(pos)


def _true_class_label(b) -> str:
    """The boat's TRUE classification, with its true threat level baked in for
    enemies (e.g. 'enemy (H)'). This is ground truth — independent of whether the
    operator has identified the boat yet."""
    if b.true_class is Classification.ENEMY:
        tier = threat_for_behavior(b.behavior, b.nominal_speed_knots())
        return f"{b.true_class.value} ({tier.value})"
    return b.true_class.value


def _rows_for_tick(world: World) -> list[dict]:
    """One dict per boat capturing the full state at this timestep."""
    rows = []
    for b in world.boats:
        agent, dist = _nearest_agent(world, b.position)
        rows.append(
            {
                "time_min": round(world.mission_time_min, 3),
                "boat_id": b.id,
                "x": round(b.position.x, 3),
                "y": round(b.position.y, 3),
                "start_x": round(b.start_pos.x, 3) if b.start_pos else "",
                "start_y": round(b.start_pos.y, 3) if b.start_pos else "",
                "goal_x": round(b.end_goal.x, 3) if b.end_goal else "",
                "goal_y": round(b.end_goal.y, 3) if b.end_goal else "",
                "true_class": _true_class_label(b),
                "behavior": b.behavior.value,
                "speed_knots": round(b.speed_knots(), 3),
                "nearest_agent": agent.id if agent else "",
                "distance": round(dist, 3),
                "within_range": dist <= identify_range(world.grid),
                # identification cells: blank until the boat is identified, then
                # filled from that tick onward
                "identified": b.identified if b.identified else "",
                "perceived_class": b.perceived_class.value if b.identified else "",
                "id_attempts": b.id_attempts if b.identified else "",
                "correctly_identified": b.is_correctly_identified() if b.identified else "",
                # escort cells: blank until an escort actually applies to the boat
                "escort_start_min": b.escort_start_min if b.escort_start_min is not None else "",
                "escort_end_min": b.escort_end_min if b.escort_end_min is not None else "",
                "left_grid": b.left_grid if b.escort_start_min is not None else "",
            }
        )
    return rows


def _usv_activity(usv, world: World):
    """Return (activity, target_boat_id, target_distance) describing what the USV
    is doing this tick. target_boat/target_distance are blank when the USV has no
    target (idle / plain straight-line), so 'blank' means not-applicable."""
    BLANK = ""
    # manual interception takes precedence for a manual USV
    if not usv.mode.is_agent() and usv.intercept_target_id:
        b = world.boat(usv.intercept_target_id)
        d = usv.position.distance_to(b.position) if b else None
        return "intercepting", usv.intercept_target_id, d
    if not usv.mode.is_agent():
        return "straight-line", BLANK, None

    # agent: describe based on its assigned boat and what it PERCEIVES it to be
    tid = usv.assigned_target_id
    if not tid:
        return "idle", BLANK, None
    b = world.boat(tid)
    if b is None:
        return "idle", BLANK, None
    d = usv.position.distance_to(b.position)
    if not b.identified:
        return "seeking", tid, d
    if b.perceived_class is Perception.ENEMY:
        return "circling", tid, d
    return "escorting", tid, d


def _usv_rows_for_tick(world: World) -> list[dict]:
    """One dict per USV capturing its full state at this timestep."""
    rows = []
    for u in world.usvs:
        activity, target, tdist = _usv_activity(u, world)
        rows.append(
            {
                "time_min": round(world.mission_time_min, 3),
                "usv_id": u.id,
                "mode": u.mode.value,
                "x": round(u.position.x, 3),
                "y": round(u.position.y, 3),
                "start_x": round(u.start_pos.x, 3) if u.start_pos else "",
                "start_y": round(u.start_pos.y, 3) if u.start_pos else "",
                # a USV's "goal" is the boat it is engaged with; blank when the
                # USV has no target (idle / plain straight-line)
                "goal_x": round(world.boat(target).position.x, 3) if target and world.boat(target) else "",
                "goal_y": round(world.boat(target).position.y, 3) if target and world.boat(target) else "",
                "heading_x": round(u.heading.x, 3),
                "heading_y": round(u.heading.y, 3),
                "speed_knots": round(u.effective_speed_knots(), 3),
                "comms_state": u.comms_state.value,
                "activity": activity,
                "target_boat": target,
                "target_distance": round(tdist, 3) if tdist is not None else "",
                "circling_since_min": round(u.circling_since_min, 3) if u.circling_since_min is not None else "",
            }
        )
    return rows


def _combined_rows_for_tick(world: World) -> list[dict]:
    """Both boats and USVs as rows in one shared shape, tagged by row_type.

    Reuses the boat and USV row builders so the per-cell rules (blank-until-
    applicable, etc.) stay identical. Each row carries only the columns that
    apply to it; the rest are blank. `entity_id` holds the boat_id or usv_id."""
    rows: list[dict] = []

    # boats -> combined shape
    for br in _rows_for_tick(world):
        row = {c: "" for c in COMBINED_COLUMNS}
        row.update(br)                      # copies shared + boat-only keys
        row["row_type"] = "boat"
        row["entity_id"] = br["boat_id"]
        row.pop("boat_id", None)
        rows.append(row)

    # usvs -> combined shape
    for ur in _usv_rows_for_tick(world):
        row = {c: "" for c in COMBINED_COLUMNS}
        row.update(ur)                      # copies shared + usv-only keys
        row["row_type"] = "usv"
        row["entity_id"] = ur["usv_id"]
        row.pop("usv_id", None)
        rows.append(row)

    return rows
def _summary(world: World) -> str:
    t = world.mission_time_min
    usv_bits = " ".join(
        f"{u.id}[{u.mode.value},{u.comms_state.value}]" for u in world.usvs
    )
    ident = sum(1 for b in world.boats if b.identified)
    return f"t={t:5.1f}m  ident={ident}/{len(world.boats)}  {usv_bits}"


def _boat_lines(world: World) -> str:
    lines = []
    for b in world.boats:
        agent, dist = _nearest_agent(world, b.position)
        agent_id = agent.id if agent else "—"
        if b.identified:
            flag = "IDENTIFIED"
        elif dist <= identify_range(world.grid):
            flag = "in range -> will id"
        else:
            flag = ""
        lines.append(
            f"      {b.id} @({b.position.x:5.1f},{b.position.y:5.1f})  "
            f"nearest agent {agent_id} d={dist:5.1f}  {flag}"
        )
    return "\n".join(lines)


async def main() -> None:
    import os
    # Pick the scenario: RESCHU_SCENARIO=demo for the full 4-USV demo, otherwise
    # the simplified 'bit by bit' scenario (6 boats, 2 stationary centre USVs).
    which = os.environ.get("RESCHU_SCENARIO", "simple").lower()
    if which == "demo":
        world = demo_world()
        assign = True
    else:
        world = simple_world()
        # the agent USV now moves, so assign it a boat to seek at start (manual
        # USV is left unassigned by assign_all regardless)
        assign = True
    engine = SimulationEngine(world, dt_min=0.1, seed=42, assign_at_start=assign)

    print("Initial boats (truth hidden from operator until identified):")
    for b in world.boats:
        print(f"  {b.id}: true={b.true_class.value:8} behavior={b.behavior.value}")
    print(f"\nIdentification range = {identify_range(world.grid)} grid units")
    print("Agent-mode USVs: " + ", ".join(
        u.id for u in world.usvs if u.mode.is_agent()
    ))
    print()

    csv_rows: list[dict] = []
    usv_csv_rows: list[dict] = []
    combined_rows: list[dict] = []
    last_print = -1.0

    async def on_tick(w: World) -> None:
        nonlocal last_print
        # capture every tick for all result files
        csv_rows.extend(_rows_for_tick(w))
        usv_csv_rows.extend(_usv_rows_for_tick(w))
        combined_rows.extend(_combined_rows_for_tick(w))
        # print only once per simulated minute
        if w.mission_time_min - last_print >= 1.0:
            print(_summary(w))
            print(_boat_lines(w))
            last_print = w.mission_time_min

    await engine.run(duration_min=12.0, realtime_factor=0.0, on_tick=on_tick)

    print("\nFinal perceived state (what the operator sees):")
    for b in world.boats:
        tier = f" tier={b.threat_tier.value}" if b.threat_tier else ""
        print(
            f"  {b.id}: perceived={b.perceived_class.value:8} "
            f"identified={b.identified}{tier}"
        )

    print("\nWhat each USV is doing now:")
    for u in world.usvs:
        if not u.mode.is_agent():
            print(f"  {u.id} (manual): holding for operator")
            continue
        if u.assigned_target_id is None:
            print(f"  {u.id} (agent): holding — nothing to do")
            continue
        tgt = world.boat(u.assigned_target_id)
        if tgt is None:
            print(f"  {u.id} (agent): holding")
        elif not tgt.identified:
            print(f"  {u.id} (agent): seeking {tgt.id} to identify")
        elif tgt.true_class is Classification.ENEMY:
            print(f"  {u.id} (agent): circling enemy {tgt.id}")
        else:
            print(f"  {u.id} (agent): escorting {tgt.true_class.value} {tgt.id}")

    print("\nCircle engagements (USV: boat, start -> end):")
    any_circle = False
    for u in world.usvs:
        for (bid, start, end) in u.circle_events:
            print(f"  {u.id}: {bid}  t={start}m -> t={end}m  ({round(end - start, 2)} min)")
            any_circle = True
    if not any_circle:
        print("  (none)")

    print("\nEscort engagements (boat: start -> left grid):")
    any_escort = False
    for b in world.boats:
        if b.escort_start_min is not None:
            end = f"{b.escort_end_min}m" if b.escort_end_min is not None else "still on grid"
            dur = f"{round(b.escort_end_min - b.escort_start_min, 2)} min" if b.escort_end_min is not None else "-"
            print(f"  {b.id} ({b.true_class.value}): t={b.escort_start_min}m -> {end}  ({dur})")
            any_escort = True
    if not any_escort:
        print("  (none)")

    # write all three CSVs (overwrite each run)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)
    with USV_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=USV_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(usv_csv_rows)
    with COMBINED_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMBINED_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined_rows)
    print(f"\nSaved {len(csv_rows)} boat rows to {CSV_PATH}")
    print(f"Saved {len(usv_csv_rows)} USV rows to {USV_CSV_PATH}")
    print(f"Saved {len(combined_rows)} combined rows to {COMBINED_CSV_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
