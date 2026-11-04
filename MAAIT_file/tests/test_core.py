"""Starter tests proving the load-bearing rules, all headless (no UI, no clock).

These are the behaviors that must stay correct as the project grows:
  * comms schedule flips a USV's state at the right mission times
  * threat tier is derived from behavior on identification
  * agent USVs auto-identify in range; manual USVs need the explicit action
  * boats move according to their behavior
"""

from __future__ import annotations

from reschu.behaviors import step_boat
from reschu.engine import SimulationEngine
from reschu.identification import IDENTIFY_RANGE, auto_identify, manual_identify
from reschu.models import Boat, CommsSchedule, CommsWindow, Grid, USV, World
from reschu.scenario import demo_world
from reschu.types import (
    Behavior,
    Classification,
    CommsState,
    Perception,
    ThreatTier,
    USVMode,
    Vec2,
)
import random


# --------------------------------------------------------------------------- #
# Comms schedule
# --------------------------------------------------------------------------- #
def test_comms_schedule_flips_at_boundaries():
    sched = CommsSchedule(
        windows=[
            CommsWindow(state=CommsState.IN_RANGE, start_min=0, duration_min=4),
            CommsWindow(state=CommsState.NEAR_EDGE, start_min=4, duration_min=4),
            CommsWindow(state=CommsState.OUT_OF_RANGE, start_min=8, duration_min=4),
        ]
    )
    assert sched.state_at(0) is CommsState.IN_RANGE
    assert sched.state_at(3.9) is CommsState.IN_RANGE
    assert sched.state_at(4) is CommsState.NEAR_EDGE
    assert sched.state_at(8) is CommsState.OUT_OF_RANGE
    assert sched.state_at(11.9) is CommsState.OUT_OF_RANGE


def test_comms_defaults_to_in_range_when_no_window_active():
    sched = CommsSchedule(
        windows=[CommsWindow(state=CommsState.OUT_OF_RANGE, start_min=5, duration_min=2)]
    )
    assert sched.state_at(0) is CommsState.IN_RANGE
    assert sched.state_at(6) is CommsState.OUT_OF_RANGE
    assert sched.state_at(9) is CommsState.IN_RANGE


# --------------------------------------------------------------------------- #
# Identification + threat tier
# --------------------------------------------------------------------------- #
def _world_one_boat(mode: USVMode, distance: float, behavior=Behavior.AGGRESSIVE,
                    standoff_speed: float = 9.0):
    cls = Classification.ENEMY if behavior in (Behavior.AGGRESSIVE, Behavior.STANDOFF) else Classification.CIVILIAN
    return World(
        # width-20 grid so range_scale = 1 and the base IDENTIFY_RANGE applies
        # directly (the ± distances in the tests mean what they say).
        grid=Grid(width=20, height=20),
        usvs=[USV(id="U", position=Vec2(x=0, y=0), mode=mode)],
        boats=[
            Boat(
                id="B",
                position=Vec2(x=distance, y=0),
                true_class=cls,
                behavior=behavior,
                standoff_speed=standoff_speed,
            )
        ],
    )


def test_agent_auto_identifies_in_range():
    w = _world_one_boat(USVMode.LLM, distance=IDENTIFY_RANGE - 1)
    newly = auto_identify(w)
    assert newly == ["B"]
    assert w.boat("B").identified
    assert w.boat("B").perceived_class is Perception.ENEMY


def test_agent_does_not_identify_out_of_range():
    w = _world_one_boat(USVMode.LLM, distance=IDENTIFY_RANGE + 5)
    assert auto_identify(w) == []
    assert not w.boat("B").identified
    assert w.boat("B").perceived_class is Perception.UNKNOWN


def test_manual_requires_action_not_auto():
    w = _world_one_boat(USVMode.MANUAL, distance=IDENTIFY_RANGE - 1)
    # auto pass does nothing for a manual USV
    assert auto_identify(w) == []
    assert not w.boat("B").identified
    # explicit action works because it is in range
    assert manual_identify(w, "U", "B") is True
    assert w.boat("B").identified


def test_manual_identify_fails_out_of_range():
    w = _world_one_boat(USVMode.MANUAL, distance=IDENTIFY_RANGE + 5)
    assert manual_identify(w, "U", "B") is False
    assert not w.boat("B").identified


def test_threat_tier_from_behavior_and_speed():
    # aggressive -> HIGH
    w_agg = _world_one_boat(USVMode.LLM, distance=1, behavior=Behavior.AGGRESSIVE)
    auto_identify(w_agg)
    assert w_agg.boat("B").threat_tier is ThreatTier.HIGH

    # standoff @ 9 kn -> MEDIUM
    w_med = _world_one_boat(
        USVMode.LLM, distance=1, behavior=Behavior.STANDOFF, standoff_speed=9.0
    )
    auto_identify(w_med)
    assert w_med.boat("B").threat_tier is ThreatTier.MEDIUM

    # standoff @ 4 kn -> LOW
    w_low = _world_one_boat(
        USVMode.LLM, distance=1, behavior=Behavior.STANDOFF, standoff_speed=4.0
    )
    auto_identify(w_low)
    assert w_low.boat("B").threat_tier is ThreatTier.LOW


def test_misidentification_keeps_true_behavior_and_counts_attempts():
    # An enemy wrongly labelled friendly still IS an enemy (behavior unchanged),
    # shows no threat tier, and the attempt is counted. A later correct call
    # fixes the label and derives the tier.
    from reschu.types import Behavior as B

    boat = Boat(
        id="B",
        position=Vec2(x=0, y=0),
        true_class=Classification.ENEMY,
        behavior=B.AGGRESSIVE,
    )
    boat.identify(as_class=Perception.FRIENDLY)  # wrong call
    assert boat.perceived_class is Perception.FRIENDLY
    assert boat.behavior is B.AGGRESSIVE          # true behavior unchanged
    assert boat.threat_tier is None               # mislabelled -> no tier shown
    assert not boat.is_correctly_identified()
    assert boat.id_attempts == 1

    boat.identify()  # correct call
    assert boat.is_correctly_identified()
    assert boat.threat_tier is ThreatTier.HIGH
    assert boat.id_attempts == 2


# --------------------------------------------------------------------------- #
# Movement
# --------------------------------------------------------------------------- #
def test_friendly_moves_and_has_a_goal():
    w = World(
        grid=Grid(width=100, height=100),
        boats=[
            Boat(
                id="F",
                position=Vec2(x=50, y=50),
                true_class=Classification.FRIENDLY,
                behavior=Behavior.FRIENDLY,
            )
        ],
    )
    rng = random.Random(0)
    before = w.boat("F").position
    nxt = step_boat(w.boat("F"), w, dt_min=1.0, rng=rng)
    # a random goal was assigned and the boat moved toward it
    assert w.boat("F").end_goal is not None
    assert nxt.distance_to(before) > 0


def test_standoff_stays_far_from_usvs():
    """A standoff enemy repositions to maximise distance from the nearest USV,
    rather than heading to a fixed corner. With a USV parked in one area, the
    boat's chosen goal should be farther from that USV than its current spot."""
    w = World(
        grid=Grid(width=100, height=60),
        usvs=[USV(id="U", position=Vec2(x=20, y=30), mode=USVMode.LLM)],
        boats=[
            Boat(
                id="S",
                position=Vec2(x=30, y=30),  # fairly near the USV
                true_class=Classification.ENEMY,
                behavior=Behavior.STANDOFF,
                standoff_speed=9.0,
            )
        ],
    )
    rng = random.Random(1)
    usv_pos = w.usv("U").position
    here = w.boat("S").position.distance_to(usv_pos)
    step_boat(w.boat("S"), w, dt_min=0.1, rng=rng)
    goal = w.boat("S").end_goal
    # the chosen goal should be farther from the USV than where it started
    assert goal.distance_to(usv_pos) > here
    # and it should keep off the very edge (interior margin)
    assert 0 < goal.x < 100 and 0 < goal.y < 60


def test_civilian_moves_randomly():
    w = World(
        grid=Grid(width=100, height=100),
        boats=[
            Boat(
                id="C",
                position=Vec2(x=50, y=50),
                true_class=Classification.CIVILIAN,
                behavior=Behavior.CIVILIAN,
                start_pos=Vec2(x=50, y=50),
            )
        ],
    )
    rng = random.Random(2)
    c = w.boat("C")
    goals = []
    # the civilian picks random goals over time; collect the distinct goals it
    # commits to across many ticks to confirm the walk is random, not fixed.
    # At real scale (5 kn -> 0.25 units/min) it must run long enough to reach and
    # re-pick several goals, so use a coarse dt over a long span.
    for _ in range(400):
        c.position = step_boat(c, w, dt_min=2.0, rng=rng)
        if c.end_goal is not None:
            goals.append((round(c.end_goal.x, 1), round(c.end_goal.y, 1)))
    # a random walk commits to several different random goals over time
    assert len(set(goals)) > 3



def test_boat_picks_new_goal_on_arrival():
    w = World(
        grid=Grid(width=100, height=100),
        boats=[
            Boat(
                id="F",
                position=Vec2(x=50, y=50),
                true_class=Classification.FRIENDLY,
                behavior=Behavior.FRIENDLY,
            )
        ],
    )
    rng = random.Random(3)
    c = w.boat("F")
    step_boat(c, w, dt_min=0.1, rng=rng)
    first_goal = c.end_goal
    # force arrival: put the boat on its goal, step again
    c.position = first_goal
    step_boat(c, w, dt_min=0.1, rng=rng)
    # a fresh goal should have been chosen
    assert c.end_goal is not None


def test_aggressive_moves_towards_nearest_usv():
    w = World(
        grid=Grid(width=100, height=100),
        usvs=[USV(id="U", position=Vec2(x=10, y=0))],
        boats=[
            Boat(
                id="A",
                position=Vec2(x=0, y=0),
                true_class=Classification.ENEMY,
                behavior=Behavior.AGGRESSIVE,
            )
        ],
    )
    rng = random.Random(0)
    nxt = step_boat(w.boat("A"), w, dt_min=1.0, rng=rng)
    assert nxt.x > 0  # closed distance towards the USV


# --------------------------------------------------------------------------- #
# Engine integration
# --------------------------------------------------------------------------- #
def test_engine_tick_advances_time_and_identifies():
    world = demo_world()
    # the runtime tick (1 s): the MDP teammate is built and evaluated at this tick
    engine = SimulationEngine(world, dt_min=1 / 60, seed=1)
    for _ in range(20 * 60):  # 20 simulated minutes
        engine.tick()
    assert world.mission_time_min > 0
    # agent USVs should have identified at least one boat by now
    assert any(b.identified for b in world.boats)


# --------------------------------------------------------------------------- #
# USV movement — Stage 1: base speed, operator speed override, move-to-target
# --------------------------------------------------------------------------- #
def test_usv_base_speed_and_operator_override():
    from reschu.models import USV, USV_BASE_SPEED_KNOTS
    from reschu.types import USVMode

    agent = USV(id="A", position=Vec2(x=0, y=0), mode=USVMode.LLM)
    manual = USV(id="M", position=Vec2(x=0, y=0), mode=USVMode.MANUAL)

    # both start at the shared base speed
    assert agent.effective_speed_knots() == USV_BASE_SPEED_KNOTS
    assert manual.effective_speed_knots() == USV_BASE_SPEED_KNOTS

    # operator can change a MANUAL USV's speed
    manual.set_operator_speed(20.0)
    assert manual.effective_speed_knots() == 20.0

    # operator speed is ignored for an AGENT USV
    agent.set_operator_speed(20.0)
    assert agent.effective_speed_knots() == USV_BASE_SPEED_KNOTS

    # clearing the override returns a manual USV to base speed
    manual.set_operator_speed(None)
    assert manual.effective_speed_knots() == USV_BASE_SPEED_KNOTS


def test_usv_moves_toward_target_and_stops_on_arrival():
    from reschu.models import USV, Grid, World
    from reschu.usv_movement import move_towards, has_arrived

    world = World(grid=Grid(width=100, height=100), usvs=[], boats=[])
    usv = USV(id="U", position=Vec2(x=0, y=0))
    target = Vec2(x=10, y=0)

    # moves toward the target
    usv.position = move_towards(usv, target, world, dt_min=0.1)
    assert 0 < usv.position.x <= 10

    # after enough ticks it arrives exactly and does not overshoot
    for _ in range(200):
        usv.position = move_towards(usv, target, world, dt_min=0.1)
    assert usv.position.distance_to(target) < 1e-6
    assert has_arrived(usv, target, world, dt_min=0.1)


def test_manual_faster_speed_covers_more_ground():
    from reschu.models import USV, Grid, World
    from reschu.usv_movement import move_towards
    from reschu.types import USVMode

    world = World(grid=Grid(width=1000, height=100), usvs=[], boats=[])
    slow = USV(id="S", position=Vec2(x=0, y=0), mode=USVMode.MANUAL)
    fast = USV(id="F", position=Vec2(x=0, y=0), mode=USVMode.MANUAL)
    fast.set_operator_speed(24.0)  # double base

    target = Vec2(x=500, y=0)
    slow.position = move_towards(slow, target, world, dt_min=0.1)
    fast.position = move_towards(fast, target, world, dt_min=0.1)
    assert fast.position.x > slow.position.x  # faster USV travelled farther


# --------------------------------------------------------------------------- #
# USV movement — Stage 2: agent seek-and-identify
# --------------------------------------------------------------------------- #
def test_agent_seek_picks_nearest_unknown():
    from reschu.models import USV, Grid, World, Boat
    from reschu.usv_movement import nearest_unknown_boat
    from reschu.types import USVMode, Classification, Behavior

    world = World(
        grid=Grid(width=100, height=100),
        usvs=[USV(id="A", position=Vec2(x=0, y=0), mode=USVMode.LLM)],
        boats=[
            Boat(id="far", position=Vec2(x=50, y=0),
                 true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN),
            Boat(id="near", position=Vec2(x=10, y=0),
                 true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN),
        ],
    )
    target = nearest_unknown_boat(world.usvs[0], world)
    assert target.id == "near"

    # once "near" is identified, the agent should switch to "far"
    world.boat("near").identify()
    target = nearest_unknown_boat(world.usvs[0], world)
    assert target.id == "far"

    # once all identified, there is nothing to seek
    world.boat("far").identify()
    assert nearest_unknown_boat(world.usvs[0], world) is None


def test_agent_seeks_and_identifies_over_time():
    from reschu.engine import SimulationEngine
    from reschu.models import USV, Grid, World, Boat
    from reschu.types import USVMode, Classification, Behavior

    # one agent, one distant standoff enemy it must travel to reach
    world = World(
        grid=Grid(width=200, height=50),
        usvs=[USV(id="A", position=Vec2(x=0, y=25), mode=USVMode.LLM)],
        boats=[
            Boat(id="E", position=Vec2(x=120, y=25),
                 true_class=Classification.ENEMY, behavior=Behavior.STANDOFF,
                 standoff_speed=4.0),
        ],
    )
    engine = SimulationEngine(world, dt_min=1.0, seed=0)
    # with passive USVs this boat would never be identified; with seeking it is.
    # At real scale (USV 12 kn -> 0.6 units/min) reaching a boat ~120 units away
    # takes ~200 sim-minutes, so step in 1-min ticks over a long enough span.
    for _ in range(600):
        engine.tick()
        if world.boat("E").identified:
            break
    assert world.boat("E").identified
    assert world.boat("E").perceived_class.value == "enemy"


def test_manual_usv_moves_straight_from_start():
    from reschu.engine import SimulationEngine
    from reschu.models import USV, Grid, World, Boat
    from reschu.types import USVMode, Classification, Behavior

    world = World(
        grid=Grid(width=200, height=200),
        usvs=[USV(id="M", position=Vec2(x=10, y=10), mode=USVMode.MANUAL,
                  manual_heading=Vec2(x=1, y=0))],
        boats=[
            Boat(id="B", position=Vec2(x=90, y=90),
                 true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN),
        ],
    )
    engine = SimulationEngine(world, dt_min=0.1, seed=0)
    start = Vec2(x=10, y=10)
    for _ in range(50):
        engine.tick()
    m = world.usvs[0]
    # manual USV travels in a straight line along its heading (x grows, y fixed)
    assert m.position.x > start.x
    assert abs(m.position.y - start.y) < 1e-6


# --------------------------------------------------------------------------- #
# USV movement — Stage 3: circle enemies, escort friendlies/civilians
# --------------------------------------------------------------------------- #
def test_circle_holds_radius_and_orbits():
    import math
    from reschu.models import USV, Grid, World, Boat
    from reschu.usv_movement import circle_step, circle_radius
    from reschu.types import USVMode, Classification, Behavior

    w = World(grid=Grid(width=100, height=100), usvs=[], boats=[])
    R = circle_radius(w.grid)   # scaled for this grid
    enemy = Boat(id="E", position=Vec2(x=50, y=50),
                 true_class=Classification.ENEMY, behavior=Behavior.AGGRESSIVE)
    u = USV(id="U", position=Vec2(x=50 + R, y=50), mode=USVMode.LLM)

    angles = []
    for _ in range(10):
        u.position = circle_step(u, enemy, w, 0.2)
        off = u.position - enemy.position
        angles.append(math.atan2(off.y, off.x))
        radius = off.length()
    # stays near the (scaled) circle radius and the angle changes (it orbits)
    assert abs(radius - R) < 1.0
    assert angles[0] != angles[-1]


def test_escort_keeps_station_near_boat():
    from reschu.models import USV, Grid, World, Boat
    from reschu.usv_movement import escort_step, escort_offset
    from reschu.types import USVMode, Classification, Behavior

    w = World(grid=Grid(width=100, height=100), usvs=[], boats=[])
    OFF = escort_offset(w.grid)   # scaled for this grid
    friendly = Boat(id="F", position=Vec2(x=50, y=50),
                    true_class=Classification.FRIENDLY, behavior=Behavior.FRIENDLY,
                    heading=Vec2(x=1, y=0), straight_heading=Vec2(x=1, y=0))
    u = USV(id="U", position=Vec2(x=10, y=10), mode=USVMode.LLM)
    # At real scale the USV must travel far to reach the boat before it can
    # settle at its escort station, so use 1-min ticks over enough span.
    for _ in range(200):
        u.position = escort_step(u, friendly, w, 1.0)
    # settles roughly one escort-offset from the boat (side-and-behind station)
    assert abs(u.position.distance_to(friendly.position) - OFF) < 1.2
    # and it must be to the SIDE or BEHIND, never ahead: the boat heads +x, so the
    # USV's x should be at or behind the boat's x (not out in front of it).
    assert u.position.x <= friendly.position.x + 0.1


def test_agent_priority_identify_then_engage():
    from reschu.models import USV, Grid, World, Boat
    from reschu.usv_movement import agent_step
    from reschu.types import USVMode, Classification, Behavior

    # one unknown enemy far away, one already-identified friendly nearby.
    # priority says: go IDENTIFY the unknown first (don't escort yet).
    w = World(
        grid=Grid(width=200, height=50),
        usvs=[USV(id="A", position=Vec2(x=100, y=25), mode=USVMode.LLM)],
        boats=[
            Boat(id="unknown", position=Vec2(x=180, y=25),
                 true_class=Classification.ENEMY, behavior=Behavior.STANDOFF,
                 standoff_speed=4.0),
            Boat(id="known", position=Vec2(x=90, y=25),
                 true_class=Classification.FRIENDLY, behavior=Behavior.FRIENDLY,
                 identified=True, perceived_class=Perception.FRIENDLY),
        ],
    )
    before_x = w.usvs[0].position.x
    w.usvs[0].position = agent_step(w.usvs[0], w, dt_min=0.1)
    # it should move toward the UNKNOWN (x increasing), not the known friendly
    assert w.usvs[0].position.x > before_x
    assert w.usvs[0].assigned_target_id == "unknown"


def test_agent_engages_enemy_after_all_identified():
    from reschu.engine import SimulationEngine
    from reschu.models import USV, Grid, World, Boat
    from reschu.types import USVMode, Classification, Behavior
    from reschu.usv_movement import circle_radius

    w = World(
        grid=Grid(width=60, height=60),
        usvs=[USV(id="A", position=Vec2(x=15, y=30), mode=USVMode.LLM)],
        boats=[
            Boat(id="E", position=Vec2(x=32, y=30),
                 true_class=Classification.ENEMY, behavior=Behavior.STANDOFF,
                 standoff_speed=4.0),
        ],
    )
    R = circle_radius(w.grid)   # scaled circling radius for this grid
    engine = SimulationEngine(w, dt_min=1.0, seed=0)
    # the agent approaches from a realistic distance, identifies E, then settles
    # onto the circle ring; at real scale the approach + manoeuvre needs 1-min
    # ticks over enough span.
    circling_seen = False
    for _ in range(120):  # 120 sim-min
        engine.tick()
        if w.boat("E").identified:
            d = w.usvs[0].position.distance_to(w.boat("E").position)
            if abs(d - R) < 2.0:
                circling_seen = True
                break
    assert w.boat("E").identified
    assert circling_seen


# --------------------------------------------------------------------------- #
# Assignment: exclusive links, closest-with-aggressive-tiebreak, re-pick
# --------------------------------------------------------------------------- #
def test_assignment_is_exclusive_and_nearest():
    from reschu.models import USV, Grid, World, Boat
    from reschu.assignment import assign_all
    from reschu.types import USVMode, Classification, Behavior

    world = World(
        grid=Grid(width=100, height=100),
        usvs=[
            USV(id="A1", position=Vec2(x=0, y=0), mode=USVMode.LLM),
            USV(id="A2", position=Vec2(x=100, y=0), mode=USVMode.LLM),
        ],
        boats=[
            Boat(id="near1", position=Vec2(x=10, y=0),
                 true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN),
            Boat(id="near2", position=Vec2(x=90, y=0),
                 true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN),
        ],
    )
    assign_all(world)
    a1 = world.usv("A1").assigned_target_id
    a2 = world.usv("A2").assigned_target_id
    assert a1 == "near1"      # each takes its closest
    assert a2 == "near2"
    assert a1 != a2           # exclusive


def test_assignment_picks_closest_no_aggression():
    from reschu.models import USV, Grid, World, Boat
    from reschu.assignment import assign_all
    from reschu.types import USVMode, Classification, Behavior

    # The planner knows nothing of true nature: it picks purely by distance.
    # The nearer boat wins even though the farther one is an aggressive enemy.
    world = World(
        grid=Grid(width=100, height=100),
        usvs=[USV(id="A", position=Vec2(x=50, y=50), mode=USVMode.LLM)],
        boats=[
            Boat(id="near", position=Vec2(x=50, y=45),
                 true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN),
            Boat(id="far", position=Vec2(x=50, y=70),
                 true_class=Classification.ENEMY, behavior=Behavior.AGGRESSIVE),
        ],
    )
    assign_all(world)
    assert world.usv("A").assigned_target_id == "near"




def test_more_usvs_than_boats_leaves_one_idle():
    from reschu.models import USV, Grid, World, Boat
    from reschu.assignment import assign_all
    from reschu.types import USVMode, Classification, Behavior

    world = World(
        grid=Grid(width=100, height=100),
        usvs=[
            USV(id="A1", position=Vec2(x=0, y=0), mode=USVMode.LLM),
            USV(id="A2", position=Vec2(x=1, y=0), mode=USVMode.LLM),
        ],
        boats=[
            Boat(id="only", position=Vec2(x=5, y=0),
                 true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN),
        ],
    )
    assign_all(world)
    assigned = [world.usv("A1").assigned_target_id, world.usv("A2").assigned_target_id]
    assert assigned.count("only") == 1     # exactly one got it
    assert None in assigned                # the other is idle


def test_operator_reassign_switches_boat():
    from reschu.models import USV, Grid, World, Boat
    from reschu.assignment import assign_all, reassign_usv
    from reschu.types import USVMode, Classification, Behavior

    world = World(
        grid=Grid(width=100, height=100),
        usvs=[USV(id="A", position=Vec2(x=0, y=0), mode=USVMode.LLM)],
        boats=[
            Boat(id="first", position=Vec2(x=10, y=0),
                 true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN),
            Boat(id="second", position=Vec2(x=20, y=0),
                 true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN),
        ],
    )
    assign_all(world)
    assert world.usv("A").assigned_target_id == "first"
    # operator double-clicks -> should switch to a different available boat
    new = reassign_usv(world, "A")
    assert new == "second"
    assert world.usv("A").assigned_target_id == "second"


# --------------------------------------------------------------------------- #
# Stage 4: manual interception
# --------------------------------------------------------------------------- #
def test_intercept_feasible_when_fast_enough():
    from reschu.models import USV, Grid, World, Boat
    from reschu.interception import evaluate_intercept
    from reschu.types import USVMode, Classification, Behavior

    grid = Grid(width=100, height=100)
    boat = Boat(id="B", position=Vec2(x=20, y=50),
                true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN,
                end_goal=Vec2(x=100, y=50))  # east at 5 kn
    usv = USV(id="U", position=Vec2(x=50, y=20), mode=USVMode.MANUAL)
    w = World(grid=grid, usvs=[usv], boats=[boat])

    fast = evaluate_intercept(usv, boat, w, speed_knots=20)
    assert fast.feasible
    assert fast.point is not None and fast.time_min is not None


def test_intercept_infeasible_when_too_slow_from_behind():
    from reschu.models import USV, Grid, World, Boat
    from reschu.interception import evaluate_intercept
    from reschu.types import USVMode, Classification, Behavior

    grid = Grid(width=100, height=100)
    boat = Boat(id="B", position=Vec2(x=50, y=50),
                true_class=Classification.ENEMY, behavior=Behavior.AGGRESSIVE,
                end_goal=Vec2(x=100, y=50))  # east at 10 kn
    usv = USV(id="U", position=Vec2(x=10, y=50), mode=USVMode.MANUAL)  # behind it
    w = World(grid=grid, usvs=[usv], boats=[boat])

    slow = evaluate_intercept(usv, boat, w, speed_knots=5)  # slower than boat
    assert not slow.feasible


def test_minimum_intercept_speed_is_a_threshold():
    from reschu.models import USV, Grid, World, Boat
    from reschu.interception import minimum_intercept_speed_knots, evaluate_intercept
    from reschu.types import USVMode, Classification, Behavior

    grid = Grid(width=100, height=100)
    boat = Boat(id="B", position=Vec2(x=20, y=50),
                true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN,
                end_goal=Vec2(x=100, y=50))
    usv = USV(id="U", position=Vec2(x=50, y=20), mode=USVMode.MANUAL)
    w = World(grid=grid, usvs=[usv], boats=[boat])

    mn = minimum_intercept_speed_knots(usv, boat, w)
    assert mn is not None
    # just above the minimum works; just below fails
    assert evaluate_intercept(usv, boat, w, speed_knots=mn + 0.5).feasible
    assert not evaluate_intercept(usv, boat, w, speed_knots=mn - 0.5).feasible


def test_commit_intercept_drives_usv_to_id_range():
    from reschu.models import USV, Grid, World, Boat
    from reschu.engine import SimulationEngine
    from reschu.interception import commit_intercept
    from reschu.identification import identify_range
    from reschu.types import USVMode, Classification, Behavior

    grid = Grid(width=120, height=80)
    boat = Boat(id="B", position=Vec2(x=10, y=40),
                true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN,
                start_pos=Vec2(x=10, y=40), end_goal=Vec2(x=110, y=40))
    usv = USV(id="M", position=Vec2(x=60, y=10), start_pos=Vec2(x=60, y=10),
              mode=USVMode.MANUAL, manual_heading=Vec2(x=0, y=1))
    w = World(grid=grid, usvs=[usv], boats=[boat])
    engine = SimulationEngine(w, dt_min=1.0, seed=1, assign_at_start=False)

    res = commit_intercept(w, "M", "B", speed_knots=12.0)
    assert res.feasible

    # At real scale the committed intercept needs 1-min ticks to close the gap.
    ID_R = identify_range(w.grid)   # scaled range for this grid
    reached = False
    for _ in range(400):
        engine.tick()
        if usv.position.distance_to(boat.position) <= ID_R:
            reached = True
            break
    assert reached  # the committed interception brought the USV to ID range


# --------------------------------------------------------------------------- #
# Misidentification: per-USV rate, wrong label, re-ID eligibility, attempts
# --------------------------------------------------------------------------- #
def test_agent_identification_is_correct():
    from reschu.models import USV, Grid, World, Boat
    from reschu.identification import auto_identify
    from reschu.types import USVMode, Classification, Behavior
    import random

    w = World(
        grid=Grid(width=50, height=50),
        usvs=[USV(id="A", position=Vec2(x=0, y=0), mode=USVMode.LLM)],
        boats=[Boat(id="B", position=Vec2(x=1, y=0),
                    true_class=Classification.ENEMY, behavior=Behavior.AGGRESSIVE)],
    )
    auto_identify(w, random.Random(0))
    assert w.boat("B").is_correctly_identified()


def test_misidentified_boat_can_be_corrected_by_another_usv():
    from reschu.models import USV, Grid, World, Boat
    from reschu.identification import auto_identify
    from reschu.types import USVMode, Classification, Behavior, Perception
    import random

    # The boat carries a WRONG label from an earlier decision (e.g. an agent that
    # committed too early, or the operator). An agent USV in range corrects it.
    w = World(
        grid=Grid(width=50, height=50),
        usvs=[USV(id="B", position=Vec2(x=2, y=0), mode=USVMode.LLM)],
        boats=[Boat(id="X", position=Vec2(x=1.5, y=0),
                    true_class=Classification.ENEMY, behavior=Behavior.STANDOFF,
                    standoff_speed=9.0)],
    )
    x = w.boat("X")
    x.identify(as_class=Perception.FRIENDLY)      # earlier wrong call
    assert not x.is_correctly_identified()
    auto_identify(w, random.Random(0))
    assert x.is_correctly_identified()            # corrected
    assert x.id_attempts == 2                     # one wrong + one correct


def test_usv_does_not_reroll_same_boat():
    from reschu.models import USV, Grid, World, Boat
    from reschu.identification import auto_identify
    from reschu.types import USVMode, Classification, Behavior
    import random

    w = World(
        grid=Grid(width=50, height=50),
        usvs=[USV(id="A", position=Vec2(x=1, y=0), mode=USVMode.LLM)],
        boats=[Boat(id="B", position=Vec2(x=1.5, y=0),
                    true_class=Classification.ENEMY, behavior=Behavior.AGGRESSIVE)],
    )
    rng = random.Random(0)
    auto_identify(w, rng)
    auto_identify(w, rng)
    auto_identify(w, rng)
    # only one attempt despite three sweeps: the USV doesn't re-roll its own boat
    assert w.boat("B").id_attempts == 1


# --------------------------------------------------------------------------- #
# Engagement lifecycle: circle 2 min then move on; escort until off-grid
# --------------------------------------------------------------------------- #
def test_usv_stops_circling_after_duration_and_reassigns():
    from reschu.engine import SimulationEngine
    from reschu.models import USV, Grid, World, Boat
    from reschu.types import USVMode, Classification, Behavior
    from reschu.usv_movement import CIRCLE_DURATION_MIN

    # one agent, two enemies. After circling the first for 2 min, it should move
    # on and identify the second.
    w = World(
        grid=Grid(width=60, height=40),
        usvs=[USV(id="A", position=Vec2(x=10, y=20), mode=USVMode.LLM)],
        boats=[
            Boat(id="E1", position=Vec2(x=13, y=20),
                 true_class=Classification.ENEMY, behavior=Behavior.STANDOFF,
                 standoff_speed=9.0, start_pos=Vec2(x=13, y=20),
                 end_goal=Vec2(x=13, y=20)),
            Boat(id="E2", position=Vec2(x=40, y=20),
                 true_class=Classification.ENEMY, behavior=Behavior.STANDOFF,
                 standoff_speed=9.0, start_pos=Vec2(x=40, y=20),
                 end_goal=Vec2(x=40, y=20)),
        ],
    )
    engine = SimulationEngine(w, dt_min=1.0, seed=0)
    # run past the circle duration, then long enough for the USV to travel to the
    # second enemy and identify it. E2 is a moving standoff enemy (9 kn) that
    # drifts while chased, so allow a generous span at real scale.
    for _ in range(int(CIRCLE_DURATION_MIN) + 150):
        engine.tick()
    # both enemies should end up identified (USV moved on to the second)
    assert w.boat("E1").identified
    assert w.boat("E2").identified


def test_escorted_boat_keeps_course_and_leaves_the_grid():
    from reschu.engine import SimulationEngine
    from reschu.models import USV, Grid, World, Boat
    from reschu.types import USVMode, Classification, Behavior

    # A friendly transiting the grid, with an agent escorting it. Being escorted
    # must NOT change the friendly's course — it keeps travelling as it was and
    # exits naturally at the far edge; the USV just keeps station on it.
    w = World(
        grid=Grid(width=60, height=40),
        usvs=[USV(id="A", position=Vec2(x=12, y=20), mode=USVMode.LLM)],
        boats=[
            Boat(id="F", position=Vec2(x=10, y=20),
                 true_class=Classification.FRIENDLY, behavior=Behavior.FRIENDLY,
                 start_pos=Vec2(x=10, y=20)),
        ],
    )
    engine = SimulationEngine(w, dt_min=1.0, seed=0)
    # let the friendly establish its stable course, then record it
    for _ in range(3):
        engine.tick()
    course_before = w.boat("F").straight_heading
    assert course_before is not None

    left = False
    for _ in range(600):
        engine.tick()
        # while still on the grid and being escorted, the stable course must not
        # have been hijacked (no divert-to-edge). Small zigzag of the *visible*
        # heading is fine; the underlying course is what must stay put.
        if not w.boat("F").left_grid and w.boat("F").being_escorted:
            c = w.boat("F").straight_heading
            assert abs(c.x - course_before.x) < 1e-6
            assert abs(c.y - course_before.y) < 1e-6
        if w.boat("F").left_grid:
            left = True
            break
    assert w.boat("F").identified
    assert left  # it still transits off the map, on its own course


def test_agents_work_through_all_boats():
    from reschu.scenario import demo_world
    from reschu.engine import SimulationEngine
    from reschu.types import Behavior

    w = demo_world()
    engine = SimulationEngine(w, dt_min=0.1, seed=42)
    # Friendlies transit across the grid and exit, so an agent only catches one
    # while it is passing through. Standoff enemies flee to the spot farthest from
    # every USV, so agents may need a while to run them down; separation nudges
    # also perturb exact paths. Run long enough to guarantee the invariants below.
    for _ in range(2500):
        engine.tick()
    # Standoff enemies actively flee to the spot farthest from every USV, so one
    # may evade for a long time — "every non-friendly identified" is no longer a
    # hard guarantee by design. Assert the meaningful invariants: agents make real
    # progress (most non-friendlies identified, and every AGGRESSIVE one, since
    # aggressive boats close on USVs), and at least one friendly was caught.
    non_friendly = [b for b in w.boats if b.behavior is not Behavior.FRIENDLY]
    identified = [b for b in non_friendly if b.identified]
    assert len(identified) >= len(non_friendly) - 1   # at most one evader
    aggressive = [b for b in w.boats if b.behavior is Behavior.AGGRESSIVE]
    assert all(b.identified for b in aggressive)
    friendlies = [b for b in w.boats if b.behavior is Behavior.FRIENDLY]
    assert any(b.identified for b in friendlies)


# --------------------------------------------------------------------------- #
# Policy seam: scripted / llm / mdp agents (llm & mdp are stubs for now)
# --------------------------------------------------------------------------- #
def test_policy_selected_by_mode():
    from reschu.models import USV
    from reschu.types import USVMode
    from reschu.policies import policy_for, LLMPolicy, MDPPolicy

    u_llm = USV(id="L", position=Vec2(x=0, y=0), mode=USVMode.LLM)
    u_mdp = USV(id="M", position=Vec2(x=0, y=0), mode=USVMode.MDP)
    assert isinstance(policy_for(u_llm), LLMPolicy)
    assert isinstance(policy_for(u_mdp), MDPPolicy)


def test_stub_modes_currently_identify_like_baseline():
    # LLM/MDP stubs run the internal baseline for now, so each identifies boats.
    from reschu.engine import SimulationEngine
    from reschu.models import USV, Grid, World, Boat
    from reschu.types import USVMode, Classification, Behavior

    def make(mode):
        return World(
            grid=Grid(width=60, height=40),
            usvs=[USV(id="A", position=Vec2(x=10, y=20), mode=mode)],
            boats=[Boat(id="B", position=Vec2(x=20, y=20),
                        true_class=Classification.ENEMY, behavior=Behavior.STANDOFF,
                        standoff_speed=9.0, start_pos=Vec2(x=20, y=20),
                        end_goal=Vec2(x=20, y=20))],
        )

    for mode in (USVMode.LLM, USVMode.MDP):
        w = make(mode)
        eng = SimulationEngine(w, dt_min=0.1, seed=0)
        for _ in range(200):
            eng.tick()
        assert w.boat("B").identified


def test_mixed_modes_in_one_run():
    from reschu.models import USV, Grid, World, Boat
    from reschu.engine import SimulationEngine
    from reschu.types import USVMode, Classification, Behavior

    w = World(
        grid=Grid(width=80, height=40),
        usvs=[
            USV(id="L", position=Vec2(x=10, y=20), mode=USVMode.LLM),
            USV(id="M", position=Vec2(x=70, y=20), mode=USVMode.MDP),
        ],
        boats=[
            Boat(id="B1", position=Vec2(x=20, y=20), true_class=Classification.CIVILIAN,
                 behavior=Behavior.CIVILIAN, start_pos=Vec2(x=20, y=20)),
            Boat(id="B2", position=Vec2(x=60, y=20), true_class=Classification.CIVILIAN,
                 behavior=Behavior.CIVILIAN, start_pos=Vec2(x=60, y=20)),
        ],
        sensor_noise=1.0,   # default sensing: the regime the MDP's cue model is fitted for
    )
    # runtime tick (1 s); the MDP decides from cues, so give it a few minutes
    eng = SimulationEngine(w, dt_min=1 / 60, seed=0)
    for _ in range(10 * 60):
        eng.tick()
    # both USVs (different modes) worked -> both boats identified
    assert all(b.identified for b in w.boats)


# --------------------------------------------------------------------------- #
# Truth-free observation boundary: agents cannot see ground truth
# --------------------------------------------------------------------------- #
def test_observation_hides_ground_truth():
    from reschu.models import USV, Grid, World, Boat
    from reschu.observation import observe, BoatView
    from reschu.types import USVMode, Classification, Behavior

    w = World(
        grid=Grid(width=100, height=60),
        usvs=[USV(id="A", position=Vec2(x=0, y=0), mode=USVMode.LLM)],
        boats=[Boat(id="B", position=Vec2(x=10, y=0),
                    true_class=Classification.ENEMY, behavior=Behavior.AGGRESSIVE,
                    standoff_speed=9.0)],
    )
    obs = observe(w.usvs[0], w, w.mission_time_min)
    ob = obs.boat("B")
    assert isinstance(ob, BoatView)
    # the observed boat exposes perceived info...
    assert ob.perceived_class.value == "unknown"  # not yet identified
    assert ob.distance == 10.0
    # ...and does NOT expose any ground-truth fields
    assert not hasattr(ob, "true_class")
    assert not hasattr(ob, "behavior")
    assert not hasattr(ob, "standoff_speed")


def test_observation_reveals_only_after_identification():
    from reschu.models import USV, Grid, World, Boat
    from reschu.observation import observe
    from reschu.types import USVMode, Classification, Behavior, Perception

    w = World(
        grid=Grid(width=100, height=60),
        usvs=[USV(id="A", position=Vec2(x=0, y=0), mode=USVMode.LLM)],
        boats=[Boat(id="B", position=Vec2(x=3, y=0),
                    true_class=Classification.ENEMY, behavior=Behavior.AGGRESSIVE)],
    )
    # before identification: perceived unknown
    assert observe(w.usvs[0], w, 0.0).boat("B").perceived_class is Perception.UNKNOWN
    # after a correct identification: perceived matches (revealed through the
    # identification mechanic, not leaked)
    w.boat("B").identify()
    assert observe(w.usvs[0], w, 0.0).boat("B").perceived_class is Perception.ENEMY


# --------------------------------------------------------------------------- #
# Simple scenario features: stationary USVs, Bernoulli identification
# --------------------------------------------------------------------------- #
def test_stationary_usv_does_not_move():
    from reschu.engine import SimulationEngine
    from reschu.models import USV, Grid, World, Boat
    from reschu.types import USVMode, Classification, Behavior

    w = World(
        grid=Grid(width=100, height=100),
        usvs=[USV(id="S", position=Vec2(x=50, y=50), start_pos=Vec2(x=50, y=50),
                  mode=USVMode.LLM, stationary=True)],
        boats=[Boat(id="B", position=Vec2(x=10, y=10),
                    true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN,
                    start_pos=Vec2(x=10, y=10))],
    )
    eng = SimulationEngine(w, dt_min=0.1, seed=0, assign_at_start=False)
    for _ in range(50):
        eng.tick()
    assert w.usvs[0].position.x == 50 and w.usvs[0].position.y == 50


def test_in_range_boat_identified_on_first_sweep():
    from reschu.identification import auto_identify
    from reschu.models import USV, Grid, World, Boat
    from reschu.types import USVMode, Classification, Behavior

    w = World(
        grid=Grid(width=50, height=50),
        usvs=[USV(id="A", position=Vec2(x=10, y=10), mode=USVMode.LLM, stationary=True)],
        boats=[Boat(id="B", position=Vec2(x=11, y=10),
                    true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN)],
    )
    auto_identify(w)                      # no randomness: identified immediately
    assert w.boat("B").is_correctly_identified()


def test_simple_world_known_enemies_preidentified():
    from reschu.scenario import simple_world

    w = simple_world()
    # B1 (known low enemy) and B3 (known high enemy) start identified with tiers
    assert w.boat("B1").identified and w.boat("B1").threat_tier.value == "L"
    assert w.boat("B3").identified and w.boat("B3").threat_tier.value == "H"
    # the unknown ones start unidentified
    for bid in ("B2", "B4", "B5", "B6"):
        assert not w.boat(bid).identified
    # the manual USV now moves from the start along a set heading (the operator
    # redirects it later); the agent USV moves via its policy
    manual = next(u for u in w.usvs if u.mode.value == "manual")
    agent = next(u for u in w.usvs if u.mode.is_agent())
    assert manual.manual_heading is not None   # moves from t=0
    assert not agent.stationary


# --------------------------------------------------------------------------- #
# Manual classify (UI): operator picks a class, may be wrong
# --------------------------------------------------------------------------- #
def test_manual_classify_records_operator_choice():
    from reschu.models import USV, Grid, World, Boat
    from reschu.identification import manual_identify
    from reschu.types import USVMode, Classification, Behavior, Perception

    w = World(
        grid=Grid(width=50, height=50),
        usvs=[USV(id="M", position=Vec2(x=10, y=10), mode=USVMode.MANUAL)],
        boats=[Boat(id="B", position=Vec2(x=12, y=10),
                    true_class=Classification.ENEMY, behavior=Behavior.AGGRESSIVE)],
    )
    # operator (correctly) calls it an enemy
    assert manual_identify(w, "M", "B", as_class=Perception.ENEMY)
    assert w.boat("B").perceived_class is Perception.ENEMY
    assert w.boat("B").is_correctly_identified()


def test_manual_classify_can_be_wrong_and_out_of_range():
    from reschu.models import USV, Grid, World, Boat
    from reschu.identification import manual_identify
    from reschu.types import USVMode, Classification, Behavior, Perception

    w = World(
        grid=Grid(width=100, height=100),
        usvs=[USV(id="M", position=Vec2(x=10, y=10), mode=USVMode.MANUAL)],
        boats=[
            Boat(id="near", position=Vec2(x=12, y=10),
                 true_class=Classification.ENEMY, behavior=Behavior.AGGRESSIVE),
            Boat(id="far", position=Vec2(x=90, y=90),
                 true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN),
        ],
    )
    # wrong call on the near boat: recorded as friendly, but truth unchanged
    assert manual_identify(w, "M", "near", as_class=Perception.FRIENDLY)
    assert w.boat("near").perceived_class is Perception.FRIENDLY
    assert not w.boat("near").is_correctly_identified()
    assert w.boat("near").behavior is Behavior.AGGRESSIVE  # true behavior intact
    # far boat is out of range -> cannot classify
    assert not manual_identify(w, "M", "far", as_class=Perception.CIVILIAN)
    assert not w.boat("far").identified


# --------------------------------------------------------------------------- #
# Ground-truth hardening: agents' act() cannot reach the World at all
# --------------------------------------------------------------------------- #
def test_agent_act_signature_excludes_world():
    """The real-agent seam act(usv, obs, dt_min) must NOT take `world`, so an
    agent's decision code physically cannot read ground truth from the world."""
    import inspect
    from reschu.policies import Policy

    params = list(inspect.signature(Policy.act).parameters)
    assert "world" not in params, (
        "act() must not receive `world` — that would let an agent reach ground "
        f"truth. Signature params: {params}"
    )
    # it should receive the truth-free observation instead
    assert "obs" in params


def test_observation_object_has_no_world_or_truth():
    """The Observation handed to an agent exposes no World handle and no boat
    ground truth, so there's no path from `obs` back to true identities."""
    from reschu.scenario import simple_world
    from reschu.observation import observe

    w = simple_world()
    agent_usv = next(u for u in w.usvs if u.mode.is_agent())
    obs = observe(agent_usv, w, w.mission_time_min)
    # no back-door reference to the world / full boats
    assert not hasattr(obs, "world")
    for bv in obs.boats:
        assert not hasattr(bv, "true_class")
        assert not hasattr(bv, "behavior")
        assert not hasattr(bv, "standoff_speed")


# --------------------------------------------------------------------------- #
# Elimination mechanics: USV in danger zone, enemy after circling
# --------------------------------------------------------------------------- #
def test_usv_eliminated_in_aggressive_danger_zone():
    from reschu.engine import SimulationEngine, DANGER_ZONE_RADIUS
    from reschu.models import USV, Grid, World, Boat
    from reschu.types import USVMode, Classification, Behavior

    # a USV sitting right on top of an aggressive boat -> eliminated next tick
    w = World(
        grid=Grid(width=100, height=60),
        usvs=[USV(id="U", position=Vec2(x=50, y=30), mode=USVMode.LLM)],
        boats=[Boat(id="A", position=Vec2(x=50, y=30),
                    true_class=Classification.ENEMY, behavior=Behavior.AGGRESSIVE,
                    start_pos=Vec2(x=50, y=30))],
    )
    eng = SimulationEngine(w, dt_min=0.1, seed=0, assign_at_start=False)
    eng.tick()
    assert len(w.usvs) == 0  # the USV was eliminated (removed from the world)


def test_enemy_eliminated_after_circling():
    from reschu.models import USV, Boat, Grid, World
    from reschu.usv_movement import assigned_agent_step, CIRCLE_DURATION_MIN
    from reschu.types import USVMode, Classification, Behavior, Perception

    w = World(
        grid=Grid(width=100, height=60),
        usvs=[USV(id="U", position=Vec2(x=57, y=30), mode=USVMode.LLM)],
        boats=[Boat(id="E", position=Vec2(x=50, y=30),
                    true_class=Classification.ENEMY, behavior=Behavior.STANDOFF,
                    standoff_speed=4.0, start_pos=Vec2(x=50, y=30),
                    identified=True, perceived_class=Perception.ENEMY)],
    )
    u = w.usv("U"); u.assigned_target_id = "E"
    now = 0.0
    for _ in range(int(CIRCLE_DURATION_MIN / 0.1) + 5):
        now += 0.1
        u.position = assigned_agent_step(u, w, 0.1, now)
    assert w.boat("E").left_grid          # eliminated -> disappears
    assert w.boat("E").eliminated


def test_agent_drives_straight_at_target_no_avoidance():
    """Agent USVs no longer steer around aggressive boats — the intended risk is
    that a USV approaches to identify a contact and gets attacked if it MISids.
    So an agent routing to its target drives straight at it, passing near an
    aggressive boat in the way rather than detouring around it."""
    from reschu.models import USV, Boat, Grid, World
    from reschu.usv_movement import assigned_agent_step
    from reschu.types import USVMode, Classification, Behavior

    w = World(
        grid=Grid(width=100, height=60),
        usvs=[USV(id="U", position=Vec2(x=10, y=30), mode=USVMode.LLM)],
        boats=[
            Boat(id="T", position=Vec2(x=90, y=30), true_class=Classification.CIVILIAN,
                 behavior=Behavior.CIVILIAN, start_pos=Vec2(x=90, y=30)),
            Boat(id="X", position=Vec2(x=50, y=30), true_class=Classification.ENEMY,
                 behavior=Behavior.AGGRESSIVE, start_pos=Vec2(x=50, y=30)),
        ],
    )
    u = w.usv("U"); u.assigned_target_id = "T"
    closest = 999.0
    for i in range(200):
        u.position = assigned_agent_step(u, w, 0.1, i * 0.1)
        closest = min(closest, u.position.distance_to(Vec2(x=50, y=30)))
        w.boat("X").position = Vec2(x=50, y=30)  # keep obstacle fixed for the test
    # the straight path from (10,30) to (90,30) runs right past the boat at
    # (50,30); with no avoidance the USV passes very close to it.
    assert closest < 2.0


# --------------------------------------------------------------------------- #
# Configurable scenario (settings page): world_from_config
# --------------------------------------------------------------------------- #
def test_default_config_reproduces_simple_scenario():
    from reschu.scenario import world_from_config, default_config
    w = world_from_config(default_config())
    assert len(w.usvs) == 4
    assert len(w.boats) == 6
    modes = [u.mode.value for u in w.usvs]
    assert modes == ["manual", "llm", "mdp", "llm"]
    # B1 and B3 are the pre-identified ("known") enemies
    assert w.boat("B1").identified and w.boat("B3").identified


def test_world_from_config_custom():
    from reschu.scenario import world_from_config
    from reschu.types import USVMode, Behavior, Classification
    cfg = {
        "grid": {"width": 80, "height": 50},
        "usvs": [{"mode": "manual", "start": [10, 10]},
                 {"mode": "mdp", "start": [70, 40]}],
        "boats": [
            {"true_class": "enemy", "behavior": "standoff", "tier": "med",
             "start": [40, 25], "known": True},
            {"true_class": "civilian", "behavior": "civilian", "start": [5, 5]},
        ],
    }
    w = world_from_config(cfg)
    assert w.grid.width == 80 and w.grid.height == 50
    assert len(w.usvs) == 2 and len(w.boats) == 2
    assert w.usvs[0].mode is USVMode.MANUAL
    assert w.usvs[1].mode is USVMode.MDP
    b1 = w.boat("B1")
    assert b1.behavior is Behavior.STANDOFF and b1.standoff_speed == 9.0  # med tier
    assert b1.identified  # known
    assert w.boat("B2").behavior is Behavior.CIVILIAN


# --------------------------------------------------------------------------- #
# Head-on separation: two entities meeting head-on each turn 10 deg to their
# right (starboard) at half a grid apart, passing port-to-port.
# --------------------------------------------------------------------------- #
def test_head_on_pair_both_turn_right():
    from reschu.models import Boat, Grid, World
    from reschu.usv_movement import resolve_separation
    from reschu.types import Classification, Behavior

    # A heads east, B heads west, on a collision course half a grid apart
    w = World(
        grid=Grid(width=80, height=50),
        usvs=[],
        boats=[
            Boat(id="A", position=Vec2(x=40, y=25), true_class=Classification.CIVILIAN,
                 behavior=Behavior.CIVILIAN, heading=Vec2(x=1, y=0)),
            Boat(id="B", position=Vec2(x=40.4, y=25), true_class=Classification.CIVILIAN,
                 behavior=Behavior.CIVILIAN, heading=Vec2(x=-1, y=0)),
        ],
    )
    resolve_separation(w, 1 / 60)
    # each turned to its right: the east-goer now has a negative-y (south)
    # component, the west-goer a positive-y (north) component -> port-to-port
    assert w.boat("A").heading.y < 0
    assert w.boat("B").heading.y > 0


def test_same_direction_pair_not_triggered():
    from reschu.models import Boat, Grid, World
    from reschu.usv_movement import resolve_separation
    from reschu.types import Classification, Behavior

    # both head east (overtaking, not head-on) -> the rule must not fire
    w = World(
        grid=Grid(width=80, height=50),
        usvs=[],
        boats=[
            Boat(id="C", position=Vec2(x=40, y=25), true_class=Classification.CIVILIAN,
                 behavior=Behavior.CIVILIAN, heading=Vec2(x=1, y=0)),
            Boat(id="D", position=Vec2(x=40.4, y=25), true_class=Classification.CIVILIAN,
                 behavior=Behavior.CIVILIAN, heading=Vec2(x=1, y=0)),
        ],
    )
    resolve_separation(w, 1 / 60)
    assert abs(w.boat("C").heading.y) < 1e-9  # unchanged
    assert abs(w.boat("D").heading.y) < 1e-9


# --------------------------------------------------------------------------- #
# MDP identification: declarations instead of auto-identify
# --------------------------------------------------------------------------- #
def _mdp_world(usv_x=10.0, true_class=None, behavior=None):
    from reschu.models import USV, Grid, World, Boat
    from reschu.types import USVMode, Classification, Behavior
    return World(
        grid=Grid(width=50, height=50),
        usvs=[USV(id="M", position=Vec2(x=usv_x, y=10), mode=USVMode.MDP, stationary=True)],
        boats=[Boat(id="B", position=Vec2(x=11, y=10),
                    true_class=true_class or Classification.ENEMY,
                    behavior=behavior or Behavior.AGGRESSIVE)],
    )


def test_mdp_usv_is_not_auto_identified():
    from reschu.identification import auto_identify
    w = _mdp_world()
    auto_identify(w)
    assert not w.boat("B").identified       # MDP must declare it itself


def test_correct_declaration_in_range_identifies():
    from reschu.identification import apply_declarations
    from reschu.types import Perception
    w = _mdp_world()
    u = w.usvs[0]
    u.pending_declaration = ("B", Perception.ENEMY)
    assert apply_declarations(w) == ["B"]
    b = w.boat("B")
    assert b.is_correctly_identified()
    assert b.threat_tier is not None        # correct enemy -> tier shown
    assert u.pending_declaration is None    # request consumed
    assert "B" in u.attempted_boat_ids


def test_wrong_declaration_is_the_agents_own_misid():
    from reschu.identification import apply_declarations
    from reschu.types import Perception, Behavior
    w = _mdp_world()
    w.usvs[0].pending_declaration = ("B", Perception.FRIENDLY)
    apply_declarations(w)
    b = w.boat("B")
    assert b.identified and not b.is_correctly_identified()
    assert b.perceived_class is Perception.FRIENDLY
    assert b.behavior is Behavior.AGGRESSIVE   # true behavior unchanged
    assert b.threat_tier is None
    assert b.id_attempts == 1


def test_declaration_out_of_range_is_dropped():
    from reschu.identification import apply_declarations
    from reschu.types import Perception
    w = _mdp_world(usv_x=40.0)                 # far beyond ID range
    u = w.usvs[0]
    u.pending_declaration = ("B", Perception.ENEMY)
    assert apply_declarations(w) == []
    assert not w.boat("B").identified
    assert u.pending_declaration is None       # dropped; policy may re-request


def test_declaring_unknown_is_a_noop():
    from reschu.identification import apply_declarations
    from reschu.types import Perception
    w = _mdp_world()
    w.usvs[0].pending_declaration = ("B", Perception.UNKNOWN)
    apply_declarations(w)
    assert not w.boat("B").identified


def test_mode_switch_clears_pending_declaration():
    from reschu.types import Perception, USVMode
    w = _mdp_world()
    u = w.usvs[0]
    u.pending_declaration = ("B", Perception.ENEMY)
    u._switch_mode(USVMode.MANUAL)
    assert u.pending_declaration is None


def test_observation_exposes_kinematics_but_not_truth():
    from reschu.observation import observe
    from reschu.types import Behavior, Classification
    w = _mdp_world(true_class=Classification.ENEMY, behavior=Behavior.STANDOFF)
    boat = w.boat("B")
    boat.standoff_speed = 9.0
    boat.heading = Vec2(x=3.0, y=4.0)
    ob = observe(w.usvs[0], w, w.mission_time_min).boat("B")
    assert ob.speed_knots == 9.0                       # observed speed
    assert abs(ob.heading.x - 0.6) < 1e-9 and abs(ob.heading.y - 0.8) < 1e-9  # unit heading
    for truth in ("true_class", "behavior", "standoff_speed"):
        assert not hasattr(ob, truth)


# --------------------------------------------------------------------------- #
# Boat speed variation: noisy observed speed, tier still from nominal speed
# --------------------------------------------------------------------------- #
def _speed_world(spread=0.0, jitter=0.0, n=40):
    from reschu.models import Grid, World, Boat
    from reschu.types import Classification, Behavior
    boats = [Boat(id=f"B{i}", position=Vec2(x=50, y=30),
                  true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN,
                  start_pos=Vec2(x=50, y=30)) for i in range(n)]
    return World(grid=Grid(width=100, height=60), boats=boats,
                 speed_spread=spread, speed_jitter=jitter)


def test_no_variation_means_exact_class_speed():
    from reschu.engine import SimulationEngine
    w = _speed_world()
    SimulationEngine(w, dt_min=1/60, seed=0).tick()
    assert all(b.speed_scale == 1.0 and b.speed_knots() == 3.0 for b in w.boats)


def test_variation_gives_each_boat_its_own_speed():
    from reschu.engine import SimulationEngine, SPEED_SCALE_MIN, SPEED_SCALE_MAX
    w = _speed_world(spread=0.15)
    SimulationEngine(w, dt_min=1/60, seed=0).tick()
    scales = [b.speed_scale for b in w.boats]
    assert len(set(scales)) > 30                          # boats differ
    assert all(SPEED_SCALE_MIN <= f <= SPEED_SCALE_MAX for f in scales)
    assert all(b.nominal_speed_knots() == 3.0 for b in w.boats)  # nominal untouched


def test_variation_is_reproducible_by_seed():
    from reschu.engine import SimulationEngine
    a, b = _speed_world(0.15, 0.05), _speed_world(0.15, 0.05)
    ea, eb = SimulationEngine(a, dt_min=1/60, seed=7), SimulationEngine(b, dt_min=1/60, seed=7)
    for _ in range(30):
        ea.tick(); eb.tick()
    assert [x.speed_knots() for x in a.boats] == [y.speed_knots() for y in b.boats]


def test_jitter_drifts_within_bounds():
    from reschu.engine import SimulationEngine, SPEED_JITTER_CLAMP
    w = _speed_world(jitter=0.05, n=5)
    eng = SimulationEngine(w, dt_min=1/60, seed=3)
    seen = []
    for _ in range(300):
        eng.tick()
        seen.append(w.boats[0].speed_jitter)
    assert all(abs(j) <= SPEED_JITTER_CLAMP * 0.05 + 1e-12 for j in seen)
    assert max(seen) != min(seen)                          # it actually drifts
    # smooth: tick-to-tick change is much smaller than the overall spread
    assert max(abs(x - y) for x, y in zip(seen, seen[1:])) < 0.06


def test_tier_comes_from_nominal_not_varied_speed():
    from reschu.models import Boat
    from reschu.types import Classification, Behavior, ThreatTier
    low = Boat(id="L", position=Vec2(x=0, y=0), true_class=Classification.ENEMY,
               behavior=Behavior.STANDOFF, standoff_speed=4.0, speed_scale=1.3)
    assert low.speed_knots() > 4.0                          # moves faster than 4...
    low.identify()
    assert low.threat_tier is ThreatTier.LOW               # ...still LOW tier


def test_observation_reports_actual_varied_speed():
    from reschu.models import USV, Grid, World, Boat
    from reschu.observation import observe
    from reschu.types import Classification, Behavior, USVMode
    w = World(grid=Grid(width=50, height=50),
              usvs=[USV(id="M", position=Vec2(x=0, y=0), mode=USVMode.MDP)],
              boats=[Boat(id="B", position=Vec2(x=5, y=0), true_class=Classification.CIVILIAN,
                          behavior=Behavior.CIVILIAN, speed_scale=1.2, speed_jitter=0.05)])
    ob = observe(w.usvs[0], w, 0.0).boat("B")
    assert abs(ob.speed_knots - 3.0 * 1.2 * 1.05) < 1e-9   # actual, not nominal 3.0


def test_config_enables_variation_by_default_and_can_disable():
    from reschu.scenario import (default_config, world_from_config,
                                 DEFAULT_SPEED_SPREAD, DEFAULT_SPEED_JITTER)
    w = world_from_config(default_config())
    assert (w.speed_spread, w.speed_jitter) == (DEFAULT_SPEED_SPREAD, DEFAULT_SPEED_JITTER)
    cfg = default_config(); cfg["speed_spread"] = 0; cfg["speed_jitter"] = 0
    w0 = world_from_config(cfg)
    assert (w0.speed_spread, w0.speed_jitter) == (0.0, 0.0)


def test_back_to_back_missions_are_identical():
    """A mission's outcome must not depend on missions run before it in the same
    process (e.g. one trial after another on the console server)."""
    from reschu.scenario import default_config, world_from_config

    def run():
        w = world_from_config(default_config())
        e = SimulationEngine(w, dt_min=1 / 60, seed=11)
        for _ in range(240):
            e.tick()
        return [(u.id, u.position.x, u.position.y) for u in w.usvs]

    assert run() == run()


def test_time_to_id_runs_from_spawn_and_skips_known_contacts():
    from reschu.scenario import default_config, world_from_config
    from reschu.web.server import _time_to_id
    w = world_from_config(default_config())
    known = [b for b in w.boats if b.known_at_start]
    assert known and all(b.identified for b in known)
    assert all(_time_to_id(b) == "" for b in known)          # no decision was needed
    e = SimulationEngine(w, dt_min=1 / 60, seed=2)
    for _ in range(60 * 12):
        e.tick()
    spawned = [b for b in w.boats if b.spawned_min > 0]
    for b in spawned:                                         # replacement friendlies
        assert not b.known_at_start
        if b.first_identified_min is not None:
            assert _time_to_id(b) == round(b.first_identified_min - b.spawned_min, 3)
