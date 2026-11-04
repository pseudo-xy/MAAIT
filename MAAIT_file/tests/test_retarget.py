"""Operator picks an agent USV's new target from the list of boats on the map."""

from reschu.assignment import retarget_usv
from reschu.engine import SimulationEngine
from reschu.identification import manual_identify
from reschu.models import USV, Boat, Grid, World
from reschu.types import Behavior, Classification, Perception, USVMode, Vec2

DT = 1 / 60


def _world(mode=USVMode.LLM):
    usvs = [USV(id="A", position=Vec2(x=5, y=5), mode=mode),
            USV(id="B", position=Vec2(x=30, y=20), mode=mode),
            USV(id="M", position=Vec2(x=20, y=5), mode=USVMode.MANUAL)]
    boats = [Boat(id=f"B{i}", position=Vec2(x=x, y=y), start_pos=Vec2(x=x, y=y),
                  true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN)
             for i, (x, y) in enumerate([(8, 8), (28, 18), (21, 6)], start=1)]
    return World(grid=Grid(width=40, height=25), usvs=usvs, boats=boats)


def test_pick_a_free_boat():
    w = _world(); w.usvs[0].assigned_target_id = "B1"
    assert retarget_usv(w, "A", "B3") == "B3"
    a = w.usv("A")
    assert a.assigned_target_id == "B3" and a.assignment_pinned


def test_pick_a_boat_engaged_by_another_usv():
    w = _world()
    w.usv("A").assigned_target_id = "B1"; w.usv("B").assigned_target_id = "B2"
    assert retarget_usv(w, "A", "B2") == "B2"
    assert w.usv("A").assigned_target_id == "B2"
    assert w.usv("B").assigned_target_id != "B2"


def test_pick_a_boat_the_operator_was_handling():
    w = _world()
    w.boats[2].true_class = Classification.ENEMY
    w.usv("M").position = Vec2(x=21.3, y=6)
    manual_identify(w, "M", "B3", Perception.ENEMY)
    assert w.boat("B3").manual_handler == "M"
    retarget_usv(w, "A", "B3")
    assert w.usv("A").assigned_target_id == "B3" and w.boat("B3").manual_handler is None


def test_invalid_picks_are_ignored():
    w = _world()
    assert retarget_usv(w, "M", "B1") is None          # manual USV: not via this menu
    assert retarget_usv(w, "A", "nope") is None
    w.boats[0].left_grid = True
    assert retarget_usv(w, "A", "B1") is None


def test_mdp_keeps_working_the_operator_pick():
    w = _world(USVMode.MDP)
    eng = SimulationEngine(w, dt_min=DT, seed=0)
    eng.tick()
    retarget_usv(w, "A", "B2", w.mission_time_min)     # far side of the map
    for _ in range(120):
        eng.tick()
        if w.boat("B2").identified:
            break
    assert w.usv("A").assigned_target_id == "B2" or w.boat("B2").identified
