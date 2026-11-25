"""A friendly heads for its exit only once it is correctly identified as
friendly. Unknown or wrongly labelled, it cruises and turns back at the edge."""
import random
from reschu.behaviors import step_boat
from reschu.models import Boat, Grid, USV, Vec2, World
from reschu.types import Behavior, Classification, Perception, USVMode


def _world():
    g = Grid(width=40.0, height=25.0)
    b = Boat(id="B1", position=Vec2(x=20, y=12), true_class=Classification.FRIENDLY,
             behavior=Behavior.FRIENDLY, exit_point=Vec2(x=40, y=12))
    u = USV(id="USV-1", position=Vec2(x=5, y=5), mode=USVMode.MANUAL)
    return World(grid=g, usvs=[u], boats=[b])


def _run(w, minutes):
    rng = random.Random(1)
    for _ in range(int(minutes * 60)):
        b = w.boat("B1")
        if b.left_grid: break
        b.position = step_boat(b, w, 1 / 60, rng)


def test_unknown_friendly_never_leaves():
    w = _world(); _run(w, 15)
    assert not w.boat("B1").left_grid


def test_wrongly_labelled_friendly_never_leaves():
    w = _world(); w.boat("B1").identify(Perception.ENEMY); _run(w, 15)
    assert not w.boat("B1").left_grid


def test_correctly_identified_friendly_exits():
    w = _world(); w.boat("B1").identify(Perception.FRIENDLY); _run(w, 15)
    assert w.boat("B1").left_grid


def test_relabel_to_friendly_then_exits():
    w = _world(); b = w.boat("B1")
    b.identify(Perception.CIVILIAN); _run(w, 5); assert not b.left_grid
    b.identify(Perception.FRIENDLY); _run(w, 15); assert b.left_grid
