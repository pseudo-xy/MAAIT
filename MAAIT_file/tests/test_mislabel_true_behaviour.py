"""A wrongly labelled boat keeps its true behaviour until it is relabelled.
In particular, circling a boat wrongly labelled ENEMY never removes it."""
from reschu.identification import manual_identify
from reschu.manual_engagement import update_manual_engagements
from reschu.models import Boat, USV, Vec2, World, Grid
from reschu.types import Behavior, Classification, Perception, USVMode
from reschu.usv_movement import CIRCLE_DURATION_MIN


def _world(true_class, behavior):
    g = Grid(width=40.0, height=25.0)
    u = USV(id="USV-1", position=Vec2(x=10, y=10), mode=USVMode.MANUAL)
    b = Boat(id="B1", position=Vec2(x=11, y=10), true_class=true_class, behavior=behavior)
    return World(grid=g, usvs=[u], boats=[b])


def _circle_out(w):
    t, dt = 0.0, 1 / 60
    while t < CIRCLE_DURATION_MIN + 0.1:
        update_manual_engagements(w, t, dt)
        t += dt


def test_civilian_labelled_enemy_is_not_removed_by_circling():
    w = _world(Classification.CIVILIAN, Behavior.CIVILIAN)
    assert manual_identify(w, "USV-1", "B1", Perception.ENEMY)
    _circle_out(w)
    b = w.boat("B1")
    assert not b.left_grid and not b.eliminated
    assert b.circle_done and b.perceived_class is Perception.ENEMY
    assert not b.needs_work()          # not circled again under the same label


def test_real_enemy_is_still_eliminated():
    w = _world(Classification.ENEMY, Behavior.STANDOFF)
    assert manual_identify(w, "USV-1", "B1", Perception.ENEMY)
    _circle_out(w)
    assert w.boat("B1").eliminated


def test_relabel_clears_circle_done():
    w = _world(Classification.CIVILIAN, Behavior.CIVILIAN)
    manual_identify(w, "USV-1", "B1", Perception.ENEMY)
    _circle_out(w)
    assert manual_identify(w, "USV-1", "B1", Perception.CIVILIAN)
    b = w.boat("B1")
    assert not b.circle_done and b.is_correctly_identified()
