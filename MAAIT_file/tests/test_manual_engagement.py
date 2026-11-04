"""Operator follow-up after a manual identification (manual_engagement.py):
civilian -> left alone; friendly -> keeps going to its exit, operator follows it;
enemy -> eliminated after the operator circles it for CIRCLE_DURATION_MIN within
identification range. AI USVs leave those contacts alone."""

from reschu.assignment import reassign_free_agents
from reschu.engine import SimulationEngine
from reschu.identification import identify_range, manual_identify
from reschu.manual_engagement import circle_progress
from reschu.models import USV, Boat, Grid, World
from reschu.types import Behavior, Classification, Perception, USVMode, Vec2
from reschu.usv_movement import CIRCLE_DURATION_MIN

DT = 1 / 60


def _world(kind="enemy", boat_xy=(20, 12), agent=True):
    tc, beh = {"enemy": (Classification.ENEMY, Behavior.STANDOFF),
               "friendly": (Classification.FRIENDLY, Behavior.FRIENDLY),
               "civilian": (Classification.CIVILIAN, Behavior.CIVILIAN)}[kind]
    bx, by = boat_xy
    usvs = [USV(id="M", position=Vec2(x=bx - 1.0, y=by), mode=USVMode.MANUAL)]
    if agent:
        usvs.append(USV(id="A", position=Vec2(x=bx + 6, y=by), mode=USVMode.LLM))
    boat = Boat(id="B1", position=Vec2(x=bx, y=by), start_pos=Vec2(x=bx, y=by),
                true_class=tc, behavior=beh, standoff_speed=0.0001)
    return World(grid=Grid(width=40, height=25), usvs=usvs, boats=[boat])


def _keep_near(w, frac=0.5):
    b, u = w.boats[0], w.usvs[0]
    u.position = Vec2(x=b.position.x - frac * identify_range(w.grid), y=b.position.y)
    u.manual_destination = None


def test_civilian_is_left_alone():
    w = _world("civilian")
    assert manual_identify(w, "M", "B1", Perception.CIVILIAN)
    assert w.boats[0].manual_handler is None and not w.boats[0].needs_work()


def test_identified_contact_is_claimed_and_ai_lets_go():
    w = _world("enemy")
    w.usvs[1].assigned_target_id = "B1"
    assert manual_identify(w, "M", "B1", Perception.ENEMY)
    b = w.boats[0]
    assert b.manual_handler == "M" and b.claimed_by == "M"
    assert w.usvs[1].assigned_target_id is None
    assert not b.needs_work()
    reassign_free_agents(w)
    assert w.usvs[1].assigned_target_id != "B1"


def test_enemy_eliminated_after_the_ai_circling_time():
    w = _world("enemy", agent=False)
    eng = SimulationEngine(w, dt_min=DT, seed=0, assign_at_start=False)
    manual_identify(w, "M", "B1", Perception.ENEMY)
    n = round(CIRCLE_DURATION_MIN / DT)
    for _ in range(n - 2):
        _keep_near(w); eng.tick()
    assert not w.boats[0].eliminated and 0.9 < circle_progress(w.boats[0]) < 1
    for _ in range(3):
        _keep_near(w); eng.tick()
    b = w.boats[0]
    assert b.eliminated and b.left_grid
    assert w.usvs[0].circle_events[-1][0] == "B1"


def test_time_out_of_range_does_not_count_but_is_kept():
    w = _world("enemy", agent=False)
    eng = SimulationEngine(w, dt_min=DT, seed=0, assign_at_start=False)
    manual_identify(w, "M", "B1", Perception.ENEMY)
    for _ in range(20):
        _keep_near(w); eng.tick()
    p = circle_progress(w.boats[0])
    for _ in range(60):
        _keep_near(w, frac=3.0); eng.tick()
    assert circle_progress(w.boats[0]) == p and not w.boats[0].eliminated


def test_friendly_keeps_going_to_its_exit_and_is_logged():
    w = _world("friendly", boat_xy=(37, 12), agent=False)
    w.boats[0].exit_point = Vec2(x=40, y=12)
    eng = SimulationEngine(w, dt_min=DT, seed=0, assign_at_start=False)
    manual_identify(w, "M", "B1", Perception.FRIENDLY)
    b = w.boats[0]
    assert b.manual_handler == "M" and b.escort_start_min is not None
    for _ in range(3000):
        eng.tick()
        if b.left_grid:
            break
    assert b.left_grid and not b.eliminated and b.escort_end_min is not None


def test_claim_released_when_manual_usv_returns_to_base():
    w = _world("friendly")
    eng = SimulationEngine(w, dt_min=DT, seed=0, assign_at_start=False)
    manual_identify(w, "M", "B1", Perception.FRIENDLY)
    w.usvs[0].rtb = True
    eng.tick()
    assert w.boats[0].manual_handler is None and w.boats[0].needs_work()


def test_claim_released_when_manual_usv_is_destroyed():
    w = _world("enemy")
    eng = SimulationEngine(w, dt_min=DT, seed=0, assign_at_start=False)
    manual_identify(w, "M", "B1", Perception.ENEMY)
    w.usvs[0].eliminated = True
    eng.tick()
    assert w.boats[0].manual_handler is None and w.boats[0].needs_work()


def test_claim_kept_while_manual_usv_is_away_from_the_contact():
    w = _world("enemy")
    eng = SimulationEngine(w, dt_min=DT, seed=0, assign_at_start=False)
    manual_identify(w, "M", "B1", Perception.ENEMY)
    for _ in range(30):
        _keep_near(w, frac=4.0); eng.tick()
    assert w.boats[0].manual_handler == "M" and w.usvs[1].assigned_target_id != "B1"


def test_timeline_switch_to_ai_keeps_the_contact_and_finishes_the_circle():
    w = _world("enemy")
    eng = SimulationEngine(w, dt_min=DT, seed=0, assign_at_start=False)
    manual_identify(w, "M", "B1", Perception.ENEMY)
    for _ in range(30):                      # half a minute of manual circling
        _keep_near(w); eng.tick()
    t = w.mission_time_min
    w.usvs[0].mode_schedule = [
        {"mode": USVMode.MANUAL, "start_min": 0.0, "end_min": t + DT / 2},
        {"mode": USVMode.LLM, "start_min": t + DT / 2, "end_min": 99.0}]
    eng.tick()
    m, b = w.usvs[0], w.boats[0]
    assert m.mode is USVMode.LLM and m.assigned_target_id == "B1"
    assert b.manual_handler is None and w.usvs[1].assigned_target_id != "B1"
    for _ in range(40):                      # the rest of the minute, by the AI
        eng.tick()
        if b.eliminated:
            break
    assert b.eliminated and m.circle_events[-1][0] == "B1"
    assert w.mission_time_min - t < 0.6     # it did not restart the full minute
