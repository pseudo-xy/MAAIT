"""The wrong-ID setting is shared by all agent USVs (MDP, Greedy, LLM). The
MDP's own mistakes count toward it; after it is reached nothing is forced and
any further MDP mistakes are kept and counted as extra."""
import random

import pytest

from reschu.engine import SimulationEngine
from reschu.identification import _agent_label, plan_agent_errors
from reschu.models import Boat, Grid, USV, World
from reschu.scenario import default_config, world_from_config
from reschu.types import Behavior, Classification, Perception, USVMode, Vec2


def _world(n_boats=6, wrong=3, plan=None):
    boats = [Boat(id=f"B{i}", position=Vec2(x=10 + i, y=10), start_pos=Vec2(x=10 + i, y=10),
                  true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN)
             for i in range(1, n_boats + 1)]
    w = World(grid=Grid(width=40, height=25), boats=boats,
              usvs=[USV(id="A", position=Vec2(x=1, y=1), mode=USVMode.MDP)], agent_wrong_ids=wrong)
    plan_agent_errors(w, random.Random(0))
    if plan is not None:
        w.agent_error_plan = plan
    return w


def _id(w, i, proposed):
    b = w.boats[i]
    label, kind = _agent_label(w, b, proposed, random.Random(1))
    b.identify(as_class=label)
    b.id_error = kind
    return label, kind


C, E = Perception.CIVILIAN, Perception.ENEMY


def test_mdp_own_mistakes_count_then_extra():
    w = _world(wrong=2, plan=[5, 6])
    assert _id(w, 0, E)[1] == "own"          # MDP wrong on its own -> counts
    assert _id(w, 1, C) == (C, None)         # right, not a planned position
    assert _id(w, 2, E)[1] == "own"          # second own mistake -> set number reached
    assert _id(w, 3, E) == (E, "extra")      # another mistake: kept, counted as extra
    assert _id(w, 4, C) == (C, None)         # planned position, but nothing left to force
    assert (w.agent_wrong_counted, w.agent_wrong_own, w.agent_wrong_forced,
            w.agent_wrong_extra) == (2, 2, 0, 1)


def test_correct_ids_forced_at_planned_points():
    w = _world(wrong=2, plan=[2, 4])
    assert _id(w, 0, C) == (C, None)
    lab, kind = _id(w, 1, C)
    assert kind == "forced" and lab is not C
    assert _id(w, 2, C) == (C, None)
    assert _id(w, 3, C)[1] == "forced"
    assert w.agent_wrong_counted == w.agent_wrong_forced == 2


def test_catch_up_when_contacts_run_out():
    w = _world(n_boats=3, wrong=2, plan=[5, 6])   # plan beyond the contacts
    assert _id(w, 0, C) == (C, None)              # 3 left, 2 owed -> still free
    assert _id(w, 1, C)[1] == "forced"            # 2 left, 2 owed -> must force
    assert _id(w, 2, C)[1] == "forced"
    assert w.agent_wrong_counted == 2


def _run(cfg_modes, wrong, seed, minutes=12):
    cfg = default_config()
    cfg["agent_wrong_ids"] = wrong
    for u, m in zip(cfg["usvs"], cfg_modes):
        u["mode"] = m
        u.pop("timeline", None)
    w = world_from_config(cfg)
    eng = SimulationEngine(w, dt_min=1 / 60, seed=seed)
    for _ in range(int(minutes * 60)):
        eng.tick()
    return w


@pytest.mark.parametrize("modes", [("manual", "mdp", "mdp", "mdp"),
                                   ("manual", "greedy", "mdp", "llm")])
@pytest.mark.parametrize("wrong", [0, 2, 3])
@pytest.mark.parametrize("seed", [1, 2])
def test_full_missions_hit_the_set_number(modes, wrong, seed):
    w = _run(modes, wrong, seed)
    agent = [b for b in w.boats if b.identified_by_mode in ("mdp", "greedy", "llm")]
    wrong_labels = [b for b in agent if not b.is_correctly_identified()]
    assert w.agent_ids_made == len(agent)
    if w.agent_ids_made >= wrong:
        assert w.agent_wrong_counted == wrong
    assert w.agent_wrong_counted == w.agent_wrong_forced + w.agent_wrong_own
    # every wrong agent label is accounted for: counted toward the number, or extra
    assert len(wrong_labels) == w.agent_wrong_counted + w.agent_wrong_extra
    assert sum(b.id_error == "extra" for b in agent) == w.agent_wrong_extra
