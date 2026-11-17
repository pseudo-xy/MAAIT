"""Greedy control mode, and the wrong-ID setting shared by all agent USVs
(World.agent_wrong_ids)."""
import pytest

from reschu.engine import SimulationEngine
from reschu.scenario import default_config, world_from_config
from reschu.types import USVMode


def _run(wrong, seed, minutes=12):
    cfg = default_config()
    cfg["agent_wrong_ids"] = wrong
    for u in cfg["usvs"]:
        u["mode"] = "greedy"
        u.pop("timeline", None)
    w = world_from_config(cfg)
    eng = SimulationEngine(w, dt_min=1 / 60, seed=seed)
    for _ in range(int(minutes * 60)):
        eng.tick()
    return w


def test_greedy_mode_is_an_agent():
    assert USVMode.GREEDY.is_agent()
    w = world_from_config({**default_config(), "usvs": [{"mode": "greedy", "start": "A1"}]})
    assert w.usvs[0].mode is USVMode.GREEDY


@pytest.mark.parametrize("wrong", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exactly_the_set_number_of_wrong_ids(wrong, seed):
    w = _run(wrong, seed)
    by_greedy = [b for b in w.boats if b.identified_by_mode == "greedy"]
    assert w.agent_ids_made == len(by_greedy) >= wrong
    bad = [b for b in by_greedy if not b.is_correctly_identified()]
    assert len(bad) == w.agent_wrong_counted == w.agent_wrong_forced == wrong
    assert all(b.id_error == "forced" for b in bad)


def test_wrong_label_sticks():
    w = _run(3, 5)
    for b in w.boats:
        if b.id_error:
            assert b.id_attempts == 1      # nobody re-identified it


def test_old_config_key_still_read():
    cfg = default_config()
    cfg["greedy_wrong_ids"] = 2
    assert world_from_config(cfg).agent_wrong_ids == 2
