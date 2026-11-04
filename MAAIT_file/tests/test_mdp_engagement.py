"""Tests for the engagement MDP (reschu.mdp.engagement / agent / episodes)."""

import inspect
import random

import numpy as np
import pytest

from reschu.engine import DANGER_ZONE_RADIUS, SimulationEngine
from reschu.identification import IDENTIFY_RANGE, identify_range
from reschu.mdp.agent import EngagementAgent, RushChooser, ThresholdChooser
from reschu.mdp.cue_model import CLASSES
from reschu.mdp.engagement import (ACTIONS, BANDS, DANGER_RATIO, DECLARE_ACTIONS,
                                   DEFAULT_REWARDS, MAP_SIZES, N_CELLS, N_STATES, Counts,
                                   EngagementMDP, PolicyTable, band_index, belief_cell,
                                   describe_state, size_for_width, state_index)
from reschu.mdp.episodes import run_engagement, summarize, transitions
from reschu.models import USV, Boat, Grid, World
from reschu.policies import MDPPolicy, policy_for
from reschu.types import Behavior, Classification, USVMode, Vec2

UNIFORM = {c: 0.2 for c in CLASSES}


def test_danger_ratio_matches_engine():
    assert DANGER_RATIO == pytest.approx(DANGER_ZONE_RADIUS / IDENTIFY_RANGE)


def test_bands_and_cells():
    assert [band_index(r) for r in (0.5, 0.75, 0.9, 1.0, 1.2, 1.4, 2.0, 3.0)] == \
        [0, 0, 1, 1, 2, 3, 4, 5]
    assert len(BANDS) * N_CELLS == N_STATES
    assert 0 <= belief_cell(UNIFORM) < N_CELLS
    sure_civ = {**{c: 0.0 for c in CLASSES}, "civilian": 1.0}
    sure_agg = {**{c: 0.0 for c in CLASSES}, "aggressive": 1.0}
    assert belief_cell(sure_civ) != belief_cell(sure_agg)
    assert "civilian" in describe_state(state_index(1, belief_cell(sure_civ)))
    assert [size_for_width(w) for w in (40, 80, 120, 200)] == ["small", "medium", "large", "large"]


def _synthetic_counts(cell_post, band=4, n=50):
    """A contact that is truly `civilian` and drifts in from `mid` to `id` when
    approached; holding keeps it where it is."""
    c = Counts()
    cell = belief_cell(cell_post)
    post = np.array([cell_post[k] for k in CLASSES])
    for b in range(len(BANDS)):
        s = state_index(b, cell)
        for _ in range(n):
            nxt = state_index(max(1, b - 1), cell)
            c.add({"s": s, "a": 0, "type": "civilian", "belief": post,
                   "outcome": "cont", "s_next": nxt, "ticks": 10})        # approach
            c.add({"s": s, "a": 1, "type": "civilian", "belief": post,
                   "outcome": "cont", "s_next": s, "ticks": 10})          # hold
    return c, cell


def test_value_iteration_on_synthetic_model():
    post = {**{c: 0.0 for c in CLASSES}, "civilian": 1.0}
    c, cell = _synthetic_counts(post)
    mdp = EngagementMDP(c)
    mdp.solve()
    # in ID range: declare civilian (+3) beats everything
    assert ACTIONS[mdp.policy[state_index(1, cell)]] == "declare_civilian"
    assert mdp.V[state_index(1, cell)] == pytest.approx(DEFAULT_REWARDS["declare"]["civilian"]["civilian"], abs=0.2)
    # far out: the policy closes in, and is worth a little less (time cost)
    far = state_index(5, cell)
    assert ACTIONS[mdp.policy[far]] in ("approach", "declare_civilian")
    assert mdp.V[far] < mdp.V[state_index(1, cell)]
    # declarations are only immediate in the declarable bands; elsewhere they
    # need simulated data (none here), so they are unavailable
    R = mdp.immediate()
    assert np.isfinite(R[state_index(1, cell), ACTIONS.index("declare_enemy")])
    assert not np.isfinite(R[far, ACTIONS.index("declare_enemy")])


def test_declare_reward_uses_type_mix():
    post = {**{c: 0.0 for c in CLASSES}, "civilian": 1.0}
    c, cell = _synthetic_counts(post)
    mdp = EngagementMDP(c)
    R = mdp.immediate()
    s = state_index(1, cell)
    assert R[s, ACTIONS.index("declare_enemy")] < 0
    assert R[s, ACTIONS.index("declare_civilian")] == pytest.approx(3.0, abs=0.01)


def test_act_never_receives_the_world():
    params = inspect.signature(EngagementAgent.act).parameters
    assert "world" not in params and "obs" in params
    assert "world" not in inspect.signature(MDPPolicy.act).parameters


def test_engagement_is_reproducible():
    a = run_engagement(11, "small", {"kind": "threshold"})
    b = run_engagement(11, "small", {"kind": "threshold"})
    assert (a["outcome"], a["time_s"], a["score"]) == (b["outcome"], b["time_s"], b["score"])


def test_decisions_are_event_driven():
    """Decisions every 10 s AND on band changes: some fall between the 10-s marks."""
    gaps = []
    for seed in range(6):
        ep = run_engagement(seed, "small", {"kind": "explore"})
        t = [d[2] * 60 for d in ep["log"]]
        gaps += [round(t2 - t1) for t1, t2 in zip(t, t[1:])]
    assert gaps and max(gaps) <= 10 and min(gaps) < 10


class _AlwaysDeclare:
    name = "always"

    def __init__(self, what):
        self.what = what

    def choose(self, band, post, obs):
        return "declare_" + self.what


def _one_on_one(ctype_behavior, true_class, dist_ratio=3.0):
    W = 40.0
    u = Vec2(x=8.0, y=12.5)
    p = Vec2(x=8.0 + dist_ratio * identify_range(Grid(width=W, height=25)), y=12.5)
    return World(grid=Grid(width=W, height=25.0),
                 usvs=[USV(id="U1", position=u, start_pos=u, mode=USVMode.MDP)],
                 boats=[Boat(id="B1", position=p, start_pos=p, true_class=true_class,
                             behavior=ctype_behavior, standoff_speed=4.0)],
                 sensor_noise=1.0, mdp_assignment="nearest")


def test_committed_declaration_applies_only_in_range(monkeypatch):
    """A declaration made from far away is a committed close-in: no further
    decisions, and the engine applies it the first tick the boat is in range."""
    import reschu.engine as engine_mod
    from reschu.mdp.episodes import _EpisodePolicy
    world = _one_on_one(Behavior.CIVILIAN, Classification.CIVILIAN)
    log = []
    agent = EngagementAgent(_AlwaysDeclare("civilian"), on_decision=lambda *a: log.append(a))
    pol = _EpisodePolicy(agent)
    monkeypatch.setattr(engine_mod, "policy_for", lambda usv: pol)
    eng = SimulationEngine(world, dt_min=1 / 60, seed=3, assign_at_start=True)
    usv, boat = world.usvs[0], world.boats[0]
    for _ in range(300):
        eng.tick()
        if boat.identified:
            break
    assert boat.identified and boat.perceived_class.value == "civilian"
    assert usv.position.distance_to(boat.position) <= identify_range(world.grid) + 1e-9
    assert len(log) == 1                     # committed: decided once


def test_transitions_record_terminal_outcomes():
    ep = run_engagement(5, "small", {"kind": "rush"})
    recs = transitions(ep)
    assert recs and all(0 <= r["s"] < N_STATES for r in recs)
    last = recs[-1]
    if ep["outcome"] in ("declared", "dead", "left"):
        assert last["outcome"] == ep["outcome"] and last["ticks"] >= 1


def test_summary_fields():
    res = [run_engagement(s, "small", {"kind": "rush"}) for s in range(10)]
    sm = summarize(res)
    assert sm["n"] == 10 and 0 <= sm["correct"] <= 1 and set(sm["by_type"]) <= set(CLASSES)


# ---------------------------------------------------------------------- #
# shipped, solved policies
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("size", MAP_SIZES)
def test_shipped_policy_tables(size):
    t = PolicyTable.load(size)
    assert len(t.policy) == N_STATES and set(t.policy) <= set(ACTIONS)
    # declarations only where they are possible or as committed close-ins
    # (never "abandon" a contact believed to be an aggressor in ID range)
    sure_agg = {**{c: 0.0 for c in CLASSES}, "aggressive": 1.0}
    assert t.action(1, belief_cell(sure_agg)) == "declare_enemy"
    sure_civ = {**{c: 0.0 for c in CLASSES}, "civilian": 1.0}
    assert t.action(1, belief_cell(sure_civ)) == "declare_civilian"
    ev = t.meta.get("eval")
    if ev:   # the solved MDP beats both heuristics on its own objective
        assert ev["mdp"]["score"] >= ev["threshold"]["score"] - 0.1
        assert ev["mdp"]["score"] > ev["rush"]["score"]


def test_mdp_policy_runs_in_a_full_mission():
    """MDP USVs in a normal multi-boat mission: runs, declares, reads no truth."""
    from reschu.scenario import default_config, world_from_config
    cfg = default_config()
    for u in cfg["usvs"]:
        u["mode"] = "mdp"
    world = world_from_config(cfg)
    eng = SimulationEngine(world, dt_min=1 / 60, seed=1)
    assert isinstance(policy_for(world.usvs[0]), MDPPolicy)
    known = {b.id for b in world.boats if b.identified}
    for _ in range(60 * 6):
        eng.tick()
    declared = [b for b in world.boats if b.identified and b.id not in known]
    assert declared, "the MDP should have committed to at least one contact in 6 min"

