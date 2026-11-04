"""Tests for the MDP team assignment (reschu.mdp.assignment + MDPPolicy wiring)."""

from reschu.assignment import assign_all, reassign_free_agents, reassign_usv
from reschu.engine import SimulationEngine
from reschu.mdp.agent import EngagementAgent, TableChooser
from reschu.mdp.assignment import SWITCH_MARGIN, TeamAssigner, TeamMember
from reschu.models import USV, Boat, Grid, World
from reschu.observation import observe
from reschu.types import Behavior, Classification, Perception, USVMode, Vec2

DT = 1 / 60


def _world(assignment="value", usvs=((10, 12), (30, 12)), boats=((12, 12), (28, 12))):
    return World(
        grid=Grid(width=40, height=25),
        usvs=[USV(id=f"U{i + 1}", position=Vec2(x=x, y=y), mode=USVMode.MDP)
              for i, (x, y) in enumerate(usvs)],
        boats=[Boat(id=f"B{i + 1}", position=Vec2(x=x, y=y), start_pos=Vec2(x=x, y=y),
                    true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN)
               for i, (x, y) in enumerate(boats)],
        sensor_noise=1.0, mdp_assignment=assignment)


def test_engine_leaves_mdp_usvs_to_the_team_assignment():
    w = _world("value")
    assign_all(w)
    reassign_free_agents(w)
    assert all(u.assigned_target_id is None for u in w.usvs)
    w2 = _world("nearest")
    assign_all(w2)
    assert all(u.assigned_target_id is not None for u in w2.usvs)


def test_mission_start_assigns_each_usv_its_own_contact():
    w = _world("value")
    eng = SimulationEngine(w, dt_min=DT, seed=0)
    eng.tick()
    ids = [u.assigned_target_id for u in w.usvs]
    assert set(ids) == {"B1", "B2"}                 # exclusive, all worked
    # the obvious pairing: each USV takes the boat right next to it
    assert ids == ["B1", "B2"]


def _assigner_and_obs(w):
    agent = EngagementAgent(TableChooser())
    obs = observe(w.usvs[0], w, 0.0)
    agent.track(obs)
    return TeamAssigner(agent, agent.chooser), obs


def test_matching_is_exclusive_and_maximises_total_gain():
    w = _world(usvs=((5, 12), (35, 12), (20, 3)), boats=((7, 12), (33, 12)))
    ta, obs = _assigner_and_obs(w)
    members = [TeamMember(u.id, u.position, 12.0, None) for u in w.usvs]
    out = ta.assign(members, obs, held=set(), dt_min=DT)
    got = [b for b in out.values() if b]
    assert sorted(got) == ["B1", "B2"]              # each contact at most once
    assert out["U1"] == "B1" and out["U2"] == "B2"  # the nearby pairs, not the far one


def test_held_contacts_are_not_reassigned():
    w = _world()
    ta, obs = _assigner_and_obs(w)
    members = [TeamMember("U1", w.usvs[0].position, 12.0, None)]
    out = ta.assign(members, obs, held={"B1"}, dt_min=DT)
    assert out["U1"] == "B2"


def test_switch_margin_keeps_current_contact():
    """Two equally good contacts: a USV keeps the one it is already working."""
    w = _world(usvs=((20, 12),), boats=((14, 12), (26, 12)))
    ta, obs = _assigner_and_obs(w)
    m = TeamMember("U1", w.usvs[0].position, 12.0, "B2")
    g_cur = ta.gain(m, obs.boat("B2"), obs, DT)
    g_other = ta.gain(m, obs.boat("B1"), obs, DT)
    assert g_cur - g_other > SWITCH_MARGIN - 0.5
    assert ta.assign([m], obs, set(), DT)["U1"] == "B2"


def test_declared_civilians_are_not_worked():
    w = _world(usvs=((10, 12),), boats=((12, 12),))
    w.boats[0].identify(as_class=Perception.CIVILIAN)
    ta, obs = _assigner_and_obs(w)
    out = ta.assign([TeamMember("U1", w.usvs[0].position, 12.0, None)], obs, set(), DT)
    assert out["U1"] is None                        # nothing worth doing: idle


def test_operator_pick_is_respected():
    w = _world()
    eng = SimulationEngine(w, dt_min=DT, seed=0)
    eng.tick()
    assert w.usvs[0].assigned_target_id == "B1"
    reassign_usv(w, "U1")                           # operator double-click
    assert w.usvs[0].assignment_pinned
    picked = w.usvs[0].assigned_target_id
    for _ in range(40):                             # several re-matching rounds
        eng.tick()
    assert w.usvs[0].assigned_target_id in (picked, None)
    if w.usvs[0].assigned_target_id is None:        # only once that contact resolved
        b = w.boat(picked)
        assert b is None or b.identified or b.left_grid


def test_full_mission_both_rules_run():
    from reschu.mdp.missions import run_mission
    for rule in ("value", "nearest"):
        r = run_mission(500_001, rule, minutes=3)
        assert r["assignment"] == rule and 0 <= r["correct"] <= 1
