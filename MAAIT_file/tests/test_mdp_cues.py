"""Tests for the MDP cue pipeline: truth-free features + cue model + belief."""

import math

import pytest

from reschu.models import USV, Boat, Grid, World
from reschu.observation import observe
from reschu.types import Behavior, Classification, Perception, USVMode, Vec2
from reschu.mdp.features import ContactTracker, WINDOW_TICKS, band_for, id_range_for_width
from reschu.mdp.cue_model import CLASSES, CueModel, PERCEPTION_OF, sequences


def _world(boat_pos=Vec2(x=20, y=10), heading=Vec2(x=-1, y=0), team=()):
    usvs = [USV(id="M", position=Vec2(x=10, y=10), mode=USVMode.MDP)]
    usvs += [USV(id=f"T{i}", position=p, mode=USVMode.MDP) for i, p in enumerate(team)]
    b = Boat(id="B", position=boat_pos, heading=heading,
             true_class=Classification.CIVILIAN, behavior=Behavior.CIVILIAN)
    return World(grid=Grid(width=40, height=25), usvs=usvs, boats=[b])


def _feed(tracker, w, n, step=Vec2(x=-0.05, y=0)):
    out = []
    for i in range(n):
        w.mission_time_min = (i + 1) / 60
        out += tracker.update(observe(w.usvs[0], w, w.mission_time_min))
        b = w.boats[0]
        b.position = Vec2(x=b.position.x + step.x, y=b.position.y + step.y)
    return out


def test_window_emitted_every_window_ticks_and_truth_free():
    w = _world()
    tr = ContactTracker()
    wins = _feed(tr, w, 3 * WINDOW_TICKS)
    assert len(wins) == 3
    for f in ("true_class", "behavior", "standoff_speed", "label"):
        assert not hasattr(wins[0], f)


def test_update_is_idempotent_per_timestamp():
    w = _world()
    tr = ContactTracker()
    obs = observe(w.usvs[0], w, 0.5)
    for _ in range(WINDOW_TICKS * 2):
        tr.update(obs)                      # same timestamp -> counted once
    assert tr.windows("B") == []
    assert tr.mean_speed("B") == pytest.approx(3.0)


def test_straight_approach_reads_as_closing_and_straight():
    w = _world(heading=Vec2(x=-1, y=0))     # heading straight at the USV
    tr = ContactTracker()
    wf = _feed(tr, w, WINDOW_TICKS)[0]
    assert wf.closing == pytest.approx(1.0)
    assert wf.turn == pytest.approx(0.0)
    assert wf.straightness == pytest.approx(1.0)


def test_closing_uses_nearest_team_usv():
    # boat heads +x; the observer is behind it, but a teammate is ahead of it
    w = _world(boat_pos=Vec2(x=20, y=10), heading=Vec2(x=1, y=0), team=(Vec2(x=24, y=10),))
    tr = ContactTracker()
    wf = _feed(tr, w, WINDOW_TICKS, step=Vec2(x=0.0, y=0))[0]
    assert wf.closing == pytest.approx(1.0)  # toward the teammate, not away from "me"


def test_identified_contacts_are_not_tracked():
    w = _world()
    w.boats[0].identify()
    tr = ContactTracker()
    assert _feed(tr, w, WINDOW_TICKS) == []


def test_bands_scale_with_map():
    assert id_range_for_width(40) == 2.0 and id_range_for_width(120) == 6.0
    assert (band_for(0.9), band_for(1.5), band_for(3.0)) == ("near", "mid", "far")


def _synthetic_rows():
    """Two clearly different types, a few windows each."""
    rows = []
    for ep in range(30):
        for t in range(4):
            rows.append(dict(episode=ep, boat_id="F", t_end=t, label="friendly", band="far",
                             speed=8.0, closing=0.0, turn=40.0, straightness=0.8))
            rows.append(dict(episode=ep, boat_id="A", t_end=t, label="aggressive", band="near",
                             speed=10.2, closing=0.9, turn=1.0, straightness=0.99))
    return rows


def test_posterior_is_normalized_and_learns():
    m = CueModel.fit(_synthetic_rows())
    friendly_win = dict(band="far", closing=0.0, turn=40.0, straightness=0.8)
    p1 = m.posterior(8.0, [friendly_win])
    assert sum(p1.values()) == pytest.approx(1.0)
    assert max(p1, key=p1.get) == "friendly"
    p3 = m.posterior(8.0, [friendly_win] * 3)
    assert p3["friendly"] > p1["friendly"]          # more evidence -> more confident


def test_declaration_probs_collapse_enemy_types():
    post = {c: 0.2 for c in CLASSES}
    dp = CueModel.declaration_probs(post)
    assert dp[Perception.ENEMY] == pytest.approx(0.6)
    assert sum(dp.values()) == pytest.approx(1.0)
    assert PERCEPTION_OF["standoff_low"] is Perception.ENEMY


def test_save_load_roundtrip(tmp_path):
    m = CueModel.fit(_synthetic_rows())
    m.tau_speed, m.tau_beh = 0.3, 0.4
    p = tmp_path / "m.json"
    m.save(p)
    m2 = CueModel.load(p)
    win = dict(band="near", closing=0.9, turn=1.0, straightness=0.99)
    assert m2.posterior(10.2, [win]) == pytest.approx(m.posterior(10.2, [win]))


def test_sequences_group_and_order():
    rows = [dict(episode=1, boat_id="B", t_end=2, label="civilian"),
            dict(episode=1, boat_id="B", t_end=1, label="civilian")]
    (label, seq), = sequences(rows)
    assert label == "civilian" and [r["t_end"] for r in seq] == [1, 2]


def test_shipped_cue_model_loads_and_is_sensible():
    from reschu.mdp.cue_model import DEFAULT_PATH
    m = CueModel.load(DEFAULT_PATH)
    # a slow, wandering contact far away -> should not look like an aggressor
    slow = dict(band="far", closing=0.0, turn=15.0, straightness=0.6)
    p = m.posterior(3.0, [slow] * 3)
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["aggressive"] < 0.05
    # a fast contact running straight at us up close -> enemy
    attack = dict(band="near", closing=0.95, turn=1.0, straightness=0.99)
    d = CueModel.declaration_probs(m.posterior(10.0, [attack] * 3))
    assert d[Perception.ENEMY] > 0.9
