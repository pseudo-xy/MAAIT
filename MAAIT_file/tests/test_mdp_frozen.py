"""Freezing / versioning the MDP teammate, and its per-decision log."""

import csv
import json
import shutil

import pytest

import reschu.mdp.frozen as fz
from reschu.engine import SimulationEngine
from reschu.mdp.decision_log import FIELDS
from reschu.policies import MDPPolicy
from reschu.scenario import default_config, world_from_config


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A copy of the policy files with its own manifest (the real one is untouched)."""
    for f in fz.POLICY_FILES:
        shutil.copy(fz.MODEL_DIR / f, tmp_path / f)
    monkeypatch.setattr(fz, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(fz, "MANIFEST", tmp_path / "FROZEN.json")
    return tmp_path


def test_freeze_check_tamper_unfreeze(sandbox):
    assert fz.check()["frozen"] is False
    man = fz.freeze("test-v1", note="unit test")
    info = fz.check()
    assert info == {"version": "test-v1", "fingerprint": man["fingerprint"], "frozen": True}
    with pytest.raises(fz.FrozenPolicyError):
        fz.freeze("test-v2")                         # can't refreeze over a freeze
    # any change to a policy file is caught
    p = sandbox / "engagement_small.json"
    d = json.loads(p.read_text())
    d["policy"][0] = "abandon" if d["policy"][0] != "abandon" else "hold"
    p.write_text(json.dumps(d))
    with pytest.raises(fz.FrozenPolicyError, match="engagement_small.json changed"):
        fz.check()
    fz.unfreeze()
    assert (sandbox / "FROZEN-test-v1.json").exists()   # record of what ran is kept
    assert fz.check()["frozen"] is False


def test_settings_change_is_caught(sandbox, monkeypatch):
    fz.freeze("test-v1")
    import reschu.mdp.assignment as A
    monkeypatch.setattr(A, "SWITCH_MARGIN", A.SWITCH_MARGIN + 1)
    with pytest.raises(fz.FrozenPolicyError, match="settings changed"):
        fz.check()


def test_build_refuses_to_overwrite_a_frozen_policy(sandbox):
    fz.freeze("test-v1")
    with pytest.raises(fz.FrozenPolicyError):
        fz.guard_write(sandbox)
    fz.guard_write(sandbox / "elsewhere")            # other directories are fine


def test_shipped_policy_is_frozen_and_intact():
    info = fz.check()                                # raises if files were altered
    assert info["frozen"] and info["version"].startswith("mdp-v")


def _mdp_mission(seed=3, minutes=6):
    cfg = default_config()
    for u in cfg["usvs"]:
        u["mode"] = "mdp"
    world = world_from_config(cfg)
    pol = MDPPolicy()                                # private instance, private log
    import reschu.engine as engine_mod
    saved = engine_mod.policy_for
    engine_mod.policy_for = lambda usv: pol
    try:
        eng = SimulationEngine(world, dt_min=1 / 60, seed=seed)
        for _ in range(int(minutes * 60)):
            eng.tick()
    finally:
        engine_mod.policy_for = saved
    return world, pol


def test_decision_log_contents_and_truth_only_at_export(tmp_path):
    world, pol = _mdp_mission()
    raw = pol.log.rows()                             # as logged during the mission
    assert raw, "MDP made no logged decisions"
    assert all(r["true_type"] == "" and r["declaration_correct"] == "" for r in raw)
    decisions = [r for r in raw if r["event"] == "decision"]
    assigns = [r for r in raw if r["event"] == "assign"]
    assert decisions and assigns
    for r in decisions:
        assert r["policy_version"] == pol.version["version"]
        assert abs(sum(r[f"p_{c}"] for c in ("civilian", "friendly", "standoff_low",
                                             "standoff_med", "aggressive")) - 1) < 1e-3
        assert r["margin"] == "" or r["margin"] >= -1e-9   # the policy picks the best
        assert r["reason"]
    path = tmp_path / "dec.csv"
    pol.log.write_csv(path, world)
    rows = list(csv.DictReader(open(path)))
    assert list(rows[0].keys()) == FIELDS
    committed = [r for r in rows if r["event"] == "decision" and r["committed"] == "1"]
    assert all(r["true_type"] for r in rows if r["boat_id"])
    assert all(r["declaration_correct"] in ("0", "1") for r in committed)


def test_same_seed_same_decisions():
    _, a = _mdp_mission(seed=7, minutes=3)
    _, b = _mdp_mission(seed=7, minutes=3)
    assert a.log.rows() == b.log.rows()


def test_action_changed_marks_real_changes():
    _, pol = _mdp_mission(seed=3, minutes=6)
    last = {}
    for r in pol.log.rows():
        if r["event"] != "decision":
            continue
        key = (r["usv_id"], r["boat_id"])
        assert r["action_changed"] == int(last.get(key) != r["action"])
        last[key] = r["action"]
