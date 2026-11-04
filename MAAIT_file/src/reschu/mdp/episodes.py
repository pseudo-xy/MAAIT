"""Single engagements in the REAL simulator: one USV vs one contact.

Used OFFLINE twice:
  * to estimate the engagement MDP's transitions (`collect`) — the true type is
    read here only to label the data;
  * to evaluate policies (`evaluate`) — the true type is read here only to score.
The agent itself (agent.py) never sees the World.

An episode: a random contact type (uniform over the five), a USV assigned to it at
a random starting distance of 1-5 ID ranges, default sensing and speed
variation. It ends when the USV declares, abandons, is destroyed, the contact
leaves the map, or time runs out.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np

from .. import engine as engine_mod
from ..engine import SimulationEngine
from ..models import USV, Boat, Grid, World
from ..policies import MDPPolicy
from ..scenario import DEFAULT_SENSOR_NOISE, DEFAULT_SPEED_JITTER, DEFAULT_SPEED_SPREAD
from ..sensing import contact_type
from ..types import Behavior, Classification, USVMode, Vec2
from .agent import (EngagementAgent, ExploreChooser, RushChooser, TableChooser,
                    ThresholdChooser)
from .cue_model import CLASSES
from .agent import SIM_ACTIONS
from .engagement import (DECISION_TICKS, DECLARE_ACTIONS, DEFAULT_REWARDS, Counts, PolicyTable,
                         belief_cell, expected_terminal, state_index)
from .features import id_range_for_width

MAP_DIMS = {"small": (40.0, 25.0), "medium": (80.0, 50.0), "large": (120.0, 75.0)}
DT_MIN = 1.0 / 60.0
_SPEC = {
    "civilian":     (Classification.CIVILIAN, Behavior.CIVILIAN, 4.0),
    "friendly":     (Classification.FRIENDLY, Behavior.FRIENDLY, 4.0),
    "standoff_low": (Classification.ENEMY, Behavior.STANDOFF, 4.0),
    "standoff_med": (Classification.ENEMY, Behavior.STANDOFF, 9.0),
    "aggressive":   (Classification.ENEMY, Behavior.AGGRESSIVE, 4.0),
}


def engagement_world(rng: random.Random, size: str, ctype: str | None = None,
                     r_range=(1.0, 5.0)) -> tuple[World, str]:
    W, H = MAP_DIMS[size]
    idr = id_range_for_width(W)
    ctype = ctype or rng.choice(CLASSES)
    margin = 0.06 * min(W, H)
    for _ in range(1000):
        u = Vec2(x=rng.uniform(margin, W - margin), y=rng.uniform(margin, H - margin))
        r, ang = rng.uniform(*r_range) * idr, rng.uniform(0, 2 * math.pi)
        p = Vec2(x=u.x + r * math.cos(ang), y=u.y + r * math.sin(ang))
        if margin <= p.x <= W - margin and margin <= p.y <= H - margin:
            break
    tc, beh, so = _SPEC[ctype]
    world = World(
        grid=Grid(width=W, height=H),
        usvs=[USV(id="U1", position=u, start_pos=u, mode=USVMode.MDP)],
        boats=[Boat(id="B1", position=p, start_pos=p, true_class=tc, behavior=beh,
                    standoff_speed=so)],
        speed_spread=DEFAULT_SPEED_SPREAD, speed_jitter=DEFAULT_SPEED_JITTER,
        sensor_noise=DEFAULT_SENSOR_NOISE, visual_cue=True,
        mdp_assignment="nearest")   # one contact: the engine's link is the engagement
    return world, ctype


class _EpisodePolicy(MDPPolicy):
    """MDPPolicy's exact control flow, with an agent of our choosing."""

    def __init__(self, agent: EngagementAgent) -> None:
        super().__init__()
        self._agent = agent


def make_chooser(spec: dict, rng: random.Random):
    kind = spec["kind"]
    if kind == "mdp":
        return TableChooser({spec["size"]: PolicyTable(spec["size"], spec["policy"])}
                            if spec.get("policy") else None)
    if kind == "rush":
        return RushChooser()
    if kind == "threshold":
        return ThresholdChooser(spec.get("threshold", 0.9))
    if kind == "explore":
        base = make_chooser(spec["base"], rng) if spec.get("base") else None
        return ExploreChooser(rng, eps=spec.get("eps", 1.0), base=base)
    raise ValueError(kind)


def run_engagement(seed: int, size: str, spec: dict, ctype: str | None = None,
                   max_decisions: int = 30, rewards: dict | None = None) -> dict:
    """Run one engagement. Returns the outcome, its score, and the decision log."""
    rewards = rewards or DEFAULT_REWARDS
    rng = random.Random(seed)
    world, ctype = engagement_world(rng, size, ctype)
    log: list[tuple] = []
    agent = EngagementAgent(make_chooser(spec, random.Random(seed + 7)),
                            on_decision=lambda *a: log.append(a))
    pol = _EpisodePolicy(agent)
    saved = engine_mod.policy_for
    engine_mod.policy_for = lambda usv: pol
    try:
        eng = SimulationEngine(world, dt_min=DT_MIN, seed=seed, assign_at_start=True)
        usv, boat = world.usvs[0], world.boats[0]
        outcome, declared = "timeout", None
        for _ in range(max_decisions * DECISION_TICKS):
            eng.tick()
            if usv.eliminated:
                outcome = "dead"
                break
            if boat.identified:
                outcome, declared = "declared", boat.perceived_class.value
                break
            if boat.left_grid:
                outcome = "left"
                break
            if log and log[-1][-1] == "abandon":
                outcome = "abandon"
                break
    finally:
        engine_mod.policy_for = saved

    # time is charged per elapsed second until the engagement resolves
    elapsed_s = world.mission_time_min * 60.0
    score = rewards["step"] * elapsed_s / DECISION_TICKS + expected_terminal(
        rewards, ctype, outcome, declared)
    return {"seed": seed, "size": size, "type": ctype, "outcome": outcome,
            "declared": declared, "decisions": len(log), "score": score,
            "time_s": round(elapsed_s), "log": log}


def transitions(ep: dict) -> list[dict]:
    """Turn an episode's decision log into MDP transition records."""
    recs = []
    log = ep["log"]
    end_min = ep["time_s"] / 60.0
    for i, (_, _, t, band, post, action) in enumerate(log):
        s = state_index(band, belief_cell(post))
        rec = {"s": s, "type": ep["type"], "belief": np.array([post[c] for c in CLASSES]),
               "a": None, "outcome": None, "s_next": None, "ticks": 0}
        if action in SIM_ACTIONS:
            a = SIM_ACTIONS.index(action)
            if i + 1 < len(log):
                nb, npost, t2 = log[i + 1][3], log[i + 1][4], log[i + 1][2]
                rec.update(a=a, outcome="cont", s_next=state_index(nb, belief_cell(npost)),
                           ticks=round((t2 - t) * 60))
            elif ep["outcome"] in ("dead", "left", "declared") or (
                    action in DECLARE_ACTIONS and ep["outcome"] == "timeout"):
                # a commitment that never resolved counts as unresolved ("left")
                out = "left" if ep["outcome"] == "timeout" else ep["outcome"]
                rec.update(a=a, outcome=out, ticks=max(1, round((end_min - t) * 60)))
        recs.append(rec)
    return recs


def _collect_one(seed: int, size: str, spec: dict, max_decisions: int) -> list[dict]:
    return transitions(run_engagement(seed, size, spec, max_decisions=max_decisions))


def collect(seeds, size: str, spec: dict, max_decisions: int = 30,
            processes: int | None = None) -> Counts:
    """Run engagements (in parallel) and accumulate their transition counts."""
    args = [(s, size, spec, max_decisions) for s in seeds]
    total = Counts()
    if processes == 1:
        for a in args:
            for r in _collect_one(*a):
                total.add(r)
        return total
    from multiprocessing import Pool
    with Pool(processes) as pool:
        for recs in pool.starmap(_collect_one, args, chunksize=50):
            for r in recs:
                total.add(r)
    return total


def _eval_one(seed, size, spec, max_decisions, rewards):
    ep = run_engagement(seed, size, spec, max_decisions=max_decisions, rewards=rewards)
    ep.pop("log")
    return ep


def evaluate(seeds, size: str, spec: dict, max_decisions: int = 60,
             rewards: dict | None = None, processes: int | None = None) -> list[dict]:
    args = [(s, size, spec, max_decisions, rewards) for s in seeds]
    if processes == 1:
        return [_eval_one(*a) for a in args]
    from multiprocessing import Pool
    with Pool(processes) as pool:
        return pool.starmap(_eval_one, args, chunksize=25)


def summarize(results: list[dict]) -> dict:
    """Mean score (with 95% CI), correct-declaration rate, losses, time."""
    n = len(results)
    sc = np.array([r["score"] for r in results])
    from .cue_model import PERCEPTION_OF
    decl = [r for r in results if r["outcome"] == "declared"]
    correct = sum(1 for r in decl if r["declared"] == PERCEPTION_OF[r["type"]].value)
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    out = {
        "n": n, "score": float(sc.mean()), "score_ci": float(1.96 * sc.std(ddof=1) / math.sqrt(n)),
        "declared": len(decl) / n, "correct_of_declared": correct / max(len(decl), 1),
        "correct": correct / n,
        "lost": sum(r["outcome"] == "dead" for r in results) / n,
        "unresolved": sum(r["outcome"] in ("abandon", "left", "timeout") for r in results) / n,
        "time_s": float(np.mean([r["time_s"] for r in results])),
        "by_type": {},
    }
    for t, rs in by_type.items():
        d = [r for r in rs if r["outcome"] == "declared"]
        out["by_type"][t] = {
            "n": len(rs), "score": float(np.mean([r["score"] for r in rs])),
            "correct": sum(1 for r in d if r["declared"] == PERCEPTION_OF[t].value) / len(rs),
            "lost": sum(r["outcome"] == "dead" for r in rs) / len(rs),
            "unresolved": sum(r["outcome"] in ("abandon", "left", "timeout") for r in rs) / len(rs),
            "time_s": float(np.mean([r["time_s"] for r in rs])),
        }
    return out
