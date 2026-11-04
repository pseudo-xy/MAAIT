"""OFFLINE data collection for fitting the MDP's cue likelihoods.

Runs randomized missions and records, for every not-yet-identified contact, the
same truth-free window features the MDP will compute at runtime (features.py),
LABELLED with the contact's true type. Ground truth is used here only to label
training data — never by the MDP at decision time.

Two engine mechanics are switched off in THIS tool only, so that complete,
natural boat behavior is observed:
  * identification — every boat stays an "undecided contact" (the regime the MDP
    actually has to decide in);
  * USV elimination — an aggressor's full lock-on approach is captured instead of
    the run ending at first contact.
USVs follow a scripted INVESTIGATOR pattern (approach a random contact, hold at a
random standoff of 0.3-2.5 ID ranges for 20-90 s, move on), which covers the
near / mid / far bands the MDP will operate in.
"""

from __future__ import annotations

import random
from dataclasses import asdict

from .. import engine as engine_mod
from ..engine import SimulationEngine
from ..models import Boat, World
from ..observation import observe
from ..scenario import world_from_config
from ..sensing import contact_type
from ..types import Behavior, Vec2
from ..usv_movement import move_towards
from .features import ContactTracker, id_range_for_width

# contact types the MDP must tell apart (labels for training only)
CLASSES = ("civilian", "friendly", "standoff_low", "standoff_med", "aggressive")
_SPEC = {  # config entry per type
    "civilian":     {"true_class": "civilian", "behavior": "civilian"},
    "friendly":     {"true_class": "friendly", "behavior": "friendly"},
    "standoff_low": {"true_class": "enemy", "behavior": "standoff", "tier": "low"},
    "standoff_med": {"true_class": "enemy", "behavior": "standoff", "tier": "med"},
    "aggressive":   {"true_class": "enemy", "behavior": "aggressive"},
}
MAP_WIDTHS = {"small": (40, 25), "medium": (80, 50), "large": (120, 75)}
DT_MIN = 1.0 / 60.0        # runtime tick: 1 simulated second
_ROWS = "ABCDEFGHIJKLMNOPQRSTUVWXY"


def true_type(b: Boat) -> str:
    """OFFLINE label from ground truth (training data only)."""
    return contact_type(b)


def _cell(rng: random.Random, edge_pad: int = 2) -> str:
    return f"{_ROWS[rng.randint(edge_pad, 24 - edge_pad)]}{rng.randint(1 + edge_pad, 40 - edge_pad)}"


def episode_config(rng: random.Random, n_per_type: int = 2, n_usvs: int = 4) -> dict:
    size = rng.choice(list(MAP_WIDTHS))
    w, h = MAP_WIDTHS[size]
    types = [t for t in CLASSES for _ in range(n_per_type)]
    rng.shuffle(types)
    return {
        "map_size": size,
        "grid": {"width": float(w), "height": float(h)},
        "mission_minutes": 60.0,
        "placement_seed": rng.randrange(10**9),
        "usvs": [{"mode": "mdp", "start": _cell(rng)} for _ in range(n_usvs)],
        "boats": [dict(_SPEC[t], start=_cell(rng)) for t in types],
    }


class _Investigator:
    """Scripted information-gathering USV behavior (collection only)."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.state: dict[str, dict] = {}

    def decide(self, usv, world: World, dt_min: float, now_min: float) -> Vec2:
        idr = id_range_for_width(world.grid.width)
        st = self.state.get(usv.id)
        live = [b for b in world.boats if not b.left_grid]
        tgt = world.boat(st["target"]) if st else None
        expired = st and st["dwell_start"] is not None and now_min - st["dwell_start"] >= st["dwell"]
        if st is None or tgt is None or tgt.left_grid or expired:
            if not live:
                return usv.position
            tgt = self.rng.choice(live)
            st = {"target": tgt.id,
                  "standoff": self.rng.uniform(0.3, 2.5) * idr,
                  "dwell": self.rng.uniform(20, 90) / 60.0,   # minutes
                  "dwell_start": None}
            self.state[usv.id] = st
        if usv.position.distance_to(tgt.position) > st["standoff"]:
            return move_towards(usv, tgt.position, world, dt_min)
        if st["dwell_start"] is None:
            st["dwell_start"] = now_min
        return usv.position


class _CollectionEngine(SimulationEngine):
    def _eliminate_usvs_in_danger(self, w: World) -> None:   # collection only
        return None


def run_episode(seed: int, minutes: float = 10.0) -> list[dict]:
    """One labelled mission. Returns one row per completed window."""
    rng = random.Random(seed)
    cfg = episode_config(rng)
    world = world_from_config(cfg)
    inv = _Investigator(random.Random(seed + 1))

    # identification off + investigator steering (restored afterwards)
    saved = (engine_mod.policy_for, engine_mod.apply_declarations, engine_mod.auto_identify)
    engine_mod.policy_for = lambda usv: inv
    engine_mod.apply_declarations = lambda w: []
    engine_mod.auto_identify = lambda w, rng=None: []
    try:
        eng = _CollectionEngine(world, dt_min=DT_MIN, seed=seed, assign_at_start=False)
        tracker = ContactTracker()
        rows: list[dict] = []
        for _ in range(int(round(minutes / DT_MIN))):
            eng.tick()
            obs = observe(world.usvs[0], world, world.mission_time_min)
            for wf in tracker.update(obs):
                boat = world.boat(wf.boat_id)
                rows.append(dict(asdict(wf), label=true_type(boat),
                                 map_size=cfg["map_size"], episode=seed))
        return rows
    finally:
        engine_mod.policy_for, engine_mod.apply_declarations, engine_mod.auto_identify = saved


def collect(seeds, minutes: float = 10.0, processes: int | None = None) -> list[dict]:
    """Run many episodes (in parallel) and concatenate their rows."""
    from multiprocessing import Pool
    seeds = list(seeds)
    if processes == 1:
        out = [run_episode(s, minutes) for s in seeds]
    else:
        with Pool(processes) as pool:
            out = pool.starmap(run_episode, [(s, minutes) for s in seeds])
    return [r for rows in out for r in rows]
