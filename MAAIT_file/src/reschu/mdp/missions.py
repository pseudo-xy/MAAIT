"""Full missions (several MDP USVs, several contacts) — OFFLINE evaluation.

Compares the MDP team assignment ("value") with the engine's closest-unworked
-boat rule ("nearest"), with the same engagement MDP driving every USV and the
same seeds for both. Ground truth is read here only to SCORE, never by agents.

Mission score (same reward table as the engagement MDP):
  each contact the team declared       reward[declared class][true type]
  each contact left unresolved          abandon[true type]   (still undeclared at
                                        the end, or it left the map undeclared)
  each USV lost                         usv_lost
Contacts known at the start are not scored (no decision was needed).

    python -m reschu.mdp.missions --missions 300 --minutes 15
"""

from __future__ import annotations

import argparse
import math
import random

import numpy as np

from ..engine import SimulationEngine
from ..models import USV, Boat, Grid, World
from ..scenario import DEFAULT_SENSOR_NOISE, DEFAULT_SPEED_JITTER, DEFAULT_SPEED_SPREAD
from ..sensing import contact_type
from ..types import USVMode, Vec2
from .cue_model import CLASSES, PERCEPTION_OF
from .engagement import DEFAULT_REWARDS, MAP_SIZES, MODEL_DIR
from .episodes import _SPEC, MAP_DIMS

DT_MIN = 1.0 / 60.0
RULES = ("value", "rate", "nearest")


def mission_world(seed: int, assignment: str, size: str | None = None,
                  n_usvs: int = 3, n_boats: tuple[int, int] = (6, 9)) -> World:
    rng = random.Random(seed)
    size = size or rng.choice(MAP_SIZES)
    W, H = MAP_DIMS[size]
    m = 0.06 * min(W, H)

    def pt():
        return Vec2(x=rng.uniform(m, W - m), y=rng.uniform(m, H - m))

    usvs = []
    for i in range(n_usvs):
        p = pt()
        usvs.append(USV(id=f"U{i + 1}", position=p, start_pos=p, mode=USVMode.MDP))
    boats = []
    for i in range(rng.randint(*n_boats)):
        t = rng.choice(CLASSES)
        tc, beh, so = _SPEC[t]
        p = pt()
        boats.append(Boat(id=f"B{i + 1}", position=p, start_pos=p, true_class=tc,
                          behavior=beh, standoff_speed=so))
    return World(grid=Grid(width=W, height=H), usvs=usvs, boats=boats,
                 speed_spread=DEFAULT_SPEED_SPREAD, speed_jitter=DEFAULT_SPEED_JITTER,
                 sensor_noise=DEFAULT_SENSOR_NOISE, visual_cue=True,
                 mdp_assignment=assignment)


def score_mission(world: World, start_usvs: int, rewards: dict | None = None) -> dict:
    rw = rewards or DEFAULT_REWARDS
    score, correct, wrong, unresolved, times = 0.0, 0, 0, 0, []
    aggr_total = aggr_done = misfire = 0
    for b in world.boats:
        t = contact_type(b)
        aggr_total += t == "aggressive"
        if b.identified and b.first_identified_min is not None:
            d = b.perceived_class.value
            score += rw["declare"][d][t]
            ok = d == PERCEPTION_OF[t].value
            correct += ok
            wrong += not ok
            misfire += d == "enemy" and t in ("friendly", "civilian")
            aggr_done += ok and t == "aggressive"
            times.append(b.first_identified_min)
        else:
            score += rw["abandon"][t]
            unresolved += 1
    lost = start_usvs - sum(1 for u in world.usvs if not u.eliminated)
    score += lost * rw["usv_lost"]
    n = len(world.boats)
    return {"score": score, "per_contact": score / n, "contacts": n, "correct": correct / n, "wrong": wrong / n,
            "unresolved": unresolved / n, "misfire": misfire, "lost": lost,
            "aggr_neutralised": aggr_done / max(aggr_total, 1),
            "mean_id_min": float(np.mean(times)) if times else float("nan")}


def set_assignment_params(params: dict | None) -> None:
    """Override the team-assignment settings (in this process) for tuning."""
    from . import assignment as A
    for k, v in (params or {}).items():
        if not hasattr(A, k):
            raise KeyError(k)
        setattr(A, k, v)


def run_mission(seed: int, assignment: str, minutes: float = 15.0,
                size: str | None = None, params: dict | None = None) -> dict:
    from . import assignment as A
    saved = {k: getattr(A, k) for k in (params or {})}
    set_assignment_params(params)
    try:
        world = mission_world(seed, assignment, size)
        eng = SimulationEngine(world, dt_min=DT_MIN, seed=seed, assign_at_start=True)
        for _ in range(int(round(minutes / DT_MIN))):
            eng.tick()
    finally:
        set_assignment_params(saved)
    out = score_mission(world, len(world.usvs))
    out.update(seed=seed, assignment=assignment, size=size_of(world), params=params or {})
    return out


def size_of(world: World) -> str:
    return {40.0: "small", 80.0: "medium", 120.0: "large"}.get(world.grid.width, "?")


def _one(args):
    return run_mission(*args)


def compare(seeds, minutes: float = 15.0, processes: int | None = None) -> dict:
    from multiprocessing import Pool
    args = [(s, a, minutes) for s in seeds for a in RULES]
    with Pool(processes) as pool:
        res = pool.map(_one, args, chunksize=4)
    out = {a: [r for r in res if r["assignment"] == a] for a in RULES}
    return out


def summary_md(res: dict, minutes: float) -> str:
    names = {"value": "value (MDP team, total gain)", "rate": "rate (MDP team, gain per minute)",
             "nearest": "nearest (engine, closest boat)"}
    rules = [r for r in RULES if res.get(r)]
    n = len(res[rules[0]])
    L = ["# MDP team assignment — mission evaluation", "",
         f"{n} random missions ({minutes:g} min, 3 MDP USVs, 6-9 contacts of random types, "
         "random map size), each run under every assignment rule on the same seed. The same "
         "engagement MDP drives every USV; only the choice of contact differs.", "",
         "| Assignment | Mission score (95% CI) | Score per contact | Correct | Wrong | Unresolved "
         "| Friendly/civilian fired on | Aggressors neutralised | USVs lost | Mean time to ID |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rules:
        rs = res[r]
        sc = np.array([x["score"] for x in rs])
        f = lambda k: np.mean([x[k] for x in rs])
        L.append(f"| {names[r]} | {sc.mean():.2f} ± {1.96 * sc.std(ddof=1) / math.sqrt(len(sc)):.2f} | "
                 f"{f('per_contact'):.2f} | {f('correct'):.1%} | {f('wrong'):.1%} | "
                 f"{f('unresolved'):.1%} | {f('misfire'):.2f} / mission | "
                 f"{f('aggr_neutralised'):.1%} | {f('lost'):.2f} / mission | "
                 f"{np.nanmean([x['mean_id_min'] for x in rs]):.1f} min |")
    L += ["", "Paired differences against the engine's nearest-boat rule (same seeds):", ""]
    base = {x["seed"]: x["score"] for x in res["nearest"]}
    for r in rules:
        if r == "nearest":
            continue
        d = np.array([x["score"] - base[x["seed"]] for x in res[r]])
        L.append(f"- {r} − nearest: {d.mean():+.2f} ± {1.96 * d.std(ddof=1) / math.sqrt(len(d)):.2f} "
                 f"per mission (better in {np.mean(d > 0):.0%}, worse in {np.mean(d < 0):.0%})")
    L += ["", "By map size (mean mission score, " + " / ".join(rules) + "):", ""]
    for size in MAP_SIZES:
        vals = [np.mean([x["score"] for x in res[r] if x["size"] == size] or [np.nan]) for r in rules]
        cnt = sum(1 for x in res[rules[0]] if x["size"] == size)
        if cnt:
            L.append(f"- {size}: " + " / ".join(f"{v:.2f}" for v in vals) + f" ({cnt} missions)")
    return "\n".join(L) + "\n"


TUNE_SEEDS = 700_000          # tuning missions (never used for reporting)
CONFIRM_SEEDS = 800_000       # fresh missions for the confirmation run
TUNE_GRID = {"SWITCH_MARGIN": (0.0, 1.0, 2.0, 4.0), "RATE_FLOOR_MIN": (0.25, 0.5, 1.0, 2.0)}


def tune(n_missions: int, minutes: float, processes=None) -> tuple[dict, str]:
    """Grid-search the 'rate' assignment settings on tuning missions."""
    from itertools import product
    from multiprocessing import Pool
    seeds = range(TUNE_SEEDS, TUNE_SEEDS + n_missions)
    grid = [dict(zip(TUNE_GRID, v)) for v in product(*TUNE_GRID.values())]
    args = [(s, "rate", minutes, None, p) for p in grid for s in seeds]
    with Pool(processes) as pool:
        res = pool.map(_one, args, chunksize=4)
    rows = []
    for p in grid:
        sc = np.array([r["score"] for r in res if r["params"] == p])
        rows.append((float(sc.mean()), 1.96 * sc.std(ddof=1) / math.sqrt(len(sc)), p))
    rows.sort(key=lambda r: -r[0])
    L = [f"Tuning on {n_missions} separate missions (gain per minute rule):", "",
         "| Switching bonus | Time floor (min) | Mission score |", "|---|---|---|"]
    for m, ci, p in rows:
        L.append(f"| {p['SWITCH_MARGIN']:g} | {p['RATE_FLOOR_MIN']:g} | {m:.2f} ± {ci:.2f} |")
    return rows[0][2], "\n".join(L) + "\n"


def confirm(params: dict, n_missions: int, minutes: float, processes=None) -> str:
    from multiprocessing import Pool
    seeds = range(CONFIRM_SEEDS, CONFIRM_SEEDS + n_missions)
    args = [(s, "rate", minutes, None, params) for s in seeds] + \
           [(s, "nearest", minutes, None, None) for s in seeds]
    with Pool(processes) as pool:
        res = pool.map(_one, args, chunksize=4)
    out = {"rate": [r for r in res if r["assignment"] == "rate"],
           "nearest": [r for r in res if r["assignment"] == "nearest"]}
    base = {r["seed"]: r for r in out["nearest"]}
    L = [f"Confirmation on {n_missions} FRESH missions, tuned settings {params}:", "",
         "| Rule | Mission score | Correct | Unresolved | Friendly/civilian fired on | "
         "Aggressors neutralised | USVs lost |", "|---|---|---|---|---|---|---|"]
    for name, rs in out.items():
        sc = np.array([r["score"] for r in rs])
        f = lambda k: np.mean([r[k] for r in rs])
        L.append(f"| {name} | {sc.mean():.2f} ± {1.96 * sc.std(ddof=1) / math.sqrt(len(sc)):.2f} | "
                 f"{f('correct'):.1%} | {f('unresolved'):.1%} | {f('misfire'):.2f} | "
                 f"{f('aggr_neutralised'):.1%} | {f('lost'):.2f} |")
    L.append("")
    for k, lab in (("score", "Mission score"), ("correct", "Correct share"),
                   ("unresolved", "Unresolved share"), ("misfire", "Fired on per mission")):
        d = np.array([r[k] - base[r["seed"]][k] for r in out["rate"]])
        se = d.std(ddof=1) / math.sqrt(len(d))
        L.append(f"- {lab}, rate − nearest: {d.mean():+.3f} ± {1.96 * se:.3f} "
                 f"(z = {d.mean() / se if se else float('nan'):.1f})")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--missions", type=int, default=300)
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--tune", type=int, default=0, help="tune on N missions, then confirm")
    a = ap.parse_args(argv)
    if a.tune:
        best, tmd = tune(a.tune, a.minutes, a.processes)
        print(tmd, flush=True)
        cmd = confirm(best, a.missions, a.minutes, a.processes)
        md = "# MDP team assignment — tuning and confirmation\n\n" + tmd + "\n" + cmd
        (MODEL_DIR / "assignment_tuning_report.md").write_text(md)
        print(cmd)
        return
    res = compare(range(500_000, 500_000 + a.missions), a.minutes, a.processes)
    md = summary_md(res, a.minutes)
    (MODEL_DIR / "assignment_report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
