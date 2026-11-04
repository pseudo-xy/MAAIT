"""Reward sensitivity of the engagement MDP — how its BEHAVIOR changes when the
key reward values are halved or doubled.

    python -m reschu.mdp.sensitivity [--sizes small medium large] [--eval 2000]

For each map size the simulator transitions are collected ONCE (same seeds and
procedure as build.py, so the counts equal those behind the shipped tables);
each reward variant is then re-solved on those counts and evaluated in the real
simulator on fresh engagements. Rewards do not change the physics, so re-using
the counts is valid; the one approximation is that the refinement episodes
followed the default-reward policy.

Reported per variant: correct declarations, friendlies/civilians declared enemy,
aggressors let through, USVs lost, unresolved, mean engagement time, and the
score under the DEFAULT rewards (so variants are compared on one yardstick).
Writes mdp/sensitivity_report.md.
"""

from __future__ import annotations

import argparse
import copy
import math

import numpy as np

from .build import SEED_BASE, SIZE_OFFSET
from .engagement import ACTIONS, DEFAULT_REWARDS, MAP_SIZES, MODEL_DIR, EngagementMDP
from .episodes import collect, evaluate


def _scale(rw: dict, path: tuple, k: float) -> dict:
    """Copy of rw with the entries at `path` multiplied by k."""
    rw = copy.deepcopy(rw)
    *head, last = path
    d = rw
    for h in head:
        d = d[h]
    if isinstance(d[last], dict):
        d[last] = {t: v * k for t, v in d[last].items()}
    else:
        d[last] = d[last] * k
    return rw


def _misfire(rw, k):
    rw = copy.deepcopy(rw)
    for t in ("friendly", "civilian"):
        rw["declare"]["enemy"][t] *= k
    return rw


def _let_through(rw, k):
    rw = copy.deepcopy(rw)
    for d in ("friendly", "civilian"):
        for t in ("aggressive", "standoff_low", "standoff_med"):
            rw["declare"][d][t] *= k
    return rw


FACTORS = {
    "time cost (step)": lambda rw, k: _scale(rw, ("step",), k),
    "USV lost": lambda rw, k: _scale(rw, ("usv_lost",), k),
    "firing on friendly/civilian": _misfire,
    "letting an enemy through": _let_through,
    "leaving a contact unresolved": lambda rw, k: _scale(rw, ("abandon",), k),
}
LEVELS = (0.5, 2.0)


def variants() -> list[tuple[str, dict]]:
    out = [("default", DEFAULT_REWARDS)]
    for name, f in FACTORS.items():
        for k in LEVELS:
            out.append((f"{name} ×{k:g}", f(DEFAULT_REWARDS, k)))
    return out


def counts_for(size: str, n_explore: int, n_refine: int, processes=None):
    off = SIZE_OFFSET[size]
    b = off + SEED_BASE["explore"]
    counts = collect(range(b, b + n_explore), size, {"kind": "explore"}, processes=processes)
    mdp = EngagementMDP(counts, DEFAULT_REWARDS)
    mdp.solve()
    b = off + SEED_BASE["refine"]
    spec = {"kind": "explore", "eps": 0.3,
            "base": {"kind": "mdp", "size": size, "policy": [ACTIONS[a] for a in mdp.policy]}}
    counts.merge(collect(range(b, b + n_refine), size, spec, processes=processes))
    return counts


def behaviour(results: list[dict]) -> dict:
    from .cue_model import PERCEPTION_OF
    n = len(results)
    sc = np.array([r["score"] for r in results])
    dec = [r for r in results if r["outcome"] == "declared"]
    return {
        "score": float(sc.mean()), "ci": float(1.96 * sc.std(ddof=1) / math.sqrt(n)),
        "correct": sum(r["declared"] == PERCEPTION_OF[r["type"]].value for r in dec) / n,
        "misfire": sum(r["declared"] == "enemy" and r["type"] in ("friendly", "civilian")
                       for r in dec) / max(1, sum(r["type"] in ("friendly", "civilian")
                                                  for r in results)),
        "let_through": sum(r["declared"] != "enemy" and PERCEPTION_OF[r["type"]].value == "enemy"
                           for r in dec) / max(1, sum(PERCEPTION_OF[r["type"]].value == "enemy"
                                                      for r in results)),
        "lost": sum(r["outcome"] == "dead" for r in results) / n,
        "unresolved": sum(r["outcome"] in ("abandon", "left", "timeout") for r in results) / n,
        "time_s": float(np.mean([r["time_s"] for r in results])),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", nargs="+", default=list(MAP_SIZES), choices=MAP_SIZES)
    ap.add_argument("--explore", type=int, default=12000)
    ap.add_argument("--refine", type=int, default=8000)
    ap.add_argument("--eval", type=int, default=2000)
    ap.add_argument("--processes", type=int, default=None)
    a = ap.parse_args(argv)
    L = ["# Engagement MDP — reward sensitivity", "",
         "Each key reward is halved (×0.5) and doubled (×2), the MDP re-solved, and the new "
         "policy run on the same fresh engagements. Score = mean reward under the DEFAULT "
         "rewards (one yardstick). Rates: misfire = share of friendlies/civilians declared "
         "enemy; let through = share of enemies declared friendly/civilian.", ""]
    for size in a.sizes:
        counts = counts_for(size, a.explore, a.refine, a.processes)
        seeds = range(SIZE_OFFSET[size] + 950_000, SIZE_OFFSET[size] + 950_000 + a.eval)
        L += [f"## {size.capitalize()} map", "",
              "| Variant | Score | Correct | Misfire | Let through | USV lost | Unresolved | Time |",
              "|---|---|---|---|---|---|---|---|"]
        for name, rw in variants():
            mdp = EngagementMDP(counts, rw)
            mdp.solve()
            spec = {"kind": "mdp", "size": size, "policy": [ACTIONS[x] for x in mdp.policy]}
            b = behaviour(evaluate(seeds, size, spec, rewards=DEFAULT_REWARDS,
                                   processes=a.processes))
            L.append(f"| {name} | {b['score']:.2f} ± {b['ci']:.2f} | {b['correct']:.1%} | "
                     f"{b['misfire']:.1%} | {b['let_through']:.1%} | {b['lost']:.1%} | "
                     f"{b['unresolved']:.1%} | {b['time_s']:.0f} s |")
            print(size, name, "done", flush=True)
        L.append("")
    (MODEL_DIR / "sensitivity_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
