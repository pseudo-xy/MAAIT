"""Estimate, solve and evaluate the engagement MDP — one reproducible command.

    python -m reschu.mdp.build                    # all map sizes, default effort
    python -m reschu.mdp.build --sizes small --explore 4000 --refine 3000 --eval 1000

For each map size:
  1. EXPLORE   run engagements with sticky random moves; count transitions
  2. SOLVE     value iteration on the estimated model
  3. REFINE    run more engagements, mostly following the solved policy
               (eps-greedy), so the states it actually visits are well estimated;
               merge the counts and solve again
  4. EVALUATE  on fresh seeds (same seeds for every policy): the MDP vs. the
               RUSH baseline (declare the most likely class on reaching ID range)
               and the THRESHOLD baseline (hold in ID range until 90% sure)
Writes mdp/engagement_<size>.json (the policy MDPPolicy loads) and
mdp/engagement_report.md. Rebuild after refitting the cue model, or after
changing rewards, sensing, speeds or behaviors.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from collections import Counter
from pathlib import Path

import numpy as np

from .engagement import (ACTIONS, BANDS, DEFAULT_REWARDS, MAP_SIZES, MODEL_DIR, N_CELLS,
                         EngagementMDP, describe_state)
from .episodes import collect, evaluate, summarize

SEED_BASE = {"explore": 100_000, "refine": 200_000, "eval": 900_000}
SIZE_OFFSET = {"small": 0, "medium": 1_000_000, "large": 2_000_000}


def build_size(size: str, n_explore: int, n_refine: int, n_eval: int,
               rewards: dict, processes: int | None, out_dir: Path) -> dict:
    off = SIZE_OFFSET[size]
    b = off + SEED_BASE["explore"]
    counts = collect(range(b, b + n_explore), size, {"kind": "explore"}, processes=processes)
    mdp = EngagementMDP(counts, rewards)
    mdp.solve()
    if n_refine:
        b = off + SEED_BASE["refine"]
        spec = {"kind": "explore", "eps": 0.3,
                "base": {"kind": "mdp", "size": size, "policy": [ACTIONS[a] for a in mdp.policy]}}
        counts.merge(collect(range(b, b + n_refine), size, spec, processes=processes))
        mdp = EngagementMDP(counts, rewards)
        mdp.solve()
    meta = {"built_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "episodes_explore": n_explore, "episodes_refine": n_refine,
            "transitions": int(counts.n_sa().sum()), "vi_iterations": mdp.iterations}
    table = mdp.table(size, meta)

    results = {}
    if n_eval:
        b = off + SEED_BASE["eval"]
        seeds = range(b, b + n_eval)
        specs = {"mdp": {"kind": "mdp", "size": size, "policy": table.policy},
                 "rush": {"kind": "rush"}, "threshold": {"kind": "threshold", "threshold": 0.9}}
        for name, spec in specs.items():
            results[name] = summarize(evaluate(seeds, size, spec, rewards=rewards,
                                               processes=processes))
        table.meta["eval"] = {k: {kk: v for kk, v in r.items() if kk != "by_type"}
                              for k, r in results.items()}
    table.save(out_dir)
    return {"mdp": mdp, "table": table, "results": results}


def _policy_map(mdp: EngagementMDP) -> list[str]:
    """Readable summary: which action the policy takes in each band, by belief."""
    lines = ["| Band | " + " | ".join(a for a in ACTIONS) + " |",
             "|---|" + "---|" * len(ACTIONS)]
    for bi, band in enumerate(BANDS):
        states = range(bi * N_CELLS, (bi + 1) * N_CELLS)
        w = Counter()
        for s in states:
            w[ACTIONS[mdp.policy[s]]] += max(1, mdp.visits[s])
        tot = sum(w.values())
        lines.append(f"| {band} | " + " | ".join(
            f"{w[a] / tot:.0%}" if w[a] else "–" for a in ACTIONS) + " |")
    return lines


def report_md(built: dict[str, dict], rewards: dict) -> str:
    L = ["# Engagement MDP — build report", "",
         f"Built {_dt.datetime.now().isoformat(timespec='seconds')}.", "",
         "Rewards: " + ", ".join(f"{k} = {v}" for k, v in rewards.items()
                                 if not isinstance(v, dict)), ""]
    for size, b in built.items():
        mdp, res = b["mdp"], b["results"]
        m = b["table"].meta
        L += [f"## {size.capitalize()} map", "",
              f"{m['episodes_explore']} exploration + {m['episodes_refine']} refinement "
              f"engagements, {m['transitions']} transitions; value iteration converged in "
              f"{m['vi_iterations']} sweeps.", ""]
        if res:
            L += ["| Policy | Score (95% CI) | Correct | Declared correctly when declaring | "
                  "USV lost | Unresolved | Mean time |", "|---|---|---|---|---|---|---|"]
            for name, r in res.items():
                L.append(f"| {name} | {r['score']:.2f} ± {r['score_ci']:.2f} | {r['correct']:.1%} | "
                         f"{r['correct_of_declared']:.1%} | {r['lost']:.1%} | "
                         f"{r['unresolved']:.1%} | {r['time_s']:.0f} s |")
            L += ["", "By contact type (score / correct / lost / unresolved / time):", "",
                  "| Type | " + " | ".join(res) + " |", "|---|" + "---|" * len(res)]
            for t in res["mdp"]["by_type"]:
                cells = []
                for name in res:
                    r = res[name]["by_type"][t]
                    cells.append(f"{r['score']:.1f} / {r['correct']:.0%} / {r['lost']:.0%} / "
                                 f"{r['unresolved']:.0%} / {r['time_s']:.0f} s")
                L.append(f"| {t} | " + " | ".join(cells) + " |")
        L += ["", "Policy by distance band (share of visited states choosing each action):", ""]
        L += _policy_map(mdp)
        # a few illustrative states
        L += ["", "Most-visited states and the chosen action:", ""]
        top = np.argsort(-mdp.visits)[:12]
        for s in top:
            L.append(f"- `{describe_state(int(s))}` → **{ACTIONS[mdp.policy[s]]}** "
                     f"(visits {int(mdp.visits[s])}, value {mdp.V[s]:.2f})")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", nargs="+", default=list(MAP_SIZES), choices=MAP_SIZES)
    ap.add_argument("--explore", type=int, default=12000)
    ap.add_argument("--refine", type=int, default=8000)
    ap.add_argument("--eval", type=int, default=3000)
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--out", type=Path, default=MODEL_DIR)
    a = ap.parse_args(argv)
    from .frozen import guard_write
    guard_write(a.out)
    built = {}
    for size in a.sizes:
        built[size] = build_size(size, a.explore, a.refine, a.eval, DEFAULT_REWARDS,
                                 a.processes, a.out)
        print(f"{size}: done")
    rep = report_md(built, DEFAULT_REWARDS)
    (a.out / "engagement_report.md").write_text(rep)
    print(rep)


if __name__ == "__main__":
    main()
