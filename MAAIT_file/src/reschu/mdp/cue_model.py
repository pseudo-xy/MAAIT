"""Cue likelihoods P(cues | contact type) and the MDP's belief update.

Fitted offline from labelled simulator windows (collect.py); used at runtime with
truth-free features only (features.py). The belief over the five contact types is

    log b(c)  =  log prior(c)
              +  tau_speed * log P(mean_speed_bin | c)                 (ONCE)
              +  tau_beh   * sum_w log P(closing, turn, straight | c, band_w)
              +  tau_cam   * sum_w log P(camera look | c, band_w)   (fresh looks only)

* Speed enters ONCE, through the contact's running-mean speed. A boat's speed
  factor is persistent, so repeated speed readings are not independent evidence;
  counting them per window would make the belief wildly overconfident.
* Behavior enters once per window, from a joint table per distance band
  (behavior means different things up close — an aggressor locks on — than far).
* Successive windows are still correlated, so both terms are TEMPERED by
  exponents tuned on held-out validation runs to minimise log-loss, which keeps
  the probabilities calibrated (a stated 80% is right about 80% of the time).

The MDP declares at the Perception level (enemy / friendly / civilian);
`declaration_probs` collapses the 5-type belief to that.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from ..types import Perception
from .features import BANDS, WindowFeatures

CLASSES = ("civilian", "friendly", "standoff_low", "standoff_med", "aggressive")
PERCEPTION_OF = {
    "civilian": Perception.CIVILIAN,
    "friendly": Perception.FRIENDLY,
    "standoff_low": Perception.ENEMY,
    "standoff_med": Perception.ENEMY,
    "aggressive": Perception.ENEMY,
}
SPEED_EDGES = [i * 0.5 for i in range(0, 31)]                 # 0..15 knt, 0.5 bins
CLOSING_EDGES = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
TURN_EDGES = [0.0, 2.0, 5.0, 10.0, 20.0, 45.0, 180.0]          # deg per tick
STRAIGHT_EDGES = [0.0, 0.5, 0.8, 0.95, 1.0]
DEFAULT_PATH = Path(__file__).with_name("cue_model.json")


def _bin(x: float, edges: list[float]) -> int:
    i = bisect_right(edges, x) - 1
    return max(0, min(len(edges) - 2, i))


def _beh_cell(closing: float, turn: float, straight: float) -> int:
    nc, nt, ns = len(CLOSING_EDGES) - 1, len(TURN_EDGES) - 1, len(STRAIGHT_EDGES) - 1
    return (_bin(closing, CLOSING_EDGES) * nt + _bin(turn, TURN_EDGES)) * ns + _bin(
        straight, STRAIGHT_EDGES)


N_SPEED = len(SPEED_EDGES) - 1
N_BEH = (len(CLOSING_EDGES) - 1) * (len(TURN_EDGES) - 1) * (len(STRAIGHT_EDGES) - 1)


class CueModel:
    def __init__(self, speed_logp: dict, beh_logp: dict, cam_logp: dict | None = None,
                 tau_speed: float = 1.0, tau_beh: float = 1.0, tau_cam: float = 1.0,
                 meta: dict | None = None) -> None:
        self.speed_logp = speed_logp      # class -> [N_SPEED]
        self.beh_logp = beh_logp          # band -> class -> [N_BEH]
        self.cam_logp = cam_logp or {}    # band -> class -> [len(CLASSES)] (camera look)
        self.tau_speed = tau_speed
        self.tau_beh = tau_beh
        self.tau_cam = tau_cam
        self.meta = meta or {}

    # ------------------------------------------------------------------ #
    # fitting (offline)
    # ------------------------------------------------------------------ #
    @classmethod
    def fit(cls, rows: list[dict], alpha: float = 0.5) -> "CueModel":
        """Laplace-smoothed tables from labelled window rows."""
        sp = {c: [alpha] * N_SPEED for c in CLASSES}
        bh = {b: {c: [alpha] * N_BEH for c in CLASSES} for b in BANDS}
        cm = {b: {c: [alpha] * len(CLASSES) for c in CLASSES} for b in BANDS}
        for r in rows:
            c = r["label"]
            sp[c][_bin(float(r["speed"]), SPEED_EDGES)] += 1
            bh[r["band"]][c][_beh_cell(float(r["closing"]), float(r["turn"]),
                                       float(r["straightness"]))] += 1
            look = r.get("camera") or None
            if look:
                cm[r["band"]][c][CLASSES.index(look)] += 1

        def norm(v):
            z = sum(v)
            return [math.log(x / z) for x in v]

        return cls({c: norm(v) for c, v in sp.items()},
                   {b: {c: norm(v) for c, v in d.items()} for b, d in bh.items()},
                   {b: {c: norm(v) for c, v in d.items()} for b, d in cm.items()},
                   meta={"n_train_windows": len(rows), "alpha": alpha,
                         "n_camera_looks": sum(1 for r in rows if r.get("camera"))})

    # ------------------------------------------------------------------ #
    # runtime belief
    # ------------------------------------------------------------------ #
    def log_scores(self, mean_speed: float | None, windows: list[WindowFeatures] | list[dict],
                   prior: dict[str, float] | None = None) -> dict[str, float]:
        out = {}
        for c in CLASSES:
            s = math.log(prior[c]) if prior else math.log(1.0 / len(CLASSES))
            if mean_speed is not None:
                s += self.tau_speed * self.speed_logp[c][_bin(mean_speed, SPEED_EDGES)]
            for w in windows:
                g = (lambda k: w.get(k)) if isinstance(w, dict) else (lambda k: getattr(w, k, None))
                band = g("band")
                s += self.tau_beh * self.beh_logp[band][c][
                    _beh_cell(float(g("closing")), float(g("turn")), float(g("straightness")))]
                look = g("camera") or None
                if look and self.cam_logp:
                    s += self.tau_cam * self.cam_logp[band][c][CLASSES.index(look)]
            out[c] = s
        return out

    def posterior(self, mean_speed, windows, prior=None) -> dict[str, float]:
        """Belief over the five contact types."""
        ls = self.log_scores(mean_speed, windows, prior)
        m = max(ls.values())
        ex = {c: math.exp(v - m) for c, v in ls.items()}
        z = sum(ex.values())
        return {c: v / z for c, v in ex.items()}

    @staticmethod
    def declaration_probs(post: dict[str, float]) -> dict[Perception, float]:
        """Collapse the 5-type belief to what can be DECLARED (enemy/friendly/civilian)."""
        out: dict[Perception, float] = defaultdict(float)
        for c, p in post.items():
            out[PERCEPTION_OF[c]] += p
        return dict(out)

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path | str = DEFAULT_PATH) -> None:
        Path(path).write_text(json.dumps({
            "classes": CLASSES, "speed_edges": SPEED_EDGES, "closing_edges": CLOSING_EDGES,
            "turn_edges": TURN_EDGES, "straight_edges": STRAIGHT_EDGES, "bands": BANDS,
            "tau_speed": self.tau_speed, "tau_beh": self.tau_beh, "tau_cam": self.tau_cam,
            "speed_logp": self.speed_logp, "beh_logp": self.beh_logp,
            "cam_logp": self.cam_logp, "meta": self.meta,
        }, indent=1))

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "CueModel":
        d = json.loads(Path(path).read_text())
        if (d["speed_edges"] != SPEED_EDGES or d["closing_edges"] != CLOSING_EDGES
                or d["turn_edges"] != TURN_EDGES or d["straight_edges"] != STRAIGHT_EDGES):
            raise ValueError("cue_model.json was fitted with different bins — refit it")
        return cls(d["speed_logp"], d["beh_logp"], d.get("cam_logp"), d["tau_speed"],
                   d["tau_beh"], d.get("tau_cam", 1.0), d["meta"])


# ---------------------------------------------------------------------- #
# offline evaluation helpers
# ---------------------------------------------------------------------- #
def sequences(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group window rows into per-contact sequences (time-ordered)."""
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        g[(r["episode"], r["boat_id"])].append(r)
    out = []
    for seq in g.values():
        seq.sort(key=lambda r: float(r["t_end"]))
        out.append((seq[0]["label"], seq))
    return out


def _prefix_beliefs(model: CueModel, seqs, ks):
    """(k, true label, posterior) for each contact observed for >= k windows."""
    for label, seq in seqs:
        for k in ks:
            if len(seq) < k:
                continue
            pre = seq[:k]
            ms = sum(float(r["speed"]) for r in pre) / k
            yield k, label, model.posterior(ms, pre)


def log_loss(model: CueModel, seqs, ks=(1, 2, 3, 6, 12)) -> float:
    tot, n = 0.0, 0
    for _, label, post in _prefix_beliefs(model, seqs, ks):
        tot -= math.log(max(post[label], 1e-12))
        n += 1
    return tot / max(n, 1)


def tune_tempering(model: CueModel, val_rows: list[dict]) -> tuple[float, float, float]:
    """Grid-search the tempering exponents on validation log-loss."""
    seqs = sequences(val_rows)
    grid = [0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
    best = None
    for ts in grid:
        for tb in grid:
            for tc in grid:
                model.tau_speed, model.tau_beh, model.tau_cam = ts, tb, tc
                ll = log_loss(model, seqs)
                if best is None or ll < best[0]:
                    best = (ll, ts, tb, tc)
    model.tau_speed, model.tau_beh, model.tau_cam = best[1], best[2], best[3]
    model.meta["val_log_loss"] = round(best[0], 4)
    return best[1], best[2], best[3]


def evaluate(model: CueModel, rows: list[dict], ks=(1, 2, 3, 6, 12, 18)) -> dict:
    """Accuracy (5-type and declaration-level), calibration, confusion."""
    seqs = sequences(rows)
    res = {}
    for k in ks:
        n = c5 = c3 = 0
        bins = [[0, 0.0, 0] for _ in range(10)]   # count, sum conf, correct
        conf = defaultdict(int)
        for kk, label, post in _prefix_beliefs(model, seqs, [k]):
            n += 1
            pred = max(post, key=post.get)
            c5 += pred == label
            dp = CueModel.declaration_probs(post)
            dpred = max(dp, key=dp.get)
            ok = dpred is PERCEPTION_OF[label]
            c3 += ok
            p = dp[dpred]
            b = min(9, int(p * 10))
            bins[b][0] += 1; bins[b][1] += p; bins[b][2] += ok
            conf[(label, dpred.value)] += 1
        ece = sum(abs(s / cnt - cor / cnt) * cnt for cnt, s, cor in bins if cnt) / max(n, 1)
        res[k] = {"n": n, "acc5": c5 / max(n, 1), "acc_declare": c3 / max(n, 1),
                  "ece_declare": ece, "confusion": dict(conf)}
    return res
