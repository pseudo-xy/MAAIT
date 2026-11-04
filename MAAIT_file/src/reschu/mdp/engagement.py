"""The ENGAGEMENT MDP: one USV working one contact, until it commits or leaves.

State   (distance band, belief cell)                       6 x 45 = 270 states
  * distance band — SENSED distance to the contact, in ID ranges:
        danger   <= 0.75   an un-neutralised aggressor kills a USV here
        id       <= 1.0    a declaration can be made
        cam_near <= 1.25   camera looks are good
        cam_far  <= 1.5    camera looks are poorer (camera range ends here)
        mid      <= 2.5    kinematic cues only
        far       > 2.5    smoothed / no kinematic cues
  * belief cell — the cue model's belief over the five contact types, summarised as
        (most likely declaration E/F/C) x (its probability, 5 bins)
        x (probability the contact is an AGGRESSOR, 3 bins)

Actions
  approach          move to 0.9 ID ranges from the contact (just inside ID range)
  hold              keep the current distance
  back_off          open the distance
  declare_enemy / declare_friendly / declare_civilian
                    commit to a class. In the danger / id bands it applies at
                    once; from further out it is a COMMITTED CLOSE-IN: drive in
                    and declare the moment the contact is in ID range, with no
                    change of mind on the way.
  abandon           leave the contact unresolved — terminal

Why commit before arriving: an aggressor locks on at ID range and reaches kill
distance within about a second, so a USV that enters ID range undecided rarely
gets to declare. The MDP must choose WHERE to make up its mind.

Decisions are event-driven (a semi-MDP): every DECISION_TICKS (10 s), and
immediately whenever the sensed distance band changes. Time is charged per
elapsed second.

Transitions are NOT hand-written. They are counted from simulated engagements
(episodes.py) run in the real simulator with real sensing: how the distance band
and the belief cell move for each action, how often the USV is destroyed, how
often the contact leaves the map. Rewards for terminal actions use the empirical
mix of TRUE types seen in each state (truth is used OFFLINE only, to score).

The model is solved by value iteration, separately for each map size (a 10-s step
covers 2 ID ranges on the small map but only 0.67 on the large one).

REWARDS are research choices — set them in DEFAULT_REWARDS or pass your own.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np

from ..types import Perception
from .cue_model import CLASSES, CueModel

# ---------------------------------------------------------------------- #
# state space
# ---------------------------------------------------------------------- #
# engine.DANGER_ZONE_RADIUS / identification.IDENTIFY_RANGE (checked by a test;
# not imported, to keep engine -> policies -> mdp free of import cycles)
DANGER_RATIO = 0.75
BAND_EDGES = (DANGER_RATIO, 1.0, 1.25, 1.5, 2.5)
BANDS = ("danger", "id", "cam_near", "cam_far", "mid", "far")
DECLARABLE_BANDS = (0, 1)                                   # danger, id

DECL = ("enemy", "friendly", "civilian")
DECL_PERCEPTION = {"enemy": Perception.ENEMY, "friendly": Perception.FRIENDLY,
                   "civilian": Perception.CIVILIAN}
CONF_EDGES = (0.6, 0.75, 0.9, 0.97)                         # 5 confidence bins
AGG_EDGES = (0.1, 0.4)                                      # 3 aggressor bins
N_CONF, N_AGG = len(CONF_EDGES) + 1, len(AGG_EDGES) + 1
N_CELLS = len(DECL) * N_CONF * N_AGG                        # 45
N_BANDS = len(BANDS)
N_STATES = N_BANDS * N_CELLS                                # 270

MOVE_ACTIONS = ("approach", "hold", "back_off")
DECLARE_ACTIONS = ("declare_enemy", "declare_friendly", "declare_civilian")
ACTIONS = MOVE_ACTIONS + DECLARE_ACTIONS + ("abandon",)
N_MOVE = len(MOVE_ACTIONS)
N_SIM = N_MOVE + len(DECLARE_ACTIONS)         # actions whose outcome is simulated

DECISION_TICKS = 10
APPROACH_RATIO = 0.9          # approach = take station at 0.9 ID ranges
MAP_SIZES = ("small", "medium", "large")
MODEL_DIR = Path(__file__).parent

# ---------------------------------------------------------------------- #
# rewards (placeholders — the researcher's choice)
# ---------------------------------------------------------------------- #
DEFAULT_REWARDS: dict = {
    # declared class -> true contact type -> reward
    "declare": {
        "enemy":    {"aggressive": 10, "standoff_low": 5, "standoff_med": 5,
                     "friendly": -20, "civilian": -20},
        "friendly": {"aggressive": -30, "standoff_low": -10, "standoff_med": -10,
                     "friendly": 5, "civilian": -1},
        "civilian": {"aggressive": -30, "standoff_low": -10, "standoff_med": -10,
                     "friendly": -2, "civilian": 3},
    },
    # contact left unresolved (abandoned, left the map, or ran out of time)
    "abandon": {"aggressive": -8, "standoff_low": -4, "standoff_med": -4,
                "friendly": -2, "civilian": -0.5},
    "usv_lost": -25,
    "step": -0.2,             # per 10-s decision step
}


def band_index(ratio: float) -> int:
    """Distance band from the sensed distance in ID ranges."""
    return bisect_left(BAND_EDGES, ratio)


def belief_cell(post: dict[str, float]) -> int:
    dp = CueModel.declaration_probs(post)
    probs = [dp.get(DECL_PERCEPTION[d], 0.0) for d in DECL]
    m = max(range(3), key=lambda i: probs[i])
    ci = bisect_right(CONF_EDGES, probs[m])
    ai = bisect_right(AGG_EDGES, post.get("aggressive", 0.0))
    return (m * N_CONF + ci) * N_AGG + ai


def state_index(band: int, cell: int) -> int:
    return band * N_CELLS + cell


def describe_state(s: int) -> str:
    band, cell = divmod(s, N_CELLS)
    m, rest = divmod(cell, N_CONF * N_AGG)
    ci, ai = divmod(rest, N_AGG)
    ce, ae = [0.0, *CONF_EDGES, 1.0], [0.0, *AGG_EDGES, 1.0]
    return (f"{BANDS[band]} | P({DECL[m]}) {ce[ci]:.2f}-{ce[ci + 1]:.2f} (most likely) | "
            f"P(aggressor) {ae[ai]:.2f}-{ae[ai + 1]:.2f}")


# ---------------------------------------------------------------------- #
# estimation from simulated transitions
# ---------------------------------------------------------------------- #
OUTCOMES = ("dead", "left", "declared")      # terminal outcomes seen in simulation


class Counts:
    """Sufficient statistics from simulated engagement transitions.

    Simulated actions are the moves AND the declarations (a declaration made
    outside ID range is a committed close-in: it takes effect on entering range).
    Terminal outcomes are kept by the contact's TRUE type, so their rewards are
    exact averages over what actually happened. Band-level counts are also kept
    per true type: they describe the PHYSICS (how each type moves, attacks or
    leaves in each band), which a rarely-visited state borrows."""

    def __init__(self) -> None:
        T, O = len(CLASSES), len(OUTCOMES)
        self.cont = np.zeros((N_STATES, N_SIM, N_STATES))
        self.term = np.zeros((N_STATES, N_SIM, O, T))
        self.dur = np.zeros((N_STATES, N_SIM))                # summed ticks
        self.types = np.zeros((N_STATES, T))                  # true type at each visit
        self.belief_sum = np.zeros((N_STATES, T))             # summed beliefs at visits
        self.cont_bt = np.zeros((N_BANDS, N_SIM, T, N_BANDS))  # band physics, per type
        self.term_bt = np.zeros((N_BANDS, N_SIM, T, O))
        self.dur_bt = np.zeros((N_BANDS, N_SIM, T))

    _FIELDS = ("cont", "term", "dur", "types", "belief_sum", "cont_bt", "term_bt", "dur_bt")

    def add(self, rec: dict) -> None:
        s, a, t = rec["s"], rec["a"], CLASSES.index(rec["type"])
        self.types[s, t] += 1
        self.belief_sum[s] += rec["belief"]
        if a is None:              # visit only (abandon / truncated)
            return
        band = s // N_CELLS
        self.dur[s, a] += rec["ticks"]
        self.dur_bt[band, a, t] += rec["ticks"]
        if rec["outcome"] == "cont":
            self.cont[s, a, rec["s_next"]] += 1
            self.cont_bt[band, a, t, rec["s_next"] // N_CELLS] += 1
        else:
            o = OUTCOMES.index(rec["outcome"])
            self.term[s, a, o, t] += 1
            self.term_bt[band, a, t, o] += 1

    def merge(self, other: "Counts") -> None:
        for k in self._FIELDS:
            getattr(self, k).__iadd__(getattr(other, k))

    def n_sa(self) -> np.ndarray:
        return self.cont.sum(axis=2) + self.term.sum(axis=(2, 3))


class EngagementMDP:
    """Estimated model + value-iteration solution for one map size (a semi-MDP:
    decisions every 10 s or on a band change; time cost is per elapsed second)."""

    def __init__(self, counts: Counts, rewards: dict | None = None,
                 gamma: float = 0.99, backoff_k: float = 10.0,
                 type_prior_k: float = 10.0) -> None:
        self.rewards = rewards or DEFAULT_REWARDS
        self.gamma = gamma
        self.counts = counts
        self._estimate(backoff_k, type_prior_k)

    # -------------------------------------------------------------- #
    def _estimate(self, K: float, KT: float) -> None:
        c = self.counts
        T, O = len(CLASSES), len(OUTCOMES)
        # P(true type | state): visit counts, shrunk toward the mean belief there
        # (and toward uniform when a state was never visited)
        visits = c.types.sum(axis=1, keepdims=True)
        mean_belief = np.where(visits > 0, c.belief_sum / np.maximum(visits, 1), 1.0 / T)
        self.ptype = (c.types + KT * mean_belief) / (visits + KT)
        self.visits = visits[:, 0]

        # Each state's own counts, shrunk (weight K) toward a prior built from
        #   P(type | state) x band-level physics of that type,
        # with the belief cell held fixed for continuing transitions.
        n = c.n_sa()
        P = np.zeros((N_STATES, N_SIM, N_STATES))
        PT = np.zeros((N_STATES, N_SIM, O, T))
        D = np.zeros((N_STATES, N_SIM))
        self.available = np.zeros((N_STATES, len(ACTIONS)), dtype=bool)
        for b in range(N_BANDS):
            for a in range(N_SIM):
                n_t = c.cont_bt[b, a].sum(axis=1) + c.term_bt[b, a].sum(axis=1)   # [T]
                nb = n_t.sum()
                if nb == 0:
                    continue
                pooled = (c.cont_bt[b, a].sum(0) / nb, c.term_bt[b, a].sum(0) / nb,
                          c.dur_bt[b, a].sum() / nb)
                rc = np.zeros((T, N_BANDS)); rt = np.zeros((T, O)); rd = np.zeros(T)
                for t in range(T):
                    if n_t[t] > 0:
                        rc[t] = c.cont_bt[b, a, t] / n_t[t]
                        rt[t] = c.term_bt[b, a, t] / n_t[t]
                        rd[t] = c.dur_bt[b, a, t] / n_t[t]
                    else:
                        rc[t], rt[t], rd[t] = pooled
                for cell in range(N_CELLS):
                    s = state_index(b, cell)
                    pt = self.ptype[s]
                    back = np.zeros(N_STATES)
                    back[[state_index(b2, cell) for b2 in range(N_BANDS)]] = pt @ rc
                    w = n[s, a]
                    P[s, a] = (c.cont[s, a] + K * back) / (w + K)
                    PT[s, a] = (c.term[s, a] + K * (rt * pt[:, None]).T) / (w + K)
                    D[s, a] = (c.dur[s, a] + K * (pt @ rd)) / (w + K)
                    self.available[s, a] = True
        self.P, self.PT, self.D = P, PT, D
        self.self_n = n
        self.available[:, ACTIONS.index("abandon")] = True

    # -------------------------------------------------------------- #
    def _vec(self, table: dict) -> np.ndarray:
        return np.array([table[t] for t in CLASSES], dtype=float)

    def immediate(self) -> np.ndarray:
        """Expected reward collected during one decision R[s, a] (terminal parts
        included). In the declarable bands a declaration applies at once, so its
        reward uses P(type | state) directly."""
        R = np.full((N_STATES, len(ACTIONS)), -np.inf)
        rw = self.rewards
        ab = self._vec(rw["abandon"])
        for a, name in enumerate(ACTIONS[:N_SIM]):
            decl = self._vec(rw["declare"][name.split("_")[1]]) if name in DECLARE_ACTIONS \
                else np.zeros(len(CLASSES))
            R[:, a] = (rw["step"] * self.D[:, a] / DECISION_TICKS
                       + self.PT[:, a, 0, :].sum(axis=1) * rw["usv_lost"]
                       + self.PT[:, a, 1, :] @ ab
                       + self.PT[:, a, 2, :] @ decl)
            if name in DECLARE_ACTIONS:
                for band in DECLARABLE_BANDS:
                    sl = slice(band * N_CELLS, (band + 1) * N_CELLS)
                    R[sl, a] = self.ptype[sl] @ decl
                    self.available[sl, a] = True
        R[:, ACTIONS.index("abandon")] = self.ptype @ ab
        R[~self.available] = -np.inf
        return R

    def continuation(self) -> np.ndarray:
        """P(continue to s') with declarations in the declarable bands terminal."""
        P = self.P.copy()
        for a, name in enumerate(ACTIONS[:N_SIM]):
            if name in DECLARE_ACTIONS:
                for band in DECLARABLE_BANDS:
                    P[band * N_CELLS:(band + 1) * N_CELLS, a] = 0.0
        return P

    def solve(self, tol: float = 1e-6, max_iter: int = 10000) -> None:
        R = self.immediate()
        P = self.continuation()
        V = np.zeros(N_STATES)
        for it in range(max_iter):
            Q = R.copy()
            for a in range(N_SIM):
                Q[:, a] = R[:, a] + self.gamma * P[:, a] @ V
            Q[~self.available] = -np.inf
            V_new = Q.max(axis=1)
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new
        self.Q, self.V, self.iterations = Q, V, it + 1
        self.policy = Q.argmax(axis=1)
        self.time_to_go = self._time_to_go(P)

    def _time_to_go(self, P: np.ndarray, max_iter: int = 5000) -> np.ndarray:
        """Expected seconds until the engagement resolves, following the policy
        (policy evaluation on the decision durations). Used by the assignment."""
        S = np.arange(N_STATES)
        a = self.policy
        sim = a < N_SIM
        dur = np.where(sim, self.D[S, np.minimum(a, N_SIM - 1)], 0.0)
        Pp = np.zeros((N_STATES, N_STATES))
        Pp[sim] = P[S[sim], a[sim]]
        T = np.zeros(N_STATES)
        for _ in range(max_iter):
            T_new = dur + 0.999 * Pp @ T
            if np.max(np.abs(T_new - T)) < 1e-6:
                break
            T = T_new
        return T_new  # ticks = seconds at the runtime tick

    # -------------------------------------------------------------- #
    def table(self, size: str, meta: dict | None = None) -> "PolicyTable":
        meta = dict(meta or {})
        meta["time_to_go_s"] = [round(float(x), 2) for x in self.time_to_go]
        return PolicyTable(size=size, policy=[ACTIONS[a] for a in self.policy],
                           q=np.where(np.isfinite(self.Q), self.Q, None).tolist(),
                           visits=self.visits.astype(int).tolist(),
                           rewards=self.rewards, meta=meta)


class PolicyTable:
    """The solved policy for one map size: state index -> action (+ Q values)."""

    def __init__(self, size: str, policy: list[str], q: list | None = None,
                 visits: list | None = None, rewards: dict | None = None,
                 meta: dict | None = None) -> None:
        self.size, self.policy = size, policy
        self.q, self.visits = q or [], visits or []
        self.rewards, self.meta = rewards or {}, meta or {}

    def action(self, band: int, cell: int) -> str:
        return self.policy[state_index(band, cell)]

    @staticmethod
    def path(size: str, directory: Path = MODEL_DIR) -> Path:
        return Path(directory) / f"engagement_{size}.json"

    def save(self, directory: Path = MODEL_DIR) -> Path:
        p = self.path(self.size, directory)
        p.write_text(json.dumps({
            "size": self.size, "bands": BANDS, "band_edges": BAND_EDGES,
            "conf_edges": CONF_EDGES, "agg_edges": AGG_EDGES, "actions": ACTIONS,
            "policy": self.policy, "q": self.q, "visits": self.visits,
            "rewards": self.rewards, "meta": self.meta}, indent=1))
        return p

    @classmethod
    def load(cls, size: str, directory: Path = MODEL_DIR) -> "PolicyTable":
        d = json.loads(cls.path(size, directory).read_text())
        if (tuple(d["band_edges"]) != BAND_EDGES or tuple(d["conf_edges"]) != CONF_EDGES
                or tuple(d["agg_edges"]) != AGG_EDGES or tuple(d["actions"]) != ACTIONS):
            raise ValueError(f"engagement_{size}.json uses a different state space — rebuild it")
        return cls(d["size"], d["policy"], d.get("q"), d.get("visits"),
                   d.get("rewards"), d.get("meta"))


def size_for_width(width: float) -> str:
    k = max(1, min(3, round(width / 40.0)))
    return MAP_SIZES[k - 1]


def expected_terminal(rewards: dict, ctype: str, outcome: str,
                      declared: str | None = None) -> float:
    """Score a finished engagement from its TRUE type (evaluation only)."""
    if outcome == "declared":
        return rewards["declare"][declared][ctype]
    if outcome == "dead":
        return rewards["usv_lost"]
    return rewards["abandon"][ctype]          # abandoned / left map / timed out


__all__ = [n for n in dir() if not n.startswith("_") and n not in ("json", "math", "np")]
