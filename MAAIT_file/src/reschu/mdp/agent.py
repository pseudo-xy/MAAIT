"""Runtime side of the engagement MDP: Observation -> belief -> action -> motion.

`EngagementAgent.act(usv, obs, dt_min)` is truth-free: it sees only the
Observation (sensed positions, measured speed / heading, camera looks), keeps the
team's cue history in a ContactTracker, turns it into a belief with the fitted
CueModel, and every DECISION_TICKS asks a *chooser* what to do with the USV's
assigned contact (immediately, too, whenever the contact's distance band
changes — the decisions are event-driven). The same agent (and the same motion code) is used

  * at runtime by MDPPolicy, with the solved policy table (TableChooser),
  * offline to explore the simulator when estimating the MDP (ExploreChooser),
  * for the comparison baselines (RushChooser, ThresholdChooser),

so the policy is trained and evaluated on exactly the mechanics it runs with.

Motion is "take station at a distance d from the contact, along the current
bearing", moving at the USV's speed:
    approach  d = APPROACH_RATIO ID ranges      hold  d = distance at decision time
    back_off  d grows every tick
    declare X d = APPROACH_RATIO ID ranges, with the declaration requested EVERY
              tick: the engine applies it on the first tick the contact is truly
              in ID range (a committed close-in when made from further out: no
              more decisions are taken on this contact until it resolves)
"""

from __future__ import annotations

import math
import random

from ..behaviors import UNITS_PER_KNOT_MIN
from ..observation import Observation
from ..types import Vec2
from .cue_model import CueModel
from .engagement import (ACTIONS, APPROACH_RATIO, DECISION_TICKS, DECL_PERCEPTION,
                         DECLARABLE_BANDS, DECLARE_ACTIONS, MOVE_ACTIONS, PolicyTable,
                         band_index, belief_cell, size_for_width, state_index)

SIM_ACTIONS = MOVE_ACTIONS + DECLARE_ACTIONS
from .features import ContactTracker, id_range_for_width

_CUE: CueModel | None = None


def shared_cue_model() -> CueModel:
    global _CUE
    if _CUE is None:
        _CUE = CueModel.load()
    return _CUE


# ---------------------------------------------------------------------- #
# choosers: (band, belief, obs) -> action name
# ---------------------------------------------------------------------- #
def _map_declaration(post: dict[str, float]) -> tuple[str, float]:
    dp = CueModel.declaration_probs(post)
    best = max(DECL_PERCEPTION, key=lambda d: dp.get(DECL_PERCEPTION[d], 0.0))
    return best, dp.get(DECL_PERCEPTION[best], 0.0)


class TableChooser:
    """The solved MDP policy (one table per map size)."""
    name = "mdp"

    def __init__(self, tables: dict[str, PolicyTable] | None = None) -> None:
        self.tables = dict(tables or {})

    def table(self, size: str) -> PolicyTable:
        if size not in self.tables:
            self.tables[size] = PolicyTable.load(size)
        return self.tables[size]

    def choose(self, band: int, post: dict, obs: Observation) -> str:
        return self.table(size_for_width(obs.grid_width)).action(band, belief_cell(post))


CAMERA_BANDS = (2, 3)          # cam_near, cam_far


class RushChooser:
    """Baseline: approach; on reaching camera range, commit to the most likely
    class (close in and declare it)."""
    name = "rush"

    def choose(self, band: int, post: dict, obs: Observation) -> str:
        if band in DECLARABLE_BANDS or band in CAMERA_BANDS:
            return "declare_" + _map_declaration(post)[0]
        return "approach"


class ThresholdChooser:
    """Baseline: approach to camera range and watch from there; commit to the
    most likely class once it is `threshold` sure (forced if already in ID range)."""
    name = "threshold"

    def __init__(self, threshold: float = 0.9) -> None:
        self.threshold = threshold

    def choose(self, band: int, post: dict, obs: Observation) -> str:
        d, p = _map_declaration(post)
        if band in DECLARABLE_BANDS or p >= self.threshold:
            return "declare_" + d
        if band in CAMERA_BANDS:
            return "hold"
        return "approach"


class ExploreChooser:
    """Offline exploration for estimating transitions: sticky random actions, or
    (with probability 1 - eps) what a base policy would do. Never abandons (its
    value needs no simulation)."""
    name = "explore"

    def __init__(self, rng: random.Random, eps: float = 1.0, base=None,
                 sticky: float = 0.5,
                 probs=(0.40, 0.25, 0.20, 0.05, 0.05, 0.05)) -> None:
        self.rng, self.eps, self.base = rng, eps, base
        self.sticky, self.probs = sticky, probs
        self._prev: str | None = None

    def choose(self, band: int, post: dict, obs: Observation) -> str:
        a = None
        if self.base is not None and self.rng.random() > self.eps:
            a = self.base.choose(band, post, obs)
            if a not in SIM_ACTIONS:
                a = None
        if a is None:
            if self._prev is not None and self.rng.random() < self.sticky:
                a = self._prev
            else:
                a = self.rng.choices(SIM_ACTIONS, weights=self.probs)[0]
        self._prev = a
        return a


# ---------------------------------------------------------------------- #
# motion (world-free: everything comes from the Observation)
# ---------------------------------------------------------------------- #
def step_toward(usv, target: Vec2, obs: Observation, dt_min: float) -> Vec2:
    tx = min(max(target.x, 0.0), obs.grid_width)
    ty = min(max(target.y, 0.0), obs.grid_height)
    dx, dy = tx - usv.position.x, ty - usv.position.y
    dist = math.hypot(dx, dy)
    step = usv.effective_speed_knots() * UNITS_PER_KNOT_MIN * dt_min
    if dist <= 1e-9:
        return usv.position
    usv.heading = Vec2(x=dx / dist, y=dy / dist)
    if dist <= step:
        return Vec2(x=tx, y=ty)
    return Vec2(x=usv.position.x + dx / dist * step, y=usv.position.y + dy / dist * step)


def station_step(usv, boat_pos: Vec2, d_goal: float, obs: Observation, dt_min: float) -> Vec2:
    """Move toward the point at distance d_goal from the contact, on our bearing."""
    ox, oy = usv.position.x - boat_pos.x, usv.position.y - boat_pos.y
    L = math.hypot(ox, oy)
    ux, uy = (ox / L, oy / L) if L > 1e-9 else (1.0, 0.0)
    return step_toward(usv, Vec2(x=boat_pos.x + ux * d_goal, y=boat_pos.y + uy * d_goal),
                       obs, dt_min)


# ---------------------------------------------------------------------- #
class EngagementAgent:
    """Shared by all MDP USVs of one mission (the cue track is team-level)."""

    def __init__(self, chooser, cue_model: CueModel | None = None, on_decision=None) -> None:
        self.chooser = chooser
        self.cue = cue_model or shared_cue_model()
        self.on_decision = on_decision
        self.reset()

    def reset(self) -> None:
        self.tracker = ContactTracker()
        self._last_now: float | None = None
        self._state: dict[str, dict] = {}          # usv id -> current decision
        self.abandoned: dict[str, set[str]] = {}   # usv id -> contacts it gave up

    def track(self, obs: Observation) -> None:
        """Feed one tick of observation (call EVERY tick, even while circling)."""
        if self._last_now is not None and obs.now_min < self._last_now - 1e-9:
            self.reset()                            # a new mission started
        self._last_now = obs.now_min
        self.tracker.update(obs)

    def committed(self, usv_id: str, boat_id: str) -> bool:
        """Is this USV committed to a declaration on this contact (close-in under way)?"""
        st = self._state.get(usv_id)
        return st is not None and st["boat"] == boat_id and st["action"] in DECLARE_ACTIONS

    def _best_kept(self, band: int, post: dict, obs: Observation) -> str:
        """Best action other than abandon (for an operator-picked contact)."""
        table = getattr(self.chooser, "table", None)
        if table is not None:
            q = table(size_for_width(obs.grid_width)).q
            if q:
                row = q[state_index(band, belief_cell(post))]
                cands = [(v, a) for a, v in zip(ACTIONS, row)
                         if v is not None and a != "abandon"]
                if cands:
                    return max(cands)[1]
        return "approach"

    def belief(self, boat_id: str) -> dict[str, float]:
        return self.cue.posterior(self.tracker.mean_speed(boat_id), self.tracker.windows(boat_id))

    def act(self, usv, obs: Observation, dt_min: float) -> Vec2:
        self.track(obs)
        tid = obs.me.assigned_target_id
        b = obs.boat(tid) if tid else None
        if b is None or b.left_grid or b.identified:
            return usv.position
        pinned = getattr(usv, "assignment_pinned", False)
        if pinned:
            # the operator picked this contact: it is worked, never abandoned
            self.abandoned.get(usv.id, set()).discard(tid)
        elif tid in self.abandoned.get(usv.id, ()):   # given up: release it
            usv.assigned_target_id = None
            return usv.position
        idr = id_range_for_width(obs.grid_width)
        st = self._state.get(usv.id)
        band = band_index(b.distance / idr)
        committed = st is not None and st["boat"] == tid and st["action"] in DECLARE_ACTIONS
        if not committed and (st is None or st["boat"] != tid or band != st["band"]
                              or obs.now_min - st["t"] >= DECISION_TICKS * dt_min - 1e-9):
            post = self.belief(tid)
            action = self.chooser.choose(band, post, obs)
            if action == "abandon" and pinned:
                action = self._best_kept(band, post, obs)
            if action not in ACTIONS:
                raise ValueError(f"unknown action {action!r}")
            st = {"boat": tid, "action": action, "t": obs.now_min, "hold_d": b.distance,
                  "band": band}
            self._state[usv.id] = st
            if self.on_decision is not None:
                self.on_decision(usv.id, tid, obs.now_min, band, post, action)

        a = st["action"]
        if a == "abandon":
            self.abandoned.setdefault(usv.id, set()).add(tid)
            usv.assigned_target_id = None
            return usv.position
        if a.startswith("declare"):
            usv.pending_declaration = (tid, DECL_PERCEPTION[a.split("_", 1)[1]])
            d_goal = APPROACH_RATIO * idr
        elif a == "approach":
            d_goal = APPROACH_RATIO * idr
        elif a == "hold":
            d_goal = st["hold_d"]
        else:                                        # back_off
            d_goal = b.distance + 3.0 * usv.effective_speed_knots() * UNITS_PER_KNOT_MIN * dt_min
        return station_step(usv, b.position, d_goal, obs, dt_min)
