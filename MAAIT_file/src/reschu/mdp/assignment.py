"""MDP team assignment: which contact each MDP USV works.

Every ASSIGN_TICKS (10 s), or at once when an MDP USV is free, the free and
re-assignable MDP USVs are matched one-to-one to contacts so as to maximise the
team's total GAIN (a weighted bipartite matching, solved exactly by
scipy.optimize.linear_sum_assignment). A USV may also stay idle (gain 0).

Gain of USV u working contact c = how much better the expected outcome is if u
works c than if c is left unworked:

  * undeclared contact: from the solved engagement MDP,
        V(s) - Q(s, abandon) - extra travel cost,
    where s = (distance band of c from u, belief cell of c) and the extra travel
    cost charges the time to cover any distance beyond FAR_REF_RATIO ID ranges
    (the MDP's "far" band lumps every distance beyond 2.5 ID ranges together;
    it was estimated from starts of 2.5-5, mean ~3.75);
  * declared contact still needing work (a perceived enemy to circle, a
    perceived friendly to escort): reward for handling it - reward for leaving
    it - travel cost. The kind of enemy comes from the shown threat tier.

Who takes part (own-side knowledge only — never a boat's truth):
  * locked, keep their contact: a USV circling / escorting a declared contact,
    one committed to a declaration (close-in under way), one whose contact the
    operator picked (`assignment_pinned`) until that contact resolves;
  * excluded: RTB, eliminated, departed USVs; contacts held by any other USV.
A USV's current undeclared contact gets SWITCH_MARGIN extra gain (hysteresis),
so USVs switch only for a clearly better use of their time.

Two scoring rules (World.mdp_assignment):
  "value"  maximise total gain;
  "rate"   (DEFAULT — clearly better in mission tests, see assignment_report.md)
           maximise total gain PER MINUTE the USV will be tied up — expected
           engagement time from the solved MDP (time_to_go of the state) plus
           travel, plus RATE_FLOOR_MIN. The classic correction for sequential
           assignment: a quick sure win can beat a slow larger one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..behaviors import UNITS_PER_KNOT_MIN
from ..observation import BoatView, Observation
from ..types import Perception, ThreatTier, Vec2
from .engagement import (ACTIONS, DECISION_TICKS, DEFAULT_REWARDS, band_index,
                         belief_cell, size_for_width, state_index)
from .features import id_range_for_width

ASSIGN_TICKS = 10
SWITCH_MARGIN = 1.0
RATE_FLOOR_MIN = 0.5           # minutes added to every engagement time ("rate")
ESCORT_MIN = 2.0               # assumed escort time for a declared friendly ("rate")
FAR_REF_RATIO = 3.75
_ABANDON = ACTIONS.index("abandon")


@dataclass
class TeamMember:
    """One MDP USV as the team coordinator sees it (own-side state only)."""
    id: str
    position: Vec2
    speed_knots: float
    assigned: str | None
    locked: bool = False


class TeamAssigner:
    def __init__(self, agent, chooser, rewards: dict | None = None, mode: str = "rate") -> None:
        self.agent, self.chooser = agent, chooser
        self.mode = mode
        self.rewards = rewards or DEFAULT_REWARDS
        self._last_t: float | None = None
        self.last_scores: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _travel_s(dist_units: float, speed_knots: float) -> float:
        return dist_units / max(1e-9, speed_knots * UNITS_PER_KNOT_MIN) * 60.0

    def _travel_cost(self, dist_units: float, speed_knots: float, dt_min: float) -> float:
        return self.rewards["step"] * self._travel_s(dist_units, speed_knots) / (
            DECISION_TICKS * dt_min * 60.0)

    def _rate(self, g: float, seconds: float) -> float:
        return g / (seconds / 60.0 + RATE_FLOOR_MIN) if self.mode == "rate" else g

    def gain(self, m: TeamMember, b: BoatView, obs: Observation, dt_min: float) -> float | None:
        """Gain of member m working contact b (None: not worth / not workable)."""
        if b.id in self.agent.abandoned.get(m.id, ()):
            return None                      # this USV already gave it up
        if b.operator_handling:
            return None                      # the operator is escorting / circling it
        idr = id_range_for_width(obs.grid_width)
        dist = m.position.distance_to(b.position)
        rw = self.rewards
        if not b.identified:
            table = self.chooser.table(size_for_width(obs.grid_width))
            post = self.agent.belief(b.id)
            s = state_index(band_index(dist / idr), belief_cell(post))
            q = [x for x in table.q[s] if x is not None] if table.q else []
            if not q or table.q[s][_ABANDON] is None:
                return None
            g = max(q) - table.q[s][_ABANDON]
            extra = max(0.0, dist - FAR_REF_RATIO * idr)
            g += self._travel_cost(extra, m.speed_knots, dt_min)
            if m.assigned == b.id:
                g += SWITCH_MARGIN
            ttg = table.meta.get("time_to_go_s")
            secs = (ttg[s] * dt_min * 60.0 if ttg else 60.0) + self._travel_s(extra, m.speed_knots)
            return self._rate(g, secs)
        # declared contacts that still need a USV
        from ..usv_movement import CIRCLE_DURATION_MIN
        if b.perceived_class is Perception.ENEMY:
            kind = "aggressive" if b.threat_tier is ThreatTier.HIGH else "standoff_low"
            g = rw["declare"]["enemy"][kind] - rw["abandon"][kind]
            work_s = CIRCLE_DURATION_MIN * 60.0
        elif b.perceived_class is Perception.FRIENDLY and not b.being_escorted:
            g = rw["declare"]["friendly"]["friendly"] - rw["abandon"]["friendly"]
            work_s = ESCORT_MIN * 60.0
        else:
            return None                      # civilians are left alone
        g += self._travel_cost(dist, m.speed_knots, dt_min)
        return self._rate(g, self._travel_s(dist, m.speed_knots) + work_s)

    # ------------------------------------------------------------------ #
    def due(self, now_min: float, members: list[TeamMember], dt_min: float) -> bool:
        if self._last_t is not None and now_min < self._last_t - 1e-9:
            self._last_t = None              # new mission
        if any(not m.locked and m.assigned is None for m in members):
            return True
        return self._last_t is None or now_min - self._last_t >= ASSIGN_TICKS * dt_min - 1e-9

    def assign(self, members: list[TeamMember], obs: Observation, held: set[str],
               dt_min: float) -> dict[str, str | None]:
        """New contact for every unlocked member (None = idle). `held` = contacts
        worked by USVs outside this matching (locked members, other agents)."""
        self._last_t = obs.now_min
        self.last_scores: dict[str, dict[str, float]] = {}
        free = [m for m in members if not m.locked]
        boats = [b for b in obs.boats if not b.left_grid and b.id not in held]
        if not free:
            return {}
        G = np.full((len(free), len(boats)), -np.inf)
        for i, m in enumerate(free):
            for j, b in enumerate(boats):
                g = self.gain(m, b, obs, dt_min)
                if g is not None:
                    G[i, j] = g
                    self.last_scores.setdefault(m.id, {})[b.id] = g
        # idle columns (gain 0) so a USV is never forced onto a worthless contact
        cost = np.hstack([np.where(np.isfinite(G), -G, 1e9), np.zeros((len(free), len(free)))])
        rows, cols = linear_sum_assignment(cost)
        out: dict[str, str | None] = {m.id: None for m in free}
        for i, j in zip(rows, cols):
            if j < len(boats) and np.isfinite(G[i, j]) and G[i, j] > 0:
                out[free[i].id] = boats[j].id
        return out
