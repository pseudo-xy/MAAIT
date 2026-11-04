"""Per-decision log of the MDP teammate, for analysing human-trial data.

Two kinds of rows, both stamped with the policy version and fingerprint:

  decision  an engagement decision by one MDP USV about its contact (made every
            10 s and whenever the distance band changes, so a USV keeping
            station near a band edge re-decides often; `action_changed` = 1
            marks the rows where it actually changed what it was doing): distance
            band and distance, the full belief, the action, its value, the
            next-best action and the MARGIN between them (how close the call
            was), and a one-line plain-language reason;
  assign    the team assignment changed a USV's contact: from, to, and the
            assignment score of the new (and old) contact.

Everything logged during the mission is what the agent knew — no truth. Ground
truth is added ONLY when the log is exported after the mission (`rows(world)`):
the contact's true type and whether a declaration was correct, for analysis.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .cue_model import CLASSES, PERCEPTION_OF, CueModel
from .engagement import ACTIONS, BANDS
from .engagement import DECL_PERCEPTION

FIELDS = [
    "mission_min", "policy_version", "policy_fingerprint", "event", "usv_id", "boat_id",
    "band", "distance_id_ranges",
    *[f"p_{c}" for c in CLASSES], "p_decl_enemy", "p_decl_friendly", "p_decl_civilian",
    "action", "action_changed", "committed", "value_chosen", "next_best_action", "value_next_best", "margin",
    "reason", "from_boat", "to_boat", "score_new", "score_old",
    # added at export, from ground truth (analysis only)
    "true_type", "declaration_correct",
]

_VERB = {"approach": "close in", "hold": "hold distance", "back_off": "open distance",
         "declare_enemy": "commit: ENEMY", "declare_friendly": "commit: FRIENDLY",
         "declare_civilian": "commit: CIVILIAN", "abandon": "give up on it"}


def _r(x, k=4):
    return "" if x is None else round(float(x), k)


class DecisionLog:
    def __init__(self, version: str = "", fingerprint: str = "") -> None:
        self.version, self.fingerprint = version, fingerprint
        self._rows: list[dict] = []
        self._last: dict[tuple, str] = {}       # (usv, boat) -> last action

    def clear(self) -> None:
        self._rows = []
        self._last = {}

    def __len__(self) -> int:
        return len(self._rows)

    def _base(self, now_min: float, event: str, usv_id: str) -> dict:
        row = dict.fromkeys(FIELDS, "")
        row.update(mission_min=round(now_min, 4), policy_version=self.version,
                   policy_fingerprint=self.fingerprint, event=event, usv_id=usv_id)
        return row

    def decision(self, now_min: float, usv_id: str, boat_id: str, band: int,
                 dist_ratio: float, post: dict, action: str, q_row: list | None) -> dict:
        row = self._base(now_min, "decision", usv_id)
        dp = CueModel.declaration_probs(post)
        pe, pf, pc = (dp.get(DECL_PERCEPTION[d], 0.0) for d in ("enemy", "friendly", "civilian"))
        row.update(boat_id=boat_id, band=BANDS[band], distance_id_ranges=_r(dist_ratio, 3),
                   **{f"p_{c}": _r(post[c]) for c in CLASSES},
                   p_decl_enemy=_r(pe), p_decl_friendly=_r(pf), p_decl_civilian=_r(pc),
                   action=action, committed=int(action.startswith("declare")),
                   action_changed=int(self._last.get((usv_id, boat_id)) != action))
        self._last[(usv_id, boat_id)] = action
        best_other, qo = None, None
        if q_row:
            qa = q_row[ACTIONS.index(action)]
            for a, q in zip(ACTIONS, q_row):
                if a != action and q is not None and (qo is None or q > qo):
                    best_other, qo = a, q
            row.update(value_chosen=_r(qa, 3), next_best_action=best_other or "",
                       value_next_best=_r(qo, 3),
                       margin=_r(qa - qo, 3) if qa is not None and qo is not None else "")
        lead = max((("enemy", pe), ("friendly", pf), ("civilian", pc)), key=lambda t: t[1])
        reason = (f"{_VERB[action]}: {lead[1]:.0%} {lead[0]}, "
                  f"{post['aggressive']:.0%} aggressor, at {dist_ratio:.1f} ID ranges")
        if best_other is not None and row["margin"] != "":
            reason += f"; beats '{_VERB[best_other]}' by {row['margin']}"
        row["reason"] = reason
        self._rows.append(row)
        return row

    def assignment(self, now_min: float, usv_id: str, old: str | None, new: str | None,
                   score_new: float | None, score_old: float | None) -> dict:
        row = self._base(now_min, "assign", usv_id)
        row.update(boat_id=new or "", from_boat=old or "", to_boat=new or "",
                   score_new=_r(score_new, 3), score_old=_r(score_old, 3),
                   reason=(f"switch {old} -> {new}" if old and new else
                           f"take {new}" if new else f"release {old}; idle"))
        self._rows.append(row)
        return row

    # ------------------------------------------------------------------ #
    # export (after the mission): ground truth is joined in HERE only
    # ------------------------------------------------------------------ #
    def rows(self, world=None) -> list[dict]:
        out = [dict(r) for r in self._rows]
        if world is None:
            return out
        from ..sensing import contact_type
        types = {b.id: contact_type(b) for b in world.boats}
        for r in out:
            t = types.get(r["boat_id"])
            if t:
                r["true_type"] = t
            if r["event"] == "decision" and r["committed"] == 1 and t:
                declared = r["action"].split("_", 1)[1]
                r["declaration_correct"] = int(declared == PERCEPTION_OF[t].value)
        return out

    def write_csv(self, path: Path | str, world=None) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows(world))
