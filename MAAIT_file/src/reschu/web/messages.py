"""Operator ⇄ USV messaging for the message box.

Two streams:
  * INCOMING — auto-generated status reports from the USVs, flowing continuously
    (each USV periodically reports its real state: fuel, comms, what it's doing).
  * OUTGOING replies — when the operator sends a message to a USV (or ALL USV),
    the addressed USV(s) reply based on their current state.

Messages are plain dicts so they serialise straight to JSON for the websocket:
    {id, sender, mission_min, text}
`sender` is a USV id ("USV-1"), "Operator", "Agent", or "BGC".
"""
from __future__ import annotations

import itertools
import random

# monotonically increasing message id so the client can de-dupe / order
_id_counter = itertools.count(1)


def _next_id() -> int:
    return next(_id_counter)


def _usv_status_phrase(usv, world) -> str:
    """A short status line describing what a USV is doing right now, truthfully
    from its live state (no ground truth about boats — only what it perceives)."""
    fuel = int(round(usv.fuel_pct))
    comms = {"in_range": "comms strong",
             "near_edge": "comms near edge",
             "out_of_range": "comms lost"}.get(usv.comms_state, "comms ok")

    if usv.left_grid:
        return "Returned to base, offline."
    if usv.rtb:
        return f"Returning to base. Fuel {fuel}%."

    # what is it doing?
    if usv.circling_since_min is not None:
        doing = "Circling a hostile contact"
    elif usv.assigned_target_id is not None:
        b = world.boat(usv.assigned_target_id)
        if b is not None and b.identified and b.perceived_class in ("friendly", "civilian"):
            doing = f"Escorting a {b.perceived_class} contact"
        elif b is not None:
            doing = "Moving to investigate a contact"
        else:
            doing = "On patrol"
    else:
        doing = "On patrol, no contacts"

    return f"{doing}. Fuel {fuel}%, {comms}."


def make_incoming(usv, world) -> dict:
    """One auto status report from a USV, reflecting its live state."""
    return {
        "id": _next_id(),
        "sender": usv.id,
        "mission_min": round(world.mission_time_min, 2),
        "kind": "incoming",
        "text": _usv_status_phrase(usv, world),
    }


def operator_message(text: str, to: str, world) -> dict:
    """Record the operator's outgoing message (shown in the thread)."""
    return {
        "id": _next_id(),
        "sender": "Operator",
        "kind": "thread",
        "to": to,
        "mission_min": round(world.mission_time_min, 2),
        "text": text,
    }


def _reply_phrase(usv, world, prompt: str, rng: random.Random) -> str:
    """A USV's reply to an operator message. For now it's a status-flavoured
    acknowledgement (the real AI/BGC replies come later)."""
    status = _usv_status_phrase(usv, world)
    openers = ["Copy.", "Acknowledged.", "Roger.", "Wilco.", "Understood."]
    return f"{rng.choice(openers)} {status}"


def replies_to(text: str, to: str, world, rng: random.Random) -> list[dict]:
    """Replies triggered by an operator message. `to` is a USV id, 'ALL USV',
    'Agent', or 'BGC'. Only USVs reply for now."""
    out: list[dict] = []
    def reachable(u):
        return (u is not None and not u.left_grid
                and u.comms_state.value != "out_of_range")
    if to == "ALL USV":
        targets = [u for u in world.usvs if reachable(u)]
    elif to in ("Agent", "BGC"):
        # not wired to real AI/BGC yet — send a simple placeholder ack so the
        # channel works and the operator sees a response.
        ack = ("Agent online. Monitoring all tracks."
               if to == "Agent" else "BGC copy. Standing by.")
        return [{
            "id": _next_id(),
            "kind": "thread",
            "sender": to,
            "mission_min": round(world.mission_time_min, 2),
            "text": ack,
        }]
    else:
        u = world.usv(to)
        targets = [u] if reachable(u) else []

    for u in targets:
        out.append({
            "id": _next_id(),
            "kind": "thread",
            "sender": u.id,
            "mission_min": round(world.mission_time_min, 2),
            "text": _reply_phrase(u, world, text, rng),
        })
    return out
