"""Truth-free cue extraction for the MDP.

A ContactTracker is fed one Observation per tick and keeps a short history per
contact. Every WINDOW_TICKS samples it emits a WindowFeatures summary for that
contact. Everything here is computed from what the Observation exposes — boat
positions, speed and heading AS MEASURED by the team's sensor track (noisier at
range, absent beyond sensor range), and our own team's positions — so it can run
inside the MDP at decision time without ever touching ground truth. Ticks where a
contact is beyond sensor range contribute no evidence.

Cues per window (all observable):
  speed        mean observed speed, knots
  closing      mean cos(angle between the boat's heading and the direction to the
               nearest USV): +1 heading straight at us, -1 heading straight away
  turn         mean absolute heading change per tick, degrees
  straightness net displacement / path length (1 = dead straight)
  dist_ratio   mean distance to the nearest USV, in units of the ID range
  band         "near" (<= 1 ID range), "mid" (<= 2), "far" (beyond)
  camera       the newest FRESH camera look during the window (a type name,
               possibly wrong), or None — each look is used at most once
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..identification import IDENTIFY_RANGE
from ..observation import Observation
from ..types import Vec2

WINDOW_TICKS = 10          # 10 ticks = 10 simulated seconds at the runtime tick
NEAR_RATIO = 1.0           # band edges, in multiples of the ID range
MID_RATIO = 2.0
BANDS = ("near", "mid", "far")


def id_range_for_width(grid_width: float) -> float:
    """The ID range for a map of this width (same scaling rule the engine uses:
    Small x1, Medium x2, Large x3). The range is a game rule, not ground truth."""
    return IDENTIFY_RANGE * max(1.0, round(grid_width / 40.0))


def band_for(dist_ratio: float) -> str:
    if dist_ratio <= NEAR_RATIO:
        return "near"
    if dist_ratio <= MID_RATIO:
        return "mid"
    return "far"


@dataclass
class WindowFeatures:
    boat_id: str
    t_end: float
    speed: float
    closing: float
    turn: float
    straightness: float
    dist_ratio: float
    band: str
    camera: str | None = None


@dataclass
class _Sample:
    pos: Vec2
    heading: Vec2
    speed: float
    closing: float
    dist: float


@dataclass
class _Track:
    buf: list[_Sample] = field(default_factory=list)
    last_heading: Vec2 | None = None
    speed_sum: float = 0.0
    speed_n: int = 0
    windows: list[WindowFeatures] = field(default_factory=list)
    used_look_id: int = 0                 # last camera look already counted
    pending_look: tuple[int, str] | None = None


def _angle_deg(a: Vec2, b: Vec2) -> float:
    la, lb = math.hypot(a.x, a.y), math.hypot(b.x, b.y)
    if la < 1e-9 or lb < 1e-9:
        return 0.0
    c = (a.x * b.x + a.y * b.y) / (la * lb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class ContactTracker:
    """Per-contact cue history, built only from Observations."""

    def __init__(self, window_ticks: int = WINDOW_TICKS) -> None:
        self.window_ticks = window_ticks
        self._tracks: dict[str, _Track] = {}
        self._last_t: float | None = None

    # ------------------------------------------------------------------ #
    def update(self, obs: Observation) -> list[WindowFeatures]:
        """Ingest one tick of observation. Idempotent per timestamp, so several
        USVs sharing one tracker can all call it in the same tick. Returns the
        windows completed by this update."""
        if self._last_t is not None and abs(obs.now_min - self._last_t) < 1e-12:
            return []
        self._last_t = obs.now_min
        idr = id_range_for_width(obs.grid_width)
        team = [obs.me.position] + [t.position for t in obs.team]
        done: list[WindowFeatures] = []
        for b in obs.boats:
            if b.left_grid or b.identified:
                continue  # only contacts still awaiting a decision are tracked
            if b.heading is None or b.speed_knots is None:
                continue  # beyond sensor range: position only, no evidence this tick
            # nearest USV and the boat's closing tendency toward it
            nearest, dist = None, float("inf")
            for p in team:
                d = b.position.distance_to(p)
                if d < dist:
                    nearest, dist = p, d
            closing = 0.0
            if nearest is not None and dist > 1e-6:
                to_usv = Vec2(x=(nearest.x - b.position.x) / dist,
                              y=(nearest.y - b.position.y) / dist)
                closing = b.heading.x * to_usv.x + b.heading.y * to_usv.y
            tr = self._tracks.setdefault(b.id, _Track())
            if b.camera_report is not None and b.camera_look_id > tr.used_look_id:
                tr.pending_look = (b.camera_look_id, b.camera_report)
            tr.buf.append(_Sample(b.position, b.heading, b.speed_knots, closing, dist))
            tr.speed_sum += b.speed_knots
            tr.speed_n += 1
            if len(tr.buf) >= self.window_ticks:
                wf = self._summarize(b.id, obs.now_min, tr, idr)
                tr.windows.append(wf)
                done.append(wf)
                tr.buf = []
        return done

    def _summarize(self, boat_id: str, t: float, tr: _Track, idr: float) -> WindowFeatures:
        buf = tr.buf
        n = len(buf)
        turns = []
        prev = tr.last_heading
        for s in buf:
            if prev is not None:
                turns.append(_angle_deg(prev, s.heading))
            prev = s.heading
        tr.last_heading = buf[-1].heading
        path = sum(buf[i].pos.distance_to(buf[i - 1].pos) for i in range(1, n))
        net = buf[-1].pos.distance_to(buf[0].pos)
        straight = 1.0 if path < 1e-9 else min(1.0, net / path)
        dist_ratio = (sum(s.dist for s in buf) / n) / idr
        camera = None
        if tr.pending_look is not None:
            tr.used_look_id, camera = tr.pending_look
            tr.pending_look = None
        return WindowFeatures(
            boat_id=boat_id,
            t_end=t,
            speed=sum(s.speed for s in buf) / n,
            closing=sum(s.closing for s in buf) / n,
            turn=(sum(turns) / len(turns)) if turns else 0.0,
            straightness=straight,
            dist_ratio=dist_ratio,
            band=band_for(dist_ratio),
            camera=camera,
        )

    # ------------------------------------------------------------------ #
    def mean_speed(self, boat_id: str) -> float | None:
        tr = self._tracks.get(boat_id)
        return None if tr is None or tr.speed_n == 0 else tr.speed_sum / tr.speed_n

    def windows(self, boat_id: str) -> list[WindowFeatures]:
        tr = self._tracks.get(boat_id)
        return [] if tr is None else list(tr.windows)

    def forget(self, boat_id: str) -> None:
        self._tracks.pop(boat_id, None)
