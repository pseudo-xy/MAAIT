"""Range-dependent sensing: what an AGENT can measure about a contact.

The team keeps one fused "track" per boat (as a real radar / EO tracker would).
Its errors grow with r = (distance from the boat to the nearest active USV) /
(ID range), and they are temporally correlated (AR(1)), like a filtered track:

  position error  std  POS_K   * r * ID range      (rho = POS_RHO)
  heading error   std  HDG_0 + HDG_K * r  degrees  (rho = HDG_RHO)
  speed error     std  SPD_0 + SPD_K * r  fraction (rho = SPD_RHO)
  beyond r > SENSOR_RANGE_RATIO: no heading / speed at all (noisy position only)

Track SMOOTHING (the main effect at range): the reported heading and speed are
an exponential moving average of the truth with weight
    alpha(r) = 1 inside ID range, falling linearly to SMOOTH_MIN by SMOOTH_FULL_R,
as a long-range tracker with sparse updates / heavy filtering would report. Fine
maneuver detail (a zigzag weave, a civilian's direction changes) disappears at
range; persistent cues (speed, moving away from us) survive.

Camera classification (World.visual_cue, on by default): within
VISUAL_RANGE_RATIO ID ranges (the camera range used by the feeds, 1.5x), the
camera produces a fresh LOOK every VISUAL_REFRESH_TICKS ticks: it names one of
the five contact types, correctly with probability
    p(r) = VIS_P_MAX - VIS_P_SLOPE * r   (floored at VIS_P_MIN),
otherwise a wrong type, preferring look-alikes (similar-speed types). Between
looks the last report is held; out of camera range there is no report. The best
looks come from up close — inside an aggressor's attack distance.

Everything scales with World.sensor_noise (0 = perfect sensing, 1 = default).

Why position is noisy too: with exact positions an agent could difference them
over time and recover heading and speed, bypassing the noise.

SCOPE: only the agent's Observation uses these tracks. Physics, identification
range checks, elimination, and the operator's map all use TRUE positions.
"""

from __future__ import annotations

import math
import random

from .identification import identify_range
from .models import Boat, World
from .types import Vec2

SENSOR_RANGE_RATIO = 5.0    # beyond this many ID ranges: no kinematics
POS_K, POS_RHO = 0.10, 0.98
HDG_0, HDG_K, HDG_RHO = 5.0, 15.0, 0.98
SPD_0, SPD_K, SPD_RHO = 0.03, 0.07, 0.98
SMOOTH_MIN, SMOOTH_FULL_R = 0.08, 4.0   # heavy filtering from 4 ID ranges out


VISUAL_RANGE_RATIO = 1.5    # camera range, in ID ranges (matches the camera feeds)
VISUAL_REFRESH_TICKS = 10   # a fresh, independent look every 10 s
VIS_P_MAX, VIS_P_SLOPE, VIS_P_MIN = 0.95, 0.30, 0.40
CONTACT_TYPES = ("civilian", "friendly", "standoff_low", "standoff_med", "aggressive")
# look-alike groups for camera mistakes: slow boats vs. fast boats
_LOOKALIKE = {"civilian": "slow", "standoff_low": "slow",
              "friendly": "fast", "standoff_med": "fast", "aggressive": "fast"}


def contact_type(boat: Boat) -> str:
    """The boat's TRUE fine-grained type (world side — never given to agents)."""
    from .types import Behavior
    if boat.behavior is Behavior.CIVILIAN:
        return "civilian"
    if boat.behavior is Behavior.FRIENDLY:
        return "friendly"
    if boat.behavior is Behavior.AGGRESSIVE:
        return "aggressive"
    return "standoff_low" if boat.nominal_speed_knots() <= 4.0 else "standoff_med"


def visual_accuracy(r: float, noise: float = 1.0) -> float:
    """Probability a camera look names the right type at range ratio r."""
    p = VIS_P_MAX - VIS_P_SLOPE * r * noise
    return max(VIS_P_MIN, min(VIS_P_MAX, p))


def _camera_look(boat: Boat, r: float, noise: float, rng: random.Random) -> str:
    truth = contact_type(boat)
    if rng.random() < visual_accuracy(r, noise):
        return truth
    others = [t for t in CONTACT_TYPES if t != truth]
    weights = [2.0 if _LOOKALIKE[t] == _LOOKALIKE[truth] else 1.0 for t in others]
    return rng.choices(others, weights=weights)[0]


def smoothing_alpha(r: float, noise: float = 1.0) -> float:
    """Track filter weight on the newest measurement: 1 = raw, small = heavy."""
    if r <= 1.0:
        return 1.0
    a = 1.0 - (1.0 - SMOOTH_MIN) * min(1.0, (r - 1.0) / (SMOOTH_FULL_R - 1.0)) * min(1.0, noise)
    return max(SMOOTH_MIN, a)


def range_ratio(boat: Boat, world: World) -> float:
    """True distance from the boat to the nearest active USV, in ID ranges."""
    idr = identify_range(world.grid)
    live = [u for u in world.usvs if not u.left_grid and not u.eliminated]
    if not live:
        return float("inf")
    return min(boat.position.distance_to(u.position) for u in live) / idr


def _ar1(prev: float, rho: float, sd: float, rng: random.Random) -> float:
    return rho * prev + math.sqrt(1.0 - rho * rho) * rng.gauss(0.0, sd)


def update_tracks(world: World, rng: random.Random) -> None:
    """Advance every boat's track errors one tick (no-op when sensing is perfect)."""
    k = world.sensor_noise
    if k <= 0:
        return
    idr = identify_range(world.grid)
    for b in world.boats:
        if b.left_grid:
            continue
        r = min(range_ratio(b, world), SENSOR_RANGE_RATIO)
        b.track_range_ratio = range_ratio(b, world)
        sd_pos = k * POS_K * r * idr
        b.track_pos_err = Vec2(x=_ar1(b.track_pos_err.x, POS_RHO, sd_pos, rng),
                               y=_ar1(b.track_pos_err.y, POS_RHO, sd_pos, rng))
        b.track_hdg_err_deg = _ar1(b.track_hdg_err_deg, HDG_RHO, k * (HDG_0 + HDG_K * r), rng)
        b.track_spd_err = _ar1(b.track_spd_err, SPD_RHO, k * (SPD_0 + SPD_K * r), rng)
        # range-dependent smoothing of the reported heading and speed
        a = smoothing_alpha(r, k)
        h = b.heading.normalized()
        if b.track_heading is None:
            b.track_heading, b.track_speed = h, b.speed_knots()
        else:
            sx = (1 - a) * b.track_heading.x + a * h.x
            sy = (1 - a) * b.track_heading.y + a * h.y
            n = math.hypot(sx, sy)
            b.track_heading = Vec2(x=sx / n, y=sy / n) if n > 1e-9 else h
            b.track_speed = (1 - a) * b.track_speed + a * b.speed_knots()
        # camera: a fresh look every VISUAL_REFRESH_TICKS while in camera range
        rr = b.track_range_ratio
        if world.visual_cue and rr <= VISUAL_RANGE_RATIO:
            b.visual_tick += 1
            if b.visual_report is None or b.visual_tick >= VISUAL_REFRESH_TICKS:
                b.visual_report = _camera_look(b, rr, k, rng)
                b.visual_tick = 0
                b.visual_look_id += 1
        else:
            b.visual_report = None
            b.visual_tick = 0


def camera_report(boat: Boat, world: World) -> tuple[str | None, int]:
    """(current camera look, its id). Look is None out of camera range / off.
    The id lets an agent tell a FRESH look from the one being held."""
    if world.sensor_noise <= 0 or not world.visual_cue:
        return None, 0
    return boat.visual_report, boat.visual_look_id


def sensed(boat: Boat, world: World) -> tuple[Vec2, Vec2 | None, float | None]:
    """(position, heading, speed) as the agent measures them. Heading/speed are
    None when the boat is beyond sensor range. Exact when sensing is perfect."""
    if world.sensor_noise <= 0:
        return boat.position, boat.heading.normalized(), boat.speed_knots()
    pos = Vec2(x=boat.position.x + boat.track_pos_err.x,
               y=boat.position.y + boat.track_pos_err.y)
    if boat.track_range_ratio > SENSOR_RANGE_RATIO:
        return pos, None, None
    h = boat.track_heading or boat.heading.normalized()
    a = math.radians(boat.track_hdg_err_deg)
    heading = Vec2(x=h.x * math.cos(a) - h.y * math.sin(a),
                   y=h.x * math.sin(a) + h.y * math.cos(a))
    base = boat.track_speed if boat.track_speed is not None else boat.speed_knots()
    speed = max(0.0, base * (1.0 + boat.track_spd_err))
    return pos, heading, speed
