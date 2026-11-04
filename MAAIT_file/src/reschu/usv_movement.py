"""USV movement — the vehicle-side counterpart to `behaviors.py` (which moves
boats). Built in stages:

  * Stage 1: travel toward a target point at effective speed, stopping on arrival
  * Stage 2: agent USVs seek the nearest unidentified boat
  * Stage 3: after identification, circle enemies / escort friendlies-civilians
  * Assignment: agents work their assigned boat (see assignment.py)
  * Manual: straight-line travel, or an operator interception course that stops
    at identification range (see interception.py)

Speed uses the same knots->grid-units conversion as boats (UNITS_PER_KNOT_MIN),
so USVs and boats share one consistent scale.
"""

from __future__ import annotations

from .behaviors import UNITS_PER_KNOT_MIN
from .identification import identify_range
from .models import USV, World
from .types import Vec2


def usv_step_distance(usv: USV, dt_min: float, speed_knots: float | None = None) -> float:
    """How far (grid units) a USV travels in one tick. Uses the USV's effective
    speed unless `speed_knots` is given (e.g. when circling/escorting, the USV
    matches the boat's speed)."""
    kn = speed_knots if speed_knots is not None else usv.effective_speed_knots()
    return kn * UNITS_PER_KNOT_MIN * dt_min


# Wave-induced sway: as a USV travels it rides the swell, weaving gently around
# the straight line to its target rather than tracking it dead-on. AMP is the
# peak lateral offset from the line (grid units); RATE sets how quickly it weaves.
WAVE_SWAY_AMP = 1.6
WAVE_SWAY_RATE = 16.0

# Per-USV sway phase (radians), kept here because the USV model doesn't persist
# ad-hoc attributes. Keyed by USV id.
_WAVE_PHASE: dict[str, float] = {}
# Last tick's wave offset per USV, so we can apply only the delta (no drift).
_WAVE_PREV: dict[str, Vec2] = {}


def reset_motion_state() -> None:
    """Forget per-USV sway state. Called when a new engine (mission) starts, so a
    mission's motion never depends on missions run earlier in the same process."""
    _WAVE_PHASE.clear()
    _WAVE_PREV.clear()


def _wave_perp_offset(usv: USV, heading: Vec2, dt_min: float) -> Vec2:
    """A fixed-amplitude sideways offset (perpendicular to heading) that
    oscillates over time — the USV weaves around its ideal track by at most
    WAVE_SWAY_AMP cells, instead of drifting. Each USV keeps its own phase so
    they don't sway in lockstep."""
    import math
    phase = _WAVE_PHASE.get(usv.id)
    if phase is None:
        # deterministic per-id seed (Python's hash() is randomised per process,
        # which would make runs non-reproducible) so different USVs start out of
        # sync but the same USV id always starts at the same phase.
        import zlib
        seed = zlib.crc32(usv.id.encode()) % 1000
        phase = seed / 1000.0 * 2 * math.pi
    phase += WAVE_SWAY_RATE * dt_min
    _WAVE_PHASE[usv.id] = phase
    perp = Vec2(x=-heading.y, y=heading.x)
    return perp * (math.sin(phase) * WAVE_SWAY_AMP)


def move_towards(usv: USV, target: Vec2, world: World, dt_min: float,
                 speed_knots: float | None = None) -> Vec2:
    """Advance a USV toward `target`, clamped to the grid. If the remaining
    distance is within one step, snap exactly onto the target (arrived) instead
    of overshooting. Also updates the USV heading to face travel direction. While
    travelling, a small wave-induced sway is added so the path isn't dead
    straight (it settles exactly on the target when arriving). `speed_knots`, if
    given, overrides the USV's own speed (to match a boat when circling/escorting)."""
    to_target = target - usv.position
    dist = to_target.length()
    if dist == 0:
        return usv.position

    step = usv_step_distance(usv, dt_min, speed_knots)
    usv.heading = to_target.normalized()

    if dist <= step:
        _WAVE_PREV.pop(usv.id, None)      # arrived: reset sway
        return world.grid.clamp(target)   # settle exactly on target (no sway)

    # base movement straight toward the target (this self-corrects toward the
    # ideal line each tick). Add only the CHANGE in the wave offset since last
    # tick, so the USV's lateral position equals the current wave value rather
    # than accumulating drift.
    cur = _wave_perp_offset(usv, usv.heading, dt_min)
    prev = _WAVE_PREV.get(usv.id, Vec2(x=0.0, y=0.0))
    _WAVE_PREV[usv.id] = cur
    nxt = usv.position + usv.heading * step + (cur - prev)
    return world.grid.clamp(nxt)


def has_arrived(usv: USV, target: Vec2, world: World, dt_min: float) -> bool:
    """True if the USV is at (or within one step of) the target point."""
    return usv.position.distance_to(target) <= usv_step_distance(usv, dt_min)


# How far outside a boat's danger zone an avoiding USV tries to stay (buffer).
AVOID_BUFFER = 1.0


def move_towards_avoiding(
    usv: USV, target: Vec2, world: World, dt_min: float, exclude_boat_id=None
) -> Vec2:
    """Advance a USV toward `target` while steering around the danger zones of
    OTHER boats (every contact is treated as potentially dangerous). The boat the
    USV is engaging (`exclude_boat_id`) is not avoided — the USV must be able to
    close on its own target. If the direct step would enter a danger zone, the
    heading is deflected tangentially around the nearest threatening zone.

    Truth-free: this uses only positions and the fixed danger radius, never a
    boat's true class. Used for AGENT transit; the manual USV does not auto-avoid.
    """
    from .engine import danger_zone_radius  # local import to avoid a cycle
    to_target = target - usv.position
    dist = to_target.length()
    if dist == 0:
        return usv.position
    step = usv_step_distance(usv, dt_min)
    desired = to_target.normalized()

    # find the nearest danger zone that the intended step would put us inside
    avoid_r = danger_zone_radius(world.grid) + AVOID_BUFFER
    next_pos = usv.position + desired * step
    threat = None
    threat_d = None
    for b in world.boats:
        if b.left_grid or b.id == exclude_boat_id:
            continue
        d = next_pos.distance_to(b.position)
        if d <= avoid_r and (threat_d is None or d < threat_d):
            threat, threat_d = b, d

    if threat is not None:
        # deflect: steer along the tangent around the threat, on the side that
        # keeps progress toward the target
        away = (usv.position - threat.position)
        if away.length() == 0:
            away = Vec2(x=1.0, y=0.0)
        away = away.normalized()
        # two tangent directions (perpendicular to 'away'); pick the one more
        # aligned with the desired travel direction
        t1 = Vec2(x=-away.y, y=away.x)
        t2 = Vec2(x=away.y, y=-away.x)
        tangent = t1 if (t1.x*desired.x + t1.y*desired.y) >= (t2.x*desired.x + t2.y*desired.y) else t2
        # blend a little outward push so we don't graze the zone
        steer = (tangent * 0.85 + away * 0.15)
        steer = steer.normalized() if steer.length() > 0 else tangent
        usv.heading = steer
        return world.grid.clamp(usv.position + steer * step)

    # clear path: go straight
    usv.heading = desired
    if dist <= step:
        return world.grid.clamp(target)
    return world.grid.clamp(usv.position + desired * step)


# --------------------------------------------------------------------------- #
# Stage 2: agent seek-and-identify.
# An AGENT USV, each tick, heads toward the nearest boat it has not yet
# identified. Auto-identification (handled in identification.py) fires when it
# gets within IDENTIFY_RANGE. Once every boat is identified there is nothing to
# seek, so the agent holds position.
# --------------------------------------------------------------------------- #
def nearest_unknown_boat(usv: USV, world: World):
    """Return the closest boat this USV has not yet identified, or None if all
    boats are already identified."""
    unknown = [b for b in world.boats if not b.identified]
    if not unknown:
        return None
    return min(unknown, key=lambda b: usv.position.distance_to(b.position))


def agent_seek_step(usv: USV, world: World, dt_min: float) -> Vec2:
    """Next position for an AGENT USV pursuing its objective: drive toward the
    nearest unidentified boat. Holds position if there is nothing left to find.

    (This does not perform identification — the engine calls auto_identify after
    moving, so a boat that comes within range this tick is identified this tick.)
    """
    target_boat = nearest_unknown_boat(usv, world)
    if target_boat is None:
        usv.assigned_target_id = None
        return usv.position
    usv.assigned_target_id = target_boat.id
    return move_towards(usv, target_boat.position, world, dt_min)


# --------------------------------------------------------------------------- #
# Stage 3: post-identification behavior.
# Once a boat is identified, a USV engaged with it either CIRCLES it (enemy) or
# ESCORTS it (friendly / civilian). Both are station-keeping moves relative to
# the (possibly moving) boat.
# --------------------------------------------------------------------------- #
import math

from .types import Classification, Perception

# Base (Small-map) circling radius in grid cells. The ACTUAL radius is this times
# grid.range_scale (Small x1 = 1.8, Medium x2 = 3.6, Large x3 = 5.4). It's set
# just OUTSIDE the scaled danger zone (base 1.5) so a circling USV isn't detected.
# Use circle_radius(world.grid) for the scaled value.
CIRCLE_RADIUS = 1.8
# Angular speed of the circling motion (radians per minute).
CIRCLE_RATE = 1.2
# How far behind/alongside an escorted boat the USV keeps station (base, Small).
# Scaled with the map like the circle radius so standoff is consistent.
ESCORT_OFFSET = 1.8
# How long a USV circles an enemy before the enemy is eliminated. Defined in REAL
# seconds (what the operator watches), converted to simulation minutes using the
# playback pacing. The slider runs the sim in real time at 1x (1 real sec = 1 sim
# sec = 1/60 sim-min), so 10 real seconds = 10 sim-seconds = 0.1667 sim-minutes.
CIRCLE_DURATION_REAL_SEC = 60.0
_SIM_MIN_PER_REAL_SEC = 1.0 / 60.0  # real-time at slider 1x
CIRCLE_DURATION_MIN = CIRCLE_DURATION_REAL_SEC * _SIM_MIN_PER_REAL_SEC  # = 1.0


def circle_radius(grid) -> float:
    """Scaled circling radius for the given grid (Small x1, Med x2, Lg x3)."""
    return CIRCLE_RADIUS * grid.range_scale


def escort_offset(grid) -> float:
    """Scaled escort standoff for the given grid."""
    return ESCORT_OFFSET * grid.range_scale


def circle_step(usv: USV, boat, world: World, dt_min: float) -> Vec2:
    """Circle an enemy boat: hold at the (scaled) circle radius and orbit around
    it. The USV's angle around the boat advances each tick, so it sweeps a ring
    around the contact even as the contact moves."""
    r = circle_radius(world.grid)
    # current angle from boat to USV
    offset = usv.position - boat.position
    angle = math.atan2(offset.y, offset.x)
    angle += CIRCLE_RATE * dt_min  # advance around the ring
    ring_point = Vec2(
        x=boat.position.x + r * math.cos(angle),
        y=boat.position.y + r * math.sin(angle),
    )
    # move toward the next point on the ring, matching the boat's own speed so
    # the USV circles at the pace of the contact rather than racing around it.
    return move_towards(usv, ring_point, world, dt_min,
                        speed_knots=boat.speed_knots())


def escort_step(usv: USV, boat, world: World, dt_min: float) -> Vec2:
    """Escort a friendly/civilian boat, keeping station to its SIDE or BEHIND —
    never ahead of it in its path (so the boat never has to pass through the
    USV). The USV rides just off the boat's quarter (abeam-and-slightly-astern)
    on whichever side it is already nearer to, and follows the boat as it travels.

    The escorted boat NEVER changes its own direction: it keeps its course and
    the USV forms up on it. We use the boat's stable course (`straight_heading`)
    when available so the escort tracks the overall direction of travel rather
    than the small per-step zigzag of the visible heading.
    """
    course = getattr(boat, "straight_heading", None) or boat.heading
    fwd = course.normalized()
    # left/right (port/starboard) unit vectors relative to the boat's course
    left = Vec2(x=-fwd.y, y=fwd.x)
    right = Vec2(x=fwd.y, y=-fwd.x)
    off = escort_offset(world.grid)

    # candidate stations: off the boat's quarter on each side — a little to the
    # side and a little BEHIND, so the USV is never in front of the boat.
    side_amt = off * 0.85
    back_amt = off * 0.5
    port = boat.position + left * side_amt - fwd * back_amt
    starboard = boat.position + right * side_amt - fwd * back_amt
    # pick the side the USV is already closer to (smooth, no swinging across)
    station = port if usv.position.distance_to(port) <= usv.position.distance_to(starboard) else starboard
    # match the escorted boat's speed so the USV keeps station instead of
    # overshooting ahead of it.
    return move_towards(usv, station, world, dt_min,
                        speed_knots=boat.speed_knots())


def _engage_step(usv: USV, boat, world: World, dt_min: float) -> Vec2:
    """Engage the boat according to what the USV PERCEIVES it to be: circle a
    perceived enemy, escort a perceived friendly/civilian. Acting on perception
    (not truth) is deliberate — a misidentified boat is treated as its wrong
    label, which is part of what makes misidentification consequential."""
    usv.assigned_target_id = boat.id
    if boat.perceived_class is Perception.ENEMY:
        return circle_step(usv, boat, world, dt_min)
    return escort_step(usv, boat, world, dt_min)


def _nearest_identified_enemy(usv: USV, world: World):
    enemies = [
        b for b in world.boats
        if b.identified and b.perceived_class is Perception.ENEMY
    ]
    if not enemies:
        return None
    return min(enemies, key=lambda b: usv.position.distance_to(b.position))


def _nearest_identified_friendly(usv: USV, world: World):
    # Only FRIENDLIES are escorted off the grid. Identified CIVILIANS are left
    # alone (no escort), so they are excluded here.
    others = [
        b for b in world.boats
        if b.identified and b.perceived_class is Perception.FRIENDLY
    ]
    if not others:
        return None
    return min(others, key=lambda b: usv.position.distance_to(b.position))


def agent_step(usv: USV, world: World, dt_min: float) -> Vec2:
    """Full AGENT decision for one tick, in objective priority order:

      1. If any boat is still unidentified -> go identify the nearest one.
      2. Else if there is an identified enemy -> circle the nearest one.
      3. Else if there is an identified friendly/civilian -> escort the nearest.
      4. Else -> hold position.
    """
    # 1) identify unknowns first
    if nearest_unknown_boat(usv, world) is not None:
        return agent_seek_step(usv, world, dt_min)

    # 2) circle enemies (higher priority than escorting)
    enemy = _nearest_identified_enemy(usv, world)
    if enemy is not None:
        return _engage_step(usv, enemy, world, dt_min)

    # 3) escort friendlies (civilians are left alone after ID)
    friendly = _nearest_identified_friendly(usv, world)
    if friendly is not None:
        return _engage_step(usv, friendly, world, dt_min)

    # 4) nothing to do
    usv.assigned_target_id = None
    return usv.position


# --------------------------------------------------------------------------- #
# Assignment-aware agent movement.
# Each agent USV is LINKED to one boat (see assignment.py). It heads to that
# boat to identify it, then engages it (circle if enemy, escort otherwise). This
# keeps USVs spread across boats — a USV sticks to its assigned boat rather than
# all piling onto the nearest one. A USV with no assignment holds.
# --------------------------------------------------------------------------- #
def assigned_agent_step(usv: USV, world: World, dt_min: float, now_min: float) -> Vec2:
    """One tick for an agent USV working its assigned boat, with engagement
    lifecycle:

      * not yet identified -> drive to it (identification happens in the engine).
      * perceived ENEMY    -> circle it for CIRCLE_DURATION_MIN, then release
                              (clear assignment) so the engine re-assigns.
      * perceived FRIENDLY/CIVILIAN -> mark it being escorted and keep station
                              until it leaves the grid, then release.
      * no assignment      -> hold (idle); the engine may assign one.

    Releasing = setting assigned_target_id to None; the engine's reassignment
    step then gives this USV a new boat by priority.
    """
    if usv.assigned_target_id is None:
        usv.circling_since_min = None
        return usv.position

    boat = world.boat(usv.assigned_target_id)
    if boat is None or boat.left_grid:
        _release(usv, boat, now_min)
        return usv.position

    # still to be identified -> approach it directly. USVs no longer steer around
    # aggressive boats: the intended risk is that a USV approaches to identify a
    # contact, and if it MISidentifies an aggressive boat it gets attacked. So we
    # drive straight at the target.
    if not boat.identified:
        usv.circling_since_min = None
        return move_towards(usv, boat.position, world, dt_min)

    # engaged: circle enemies (timed -> elimination), escort others (until exit)
    if boat.perceived_class is Perception.ENEMY:
        if usv.circling_since_min is None:
            usv.circling_since_min = now_min  # circle START
        elapsed = now_min - usv.circling_since_min
        if elapsed >= CIRCLE_DURATION_MIN:
            # circle END: log the window, then the enemy is ELIMINATED (it
            # disappears from the grid). The USV releases and is re-assigned.
            usv.circle_events.append(
                (boat.id, round(usv.circling_since_min, 3), round(now_min, 3))
            )
            boat.left_grid = True          # eliminated -> disappears
            boat.eliminated = True
            _release(usv, boat, now_min)
            return usv.position
        return circle_step(usv, boat, world, dt_min)
    elif boat.perceived_class is Perception.CIVILIAN:
        # civilians are left alone after identification — no escort. Release so
        # the USV goes to the next boat that actually needs work.
        _release(usv, boat, now_min)
        return usv.position
    else:
        # escort friendlies: mark the boat being escorted (record start once) and
        # keep station on it until it leaves the grid.
        if not boat.being_escorted and boat.escort_start_min is None:
            boat.escort_start_min = round(now_min, 3)  # escort START
        boat.being_escorted = True
        pos = escort_step(usv, boat, world, dt_min)
        # if the boat left the grid this tick, record the end time
        if boat.left_grid and boat.escort_end_min is None:
            boat.escort_end_min = round(now_min, 3)  # escort END (left grid)
        return pos


def _release(usv: USV, boat, now_min: float | None = None) -> None:
    """Finish an engagement: stop escorting the boat and clear the USV's link so
    it can be re-assigned. If a circle was in progress but not yet logged (early
    release), log it now."""
    if usv.circling_since_min is not None and boat is not None and now_min is not None:
        # log an interrupted circle window (release before the timer elapsed)
        already = usv.circle_events and usv.circle_events[-1][0] == boat.id and \
            abs(usv.circle_events[-1][2] - now_min) < 1e-9
        if not already:
            usv.circle_events.append(
                (boat.id, round(usv.circling_since_min, 3), round(now_min, 3))
            )
    if boat is not None and boat.being_escorted and not boat.left_grid:
        # released before exit -> let the boat resume normal behavior
        boat.being_escorted = False
    usv.assigned_target_id = None
    usv.circling_since_min = None


# --------------------------------------------------------------------------- #
# Manual USV movement: if the operator has committed an interception, steer to
# the intercept point and stop at identification range (the operator then clicks
# to identify). Otherwise travel in a straight line from the start.
# --------------------------------------------------------------------------- #
def manual_straight_step(usv: USV, world: World, dt_min: float) -> Vec2:
    # No operator heading set -> hold position (wait for an operator command)
    if usv.manual_heading is None:
        return usv.position
    direction = usv.manual_heading.normalized()
    if direction.length() == 0:
        return usv.position
    # reflect the heading inward at the grid edges so the USV keeps moving
    # (patrols) instead of freezing against a wall until the operator redirects it
    w, h = world.grid.width, world.grid.height
    m = 1.0
    hx, hy = direction.x, direction.y
    if (usv.position.x <= m and hx < 0) or (usv.position.x >= w - m and hx > 0):
        hx = -hx
    if (usv.position.y <= m and hy < 0) or (usv.position.y >= h - m and hy > 0):
        hy = -hy
    direction = Vec2(x=hx, y=hy)
    usv.manual_heading = direction   # persist the reflected heading
    usv.heading = direction
    step = usv_step_distance(usv, dt_min)
    return world.grid.clamp(usv.position + direction * step)


def manual_step(usv: USV, world: World, dt_min: float) -> Vec2:
    """One tick for a MANUAL USV.

    Priority:
      1. If the operator has drawn a destination point, travel toward it and
         clear it on arrival (so the USV then holds position).
      2. Else if an intercept target is set, steer toward that boat, stopping at
         identification range for the operator to identify.
      3. Else travel in a straight line along the manual heading.
    """
    # 1) operator-drawn destination
    if usv.manual_destination is not None:
        dest = usv.manual_destination
        if usv.position.distance_to(dest) <= 1e-6:
            usv.manual_destination = None      # arrived -> resume patrol heading
            return usv.position
        # set heading toward the destination so that, after arriving, the USV
        # continues along that operator-chosen direction (its new course)
        to_dest = (dest - usv.position)
        if to_dest.length() > 0:
            usv.manual_heading = to_dest.normalized()
        nxt = move_towards(usv, dest, world, dt_min)
        if nxt.distance_to(dest) <= 1e-6:
            usv.manual_destination = None      # arrived this tick
        return nxt

    # 2) intercept a specific boat
    if usv.intercept_target_id is not None:
        boat = world.boat(usv.intercept_target_id)
        if boat is not None:
            dist = usv.position.distance_to(boat.position)
            if dist <= identify_range(world.grid):
                return usv.position  # arrived at ID range: wait for operator click
            return move_towards(usv, boat.position, world, dt_min)
        usv.intercept_target_id = None  # target gone; fall through to straight line

    # 3) straight-line travel
    return manual_straight_step(usv, world, dt_min)


# --------------------------------------------------------------------------- #
# Collision separation
# --------------------------------------------------------------------------- #
# When two entities (boat-boat, USV-USV, or boat-USV) approach HEAD-ON and get
# within SEPARATION_DIST of each other, each turns SEPARATION_TURN_DEG to its
# RIGHT (starboard) so they pass port-to-port, the standard maritime rule. Only
# a genuine head-on / closing encounter triggers it (two vessels merely near each
# other while travelling the same way, or moving apart, are left alone).
#
# EXEMPT (being close is intentional, so they don't turn):
#   * a USV that is circling an enemy (it orbits at the circle radius),
#   * a USV that is escorting a friendly/civilian (it keeps station alongside),
#   * the MANUAL USV (the operator is responsible for avoiding overlaps).
SEPARATION_DIST = 0.5           # grid units: closer than this triggers the turn
SEPARATION_TURN_DEG = 10.0      # degrees each vessel turns to its right
# how "head-on" the approach must be to count: dot(headingA, headingB) < this
# (i.e. their headings oppose each other). -0.5 ≈ within 120° of head-on.
HEADON_DOT_MAX = -0.3

# Hard anti-overlap ("no-go bubble"): every entity carries an invisible no-go
# zone; no two entities may end a tick with centres closer than the bubble.
#
# IMPORTANT: on-screen markers are drawn at a fixed PIXEL size, but grid cells
# render smaller on bigger maps — so a marker spans MORE grid cells on the Large
# map than on the Small one. A fixed grid-unit bubble therefore covers the marker
# on Small but not on Large/Medium (where the icons visibly overlapped). So the
# bubble scales with the map (via grid.range_scale: Small x1, Medium x2, Large x3)
# to match the marker's grid size on each map. Use min_separation(grid).
#
# Applies to EVERY pair — including two exempt USVs (circling/escorting/manual) —
# so none can overlap. The one exception is a USV and the single boat it is
# actively circling/escorting, which is meant to stay close.
MIN_SEPARATION_BASE = 1.6       # grid units on the Small map (x1)
_SEPARATION_PASSES = 30         # relaxation iterations to fully resolve pile-ups


def min_separation(grid) -> float:
    """No-go bubble diameter (grid units) for this map, scaled so it always
    covers the fixed-pixel on-screen markers (bigger on larger maps)."""
    return MIN_SEPARATION_BASE * grid.range_scale


def _usv_is_exempt(usv: USV, world: World) -> bool:
    from .types import USVMode, Perception
    if usv.mode is USVMode.MANUAL:
        return True
    if usv.circling_since_min is not None:       # circling an enemy
        return True
    # escorting an identified friendly/civilian
    if usv.assigned_target_id is not None:
        tgt = world.boat(usv.assigned_target_id)
        if (tgt is not None and tgt.identified
                and tgt.perceived_class in (Perception.FRIENDLY, Perception.CIVILIAN)):
            return True
    return False


def _turn_right(heading: Vec2, deg: float) -> Vec2:
    """Rotate a heading `deg` degrees to its RIGHT (starboard). In these coords
    (x = right, y = up) a right/clockwise turn is a negative-angle rotation."""
    import math
    r = math.radians(-deg)   # clockwise = to the right
    c, s = math.cos(r), math.sin(r)
    return Vec2(x=heading.x * c - heading.y * s, y=heading.x * s + heading.y * c)


def _engaged_pair(ea, kind_a, eb, kind_b, grid) -> bool:
    """True if the two entities should be allowed to breach the no-go bubble.
    Only an AGGRESSIVE boat actively closing on a USV (within ID range) qualifies,
    so it can drive into the danger zone and attack.

    Circling and escorting are NOT exempt here: their standoff radius (circle /
    escort offset) is larger than the no-go bubble, so the USV naturally keeps
    clear of the boat — and keeping the bubble active stops the USV visually
    overlapping the boat during the moment it settles onto its station/ring.
    """
    from .types import Behavior
    from .identification import identify_range

    if kind_a == "usv" and kind_b == "boat":
        usv, boat = ea, eb
    elif kind_a == "boat" and kind_b == "usv":
        usv, boat = eb, ea
    else:
        return False

    if boat.behavior is Behavior.AGGRESSIVE:
        if usv.position.distance_to(boat.position) <= identify_range(grid):
            return True
    return False


def resolve_separation(world: World, dt_min: float) -> None:
    """Nudge any two non-exempt entities that are closer than SEPARATION_DIST
    apart: each veers SEPARATION_TURN_DEG away from the other (opposite senses),
    then steps a little along the veer so they physically separate. Operates on
    the entities' current positions after they've moved this tick."""
    # gather movable entities with an exempt flag and a per-entity speed
    ents = []
    for b in world.boats:
        if b.left_grid:
            continue
        ents.append(("boat", b, False))
    for u in world.usvs:
        if getattr(u, "stationary", False):
            continue
        ents.append(("usv", u, _usv_is_exempt(u, world)))

    n = len(ents)
    for i in range(n):
        kind_i, ei, exempt_i = ents[i]
        for j in range(i + 1, n):
            kind_j, ej, exempt_j = ents[j]
            if exempt_i and exempt_j:
                continue
            # skip a USV and the specific boat it is engaged with (circling or
            # escorting) — they are meant to be close, so neither turns.
            if _engaged_pair(ei, kind_i, ej, kind_j, world.grid):
                continue
            off = ej.position - ei.position
            d = off.length()
            if d == 0 or d >= SEPARATION_DIST:
                continue

            hi = getattr(ei, "heading", None)
            hj = getattr(ej, "heading", None)
            if hi is None or hj is None:
                continue
            hi = hi.normalized()
            hj = hj.normalized()

            # head-on test: their headings must oppose (dot < HEADON_DOT_MAX) AND
            # they must actually be closing on each other (each heading points
            # roughly toward the other). If not head-on, leave them (this rule is
            # only for head-to-head meetings).
            unit = off.normalized()
            closing_i = hi.x * unit.x + hi.y * unit.y > 0        # ei heads toward ej
            closing_j = hj.x * (-unit.x) + hj.y * (-unit.y) > 0  # ej heads toward ei
            head_on = (hi.x * hj.x + hi.y * hj.y) < HEADON_DOT_MAX
            if not (head_on and closing_i and closing_j):
                continue

            # both turn 10 degrees to their RIGHT (starboard) and step along the
            # new heading so they pass port-to-port. "Right" of heading (hx,hy)
            # in these coords (x right, y up) is a -90 deg rotation of the
            # heading; _turn_right applies the SEPARATION_TURN_DEG deflection.
            step = SEPARATION_DIST * 0.6
            if not exempt_i:
                ei.heading = _turn_right(hi, SEPARATION_TURN_DEG)
                ei.position = world.grid.clamp(ei.position + ei.heading * step)
            if not exempt_j:
                ej.heading = _turn_right(hj, SEPARATION_TURN_DEG)
                ej.position = world.grid.clamp(ej.position + ej.heading * step)

    # ---- hard anti-overlap guarantee ------------------------------------- #
    # The head-on rule above only handles nose-to-nose meetings. This pass
    # guarantees NOTHING overlaps: any two non-exempt entities still closer than
    # MIN_SEPARATION (overtaking, crossing, side-swipe, any angle) are pushed
    # directly apart to that distance. A few relaxation passes resolve chains
    # (three-way pile-ups) cleanly. Exempt entities (circling/escorting/manual)
    # hold their position; the other one moves the full amount to clear.
    _enforce_min_separation(ents, world)


def _enforce_min_separation(ents, world) -> None:
    # relax until no non-exempt pair overlaps (converges), capped for safety.
    minsep = min_separation(world.grid)
    for _ in range(_SEPARATION_PASSES):
        moved = False
        n = len(ents)
        for i in range(n):
            kind_i, ei, exempt_i = ents[i]
            for j in range(i + 1, n):
                kind_j, ej, exempt_j = ents[j]
                # NOTE: unlike the head-on pass we do NOT skip exempt-exempt
                # pairs here — two circling/escorting/manual USVs still must not
                # sit on top of each other. Only a genuinely engaged USV+its-boat
                # pair (which is meant to stay close) is skipped.
                if _engaged_pair(ei, kind_i, ej, kind_j, world.grid):
                    continue
                off = ej.position - ei.position
                d = off.length()
                if d >= minsep:
                    continue
                if d == 0:
                    unit = Vec2(x=1.0, y=0.0)
                    d = 0.0
                else:
                    unit = Vec2(x=off.x / d, y=off.y / d)
                overlap = minsep - d
                moved = True
                # both exempt (e.g. two circling USVs): split the push so neither
                # is favoured. One exempt + one not: the non-exempt one yields.
                if exempt_i and not exempt_j:
                    ej.position = world.grid.clamp(ej.position + unit * overlap)
                elif exempt_j and not exempt_i:
                    ei.position = world.grid.clamp(ei.position - unit * overlap)
                else:
                    half = overlap * 0.5
                    ei.position = world.grid.clamp(ei.position - unit * half)
                    ej.position = world.grid.clamp(ej.position + unit * half)
        if not moved:
            break


def rtb_step(usv: USV, world: World, dt_min: float) -> Vec2:
    """Return-to-base movement: head straight for the nearest point on the grid
    boundary at USV speed and exit there. Once the engine sees the USV on the
    boundary it marks it left_grid."""
    x, y = usv.position.x, usv.position.y
    w, h = world.grid.width, world.grid.height
    edges = [
        Vec2(x=0.0, y=y),   # left
        Vec2(x=w, y=y),     # right
        Vec2(x=x, y=0.0),   # bottom
        Vec2(x=x, y=h),     # top
    ]
    target = min(edges, key=lambda p: usv.position.distance_to(p))
    return move_towards(usv, target, world, dt_min)
