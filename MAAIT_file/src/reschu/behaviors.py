"""Autonomous movement strategies for boats — organised around each boat's
**end goal**, set in the scenario/config.

Every boat heads toward its end goal but still reacts as it goes; when it reaches
the goal it picks a new one by its type's rule and keeps going. The per-type
rules for choosing a goal:

  * aggressive enemy -> attack a USV: goal is the nearest USV's position, re-aimed
                        each tick (so it keeps chasing a moving USV)
  * standoff enemy   -> a grid in a far CORNER of the map
  * civilian         -> a grid in a straight line from where it started
  * friendly         -> a random grid

Speed uses the shared knots->grid-units conversion (UNITS_PER_KNOT_MIN), so boats
and USVs move on one consistent scale.
"""

from __future__ import annotations

import random

from .models import Boat, World
from .types import Behavior, Classification, Perception, Vec2

# ---- Movement pace ---------------------------------------------------------
# Boat/USV speed in "knots" is now just a RELATIVE speed (a 12-kn USV is faster
# than a 3-kn civilian); it is no longer tied to real nautical miles. The pace
# factor below controls how many grid cells one knot covers per simulation
# minute — turn it UP to make everything move faster, DOWN to slow it. It is
# tuned so the action is comfortably watchable at 1x (real time): a 12-kn USV
# crosses the 40-wide small map in ~1.7 min, the 80-wide medium in ~3.3 min, and
# the 120-wide large in ~5 min. The speed slider adjusts around this at run time.
UNITS_PER_KNOT_MIN = 2.0         # grid cells per knot per simulation minute


def _step_distance(boat: Boat, dt_min: float) -> float:
    return boat.speed_knots() * UNITS_PER_KNOT_MIN * dt_min


def _move_towards(boat: Boat, target: Vec2, dt_min: float, world: World) -> Vec2:
    to_target = target - boat.position
    dist = to_target.length()
    if dist == 0:
        return boat.position
    boat.heading = to_target.normalized()
    step = _step_distance(boat, dt_min)
    # Friendlies transit the map and must reach the exact edge to exit, so they
    # use the hard clamp. Every other boat (civilian / standoff / aggressive) is
    # kept a small margin off the edge so it never pins to the wall and leaves a
    # visual smudge there.
    _clamp = (world.grid.clamp if boat.behavior is Behavior.FRIENDLY
              else world.grid.clamp_inset)
    if dist <= step:
        return _clamp(target)   # arrived: land on the goal, no overshoot
    return _clamp(boat.position + boat.heading * step)


# --------------------------------------------------------------------------- #
# Goal-choosing rules, one per behavior. Each returns a fresh end-goal point.
# --------------------------------------------------------------------------- #
# Aggressive boats detect and attack a target (USV or friendly) once it comes
# within the USV identification range (see identify_range). Beyond that they
# can't detect it and just search (random wander). Using the ID range ties the
# threat directly to the identify mechanic: a USV that approaches to identify an
# aggressive boat is exactly at the distance where the boat can attack it, so the
# operator must identify it CORRECTLY (-> circle -> neutralised) before it closes.


def _aggressive_targets(boat: Boat, world: World) -> list:
    """Potential targets for an aggressive boat: all USVs plus all friendly boats
    still on the grid (a friendly is identified by its true behavior — this is
    engine-side movement logic, not the truth-free planner)."""
    targets = list(world.usvs)
    targets += [
        b for b in world.boats
        if b.behavior is Behavior.FRIENDLY and not b.left_grid and b.id != boat.id
    ]
    return targets


def _aggressive_sees_target(boat: Boat, world: World) -> bool:
    """True if a USV or friendly boat is within IDENTIFICATION range right now
    (i.e. the aggressive boat is attacking rather than seeking)."""
    from .identification import identify_range
    targets = _aggressive_targets(boat, world)
    if not targets:
        return False
    nearest = min(targets, key=lambda t: boat.position.distance_to(t.position))
    return boat.position.distance_to(nearest.position) <= identify_range(world.grid)


def _goal_aggressive(boat: Boat, world: World, rng: random.Random) -> Vec2:
    """Wander/seek by default (a search pattern), and lock on to attack ONLY when
    a USV comes within IDENTIFICATION range. The idea: an unidentified aggressive
    boat is a threat you must classify — if a USV gets close enough to ID it and
    IDs it correctly, that USV circles it (safe). If the USV is close enough to be
    in ID range but hasn't correctly identified it, the boat homes in and can
    eliminate it (see the engine's danger-zone check). The lock is re-evaluated
    every tick, so a target that pulls away is dropped and the boat resumes
    seeking."""
    from .identification import identify_range
    targets = _aggressive_targets(boat, world)
    if targets:
        nearest = min(targets, key=lambda t: boat.position.distance_to(t.position))
        if boat.position.distance_to(nearest.position) <= identify_range(world.grid):
            return nearest.position   # within ID range -> attack (home in)
    # nothing in ID range -> seek: wander randomly like a search pattern
    return _wander_goal(boat, world, rng)


def _wander_goal(boat: Boat, world: World, rng: random.Random) -> Vec2:
    """A random point a moderate distance away in any direction — used as a
    search pattern. Shared shape with the civilian random walk."""
    import math
    reach = 0.20 * (world.grid.width + world.grid.height)
    angle = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(0, reach)
    boat.heading = Vec2(x=math.cos(angle), y=math.sin(angle))
    return world.grid.clamp(
        Vec2(x=boat.position.x + math.cos(angle) * dist,
             y=boat.position.y + math.sin(angle) * dist)
    )


def _goal_standoff(boat: Boat, world: World, rng: random.Random) -> Vec2:
    """Stay as FAR as possible from every USV. Each refresh the boat samples a
    number of candidate points across the whole map (not just edges) and picks
    the one that maximises the distance to the NEAREST USV — i.e. the safest spot
    right now. As USVs move, the safest spot moves, so the boat keeps
    repositioning and never settles. It can be anywhere on the map, kept a small
    margin off the very edge so it doesn't pin itself to a wall."""
    w, h = world.grid.width, world.grid.height
    usvs = [u for u in world.usvs if not getattr(u, "eliminated", False)]
    margin = 0.04 * min(w, h)
    lo_x, hi_x = margin, w - margin
    lo_y, hi_y = margin, h - margin
    if not usvs:
        # no USVs to flee: just wander a little
        return Vec2(x=rng.uniform(lo_x, hi_x), y=rng.uniform(lo_y, hi_y))

    def min_usv_dist(p: Vec2) -> float:
        return min(p.distance_to(u.position) for u in usvs)

    # sample candidate points and keep the one furthest from the nearest USV.
    # bias sampling toward the boat's neighbourhood so movement is smooth rather
    # than teleport-like, but always include some fully-random points so it can
    # relocate across the map when USVs close in.
    best = None
    best_d = -1.0
    for _ in range(24):
        if rng.random() < 0.6:
            # a point a moderate step from the current position
            import math
            ang = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.1, 0.35) * (w + h) / 2
            cand = Vec2(x=boat.position.x + math.cos(ang) * r,
                        y=boat.position.y + math.sin(ang) * r)
        else:
            cand = Vec2(x=rng.uniform(lo_x, hi_x), y=rng.uniform(lo_y, hi_y))
        cand = Vec2(x=min(max(cand.x, lo_x), hi_x),
                    y=min(max(cand.y, lo_y), hi_y))
        d = min_usv_dist(cand)
        if d > best_d:
            best_d = d
            best = cand
    return best


def _goal_civilian(boat: Boat, world: World, rng: random.Random) -> Vec2:
    """Random walk that changes direction on a fixed cadence (see step_boat).
    Each pick is a point in a random direction a good distance away, kept inside
    an interior box so the boat ranges over a large area but never drives onto an
    edge and sticks there. Reflecting (not clamping) off the interior bounds is
    what keeps it off the walls."""
    import math
    w, h = world.grid.width, world.grid.height
    margin = 0.08 * min(w, h)
    reach = rng.uniform(0.25, 0.5) * (w + h) / 2   # cover a large area
    angle = rng.uniform(0, 2 * math.pi)
    boat.heading = Vec2(x=math.cos(angle), y=math.sin(angle))
    gx = boat.position.x + math.cos(angle) * reach
    gy = boat.position.y + math.sin(angle) * reach
    lo_x, hi_x = margin, w - margin
    lo_y, hi_y = margin, h - margin
    if gx < lo_x: gx = lo_x + (lo_x - gx)
    if gx > hi_x: gx = hi_x - (gx - hi_x)
    if gy < lo_y: gy = lo_y + (lo_y - gy)
    if gy > hi_y: gy = hi_y - (gy - hi_y)
    gx = min(max(gx, lo_x), hi_x)
    gy = min(max(gy, lo_y), hi_y)
    return Vec2(x=gx, y=gy)


# How much a friendly boat's heading may jitter each step (radians). Large enough
# that the zigzag is clearly visible, while the (unchanged) base course still
# carries it forward across the map so it doesn't just thrash in place.
FRIENDLY_JITTER_RAD = 0.9

# A civilian changes direction every this-many ticks (fixed cadence), so its path
# is a sequence of short legs in different random directions — obviously random
# movement rather than one long committed hop.
CIVILIAN_DIR_CHANGE_TICKS = 5

# Standoff enemies re-evaluate their "furthest from all USVs" goal every this-many
# ticks, so they keep repositioning as the USVs move (never settling).
STANDOFF_REPOSITION_TICKS = 5

# How close (grid units) a boat must get to its goal to count as "arrived" and
# pick a new one. Small but non-zero so it triggers at the fine runtime tick rate.
GOAL_ARRIVE_RADIUS = 1.0


def _bounce_off_walls(boat: Boat, world: World, margin: float = 2.0) -> None:
    """If the boat is at/near a grid edge, reflect its stored heading inward so
    it turns back rather than pushing into the wall and getting stuck."""
    import math
    h = boat.straight_heading
    if h is None:
        return
    hx, hy = h.x, h.y
    if boat.position.x <= margin and hx < 0:
        hx = -hx
    elif boat.position.x >= world.grid.width - margin and hx > 0:
        hx = -hx
    if boat.position.y <= margin and hy < 0:
        hy = -hy
    elif boat.position.y >= world.grid.height - margin and hy > 0:
        hy = -hy
    boat.straight_heading = Vec2(x=hx, y=hy)


def _goal_friendly(boat: Boat, world: World, rng: random.Random) -> Vec2:
    """Zigzag transit: the friendly travels along a stable base heading, weaving
    with a small random deviation each step (the zigzag) WITHOUT the deviations
    compounding — so it holds its overall course across the grid. It does NOT
    bounce off walls: it transits to the far edge and exits there (the engine
    then despawns it and spawns a fresh friendly entering from a random edge)."""
    import math
    if boat.straight_heading is None:
        if boat.exit_point is not None:
            # course aimed straight at the admin-set exit cell on the edge
            to_exit = boat.exit_point - boat.position
            if to_exit.length() > 1e-6:
                boat.straight_heading = to_exit.normalized()
            else:
                angle = rng.uniform(0, 2 * math.pi)
                boat.straight_heading = Vec2(x=math.cos(angle), y=math.sin(angle))
        else:
            angle = rng.uniform(0, 2 * math.pi)
            boat.straight_heading = Vec2(x=math.cos(angle), y=math.sin(angle))

    # jitter is a temporary deviation from the stable base heading (not persisted)
    base = boat.straight_heading
    base_angle = math.atan2(base.y, base.x)
    jittered = base_angle + rng.uniform(-FRIENDLY_JITTER_RAD, FRIENDLY_JITTER_RAD)
    step_dir = Vec2(x=math.cos(jittered), y=math.sin(jittered))
    boat.heading = step_dir  # visible heading weaves...
    # ...but boat.straight_heading (the course) is left unchanged, so the zigzag
    # oscillates around the course rather than wandering off it.

    reach = 0.15 * (world.grid.width + world.grid.height)
    return world.grid.clamp(
        Vec2(x=boat.position.x + step_dir.x * reach,
             y=boat.position.y + step_dir.y * reach)
    )


_GOAL_RULE = {
    Behavior.AGGRESSIVE: _goal_aggressive,
    Behavior.STANDOFF: _goal_standoff,
    Behavior.CIVILIAN: _goal_civilian,
    Behavior.FRIENDLY: _goal_friendly,
}


def choose_goal(boat: Boat, world: World, rng: random.Random) -> Vec2:
    """Pick a fresh end goal for this boat according to its behavior."""
    return _GOAL_RULE[boat.behavior](boat, world, rng)


# --------------------------------------------------------------------------- #
# Escort routing. When a boat is being escorted off the grid it abandons its
# normal behavior: it heads for the NEAREST edge, but steers away from enemy and
# unidentified boats, keeping at least ESCORT_DANGER_RADIUS from them.
# --------------------------------------------------------------------------- #
ESCORT_DANGER_RADIUS = 12.0  # keep at least this far from threats while exiting


def _nearest_edge_point(boat: Boat, world: World) -> Vec2:
    """The closest point on the grid boundary to the boat (shortest way out)."""
    x, y = boat.position.x, boat.position.y
    w, h = world.grid.width, world.grid.height
    # distance to each of the four edges
    candidates = [
        Vec2(x=0.0, y=y),   # left
        Vec2(x=w, y=y),     # right
        Vec2(x=x, y=0.0),   # bottom
        Vec2(x=x, y=h),     # top
    ]
    return min(candidates, key=lambda p: boat.position.distance_to(p))


def _threat_positions(boat: Boat, world: World) -> list[Vec2]:
    """Positions of boats to avoid while escorting: enemies (perceived or true)
    and any still-unidentified boat."""
    threats = []
    for other in world.boats:
        if other.id == boat.id:
            continue
        if other.left_grid:
            continue
        is_enemy = (
            other.perceived_class is Perception.ENEMY
            or other.true_class is Classification.ENEMY
        )
        if is_enemy or not other.identified:
            threats.append(other.position)
    return threats


def escort_exit_step(boat: Boat, world: World, dt_min: float) -> Vec2:
    """Move the escorted boat toward the nearest edge, pushed away from nearby
    threats. The exit direction is the straight line to the nearest edge; if any
    threat is within ESCORT_DANGER_RADIUS, add a repulsion vector away from it so
    the boat routes around it."""
    exit_point = _nearest_edge_point(boat, world)
    to_exit = (exit_point - boat.position).normalized()

    # sum repulsion from nearby threats
    repulsion = Vec2(x=0.0, y=0.0)
    for tp in _threat_positions(boat, world):
        away = boat.position - tp
        d = away.length()
        if 0 < d < ESCORT_DANGER_RADIUS:
            # stronger push the closer the threat (inverse of distance)
            strength = (ESCORT_DANGER_RADIUS - d) / ESCORT_DANGER_RADIUS
            repulsion = repulsion + away.normalized() * strength

    direction = (to_exit + repulsion).normalized()
    if direction.length() == 0:
        direction = to_exit
    boat.heading = direction
    step = _step_distance(boat, dt_min)
    return world.grid.clamp(boat.position + direction * step)


# --------------------------------------------------------------------------- #
# The single movement step, shared by all behaviors:
#   1. make sure the boat has a goal (pick one if not)
#   2. aggressive re-aims at the (moving) nearest USV every tick
#   3. move toward the goal
#   4. if we've arrived, pick a new goal by the type's rule
# --------------------------------------------------------------------------- #
def step_boat(boat: Boat, world: World, dt_min: float, rng: random.Random) -> Vec2:
    if boat.start_pos is None:
        boat.start_pos = boat.position

    # once a boat has left the grid it stays gone (no more movement)
    if boat.left_grid:
        return boat.position

    boat.tick_count += 1

    # --- escorted boats leave the map --------------------------------------- #
    # A FRIENDLY, once identified, is escorted out: it keeps its own transit
    # course and exits at the far edge naturally; the USV escorts alongside.
    # A CIVILIAN is NOT escorted — after identification it is simply left alone
    # and continues its normal random-walk behaviour (no USV is assigned to it).

    # Decide whether to refresh the goal this tick.
    #  - friendly (zigzag) recomputes every tick.
    #  - aggressive recomputes every tick WHILE ATTACKING (a seen target moves),
    #    but while SEEKING commits to a wander point until it arrives.
    #  - civilian changes direction on a fixed cadence (every N ticks).
    #  - standoff re-picks its "furthest from USVs" goal on a fixed cadence, so it
    #    keeps repositioning as the USVs move.
    refresh = boat.end_goal is None or boat.behavior is Behavior.FRIENDLY
    if boat.behavior is Behavior.AGGRESSIVE and _aggressive_sees_target(boat, world):
        refresh = True
    if (boat.behavior is Behavior.CIVILIAN
            and boat.tick_count % CIVILIAN_DIR_CHANGE_TICKS == 0):
        refresh = True
    if (boat.behavior is Behavior.STANDOFF
            and boat.tick_count % STANDOFF_REPOSITION_TICKS == 0):
        refresh = True
    if refresh:
        boat.end_goal = choose_goal(boat, world, rng)

    nxt = _move_towards(boat, boat.end_goal, dt_min, world)

    # reached the goal? pick a new one (keep going). Small arrival radius so it
    # triggers at the fine runtime tick rate.
    if nxt.distance_to(boat.end_goal) <= GOAL_ARRIVE_RADIUS:
        boat.end_goal = choose_goal(boat, world, rng)

    # A friendly transits across the grid and exits at the far edge. It may START
    # on an edge (it enters from one), so only count it as "exited" once it has
    # first moved into the interior and then returns to a boundary.
    if boat.behavior is Behavior.FRIENDLY:
        if not _on_boundary(nxt, world):
            boat.entered_grid = True
        elif boat.entered_grid:
            boat.left_grid = True

    return nxt


def _on_boundary(p: Vec2, world: World) -> bool:
    w, h = world.grid.width, world.grid.height
    return p.x <= 1e-6 or p.y <= 1e-6 or p.x >= w - 1e-6 or p.y >= h - 1e-6
