"""The simulation engine.

`SimulationEngine.tick()` advances the world by one time step: it moves every
boat by its behavior, updates each USV's comms state from its schedule, and runs
automatic identification for agent-mode USVs. `run()` is the asyncio loop that
calls tick() at a fixed cadence — this is the piece the web layer will later
wrap, streaming the world to the browser after each tick.

The design deliberately separates:
  * tick()  -> pure state advance, fully testable with no clock involved
  * run()   -> the async timing wrapper around tick()
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from .assignment import assign_all, reassign_free_agents
from .manual_engagement import keep_on_mode_switch, update_manual_engagements
from .behaviors import step_boat
from .identification import apply_declarations, auto_identify
from .models import Boat, World
from .types import Behavior, Classification, USVMode, Vec2
from .policies import policy_for
from .sensing import update_tracks
from .usv_movement import manual_step, resolve_separation, rtb_step

# Boat speed variation (strength set per world: World.speed_spread / speed_jitter)
SPEED_SCALE_MIN, SPEED_SCALE_MAX = 0.6, 1.4  # clamp on a boat's persistent factor
SPEED_JITTER_RHO = 0.95   # per-tick persistence of the fluctuation (~20 s memory at
                          # 1 tick = 1 sim-second), so speed drifts, not jitters
SPEED_JITTER_CLAMP = 3.0  # clip the fluctuation at +/- this many std-devs

# How close a USV can get to an aggressive boat before that boat eliminates it.
# Base (Small-map) danger radius in grid cells. The ACTUAL radius on a map is this
# times grid.range_scale (Small x1 = 1.5, Medium x2 = 3.0, Large x3 = 4.5) so it
# stays the same fraction of each map. Use danger_zone_radius(grid) for the scaled
# value. The circling radius (see usv_movement) is scaled to stay just outside it.
DANGER_ZONE_RADIUS = 1.5


def danger_zone_radius(grid) -> float:
    """Scaled danger-zone radius for the given grid (Small x1, Med x2, Lg x3)."""
    return DANGER_ZONE_RADIUS * grid.range_scale


class SimulationEngine:
    def __init__(
        self,
        world: World,
        dt_min: float = 0.1,
        seed: int | None = None,
        assign_at_start: bool = True,
    ) -> None:
        """
        world           : the initial scenario state (mutated in place each tick)
        dt_min          : simulated minutes advanced per tick
        seed            : RNG seed for reproducible runs (tests + research)
        assign_at_start : link each agent USV to a boat at setup (closest, with
                          aggressiveness as tie-break; exclusive per boat)
        """
        self.world = world
        self.dt_min = dt_min
        # a new mission must not inherit motion state from an earlier one
        from .usv_movement import reset_motion_state
        reset_motion_state()
        self.rng = random.Random(seed)
        # separate stream for boat speed noise, so switching variation on/off
        # does not reshuffle any other random behavior in a seeded run
        self.speed_rng = random.Random(None if seed is None else seed * 7919 + 17)
        # separate stream for sensor-track noise (agent sensing only)
        self.sensor_rng = random.Random(None if seed is None else seed * 104729 + 31)
        self._running = False
        # next numeric suffix for a respawned friendly (B7, B8, ... after the
        # six starting boats). Start past the highest existing B-number.
        self._next_boat_num = self._highest_boat_number() + 1
        # capture start positions for USVs (boats capture theirs on first step)
        for u in world.usvs:
            if u.start_pos is None:
                u.start_pos = u.position
        if assign_at_start:
            assign_all(world)

    def _highest_boat_number(self) -> int:
        """Largest numeric suffix among boats named like 'B<n>' (else 0)."""
        best = 0
        for b in self.world.boats:
            if b.id.startswith("B") and b.id[1:].isdigit():
                best = max(best, int(b.id[1:]))
        return best

    def _spawn_friendly(self) -> None:
        """Add a new friendly boat entering from a random edge, heading inward.
        Named B7, B8, ... continuing the sequence. Keeps the friendly count on
        the grid constant: one leaves, one enters."""
        w, h = self.world.grid.width, self.world.grid.height
        rng = self.rng
        edge = rng.choice(["left", "right", "bottom", "top"])
        if edge == "left":
            pos = Vec2(x=0.0, y=rng.uniform(0, h)); head = Vec2(x=1.0, y=0.0)
        elif edge == "right":
            pos = Vec2(x=w, y=rng.uniform(0, h)); head = Vec2(x=-1.0, y=0.0)
        elif edge == "bottom":
            pos = Vec2(x=rng.uniform(0, w), y=0.0); head = Vec2(x=0.0, y=1.0)
        else:  # top
            pos = Vec2(x=rng.uniform(0, w), y=h); head = Vec2(x=0.0, y=-1.0)
        new_id = f"B{self._next_boat_num}"
        self._next_boat_num += 1
        self.world.boats.append(
            Boat(
                id=new_id,
                position=pos,
                start_pos=pos,
                true_class=Classification.FRIENDLY,
                behavior=Behavior.FRIENDLY,
                straight_heading=head,  # aim inward across the grid
                spawned_min=round(self.world.mission_time_min, 3),
            )
        )

    # ------------------------------------------------------------------ #
    # One deterministic step. No sleeping, no clock — just state -> state.
    # ------------------------------------------------------------------ #
    def tick(self) -> None:
        w = self.world

        # 1) advance mission time
        w.mission_time_min += self.dt_min

        # 1b) switch each USV's control mode per its timeline (no-op if unset)
        for usv in w.usvs:
            was_manual = usv.mode is USVMode.MANUAL
            usv.update_mode(w.mission_time_min)
            if was_manual and usv.mode.is_agent():
                # the timeline hands this USV to an AI: it keeps (and finishes)
                # the contact the operator was following up
                keep_on_mode_switch(w, usv, w.mission_time_min)

        # 1c) per-boat speed variation: draw any new boat's persistent speed
        #     factor, and drift every boat's slow fluctuation (no-op when off)
        self._update_boat_speeds(w)

        # 2) move every boat by its behavior. Iterate a snapshot: a friendly
        #    reaching the edge appends a new boat, which must not be processed
        #    until the next tick.
        for boat in list(w.boats):
            was_on_grid = not boat.left_grid
            boat.position = step_boat(boat, w, self.dt_min, self.rng)
            # record the escort end time the moment an escorted boat exits
            if was_on_grid and boat.left_grid and boat.escort_end_min is None:
                boat.escort_end_min = round(w.mission_time_min, 3)
            # 2b) a friendly that reaches the edge despawns; spawn a fresh friendly
            #     entering from a random edge so the friendly count stays constant
            if (
                was_on_grid
                and boat.left_grid
                and boat.behavior is Behavior.FRIENDLY
            ):
                self._spawn_friendly()

        # 3) move USVs. AGENT USVs are driven by their decision agent (policy):
        #    scripted / llm / mdp. The policy decides target, speed, motion, and
        #    engagement; the mechanics live in usv_movement. MANUAL USVs travel
        #    straight / intercept.
        for usv in w.usvs:
            if usv.left_grid:
                continue  # returned to base: gone from the sim
            # drain fuel: 1% per 10 ticks
            usv.fuel_pct = max(0.0, usv.fuel_pct - 0.1)
            # RTB: head straight for the nearest edge and exit
            if usv.rtb:
                usv.position = rtb_step(usv, w, self.dt_min)
                gw, gh = w.grid.width, w.grid.height
                p = usv.position
                if p.x <= 1e-6 or p.y <= 1e-6 or p.x >= gw - 1e-6 or p.y >= gh - 1e-6:
                    usv.left_grid = True
                continue
            if usv.stationary:
                continue  # fixed observation post: never moves, but still IDs
            if usv.mode.is_agent():
                usv.position = policy_for(usv).decide(
                    usv, w, self.dt_min, w.mission_time_min
                )
            else:
                usv.position = manual_step(usv, w, self.dt_min)

        # 3a) separate any entities that would overlap: each veers 10 degrees
        #     away from the other so they slide past (circling/escorting USVs and
        #     the manual USV are exempt — being close there is intentional or the
        #     operator's responsibility).
        resolve_separation(w, self.dt_min)

        # 3b) eliminate any USV that has entered an aggressive boat's danger zone
        #     (it "sees" the USV and attacks). Eliminated USVs are removed from
        #     the grid (they disappear). Applies to all USVs, manual and agent.
        self._eliminate_usvs_in_danger(w)

        # 3b') operator follow-up: time the manual USV spends circling an enemy
        #      it identified (manual_engagement.py)
        update_manual_engagements(w, w.mission_time_min, self.dt_min)

        # 3c) any agent that finished its engagement (assignment cleared) picks a
        #     new boat by priority, if one is available.
        reassign_free_agents(w)

        # 3d) advance the team's sensor tracks of every boat (agent sensing only;
        #     no-op when the world's sensor_noise is 0)
        update_tracks(w, self.sensor_rng)

        # 4) refresh each USV's comms state from its schedule
        for usv in w.usvs:
            usv.update_comms(w.mission_time_min)

        # 5) identification:
        #    5a) MDP USVs: apply the class declarations their policy requested
        #        (in range only; the declared class may be wrong — the MDP's call)
        apply_declarations(w)
        #    5b) LLM USVs: auto-identify boats now in range
        auto_identify(w, self.rng)

        # stamp the time of FIRST identification on any boat just identified
        for boat in w.boats:
            if boat.identified and boat.first_identified_min is None:
                boat.first_identified_min = round(w.mission_time_min, 3)

    def _update_boat_speeds(self, w: World) -> None:
        """Give boats realistic, noisy speeds (when the world enables it).

        * Each boat gets ONE persistent factor ~ N(1, speed_spread), clamped to
          [SPEED_SCALE_MIN, SPEED_SCALE_MAX] — drawn the first tick it exists, so
          starting boats and respawned friendlies are handled the same way.
        * Each tick its fluctuation follows a slow AR(1) drift with stationary
          std-dev speed_jitter, so speed wanders smoothly rather than jumping.
        The threat tier is unaffected: it comes from the NOMINAL speed.
        """
        spread, jit = w.speed_spread, w.speed_jitter
        rho = SPEED_JITTER_RHO
        innov = (1.0 - rho * rho) ** 0.5
        for b in w.boats:
            if b.speed_scale is None:
                if spread > 0:
                    f = self.speed_rng.gauss(1.0, spread)
                    b.speed_scale = min(SPEED_SCALE_MAX, max(SPEED_SCALE_MIN, f))
                else:
                    b.speed_scale = 1.0
            if jit > 0:
                j = rho * b.speed_jitter + innov * self.speed_rng.gauss(0.0, jit)
                lim = SPEED_JITTER_CLAMP * jit
                b.speed_jitter = min(lim, max(-lim, j))
            else:
                b.speed_jitter = 0.0

    def _eliminate_usvs_in_danger(self, w: World) -> None:
        """Remove any USV that a hostile aggressive boat catches in its danger
        zone. The race is: a USV must CORRECTLY identify an aggressive boat before
        it closes in. An aggressive boat that has been correctly identified as an
        enemy is being circled by its USV (at the circle radius, just outside the
        danger zone) and no longer eliminates that USV — the operator won that
        exchange. An aggressive boat that is NOT correctly identified still
        attacks: any USV inside its danger zone is eliminated (a misID, or simply
        getting too close before identifying, gets the USV destroyed)."""
        aggressors = [
            b for b in w.boats
            if b.behavior is Behavior.AGGRESSIVE and not b.left_grid
            and not b.is_correctly_identified()   # correctly-IDed enemies are neutralised
        ]
        if not aggressors:
            return
        survivors = []
        _danger = danger_zone_radius(w.grid)
        for u in w.usvs:
            hit = any(
                u.position.distance_to(b.position) <= _danger
                for b in aggressors
            )
            if hit:
                u.eliminated = True   # dropped from the world (disappears)
            else:
                survivors.append(u)
        if len(survivors) != len(w.usvs):
            w.usvs[:] = survivors

    # ------------------------------------------------------------------ #
    async def run(
        self,
        duration_min: float | None = None,
        realtime_factor: float = 0.0,
        on_tick: Callable[[World], Awaitable[None]] | None = None,
    ) -> None:
        """
        duration_min    : stop after this much simulated time (None = forever)
        realtime_factor : wall-clock seconds to sleep per simulated minute.
                          0.0 = run as fast as possible (good for headless tests
                          and batch research runs); e.g. 1.0 makes one simulated
                          minute take one real second for a live demo.
        on_tick         : optional async callback invoked with the world after
                          each tick (the future WebSocket broadcast hook).
        """
        self._running = True
        start = self.world.mission_time_min
        while self._running:
            self.tick()
            if on_tick is not None:
                await on_tick(self.world)
            if realtime_factor > 0:
                await asyncio.sleep(self.dt_min * realtime_factor)
            else:
                await asyncio.sleep(0)  # yield control without wall-clock delay
            if duration_min is not None and (
                self.world.mission_time_min - start >= duration_min
            ):
                break
        self._running = False

    def stop(self) -> None:
        self._running = False
