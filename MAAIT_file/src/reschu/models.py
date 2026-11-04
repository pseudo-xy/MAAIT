"""Domain entities for the RESCHU-MAAIT simulation.

The central design idea, driven by the identification rules: every boat carries
a *true* type known to the engine from t=0, and a *perceived* state that is what
the operator actually sees. Identification is the bridge — it copies truth into
perception. Until then the operator sees "unknown" (a perception, not a type).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .types import (
    Behavior,
    Classification,
    CommsState,
    Perception,
    ThreatTier,
    USVMode,
    Vec2,
    threat_for_behavior,
)


# --------------------------------------------------------------------------- #
# Grid: the bounded, continuous world. No lat/long — just a rectangle in grid
# units with an origin at (0, 0).
# --------------------------------------------------------------------------- #
class Grid(BaseModel):
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    def contains(self, p: Vec2) -> bool:
        return 0.0 <= p.x <= self.width and 0.0 <= p.y <= self.height

    def clamp_inset(self, p: Vec2, margin: float = 0.6) -> Vec2:
        """Like clamp, but keeps a small margin off the very edge so a marker
        never sits pinned exactly on the boundary line (which leaves a visual
        smudge where boats pile up against the wall)."""
        return Vec2(
            x=min(max(p.x, margin), self.width - margin),
            y=min(max(p.y, margin), self.height - margin),
        )

    def clamp(self, p: Vec2) -> Vec2:
        """Keep a point inside the grid bounds."""
        return Vec2(
            x=min(max(p.x, 0.0), self.width),
            y=min(max(p.y, 0.0), self.height),
        )

    # ------------------------------------------------------------------ #
    # Interaction radii (identification range, danger zone, circle radius)
    # scale with the map size so they occupy the same fraction of each map:
    # Small (40 wide) x1, Medium (80) x2, Large (120) x3. Base values are the
    # Small-map values; multiply by this factor for the actual radius on this
    # grid. Derived from width / 40 so it degrades gracefully for other sizes.
    # ------------------------------------------------------------------ #
    @property
    def range_scale(self) -> float:
        return max(1.0, round(self.width / 40.0))


# --------------------------------------------------------------------------- #
# Comms schedule: per-USV, per-state timing that drives the comms signal shown
# in Health & Status. Matches the admin-screen grid exactly — for each state a
# start time (minutes into the mission) and a duration (minutes).
# --------------------------------------------------------------------------- #
class CommsWindow(BaseModel):
    """One comms state active from `start` for `duration` minutes."""

    state: CommsState
    start_min: float = Field(ge=0)
    duration_min: float = Field(ge=0)

    def active_at(self, t_min: float) -> bool:
        return self.duration_min > 0 and self.start_min <= t_min < self.start_min + self.duration_min


class CommsSchedule(BaseModel):
    """A USV's full comms schedule: the three windows (in range / near edge /
    out of range) with their own start + duration."""

    windows: list[CommsWindow] = Field(default_factory=list)

    def state_at(self, t_min: float) -> CommsState:
        """Comms state at mission time `t_min`. If several windows overlap, the
        most-degraded active state wins (out_of_range > near_edge > in_range);
        if none is active, default to in range."""
        priority = {
            CommsState.OUT_OF_RANGE: 3,
            CommsState.NEAR_EDGE: 2,
            CommsState.IN_RANGE: 1,
        }
        active = [w.state for w in self.windows if w.active_at(t_min)]
        if not active:
            return CommsState.IN_RANGE
        return max(active, key=lambda s: priority[s])


# --------------------------------------------------------------------------- #
# Boat: any contact on the map that is not one of our USVs. Holds both the true
# type (engine-side) and the perceived state (operator-side).
# --------------------------------------------------------------------------- #
class Boat(BaseModel):
    id: str
    position: Vec2
    heading: Vec2 = Vec2(x=1.0, y=0.0)  # unit-ish direction of travel

    # --- ground truth (known to the engine, hidden from the operator) ------ #
    true_class: Classification
    behavior: Behavior

    # Standoff enemies come in two speeds: 4 kn (low threat) or 9 kn (medium
    # threat). Only meaningful when behavior is STANDOFF; ignored otherwise.
    standoff_speed: float = 9.0

    # Per-boat speed variation (drawn/updated by the engine; see
    # SimulationEngine._update_boat_speeds). Makes the observed speed an honest,
    # noisy cue rather than an exact giveaway of class.
    #   speed_scale  : persistent multiplier on the nominal speed, drawn once
    #                  (None = not drawn yet -> treated as 1.0)
    #   speed_jitter : slow fluctuation (fraction of speed), updated every tick
    speed_scale: float | None = None
    speed_jitter: float = 0.0

    # The team's sensor track of this boat (see sensing.py): correlated errors
    # that grow with range. Agents observe position/heading/speed THROUGH these;
    # physics and the operator's map use the true values.
    track_pos_err: Vec2 = Vec2(x=0.0, y=0.0)
    track_hdg_err_deg: float = 0.0
    track_spd_err: float = 0.0
    track_range_ratio: float = 0.0   # true distance to nearest USV, in ID ranges
    track_heading: Vec2 | None = None  # filtered (range-smoothed) heading
    track_speed: float | None = None   # filtered (range-smoothed) speed
    visual_report: str | None = None  # latest camera look (a type name; may be wrong)
    visual_tick: int = 0              # ticks since that look
    visual_look_id: int = 0           # increments with every fresh look

    # --- movement goal --------------------------------------------------- #
    # Where the boat is heading. Boats move toward this point (while still
    # reacting, e.g. an aggressive boat re-aims at a moving USV each tick). When
    # a boat reaches its goal it picks a new one by its type's rule. `start_pos`
    # records where the boat began, used by the civilian straight-line rule.
    start_pos: Vec2 | None = None
    end_goal: Vec2 | None = None
    # Civilian boats fix one straight-line heading (chosen at first goal pick)
    # and hold it, so they track a straight course from their start.
    straight_heading: Vec2 | None = None

    # --- perceived state (what the operator sees) -------------------------- #
    identified: bool = False
    perceived_class: Perception = Perception.UNKNOWN
    threat_tier: ThreatTier | None = None  # only meaningful once enemy + identified

    # --- identification accounting (for the "how many tries to get it right"
    #     research question). A misidentification does NOT change behavior — the
    #     boat keeps its true behavior; only what the operator sees changes. -- #
    id_attempts: int = 0  # how many times this boat has been (re)identified

    # ticks this boat has been stepped (used for cadence-based behaviors, e.g.
    # a civilian changing direction every N ticks).
    tick_count: int = 0

    # --- escort state (friendly/civilian being escorted off the grid) ------ #
    # When True, the boat abandons its normal behavior and routes to the nearest
    # safe edge (avoiding enemy/unidentified boats) until it exits the grid.
    being_escorted: bool = False
    # The camera video that was playing on a USV feed for this boat just before it
    # was identified — captured each tick while still unidentified, so at the
    # moment of ID it holds the last pre-ID clip. The operator can replay it from
    # the Contacts panel. None if the boat was never seen on camera before ID.
    preview_video: str | None = None
    left_grid: bool = False  # set once the boat has exited the map
    eliminated: bool = False  # true if removed by a completed enemy-circle (vs. exited)
    entered_grid: bool = False  # friendly: has moved into the interior at least once
    escort_start_min: float | None = None  # mission time escort began
    escort_end_min: float | None = None    # mission time the boat left the grid
    # mission time (minutes) when this boat was FIRST identified (correctly or
    # not); stays at the first value even if later re-identified. None if never.
    first_identified_min: float | None = None
    # mission time (minutes) the boat APPEARED: 0 for starting boats, the spawn
    # time for a replacement friendly. Time-to-ID is measured from here.
    spawned_min: float = 0.0
    # True for a contact the team already knew at the start (pre-identified):
    # no identification decision was needed, so it is excluded from ID metrics.
    known_at_start: bool = False
    # --- operator follow-up after a MANUAL identification ------------------ #
    # When the operator identifies a contact as ENEMY or FRIENDLY with the manual
    # USV, the operator handles it manually (manual_engagement.py): follow a
    # friendly out, circle an enemy for the AI's circling time. AI USVs leave it.
    manual_handler: str | None = None      # manual USV currently owning the follow-up
    claimed_by: str | None = None          # (log) manual USV that claimed it at ID time
    manual_circle_min: float = 0.0         # time the handler has spent circling it
    manual_circle_start_min: float | None = None
    # For a FRIENDLY: the edge point (on the boundary) it should exit through,
    # from the admin-set exit cell. None -> it exits at the far edge naturally.
    exit_point: Vec2 | None = None

    def nominal_speed_knots(self) -> float:
        """The boat's ASSIGNED speed for its class/tier: standoff boats use their
        standoff_speed (4 = low tier, 9 = medium tier); the rest are fixed by
        behavior. This is what defines the threat tier — never the varied speed."""
        if self.behavior is Behavior.STANDOFF:
            return self.standoff_speed
        return {
            Behavior.CIVILIAN: 3.0,   # slowed per professor's request (was 5.0)
            Behavior.AGGRESSIVE: 10.0,
            Behavior.FRIENDLY: 8.0,
        }[self.behavior]

    def speed_knots(self) -> float:
        """The speed the boat ACTUALLY moves at (and that sensors/agents observe):
        its nominal speed x its persistent per-boat factor x (1 + the current
        slow fluctuation). With variation switched off this equals the nominal
        speed. Kept strictly positive."""
        scale = 1.0 if self.speed_scale is None else self.speed_scale
        return max(0.1, self.nominal_speed_knots() * scale * (1.0 + self.speed_jitter))

    def is_correctly_identified(self) -> bool:
        """True once the operator's perceived class matches the boat's truth."""
        return self.identified and self.perceived_class.value == self.true_class.value

    def needs_work(self) -> bool:
        """Truth-free: does this boat still need an agent to work it? Uses
        perceived class only, never ground truth.
          * left the grid              -> done.
          * identified as CIVILIAN     -> done (civilians are left alone after ID,
                                          not escorted).
          * everything else on grid    -> needs work: unidentified (needs ID),
                                          identified enemy (needs circling), or
                                          identified friendly (needs escorting off).
        """
        if self.left_grid:
            return False
        if self.identified and self.perceived_class.value == "civilian":
            return False
        if self.manual_handler is not None:
            return False          # the operator's manual USV is handling it
        return True

    def identify(self, as_class: Perception | None = None) -> None:
        """Record an identification of this boat.

        as_class = None  -> a *correct* identification: perceived class is set to
                            the boat's true class (this is what an in-range agent
                            or a correct operator call does).
        as_class = X     -> the operator/agent labelled it X, which may be WRONG.
                            The label is recorded for display, but the boat's true
                            class and behavior are untouched — it keeps behaving
                            as it truly is. This lets us count how many attempts
                            it takes before the label finally matches the truth.

        Every call counts as one attempt. For a correctly-identified enemy the
        threat tier is derived from behavior + speed; otherwise threat_tier is
        cleared (an enemy mislabelled friendly shows no threat tier).
        """
        self.id_attempts += 1
        self.identified = True
        if as_class is None:
            self.perceived_class = Perception.from_classification(self.true_class)
        else:
            self.perceived_class = as_class

        if self.is_correctly_identified() and self.true_class is Classification.ENEMY:
            self.threat_tier = threat_for_behavior(self.behavior, self.nominal_speed_knots())
        else:
            self.threat_tier = None


# --------------------------------------------------------------------------- #
# USV: one of our vehicles. Carries mode (manual/agent), a live comms state
# (computed from its schedule each tick), and an optional assigned target.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# USV: one of our vehicles. All USVs share one base speed; a MANUAL USV can have
# that speed overridden by the operator (used for interception). An AGENT USV
# always runs at the base speed.
# --------------------------------------------------------------------------- #
# The single base speed every USV uses unless an operator overrides it.
USV_BASE_SPEED_KNOTS = 12.0
# Bounds and step for the operator's manual speed control (the speed stepper).
MANUAL_SPEED_MIN = 0.5
MANUAL_SPEED_MAX = 30.0
MANUAL_SPEED_STEP = 0.5


RTB_MIN_FUEL_PCT = 20.0   # below this fuel level a USV can no longer RTB


class USV(BaseModel):
    id: str
    position: Vec2
    heading: Vec2 = Vec2(x=1.0, y=0.0)
    mode: USVMode = USVMode.MANUAL

    # Where the USV began the run. Set in the scenario/config; if left None it
    # defaults to `position` at first tick. Configurable per run like end goals.
    start_pos: Vec2 | None = None

    # Operator-drawn destination (MANUAL only). When set, the USV travels toward
    # this point each tick (the operator double-clicks the USV and drags a line to
    # a spot in a boat's projected path). Cleared once the USV arrives.
    manual_destination: Vec2 | None = None

    # Straight-line heading a MANUAL USV travels along from the start (until the
    # operator intervenes). Set in config, or defaults to the initial heading.
    manual_heading: Vec2 | None = None

    # Operator speed override (MANUAL only). None means "use the base speed".
    # The operator raises/lowers this to intercept a boat in time.
    operator_speed_knots: float | None = None

    # If True, this USV never moves (holds its start position) regardless of
    # mode — used for scenarios with fixed observation posts. It still identifies
    # boats that come within range.
    stationary: bool = False

    # Set true when this USV is eliminated by entering an aggressive boat's danger
    # zone. The engine drops eliminated USVs from the world (they disappear).
    eliminated: bool = False


    # Boats this USV has already ATTEMPTED to identify (so it doesn't re-roll the
    # same boat every tick). A different USV can still attempt a misidentified boat.
    attempted_boat_ids: list[str] = Field(default_factory=list)

    # A class DECLARATION requested by this USV's decision agent (MDP): the
    # (boat_id, declared class) it wants to commit to. Agents never touch the
    # world, so they express the declaration as intent here; the engine applies
    # it (only while the boat is inside ID range) and then clears it. The
    # declared class may be WRONG — any misidentification comes from the
    # agent's own decision, never from injected noise.
    pending_declaration: tuple[str, Perception] | None = None
    # True when the OPERATOR picked this USV's current contact (double-click).
    # The MDP team assignment then leaves the pairing alone until it resolves.
    assignment_pinned: bool = False

    comms_schedule: CommsSchedule = Field(default_factory=CommsSchedule)
    comms_state: CommsState = CommsState.IN_RANGE  # updated each tick

    # Timeline of control-mode blocks over the mission: ordered list of
    # {"mode": USVMode, "start_min": float, "end_min": float}. When set, the USV's
    # mode is switched to the block covering the current mission time. Empty = the
    # USV keeps its fixed starting mode for the whole mission.
    mode_schedule: list[dict] = Field(default_factory=list)

    assigned_target_id: str | None = None  # agent-chosen or operator-set

    # Manual interception: the boat this manual USV is trying to intercept. When
    # set, the USV steers toward the intercept point instead of its straight-line
    # heading, and stops at identification range for the operator to identify.
    intercept_target_id: str | None = None

    # Engagement tracking. When an agent starts circling an enemy, we record the
    # mission time it began, so it can stop after CIRCLE_DURATION_MIN and move on.
    circling_since_min: float | None = None

    # Log of completed circle engagements: each entry (boat_id, start_min,
    # end_min). Appended when a circle finishes (timer elapses or boat released).
    circle_events: list[tuple[str, float, float]] = Field(default_factory=list)

    # Fuel remaining, as a percentage (100 = full). Drains slowly each tick.
    fuel_pct: float = 100.0          # set per USV in settings ("fuel start %")

    # Return-to-base state. When the operator clicks RTB, rtb is set True and the
    # USV heads for the nearest edge and exits; once off the grid, left_grid is
    # set and the USV stops being drawn / its health row greys out.
    rtb: bool = False
    left_grid: bool = False

    def effective_speed_knots(self) -> float:
        """The speed this USV actually travels at this tick. Agents always use
        the base speed; a manual USV uses the operator's override if one is set,
        otherwise the base speed."""
        if self.mode is USVMode.MANUAL and self.operator_speed_knots is not None:
            return self.operator_speed_knots
        return USV_BASE_SPEED_KNOTS

    def set_operator_speed(self, knots: float | None) -> None:
        """Operator action: set (or clear, with None) the manual speed override.
        Ignored for agent USVs, which always run at base speed. Clamped to a
        sensible range so the stepper can't drive it negative or absurdly high."""
        if self.mode is USVMode.MANUAL:
            if knots is not None:
                knots = max(MANUAL_SPEED_MIN, min(MANUAL_SPEED_MAX, knots))
            self.operator_speed_knots = knots

    def rtb_blocked_reason(self) -> str | None:
        """Why an RTB order cannot be carried out right now (None = it can).
        The order needs a comms link, and enough fuel to make it home: below
        RTB_MIN_FUEL_PCT the USV can no longer return to base."""
        if self.comms_state.value == "out_of_range":
            return "out of comms range"
        if self.fuel_pct < RTB_MIN_FUEL_PCT:
            return f"fuel below {RTB_MIN_FUEL_PCT:g}%"
        return None

    def update_comms(self, t_min: float) -> None:
        self.comms_state = self.comms_schedule.state_at(t_min)

    def update_mode(self, t_min: float) -> None:
        """Switch control mode according to the timeline block covering t_min.
        Manual is never auto-switched AWAY from by the operator's own control, but
        the timeline is authoritative: if a block says Manual/LLM/MDP for this
        stretch of the mission, that's the USV's mode. No schedule -> mode is
        left unchanged (its fixed starting mode)."""
        if not self.mode_schedule:
            return
        for blk in self.mode_schedule:
            if blk["start_min"] <= t_min < blk["end_min"]:
                if self.mode is not blk["mode"]:
                    self._switch_mode(blk["mode"])
                return
        # past the last block: hold the final block's mode
        last = self.mode_schedule[-1]
        if self.mode is not last["mode"]:
            self._switch_mode(last["mode"])

    def _switch_mode(self, new_mode: "USVMode") -> None:
        """Apply a control-mode change mid-mission, clearing state that only makes
        sense for the previous mode so the new controller starts clean."""
        self.mode = new_mode
        # dropping an agent engagement / manual course on a mode change
        self.assigned_target_id = None
        self.circling_since_min = None
        self.intercept_target_id = None
        self.pending_declaration = None
        if new_mode is USVMode.MANUAL:
            # a freshly-manual USV holds position until the operator directs it
            self.manual_destination = None
            if self.manual_heading is None:
                self.manual_heading = Vec2(x=1.0, y=0.0)
        else:
            self.operator_speed_knots = None


# --------------------------------------------------------------------------- #
# World: the whole scenario state at an instant. This is the object that will
# later be serialized to the browser each tick.
# --------------------------------------------------------------------------- #
class World(BaseModel):
    grid: Grid
    usvs: list[USV] = Field(default_factory=list)
    boats: list[Boat] = Field(default_factory=list)
    mission_time_min: float = 0.0  # minutes since mission start
    # Boat speed variation (0 = off). speed_spread: std-dev of each boat's
    # persistent speed factor around 1.0 (e.g. 0.15 = +/-15%). speed_jitter:
    # std-dev of the slow per-tick fluctuation (e.g. 0.05 = +/-5%).
    speed_spread: float = 0.0
    speed_jitter: float = 0.0
    # Agent sensing quality (sensing.py): 0 = perfect, 1 = default range-dependent
    # noise; larger = noisier. Only affects what agents OBSERVE.
    sensor_noise: float = 0.0
    # Camera classification cue for agents (sensing.py); only active when
    # sensor_noise > 0.
    visual_cue: bool = True
    # Who chooses MDP USVs' contacts: "value" / "rate" = the MDP team assignment
    # (mdp/assignment.py; total gain, or gain per minute of USV time);
    # "nearest" = the engine's closest-unworked-boat rule.
    mdp_assignment: str = "rate"

    def usv(self, usv_id: str) -> USV | None:
        return next((u for u in self.usvs if u.id == usv_id), None)

    def boat(self, boat_id: str) -> Boat | None:
        return next((b for b in self.boats if b.id == boat_id), None)
