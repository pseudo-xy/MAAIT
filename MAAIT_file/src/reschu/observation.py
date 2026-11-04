"""The truth-free observation an agent is allowed to see.

Ground truth — a boat's `true_class`, `behavior`, `standoff_speed`, and the true
threat tier — lives on the engine side and is used ONLY to reveal identity through
identification and to score results. No agent's decision-making may touch it.

This module builds the operator's-eye view: for a given USV, what it perceives
about the world. Agents (LLM / MDP / the scripted baseline) receive an
`Observation` and decide from that alone, so every agent plays by the same rules
and the comparison is fair. When the real LLM/MDP agents are implemented, this is
the exact, already-truth-free input they consume.

What's included (perceived only):
  * the USV's own state (position, heading, speed, mode, comms, current target)
  * each boat's id, position, distance, whether it's been identified, and its
    PERCEIVED class (unknown until identified) and shown threat tier
  * each boat's position, speed and heading AS MEASURED by the team's sensor
    track (sensing.py): errors grow with range, and beyond sensor range there is
    no speed/heading at all. These are cues, not identity — an agent (e.g. the
    MDP) uses them to infer a boat's likely class before declaring one, and
    must get closer to see clearly. (The operator's map shows true positions.)
  * the positions of the OTHER USVs on the team (own-side knowledge — the
    operator sees every USV on the map; it says nothing about any boat)
  * the grid bounds
Deliberately excluded: any boat's true_class, behavior, standoff_speed, or true
tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import USV, World
from .sensing import camera_report, sensed
from .types import CommsState, Perception, ThreatTier, USVMode, Vec2


@dataclass
class BoatView:
    """What a USV perceives about one boat — no ground truth."""
    id: str
    position: Vec2
    distance: float               # from the observing USV
    identified: bool
    perceived_class: Perception   # unknown until identified
    threat_tier: ThreatTier | None  # only shown once a correct enemy ID is made
    being_escorted: bool
    left_grid: bool
    speed_knots: float | None     # MEASURED speed; None beyond sensor range
    heading: Vec2 | None          # MEASURED direction of travel; None beyond range
    camera_report: str | None = None  # latest camera look (a type name; may be WRONG)
    camera_look_id: int = 0           # id of that look (new id = fresh look)
    operator_handling: bool = False   # the operator's manual USV owns its follow-up


@dataclass
class SelfView:
    """The observing USV's own state."""
    id: str
    position: Vec2
    heading: Vec2
    speed_knots: float
    mode: USVMode
    comms_state: CommsState
    assigned_target_id: str | None


@dataclass
class TeamView:
    """Another USV on our own team (own-side knowledge, no boat ground truth)."""
    id: str
    position: Vec2


@dataclass
class Observation:
    """Everything one USV is allowed to know this tick. Truth-free by
    construction — an agent physically cannot read a boat's true identity from
    this object."""
    now_min: float
    grid_width: float
    grid_height: float
    me: SelfView
    boats: list[BoatView]
    team: list[TeamView] = field(default_factory=list)   # other active USVs

    def boat(self, boat_id: str) -> BoatView | None:
        return next((b for b in self.boats if b.id == boat_id), None)

    def unidentified(self) -> list[BoatView]:
        return [b for b in self.boats if not b.identified and not b.left_grid]

    def perceived_enemies(self) -> list[BoatView]:
        return [
            b for b in self.boats
            if b.identified and b.perceived_class is Perception.ENEMY and not b.left_grid
        ]

    def perceived_nonenemies(self) -> list[BoatView]:
        return [
            b for b in self.boats
            if b.identified and b.perceived_class is not Perception.ENEMY and not b.left_grid
        ]


def observe(usv: USV, world: World, now_min: float) -> Observation:
    """Build the truth-free observation for `usv` from the world state."""
    boats = []
    for b in world.boats:
        pos, heading, speed = sensed(b, world)   # range-dependent track (sensing.py)
        look, look_id = camera_report(b, world)
        boats.append(BoatView(
            id=b.id,
            position=pos,
            distance=usv.position.distance_to(pos),
            identified=b.identified,
            perceived_class=b.perceived_class,
            threat_tier=b.threat_tier,
            being_escorted=b.being_escorted,
            left_grid=b.left_grid,
            speed_knots=speed,
            heading=heading,
            camera_report=look,
            camera_look_id=look_id,
            operator_handling=b.manual_handler is not None,
        ))
    me = SelfView(
        id=usv.id,
        position=usv.position,
        heading=usv.heading,
        speed_knots=usv.effective_speed_knots(),
        mode=usv.mode,
        comms_state=usv.comms_state,
        assigned_target_id=usv.assigned_target_id,
    )
    team = [
        TeamView(id=u.id, position=u.position)
        for u in world.usvs
        if u.id != usv.id and not u.left_grid and not u.eliminated
    ]
    return Observation(
        now_min=now_min,
        grid_width=world.grid.width,
        grid_height=world.grid.height,
        me=me,
        boats=boats,
        team=team,
    )
