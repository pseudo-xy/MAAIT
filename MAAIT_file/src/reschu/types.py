"""Core vocabulary and geometry primitives for the RESCHU-MAAIT simulation.

Everything the rest of the system speaks in — classifications, behaviors, threat
tiers, USV modes — lives here as enums so the vocabulary matches the operator
interface exactly (unknown / enemy / friendly, L / M / H, in range / near edge /
out of range, manual / agent).
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel


# --------------------------------------------------------------------------- #
# Classification: what a boat *is*. Exactly three real types. A boat is always
# operator and is revealed to one of the three real classes by identification.
#
# There are exactly THREE real types. "unknown" is NOT a type — it is the
# operator's *perception* of a boat that has not been identified yet. So the
# real classification enum has three members, and perception is modelled
# separately below (a boat is always one of the three; the operator may just
# not know which yet).
# --------------------------------------------------------------------------- #
class Classification(str, Enum):
    CIVILIAN = "civilian"
    FRIENDLY = "friendly"
    ENEMY = "enemy"


# What the operator perceives: either "unknown" (not yet identified) or, once
# identified, the boat's real classification. Kept as a separate string enum so
# the interface can render "unknown" without it ever being a real boat type.
class Perception(str, Enum):
    UNKNOWN = "unknown"
    CIVILIAN = "civilian"
    FRIENDLY = "friendly"
    ENEMY = "enemy"

    @classmethod
    def from_classification(cls, c: "Classification") -> "Perception":
        return cls(c.value)


# --------------------------------------------------------------------------- #
# Behavior: how a boat moves autonomously each tick. Only meaningful once its
# true class is known to the engine; it drives the movement strategy.
#   - civilian  -> wander slowly (5 kn)
#   - aggressive -> chase nearest USV/friendly (10 kn)   [enemy]
#   - standoff   -> hug the nearest danger-envelope edge  [enemy]
#   - friendly  -> head to a random grid (picks a new one on arrival)
# --------------------------------------------------------------------------- #
class Behavior(str, Enum):
    CIVILIAN = "civilian"
    AGGRESSIVE = "aggressive"
    STANDOFF = "standoff"
    FRIENDLY = "friendly"


# --------------------------------------------------------------------------- #
# Threat tier for enemy boats. Derived from behavior AND speed:
#   * aggressive           -> HIGH
#   * standoff at 9 knots   -> MEDIUM
#   * standoff at 4 knots   -> LOW
# So standoff enemies come in two speeds, and the speed is what distinguishes
# a low-threat from a medium-threat standoff boat. Revisable at any time.
# --------------------------------------------------------------------------- #
class ThreatTier(str, Enum):
    LOW = "L"
    MEDIUM = "M"
    HIGH = "H"


# Standoff speeds that map to a tier. Kept as named constants so the rule is in
# one place if the speeds ever change.
STANDOFF_SPEED_LOW = 4.0
STANDOFF_SPEED_MEDIUM = 9.0


def threat_for_behavior(behavior: Behavior, speed_knots: float | None = None) -> ThreatTier:
    """Threat tier implied by an enemy boat's behavior (and, for standoff boats,
    its speed).

    aggressive          -> HIGH
    standoff @ 9 kn      -> MEDIUM
    standoff @ 4 kn      -> LOW
    (any other standoff speed falls back to MEDIUM)
    """
    if behavior is Behavior.AGGRESSIVE:
        return ThreatTier.HIGH
    if behavior is Behavior.STANDOFF:
        if speed_knots is not None and speed_knots <= STANDOFF_SPEED_LOW:
            return ThreatTier.LOW
        return ThreatTier.MEDIUM
    return ThreatTier.MEDIUM


# --------------------------------------------------------------------------- #
# USV control mode. Three flat choices, set per USV per run (one mode per USV
# for now; a run may mix modes across USVs):
#   - MANUAL : operator-driven (straight-line travel, operator interception)
#   - LLM    : driven by the LLM agent (not trained/connected yet -> stub)
#   - MDP    : driven by the MDP agent (solved engagement MDP, reschu.mdp)
#   - GREEDY : greedy baseline: nearest free boat, instant identification, with
#              a set number of WRONG identifications per mission
#              (World.greedy_wrong_ids, shared by all greedy USVs)
# Identification: LLM/MDP USVs auto-identify nearby boats; MANUAL needs the
# operator action. The untrained LLM/MDP stubs currently run on a shared internal
# baseline so the sim still works; that baseline is not a selectable mode.
# --------------------------------------------------------------------------- #
class USVMode(str, Enum):
    MANUAL = "manual"
    LLM = "llm"
    MDP = "mdp"
    GREEDY = "greedy"

    def is_agent(self) -> bool:
        """True if this mode is an autonomous agent (LLM, MDP or Greedy), not manual."""
        return self in (USVMode.LLM, USVMode.MDP, USVMode.GREEDY)


# --------------------------------------------------------------------------- #
# Comms state shown in Health & Status, driven by each USV's comms schedule.
# --------------------------------------------------------------------------- #
class CommsState(str, Enum):
    IN_RANGE = "in_range"
    NEAR_EDGE = "near_edge"
    OUT_OF_RANGE = "out_of_range"


# --------------------------------------------------------------------------- #
# Geometry: continuous 2-D coordinates on a bounded grid.
# --------------------------------------------------------------------------- #
class Vec2(BaseModel):
    """An immutable 2-D point/vector in grid units."""

    x: float
    y: float

    model_config = {"frozen": True}

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(x=self.x - other.x, y=self.y - other.y)

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(x=self.x * k, y=self.y * k)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def distance_to(self, other: "Vec2") -> float:
        return (self - other).length()

    def normalized(self) -> "Vec2":
        """Unit vector in the same direction (zero vector stays zero)."""
        d = self.length()
        if d == 0:
            return Vec2(x=0.0, y=0.0)
        return Vec2(x=self.x / d, y=self.y / d)
