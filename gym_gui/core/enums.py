from __future__ import annotations

"""Enumerations that describe the Gym GUI domain model."""

from enum import Enum, auto
from typing import Iterable


class StrEnum(str, Enum):
    """Minimal stand-in for :class:`enum.StrEnum` (Python 3.11+)."""

    def __new__(cls, value: str) -> "StrEnum":
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj


class EnvironmentFamily(StrEnum):
    """High-level groupings that organise Gymnasium environments."""

    TOY_TEXT = "toy_text"
    BOX2D = "box2d"
    CLASSIC_CONTROL = "classic_control"
    ATARI = "atari"
    MUJOCO = "mujoco"
    OTHER = "other"


class GameId(StrEnum):
    """Canonical Gymnasium environment identifiers supported by the GUI."""

    FROZEN_LAKE = "FrozenLake-v1"
    FROZEN_LAKE_V2 = "FrozenLake-v2"
    CLIFF_WALKING = "CliffWalking-v1"
    TAXI = "Taxi-v3"
    BLACKJACK = "Blackjack-v1"
    LUNAR_LANDER = "LunarLander-v3"
    CAR_RACING = "CarRacing-v3"
    BIPEDAL_WALKER = "BipedalWalker-v3"


class ControlMode(StrEnum):
    """Who is currently in control of the environment."""

    HUMAN_ONLY = "human_only"
    AGENT_ONLY = "agent_only"
    HYBRID_TURN_BASED = "hybrid_turn_based"
    HYBRID_HUMAN_AGENT = "hybrid_human_agent"
    MULTI_AGENT_COOP = "multi_agent_coop"
    MULTI_AGENT_COMPETITIVE = "multi_agent_competitive"


class RenderMode(StrEnum):
    """Rendering strategies supported by the UI."""

    ASCII = "ascii"
    GRID = "grid"
    RGB_ARRAY = "rgb_array"
    SURFACE = "surface"


class ActionSpaceType(StrEnum):
    """Simplified view over Gymnasium action space types."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_BINARY = "multi_binary"
    MULTI_DISCRETE = "multi_discrete"


class AgentRole(StrEnum):
    """Roles an agent can play when multiple controllers are present."""

    PRIMARY = "primary"
    ASSIST = "assist"
    SPECTATOR = "spectator"


class AdapterCapability(Enum):
    """Flags that describe optional behaviours for adapters."""

    RECORD_SUPPORT = auto()
    FAST_RESET = auto()
    MULTI_AGENT = auto()


ENVIRONMENT_FAMILY_BY_GAME: dict[GameId, EnvironmentFamily] = {
    GameId.FROZEN_LAKE: EnvironmentFamily.TOY_TEXT,
    GameId.FROZEN_LAKE_V2: EnvironmentFamily.TOY_TEXT,
    GameId.CLIFF_WALKING: EnvironmentFamily.TOY_TEXT,
    GameId.TAXI: EnvironmentFamily.TOY_TEXT,
    GameId.BLACKJACK: EnvironmentFamily.TOY_TEXT,
    GameId.LUNAR_LANDER: EnvironmentFamily.BOX2D,
    GameId.CAR_RACING: EnvironmentFamily.BOX2D,
    GameId.BIPEDAL_WALKER: EnvironmentFamily.BOX2D,
}


DEFAULT_RENDER_MODES: dict[GameId, RenderMode] = {
    GameId.FROZEN_LAKE: RenderMode.GRID,
    GameId.FROZEN_LAKE_V2: RenderMode.GRID,
    GameId.CLIFF_WALKING: RenderMode.GRID,
    GameId.TAXI: RenderMode.GRID,
    GameId.BLACKJACK: RenderMode.RGB_ARRAY,
    GameId.LUNAR_LANDER: RenderMode.RGB_ARRAY,
    GameId.CAR_RACING: RenderMode.RGB_ARRAY,
    GameId.BIPEDAL_WALKER: RenderMode.RGB_ARRAY,
}


DEFAULT_CONTROL_MODES: dict[GameId, Iterable[ControlMode]] = {
    GameId.FROZEN_LAKE: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.FROZEN_LAKE_V2: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.CLIFF_WALKING: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.TAXI: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.BLACKJACK: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    ),
    GameId.LUNAR_LANDER: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.CAR_RACING: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.BIPEDAL_WALKER: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
}


__all__ = [
    "EnvironmentFamily",
    "GameId",
    "ControlMode",
    "RenderMode",
    "ActionSpaceType",
    "AgentRole",
    "AdapterCapability",
    "ENVIRONMENT_FAMILY_BY_GAME",
    "DEFAULT_RENDER_MODES",
    "DEFAULT_CONTROL_MODES",
]
