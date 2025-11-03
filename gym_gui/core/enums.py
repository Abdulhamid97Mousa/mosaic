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
    MINIGRID = "minigrid"
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
    MINIGRID_EMPTY_5x5 = "MiniGrid-Empty-5x5-v0"
    MINIGRID_EMPTY_RANDOM_5x5 = "MiniGrid-Empty-Random-5x5-v0"
    MINIGRID_EMPTY_6x6 = "MiniGrid-Empty-6x6-v0"
    MINIGRID_EMPTY_RANDOM_6x6 = "MiniGrid-Empty-Random-6x6-v0"
    MINIGRID_EMPTY_8x8 = "MiniGrid-Empty-8x8-v0"
    MINIGRID_EMPTY_16x16 = "MiniGrid-Empty-16x16-v0"
    MINIGRID_DOORKEY_5x5 = "MiniGrid-DoorKey-5x5-v0"
    MINIGRID_DOORKEY_6x6 = "MiniGrid-DoorKey-6x6-v0"
    MINIGRID_DOORKEY_8x8 = "MiniGrid-DoorKey-8x8-v0"
    MINIGRID_DOORKEY_16x16 = "MiniGrid-DoorKey-16x16-v0"
    MINIGRID_LAVAGAP_S5 = "MiniGrid-LavaGapS5-v0"
    MINIGRID_LAVAGAP_S6 = "MiniGrid-LavaGapS6-v0"
    MINIGRID_LAVAGAP_S7 = "MiniGrid-LavaGapS7-v0"
    MINIGRID_DYNAMIC_OBSTACLES_5X5 = "MiniGrid-Dynamic-Obstacles-5x5-v0"
    MINIGRID_DYNAMIC_OBSTACLES_RANDOM_5X5 = "MiniGrid-Dynamic-Obstacles-Random-5x5-v0"
    MINIGRID_DYNAMIC_OBSTACLES_6X6 = "MiniGrid-Dynamic-Obstacles-6x6-v0"
    MINIGRID_DYNAMIC_OBSTACLES_RANDOM_6X6 = "MiniGrid-Dynamic-Obstacles-Random-6x6-v0"
    MINIGRID_DYNAMIC_OBSTACLES_8X8 = "MiniGrid-Dynamic-Obstacles-8x8-v0"
    MINIGRID_DYNAMIC_OBSTACLES_16X16 = "MiniGrid-Dynamic-Obstacles-16x16-v0"


def get_game_display_name(game_id: GameId) -> str:
    """Get the display name for a GameId with family prefix.
    
    Converts actual environment IDs to user-friendly display names:
    - 'FrozenLake-v1' → 'Gym-ToyText-FrozenLake-v1'
    - 'LunarLander-v3' → 'Gym-Box2D-LunarLander-v3'
    - 'MiniGrid-Empty-5x5-v0' → 'MiniGrid-Empty-5x5-v0' (no prefix, separate library)
    """
    value = game_id.value
    
    # MiniGrid is a separate library, not part of Gym - keep as-is
    if value.startswith("MiniGrid-"):
        return value
    
    # Determine Gym family based on enum
    if game_id in (GameId.FROZEN_LAKE, GameId.FROZEN_LAKE_V2, GameId.CLIFF_WALKING, 
                   GameId.TAXI, GameId.BLACKJACK):
        return f"Gym-ToyText-{value}"
    elif game_id in (GameId.LUNAR_LANDER, GameId.CAR_RACING, GameId.BIPEDAL_WALKER):
        return f"Gym-Box2D-{value}"
    else:
        return f"Gym-{value}"


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
    GameId.MINIGRID_EMPTY_5x5: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_EMPTY_RANDOM_5x5: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_EMPTY_6x6: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_EMPTY_RANDOM_6x6: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_EMPTY_8x8: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_EMPTY_16x16: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_DOORKEY_5x5: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_DOORKEY_6x6: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_DOORKEY_8x8: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_DOORKEY_16x16: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVAGAP_S5: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVAGAP_S6: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVAGAP_S7: EnvironmentFamily.MINIGRID,
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
    GameId.MINIGRID_EMPTY_5x5: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_EMPTY_RANDOM_5x5: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_EMPTY_6x6: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_EMPTY_RANDOM_6x6: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_EMPTY_8x8: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_EMPTY_16x16: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_DOORKEY_5x5: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_DOORKEY_6x6: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_DOORKEY_8x8: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_DOORKEY_16x16: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVAGAP_S5: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVAGAP_S6: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVAGAP_S7: RenderMode.RGB_ARRAY,
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
    GameId.MINIGRID_EMPTY_5x5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_EMPTY_RANDOM_5x5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_EMPTY_6x6: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_EMPTY_RANDOM_6x6: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_EMPTY_8x8: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_EMPTY_16x16: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_DOORKEY_5x5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_DOORKEY_6x6: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_DOORKEY_8x8: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_DOORKEY_16x16: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVAGAP_S5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVAGAP_S6: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVAGAP_S7: (
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
    "get_game_display_name",
]
