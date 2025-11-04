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
    CART_POLE = "CartPole-v1"
    ACROBOT = "Acrobot-v1"
    MOUNTAIN_CAR = "MountainCar-v0"
    PONG_NO_FRAMESKIP = "PongNoFrameskip-v4"
    BREAKOUT_NO_FRAMESKIP = "BreakoutNoFrameskip-v4"
    PROCGEN_COINRUN = "procgen:procgen-coinrun-v0"
    PROCGEN_MAZE = "procgen:procgen-maze-v0"
    ANT = "Ant-v5"
    HALF_CHEETAH = "HalfCheetah-v5"
    HOPPER = "Hopper-v5"
    HUMANOID = "Humanoid-v5"
    HUMANOID_STANDUP = "HumanoidStandup-v5"
    INVERTED_DOUBLE_PENDULUM = "InvertedDoublePendulum-v5"
    INVERTED_PENDULUM = "InvertedPendulum-v5"
    PUSHER = "Pusher-v5"
    REACHER = "Reacher-v5"
    SWIMMER = "Swimmer-v5"
    WALKER2D = "Walker2d-v5"
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
    MINIGRID_BLOCKED_UNLOCK_PICKUP = "MiniGrid-BlockedUnlockPickup-v0"
    MINIGRID_MULTIROOM_N2_S4 = "MiniGrid-MultiRoom-N2-S4-v0"
    MINIGRID_MULTIROOM_N4_S5 = "MiniGrid-MultiRoom-N4-S5-v0"
    MINIGRID_MULTIROOM_N6 = "MiniGrid-MultiRoom-N6-v0"
    MINIGRID_OBSTRUCTED_MAZE_1DLHB = "MiniGrid-ObstructedMaze-1Dlhb-v1"
    MINIGRID_OBSTRUCTED_MAZE_FULL = "MiniGrid-ObstructedMaze-Full-v1"
    MINIGRID_LAVA_CROSSING_S9N1 = "MiniGrid-LavaCrossingS9N1-v0"
    MINIGRID_LAVA_CROSSING_S9N2 = "MiniGrid-LavaCrossingS9N2-v0"
    MINIGRID_LAVA_CROSSING_S9N3 = "MiniGrid-LavaCrossingS9N3-v0"
    MINIGRID_LAVA_CROSSING_S11N5 = "MiniGrid-LavaCrossingS11N5-v0"
    MINIGRID_SIMPLE_CROSSING_S9N1 = "MiniGrid-SimpleCrossingS9N1-v0"
    MINIGRID_SIMPLE_CROSSING_S9N2 = "MiniGrid-SimpleCrossingS9N2-v0"
    MINIGRID_SIMPLE_CROSSING_S9N3 = "MiniGrid-SimpleCrossingS9N3-v0"
    MINIGRID_SIMPLE_CROSSING_S11N5 = "MiniGrid-SimpleCrossingS11N5-v0"


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
    elif game_id in (
        GameId.CART_POLE,
        GameId.ACROBOT,
        GameId.MOUNTAIN_CAR,
    ):
        return f"Gym-ClassicControl-{value}"
    elif game_id in (
        GameId.PONG_NO_FRAMESKIP,
        GameId.BREAKOUT_NO_FRAMESKIP,
    ):
        return f"Gym-Atari-{value}"
    elif game_id in (
        GameId.PROCGEN_COINRUN,
        GameId.PROCGEN_MAZE,
    ):
        return f"Procgen-{value}"
    elif game_id in (
        GameId.ANT,
        GameId.HALF_CHEETAH,
        GameId.HOPPER,
        GameId.HUMANOID,
        GameId.HUMANOID_STANDUP,
        GameId.INVERTED_DOUBLE_PENDULUM,
        GameId.INVERTED_PENDULUM,
        GameId.PUSHER,
        GameId.REACHER,
        GameId.SWIMMER,
        GameId.WALKER2D,
    ):
        return f"Gym-MuJoCo-{value}"
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
    GameId.CART_POLE: EnvironmentFamily.CLASSIC_CONTROL,
    GameId.ACROBOT: EnvironmentFamily.CLASSIC_CONTROL,
    GameId.MOUNTAIN_CAR: EnvironmentFamily.CLASSIC_CONTROL,
    GameId.PONG_NO_FRAMESKIP: EnvironmentFamily.ATARI,
    GameId.BREAKOUT_NO_FRAMESKIP: EnvironmentFamily.ATARI,
    GameId.PROCGEN_COINRUN: EnvironmentFamily.OTHER,
    GameId.PROCGEN_MAZE: EnvironmentFamily.OTHER,
    GameId.ANT: EnvironmentFamily.MUJOCO,
    GameId.HALF_CHEETAH: EnvironmentFamily.MUJOCO,
    GameId.HOPPER: EnvironmentFamily.MUJOCO,
    GameId.HUMANOID: EnvironmentFamily.MUJOCO,
    GameId.HUMANOID_STANDUP: EnvironmentFamily.MUJOCO,
    GameId.INVERTED_DOUBLE_PENDULUM: EnvironmentFamily.MUJOCO,
    GameId.INVERTED_PENDULUM: EnvironmentFamily.MUJOCO,
    GameId.PUSHER: EnvironmentFamily.MUJOCO,
    GameId.REACHER: EnvironmentFamily.MUJOCO,
    GameId.SWIMMER: EnvironmentFamily.MUJOCO,
    GameId.WALKER2D: EnvironmentFamily.MUJOCO,
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
    GameId.MINIGRID_LAVA_CROSSING_S9N1: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVA_CROSSING_S9N2: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVA_CROSSING_S9N3: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_LAVA_CROSSING_S11N5: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N1: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N2: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N3: EnvironmentFamily.MINIGRID,
    GameId.MINIGRID_SIMPLE_CROSSING_S11N5: EnvironmentFamily.MINIGRID,
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
    GameId.CART_POLE: RenderMode.RGB_ARRAY,
    GameId.ACROBOT: RenderMode.RGB_ARRAY,
    GameId.MOUNTAIN_CAR: RenderMode.RGB_ARRAY,
    GameId.PONG_NO_FRAMESKIP: RenderMode.RGB_ARRAY,
    GameId.BREAKOUT_NO_FRAMESKIP: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_COINRUN: RenderMode.RGB_ARRAY,
    GameId.PROCGEN_MAZE: RenderMode.RGB_ARRAY,
    GameId.ANT: RenderMode.RGB_ARRAY,
    GameId.HALF_CHEETAH: RenderMode.RGB_ARRAY,
    GameId.HOPPER: RenderMode.RGB_ARRAY,
    GameId.HUMANOID: RenderMode.RGB_ARRAY,
    GameId.HUMANOID_STANDUP: RenderMode.RGB_ARRAY,
    GameId.INVERTED_DOUBLE_PENDULUM: RenderMode.RGB_ARRAY,
    GameId.INVERTED_PENDULUM: RenderMode.RGB_ARRAY,
    GameId.PUSHER: RenderMode.RGB_ARRAY,
    GameId.REACHER: RenderMode.RGB_ARRAY,
    GameId.SWIMMER: RenderMode.RGB_ARRAY,
    GameId.WALKER2D: RenderMode.RGB_ARRAY,
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
    GameId.MINIGRID_LAVA_CROSSING_S9N1: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVA_CROSSING_S9N2: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVA_CROSSING_S9N3: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_LAVA_CROSSING_S11N5: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N1: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N2: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_SIMPLE_CROSSING_S9N3: RenderMode.RGB_ARRAY,
    GameId.MINIGRID_SIMPLE_CROSSING_S11N5: RenderMode.RGB_ARRAY,
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
    GameId.CART_POLE: (ControlMode.AGENT_ONLY,),
    GameId.ACROBOT: (ControlMode.AGENT_ONLY,),
    GameId.MOUNTAIN_CAR: (ControlMode.AGENT_ONLY,),
    GameId.PONG_NO_FRAMESKIP: (ControlMode.AGENT_ONLY,),
    GameId.BREAKOUT_NO_FRAMESKIP: (ControlMode.AGENT_ONLY,),
    GameId.PROCGEN_COINRUN: (ControlMode.AGENT_ONLY,),
    GameId.PROCGEN_MAZE: (ControlMode.AGENT_ONLY,),
    GameId.ANT: (ControlMode.AGENT_ONLY,),
    GameId.HALF_CHEETAH: (ControlMode.AGENT_ONLY,),
    GameId.HOPPER: (ControlMode.AGENT_ONLY,),
    GameId.HUMANOID: (ControlMode.AGENT_ONLY,),
    GameId.HUMANOID_STANDUP: (ControlMode.AGENT_ONLY,),
    GameId.INVERTED_DOUBLE_PENDULUM: (ControlMode.AGENT_ONLY,),
    GameId.INVERTED_PENDULUM: (ControlMode.AGENT_ONLY,),
    GameId.PUSHER: (ControlMode.AGENT_ONLY,),
    GameId.REACHER: (ControlMode.AGENT_ONLY,),
    GameId.SWIMMER: (ControlMode.AGENT_ONLY,),
    GameId.WALKER2D: (ControlMode.AGENT_ONLY,),
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
    GameId.MINIGRID_LAVA_CROSSING_S9N1: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVA_CROSSING_S9N2: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVA_CROSSING_S9N3: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_LAVA_CROSSING_S11N5: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_SIMPLE_CROSSING_S9N1: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_SIMPLE_CROSSING_S9N2: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_SIMPLE_CROSSING_S9N3: (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    ),
    GameId.MINIGRID_SIMPLE_CROSSING_S11N5: (
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
