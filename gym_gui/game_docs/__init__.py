"""Aggregated environment documentation helpers."""

from __future__ import annotations

from typing import Dict

from gym_gui.core.enums import GameId
from gym_gui.game_docs.Gymnasium.ToyText import (
    TAXI_HTML,
    FROZEN_HTML,
    FROZEN_V2_HTML,
    CLIFF_HTML,
    BLACKJACK_HTML,
)
from gym_gui.game_docs.Gymnasium.Box2D import (
    LUNAR_LANDER_HTML,
    CAR_RACING_HTML,
    BIPEDAL_WALKER_HTML,
)
from gym_gui.game_docs.Gymnasium.MuJuCo import (
    ANT_HTML,
    HALF_CHEETAH_HTML,
    HOPPER_HTML,
    HUMANOID_HTML,
    HUMANOID_STANDUP_HTML,
    INVERTED_DOUBLE_PENDULUM_HTML,
    INVERTED_PENDULUM_HTML,
    PUSHER_HTML,
    REACHER_HTML,
    SWIMMER_HTML,
    WALKER2D_HTML,
)
from gym_gui.game_docs.MiniGrid import (
    MINIGRID_EMPTY_HTML,
    get_empty_html,
    MINIGRID_DOORKEY_HTML,
    get_doorkey_html,
    MINIGRID_LAVAGAP_HTML,
    get_lavagap_html,
    MINIGRID_DYNAMIC_OBSTACLES_HTML,
    get_dynamic_obstacles_html,
    MINIGRID_BLOCKED_UNLOCK_PICKUP_HTML,
    MINIGRID_MULTIROOM_HTML,
    get_multiroom_html,
    MINIGRID_OBSTRUCTED_MAZE_HTML,
    get_obstructed_maze_html,
    MINIGRID_CROSSING_HTML,
    get_crossing_html,
)
from gym_gui.game_docs.ALE import (
    ADVENTURE_HTML,
    AIR_RAID_HTML,
    ASSAULT_HTML,
)

_DEFAULT_DOC = (
    "<h3>Documentation unavailable</h3>"
    "<p>This environment does not yet have a descriptive blurb."
    " Check the upstream project for details.</p>"
)

GAME_INFO: Dict[GameId, str] = {
    GameId.TAXI: TAXI_HTML,
    GameId.FROZEN_LAKE: FROZEN_HTML,
    GameId.FROZEN_LAKE_V2: FROZEN_V2_HTML,
    GameId.CLIFF_WALKING: CLIFF_HTML,
    GameId.BLACKJACK: BLACKJACK_HTML,
    GameId.LUNAR_LANDER: LUNAR_LANDER_HTML,
    GameId.CAR_RACING: CAR_RACING_HTML,
    GameId.BIPEDAL_WALKER: BIPEDAL_WALKER_HTML,
    GameId.ANT: ANT_HTML,
    GameId.HALF_CHEETAH: HALF_CHEETAH_HTML,
    GameId.HOPPER: HOPPER_HTML,
    GameId.HUMANOID: HUMANOID_HTML,
    GameId.HUMANOID_STANDUP: HUMANOID_STANDUP_HTML,
    GameId.INVERTED_DOUBLE_PENDULUM: INVERTED_DOUBLE_PENDULUM_HTML,
    GameId.INVERTED_PENDULUM: INVERTED_PENDULUM_HTML,
    GameId.PUSHER: PUSHER_HTML,
    GameId.REACHER: REACHER_HTML,
    GameId.SWIMMER: SWIMMER_HTML,
    GameId.WALKER2D: WALKER2D_HTML,
}

_MINIGRID_EMPTY_VARIANTS = (
    GameId.MINIGRID_EMPTY_5x5,
    GameId.MINIGRID_EMPTY_RANDOM_5x5,
    GameId.MINIGRID_EMPTY_6x6,
    GameId.MINIGRID_EMPTY_RANDOM_6x6,
    GameId.MINIGRID_EMPTY_8x8,
    GameId.MINIGRID_EMPTY_16x16,
)
GAME_INFO.update({gid: MINIGRID_EMPTY_HTML for gid in _MINIGRID_EMPTY_VARIANTS})

_MINIGRID_DOORKEY_VARIANTS = (
    GameId.MINIGRID_DOORKEY_5x5,
    GameId.MINIGRID_DOORKEY_6x6,
    GameId.MINIGRID_DOORKEY_8x8,
    GameId.MINIGRID_DOORKEY_16x16,
)
GAME_INFO.update({gid: MINIGRID_DOORKEY_HTML for gid in _MINIGRID_DOORKEY_VARIANTS})

_MINIGRID_LAVAGAP_VARIANTS = (
    GameId.MINIGRID_LAVAGAP_S5,
    GameId.MINIGRID_LAVAGAP_S6,
    GameId.MINIGRID_LAVAGAP_S7,
)
GAME_INFO.update({gid: MINIGRID_LAVAGAP_HTML for gid in _MINIGRID_LAVAGAP_VARIANTS})

_MINIGRID_DYNAMIC_OBSTACLES_VARIANTS = (
    GameId.MINIGRID_DYNAMIC_OBSTACLES_5X5,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_5X5,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_6X6,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_6X6,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_8X8,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_16X16,
)
GAME_INFO.update({gid: MINIGRID_DYNAMIC_OBSTACLES_HTML for gid in _MINIGRID_DYNAMIC_OBSTACLES_VARIANTS})

GAME_INFO[GameId.MINIGRID_BLOCKED_UNLOCK_PICKUP] = MINIGRID_BLOCKED_UNLOCK_PICKUP_HTML

_MINIGRID_MULTIROOM_VARIANTS = (
    GameId.MINIGRID_MULTIROOM_N2_S4,
    GameId.MINIGRID_MULTIROOM_N4_S5,
    GameId.MINIGRID_MULTIROOM_N6,
)
GAME_INFO.update({gid: MINIGRID_MULTIROOM_HTML for gid in _MINIGRID_MULTIROOM_VARIANTS})

GAME_INFO[GameId.MINIGRID_OBSTRUCTED_MAZE_1DLHB] = MINIGRID_OBSTRUCTED_MAZE_HTML
GAME_INFO[GameId.MINIGRID_OBSTRUCTED_MAZE_FULL] = MINIGRID_OBSTRUCTED_MAZE_HTML

# ALE mappings
GAME_INFO[GameId.ADVENTURE_V4] = ADVENTURE_HTML
GAME_INFO[GameId.ALE_ADVENTURE_V5] = ADVENTURE_HTML
GAME_INFO[GameId.AIR_RAID_V4] = AIR_RAID_HTML
GAME_INFO[GameId.ALE_AIR_RAID_V5] = AIR_RAID_HTML
GAME_INFO[GameId.ASSAULT_V4] = ASSAULT_HTML
GAME_INFO[GameId.ALE_ASSAULT_V5] = ASSAULT_HTML


def get_game_info(game_id: GameId) -> str:
    """Return HTML documentation for the specified environment."""
    
    # Handle Empty variants dynamically
    if game_id in (GameId.MINIGRID_EMPTY_5x5, GameId.MINIGRID_EMPTY_RANDOM_5x5,
                   GameId.MINIGRID_EMPTY_6x6, GameId.MINIGRID_EMPTY_RANDOM_6x6,
                   GameId.MINIGRID_EMPTY_8x8, GameId.MINIGRID_EMPTY_16x16):
        return get_empty_html(game_id.value)
    
    # Handle DoorKey variants dynamically
    if game_id in (GameId.MINIGRID_DOORKEY_5x5, GameId.MINIGRID_DOORKEY_6x6, 
                   GameId.MINIGRID_DOORKEY_8x8, GameId.MINIGRID_DOORKEY_16x16):
        return get_doorkey_html(game_id.value)
    
    # Handle LavaGap variants dynamically
    if game_id in (GameId.MINIGRID_LAVAGAP_S5, GameId.MINIGRID_LAVAGAP_S6, 
                   GameId.MINIGRID_LAVAGAP_S7):
        return get_lavagap_html(game_id.value)
    
    # Handle Dynamic Obstacles variants dynamically
    if game_id in (GameId.MINIGRID_DYNAMIC_OBSTACLES_5X5, GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_5X5,
                   GameId.MINIGRID_DYNAMIC_OBSTACLES_6X6, GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_6X6,
                   GameId.MINIGRID_DYNAMIC_OBSTACLES_8X8, GameId.MINIGRID_DYNAMIC_OBSTACLES_16X16):
        return get_dynamic_obstacles_html(game_id.value)
    
    # Handle MultiRoom variants dynamically
    if game_id in (GameId.MINIGRID_MULTIROOM_N2_S4, GameId.MINIGRID_MULTIROOM_N4_S5,
                   GameId.MINIGRID_MULTIROOM_N6):
        return get_multiroom_html(game_id.value)
    
    # Handle Obstructed Maze variants dynamically
    if game_id in (GameId.MINIGRID_OBSTRUCTED_MAZE_1DLHB, GameId.MINIGRID_OBSTRUCTED_MAZE_FULL):
        return get_obstructed_maze_html(game_id.value)
    
    # Handle Crossing variants dynamically
    if game_id in (GameId.MINIGRID_LAVA_CROSSING_S9N1, GameId.MINIGRID_LAVA_CROSSING_S9N2,
                   GameId.MINIGRID_LAVA_CROSSING_S9N3, GameId.MINIGRID_LAVA_CROSSING_S11N5,
                   GameId.MINIGRID_SIMPLE_CROSSING_S9N1, GameId.MINIGRID_SIMPLE_CROSSING_S9N2,
                   GameId.MINIGRID_SIMPLE_CROSSING_S9N3, GameId.MINIGRID_SIMPLE_CROSSING_S11N5):
        return get_crossing_html(game_id.value)
    
    return GAME_INFO.get(game_id, _DEFAULT_DOC)


__all__ = ["GAME_INFO", "get_game_info"]
