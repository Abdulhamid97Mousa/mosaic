"""MiniGrid game documentation module."""
from __future__ import annotations

from .EmptyEnv import MINIGRID_EMPTY_HTML, get_empty_html
from .DoorKeyEnv import MINIGRID_DOORKEY_HTML, get_doorkey_html
from .LavaGapEnv import MINIGRID_LAVAGAP_HTML, get_lavagap_html
from .DynamicObstaclesEnv import MINIGRID_DYNAMIC_OBSTACLES_HTML, get_dynamic_obstacles_html
from .BlockedUnlockPickupEnv import MINIGRID_BLOCKED_UNLOCK_PICKUP_HTML
from .MultiRoomEnv import MINIGRID_MULTIROOM_HTML, get_multiroom_html
from .ObstructedMazeEnv import MINIGRID_OBSTRUCTED_MAZE_HTML, get_obstructed_maze_html
from .CrossingEnv import MINIGRID_CROSSING_HTML, get_crossing_html
from .RedBlueDoorEnv import MINIGRID_REDBLUEDOORS_HTML, get_redbluedoors_html

__all__ = [
    "MINIGRID_EMPTY_HTML",
    "get_empty_html",
    "MINIGRID_DOORKEY_HTML",
    "get_doorkey_html",
    "MINIGRID_LAVAGAP_HTML",
    "get_lavagap_html",
    "MINIGRID_DYNAMIC_OBSTACLES_HTML",
    "get_dynamic_obstacles_html",
    "MINIGRID_BLOCKED_UNLOCK_PICKUP_HTML",
    "MINIGRID_MULTIROOM_HTML",
    "get_multiroom_html",
    "MINIGRID_OBSTRUCTED_MAZE_HTML",
    "get_obstructed_maze_html",
    "MINIGRID_CROSSING_HTML",
    "get_crossing_html",
    "MINIGRID_REDBLUEDOORS_HTML",
    "get_redbluedoors_html",
]
