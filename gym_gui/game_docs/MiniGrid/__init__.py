"""MiniGrid game documentation module."""
from __future__ import annotations

from .EmptyEnv import MINIGRID_EMPTY_HTML
from .DoorKeyEnv import MINIGRID_DOORKEY_HTML
from .LavaGapEnv import MINIGRID_LAVAGAP_HTML
from .DynamicObstaclesEnv import MINIGRID_DYNAMIC_OBSTACLES_HTML
from .BlockedUnlockPickupEnv import MINIGRID_BLOCKED_UNLOCK_PICKUP_HTML
from .MultiRoomEnv import MINIGRID_MULTIROOM_HTML
from .ObstructedMazeEnv import MINIGRID_OBSTRUCTED_MAZE_HTML

__all__ = [
    "MINIGRID_EMPTY_HTML",
    "MINIGRID_DOORKEY_HTML",
    "MINIGRID_LAVAGAP_HTML",
    "MINIGRID_DYNAMIC_OBSTACLES_HTML",
    "MINIGRID_BLOCKED_UNLOCK_PICKUP_HTML",
    "MINIGRID_MULTIROOM_HTML",
    "MINIGRID_OBSTRUCTED_MAZE_HTML",
]
