"""Documentation for MiniHack environments.

MiniHack is a sandbox framework for RL research built on the NetHack Learning
Environment (NLE). It provides customizable, simplified environments for
navigation, skill acquisition, and exploration tasks.

NOTE: This is separate from full NetHack (see game_docs/NetHack/).
"""

from __future__ import annotations

from .controls import MINIHACK_CONTROLS_HTML
from .navigation import (
    MINIHACK_ROOM_HTML,
    MINIHACK_CORRIDOR_HTML,
    MINIHACK_MAZEWALK_HTML,
    MINIHACK_RIVER_HTML,
)
from .skills import (
    MINIHACK_SKILLS_SIMPLE_HTML,
    MINIHACK_SKILLS_LAVA_HTML,
    MINIHACK_SKILLS_WOD_HTML,
    MINIHACK_SKILLS_QUEST_HTML,
)
from .exploration import (
    MINIHACK_EXPLORE_MAZE_HTML,
    MINIHACK_HIDENSEEK_HTML,
    MINIHACK_MEMENTO_HTML,
)

__all__ = [
    "MINIHACK_CONTROLS_HTML",
    # Navigation
    "MINIHACK_ROOM_HTML",
    "MINIHACK_CORRIDOR_HTML",
    "MINIHACK_MAZEWALK_HTML",
    "MINIHACK_RIVER_HTML",
    # Skills
    "MINIHACK_SKILLS_SIMPLE_HTML",
    "MINIHACK_SKILLS_LAVA_HTML",
    "MINIHACK_SKILLS_WOD_HTML",
    "MINIHACK_SKILLS_QUEST_HTML",
    # Exploration
    "MINIHACK_EXPLORE_MAZE_HTML",
    "MINIHACK_HIDENSEEK_HTML",
    "MINIHACK_MEMENTO_HTML",
]
