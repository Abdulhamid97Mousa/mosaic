"""Documentation for NetHack (full game via NLE).

NetHack is a classic roguelike dungeon crawler (1987). This module provides
documentation for the full NetHack game accessed via the NetHack Learning
Environment (NLE).

NOTE: For simplified RL environments, see game_docs/MiniHack/.
"""

from __future__ import annotations

from .controls import NETHACK_CONTROLS_HTML
from .game import NETHACK_FULL_HTML

__all__ = [
    "NETHACK_CONTROLS_HTML",
    "NETHACK_FULL_HTML",
]
