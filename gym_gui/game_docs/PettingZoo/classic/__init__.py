"""PettingZoo Classic game documentation module.

This module provides HTML documentation strings for PettingZoo Classic environments.
Classic environments are turn-based board games and card games using the AEC API.

Supported games:
- Chess (chess_v6): Standard chess with AlphaZero-style observations
- Connect Four (connect_four_v3): Classic 4-in-a-row game
- Go (go_v5): Ancient board game (9x9, 13x13, 19x19)
- Tic-Tac-Toe (tictactoe_v3): Classic 3x3 grid game
"""
from __future__ import annotations

from .chess import CHESS_HTML, get_chess_html
from .connect_four import CONNECT_FOUR_HTML, get_connect_four_html
from .go import GO_HTML, get_go_html
from .tictactoe import TICTACTOE_HTML, get_tictactoe_html

__all__ = [
    # Chess
    "CHESS_HTML",
    "get_chess_html",
    # Connect Four
    "CONNECT_FOUR_HTML",
    "get_connect_four_html",
    # Go
    "GO_HTML",
    "get_go_html",
    # Tic-Tac-Toe
    "TICTACTOE_HTML",
    "get_tictactoe_html",
]
