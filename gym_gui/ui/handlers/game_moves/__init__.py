"""Game move handlers for Human Control Mode.

These handlers process user input on board games when playing in Human Control Mode
(where both players are human, using SessionController).

Available handlers:
- ChessHandler: Chess move handling
- GoHandler: Go move handling
- ConnectFourHandler: Connect Four move handling
"""

from gym_gui.ui.handlers.game_moves.chess import ChessHandler
from gym_gui.ui.handlers.game_moves.go import GoHandler
from gym_gui.ui.handlers.game_moves.connect_four import ConnectFourHandler

__all__ = [
    "ChessHandler",
    "GoHandler",
    "ConnectFourHandler",
]
