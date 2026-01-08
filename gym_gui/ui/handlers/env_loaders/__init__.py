"""Environment loaders for loading and initializing environments.

These loaders handle environment-specific setup including:
- Widget creation
- Signal connections
- Resource initialization
- Lifecycle management

Available loaders:
- CheckersEnvLoader: Human vs Agent checkers game loading
- ChessEnvLoader: Human vs Agent chess game loading
- ConnectFourEnvLoader: Human vs Agent Connect Four game loading
- GoEnvLoader: Human vs Agent Go game loading
- TicTacToeEnvLoader: Human vs Agent Tic-Tac-Toe game loading
- VizdoomEnvLoader: ViZDoom environment loading with mouse capture
"""

from gym_gui.ui.handlers.env_loaders.checkers import CheckersEnvLoader
from gym_gui.ui.handlers.env_loaders.chess import ChessEnvLoader
from gym_gui.ui.handlers.env_loaders.connect_four import ConnectFourEnvLoader
from gym_gui.ui.handlers.env_loaders.go import GoEnvLoader
from gym_gui.ui.handlers.env_loaders.tictactoe import TicTacToeEnvLoader
from gym_gui.ui.handlers.env_loaders.vizdoom import VizdoomEnvLoader

__all__ = [
    "CheckersEnvLoader",
    "ChessEnvLoader",
    "ConnectFourEnvLoader",
    "GoEnvLoader",
    "TicTacToeEnvLoader",
    "VizdoomEnvLoader",
]
