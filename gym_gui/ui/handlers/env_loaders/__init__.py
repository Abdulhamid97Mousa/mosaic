"""Environment loaders for loading and initializing environments.

These loaders handle environment-specific setup including:
- Widget creation
- Signal connections
- Resource initialization
- Lifecycle management

Available loaders:
- ChessEnvLoader: Human vs Agent chess game loading
- VizdoomEnvLoader: ViZDoom environment loading with mouse capture
"""

from gym_gui.ui.handlers.env_loaders.chess import ChessEnvLoader
from gym_gui.ui.handlers.env_loaders.vizdoom import VizdoomEnvLoader

__all__ = [
    "ChessEnvLoader",
    "VizdoomEnvLoader",
]
