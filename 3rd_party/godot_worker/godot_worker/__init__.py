"""Godot Worker - Integration wrapper for MOSAIC BDI-RL framework.

This module provides a wrapper around the Godot game engine,
enabling Godot-based RL environments within the GUI_BDI_RL application.

The Godot binary lives in ../bin/ and is launched by this worker.
All customizations and integrations belong in this godot_worker package.
"""

__version__ = "0.1.0"

from godot_worker.config import GodotConfig, GodotRenderMode
from godot_worker.launcher import GodotLauncher, GodotProcess, get_launcher

__all__ = [
    "__version__",
    "GodotConfig",
    "GodotRenderMode",
    "GodotLauncher",
    "GodotProcess",
    "get_launcher",
]
