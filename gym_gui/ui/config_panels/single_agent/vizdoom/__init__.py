"""UI integration helpers for ViZDoom control panels."""

from .config_panel import (
    VIZDOOM_GAME_IDS,
    ControlCallbacks,
    build_vizdoom_controls,
)

__all__ = [
    "VIZDOOM_GAME_IDS",
    "ControlCallbacks",
    "build_vizdoom_controls",
]
