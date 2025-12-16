"""Multi-agent environment UI helpers.

This module provides UI configuration panels and helpers for multi-agent
environments like PettingZoo.
"""

# Re-export PettingZoo family helpers
from .pettingzoo import (
    PETTINGZOO_GAME_IDS,
    ControlCallbacks as PettingZooControlCallbacks,
    PettingZooConfig,
    build_pettingzoo_controls,
    build_pettingzoo_config_panel,
    build_classic_config_panel,
    build_mpe_config_panel,
    build_sisl_config_panel,
    build_butterfly_config_panel,
    build_atari_config_panel,
    get_pettingzoo_display_name,
)

__all__ = [
    # PettingZoo family
    "PETTINGZOO_GAME_IDS",
    "PettingZooControlCallbacks",
    "PettingZooConfig",
    "build_pettingzoo_controls",
    "build_pettingzoo_config_panel",
    "build_classic_config_panel",
    "build_mpe_config_panel",
    "build_sisl_config_panel",
    "build_butterfly_config_panel",
    "build_atari_config_panel",
    "get_pettingzoo_display_name",
]
