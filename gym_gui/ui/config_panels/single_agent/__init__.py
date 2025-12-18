"""Single-agent environment UI helpers."""

# Re-export gym family helpers
from .gym import (
    TOY_TEXT_FAMILY,
    BOX2D_FAMILY,
    build_frozenlake_controls,
    build_frozenlake_v2_controls,
    build_taxi_controls,
    build_cliff_controls,
    build_lunarlander_controls,
    build_car_racing_controls,
    build_bipedal_controls,
)

# Re-export minigrid helpers
from .minigrid import (
    MINIGRID_GAME_IDS,
    ControlCallbacks,
    build_minigrid_controls,
    resolve_default_config,
)

# Re-export ALE helpers (Atari via ALE)
from .ale import (
    ALE_GAME_IDS,
    ControlCallbacks as ALEControlCallbacks,
    build_ale_controls,
)

from .vizdoom import (
    VIZDOOM_GAME_IDS,
    ControlCallbacks as ViZDoomControlCallbacks,
    build_vizdoom_controls,
)

# Re-export Crafter helpers
from .crafter import (
    CRAFTER_GAME_IDS,
    ControlCallbacks as CrafterControlCallbacks,
    build_crafter_controls,
)

__all__ = [
    # Gym family
    "TOY_TEXT_FAMILY",
    "BOX2D_FAMILY",
    "build_frozenlake_controls",
    "build_frozenlake_v2_controls",
    "build_taxi_controls",
    "build_cliff_controls",
    "build_lunarlander_controls",
    "build_car_racing_controls",
    "build_bipedal_controls",
    # MiniGrid family
    "MINIGRID_GAME_IDS",
    "ControlCallbacks",
    "build_minigrid_controls",
    "resolve_default_config",
    # ALE family
    "ALE_GAME_IDS",
    "ALEControlCallbacks",
    "build_ale_controls",
    "VIZDOOM_GAME_IDS",
    "ViZDoomControlCallbacks",
    "build_vizdoom_controls",
    # Crafter family
    "CRAFTER_GAME_IDS",
    "CrafterControlCallbacks",
    "build_crafter_controls",
]
