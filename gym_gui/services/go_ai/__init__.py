"""Go AI services for Human vs Agent gameplay.

This package provides AI engines that can play Go against human players:

- **KataGoService**: Superhuman-strength Go AI using neural networks
  - Requires: katago binary + neural network model file
  - Installation: sudo apt install katago + download model to var/models/katago/

- **GnuGoService**: Classical Go AI (no neural network required)
  - Requires: gnugo binary only
  - Installation: sudo apt install gnugo
  - Strength: Amateur dan level (weaker than KataGo)

Both services implement the same interface and can be used with GoGameController:

    from gym_gui.services.go_ai import KataGoService, GnuGoService

    # Try KataGo first, fall back to GNU Go
    service = KataGoService()
    if not service.is_available():
        service = GnuGoService()

    if service.is_available() and service.start():
        controller.set_ai_action_provider(service.get_best_move)
"""

from gym_gui.services.go_ai.gtp_engine import (
    GTPEngine,
    GTPError,
    action_to_vertex,
    vertex_to_action,
    coords_to_vertex,
    vertex_to_coords,
    board_to_gtp_moves,
)
from gym_gui.services.go_ai.katago_service import (
    KataGoConfig,
    KataGoService,
    KATAGO_DIFFICULTY_PRESETS,
    KATAGO_DIFFICULTY_DESCRIPTIONS,
    create_katago_provider,
)
from gym_gui.services.go_ai.gnugo_service import (
    GnuGoConfig,
    GnuGoService,
    GNUGO_DIFFICULTY_PRESETS,
    GNUGO_DIFFICULTY_DESCRIPTIONS,
    create_gnugo_provider,
)

__all__ = [
    # GTP Engine
    "GTPEngine",
    "GTPError",
    "action_to_vertex",
    "vertex_to_action",
    "coords_to_vertex",
    "vertex_to_coords",
    "board_to_gtp_moves",
    # KataGo
    "KataGoConfig",
    "KataGoService",
    "KATAGO_DIFFICULTY_PRESETS",
    "KATAGO_DIFFICULTY_DESCRIPTIONS",
    "create_katago_provider",
    # GNU Go
    "GnuGoConfig",
    "GnuGoService",
    "GNUGO_DIFFICULTY_PRESETS",
    "GNUGO_DIFFICULTY_DESCRIPTIONS",
    "create_gnugo_provider",
]
