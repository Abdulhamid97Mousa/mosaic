"""Configuration widgets for MeltingPot multi-agent environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.config.game_configs import MeltingPotConfig
from gym_gui.core.enums import GameId


MELTINGPOT_GAME_IDS: tuple[GameId, ...] = (
    # All 49 MeltingPot substrates
    GameId.MELTINGPOT_ALLELOPATHIC_HARVEST__OPEN,
    GameId.MELTINGPOT_BACH_OR_STRAVINSKY_IN_THE_MATRIX__ARENA,
    GameId.MELTINGPOT_BACH_OR_STRAVINSKY_IN_THE_MATRIX__REPEATED,
    GameId.MELTINGPOT_BOAT_RACE__EIGHT_RACES,
    GameId.MELTINGPOT_CHEMISTRY__THREE_METABOLIC_CYCLES,
    GameId.MELTINGPOT_CHEMISTRY__THREE_METABOLIC_CYCLES_WITH_PLENTIFUL_DISTRACTORS,
    GameId.MELTINGPOT_CHEMISTRY__TWO_METABOLIC_CYCLES,
    GameId.MELTINGPOT_CHEMISTRY__TWO_METABOLIC_CYCLES_WITH_DISTRACTORS,
    GameId.MELTINGPOT_CHICKEN_IN_THE_MATRIX__ARENA,
    GameId.MELTINGPOT_CHICKEN_IN_THE_MATRIX__REPEATED,
    GameId.MELTINGPOT_CLEAN_UP,
    GameId.MELTINGPOT_COINS,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__ASYMMETRIC,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__CIRCUIT,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__CRAMPED,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__CROWDED,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__FIGURE_EIGHT,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__FORCED,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__RING,
    GameId.MELTINGPOT_COMMONS_HARVEST__CLOSED,
    GameId.MELTINGPOT_COMMONS_HARVEST__OPEN,
    GameId.MELTINGPOT_COMMONS_HARVEST__PARTNERSHIP,
    GameId.MELTINGPOT_COOP_MINING,
    GameId.MELTINGPOT_DAYCARE,
    GameId.MELTINGPOT_EXTERNALITY_MUSHROOMS__DENSE,
    GameId.MELTINGPOT_FACTORY_COMMONS__EITHER_OR,
    GameId.MELTINGPOT_FRUIT_MARKET__CONCENTRIC_RIVERS,
    GameId.MELTINGPOT_GIFT_REFINEMENTS,
    GameId.MELTINGPOT_HIDDEN_AGENDA,
    GameId.MELTINGPOT_PAINTBALL__CAPTURE_THE_FLAG,
    GameId.MELTINGPOT_PAINTBALL__KING_OF_THE_HILL,
    GameId.MELTINGPOT_PREDATOR_PREY__ALLEY_HUNT,
    GameId.MELTINGPOT_PREDATOR_PREY__OPEN,
    GameId.MELTINGPOT_PREDATOR_PREY__ORCHARD,
    GameId.MELTINGPOT_PREDATOR_PREY__RANDOM_FOREST,
    GameId.MELTINGPOT_PRISONERS_DILEMMA_IN_THE_MATRIX__ARENA,
    GameId.MELTINGPOT_PRISONERS_DILEMMA_IN_THE_MATRIX__REPEATED,
    GameId.MELTINGPOT_PURE_COORDINATION_IN_THE_MATRIX__ARENA,
    GameId.MELTINGPOT_PURE_COORDINATION_IN_THE_MATRIX__REPEATED,
    GameId.MELTINGPOT_RATIONALIZABLE_COORDINATION_IN_THE_MATRIX__ARENA,
    GameId.MELTINGPOT_RATIONALIZABLE_COORDINATION_IN_THE_MATRIX__REPEATED,
    GameId.MELTINGPOT_RUNNING_WITH_SCISSORS_IN_THE_MATRIX__ARENA,
    GameId.MELTINGPOT_RUNNING_WITH_SCISSORS_IN_THE_MATRIX__ONE_SHOT,
    GameId.MELTINGPOT_RUNNING_WITH_SCISSORS_IN_THE_MATRIX__REPEATED,
    GameId.MELTINGPOT_STAG_HUNT_IN_THE_MATRIX__ARENA,
    GameId.MELTINGPOT_STAG_HUNT_IN_THE_MATRIX__REPEATED,
    GameId.MELTINGPOT_TERRITORY__INSIDE_OUT,
    GameId.MELTINGPOT_TERRITORY__OPEN,
    GameId.MELTINGPOT_TERRITORY__ROOMS,
)

# Resolution options: display name -> render_scale value
# Native resolution varies by substrate (40×72 to 312×184)
RESOLUTION_OPTIONS: dict[str, int] = {
    "Native (1x)": 1,
    "2x Scale (Recommended)": 2,
    "4x Scale": 4,
    "8x Scale": 8,
}


@dataclass(slots=True)
class ControlCallbacks:
    """Bridge callbacks for propagating UI changes to session state."""

    on_change: Callable[[str, Any], None]


def build_meltingpot_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    game_id: GameId,
    overrides: Dict[str, Any],
    defaults: MeltingPotConfig | None = None,
    callbacks: ControlCallbacks | None = None,
) -> None:
    """Populate MeltingPot-specific configuration widgets.

    Args:
        parent: Parent widget for controls.
        layout: Form layout to add controls to.
        game_id: The selected MeltingPot game ID.
        overrides: Dictionary to store user-modified values.
        defaults: Default configuration values.
        callbacks: Optional callbacks for value changes.
    """
    del game_id  # All MeltingPot substrates share the same config options

    def emit(key: str, value: Any) -> None:
        overrides[key] = value
        if callbacks:
            callbacks.on_change(key, value)

    cfg = defaults if isinstance(defaults, MeltingPotConfig) else MeltingPotConfig()

    # -------- Display Resolution (render_scale) --------
    resolution_combo = QtWidgets.QComboBox(parent)
    resolution_combo.addItems(list(RESOLUTION_OPTIONS.keys()))

    # Find current resolution based on render_scale
    current_scale = overrides.get("render_scale", cfg.render_scale)
    current_resolution = "2x Scale (Recommended)"  # default
    for name, scale in RESOLUTION_OPTIONS.items():
        if scale == current_scale:
            current_resolution = name
            break
    resolution_combo.setCurrentText(current_resolution)

    def on_resolution_changed(text: str) -> None:
        scale = RESOLUTION_OPTIONS.get(text, 2)
        emit("render_scale", scale)

    resolution_combo.currentTextChanged.connect(on_resolution_changed)
    resolution_combo.setToolTip(
        "Scale factor for rendered image.\n"
        "Higher values improve visibility but may impact performance.\n"
        "Native resolution varies by substrate (40×72 to 312×184)."
    )
    layout.addRow("Display Resolution", resolution_combo)

    # -------- Info Label --------
    info_label = QtWidgets.QLabel(
        "<i>MeltingPot: Multi-agent social scenarios by DeepMind.<br>"
        "All agents act simultaneously (parallel stepping).<br><br>"
        "<b>Controls:</b> WASD (move/strafe), Q/E (turn), Space (interact)<br>"
        "Some substrates have extra actions: Z/1 (fire1), C/2 (fire2), X/3 (fire3)</i>",
        parent,
    )
    info_label.setWordWrap(True)
    layout.addRow("", info_label)
