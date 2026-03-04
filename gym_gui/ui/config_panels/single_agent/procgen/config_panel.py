"""Human control configuration widgets for Procgen environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.config.game_configs import ProcgenConfig
from gym_gui.core.enums import GameId

PROCGEN_GAME_IDS: tuple[GameId, ...] = (
    GameId.PROCGEN_BIGFISH,
    GameId.PROCGEN_BOSSFIGHT,
    GameId.PROCGEN_CAVEFLYER,
    GameId.PROCGEN_CHASER,
    GameId.PROCGEN_CLIMBER,
    GameId.PROCGEN_COINRUN,
    GameId.PROCGEN_DODGEBALL,
    GameId.PROCGEN_FRUITBOT,
    GameId.PROCGEN_HEIST,
    GameId.PROCGEN_JUMPER,
    GameId.PROCGEN_LEAPER,
    GameId.PROCGEN_MAZE,
    GameId.PROCGEN_MINER,
    GameId.PROCGEN_NINJA,
    GameId.PROCGEN_PLUNDER,
    GameId.PROCGEN_STARPILOT,
)

# Resolution options: display name -> render_scale value
# Base resolution is 512x512 from info["rgb"]
RESOLUTION_OPTIONS: dict[str, int] = {
    "512x512 (Native)": 1,
    "1024x1024 (2x)": 2,
    "2048x2048 (4x)": 4,
    "4096x4096 (8x)": 8,
}


@dataclass(slots=True)
class ControlCallbacks:
    """Bridge callbacks for propagating UI changes to session state."""

    on_change: Callable[[str, Any], None]


def build_procgen_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    game_id: GameId,
    overrides: Dict[str, Any],
    defaults: ProcgenConfig | None = None,
    callbacks: ControlCallbacks | None = None,
) -> None:
    """Populate Procgen-specific configuration widgets."""

    del game_id  # All Procgen games share the same knobs

    def emit(key: str, value: Any) -> None:
        overrides[key] = value
        if callbacks:
            callbacks.on_change(key, value)

    cfg = defaults if isinstance(defaults, ProcgenConfig) else ProcgenConfig()

    # -------- Display Resolution --------
    resolution_combo = QtWidgets.QComboBox(parent)
    resolution_combo.addItems(list(RESOLUTION_OPTIONS.keys()))

    # Find current resolution based on render_scale
    current_scale = overrides.get("render_scale", cfg.render_scale)
    current_resolution = "2048x2048 (4x)"  # default
    for name, scale in RESOLUTION_OPTIONS.items():
        if scale == current_scale:
            current_resolution = name
            break
    resolution_combo.setCurrentText(current_resolution)

    def on_resolution_changed(text: str) -> None:
        scale = RESOLUTION_OPTIONS.get(text, 4)
        emit("render_scale", scale)

    resolution_combo.currentTextChanged.connect(on_resolution_changed)
    layout.addRow("Display Resolution", resolution_combo)

    # -------- Distribution Mode --------
    mode_combo = QtWidgets.QComboBox(parent)
    mode_options = ["easy", "hard", "extreme", "memory", "exploration"]
    mode_combo.addItems(mode_options)
    mode_combo.setCurrentText(overrides.get("distribution_mode", cfg.distribution_mode))
    mode_combo.currentTextChanged.connect(lambda text: emit("distribution_mode", str(text)))
    layout.addRow("Difficulty", mode_combo)

    # -------- Number of Levels --------
    levels_spin = QtWidgets.QSpinBox(parent)
    levels_spin.setRange(0, 100000)
    levels_spin.setSpecialValueText("Unlimited")
    levels_spin.setValue(int(overrides.get("num_levels", cfg.num_levels)))
    levels_spin.setToolTip("0 = unlimited levels (for generalization testing)")
    levels_spin.valueChanged.connect(lambda value: emit("num_levels", int(value)))
    layout.addRow("Number of Levels", levels_spin)

    # -------- Start Level --------
    start_spin = QtWidgets.QSpinBox(parent)
    start_spin.setRange(0, 100000)
    start_spin.setValue(int(overrides.get("start_level", cfg.start_level)))
    start_spin.setToolTip("Starting level seed for reproducibility")
    start_spin.valueChanged.connect(lambda value: emit("start_level", int(value)))
    layout.addRow("Start Level", start_spin)

    # -------- Visual Options --------
    for label, key, default in (
        ("Use backgrounds", "use_backgrounds", cfg.use_backgrounds),
        ("Center agent", "center_agent", cfg.center_agent),
        ("Sequential levels", "use_sequential_levels", cfg.use_sequential_levels),
        ("Paint velocity info", "paint_vel_info", cfg.paint_vel_info),
    ):
        checkbox = QtWidgets.QCheckBox(label, parent)
        checkbox.setChecked(bool(overrides.get(key, default)))
        checkbox.toggled.connect(lambda checked, name=key: emit(name, bool(checked)))
        layout.addRow(checkbox)

    # Brief tip
    guidance = QtWidgets.QLabel(
        "<i>See Game Info tab for keyboard controls.</i>",
        parent,
    )
    guidance.setWordWrap(True)
    layout.addRow("", guidance)
