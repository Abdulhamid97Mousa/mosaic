"""Human control configuration widgets for ViZDoom environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.core.adapters.vizdoom import ViZDoomConfig
from gym_gui.core.enums import GameId

VIZDOOM_GAME_IDS: tuple[GameId, ...] = (
    GameId.VIZDOOM_BASIC,
    GameId.VIZDOOM_DEADLY_CORRIDOR,
    GameId.VIZDOOM_DEFEND_THE_CENTER,
    GameId.VIZDOOM_DEFEND_THE_LINE,
    GameId.VIZDOOM_HEALTH_GATHERING,
    GameId.VIZDOOM_HEALTH_GATHERING_SUPREME,
    GameId.VIZDOOM_MY_WAY_HOME,
    GameId.VIZDOOM_PREDICT_POSITION,
    GameId.VIZDOOM_TAKE_COVER,
    GameId.VIZDOOM_DEATHMATCH,
)


@dataclass(slots=True)
class ControlCallbacks:
    """Bridge callbacks for propagating UI changes to session state."""

    on_change: Callable[[str, Any], None]


def build_vizdoom_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    game_id: GameId,
    overrides: Dict[str, Any],
    defaults: ViZDoomConfig | None = None,
    callbacks: ControlCallbacks | None = None,
) -> None:
    """Populate ViZDoom-specific configuration widgets."""

    del game_id  # All scenarios share the same knobs for now

    def emit(key: str, value: Any) -> None:
        overrides[key] = value
        if callbacks:
            callbacks.on_change(key, value)

    cfg = defaults if isinstance(defaults, ViZDoomConfig) else ViZDoomConfig()

    # -------- Screen resolution --------
    resolution_combo = QtWidgets.QComboBox(parent)
    resolution_options = [
        "RES_320X240",
        "RES_640X480",
        "RES_800X600",
        "RES_1024X768",
    ]
    if cfg.screen_resolution not in resolution_options:
        resolution_options.append(cfg.screen_resolution)
    resolution_combo.addItems(resolution_options)
    resolution_combo.setCurrentText(cfg.screen_resolution)
    resolution_combo.currentTextChanged.connect(lambda text: emit("screen_resolution", str(text)))
    layout.addRow("Resolution", resolution_combo)

    # -------- Screen format --------
    fmt_combo = QtWidgets.QComboBox(parent)
    fmt_combo.addItems(["RGB24", "RGBA32", "GRAY8"])
    fmt_combo.setCurrentText(cfg.screen_format if isinstance(cfg.screen_format, str) else "RGB24")
    fmt_combo.currentTextChanged.connect(lambda text: emit("screen_format", str(text)))
    layout.addRow("Format", fmt_combo)

    # -------- Render toggles --------
    for label, key, default in (
        ("Show HUD", "render_hud", cfg.render_hud),
        ("Show weapon", "render_weapon", cfg.render_weapon),
        ("Show crosshair", "render_crosshair", cfg.render_crosshair),
        ("Particles", "render_particles", cfg.render_particles),
        ("Decals", "render_decals", cfg.render_decals),
        ("Depth buffer", "depth_buffer", cfg.depth_buffer),
        ("Labels buffer", "labels_buffer", cfg.labels_buffer),
        ("Automap buffer", "automap_buffer", cfg.automap_buffer),
    ):
        checkbox = QtWidgets.QCheckBox(label, parent)
        checkbox.setChecked(bool(overrides.get(key, default)))
        checkbox.toggled.connect(lambda checked, name=key: emit(name, bool(checked)))
        layout.addRow(checkbox)

    # -------- Sound option (disabled - causes crashes in embedded mode) --------
    sound_checkbox = QtWidgets.QCheckBox("Enable sound", parent)
    sound_checkbox.setChecked(False)
    sound_checkbox.setEnabled(False)
    sound_checkbox.setToolTip(
        "Sound is disabled: ViZDoom's internal sound system is incompatible\n"
        "with embedded Qt applications and causes crashes."
    )
    # Force sound_enabled to False in overrides to prevent crashes
    overrides["sound_enabled"] = False
    layout.addRow(sound_checkbox)

    # -------- Episode timeout (disabled by default - play until death) --------
    timeout_container = QtWidgets.QWidget(parent)
    timeout_layout = QtWidgets.QHBoxLayout(timeout_container)
    timeout_layout.setContentsMargins(0, 0, 0, 0)

    timeout_checkbox = QtWidgets.QCheckBox("Enable timeout", timeout_container)
    timeout_spin = QtWidgets.QSpinBox(timeout_container)
    timeout_spin.setRange(100, 10000)
    timeout_spin.setValue(int(overrides.get("episode_timeout", cfg.episode_timeout)) or 2100)
    timeout_spin.setSuffix(" tics")

    # Default to disabled (play until death)
    has_timeout = overrides.get("episode_timeout", 0) > 0
    timeout_checkbox.setChecked(has_timeout)
    timeout_spin.setEnabled(has_timeout)

    def on_timeout_toggle(checked: bool) -> None:
        timeout_spin.setEnabled(checked)
        if checked:
            emit("episode_timeout", timeout_spin.value())
        else:
            emit("episode_timeout", 0)  # 0 = no timeout

    timeout_checkbox.toggled.connect(on_timeout_toggle)
    timeout_spin.valueChanged.connect(lambda value: emit("episode_timeout", int(value)) if timeout_checkbox.isChecked() else None)

    timeout_layout.addWidget(timeout_checkbox)
    timeout_layout.addWidget(timeout_spin)
    timeout_layout.addStretch()
    layout.addRow("Episode timeout", timeout_container)

    # Set default to no timeout if not already set
    if "episode_timeout" not in overrides:
        overrides["episode_timeout"] = 0

    # -------- Living reward --------
    living_spin = QtWidgets.QDoubleSpinBox(parent)
    living_spin.setRange(-10.0, 10.0)
    living_spin.setSingleStep(0.1)
    living_spin.setValue(float(overrides.get("living_reward", cfg.living_reward)))
    living_spin.valueChanged.connect(lambda value: emit("living_reward", float(value)))
    layout.addRow("Living reward", living_spin)

    # -------- Death penalty --------
    death_spin = QtWidgets.QDoubleSpinBox(parent)
    death_spin.setRange(0.0, 500.0)
    death_spin.setSingleStep(5.0)
    death_spin.setValue(float(overrides.get("death_penalty", cfg.death_penalty)))
    death_spin.valueChanged.connect(lambda value: emit("death_penalty", float(value)))
    layout.addRow("Death penalty", death_spin)

    # Brief tip - detailed controls are in Game Info tab
    guidance = QtWidgets.QLabel(
        "<i>See Game Info tab for keyboard controls and mouse capture instructions.</i>",
        parent,
    )
    guidance.setWordWrap(True)
    layout.addRow("", guidance)
