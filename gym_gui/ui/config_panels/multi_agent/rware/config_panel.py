"""Configuration widgets for RWARE (Robotic Warehouse) multi-agent environments."""

from __future__ import annotations

from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.config.game_configs import RWAREConfig
from gym_gui.core.enums import GameId


ALL_RWARE_GAME_IDS: tuple[GameId, ...] = (
    GameId.RWARE_TINY_2AG,
    GameId.RWARE_TINY_4AG,
    GameId.RWARE_SMALL_2AG,
    GameId.RWARE_SMALL_4AG,
    GameId.RWARE_MEDIUM_2AG,
    GameId.RWARE_MEDIUM_4AG,
    GameId.RWARE_MEDIUM_4AG_EASY,
    GameId.RWARE_MEDIUM_4AG_HARD,
    GameId.RWARE_LARGE_4AG,
    GameId.RWARE_LARGE_4AG_HARD,
    GameId.RWARE_LARGE_8AG,
    GameId.RWARE_LARGE_8AG_HARD,
)

# Per-variant metadata: GameId -> (n_agents, size_label, difficulty)
_MAP_INFO: dict[GameId, tuple[int, str, str]] = {
    GameId.RWARE_TINY_2AG: (2, "Tiny (1x3)", "Normal"),
    GameId.RWARE_TINY_4AG: (4, "Tiny (1x3)", "Normal"),
    GameId.RWARE_SMALL_2AG: (2, "Small (2x3)", "Normal"),
    GameId.RWARE_SMALL_4AG: (4, "Small (2x3)", "Normal"),
    GameId.RWARE_MEDIUM_2AG: (2, "Medium (2x5)", "Normal"),
    GameId.RWARE_MEDIUM_4AG: (4, "Medium (2x5)", "Normal"),
    GameId.RWARE_MEDIUM_4AG_EASY: (4, "Medium (2x5)", "Easy"),
    GameId.RWARE_MEDIUM_4AG_HARD: (4, "Medium (2x5)", "Hard"),
    GameId.RWARE_LARGE_4AG: (4, "Large (3x5)", "Normal"),
    GameId.RWARE_LARGE_4AG_HARD: (4, "Large (3x5)", "Hard"),
    GameId.RWARE_LARGE_8AG: (8, "Large (3x5)", "Normal"),
    GameId.RWARE_LARGE_8AG_HARD: (8, "Large (3x5)", "Hard"),
}

# Observation type display -> config value
_OBS_OPTIONS: dict[str, str] = {
    "Flattened (1D vector)": "flattened",
    "Dict (nested)": "dict",
    "Image (multi-channel grid)": "image",
    "Image + Dict": "image_dict",
}

# Reward type display -> config value
_REWARD_OPTIONS: dict[str, str] = {
    "Individual (per-agent)": "individual",
    "Global (shared)": "global",
    "Two-Stage (delivery + return)": "two_stage",
}


def build_rware_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    game_id: GameId,
    overrides: Dict[str, Any],
    on_change: Callable[[Dict[str, Any]], None] | None = None,
) -> None:
    """Populate RWARE-specific configuration widgets.

    Args:
        parent: Parent widget for new controls.
        layout: Form layout to add rows to.
        game_id: Currently selected RWARE GameId.
        overrides: Mutable dict of current config overrides.
        on_change: Callback invoked with updated overrides dict.
    """
    defaults = RWAREConfig()

    def _emit() -> None:
        if on_change is not None:
            on_change(overrides)

    # ── Map info (read-only) ─────────────────────────────────────────
    info = _MAP_INFO.get(game_id)
    if info is not None:
        n_agents, size_label, difficulty = info
        info_label = QtWidgets.QLabel(
            f"{size_label} | {n_agents} agents | {difficulty}",
            parent,
        )
        info_label.setStyleSheet("color: #aaa; font-style: italic;")
        layout.addRow("Map:", info_label)

    # ── Observation type ─────────────────────────────────────────────
    obs_combo = QtWidgets.QComboBox(parent)
    obs_combo.addItems(list(_OBS_OPTIONS.keys()))
    current_obs = overrides.get("observation_type", defaults.observation_type)
    # Find and set the matching display key
    for idx, (display, value) in enumerate(_OBS_OPTIONS.items()):
        if value == current_obs:
            obs_combo.setCurrentIndex(idx)
            break

    def _on_obs_changed(index: int) -> None:
        values = list(_OBS_OPTIONS.values())
        overrides["observation_type"] = values[index]
        _emit()

    obs_combo.currentIndexChanged.connect(_on_obs_changed)
    layout.addRow("Observation:", obs_combo)

    # ── Sensor range ─────────────────────────────────────────────────
    sensor_spin = QtWidgets.QSpinBox(parent)
    sensor_spin.setRange(1, 5)
    sensor_spin.setValue(int(overrides.get("sensor_range", defaults.sensor_range)))
    sensor_spin.setToolTip("Number of cells visible around each agent (1-5)")

    def _on_sensor_changed(val: int) -> None:
        overrides["sensor_range"] = val
        _emit()

    sensor_spin.valueChanged.connect(_on_sensor_changed)
    layout.addRow("Sensor Range:", sensor_spin)

    # ── Reward type ──────────────────────────────────────────────────
    reward_combo = QtWidgets.QComboBox(parent)
    reward_combo.addItems(list(_REWARD_OPTIONS.keys()))
    current_reward = overrides.get("reward_type", defaults.reward_type)
    for idx, (display, value) in enumerate(_REWARD_OPTIONS.items()):
        if value == current_reward:
            reward_combo.setCurrentIndex(idx)
            break

    def _on_reward_changed(index: int) -> None:
        values = list(_REWARD_OPTIONS.values())
        overrides["reward_type"] = values[index]
        _emit()

    reward_combo.currentIndexChanged.connect(_on_reward_changed)
    layout.addRow("Reward:", reward_combo)

    # ── Communication bits ───────────────────────────────────────────
    msg_spin = QtWidgets.QSpinBox(parent)
    msg_spin.setRange(0, 8)
    msg_spin.setValue(int(overrides.get("msg_bits", defaults.msg_bits)))
    msg_spin.setToolTip("Communication channels per agent (0 = silent)")

    def _on_msg_changed(val: int) -> None:
        overrides["msg_bits"] = val
        _emit()

    msg_spin.valueChanged.connect(_on_msg_changed)
    layout.addRow("Comm Bits:", msg_spin)

    # ── Max steps ────────────────────────────────────────────────────
    steps_spin = QtWidgets.QSpinBox(parent)
    steps_spin.setRange(100, 10000)
    steps_spin.setSingleStep(100)
    steps_spin.setValue(int(overrides.get("max_steps", defaults.max_steps)))

    def _on_steps_changed(val: int) -> None:
        overrides["max_steps"] = val
        _emit()

    steps_spin.valueChanged.connect(_on_steps_changed)
    layout.addRow("Max Steps:", steps_spin)

    # ── Seed ─────────────────────────────────────────────────────────
    seed_spin = QtWidgets.QSpinBox(parent)
    seed_spin.setRange(-1, 99999)
    seed_spin.setSpecialValueText("Random")
    seed_val = overrides.get("seed")
    seed_spin.setValue(int(seed_val) if seed_val is not None and int(seed_val) >= 0 else -1)
    seed_spin.setToolTip("-1 = random seed each episode")

    def _on_seed_changed(val: int) -> None:
        overrides["seed"] = val if val >= 0 else None
        _emit()

    seed_spin.valueChanged.connect(_on_seed_changed)
    layout.addRow("Seed:", seed_spin)


__all__ = [
    "ALL_RWARE_GAME_IDS",
    "build_rware_controls",
]
