"""UI helpers for classic Gym environment configuration panels."""

from __future__ import annotations

from typing import Any, Dict

from PyQt6 import QtCore, QtWidgets

from gym_gui.config.game_configs import (
    BipedalWalkerConfig,
    CarRacingConfig,
    CliffWalkingConfig,
    FrozenLakeConfig,
    LunarLanderConfig,
    TaxiConfig,
    DEFAULT_FROZEN_LAKE_V2_CONFIG,
)
from gym_gui.core.enums import GameId


TOY_TEXT_FAMILY = {
    GameId.FROZEN_LAKE,
    GameId.FROZEN_LAKE_V2,
    GameId.CLIFF_WALKING,
    GameId.TAXI,
    GameId.BLACKJACK,
}

BOX2D_FAMILY = {
    GameId.LUNAR_LANDER,
    GameId.CAR_RACING,
    GameId.BIPEDAL_WALKER,
}


def build_frozenlake_controls(
    *,
    layout: QtWidgets.QFormLayout,
    group: QtWidgets.QWidget,
    overrides: Dict[str, Any],
    config: FrozenLakeConfig,
    on_slippery_toggled: Any,
) -> QtWidgets.QCheckBox:
    overrides.setdefault("is_slippery", config.is_slippery)
    checkbox = QtWidgets.QCheckBox("Enable slippery ice (stochastic)", group)
    checkbox.setChecked(bool(overrides["is_slippery"]))
    checkbox.stateChanged.connect(
        lambda state: on_slippery_toggled(state == QtCore.Qt.CheckState.Checked.value)
    )
    layout.addRow("Slippery ice", checkbox)
    return checkbox


def build_frozenlake_v2_controls(
    *,
    layout: QtWidgets.QFormLayout,
    group: QtWidgets.QWidget,
    overrides: Dict[str, Any],
    on_change: Any,
) -> None:
    defaults = DEFAULT_FROZEN_LAKE_V2_CONFIG

    def emit(name: str, value: Any) -> None:
        on_change(name, value)

    overrides.setdefault("is_slippery", defaults.is_slippery)
    overrides.setdefault("grid_height", defaults.grid_height)
    overrides.setdefault("grid_width", defaults.grid_width)
    overrides.setdefault("start_position", defaults.start_position or (0, 0))
    overrides.setdefault(
        "goal_position",
        defaults.goal_position or (defaults.grid_height - 1, defaults.grid_width - 1),
    )
    overrides.setdefault("hole_count", defaults.hole_count or 10)
    overrides.setdefault("random_holes", defaults.random_holes)

    slippery_checkbox = QtWidgets.QCheckBox("Enable slippery ice (stochastic)", group)
    slippery_checkbox.setChecked(bool(overrides["is_slippery"]))
    slippery_checkbox.stateChanged.connect(
        lambda state: emit("is_slippery", state == QtCore.Qt.CheckState.Checked.value)
    )
    layout.addRow("Slippery ice", slippery_checkbox)

    height_spin = QtWidgets.QSpinBox(group)
    height_spin.setRange(4, 20)
    height_spin.setValue(int(overrides["grid_height"]))
    height_spin.valueChanged.connect(lambda value: emit("grid_height", int(value)))
    layout.addRow("Grid Height", height_spin)

    width_spin = QtWidgets.QSpinBox(group)
    width_spin.setRange(4, 20)
    width_spin.setValue(int(overrides["grid_width"]))
    width_spin.valueChanged.connect(lambda value: emit("grid_width", int(value)))
    layout.addRow("Grid Width", width_spin)

    start_combo = QtWidgets.QComboBox(group)
    start_positions = [
        (r, c)
        for r in range(height_spin.value())
        for c in range(width_spin.value())
    ]
    for pos in start_positions:
        start_combo.addItem(f"({pos[0]}, {pos[1]})", pos)
    start_idx = start_combo.findData(overrides["start_position"])
    start_combo.setCurrentIndex(start_idx if start_idx >= 0 else 0)
    start_combo.currentIndexChanged.connect(
        lambda idx: emit("start_position", start_combo.itemData(idx))
    )
    layout.addRow("Start Position", start_combo)

    goal_combo = QtWidgets.QComboBox(group)
    for pos in start_positions:
        if pos != overrides["start_position"]:
            goal_combo.addItem(f"({pos[0]}, {pos[1]})", pos)
    goal_idx = goal_combo.findData(overrides["goal_position"])
    goal_combo.setCurrentIndex(goal_idx if goal_idx >= 0 else goal_combo.count() - 1)
    goal_combo.currentIndexChanged.connect(
        lambda idx: emit("goal_position", goal_combo.itemData(idx))
    )
    layout.addRow("Goal Position", goal_combo)

    hole_spin = QtWidgets.QSpinBox(group)
    hole_spin.setRange(0, (height_spin.value() * width_spin.value()) - 2)
    hole_spin.setValue(int(overrides["hole_count"]))
    hole_spin.valueChanged.connect(lambda value: emit("hole_count", int(value)))
    layout.addRow("Hole Count", hole_spin)

    random_checkbox = QtWidgets.QCheckBox("Random hole placement", group)
    random_checkbox.setChecked(bool(overrides["random_holes"]))
    random_checkbox.stateChanged.connect(
        lambda state: emit("random_holes", state == QtCore.Qt.CheckState.Checked.value)
    )
    layout.addRow("Random Holes", random_checkbox)


def build_taxi_controls(
    *,
    layout: QtWidgets.QFormLayout,
    group: QtWidgets.QWidget,
    overrides: Dict[str, Any],
    config: TaxiConfig,
    on_change: Any,
) -> None:
    overrides.setdefault("is_raining", config.is_raining)
    overrides.setdefault("fickle_passenger", config.fickle_passenger)

    rain_checkbox = QtWidgets.QCheckBox("Enable rain (80% move success)", group)
    rain_checkbox.setChecked(bool(overrides["is_raining"]))
    rain_checkbox.stateChanged.connect(
        lambda state: on_change("is_raining", state == QtCore.Qt.CheckState.Checked.value)
    )
    layout.addRow("Rain", rain_checkbox)

    fickle_checkbox = QtWidgets.QCheckBox("Fickle passenger (30% dest change)", group)
    fickle_checkbox.setChecked(bool(overrides["fickle_passenger"]))
    fickle_checkbox.stateChanged.connect(
        lambda state: on_change("fickle_passenger", state == QtCore.Qt.CheckState.Checked.value)
    )
    layout.addRow("Fickle", fickle_checkbox)


def build_cliff_controls(
    *,
    layout: QtWidgets.QFormLayout,
    group: QtWidgets.QWidget,
    overrides: Dict[str, Any],
    config: CliffWalkingConfig,
    on_change: Any,
) -> None:
    overrides.setdefault("is_slippery", config.is_slippery)
    checkbox = QtWidgets.QCheckBox("Enable slippery cliff (stochastic)", group)
    checkbox.setChecked(bool(overrides["is_slippery"]))
    checkbox.stateChanged.connect(
        lambda state: on_change("is_slippery", state == QtCore.Qt.CheckState.Checked.value)
    )
    layout.addRow("Slippery", checkbox)


def build_lunarlander_controls(
    *,
    layout: QtWidgets.QFormLayout,
    group: QtWidgets.QWidget,
    overrides: Dict[str, Any],
    config: LunarLanderConfig,
    on_change: Any,
) -> None:
    overrides.setdefault("continuous", config.continuous)
    overrides.setdefault("gravity", config.gravity)
    overrides.setdefault("enable_wind", config.enable_wind)
    overrides.setdefault("wind_power", config.wind_power)
    overrides.setdefault("turbulence_power", config.turbulence_power)
    overrides.setdefault("max_episode_steps", config.max_episode_steps)

    def emit(name: str, value: Any) -> None:
        on_change(name, value)

    continuous_checkbox = QtWidgets.QCheckBox("Continuous control (Box actions)", group)
    continuous_checkbox.setChecked(bool(overrides["continuous"]))
    continuous_checkbox.toggled.connect(lambda checked: emit("continuous", bool(checked)))
    layout.addRow("Action space", continuous_checkbox)

    gravity_spin = QtWidgets.QDoubleSpinBox(group)
    gravity_spin.setRange(-12.0, 0.0)
    gravity_spin.setSingleStep(0.1)
    gravity_spin.setValue(float(overrides["gravity"]))
    gravity_spin.valueChanged.connect(lambda value: emit("gravity", float(value)))
    layout.addRow("Gravity", gravity_spin)

    wind_checkbox = QtWidgets.QCheckBox("Enable wind", group)
    wind_checkbox.setChecked(bool(overrides["enable_wind"]))
    layout.addRow("Wind", wind_checkbox)

    wind_spin = QtWidgets.QDoubleSpinBox(group)
    wind_spin.setRange(0.0, 20.0)
    wind_spin.setSingleStep(0.5)
    wind_spin.setDecimals(2)
    wind_spin.setValue(float(overrides["wind_power"]))
    wind_spin.setEnabled(wind_checkbox.isChecked())
    wind_spin.valueChanged.connect(lambda value: emit("wind_power", float(value)))
    layout.addRow("Wind power", wind_spin)

    turbulence_spin = QtWidgets.QDoubleSpinBox(group)
    turbulence_spin.setRange(0.0, 5.0)
    turbulence_spin.setSingleStep(0.1)
    turbulence_spin.setDecimals(2)
    turbulence_spin.setValue(float(overrides["turbulence_power"]))
    turbulence_spin.setEnabled(wind_checkbox.isChecked())
    turbulence_spin.valueChanged.connect(lambda value: emit("turbulence_power", float(value)))
    layout.addRow("Turbulence", turbulence_spin)

    def handle_wind(state: int) -> None:
        enabled = state == QtCore.Qt.CheckState.Checked.value
        emit("enable_wind", enabled)
        wind_spin.setEnabled(enabled)
        turbulence_spin.setEnabled(enabled)

    wind_checkbox.stateChanged.connect(handle_wind)


def build_car_racing_controls(
    *,
    layout: QtWidgets.QFormLayout,
    group: QtWidgets.QWidget,
    overrides: Dict[str, Any],
    config: CarRacingConfig,
    on_change: Any,
) -> None:
    overrides.setdefault("continuous", config.continuous)
    overrides.setdefault("domain_randomize", config.domain_randomize)
    overrides.setdefault("lap_complete_percent", config.lap_complete_percent)
    overrides.setdefault("max_episode_steps", config.max_episode_steps)
    overrides.setdefault("max_episode_seconds", config.max_episode_seconds)

    def emit(name: str, value: Any) -> None:
        on_change(name, value)

    continuous_checkbox = QtWidgets.QCheckBox("Continuous control (Box actions)", group)
    continuous_checkbox.setChecked(bool(overrides["continuous"]))
    continuous_checkbox.toggled.connect(lambda checked: emit("continuous", bool(checked)))
    layout.addRow("Action space", continuous_checkbox)

    domain_checkbox = QtWidgets.QCheckBox("Enable domain randomization", group)
    domain_checkbox.setChecked(bool(overrides["domain_randomize"]))
    domain_checkbox.toggled.connect(lambda checked: emit("domain_randomize", bool(checked)))
    layout.addRow("Domain randomize", domain_checkbox)

    lap_spin = QtWidgets.QDoubleSpinBox(group)
    lap_spin.setRange(0.50, 1.00)
    lap_spin.setSingleStep(0.01)
    lap_spin.setDecimals(2)
    lap_spin.setValue(float(overrides["lap_complete_percent"]))
    lap_spin.valueChanged.connect(lambda value: emit("lap_complete_percent", float(value)))
    layout.addRow("Lap completion", lap_spin)

    steps_spin = QtWidgets.QSpinBox(group)
    steps_value = overrides["max_episode_steps"] or 0
    steps_spin.setRange(0, 20000)
    steps_spin.setSpecialValueText("Disabled (unlimited)")
    steps_spin.setValue(int(steps_value) if steps_value else 0)
    steps_spin.valueChanged.connect(
        lambda value: emit("max_episode_steps", None if value == 0 else int(value))
    )
    layout.addRow("Max steps", steps_spin)

    seconds_spin = QtWidgets.QDoubleSpinBox(group)
    seconds_value = overrides["max_episode_seconds"] or 0.0
    seconds_spin.setRange(0.0, 3600.0)
    seconds_spin.setSingleStep(5.0)
    seconds_spin.setDecimals(1)
    seconds_spin.setSpecialValueText("Use Gym default (disabled)")
    seconds_spin.setValue(float(seconds_value) if seconds_value else 0.0)
    seconds_spin.valueChanged.connect(
        lambda value: emit("max_episode_seconds", None if value == 0 else float(value))
    )
    layout.addRow("Time limit (s)", seconds_spin)


def build_bipedal_controls(
    *,
    layout: QtWidgets.QFormLayout,
    group: QtWidgets.QWidget,
    overrides: Dict[str, Any],
    config: BipedalWalkerConfig,
    on_change: Any,
) -> None:
    overrides.setdefault("hardcore", config.hardcore)
    overrides.setdefault("max_episode_steps", config.max_episode_steps)
    overrides.setdefault("max_episode_seconds", config.max_episode_seconds)

    def emit(name: str, value: Any) -> None:
        on_change(name, value)

    hardcore_checkbox = QtWidgets.QCheckBox("Enable hardcore terrain", group)
    hardcore_checkbox.setChecked(bool(overrides["hardcore"]))
    hardcore_checkbox.toggled.connect(lambda checked: emit("hardcore", bool(checked)))
    layout.addRow("Hardcore mode", hardcore_checkbox)

    steps_spin = QtWidgets.QSpinBox(group)
    steps_value = overrides["max_episode_steps"] or 0
    steps_spin.setRange(0, 20000)
    steps_spin.setSpecialValueText("Use Gym default (disabled)")
    steps_spin.setValue(int(steps_value) if steps_value else 0)
    steps_spin.valueChanged.connect(
        lambda value: emit("max_episode_steps", None if value == 0 else int(value))
    )
    layout.addRow("Max steps", steps_spin)

    seconds_spin = QtWidgets.QDoubleSpinBox(group)
    seconds_value = overrides["max_episode_seconds"] or 0.0
    seconds_spin.setRange(0.0, 3600.0)
    seconds_spin.setSingleStep(5.0)
    seconds_spin.setDecimals(1)
    seconds_spin.setSpecialValueText("Use Gym default (disabled)")
    seconds_spin.setValue(float(seconds_value) if seconds_value else 0.0)
    seconds_spin.valueChanged.connect(
        lambda value: emit("max_episode_seconds", None if value == 0 else float(value))
    )
    layout.addRow("Time limit (s)", seconds_spin)


__all__ = [
    "TOY_TEXT_FAMILY",
    "BOX2D_FAMILY",
    "build_frozenlake_controls",
    "build_frozenlake_v2_controls",
    "build_taxi_controls",
    "build_cliff_controls",
    "build_lunarlander_controls",
    "build_car_racing_controls",
    "build_bipedal_controls",
]
