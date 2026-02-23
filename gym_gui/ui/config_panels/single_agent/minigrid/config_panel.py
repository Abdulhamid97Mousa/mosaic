"""UI helpers for MiniGrid environment configuration panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.config.game_configs import (
    DEFAULT_MINIGRID_DOORKEY_16x16_CONFIG,
    DEFAULT_MINIGRID_DOORKEY_5x5_CONFIG,
    DEFAULT_MINIGRID_DOORKEY_6x6_CONFIG,
    DEFAULT_MINIGRID_DOORKEY_8x8_CONFIG,
    DEFAULT_MINIGRID_EMPTY_5x5_CONFIG,
    DEFAULT_MINIGRID_EMPTY_RANDOM_5x5_CONFIG,
    DEFAULT_MINIGRID_EMPTY_6x6_CONFIG,
    DEFAULT_MINIGRID_EMPTY_RANDOM_6x6_CONFIG,
    DEFAULT_MINIGRID_EMPTY_8x8_CONFIG,
    DEFAULT_MINIGRID_EMPTY_16x16_CONFIG,
    DEFAULT_MINIGRID_LAVAGAP_S7_CONFIG,
    DEFAULT_MINIGRID_REDBLUE_DOORS_6x6_CONFIG,
    DEFAULT_MINIGRID_REDBLUE_DOORS_8x8_CONFIG,
    MiniGridConfig,
)
from gym_gui.core.enums import GameId


MINIGRID_GAME_IDS: tuple[GameId, ...] = (
    GameId.MINIGRID_EMPTY_5x5,
    GameId.MINIGRID_EMPTY_RANDOM_5x5,
    GameId.MINIGRID_EMPTY_6x6,
    GameId.MINIGRID_EMPTY_RANDOM_6x6,
    GameId.MINIGRID_EMPTY_8x8,
    GameId.MINIGRID_EMPTY_16x16,
    GameId.MINIGRID_DOORKEY_5x5,
    GameId.MINIGRID_DOORKEY_6x6,
    GameId.MINIGRID_DOORKEY_8x8,
    GameId.MINIGRID_DOORKEY_16x16,
    GameId.MINIGRID_LAVAGAP_S7,
    GameId.MINIGRID_REDBLUE_DOORS_6x6,
    GameId.MINIGRID_REDBLUE_DOORS_8x8,
)


_DEFAULT_LOOKUP: Dict[GameId, MiniGridConfig] = {
    GameId.MINIGRID_EMPTY_5x5: DEFAULT_MINIGRID_EMPTY_5x5_CONFIG,
    GameId.MINIGRID_EMPTY_RANDOM_5x5: DEFAULT_MINIGRID_EMPTY_RANDOM_5x5_CONFIG,
    GameId.MINIGRID_EMPTY_6x6: DEFAULT_MINIGRID_EMPTY_6x6_CONFIG,
    GameId.MINIGRID_EMPTY_RANDOM_6x6: DEFAULT_MINIGRID_EMPTY_RANDOM_6x6_CONFIG,
    GameId.MINIGRID_EMPTY_8x8: DEFAULT_MINIGRID_EMPTY_8x8_CONFIG,
    GameId.MINIGRID_EMPTY_16x16: DEFAULT_MINIGRID_EMPTY_16x16_CONFIG,
    GameId.MINIGRID_DOORKEY_5x5: DEFAULT_MINIGRID_DOORKEY_5x5_CONFIG,
    GameId.MINIGRID_DOORKEY_6x6: DEFAULT_MINIGRID_DOORKEY_6x6_CONFIG,
    GameId.MINIGRID_DOORKEY_8x8: DEFAULT_MINIGRID_DOORKEY_8x8_CONFIG,
    GameId.MINIGRID_DOORKEY_16x16: DEFAULT_MINIGRID_DOORKEY_16x16_CONFIG,
    GameId.MINIGRID_LAVAGAP_S7: DEFAULT_MINIGRID_LAVAGAP_S7_CONFIG,
    GameId.MINIGRID_REDBLUE_DOORS_6x6: DEFAULT_MINIGRID_REDBLUE_DOORS_6x6_CONFIG,
    GameId.MINIGRID_REDBLUE_DOORS_8x8: DEFAULT_MINIGRID_REDBLUE_DOORS_8x8_CONFIG,
}


def resolve_default_config(game_id: GameId) -> MiniGridConfig:
    """Return the default configuration for the given MiniGrid environment."""

    return _DEFAULT_LOOKUP.get(game_id, DEFAULT_MINIGRID_EMPTY_5x5_CONFIG)


@dataclass(slots=True)
class ControlCallbacks:
    """Callback container used to notify control panel of config changes."""

    on_change: Callable[[str, Any], None]


def build_minigrid_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    game_id: GameId,
    overrides: Dict[str, Any],
    defaults: MiniGridConfig,
    callbacks: ControlCallbacks,
) -> None:
    """Populate MiniGrid-specific controls into the provided layout."""

    def emit_change(key: str, value: Any) -> None:
        callbacks.on_change(key, value)

    partial = bool(overrides.get("partial_observation", defaults.partial_observation))
    overrides["partial_observation"] = partial
    partial_checkbox = QtWidgets.QCheckBox("Agent-centric partial observation", parent)
    partial_checkbox.setChecked(partial)
    partial_checkbox.toggled.connect(lambda checked: emit_change("partial_observation", bool(checked)))
    partial_checkbox.setToolTip("Use MiniGrid's RGBImgPartialObsWrapper for egocentric views.")
    layout.addRow("Partial View", partial_checkbox)

    image_obs = bool(overrides.get("image_observation", defaults.image_observation))
    overrides["image_observation"] = image_obs
    image_checkbox = QtWidgets.QCheckBox("Flatten RGB image observations", parent)
    image_checkbox.setChecked(image_obs)
    image_checkbox.toggled.connect(lambda checked: emit_change("image_observation", bool(checked)))
    image_checkbox.setToolTip("Apply ImgObsWrapper before flattening observations for agents.")
    layout.addRow("Image Wrapper", image_checkbox)

    reward_raw: Any = overrides.get("reward_multiplier", defaults.reward_multiplier)
    try:
        reward_multiplier = float(reward_raw)
    except (TypeError, ValueError):
        reward_multiplier = float(defaults.reward_multiplier)
    overrides["reward_multiplier"] = reward_multiplier
    reward_spin = QtWidgets.QDoubleSpinBox(parent)
    reward_spin.setRange(0.1, 100.0)
    reward_spin.setSingleStep(0.5)
    reward_spin.setDecimals(2)
    reward_spin.setValue(reward_multiplier)
    reward_spin.valueChanged.connect(lambda value: emit_change("reward_multiplier", float(value)))
    reward_spin.setToolTip("Scale environment rewards (xuance default = 10).")
    layout.addRow("Reward ×", reward_spin)

    agent_view_raw: Any = overrides.get("agent_view_size", defaults.agent_view_size)
    agent_view_value = (
        int(agent_view_raw)
        if isinstance(agent_view_raw, (int, float)) and int(agent_view_raw) > 0
        else 0
    )
    overrides["agent_view_size"] = agent_view_raw
    agent_view_spin = QtWidgets.QSpinBox(parent)
    agent_view_spin.setRange(0, 15)
    agent_view_spin.setSpecialValueText("Environment default")
    agent_view_spin.setValue(agent_view_value)
    agent_view_spin.valueChanged.connect(
        lambda value: emit_change("agent_view_size", None if int(value) == 0 else int(value))
    )
    agent_view_spin.setToolTip("Override agent view size (0 keeps MiniGrid default of 7).")
    layout.addRow("Agent View", agent_view_spin)

    step_limit_raw: Any = overrides.get("max_episode_steps", defaults.max_episode_steps)
    step_limit_value = (
        int(step_limit_raw)
        if isinstance(step_limit_raw, (int, float)) and int(step_limit_raw) > 0
        else 0
    )
    overrides["max_episode_steps"] = step_limit_raw
    step_limit_spin = QtWidgets.QSpinBox(parent)
    step_limit_spin.setRange(0, 20000)
    step_limit_spin.setSpecialValueText("Environment default")
    step_limit_spin.setValue(step_limit_value)
    step_limit_spin.valueChanged.connect(
        lambda value: emit_change("max_episode_steps", None if int(value) == 0 else int(value))
    )
    step_limit_spin.setToolTip("Override episode truncation length; 0 keeps MiniGrid default.")
    layout.addRow("Max Steps", step_limit_spin)

    guidance = QtWidgets.QLabel(
        "MiniGrid rewards are sparse; consider keeping the 10× multiplier for visibility.",
        parent,
    )
    guidance.setWordWrap(True)
    layout.addRow("", guidance)


__all__ = [
    "MINIGRID_GAME_IDS",
    "ControlCallbacks",
    "build_minigrid_controls",
    "resolve_default_config",
]
