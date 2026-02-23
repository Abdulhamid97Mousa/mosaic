"""UI helpers for Crafter environment configuration panels.

Crafter is an open-world survival game benchmark for reinforcement learning.
Paper: Hafner, D. (2022). Benchmarking the Spectrum of Agent Capabilities. ICLR 2022.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.config.game_configs import CrafterConfig
from gym_gui.core.enums import GameId


CRAFTER_GAME_IDS: tuple[GameId, ...] = (
    GameId.CRAFTER_REWARD,
    GameId.CRAFTER_NO_REWARD,
)


# Default configurations for Crafter variants
_DEFAULT_CRAFTER_REWARD = CrafterConfig(
    env_id=GameId.CRAFTER_REWARD.value,
    reward=True,
)

_DEFAULT_CRAFTER_NO_REWARD = CrafterConfig(
    env_id=GameId.CRAFTER_NO_REWARD.value,
    reward=False,
)

_DEFAULT_LOOKUP: Dict[GameId, CrafterConfig] = {
    GameId.CRAFTER_REWARD: _DEFAULT_CRAFTER_REWARD,
    GameId.CRAFTER_NO_REWARD: _DEFAULT_CRAFTER_NO_REWARD,
}


def resolve_default_config(game_id: GameId) -> CrafterConfig:
    """Return the default configuration for the given Crafter environment."""
    return _DEFAULT_LOOKUP.get(game_id, _DEFAULT_CRAFTER_REWARD)


@dataclass(slots=True)
class ControlCallbacks:
    """Callback container used to notify control panel of config changes."""

    on_change: Callable[[str, Any], None]


def build_crafter_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    game_id: GameId,
    overrides: Dict[str, Any],
    defaults: CrafterConfig,
    callbacks: ControlCallbacks,
) -> None:
    """Populate Crafter-specific controls into the provided layout."""

    def emit_change(key: str, value: Any) -> None:
        callbacks.on_change(key, value)

    # Reward multiplier
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
    reward_spin.setToolTip("Scale environment rewards (default = 1.0).")
    layout.addRow("Reward ×", reward_spin)

    # Max episode steps
    length_raw: Any = overrides.get("length", defaults.length)
    try:
        length_value = int(length_raw)
    except (TypeError, ValueError):
        length_value = int(defaults.length)
    overrides["length"] = length_value
    length_spin = QtWidgets.QSpinBox(parent)
    length_spin.setRange(1000, 100000)
    length_spin.setSingleStep(1000)
    length_spin.setValue(length_value)
    length_spin.valueChanged.connect(lambda value: emit_change("length", int(value)))
    length_spin.setToolTip("Maximum episode steps (default = 10,000).")
    layout.addRow("Max Steps", length_spin)

    # Seed input
    seed_raw: Any = overrides.get("seed", defaults.seed)
    seed_value = int(seed_raw) if seed_raw is not None else 0
    overrides["seed"] = seed_raw
    seed_spin = QtWidgets.QSpinBox(parent)
    seed_spin.setRange(0, 999999)
    seed_spin.setSpecialValueText("Random")
    seed_spin.setValue(seed_value)
    seed_spin.valueChanged.connect(
        lambda value: emit_change("seed", None if int(value) == 0 else int(value))
    )
    seed_spin.setToolTip("Random seed for world generation (0 = random).")
    layout.addRow("Seed", seed_spin)

    # Keyboard controls reference
    controls_label = QtWidgets.QLabel(
        "<b>Keyboard Controls:</b><br>"
        "WASD/Arrows: Move | Space: Interact | R: Sleep<br>"
        "1-4: Place (stone/table/furnace/plant)<br>"
        "Q/E/F: Make pickaxes (wood/stone/iron)<br>"
        "Z/X/C: Make swords (wood/stone/iron)",
        parent,
    )
    controls_label.setWordWrap(True)
    controls_label.setStyleSheet("color: #666; font-size: 10px;")
    layout.addRow("", controls_label)

    # Achievement guidance
    guidance = QtWidgets.QLabel(
        "Crafter has 22 achievements. Score = geometric mean of success rates.",
        parent,
    )
    guidance.setWordWrap(True)
    guidance.setStyleSheet("color: #888; font-size: 9px;")
    layout.addRow("", guidance)


__all__ = [
    "CRAFTER_GAME_IDS",
    "ControlCallbacks",
    "build_crafter_controls",
    "resolve_default_config",
]
