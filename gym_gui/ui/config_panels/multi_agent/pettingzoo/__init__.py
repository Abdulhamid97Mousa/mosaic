"""PettingZoo environment UI configuration helpers.

Provides config panel builders and game ID mappings for PettingZoo
multi-agent environments.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Set

from PyQt6 import QtWidgets

from gym_gui.core.pettingzoo_enums import (
    PETTINGZOO_ENV_METADATA,
    PettingZooAPIType,
    PettingZooEnvId,
    PettingZooFamily,
    get_api_type,
    get_description,
    get_display_name,
    get_envs_by_family,
)
from .config_panel import (
    PettingZooConfig,
    build_pettingzoo_config_panel,
    build_classic_config_panel,
    build_mpe_config_panel,
    build_sisl_config_panel,
    build_butterfly_config_panel,
    build_atari_config_panel,
)
# All PettingZoo environment IDs
PETTINGZOO_GAME_IDS: Set[PettingZooEnvId] = set(PettingZooEnvId)


class ControlCallbacks:
    """Callbacks for PettingZoo configuration changes."""

    def __init__(
        self,
        on_change: Callable[[str, object], None] | None = None,
    ) -> None:
        self.on_change = on_change or (lambda k, v: None)


def get_pettingzoo_display_name(env_id: PettingZooEnvId) -> str:
    """Get display name for a PettingZoo environment."""
    return get_display_name(env_id)


def build_pettingzoo_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    env_id: PettingZooEnvId,
    overrides: Dict[str, object],
    callbacks: ControlCallbacks,
) -> None:
    """Build configuration controls for a PettingZoo environment.

    Args:
        parent: Parent widget for created controls
        layout: Form layout to add controls to
        env_id: PettingZoo environment ID
        overrides: Current configuration overrides
        callbacks: Callbacks for configuration changes
    """
    # Clear existing controls
    while layout.count():
        item = layout.takeAt(0)
        if item is not None:
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    # Get metadata for this environment
    metadata = PETTINGZOO_ENV_METADATA.get(env_id)
    if metadata is None:
        label = QtWidgets.QLabel(
            "No configuration available for this environment.",
            parent,
        )
        label.setWordWrap(True)
        layout.addRow("", label)
        return

    family, api_type, display_name, description = metadata

    # Environment info section
    info_label = QtWidgets.QLabel(
        f"<b>{display_name}</b><br/>"
        f"{description}<br/>"
        f"<i>Family: {family.value.title()} | API: {api_type.value.upper()}</i>",
        parent,
    )
    info_label.setWordWrap(True)
    info_label.setStyleSheet(
        "color: #555; font-size: 11px; padding: 4px; "
        "background-color: #f8f8f8; border-radius: 4px;"
    )
    layout.addRow(info_label)

    # API-specific configuration
    if api_type == PettingZooAPIType.AEC:
        # AEC (Agent Environment Cycle) - turn-based games
        _build_aec_controls(parent, layout, env_id, overrides, callbacks)
    else:
        # Parallel - simultaneous action games
        _build_parallel_controls(parent, layout, env_id, overrides, callbacks)


def _build_aec_controls(
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    env_id: PettingZooEnvId,
    overrides: Dict[str, object],
    callbacks: ControlCallbacks,
) -> None:
    """Build controls specific to AEC (turn-based) environments."""
    # Max cycles (episode length)
    max_cycles_spin = QtWidgets.QSpinBox(parent)
    max_cycles_spin.setRange(1, 100000)
    max_cycles_spin.setValue(int(overrides.get("max_cycles", 1000)))
    max_cycles_spin.setToolTip(
        "Maximum number of agent turns before episode terminates"
    )
    max_cycles_spin.valueChanged.connect(
        lambda v: callbacks.on_change("max_cycles", v)
    )
    layout.addRow("Max Cycles:", max_cycles_spin)

    # Render mode selection
    render_combo = QtWidgets.QComboBox(parent)
    render_combo.addItem("RGB Array", "rgb_array")
    render_combo.addItem("Human (Pygame)", "human")
    render_combo.addItem("ANSI (Text)", "ansi")

    current_render = overrides.get("render_mode", "rgb_array")
    index = render_combo.findData(current_render)
    if index >= 0:
        render_combo.setCurrentIndex(index)

    render_combo.currentIndexChanged.connect(
        lambda i: callbacks.on_change("render_mode", render_combo.itemData(i))
    )
    layout.addRow("Render Mode:", render_combo)


def _build_parallel_controls(
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    env_id: PettingZooEnvId,
    overrides: Dict[str, object],
    callbacks: ControlCallbacks,
) -> None:
    """Build controls specific to Parallel (simultaneous action) environments."""
    # Max cycles (episode length)
    max_cycles_spin = QtWidgets.QSpinBox(parent)
    max_cycles_spin.setRange(1, 100000)
    max_cycles_spin.setValue(int(overrides.get("max_cycles", 500)))
    max_cycles_spin.setToolTip(
        "Maximum number of steps before episode terminates"
    )
    max_cycles_spin.valueChanged.connect(
        lambda v: callbacks.on_change("max_cycles", v)
    )
    layout.addRow("Max Cycles:", max_cycles_spin)

    # Render mode selection
    render_combo = QtWidgets.QComboBox(parent)
    render_combo.addItem("RGB Array", "rgb_array")
    render_combo.addItem("Human (Pygame)", "human")

    current_render = overrides.get("render_mode", "rgb_array")
    index = render_combo.findData(current_render)
    if index >= 0:
        render_combo.setCurrentIndex(index)

    render_combo.currentIndexChanged.connect(
        lambda i: callbacks.on_change("render_mode", render_combo.itemData(i))
    )
    layout.addRow("Render Mode:", render_combo)

    # Number of agents (for environments that support it)
    if env_id in (
        PettingZooEnvId.KNIGHTS_ARCHERS_ZOMBIES,
        PettingZooEnvId.WATERWORLD,
        PettingZooEnvId.PURSUIT,
    ):
        num_agents_spin = QtWidgets.QSpinBox(parent)
        num_agents_spin.setRange(2, 20)
        num_agents_spin.setValue(int(overrides.get("num_agents", 4)))
        num_agents_spin.setToolTip("Number of cooperative agents")
        num_agents_spin.valueChanged.connect(
            lambda v: callbacks.on_change("num_agents", v)
        )
        layout.addRow("Num Agents:", num_agents_spin)


__all__ = [
    "PETTINGZOO_GAME_IDS",
    "ControlCallbacks",
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
