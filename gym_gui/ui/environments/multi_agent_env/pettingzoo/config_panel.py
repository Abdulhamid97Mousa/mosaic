"""Configuration panel builders for PettingZoo environments.

Provides specialized configuration UIs for different PettingZoo families
(Classic, MPE, SISL, Butterfly, Atari).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from PyQt6 import QtWidgets

from gym_gui.core.pettingzoo_enums import (
    PETTINGZOO_ENV_METADATA,
    PettingZooAPIType,
    PettingZooEnvId,
    PettingZooFamily,
    PettingZooGameType,
    get_api_type,
    get_description,
    get_display_name,
    get_game_type,
)


@dataclass
class PettingZooConfig:
    """Configuration for a PettingZoo environment."""

    env_id: str
    max_cycles: int = 1000
    render_mode: str = "rgb_array"
    num_agents: Optional[int] = None
    # Classic-specific
    screen_scaling: int = 4
    # MPE-specific
    continuous_actions: bool = False
    # Atari-specific
    obs_type: str = "rgb"  # "rgb", "grayscale", "ram"
    auto_rom_install_path: Optional[str] = None


def build_classic_config_panel(
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    env_id: PettingZooEnvId,
    overrides: Dict[str, object],
    on_change: Callable[[str, object], None],
) -> None:
    """Build config panel for Classic (board game) environments."""
    # Screen scaling for rendering
    scale_spin = QtWidgets.QSpinBox(parent)
    scale_spin.setRange(1, 10)
    scale_spin.setValue(int(overrides.get("screen_scaling", 4)))
    scale_spin.setToolTip("Rendering scale factor for board visualization")
    scale_spin.valueChanged.connect(lambda v: on_change("screen_scaling", v))
    layout.addRow("Screen Scale:", scale_spin)

    # Max cycles for AEC games
    cycles_spin = QtWidgets.QSpinBox(parent)
    cycles_spin.setRange(1, 100000)
    cycles_spin.setValue(int(overrides.get("max_cycles", 1000)))
    cycles_spin.setToolTip("Maximum turns before game ends (prevents infinite games)")
    cycles_spin.valueChanged.connect(lambda v: on_change("max_cycles", v))
    layout.addRow("Max Turns:", cycles_spin)


def build_mpe_config_panel(
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    env_id: PettingZooEnvId,
    overrides: Dict[str, object],
    on_change: Callable[[str, object], None],
) -> None:
    """Build config panel for MPE (Multi-Particle) environments."""
    # Number of agents (most MPE envs support this)
    agents_spin = QtWidgets.QSpinBox(parent)
    agents_spin.setRange(2, 20)
    agents_spin.setValue(int(overrides.get("num_agents", 3)))
    agents_spin.setToolTip("Number of agents in the environment")
    agents_spin.valueChanged.connect(lambda v: on_change("num_agents", v))
    layout.addRow("Num Agents:", agents_spin)

    # Continuous vs discrete actions
    continuous_check = QtWidgets.QCheckBox("Continuous Actions", parent)
    continuous_check.setChecked(bool(overrides.get("continuous_actions", False)))
    continuous_check.setToolTip(
        "Use continuous action space instead of discrete. "
        "Continuous is harder but allows finer control."
    )
    continuous_check.stateChanged.connect(
        lambda s: on_change("continuous_actions", s == 2)
    )
    layout.addRow("", continuous_check)

    # Max cycles
    cycles_spin = QtWidgets.QSpinBox(parent)
    cycles_spin.setRange(1, 10000)
    cycles_spin.setValue(int(overrides.get("max_cycles", 100)))
    cycles_spin.setToolTip("Maximum steps per episode")
    cycles_spin.valueChanged.connect(lambda v: on_change("max_cycles", v))
    layout.addRow("Max Cycles:", cycles_spin)


def build_sisl_config_panel(
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    env_id: PettingZooEnvId,
    overrides: Dict[str, object],
    on_change: Callable[[str, object], None],
) -> None:
    """Build config panel for SISL environments (Multiwalker, Waterworld, Pursuit)."""
    # Number of agents
    agents_spin = QtWidgets.QSpinBox(parent)

    if env_id == PettingZooEnvId.MULTIWALKER:
        agents_spin.setRange(2, 10)
        agents_spin.setValue(int(overrides.get("n_walkers", 3)))
        agents_spin.setToolTip("Number of bipedal walkers carrying the package")
        agents_spin.valueChanged.connect(lambda v: on_change("n_walkers", v))
        layout.addRow("Num Walkers:", agents_spin)

        # Package mass
        mass_spin = QtWidgets.QDoubleSpinBox(parent)
        mass_spin.setRange(0.1, 100.0)
        mass_spin.setValue(float(overrides.get("package_mass", 1.0)))
        mass_spin.setSingleStep(0.1)
        mass_spin.setToolTip("Mass of the package being carried")
        mass_spin.valueChanged.connect(lambda v: on_change("package_mass", v))
        layout.addRow("Package Mass:", mass_spin)

    elif env_id == PettingZooEnvId.WATERWORLD:
        agents_spin.setRange(2, 20)
        agents_spin.setValue(int(overrides.get("n_pursuers", 5)))
        agents_spin.setToolTip("Number of pursuer agents")
        agents_spin.valueChanged.connect(lambda v: on_change("n_pursuers", v))
        layout.addRow("Num Pursuers:", agents_spin)

        # Number of evaders
        evaders_spin = QtWidgets.QSpinBox(parent)
        evaders_spin.setRange(1, 20)
        evaders_spin.setValue(int(overrides.get("n_evaders", 5)))
        evaders_spin.setToolTip("Number of evader targets")
        evaders_spin.valueChanged.connect(lambda v: on_change("n_evaders", v))
        layout.addRow("Num Evaders:", evaders_spin)

    elif env_id == PettingZooEnvId.PURSUIT:
        agents_spin.setRange(2, 16)
        agents_spin.setValue(int(overrides.get("n_pursuers", 8)))
        agents_spin.setToolTip("Number of pursuer agents")
        agents_spin.valueChanged.connect(lambda v: on_change("n_pursuers", v))
        layout.addRow("Num Pursuers:", agents_spin)

        # Number of evaders
        evaders_spin = QtWidgets.QSpinBox(parent)
        evaders_spin.setRange(1, 30)
        evaders_spin.setValue(int(overrides.get("n_evaders", 30)))
        evaders_spin.setToolTip("Number of evader targets")
        evaders_spin.valueChanged.connect(lambda v: on_change("n_evaders", v))
        layout.addRow("Num Evaders:", evaders_spin)

    # Max cycles
    cycles_spin = QtWidgets.QSpinBox(parent)
    cycles_spin.setRange(1, 10000)
    cycles_spin.setValue(int(overrides.get("max_cycles", 500)))
    cycles_spin.setToolTip("Maximum steps per episode")
    cycles_spin.valueChanged.connect(lambda v: on_change("max_cycles", v))
    layout.addRow("Max Cycles:", cycles_spin)


def build_butterfly_config_panel(
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    env_id: PettingZooEnvId,
    overrides: Dict[str, object],
    on_change: Callable[[str, object], None],
) -> None:
    """Build config panel for Butterfly environments."""
    if env_id == PettingZooEnvId.KNIGHTS_ARCHERS_ZOMBIES:
        # Number of knights
        knights_spin = QtWidgets.QSpinBox(parent)
        knights_spin.setRange(1, 10)
        knights_spin.setValue(int(overrides.get("num_knights", 2)))
        knights_spin.setToolTip("Number of knight agents")
        knights_spin.valueChanged.connect(lambda v: on_change("num_knights", v))
        layout.addRow("Num Knights:", knights_spin)

        # Number of archers
        archers_spin = QtWidgets.QSpinBox(parent)
        archers_spin.setRange(1, 10)
        archers_spin.setValue(int(overrides.get("num_archers", 2)))
        archers_spin.setToolTip("Number of archer agents")
        archers_spin.valueChanged.connect(lambda v: on_change("num_archers", v))
        layout.addRow("Num Archers:", archers_spin)

    elif env_id == PettingZooEnvId.PISTONBALL:
        # Number of pistons
        pistons_spin = QtWidgets.QSpinBox(parent)
        pistons_spin.setRange(2, 20)
        pistons_spin.setValue(int(overrides.get("n_pistons", 15)))
        pistons_spin.setToolTip("Number of piston agents")
        pistons_spin.valueChanged.connect(lambda v: on_change("n_pistons", v))
        layout.addRow("Num Pistons:", pistons_spin)

        # Continuous actions
        continuous_check = QtWidgets.QCheckBox("Continuous Actions", parent)
        continuous_check.setChecked(bool(overrides.get("continuous", False)))
        continuous_check.stateChanged.connect(
            lambda s: on_change("continuous", s == 2)
        )
        layout.addRow("", continuous_check)

    elif env_id == PettingZooEnvId.COOPERATIVE_PONG:
        # Paddle height
        paddle_spin = QtWidgets.QSpinBox(parent)
        paddle_spin.setRange(5, 50)
        paddle_spin.setValue(int(overrides.get("paddle_height", 15)))
        paddle_spin.setToolTip("Height of each paddle in pixels")
        paddle_spin.valueChanged.connect(lambda v: on_change("paddle_height", v))
        layout.addRow("Paddle Height:", paddle_spin)

    # Max cycles
    cycles_spin = QtWidgets.QSpinBox(parent)
    cycles_spin.setRange(1, 10000)
    cycles_spin.setValue(int(overrides.get("max_cycles", 900)))
    cycles_spin.setToolTip("Maximum steps per episode")
    cycles_spin.valueChanged.connect(lambda v: on_change("max_cycles", v))
    layout.addRow("Max Cycles:", cycles_spin)


def build_atari_config_panel(
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    env_id: PettingZooEnvId,
    overrides: Dict[str, object],
    on_change: Callable[[str, object], None],
) -> None:
    """Build config panel for Atari (2-player) environments."""
    # Observation type
    obs_combo = QtWidgets.QComboBox(parent)
    obs_combo.addItem("RGB (Color)", "rgb")
    obs_combo.addItem("Grayscale", "grayscale")
    obs_combo.addItem("RAM", "ram")

    current_obs = overrides.get("obs_type", "rgb")
    index = obs_combo.findData(current_obs)
    if index >= 0:
        obs_combo.setCurrentIndex(index)

    obs_combo.currentIndexChanged.connect(
        lambda i: on_change("obs_type", obs_combo.itemData(i))
    )
    obs_combo.setToolTip("Observation format for the agent")
    layout.addRow("Observation:", obs_combo)

    # Full action set
    full_actions_check = QtWidgets.QCheckBox("Full Action Set", parent)
    full_actions_check.setChecked(bool(overrides.get("full_action_space", False)))
    full_actions_check.setToolTip(
        "Use full 18-action set instead of minimal game-specific actions"
    )
    full_actions_check.stateChanged.connect(
        lambda s: on_change("full_action_space", s == 2)
    )
    layout.addRow("", full_actions_check)

    # Max cycles
    cycles_spin = QtWidgets.QSpinBox(parent)
    cycles_spin.setRange(1, 100000)
    cycles_spin.setValue(int(overrides.get("max_cycles", 100000)))
    cycles_spin.setToolTip("Maximum frames before episode ends")
    cycles_spin.valueChanged.connect(lambda v: on_change("max_cycles", v))
    layout.addRow("Max Cycles:", cycles_spin)


def build_pettingzoo_config_panel(
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    env_id: PettingZooEnvId,
    overrides: Dict[str, object],
    on_change: Callable[[str, object], None],
) -> None:
    """Build the appropriate config panel based on environment family.

    This is the main entry point for building PettingZoo config panels.
    It dispatches to family-specific builders based on the environment ID.

    Args:
        parent: Parent widget
        layout: Form layout to add controls to
        env_id: PettingZoo environment ID
        overrides: Current configuration overrides
        on_change: Callback when configuration changes
    """
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

    # Environment info header
    game_type = get_game_type(env_id)
    type_label = {
        PettingZooGameType.COOPERATIVE: "Cooperative",
        PettingZooGameType.COMPETITIVE: "Competitive",
        PettingZooGameType.MIXED: "Mixed",
    }.get(game_type, "Unknown")

    info_label = QtWidgets.QLabel(
        f"<b>{display_name}</b><br/>"
        f"{description}<br/>"
        f"<i>Family: {family.value.title()} | "
        f"Type: {type_label} | "
        f"API: {api_type.value.upper()}</i>",
        parent,
    )
    info_label.setWordWrap(True)
    info_label.setStyleSheet(
        "color: #555; font-size: 11px; padding: 8px; "
        "background-color: #f0f4f8; border-radius: 4px; "
        "margin-bottom: 8px;"
    )
    layout.addRow(info_label)

    # Dispatch to family-specific panel builder
    if family == PettingZooFamily.CLASSIC:
        build_classic_config_panel(parent, layout, env_id, overrides, on_change)
    elif family == PettingZooFamily.MPE:
        build_mpe_config_panel(parent, layout, env_id, overrides, on_change)
    elif family == PettingZooFamily.SISL:
        build_sisl_config_panel(parent, layout, env_id, overrides, on_change)
    elif family == PettingZooFamily.BUTTERFLY:
        build_butterfly_config_panel(parent, layout, env_id, overrides, on_change)
    elif family == PettingZooFamily.ATARI:
        build_atari_config_panel(parent, layout, env_id, overrides, on_change)
    else:
        # Generic fallback
        cycles_spin = QtWidgets.QSpinBox(parent)
        cycles_spin.setRange(1, 100000)
        cycles_spin.setValue(int(overrides.get("max_cycles", 1000)))
        cycles_spin.setToolTip("Maximum steps per episode")
        cycles_spin.valueChanged.connect(lambda v: on_change("max_cycles", v))
        layout.addRow("Max Cycles:", cycles_spin)


__all__ = [
    "PettingZooConfig",
    "build_pettingzoo_config_panel",
    "build_classic_config_panel",
    "build_mpe_config_panel",
    "build_sisl_config_panel",
    "build_butterfly_config_panel",
    "build_atari_config_panel",
]
