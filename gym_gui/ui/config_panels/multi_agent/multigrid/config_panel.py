"""Configuration widgets for MultiGrid multi-agent environments.

MultiGrid is a multi-agent extension of MiniGrid. Multiple agents act
simultaneously on a shared grid with walls, doors, keys, goals, etc.

Two types of environments:
- INI multigrid (2023+): Gymnasium-based, configurable agent count (default: 2)
- Legacy gym-multigrid: Fixed agent counts (Soccer=4, Collect=3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from PyQt6 import QtWidgets

from gym_gui.config.game_configs import MultiGridConfig
from gym_gui.core.enums import GameId


# All MultiGrid game IDs
MULTIGRID_GAME_IDS: tuple[GameId, ...] = (
    # MOSAIC multigrid (fixed agent counts)
    GameId.MOSAIC_MULTIGRID_SOCCER,      # 4 agents (2v2) - Deprecated
    GameId.MOSAIC_MULTIGRID_COLLECT,     # 3 agents - Deprecated
    GameId.MOSAIC_MULTIGRID_COLLECT2VS2,  # 4 agents (2v2) - Deprecated
    GameId.MOSAIC_MULTIGRID_SOCCER_2VS2_INDAGOBS,      # 4 agents (2v2) - IndAgObs
    GameId.MOSAIC_MULTIGRID_COLLECT_INDAGOBS,     # 3 agents - IndAgObs
    GameId.MOSAIC_MULTIGRID_COLLECT2VS2_INDAGOBS,  # 4 agents (2v2) - IndAgObs
    GameId.MOSAIC_MULTIGRID_SOCCER_2VS2_TEAMOBS,       # 4 agents (2v2) - TeamObs
    GameId.MOSAIC_MULTIGRID_COLLECT2VS2_TEAMOBS,   # 4 agents (2v2) - TeamObs
    # INI multigrid (configurable agent count, default 2)
    GameId.INI_MULTIGRID_BLOCKED_UNLOCK_PICKUP,
    GameId.INI_MULTIGRID_EMPTY_5X5,
    GameId.INI_MULTIGRID_EMPTY_RANDOM_5X5,
    GameId.INI_MULTIGRID_EMPTY_6X6,
    GameId.INI_MULTIGRID_EMPTY_RANDOM_6X6,
    GameId.INI_MULTIGRID_EMPTY_8X8,
    GameId.INI_MULTIGRID_EMPTY_16X16,
    GameId.INI_MULTIGRID_LOCKED_HALLWAY_2ROOMS,
    GameId.INI_MULTIGRID_LOCKED_HALLWAY_4ROOMS,
    GameId.INI_MULTIGRID_LOCKED_HALLWAY_6ROOMS,
    GameId.INI_MULTIGRID_PLAYGROUND,
    GameId.INI_MULTIGRID_RED_BLUE_DOORS_6X6,
    GameId.INI_MULTIGRID_RED_BLUE_DOORS_8X8,
)

# MOSAIC environments with  agent counts (cannot be changed)
MOSAIC_AGENT_COUNTS: dict[GameId, int] = {
    GameId.MOSAIC_MULTIGRID_SOCCER: 4,   # 2v2 teams
    GameId.MOSAIC_MULTIGRID_COLLECT: 3,  # 3 collectors
    GameId.MOSAIC_MULTIGRID_COLLECT2VS2: 4,  # 2v2 teams
    GameId.MOSAIC_MULTIGRID_SOCCER_2VS2_INDAGOBS: 4,   # 2v2 teams
    GameId.MOSAIC_MULTIGRID_COLLECT_INDAGOBS: 3,  # 3 collectors
    GameId.MOSAIC_MULTIGRID_COLLECT2VS2_INDAGOBS: 4,  # 2v2 teams
    GameId.MOSAIC_MULTIGRID_SOCCER_2VS2_TEAMOBS: 4,   # 2v2 teams
    GameId.MOSAIC_MULTIGRID_COLLECT2VS2_TEAMOBS: 4,  # 2v2 teams
}

# INI environments that support configurable agent count
INI_CONFIGURABLE_GAMES: tuple[GameId, ...] = tuple(
    gid for gid in MULTIGRID_GAME_IDS if gid not in MOSAIC_AGENT_COUNTS
)


@dataclass(slots=True)
class ControlCallbacks:
    """Bridge callbacks for propagating UI changes to session state."""

    on_change: Callable[[str, Any], None]


def build_multigrid_controls(
    *,
    parent: QtWidgets.QWidget,
    layout: QtWidgets.QFormLayout,
    game_id: GameId,
    overrides: Dict[str, Any],
    defaults: MultiGridConfig | None = None,
    callbacks: ControlCallbacks | None = None,
) -> None:
    """Populate MultiGrid-specific configuration widgets.

    Args:
        parent: Parent widget for controls.
        layout: Form layout to add controls to.
        game_id: The selected MultiGrid game ID.
        overrides: Dictionary to store user-modified values.
        defaults: Default configuration values.
        callbacks: Optional callbacks for value changes.
    """
    def emit(key: str, value: Any) -> None:
        overrides[key] = value
        if callbacks:
            callbacks.on_change(key, value)

    cfg = defaults if isinstance(defaults, MultiGridConfig) else MultiGridConfig()
    is_mosaic = game_id in MOSAIC_AGENT_COUNTS

    # -------- Number of Agents --------
    if is_mosaic:
        # MOSAIC environments: Show agent count (read-only)
        agent_count = MOSAIC_AGENT_COUNTS[game_id]
        agents_label = QtWidgets.QLabel(f"{agent_count}", parent)
        agents_label.setToolTip(
            f"This MOSAIC environment has {agent_count} agents.\n"
            "MOSAIC multigrid environments have predefined agent counts."
        )
        layout.addRow("Number of Agents", agents_label)
    else:
        # INI environments: Spinbox for configurable agent count
        agents_spin = QtWidgets.QSpinBox(parent)
        agents_spin.setRange(2, 16)  # Min 2 for multi-agent, max 16 reasonable limit

        # Get current value from overrides or use default of 2
        current_agents = overrides.get("num_agents", cfg.num_agents)
        if current_agents is None:
            current_agents = 2  # Default for INI environments
        agents_spin.setValue(current_agents)

        def on_agents_changed(value: int) -> None:
            emit("num_agents", value)

        agents_spin.valueChanged.connect(on_agents_changed)
        agents_spin.setToolTip(
            "Number of agents in the environment.\n"
            "INI multigrid environments support 2-16 agents.\n"
            "Default: 2 (cooperative pair)"
        )
        layout.addRow("Number of Agents", agents_spin)

        # Set initial value in overrides if not already set
        if "num_agents" not in overrides:
            overrides["num_agents"] = current_agents

    # -------- Seed (optional) --------
    seed_spin = QtWidgets.QSpinBox(parent)
    seed_spin.setRange(0, 2147483647)
    seed_spin.setSpecialValueText("Random")

    current_seed = overrides.get("seed", cfg.seed)
    if current_seed is None:
        seed_spin.setValue(0)  # Shows "Random"
    else:
        seed_spin.setValue(current_seed)

    def on_seed_changed(value: int) -> None:
        emit("seed", value if value > 0 else None)

    seed_spin.valueChanged.connect(on_seed_changed)
    seed_spin.setToolTip(
        "Random seed for reproducibility.\n"
        "Set to 0 for random seed, or specify a value for reproducible runs."
    )
    layout.addRow("Seed", seed_spin)

    # -------- Highlight Agent Views --------
    highlight_check = QtWidgets.QCheckBox("Highlight agent view cones", parent)
    current_highlight = overrides.get("highlight", cfg.highlight)
    highlight_check.setChecked(current_highlight)

    def on_highlight_changed(checked: bool) -> None:
        emit("highlight", checked)

    highlight_check.stateChanged.connect(lambda state: on_highlight_changed(state == 2))
    highlight_check.setToolTip(
        "When enabled, renders colored overlays showing each agent's field of view."
    )
    layout.addRow("", highlight_check)

    # -------- Info Label --------
    if is_mosaic:
        env_type = "MOSAIC MultiGrid"
        api_info = "Team-based competitive (Soccer/Collect)"
    else:
        env_type = "INI MultiGrid"
        api_info = "Cooperative exploration"

    info_label = QtWidgets.QLabel(
        f"<i><b>{env_type}</b> ({api_info})<br><br>"
        "<b>Controls:</b> Arrow keys (move), Space (toggle/pickup/drop)<br>"
        "For multi-human play, assign keyboards in Human Control tab.</i>",
        parent,
    )
    info_label.setWordWrap(True)
    layout.addRow("", info_label)
