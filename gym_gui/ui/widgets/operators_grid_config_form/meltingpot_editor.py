"""MeltingPot environment configuration dialog.

MeltingPot is DeepMind's suite of multi-agent social scenarios for testing
generalization to novel social situations. Unlike grid environments,
MeltingPot substrates are complex simulations that don't have simple
editable grid states.

This editor provides:
- Substrate selection and info
- Player count configuration
- Scenario-specific settings

Supported substrates:
- collaborative_cooking: Cooperative cooking tasks
- clean_up: Environmental cleanup
- commons_harvest: Resource gathering dilemma
- territory: Area control
- prisoners_dilemma: Classic game theory
- stag_hunt: Coordination game
- allelopathic_harvest: Competitive harvesting
- king_of_the_hill: Competitive territory control
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

from PyQt6 import QtWidgets, QtCore, QtGui

from .base import (
    GridState,
    GridCell,
    GridObject,
    GridEditorWidget,
    GridObjectTray,
    GridConfigDialog,
    ObjectType,
    ObjectColor,
)

_LOGGER = logging.getLogger(__name__)


# MeltingPot substrate information
MELTINGPOT_SUBSTRATES: Dict[str, Dict[str, Any]] = {
    "collaborative_cooking": {
        "name": "Collaborative Cooking",
        "description": "Agents must cooperate to prepare and deliver dishes in a shared kitchen.",
        "min_players": 2,
        "max_players": 4,
        "default_players": 2,
        "tags": ["cooperative", "coordination"],
    },
    "clean_up": {
        "name": "Clean Up",
        "description": "Agents must clean pollution to maintain apple growth. Tests social dilemma.",
        "min_players": 2,
        "max_players": 7,
        "default_players": 4,
        "tags": ["social_dilemma", "public_goods"],
    },
    "commons_harvest": {
        "name": "Commons Harvest",
        "description": "Agents harvest apples from a shared orchard. Tests sustainable resource use.",
        "min_players": 2,
        "max_players": 16,
        "default_players": 4,
        "tags": ["social_dilemma", "commons"],
    },
    "territory": {
        "name": "Territory",
        "description": "Agents claim and defend territory. Tests competitive dynamics.",
        "min_players": 2,
        "max_players": 8,
        "default_players": 4,
        "tags": ["competitive", "territory"],
    },
    "prisoners_dilemma": {
        "name": "Prisoner's Dilemma",
        "description": "Classic iterated prisoner's dilemma in spatial setting.",
        "min_players": 2,
        "max_players": 8,
        "default_players": 2,
        "tags": ["game_theory", "cooperation"],
    },
    "stag_hunt": {
        "name": "Stag Hunt",
        "description": "Coordination game: hunt stag together or hunt rabbits alone.",
        "min_players": 2,
        "max_players": 8,
        "default_players": 4,
        "tags": ["coordination", "game_theory"],
    },
    "allelopathic_harvest": {
        "name": "Allelopathic Harvest",
        "description": "Agents can plant colored berries that inhibit other colors.",
        "min_players": 2,
        "max_players": 8,
        "default_players": 4,
        "tags": ["competitive", "resource"],
    },
    "king_of_the_hill": {
        "name": "King of the Hill",
        "description": "Agents compete to control a central area.",
        "min_players": 2,
        "max_players": 8,
        "default_players": 4,
        "tags": ["competitive", "territory"],
    },
}


class MeltingPotState(GridState):
    """Configuration state for MeltingPot environments.

    Unlike grid-based states, MeltingPot state primarily captures:
    - Selected substrate
    - Number of players
    - Substrate-specific settings
    """

    def __init__(self, substrate: str = "commons_harvest", num_players: int = 4):
        self._substrate = substrate
        self._num_players = num_players
        self._settings: Dict[str, Any] = {}

    def get_cell(self, row: int, col: int) -> GridCell:
        # MeltingPot doesn't use a simple grid
        return GridCell(row, col, [])

    def set_cell(self, cell: GridCell) -> None:
        pass  # Not applicable

    def get_agent_position(self) -> Optional[Tuple[int, int]]:
        return None  # Agents are managed by substrate

    def set_agent_position(self, row: int, col: int, direction: int = 0) -> None:
        pass  # Not applicable

    def place_object(self, row: int, col: int, obj: GridObject) -> bool:
        return False  # Not applicable

    def remove_object(
        self, row: int, col: int, obj_type: Optional[ObjectType] = None
    ) -> Optional[GridObject]:
        return None  # Not applicable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "substrate": self._substrate,
            "num_players": self._num_players,
            "settings": self._settings,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self._substrate = data.get("substrate", "commons_harvest")
        self._num_players = data.get("num_players", 4)
        self._settings = data.get("settings", {})

    def get_dimensions(self) -> Tuple[int, int]:
        return (0, 0)  # Not a fixed grid

    def clear(self) -> None:
        self._settings = {}

    def copy(self) -> "MeltingPotState":
        new_state = MeltingPotState(self._substrate, self._num_players)
        new_state._settings = self._settings.copy()
        return new_state

    def get_available_objects(self) -> List[GridObject]:
        return []  # Not applicable

    @property
    def substrate(self) -> str:
        return self._substrate

    @substrate.setter
    def substrate(self, value: str) -> None:
        self._substrate = value

    @property
    def num_players(self) -> int:
        return self._num_players

    @num_players.setter
    def num_players(self, value: int) -> None:
        self._num_players = value


class MeltingPotInfoWidget(GridEditorWidget):
    """Display widget showing MeltingPot substrate information.

    Since MeltingPot substrates don't have simple editable grids,
    this widget shows substrate description and preview.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._state: Optional[MeltingPotState] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Substrate info card
        self._info_card = QtWidgets.QFrame()
        self._info_card.setStyleSheet("""
            QFrame {
                background-color: #F8F9FA;
                border: 1px solid #DEE2E6;
                border-radius: 8px;
                padding: 16px;
            }
        """)
        card_layout = QtWidgets.QVBoxLayout(self._info_card)

        self._title_label = QtWidgets.QLabel("Select a substrate")
        self._title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #212529;")
        card_layout.addWidget(self._title_label)

        self._desc_label = QtWidgets.QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("font-size: 13px; color: #495057; margin-top: 8px;")
        card_layout.addWidget(self._desc_label)

        self._tags_label = QtWidgets.QLabel()
        self._tags_label.setStyleSheet("font-size: 11px; color: #6C757D; margin-top: 12px;")
        card_layout.addWidget(self._tags_label)

        card_layout.addStretch()

        # Player info
        self._player_info = QtWidgets.QLabel()
        self._player_info.setStyleSheet("font-size: 12px; color: #495057;")
        card_layout.addWidget(self._player_info)

        layout.addWidget(self._info_card)

        # Note about MeltingPot
        note = QtWidgets.QLabel(
            "Note: MeltingPot substrates have complex procedural layouts. "
            "Configuration options are limited to player count and substrate selection."
        )
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 11px; color: #868E96; font-style: italic;")
        layout.addWidget(note)

    def set_state(self, state: GridState) -> None:
        if not isinstance(state, MeltingPotState):
            raise TypeError("Expected MeltingPotState")
        self._state = state
        self._update_display()

    def get_state(self) -> GridState:
        if self._state is None:
            self._state = MeltingPotState()
        return self._state

    def _update_display(self) -> None:
        if self._state is None:
            return

        substrate = self._state.substrate
        info = MELTINGPOT_SUBSTRATES.get(substrate, {})

        self._title_label.setText(info.get("name", substrate))
        self._desc_label.setText(info.get("description", "No description available."))

        tags = info.get("tags", [])
        if tags:
            tag_text = "Tags: " + ", ".join(f"#{t}" for t in tags)
            self._tags_label.setText(tag_text)
        else:
            self._tags_label.setText("")

        min_p = info.get("min_players", 2)
        max_p = info.get("max_players", 8)
        self._player_info.setText(f"Players: {self._state.num_players} (range: {min_p}-{max_p})")

    # Required abstract methods (not used for MeltingPot)
    def _get_cell_size(self) -> int:
        return 0

    def _pos_to_cell(self, pos: QtCore.QPoint) -> Tuple[int, int]:
        return (0, 0)

    def _cell_to_rect(self, row: int, col: int) -> QtCore.QRect:
        return QtCore.QRect()

    def _is_valid_cell(self, row: int, col: int) -> bool:
        return False

    def _draw_cell(
        self,
        painter: QtGui.QPainter,
        row: int,
        col: int,
        rect: QtCore.QRect
    ) -> None:
        pass  # No grid to draw


class MeltingPotSubstrateSelector(GridObjectTray):
    """Substrate selector for MeltingPot environments."""

    substrate_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._selected_substrate: str = "commons_harvest"
        self._buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        label = QtWidgets.QLabel("Substrates")
        label.setStyleSheet("font-weight: bold; font-size: 13px; color: #495057;")
        layout.addWidget(label)

        # Substrate buttons
        for substrate_id, info in MELTINGPOT_SUBSTRATES.items():
            btn = QtWidgets.QPushButton(info["name"])
            btn.setCheckable(True)
            btn.setFixedHeight(32)

            # Color by type
            tags = info.get("tags", [])
            if "cooperative" in tags or "coordination" in tags:
                color = "#51CF66"  # Green for cooperative
            elif "competitive" in tags:
                color = "#FF6B6B"  # Red for competitive
            else:
                color = "#74C0FC"  # Blue for social dilemma

            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color}40;
                    border: 1px solid {color};
                    border-radius: 4px;
                    font-size: 11px;
                    text-align: left;
                    padding-left: 8px;
                }}
                QPushButton:checked {{
                    background-color: {color};
                    color: white;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {color}80;
                }}
            """)

            btn.clicked.connect(
                lambda checked, sid=substrate_id: self._on_substrate_clicked(sid)
            )
            self._buttons[substrate_id] = btn
            layout.addWidget(btn)

        layout.addStretch()

        # Select default
        self._buttons["commons_harvest"].setChecked(True)

    def _on_substrate_clicked(self, substrate_id: str) -> None:
        for sid, btn in self._buttons.items():
            btn.setChecked(sid == substrate_id)

        self._selected_substrate = substrate_id
        self.substrate_changed.emit(substrate_id)

    def get_selected_substrate(self) -> str:
        return self._selected_substrate

    def set_selected_substrate(self, substrate_id: str) -> None:
        if substrate_id in self._buttons:
            self._on_substrate_clicked(substrate_id)

    # Required abstract methods
    def set_available_objects(self, objects: List[GridObject]) -> None:
        pass

    def get_selected_object(self) -> Optional[GridObject]:
        return None


class MeltingPotConfigDialog(GridConfigDialog):
    """Configuration dialog for MeltingPot environments."""

    def __init__(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        env_id: str = "commons_harvest"
    ):
        self._env_id = env_id
        self._player_spinbox: Optional[QtWidgets.QSpinBox] = None
        super().__init__(initial_state, parent)

    def _get_title(self) -> str:
        return f"Configure MeltingPot - {self._env_id}"

    def _create_grid_editor(self) -> GridEditorWidget:
        return MeltingPotInfoWidget()

    def _create_object_tray(self) -> GridObjectTray:
        tray = MeltingPotSubstrateSelector()
        tray.substrate_changed.connect(self._on_substrate_changed)
        return tray

    def _create_extra_controls(self) -> Optional[QtWidgets.QWidget]:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 10, 0, 0)

        # Player count control
        player_group = QtWidgets.QGroupBox("Players")
        player_layout = QtWidgets.QHBoxLayout(player_group)

        player_layout.addWidget(QtWidgets.QLabel("Count:"))

        self._player_spinbox = QtWidgets.QSpinBox()
        self._player_spinbox.setRange(2, 16)
        self._player_spinbox.setValue(4)
        self._player_spinbox.valueChanged.connect(self._on_player_count_changed)
        player_layout.addWidget(self._player_spinbox)

        player_layout.addStretch()
        layout.addWidget(player_group)

        return widget

    def _on_substrate_changed(self, substrate_id: str) -> None:
        """Handle substrate selection change."""
        state = self._grid_editor.get_state()
        if isinstance(state, MeltingPotState):
            state.substrate = substrate_id

            # Update player range
            info = MELTINGPOT_SUBSTRATES.get(substrate_id, {})
            min_p = info.get("min_players", 2)
            max_p = info.get("max_players", 8)
            default_p = info.get("default_players", 4)

            if self._player_spinbox:
                self._player_spinbox.setRange(min_p, max_p)
                self._player_spinbox.setValue(default_p)
                state.num_players = default_p

            self._grid_editor.set_state(state)

    def _on_player_count_changed(self, count: int) -> None:
        """Handle player count change."""
        state = self._grid_editor.get_state()
        if isinstance(state, MeltingPotState):
            state.num_players = count
            self._grid_editor.set_state(state)

    def _get_presets(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            ("Commons 4p", {"substrate": "commons_harvest", "num_players": 4, "settings": {}}),
            ("Cooking 2p", {"substrate": "collaborative_cooking", "num_players": 2, "settings": {}}),
            ("Territory 4p", {"substrate": "territory", "num_players": 4, "settings": {}}),
            ("Stag Hunt 4p", {"substrate": "stag_hunt", "num_players": 4, "settings": {}}),
        ]

    def _validate_state(self, state_dict: Dict[str, Any]) -> List[str]:
        errors = []

        substrate = state_dict.get("substrate", "")
        if substrate not in MELTINGPOT_SUBSTRATES:
            errors.append(f"Unknown substrate: {substrate}")

        num_players = state_dict.get("num_players", 0)
        info = MELTINGPOT_SUBSTRATES.get(substrate, {})
        min_p = info.get("min_players", 2)
        max_p = info.get("max_players", 8)

        if num_players < min_p or num_players > max_p:
            errors.append(f"Player count must be between {min_p} and {max_p} for {substrate}")

        return errors

    def _create_state_from_dict(self, data: Dict[str, Any]) -> GridState:
        state = MeltingPotState()
        state.from_dict(data)

        # Update UI to match state
        if self._player_spinbox:
            self._player_spinbox.setValue(state.num_players)

        if isinstance(self._object_tray, MeltingPotSubstrateSelector):
            self._object_tray.set_selected_substrate(state.substrate)

        return state
