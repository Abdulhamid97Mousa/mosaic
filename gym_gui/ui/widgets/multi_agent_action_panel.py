"""Multi-agent action panel for simultaneous environments.

This widget provides action selection for multiple human-controlled agents
in parallel/simultaneous environments (MultiGrid, MeltingPot, Overcooked).

Each human agent gets a row of action buttons, and all selections must be
made before the environment can step.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import pyqtSignal
from qtpy import QtCore, QtWidgets

_LOGGER = logging.getLogger(__name__)

# Named color palette for agent customization.
# The 6 names match MosaicMultiGrid's core/constants.py COLORS dict.
# Hex values are Material Design approximations for readable UI buttons.
COLOR_PALETTE: Dict[str, Tuple[str, str]] = {
    "red":    ("#e53935", "#ffcdd2"),
    "green":  ("#43a047", "#c8e6c9"),
    "blue":   ("#1e88e5", "#bbdefb"),
    "purple": ("#8e24aa", "#e1bee7"),
    "yellow": ("#fdd835", "#fff9c4"),
    "grey":   ("#757575", "#e0e0e0"),
}

# Default agent -> color_name mapping (preserves current behaviour).
DEFAULT_AGENT_COLOR_NAMES: Dict[str, str] = {
    # MosaicMultiGrid / PettingZoo agent IDs
    "agent_0": "green",
    "agent_1": "blue",
    "agent_2": "red",
    "agent_3": "yellow",
    "agent_4": "purple",
    "agent_5": "grey",
    # Chess / classic game player IDs
    "player_0": "grey",   # White
    "player_1": "blue",   # Black
}

# Legacy dict used internally — kept for fallback.
AGENT_COLORS = {
    f"agent_{i}": COLOR_PALETTE[name]
    for i, name in enumerate(DEFAULT_AGENT_COLOR_NAMES.values())
}


class AgentActionRow(QtWidgets.QWidget):
    """Row of action buttons for a single agent.

    Displays the agent name/color and action buttons.
    Tracks which action is selected (highlighted).
    """

    action_selected = pyqtSignal(str, int)  # agent_id, action_index

    def __init__(
        self,
        agent_id: str,
        agent_label: str,
        action_labels: List[str],
        color_override: Optional[Tuple[str, str]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize agent action row.

        Args:
            agent_id: The agent identifier (e.g., "agent_0").
            agent_label: Human-readable label (e.g., "Agent 0 (Red)").
            action_labels: List of action names (e.g., ["Still", "Left", "Right", ...]).
            color_override: Optional (primary_hex, bg_hex) to use instead of default.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._agent_id = agent_id
        self._agent_label = agent_label
        self._action_labels = action_labels
        self._selected_action: Optional[int] = None
        self._action_buttons: List[QtWidgets.QPushButton] = []

        # Get agent color scheme — use override if provided
        if color_override is not None:
            primary_color, bg_color = color_override
        else:
            primary_color, bg_color = AGENT_COLORS.get(agent_id, ("#666666", "#e0e0e0"))
        self._primary_color = primary_color
        self._bg_color = bg_color

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the action row UI."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Agent label with color indicator
        label_container = QtWidgets.QWidget(self)
        label_layout = QtWidgets.QHBoxLayout(label_container)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(4)

        # Color badge
        color_badge = QtWidgets.QLabel(self)
        color_badge.setFixedSize(12, 12)
        color_badge.setStyleSheet(
            f"background-color: {self._primary_color}; "
            f"border-radius: 6px; border: 1px solid {self._primary_color};"
        )
        label_layout.addWidget(color_badge)

        # Agent name
        name_label = QtWidgets.QLabel(self._agent_label, self)
        name_label.setStyleSheet(f"font-weight: bold; color: {self._primary_color};")
        name_label.setMinimumWidth(80)
        label_layout.addWidget(name_label)

        layout.addWidget(label_container)

        # Separator
        sep = QtWidgets.QFrame(self)
        sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Action buttons container
        buttons_container = QtWidgets.QWidget(self)
        buttons_layout = QtWidgets.QHBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(2)

        # Create action buttons (limit to 10 for UI)
        max_buttons = 10
        for i, label in enumerate(self._action_labels[:max_buttons]):
            btn = QtWidgets.QPushButton(label, buttons_container)
            btn.setFixedHeight(24)
            btn.setMinimumWidth(50)
            btn.setToolTip(f"Action {i}: {label}")
            btn.setStyleSheet(self._get_button_style(selected=False))
            btn.clicked.connect(lambda checked, idx=i: self._on_action_clicked(idx))
            buttons_layout.addWidget(btn)
            self._action_buttons.append(btn)

        # Overflow indicator
        if len(self._action_labels) > max_buttons:
            overflow = QtWidgets.QLabel(f"+{len(self._action_labels) - max_buttons}", self)
            overflow.setStyleSheet("font-size: 10px; color: #666; padding: 0 4px;")
            buttons_layout.addWidget(overflow)

        layout.addWidget(buttons_container)
        layout.addStretch()

        # Selection indicator
        self._selection_label = QtWidgets.QLabel("", self)
        self._selection_label.setStyleSheet(
            f"font-size: 10px; color: {self._primary_color}; font-weight: bold;"
        )
        self._selection_label.setMinimumWidth(80)
        layout.addWidget(self._selection_label)

    def _get_button_style(self, selected: bool) -> str:
        """Get button stylesheet based on selection state."""
        if selected:
            return (
                f"QPushButton {{ font-size: 10px; padding: 2px 6px; "
                f"background-color: {self._primary_color}; color: white; "
                f"border: 2px solid {self._primary_color}; border-radius: 3px; font-weight: bold; }}"
            )
        else:
            return (
                f"QPushButton {{ font-size: 10px; padding: 2px 6px; "
                f"background-color: {self._bg_color}; color: #333; "
                f"border: 1px solid {self._primary_color}; border-radius: 3px; }}"
                f"QPushButton:hover {{ background-color: {self._primary_color}; color: white; }}"
            )

    def _on_action_clicked(self, action_index: int) -> None:
        """Handle action button click."""
        # Update selection
        self._selected_action = action_index

        # Update button styles
        for i, btn in enumerate(self._action_buttons):
            btn.setStyleSheet(self._get_button_style(selected=(i == action_index)))

        # Update selection label
        if action_index < len(self._action_labels):
            self._selection_label.setText(f"Selected: {self._action_labels[action_index]}")

        # Emit signal
        self.action_selected.emit(self._agent_id, action_index)
        _LOGGER.debug(f"Agent {self._agent_id} selected action {action_index}")

    @property
    def agent_id(self) -> str:
        """Get agent ID."""
        return self._agent_id

    @property
    def selected_action(self) -> Optional[int]:
        """Get selected action index, or None if not selected."""
        return self._selected_action

    def set_action(self, action_index: int) -> None:
        """Programmatically select an action (e.g., from arrow key input).

        Updates the button highlight and selection label exactly like a click,
        then emits :pyqtSignal:`action_selected`.

        Args:
            action_index: The action index to select.
        """
        if 0 <= action_index < len(self._action_buttons):
            self._on_action_clicked(action_index)

    def clear_selection(self) -> None:
        """Clear the current selection."""
        self._selected_action = None
        for btn in self._action_buttons:
            btn.setStyleSheet(self._get_button_style(selected=False))
        self._selection_label.setText("")


class MultiAgentActionPanel(QtWidgets.QWidget):
    """Panel for collecting actions from multiple human agents.

    Displays action rows for each human-controlled agent and a
    "Submit All" button that enables when all agents have selected.

    Signals:
        all_actions_submitted: Emitted when user clicks Submit with all actions.
            Passes Dict[str, int] mapping agent_id to action.
    """

    all_actions_submitted = pyqtSignal(dict)  # {agent_id: action_index}

    def __init__(
        self,
        human_agents: List[str],
        action_labels: List[str],
        agent_labels: Optional[Dict[str, str]] = None,
        agent_colors: Optional[Dict[str, Tuple[str, str]]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize multi-agent action panel.

        Args:
            human_agents: List of human-controlled agent IDs.
            action_labels: List of action names (same for all agents).
            agent_labels: Optional dict mapping agent_id to display label.
            agent_colors: Optional per-agent color overrides
                {agent_id: (primary_hex, bg_hex)}.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._human_agents = human_agents
        self._action_labels = action_labels
        self._agent_labels = agent_labels or {}
        self._agent_colors: Dict[str, Tuple[str, str]] = agent_colors or {}
        self._agent_rows: Dict[str, AgentActionRow] = {}
        self._pending_actions: Dict[str, int] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the panel UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        header = QtWidgets.QLabel(
            f"Waiting for Human Actions ({len(self._human_agents)} agent(s))",
            self,
        )
        header.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #333; padding: 4px 0;"
        )
        layout.addWidget(header)
        self._header_label = header

        # Separator
        sep = QtWidgets.QFrame(self)
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Agent rows container with scroll area (in case of many agents)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setMaximumHeight(200)

        rows_container = QtWidgets.QWidget(scroll)
        rows_layout = QtWidgets.QVBoxLayout(rows_container)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(4)

        # Create row for each human agent
        for agent_id in self._human_agents:
            # Get label with color hint
            default_label = f"Agent {agent_id.split('_')[-1]}"
            label = self._agent_labels.get(agent_id, default_label)
            # Add color hint
            color_name = self._get_color_name(agent_id)
            if color_name:
                label = f"{label} ({color_name})"

            row = AgentActionRow(
                agent_id=agent_id,
                agent_label=label,
                action_labels=self._action_labels,
                color_override=self._agent_colors.get(agent_id),
                parent=rows_container,
            )
            row.action_selected.connect(self._on_agent_action_selected)
            rows_layout.addWidget(row)
            self._agent_rows[agent_id] = row

        scroll.setWidget(rows_container)
        layout.addWidget(scroll)

        # Status and submit button row
        button_row = QtWidgets.QHBoxLayout()

        # Status label
        self._status_label = QtWidgets.QLabel("0 of {} selected".format(len(self._human_agents)), self)
        self._status_label.setStyleSheet("font-size: 11px; color: #666;")
        button_row.addWidget(self._status_label)

        button_row.addStretch()

        # Clear button
        self._clear_btn = QtWidgets.QPushButton("Clear All", self)
        self._clear_btn.setFixedHeight(28)
        self._clear_btn.setStyleSheet(
            "QPushButton { padding: 4px 12px; background-color: #f5f5f5; "
            "border: 1px solid #ccc; border-radius: 4px; }"
            "QPushButton:hover { background-color: #e0e0e0; }"
        )
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        button_row.addWidget(self._clear_btn)

        # Submit button
        self._submit_btn = QtWidgets.QPushButton("Submit All Actions", self)
        self._submit_btn.setFixedHeight(28)
        self._submit_btn.setEnabled(False)
        self._submit_btn.setStyleSheet(
            "QPushButton { padding: 4px 16px; background-color: #4caf50; color: white; "
            "border: none; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #43a047; }"
            "QPushButton:disabled { background-color: #c8e6c9; color: #666; }"
        )
        self._submit_btn.clicked.connect(self._on_submit_clicked)
        button_row.addWidget(self._submit_btn)

        layout.addLayout(button_row)

    def _get_color_name(self, agent_id: str) -> Optional[str]:
        """Get display color name for an agent."""
        # Check custom override first
        if agent_id in self._agent_colors:
            primary, _ = self._agent_colors[agent_id]
            for name, (p, _) in COLOR_PALETTE.items():
                if p == primary:
                    return name.replace("_", " ").title()
            return None
        # Fall back to default mapping
        default_name = DEFAULT_AGENT_COLOR_NAMES.get(agent_id)
        if default_name:
            return default_name.replace("_", " ").title()
        return None

    def _on_agent_action_selected(self, agent_id: str, action: int) -> None:
        """Handle action selection from an agent row."""
        self._pending_actions[agent_id] = action
        self._update_status()

    def _update_status(self) -> None:
        """Update status label and submit button state."""
        selected = len(self._pending_actions)
        total = len(self._human_agents)
        pending = total - selected

        if pending == 0:
            self._status_label.setText("All agents ready!")
            self._status_label.setStyleSheet("font-size: 11px; color: #4caf50; font-weight: bold;")
            self._header_label.setText("All Actions Selected - Ready to Submit")
        else:
            self._status_label.setText(f"{selected} of {total} selected ({pending} pending)")
            self._status_label.setStyleSheet("font-size: 11px; color: #666;")
            self._header_label.setText(f"Waiting for Human Actions ({pending} pending)")

        self._submit_btn.setEnabled(pending == 0)

    def _on_clear_clicked(self) -> None:
        """Clear all selections."""
        self._pending_actions.clear()
        for row in self._agent_rows.values():
            row.clear_selection()
        self._update_status()

    def _on_submit_clicked(self) -> None:
        """Submit all collected actions."""
        if len(self._pending_actions) != len(self._human_agents):
            _LOGGER.warning("Cannot submit: not all agents have selected actions")
            return

        _LOGGER.info(f"Submitting multi-agent actions: {self._pending_actions}")
        self.all_actions_submitted.emit(dict(self._pending_actions))

    def get_actions(self) -> Dict[str, int]:
        """Get currently selected actions.

        Returns:
            Dict mapping agent_id to action index for agents that have selected.
        """
        return dict(self._pending_actions)

    def is_complete(self) -> bool:
        """Check if all agents have selected an action."""
        return len(self._pending_actions) == len(self._human_agents)

    def pending_agents(self) -> List[str]:
        """Get list of agents that haven't selected yet."""
        return [a for a in self._human_agents if a not in self._pending_actions]

    def reset(self) -> None:
        """Reset the panel for a new step."""
        self._on_clear_clicked()


__all__ = [
    "MultiAgentActionPanel",
    "AgentActionRow",
    "AGENT_COLORS",
]
