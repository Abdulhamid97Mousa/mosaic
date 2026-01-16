"""Operators tab widget for configuring action-selecting entities.

This module provides the OperatorsTab widget that allows configuring and
running multiple operators (LLM agents, RL policies) for scientific comparison.

Operators are the action-selecting entities in MOSAIC - they represent
who or what controls the agent in an environment.
"""

from __future__ import annotations

import logging
import random
from functools import partial
from typing import Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal

from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_UI_BOARD_CONFIG_DIALOG_OPENED,
    LOG_UI_BOARD_CONFIG_STATE_APPLIED,
    LOG_OP_GRID_CONFIG_DIALOG_OPENED,
    LOG_OP_GRID_CONFIG_STATE_APPLIED,
)
from gym_gui.services.operator import OperatorConfig
from gym_gui.ui.widgets.operator_config_widget import OperatorConfigWidget, VLLMServerInfo
from gym_gui.ui.widgets.vllm_server_widget import VLLMServerWidget

# Use operators namespace for dedicated operators.log routing
_LOGGER = logging.getLogger("gym_gui.operators.operators_tab")
_log = partial(log_constant, _LOGGER)


class CollapsibleSection(QtWidgets.QWidget):
    """A collapsible section with a header button to show/hide content.

    The section starts collapsed by default to save vertical space.
    """

    def __init__(
        self,
        title: str,
        parent: Optional[QtWidgets.QWidget] = None,
        collapsed: bool = True,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._is_collapsed = collapsed

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the collapsible section UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with toggle button
        self._header = QtWidgets.QPushButton(self)
        self._header.setCheckable(True)
        self._header.setChecked(not self._is_collapsed)
        self._header.clicked.connect(self._on_toggle)
        self._header.setStyleSheet(
            """
            QPushButton {
                text-align: left;
                padding: 8px 12px;
                font-size: 13px;
                font-weight: 600;
                color: #333;
                background-color: #f5f5f5;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #ebebeb;
                border-color: #bbb;
            }
            QPushButton:checked {
                background-color: #f0f0f0;
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
                border-bottom-color: transparent;
            }
            """
        )
        layout.addWidget(self._header)

        # Content container
        self._content_widget = QtWidgets.QWidget(self)
        self._content_layout = QtWidgets.QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(12, 12, 12, 12)
        self._content_widget.setStyleSheet(
            """
            QWidget {
                background-color: #fafafa;
                border: 1px solid #d0d0d0;
                border-top: none;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            QLabel {
                background: transparent;
                border: none;
                font-size: 12px;
                color: #444;
            }
            """
        )
        layout.addWidget(self._content_widget)

        # Set initial state
        self._update_header_text()
        self._content_widget.setVisible(not self._is_collapsed)

    def _update_header_text(self) -> None:
        """Update header text with expand/collapse indicator."""
        arrow = "▼" if not self._is_collapsed else "▶"
        action = "Hide" if not self._is_collapsed else "Show"
        self._header.setText(f"{arrow} {self._title} ({action})")

    def _on_toggle(self) -> None:
        """Toggle the collapsed state."""
        self._is_collapsed = not self._is_collapsed
        self._content_widget.setVisible(not self._is_collapsed)
        self._header.setChecked(not self._is_collapsed)
        self._update_header_text()

    def set_content_widget(self, widget: QtWidgets.QWidget) -> None:
        """Set the content widget for this collapsible section."""
        # Clear existing content
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        self._content_layout.addWidget(widget)

    def add_widget(self, widget: QtWidgets.QWidget) -> None:
        """Add a widget to the content area."""
        self._content_layout.addWidget(widget)

    def content_layout(self) -> QtWidgets.QVBoxLayout:
        """Get the content layout for adding widgets."""
        return self._content_layout

    def is_collapsed(self) -> bool:
        """Return whether the section is collapsed."""
        return self._is_collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        """Set the collapsed state."""
        if self._is_collapsed != collapsed:
            self._on_toggle()


class OperatorsTab(QtWidgets.QWidget):
    """Main tab for configuring Operators.

    Operators are the action-selecting entities in MOSAIC - they represent
    who or what controls the agent in an environment.

    Scientific Execution Model (inspired by BALROG):
    - Shared seed ensures all operators start with identical initial conditions
    - Step All advances all operators by exactly one step (lock-step execution)
    - Reset All resets all operators to the same seed for fair comparison
    - No arbitrary timing delays - steps are explicitly controlled
    """

    # Signals
    operators_changed = pyqtSignal(list)  # List[OperatorConfig]
    step_all_requested = pyqtSignal(int)  # Emit with current seed
    reset_all_requested = pyqtSignal(int)  # Emit with current seed
    stop_operators_requested = pyqtSignal()
    initialize_operator_requested = pyqtSignal(str, object, int)  # operator_id, config, seed
    step_player_requested = pyqtSignal(str, int)  # player_id, seed
    human_step_completed = pyqtSignal(str)  # operator_id - emitted when human completes their step
    human_action_requested = pyqtSignal(str, int)  # operator_id, action_index - request to step human operator

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        # Allow this widget to expand vertically
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self._step_count = 0
        self._is_running = False
        self._pettingzoo_mode = False
        self._current_player: str = ""
        # Human operator tracking
        self._has_human_operator = False
        self._human_step_completed = False
        self._human_operator_ids: list[str] = []  # IDs of human operators
        self._human_available_actions: list[int] = []  # Available action indices
        self._human_action_labels: list[str] = []  # Human-readable action labels

        # Turn-based execution tracking
        self._operator_order: list[str] = []  # Ordered list of operator IDs
        self._operator_types: dict[str, str] = {}  # operator_id -> "human", "llm", "rl", etc.
        self._current_agent_index: int = 0  # Which agent's turn it is (0-based)

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Explanation section (collapsible, starts collapsed)
        self._explanation_section = CollapsibleSection("What are Operators?", self, collapsed=True)
        explanation_text = QtWidgets.QLabel(
            "<p><b>Operators</b> are action-selecting entities that control agents in environments. "
            "MOSAIC introduces a unified <i>Operator</i> abstraction that bridges:</p>"
            "<ul>"
            "<li><b>LLM Agents</b> - GPT-4, Claude, Gemini making decisions</li>"
            "<li><b>RL Policies</b> - Trained neural network policies</li>"
            "</ul>"
            "<p>Add multiple operators below to run them in parallel and compare their performance side-by-side.</p>",
            self
        )
        explanation_text.setWordWrap(True)
        explanation_text.setTextFormat(QtCore.Qt.TextFormat.RichText)
        explanation_text.setStyleSheet("QLabel { color: #555; padding: 8px; background: white; }")
        self._explanation_section.add_widget(explanation_text)
        layout.addWidget(self._explanation_section)

        # vLLM Servers section (collapsible, starts collapsed)
        self._vllm_section = CollapsibleSection("vLLM Servers (Local Inference)", self, collapsed=True)
        self._vllm_server_widget = VLLMServerWidget(max_servers=2, parent=self)
        self._vllm_server_widget.server_status_changed.connect(self._on_vllm_server_status_changed)
        self._vllm_section.add_widget(self._vllm_server_widget)
        layout.addWidget(self._vllm_section)

        # Configure Operators section (always visible, takes remaining space)
        config_group = QtWidgets.QGroupBox("Configure Operators", self)
        config_group.setStyleSheet(
            """
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                background-color: #fafafa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                background-color: #fafafa;
                color: #333;
            }
            """
        )
        config_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        config_layout = QtWidgets.QVBoxLayout(config_group)
        config_layout.setContentsMargins(8, 12, 8, 8)

        # Multi-operator configuration widget - expands to fill available space
        self._operator_config_widget = OperatorConfigWidget(max_operators=8, parent=config_group)
        self._operator_config_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self._operator_config_widget.operators_changed.connect(self._on_operators_config_changed)
        self._operator_config_widget.initialize_requested.connect(self._on_initialize_requested)
        self._operator_config_widget.configure_requested.connect(self._on_configure_requested)
        self._operator_config_widget.vllm_refresh_requested.connect(self.refresh_vllm_servers)
        config_layout.addWidget(self._operator_config_widget, 1)  # Stretch factor for expansion

        # Add with stretch=1 so config_group expands to fill available vertical space
        layout.addWidget(config_group, 1)

        # Scientific Execution Controls (inspired by BALROG methodology)
        exec_group = QtWidgets.QGroupBox("Execution Controls (Scientific Comparison)", self)
        exec_group.setStyleSheet(
            """
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                background-color: #fafafa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                background-color: #fafafa;
                color: #333;
            }
            QLabel {
                font-size: 12px;
                color: #444;
                background: transparent;
                border: none;
            }
            QSpinBox {
                font-size: 12px;
                padding: 4px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            """
        )
        exec_layout = QtWidgets.QVBoxLayout(exec_group)
        exec_layout.setContentsMargins(12, 12, 12, 12)

        # Seed row - for reproducibility like BALROG
        seed_row = QtWidgets.QHBoxLayout()
        seed_row.setSpacing(8)

        seed_label = QtWidgets.QLabel("Shared Seed:", exec_group)
        seed_label.setToolTip(
            "All operators use this seed for identical initial conditions.\n"
            "This ensures fair, reproducible side-by-side comparison."
        )
        seed_row.addWidget(seed_label)

        self._seed_spin = QtWidgets.QSpinBox(exec_group)
        self._seed_spin.setRange(0, 2147483647)  # Max int32
        self._seed_spin.setValue(42)  # Default seed like BALROG
        self._seed_spin.setToolTip(
            "Seed for random number generators.\n"
            "Same seed = same initial environment state for all operators."
        )
        seed_row.addWidget(self._seed_spin, 1)

        self._random_seed_button = QtWidgets.QPushButton("Random", exec_group)
        self._random_seed_button.setToolTip("Generate a new random seed")
        self._random_seed_button.setStyleSheet("QPushButton { padding: 4px 8px; }")
        self._random_seed_button.clicked.connect(self._on_random_seed_clicked)
        seed_row.addWidget(self._random_seed_button)

        exec_layout.addLayout(seed_row)

        # Control buttons row
        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(8)

        self._reset_all_button = QtWidgets.QPushButton("Reset All", exec_group)
        self._reset_all_button.setToolTip(
            "Reset all operators to initial state with the shared seed.\n"
            "All environments will have identical starting conditions."
        )
        self._reset_all_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 12px; background-color: #FF9800; color: white; }"
            "QPushButton:hover { background-color: #F57C00; }"
            "QPushButton:pressed { background-color: #EF6C00; }"
            "QPushButton:disabled { background-color: #FFCC80; color: #FFF3E0; }"
        )
        self._reset_all_button.setEnabled(False)
        self._reset_all_button.clicked.connect(self._on_reset_all_clicked)
        button_row.addWidget(self._reset_all_button)

        self._step_all_button = QtWidgets.QPushButton("Step All", exec_group)
        self._step_all_button.setToolTip(
            "Step all AI operators (RL, LLM, VLM) until the next Human operator.\n"
            "Disabled when waiting for a Human to take their turn."
        )
        self._step_all_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 12px; background-color: #4CAF50; color: white; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:pressed { background-color: #388E3C; }"
            "QPushButton:disabled { background-color: #A5D6A7; color: #E8F5E9; }"
        )
        self._step_all_button.setEnabled(False)
        self._step_all_button.clicked.connect(self._on_step_all_clicked)
        button_row.addWidget(self._step_all_button)

        # Human Step button - for human operators to take their turn
        self._human_step_button = QtWidgets.QPushButton("Human Step", exec_group)
        self._human_step_button.setToolTip(
            "Take a step as the current Human operator.\n"
            "Use keyboard (WASD/Arrows) or click action buttons below.\n"
            "Enabled only when it's a Human operator's turn."
        )
        self._human_step_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 12px; background-color: #FF5722; color: white; }"
            "QPushButton:hover { background-color: #E64A19; }"
            "QPushButton:pressed { background-color: #D84315; }"
            "QPushButton:disabled { background-color: #FFAB91; color: #FBE9E7; }"
        )
        self._human_step_button.setEnabled(False)
        self._human_step_button.setVisible(False)  # Hidden until human operator is configured
        self._human_step_button.clicked.connect(self._on_human_step_button_clicked)
        button_row.addWidget(self._human_step_button)

        # PettingZoo player-specific step buttons (hidden by default)
        self._step_player_0_btn = QtWidgets.QPushButton("Step White", exec_group)
        self._step_player_0_btn.setToolTip("Step player_0 (White) - their turn to move")
        self._step_player_0_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 12px; background-color: #4CAF50; color: white; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:pressed { background-color: #388E3C; }"
            "QPushButton:disabled { background-color: #A5D6A7; color: #E8F5E9; }"
        )
        self._step_player_0_btn.setEnabled(False)
        self._step_player_0_btn.setVisible(False)  # Hidden by default
        self._step_player_0_btn.clicked.connect(lambda: self._on_step_player_clicked("player_0"))
        button_row.addWidget(self._step_player_0_btn)

        self._step_player_1_btn = QtWidgets.QPushButton("Step Black", exec_group)
        self._step_player_1_btn.setToolTip("Step player_1 (Black) - their turn to move")
        self._step_player_1_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 12px; background-color: #4CAF50; color: white; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:pressed { background-color: #388E3C; }"
            "QPushButton:disabled { background-color: #A5D6A7; color: #E8F5E9; }"
        )
        self._step_player_1_btn.setEnabled(False)
        self._step_player_1_btn.setVisible(False)  # Hidden by default
        self._step_player_1_btn.clicked.connect(lambda: self._on_step_player_clicked("player_1"))
        button_row.addWidget(self._step_player_1_btn)

        self._stop_operators_button = QtWidgets.QPushButton("Stop All", exec_group)
        self._stop_operators_button.setToolTip("Stop all running operators and release resources")
        self._stop_operators_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px 12px; background-color: #F44336; color: white; }"
            "QPushButton:hover { background-color: #E53935; }"
            "QPushButton:pressed { background-color: #D32F2F; }"
            "QPushButton:disabled { background-color: #EF9A9A; color: #FFEBEE; }"
        )
        self._stop_operators_button.setEnabled(False)
        self._stop_operators_button.clicked.connect(self._on_stop_operators_clicked)
        button_row.addWidget(self._stop_operators_button)

        exec_layout.addLayout(button_row)

        # Step counter and status row
        status_row = QtWidgets.QHBoxLayout()
        status_row.setSpacing(12)

        self._step_count_label = QtWidgets.QLabel("Steps: 0", exec_group)
        self._step_count_label.setStyleSheet(
            "QLabel { font-weight: bold; color: #1976D2; padding: 4px 8px; "
            "background-color: #E3F2FD; border-radius: 4px; }"
        )
        status_row.addWidget(self._step_count_label)

        self._status_label = QtWidgets.QLabel("Ready", exec_group)
        self._status_label.setStyleSheet(
            "QLabel { color: #666; padding: 4px; font-style: italic; }"
        )
        status_row.addWidget(self._status_label)

        # Turn indicator for PettingZoo multi-agent games
        self._turn_indicator_label = QtWidgets.QLabel("", exec_group)
        self._turn_indicator_label.setStyleSheet(
            "QLabel { font-weight: bold; color: #F57C00; padding: 4px 8px; "
            "background-color: #FFF3E0; border-radius: 4px; }"
        )
        self._turn_indicator_label.setVisible(False)  # Hidden by default
        status_row.addWidget(self._turn_indicator_label)

        status_row.addStretch(1)

        exec_layout.addLayout(status_row)

        layout.addWidget(exec_group)

        # Human Actions Panel (visible only when human operator is waiting)
        self._human_actions_group = QtWidgets.QGroupBox("Human Actions", self)
        self._human_actions_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 2px solid #FF5722; border-radius: 6px; "
            "margin-top: 8px; padding: 8px; background-color: #FFF3E0; }"
            "QGroupBox::title { color: #FF5722; subcontrol-origin: margin; "
            "subcontrol-position: top left; padding: 2px 8px; }"
        )
        self._human_actions_group.setVisible(False)  # Hidden by default

        human_actions_layout = QtWidgets.QVBoxLayout(self._human_actions_group)
        human_actions_layout.setContentsMargins(8, 12, 8, 8)
        human_actions_layout.setSpacing(8)

        # Instructions label
        self._human_instructions_label = QtWidgets.QLabel(
            "Your turn! Click an action button or use keyboard shortcuts:",
            self._human_actions_group
        )
        self._human_instructions_label.setStyleSheet("font-weight: normal; color: #555;")
        self._human_instructions_label.setWordWrap(True)
        human_actions_layout.addWidget(self._human_instructions_label)

        # Action buttons container
        self._human_action_buttons_widget = QtWidgets.QWidget(self._human_actions_group)
        self._human_action_buttons_layout = QtWidgets.QGridLayout(self._human_action_buttons_widget)
        self._human_action_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._human_action_buttons_layout.setSpacing(4)
        human_actions_layout.addWidget(self._human_action_buttons_widget)

        # Keyboard shortcut hint
        self._keyboard_hint_label = QtWidgets.QLabel(
            "Keyboard: WASD/Arrows for movement, Space/G for pickup, E for toggle",
            self._human_actions_group
        )
        self._keyboard_hint_label.setStyleSheet("font-size: 9px; color: #888; font-style: italic;")
        self._keyboard_hint_label.setWordWrap(True)
        human_actions_layout.addWidget(self._keyboard_hint_label)

        layout.addWidget(self._human_actions_group)

        # No stretch at end - let config_group (with stretch=1) fill available space

    def _on_operators_config_changed(self, configs: list) -> None:
        """Handle operator configuration changes."""
        self.operators_changed.emit(configs)
        has_operators = len(configs) > 0
        self._reset_all_button.setEnabled(has_operators)

        # Build operator order and types for turn-based execution
        self._operator_order = [cfg.operator_id for cfg in configs]
        self._operator_types = {}
        for cfg in configs:
            # Determine operator type based on worker_id
            if cfg.worker_id == "human_worker":
                self._operator_types[cfg.operator_id] = "human"
            elif cfg.worker_id in ("cleanrl_worker", "rl_worker", "sb3_worker"):
                self._operator_types[cfg.operator_id] = "rl"
            else:
                # LLM, VLM, etc.
                self._operator_types[cfg.operator_id] = "llm"

        # Detect human operators (worker_id == "human_worker")
        self._human_operator_ids = [
            cfg.operator_id for cfg in configs
            if cfg.worker_id == "human_worker"
        ]
        self._has_human_operator = len(self._human_operator_ids) > 0
        self._human_step_completed = False  # Reset on config change
        self._current_agent_index = 0  # Reset to first agent

        # Show/hide Human Step button based on whether there are human operators
        self._human_step_button.setVisible(self._has_human_operator)

        # Step All only enabled after Reset All is done
        if not has_operators:
            self._step_all_button.setEnabled(False)
            self._human_step_button.setEnabled(False)
            self._stop_operators_button.setEnabled(False)

        # Update status to show human operator info
        if self._has_human_operator:
            self._status_label.setText(f"Human operator(s): {len(self._human_operator_ids)}")

    def _on_random_seed_clicked(self) -> None:
        """Generate a random seed."""
        new_seed = random.randint(0, 2147483647)
        self._seed_spin.setValue(new_seed)

    def _on_reset_all_clicked(self) -> None:
        """Handle Reset All button click."""
        seed = self._seed_spin.value()
        self._step_count = 0
        self._step_count_label.setText("Steps: 0")
        self._status_label.setText(f"Resetting with seed {seed}...")
        self._is_running = True
        self._human_step_completed = False  # Reset human step state
        self.reset_all_requested.emit(seed)

        # Enable stop button
        self._stop_operators_button.setEnabled(True)

        # Step All: enabled only if no human operators, or after human steps
        if self._has_human_operator:
            self._step_all_button.setEnabled(False)
            self._status_label.setText(f"Waiting for Human... (seed={seed})")
        else:
            self._step_all_button.setEnabled(True)
            self._status_label.setText(f"Running (seed={seed})")

    def _on_step_all_clicked(self) -> None:
        """Handle Step All button click.

        For Human operators: After AI operators step, Step All is disabled
        until the human makes their next move.
        """
        if not self._is_running:
            return
        seed = self._seed_spin.value()
        self._step_count += 1
        self._step_count_label.setText(f"Steps: {self._step_count}")
        self.step_all_requested.emit(seed)

        # If there are human operators, disable Step All until human steps again
        if self._has_human_operator:
            self._human_step_completed = False
            self._step_all_button.setEnabled(False)
            self._status_label.setText(f"Waiting for Human... (step {self._step_count})")

    def _on_stop_operators_clicked(self) -> None:
        """Handle Stop All button click."""
        self.stop_operators_requested.emit()
        self._is_running = False
        self._step_all_button.setEnabled(False)
        self._stop_operators_button.setEnabled(False)
        self._status_label.setText("Stopped")

    def _on_initialize_requested(self, operator_id: str, config: OperatorConfig) -> None:
        """Handle initialize request from an operator row.

        Passes the shared seed for controlled scientific comparison.
        """
        seed = self.get_current_seed()
        self.initialize_operator_requested.emit(operator_id, config, seed)

    def _on_configure_requested(self, operator_id: str, config: OperatorConfig) -> None:
        """Handle configure request from an operator row.

        Opens an environment-specific configuration dialog to set up custom
        starting positions/states. Supports both:
        - Board games (chess, Go, checkers) via BoardConfigDialogFactory
        - Grid environments (MiniGrid, BabyAI) via GridConfigDialogFactory
        """
        from gym_gui.ui.widgets.operators_board_config_form import BoardConfigDialogFactory
        from gym_gui.ui.widgets.operators_grid_config_form import GridConfigDialogFactory

        env_id = config.task  # e.g., "chess_v6" or "MiniGrid-Empty-8x8-v0"

        # Get current state if any (from worker settings)
        current_state = None
        if config.workers:
            first_worker_id = next(iter(config.workers.keys()))
            current_state = config.workers[first_worker_id].settings.get("initial_state")

        # Try board game factory first (chess, Go, checkers, etc.)
        if BoardConfigDialogFactory.supports(env_id):
            _log(
                LOG_UI_BOARD_CONFIG_DIALOG_OPENED,
                extra={"operator_id": operator_id, "game_id": env_id},
            )

            dialog = BoardConfigDialogFactory.create(env_id, current_state, self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                custom_state = dialog.get_state()
                _log(
                    LOG_UI_BOARD_CONFIG_STATE_APPLIED,
                    extra={
                        "operator_id": operator_id,
                        "game_id": env_id,
                        "state_preview": custom_state[:50] if custom_state else "",
                    },
                )
                self._operator_config_widget.set_operator_initial_state(operator_id, custom_state)
            return

        # Try grid environment factory (MiniGrid, BabyAI, etc.)
        if GridConfigDialogFactory.supports(env_id):
            _log(
                LOG_OP_GRID_CONFIG_DIALOG_OPENED,
                extra={"operator_id": operator_id, "env_id": env_id},
            )

            # For grid environments, current_state is a dict (JSON serializable)
            import json
            initial_dict = None
            if current_state:
                try:
                    initial_dict = json.loads(current_state) if isinstance(current_state, str) else current_state
                except json.JSONDecodeError:
                    _LOGGER.warning(f"Could not parse initial state as JSON: {current_state[:50]}")

            dialog = GridConfigDialogFactory.create(env_id, initial_dict, self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                custom_state_dict = dialog.get_state()
                # Store as JSON string for consistency with board games
                custom_state = json.dumps(custom_state_dict)
                _log(
                    LOG_OP_GRID_CONFIG_STATE_APPLIED,
                    extra={
                        "operator_id": operator_id,
                        "env_id": env_id,
                        "grid_size": f"{custom_state_dict.get('rows', '?')}x{custom_state_dict.get('cols', '?')}",
                    },
                )
                self._operator_config_widget.set_operator_initial_state(operator_id, custom_state)
            return

        # No factory supports this environment
        _LOGGER.warning(f"No configuration dialog for environment: {env_id}")

    def set_step_count(self, count: int) -> None:
        """Set the step count (called externally when steps complete)."""
        self._step_count = count
        self._step_count_label.setText(f"Steps: {count}")

    def set_status(self, status: str) -> None:
        """Set the status label text."""
        self._status_label.setText(status)

    def set_turn_indicator(self, player_id: str, visible: bool = True) -> None:
        """Set the turn indicator for PettingZoo multi-agent games.

        Args:
            player_id: Current player (e.g., "player_0", "player_1").
            visible: Whether to show the indicator.
        """
        if visible and player_id:
            # Map player_id to friendly name
            player_names = {
                "player_0": "White",
                "player_1": "Black",
            }
            friendly_name = player_names.get(player_id, player_id)
            self._turn_indicator_label.setText(f"Next: {friendly_name} ({player_id})")
            self._turn_indicator_label.setVisible(True)
        else:
            self._turn_indicator_label.setVisible(False)

    def set_pettingzoo_mode(self, enabled: bool) -> None:
        """Enable or disable PettingZoo multi-agent mode.

        When enabled:
        - Hides "Step All" button
        - Shows player-specific step buttons (Step White, Step Black)

        Args:
            enabled: True to enable PettingZoo mode, False for normal mode.
        """
        _LOGGER.debug("set_pettingzoo_mode: enabled=%s", enabled)
        self._pettingzoo_mode = enabled
        self._step_all_button.setVisible(not enabled)
        self._step_player_0_btn.setVisible(enabled)
        self._step_player_1_btn.setVisible(enabled)
        _LOGGER.debug(
            "set_pettingzoo_mode: step_all visible=%s, player btns visible=%s",
            not enabled, enabled,
        )

        if not enabled:
            # Reset to normal mode
            self._step_player_0_btn.setEnabled(False)
            self._step_player_1_btn.setEnabled(False)
            self._current_player = ""

    def set_current_player(self, player_id: str) -> None:
        """Set which player's turn it is (enables their button, disables the other).

        Args:
            player_id: The player whose turn it is ("player_0" or "player_1").
        """
        _LOGGER.debug(
            "set_current_player: player_id=%s",
            player_id,
            extra={"agent_id": player_id},
        )
        self._current_player = player_id

        if player_id == "player_0":
            self._step_player_0_btn.setEnabled(True)
            self._step_player_1_btn.setEnabled(False)
            _LOGGER.debug("Enabled player_0 button, disabled player_1 button")
        elif player_id == "player_1":
            self._step_player_0_btn.setEnabled(False)
            self._step_player_1_btn.setEnabled(True)
            _LOGGER.debug("Disabled player_0 button, enabled player_1 button")
        else:
            # Unknown player or game over
            self._step_player_0_btn.setEnabled(False)
            self._step_player_1_btn.setEnabled(False)
            _LOGGER.debug("Disabled both buttons (game over or unknown player)")

    def _on_step_player_clicked(self, player_id: str) -> None:
        """Handle click on a player-specific step button.

        Args:
            player_id: Which player's button was clicked.
        """
        seed = self._seed_spin.value()
        # Disable button immediately to prevent double-clicks
        if player_id == "player_0":
            self._step_player_0_btn.setEnabled(False)
        else:
            self._step_player_1_btn.setEnabled(False)
        self._step_count += 1
        self._step_count_label.setText(f"Steps: {self._step_count}")
        self.step_player_requested.emit(player_id, seed)

    def get_current_seed(self) -> int:
        """Get the current seed value."""
        return self._seed_spin.value()

    def on_human_action_received(self, operator_id: str) -> None:
        """Called when a human operator completes their action via the UI.

        This enables the Step All button so AI operators can take their turn.

        Args:
            operator_id: The ID of the human operator that acted.
        """
        if not self._has_human_operator:
            return

        if operator_id not in self._human_operator_ids:
            _LOGGER.warning(f"Unknown human operator: {operator_id}")
            return

        _LOGGER.info(f"Human operator {operator_id} completed action")
        self._human_step_completed = True
        self._step_all_button.setEnabled(True)
        self._status_label.setText(f"Human acted - Ready to Step All (step {self._step_count})")

        # Hide the human actions panel until next step
        self._human_actions_group.setVisible(False)

        # Emit signal for any listeners
        self.human_step_completed.emit(operator_id)

    @property
    def has_human_operator(self) -> bool:
        """Check if there are any human operators configured."""
        return self._has_human_operator

    @property
    def human_operator_ids(self) -> list[str]:
        """Get the list of human operator IDs."""
        return self._human_operator_ids.copy()

    def _on_vllm_server_status_changed(self, server_id: int, status: str, base_url: str) -> None:
        """Handle vLLM server status changes.

        This allows the UI to update operator dropdowns when servers start/stop.
        Collects all server states and passes them to the operator config widget.
        """
        _LOGGER.info(f"vLLM Server {server_id} status: {status}, URL: {base_url}")

        # Collect all server info from the vLLM server widget
        servers: list[VLLMServerInfo] = []
        for sid, state in self._vllm_server_widget._server_states.items():
            if state.model_id:  # Only include servers with a model selected
                server_info = VLLMServerInfo(
                    server_id=sid,
                    port=state.port,
                    model_id=state.model_id,
                    base_url=f"http://127.0.0.1:{state.port}/v1",
                    status=state.status,
                )
                servers.append(server_info)

        # Update operator config widget with available servers
        self._operator_config_widget.set_vllm_servers(servers)

    def refresh_vllm_servers(self) -> None:
        """Manually refresh the vLLM server list for operator dropdowns.

        Call this to sync operator config with current vLLM server state.
        Useful when servers were started before operators were configured.
        """
        servers: list[VLLMServerInfo] = []
        for sid, state in self._vllm_server_widget._server_states.items():
            if state.model_id:  # Only include servers with a model selected
                server_info = VLLMServerInfo(
                    server_id=sid,
                    port=state.port,
                    model_id=state.model_id,
                    base_url=f"http://127.0.0.1:{state.port}/v1",
                    status=state.status,
                )
                servers.append(server_info)

        _LOGGER.debug(f"Refreshing vLLM servers: {len(servers)} available")
        self._operator_config_widget.set_vllm_servers(servers)

    def set_operator_environment_size(
        self, operator_id: str, width: int, height: int, container_size: int | None = None
    ) -> None:
        """Set the environment size for a specific operator.

        Args:
            operator_id: The operator's unique ID
            width: Rendered environment width in pixels (image size)
            height: Rendered environment height in pixels (image size)
            container_size: Optional container display size in pixels
        """
        self._operator_config_widget.set_operator_environment_size(
            operator_id, width, height, container_size
        )

    @property
    def vllm_server_widget(self) -> VLLMServerWidget:
        """Get the vLLM server management widget."""
        return self._vllm_server_widget

    @property
    def operator_config_widget(self) -> OperatorConfigWidget:
        """Get the operator configuration widget."""
        return self._operator_config_widget

    # --- Human Action Panel Methods ---

    def set_human_available_actions(
        self, actions: list[int], labels: list[str], show_panel: bool = True
    ) -> None:
        """Set the available actions for human operators and populate buttons.

        Args:
            actions: List of valid action indices.
            labels: Human-readable labels for each action.
            show_panel: Whether to show the action panel.
        """
        self._human_available_actions = actions
        self._human_action_labels = labels
        self._populate_human_action_buttons(actions, labels)

        if show_panel and self._has_human_operator:
            self._human_actions_group.setVisible(True)
            self._status_label.setText(f"Waiting for Human... (step {self._step_count})")

    def _populate_human_action_buttons(
        self, actions: list[int], labels: list[str]
    ) -> None:
        """Populate the human action buttons grid.

        Args:
            actions: List of action indices.
            labels: Human-readable labels for each action.
        """
        # Clear existing buttons
        while self._human_action_buttons_layout.count():
            item = self._human_action_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create buttons in a grid (4 columns max)
        cols = 4
        for i, (action_idx, label) in enumerate(zip(actions, labels)):
            row = i // cols
            col = i % cols

            btn = QtWidgets.QPushButton(f"{label}", self._human_action_buttons_widget)
            btn.setToolTip(f"Action {action_idx}: {label}")
            btn.setStyleSheet(
                "QPushButton { padding: 8px 12px; background-color: #FF5722; color: white; "
                "font-weight: bold; border-radius: 4px; min-width: 80px; }"
                "QPushButton:hover { background-color: #E64A19; }"
                "QPushButton:pressed { background-color: #D84315; }"
            )
            btn.clicked.connect(lambda checked, a=action_idx: self._on_human_action_button_clicked(a))
            self._human_action_buttons_layout.addWidget(btn, row, col)

    def _on_human_action_button_clicked(self, action: int) -> None:
        """Handle click on a human action button.

        Args:
            action: The action index that was clicked.
        """
        if not self._human_operator_ids:
            _LOGGER.warning("Human action clicked but no human operators configured")
            return

        # Get the first human operator (for now, support single human)
        operator_id = self._human_operator_ids[0]
        _LOGGER.info(f"Human action button clicked: operator={operator_id}, action={action}")

        # Emit signal to main_window to process the action
        self.human_action_requested.emit(operator_id, action)

    def _on_human_step_button_clicked(self) -> None:
        """Handle Human Step button click.

        This button is used in multi-operator sequential execution:
        - When it's a Human operator's turn in the sequence
        - Step All is disabled
        - Human Step enables keyboard/mouse controller
        - Human makes ONE action
        - Step All re-enables to continue to next operator
        """
        if not self._human_operator_ids:
            _LOGGER.warning("Human Step clicked but no human operators configured")
            return

        _LOGGER.info("Human Step button clicked - activating controller for human operator")

        # Show the action panel for discrete action environments
        self.show_human_actions_panel(show=True)

        # Update status
        self._status_label.setText(f"Human's Turn - Use keyboard/mouse or click action button")

    def show_human_actions_panel(self, show: bool = True) -> None:
        """Show or hide the human actions panel.

        Args:
            show: True to show, False to hide.
        """
        if self._has_human_operator:
            self._human_actions_group.setVisible(show)
            if show:
                self._status_label.setText(f"Waiting for Human... (step {self._step_count})")

    def hide_human_actions_panel(self) -> None:
        """Hide the human actions panel after action is taken."""
        self._human_actions_group.setVisible(False)


__all__ = ["OperatorsTab"]
