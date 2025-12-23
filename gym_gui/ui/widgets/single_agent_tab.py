"""Single-Agent Mode tab with subtabs for Operators and Workers.

This module provides the SingleAgentTab widget that organizes single-agent
mode into two logical subtabs:
- Operators: Configure action-selecting entities (Human, LLM, RL policies)
- Workers: Select worker backends and control training/evaluation

This follows the same pattern as MultiAgentTab which has subtabs for
Human vs Agent, Cooperation, and Competition modes.
"""

from __future__ import annotations

import random
from typing import Optional, List

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal

from gym_gui.services.operator import OperatorConfig
from gym_gui.ui.widgets.operator_config_widget import OperatorConfigWidget
from gym_gui.ui.worker_catalog import WorkerDefinition, get_worker_catalog


class OperatorsSubTab(QtWidgets.QWidget):
    """Subtab for configuring Operators.

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
    initialize_operator_requested = pyqtSignal(str, object)  # operator_id, config

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._step_count = 0
        self._is_running = False
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Explanation section
        explanation_group = QtWidgets.QGroupBox("What are Operators?", self)
        explanation_layout = QtWidgets.QVBoxLayout(explanation_group)

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
        explanation_text.setStyleSheet("QLabel { color: #555; padding: 4px; }")
        explanation_layout.addWidget(explanation_text)
        layout.addWidget(explanation_group)

        # Configure Operators section
        config_group = QtWidgets.QGroupBox("Configure Operators", self)
        config_layout = QtWidgets.QVBoxLayout(config_group)

        # Multi-operator configuration widget
        self._operator_config_widget = OperatorConfigWidget(max_operators=8, parent=config_group)
        self._operator_config_widget.operators_changed.connect(self._on_operators_config_changed)
        self._operator_config_widget.initialize_requested.connect(self._on_initialize_requested)
        config_layout.addWidget(self._operator_config_widget)

        layout.addWidget(config_group)

        # Scientific Execution Controls (inspired by BALROG methodology)
        exec_group = QtWidgets.QGroupBox("Execution Controls (Scientific Comparison)", self)
        exec_layout = QtWidgets.QVBoxLayout(exec_group)

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
            "Advance ALL operators by exactly one step (lock-step execution).\n"
            "Each operator's agent selects one action simultaneously.\n"
            "This ensures scientifically fair side-by-side comparison."
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

        status_row.addStretch(1)

        exec_layout.addLayout(status_row)

        layout.addWidget(exec_group)

        layout.addStretch(1)

    def _on_operators_config_changed(self, configs: list) -> None:
        """Handle operator configuration changes."""
        self.operators_changed.emit(configs)
        has_operators = len(configs) > 0
        self._reset_all_button.setEnabled(has_operators)
        # Step All only enabled after Reset All is done
        if not has_operators:
            self._step_all_button.setEnabled(False)
            self._stop_operators_button.setEnabled(False)

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
        self.reset_all_requested.emit(seed)
        # Enable step button after reset
        self._step_all_button.setEnabled(True)
        self._stop_operators_button.setEnabled(True)
        self._status_label.setText(f"Running (seed={seed})")

    def _on_step_all_clicked(self) -> None:
        """Handle Step All button click."""
        if not self._is_running:
            return
        seed = self._seed_spin.value()
        self._step_count += 1
        self._step_count_label.setText(f"Steps: {self._step_count}")
        self.step_all_requested.emit(seed)

    def _on_stop_operators_clicked(self) -> None:
        """Handle Stop All button click."""
        self.stop_operators_requested.emit()
        self._is_running = False
        self._step_all_button.setEnabled(False)
        self._stop_operators_button.setEnabled(False)
        self._status_label.setText("Stopped")

    def _on_initialize_requested(self, operator_id: str, config) -> None:
        """Handle initialize request from an operator row."""
        self.initialize_operator_requested.emit(operator_id, config)

    def set_step_count(self, count: int) -> None:
        """Set the step count (called externally when steps complete)."""
        self._step_count = count
        self._step_count_label.setText(f"Steps: {count}")

    def set_status(self, status: str) -> None:
        """Set the status label text."""
        self._status_label.setText(status)

    def get_current_seed(self) -> int:
        """Get the current seed value."""
        return self._seed_spin.value()

    @property
    def operator_config_widget(self) -> OperatorConfigWidget:
        """Get the operator configuration widget."""
        return self._operator_config_widget


class WorkersSubTab(QtWidgets.QWidget):
    """Subtab for Worker integration and training controls.

    Workers are subprocess backends that execute training and inference.
    """

    # Signals
    worker_changed = pyqtSignal(str)  # worker_id
    train_requested = pyqtSignal()
    evaluate_requested = pyqtSignal()
    resume_requested = pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._worker_definitions: List[WorkerDefinition] = list(get_worker_catalog())
        self._build_ui()
        self._populate_worker_combo()
        self._connect_signals()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Explanation section
        explanation_group = QtWidgets.QGroupBox("What are Workers?", self)
        explanation_layout = QtWidgets.QVBoxLayout(explanation_group)

        explanation_text = QtWidgets.QLabel(
            "<p><b>Workers</b> are subprocess backends that handle training and inference. "
            "Each worker provides a specific implementation:</p>"
            "<ul>"
            "<li><b>BARLOG LLM Worker</b> - LLM agents using BALROG (OpenAI, Claude, Gemini)</li>"
            "<li><b>CleanRL Worker</b> - Single-file RL implementations (PPO, DQN, SAC)</li>"
            "<li><b>XuanCe Worker</b> - Comprehensive RL library with 50+ algorithms</li>"
            "<li><b>Ray RLlib Worker</b> - Scalable distributed RL training</li>"
            "</ul>"
            "<p>Workers run as isolated subprocesses, providing process isolation, "
            "separate Python environments, and GPU isolation.</p>",
            self
        )
        explanation_text.setWordWrap(True)
        explanation_text.setTextFormat(QtCore.Qt.TextFormat.RichText)
        explanation_text.setStyleSheet("QLabel { color: #555; padding: 4px; }")
        explanation_layout.addWidget(explanation_text)
        layout.addWidget(explanation_group)

        # Worker Selection section
        worker_group = QtWidgets.QGroupBox("Worker Selection", self)
        worker_layout = QtWidgets.QVBoxLayout(worker_group)

        self._worker_combo = QtWidgets.QComboBox(worker_group)
        self._worker_combo.setEnabled(bool(self._worker_definitions))
        worker_layout.addWidget(self._worker_combo)

        self._worker_description = QtWidgets.QLabel("Select a worker to view capabilities.", worker_group)
        self._worker_description.setWordWrap(True)
        self._worker_description.setStyleSheet(
            "QLabel { color: #666; padding: 8px; background-color: #f5f5f5; border-radius: 4px; }"
        )
        worker_layout.addWidget(self._worker_description)

        layout.addWidget(worker_group)

        # Training section
        training_group = QtWidgets.QGroupBox("Headless Training", self)
        training_layout = QtWidgets.QVBoxLayout(training_group)

        self._train_agent_button = QtWidgets.QPushButton("Train Agent", training_group)
        self._train_agent_button.setToolTip(
            "Start a fresh headless training run.\n"
            "Training will run in the background with live telemetry streaming."
        )
        self._train_agent_button.setEnabled(False)
        self._train_agent_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 8px; background-color: #1976d2; color: white; }"
            "QPushButton:hover { background-color: #1565c0; }"
            "QPushButton:pressed { background-color: #0d47a1; }"
            "QPushButton:disabled { background-color: #90caf9; color: #E3F2FD; }"
        )
        training_layout.addWidget(self._train_agent_button)

        self._evaluate_policy_button = QtWidgets.QPushButton("Evaluate Policy", training_group)
        self._evaluate_policy_button.setToolTip(
            "Select an existing policy or checkpoint to evaluate inside the GUI."
        )
        self._evaluate_policy_button.setEnabled(False)
        self._evaluate_policy_button.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; background-color: #388e3c; color: white; }"
            "QPushButton:hover { background-color: #2e7d32; }"
            "QPushButton:pressed { background-color: #1b5e20; }"
            "QPushButton:disabled { background-color: #a5d6a7; color: #E8F5E9; }"
        )
        training_layout.addWidget(self._evaluate_policy_button)

        self._resume_training_button = QtWidgets.QPushButton("Resume Training", training_group)
        self._resume_training_button.setToolTip(
            "Load a checkpoint and continue training from where it left off."
        )
        self._resume_training_button.setEnabled(False)
        self._resume_training_button.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; background-color: #f57c00; color: white; }"
            "QPushButton:hover { background-color: #ef6c00; }"
            "QPushButton:pressed { background-color: #e65100; }"
            "QPushButton:disabled { background-color: #ffcc80; color: #FFF3E0; }"
        )
        training_layout.addWidget(self._resume_training_button)

        layout.addWidget(training_group)

        layout.addStretch(1)

    def _populate_worker_combo(self) -> None:
        """Populate the worker combo box."""
        self._worker_combo.clear()
        for worker in self._worker_definitions:
            self._worker_combo.addItem(worker.display_name, worker.worker_id)
        if self._worker_definitions:
            self._update_worker_description(0)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._worker_combo.currentIndexChanged.connect(self._on_worker_selection_changed)
        self._train_agent_button.clicked.connect(self._on_train_clicked)
        self._evaluate_policy_button.clicked.connect(self._on_evaluate_clicked)
        self._resume_training_button.clicked.connect(self._on_resume_clicked)

    def _on_worker_selection_changed(self, index: int) -> None:
        """Handle worker selection change."""
        self._update_worker_description(index)
        if 0 <= index < len(self._worker_definitions):
            worker = self._worker_definitions[index]
            self.worker_changed.emit(worker.worker_id)
            # Enable buttons based on worker capabilities
            self._train_agent_button.setEnabled(worker.supports_training)
            self._evaluate_policy_button.setEnabled(worker.supports_policy_load)
            self._resume_training_button.setEnabled(worker.supports_training)

    def _update_worker_description(self, index: int) -> None:
        """Update the worker description label."""
        if index < 0 or index >= len(self._worker_definitions):
            self._worker_description.setText("Select a worker to view capabilities.")
            return

        worker = self._worker_definitions[index]
        capabilities = []
        if worker.supports_training:
            capabilities.append("Training")
        if worker.supports_policy_load:
            capabilities.append("Policy evaluation")
        if worker.requires_live_telemetry:
            capabilities.append("Live telemetry")

        caps_text = f"<b>Capabilities:</b> {', '.join(capabilities)}" if capabilities else ""

        self._worker_description.setText(
            f"{worker.description}<br><br>{caps_text}"
        )

    def _on_train_clicked(self) -> None:
        """Handle train button click."""
        self.train_requested.emit()

    def _on_evaluate_clicked(self) -> None:
        """Handle evaluate button click."""
        self.evaluate_requested.emit()

    def _on_resume_clicked(self) -> None:
        """Handle resume button click."""
        self.resume_requested.emit()

    @property
    def current_worker_id(self) -> Optional[str]:
        """Get the currently selected worker ID."""
        index = self._worker_combo.currentIndex()
        if 0 <= index < len(self._worker_definitions):
            return self._worker_definitions[index].worker_id
        return None

    @property
    def worker_combo(self) -> QtWidgets.QComboBox:
        """Get the worker combo box (for backward compatibility)."""
        return self._worker_combo

    @property
    def worker_description(self) -> QtWidgets.QLabel:
        """Get the worker description label (for backward compatibility)."""
        return self._worker_description

    @property
    def train_agent_button(self) -> QtWidgets.QPushButton:
        """Get the train agent button (for backward compatibility)."""
        return self._train_agent_button

    @property
    def evaluate_policy_button(self) -> QtWidgets.QPushButton:
        """Get the evaluate policy button (for backward compatibility)."""
        return self._evaluate_policy_button

    @property
    def resume_training_button(self) -> QtWidgets.QPushButton:
        """Get the resume training button (for backward compatibility)."""
        return self._resume_training_button


class SingleAgentTab(QtWidgets.QWidget):
    """Main Single-Agent Mode tab with subtabs for Operators and Workers.

    Contains:
    - Operators: Configure action-selecting entities
    - Workers: Select backends and control training

    This follows the same pattern as MultiAgentTab which has subtabs for
    Human vs Agent, Cooperation, and Competition modes.

    Scientific Execution Model (forwarded from OperatorsSubTab):
    - step_all_requested: Advance all operators by one step (lock-step)
    - reset_all_requested: Reset all operators with shared seed
    """

    # Forwarded signals from Operators subtab
    operators_changed = pyqtSignal(list)  # List[OperatorConfig]
    step_all_requested = pyqtSignal(int)  # Emit with seed for lock-step execution
    reset_all_requested = pyqtSignal(int)  # Emit with seed for fair reset
    stop_operators_requested = pyqtSignal()
    initialize_operator_requested = pyqtSignal(str, object)  # operator_id, config

    # Forwarded signals from Workers subtab
    worker_changed = pyqtSignal(str)  # worker_id
    train_requested = pyqtSignal()
    evaluate_requested = pyqtSignal()
    resume_requested = pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Subtab widget
        self._subtabs = QtWidgets.QTabWidget(self)
        self._subtabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)

        # Create subtabs
        self._operators_tab = OperatorsSubTab(self)
        self._workers_tab = WorkersSubTab(self)

        # Add subtabs
        self._subtabs.addTab(self._operators_tab, "Operators")
        self._subtabs.addTab(self._workers_tab, "Workers")

        layout.addWidget(self._subtabs)

    def _connect_signals(self) -> None:
        """Connect signals from subtabs."""
        # Operators subtab - scientific execution controls
        self._operators_tab.operators_changed.connect(self.operators_changed)
        self._operators_tab.step_all_requested.connect(self.step_all_requested)
        self._operators_tab.reset_all_requested.connect(self.reset_all_requested)
        self._operators_tab.stop_operators_requested.connect(self.stop_operators_requested)
        self._operators_tab.initialize_operator_requested.connect(self.initialize_operator_requested)

        # Workers subtab
        self._workers_tab.worker_changed.connect(self.worker_changed)
        self._workers_tab.train_requested.connect(self.train_requested)
        self._workers_tab.evaluate_requested.connect(self.evaluate_requested)
        self._workers_tab.resume_requested.connect(self.resume_requested)

    @property
    def operators_subtab(self) -> OperatorsSubTab:
        """Get the Operators subtab."""
        return self._operators_tab

    @property
    def workers_subtab(self) -> WorkersSubTab:
        """Get the Workers subtab."""
        return self._workers_tab

    # Backward compatibility properties for control_panel.py
    @property
    def operator_config_widget(self) -> OperatorConfigWidget:
        """Get the operator configuration widget."""
        return self._operators_tab.operator_config_widget

    @property
    def worker_combo(self) -> QtWidgets.QComboBox:
        """Get the worker combo box."""
        return self._workers_tab.worker_combo

    @property
    def worker_description(self) -> QtWidgets.QLabel:
        """Get the worker description label."""
        return self._workers_tab.worker_description

    @property
    def train_agent_button(self) -> QtWidgets.QPushButton:
        """Get the train agent button."""
        return self._workers_tab.train_agent_button

    @property
    def evaluate_policy_button(self) -> QtWidgets.QPushButton:
        """Get the evaluate policy button."""
        return self._workers_tab.evaluate_policy_button

    @property
    def resume_training_button(self) -> QtWidgets.QPushButton:
        """Get the resume training button."""
        return self._workers_tab.resume_training_button


__all__ = [
    "SingleAgentTab",
    "OperatorsSubTab",
    "WorkersSubTab",
]
