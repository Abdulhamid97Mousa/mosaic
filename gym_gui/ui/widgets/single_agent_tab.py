"""Single-Agent Mode tab with Workers subtab for training backends.

This module provides:
- WorkersSubTab: Select worker backends and control training/evaluation
- SingleAgentTab: Container for Workers subtab (training mode)

Note: Operators have been moved to their own main tab (see operators_tab.py).
Single-Agent Mode now focuses on training via Workers.
"""

from __future__ import annotations

import logging
from typing import Optional, List

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal

from gym_gui.ui.worker_catalog import WorkerDefinition, get_worker_catalog

_LOGGER = logging.getLogger("gym_gui.ui.single_agent_tab")


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
            "<li><b>BALROG LLM Worker</b> - LLM agents using BALROG (OpenAI, Claude, Gemini)</li>"
            "<li><b>CleanRL Worker</b> - Single-file RL implementations (PPO, DQN, SAC)</li>"
            "<li><b>XuanCe Worker</b> - Comprehensive deep RL library with 50+ algorithms</li>"
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
    """Single-Agent Mode tab with Workers subtab for training backends.

    Contains:
    - Workers: Select worker backends and control training/evaluation

    Note: Operators have been moved to their own main tab in the control panel.
    """

    # Signals from Workers subtab
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

        # Subtab widget (keeping subtab structure for future extensibility)
        self._subtabs = QtWidgets.QTabWidget(self)
        self._subtabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)

        # Create Workers subtab
        self._workers_tab = WorkersSubTab(self)

        # Add subtab
        self._subtabs.addTab(self._workers_tab, "Workers")

        layout.addWidget(self._subtabs)

    def _connect_signals(self) -> None:
        """Connect signals from subtabs."""
        # Workers subtab
        self._workers_tab.worker_changed.connect(self.worker_changed)
        self._workers_tab.train_requested.connect(self.train_requested)
        self._workers_tab.evaluate_requested.connect(self.evaluate_requested)
        self._workers_tab.resume_requested.connect(self.resume_requested)

    @property
    def workers_subtab(self) -> WorkersSubTab:
        """Get the Workers subtab."""
        return self._workers_tab

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
    "WorkersSubTab",
]
