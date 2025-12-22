"""Multi-operator configuration widget for side-by-side agent comparison.

This module provides UI widgets for configuring N operators (LLM or RL workers)
that can run in parallel, each with its own render container.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]
from qtpy import QtCore, QtWidgets

from gym_gui.services.operator import OperatorConfig
from gym_gui.ui.worker_catalog.catalog import get_worker_catalog, WorkerDefinition
from gym_gui.constants.constants_operator import (
    BARLOG_SUPPORTED_ENVS,
    BARLOG_DEFAULT_TASK,
)

_LOGGER = logging.getLogger(__name__)

# Maximum number of operators allowed
MAX_OPERATORS = 8

# Default environments for RL workers (Gymnasium)
RL_SUPPORTED_ENVS = (
    "FrozenLake-v1",
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
    "LunarLander-v2",
    "BipedalWalker-v3",
    "Taxi-v3",
    "CliffWalking-v0",
)


def _get_llm_workers() -> List[WorkerDefinition]:
    """Get LLM workers from catalog (supports_training=False)."""
    return [w for w in get_worker_catalog() if not w.supports_training]


def _get_rl_workers() -> List[WorkerDefinition]:
    """Get RL workers from catalog (supports_training=True)."""
    return [w for w in get_worker_catalog() if w.supports_training]


class OperatorConfigRow(QtWidgets.QWidget):
    """Single row in the operator configuration list.

    Each row represents one operator with:
    - Display name
    - Type selector (LLM / RL)
    - Worker dropdown (filtered by type)
    - Environment dropdown
    - Task dropdown (for LLM only)
    - Settings button
    - Remove button
    """

    config_changed = pyqtSignal(str, object)  # operator_id, new_config
    remove_requested = pyqtSignal(str)  # operator_id

    def __init__(
        self,
        operator_id: str,
        initial_config: Optional[OperatorConfig] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._operator_id = operator_id
        self._updating = False  # Prevent signal loops

        self._build_ui()
        self._connect_signals()

        if initial_config:
            self._load_config(initial_config)

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Operator index label
        self._index_label = QtWidgets.QLabel(f"#{self._operator_id[-1]}", self)
        self._index_label.setFixedWidth(24)
        self._index_label.setStyleSheet("font-weight: bold; color: #666;")
        layout.addWidget(self._index_label)

        # Display name
        self._name_edit = QtWidgets.QLineEdit(self)
        self._name_edit.setPlaceholderText("Operator Name")
        self._name_edit.setFixedWidth(120)
        layout.addWidget(self._name_edit)

        # Type selector (LLM / RL)
        self._type_combo = QtWidgets.QComboBox(self)
        self._type_combo.addItems(["LLM", "RL"])
        self._type_combo.setFixedWidth(60)
        layout.addWidget(self._type_combo)

        # Worker dropdown
        self._worker_combo = QtWidgets.QComboBox(self)
        self._worker_combo.setFixedWidth(140)
        layout.addWidget(self._worker_combo)

        # Environment dropdown
        self._env_combo = QtWidgets.QComboBox(self)
        self._env_combo.setFixedWidth(140)
        layout.addWidget(self._env_combo)

        # Task dropdown (for LLM environments like BabyAI)
        self._task_combo = QtWidgets.QComboBox(self)
        self._task_combo.setFixedWidth(180)
        layout.addWidget(self._task_combo)

        # Settings button
        self._settings_btn = QtWidgets.QPushButton("Settings", self)
        self._settings_btn.setFixedWidth(70)
        self._settings_btn.setToolTip("Configure operator-specific settings")
        layout.addWidget(self._settings_btn)

        # Remove button
        self._remove_btn = QtWidgets.QPushButton("X", self)
        self._remove_btn.setFixedWidth(30)
        self._remove_btn.setToolTip("Remove this operator")
        self._remove_btn.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(self._remove_btn)

        # Initialize dropdowns
        self._update_worker_dropdown()
        self._update_env_dropdown()
        self._update_task_dropdown()

    def _connect_signals(self) -> None:
        self._name_edit.textChanged.connect(self._on_config_changed)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        self._worker_combo.currentIndexChanged.connect(self._on_config_changed)
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)
        self._task_combo.currentIndexChanged.connect(self._on_config_changed)
        self._settings_btn.clicked.connect(self._on_settings_clicked)
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._operator_id))

    def _on_type_changed(self) -> None:
        """Handle operator type change (LLM <-> RL)."""
        if self._updating:
            return
        self._update_worker_dropdown()
        self._update_env_dropdown()
        self._update_task_dropdown()
        self._on_config_changed()

    def _on_env_changed(self) -> None:
        """Handle environment change."""
        if self._updating:
            return
        self._update_task_dropdown()
        self._on_config_changed()

    def _on_config_changed(self) -> None:
        """Emit config_changed signal with current configuration."""
        if self._updating:
            return
        config = self.get_config()
        self.config_changed.emit(self._operator_id, config)

    def _on_settings_clicked(self) -> None:
        """Open settings dialog for this operator."""
        # TODO: Implement operator-specific settings dialog
        QtWidgets.QMessageBox.information(
            self,
            "Operator Settings",
            f"Settings for {self._name_edit.text() or self._operator_id}\n\n"
            f"Type: {self._type_combo.currentText()}\n"
            f"Worker: {self._worker_combo.currentText()}\n"
            f"Environment: {self._env_combo.currentText()}\n"
            f"Task: {self._task_combo.currentText()}"
        )

    def _update_worker_dropdown(self) -> None:
        """Update worker dropdown based on selected type."""
        self._updating = True
        current_worker = self._worker_combo.currentData()
        self._worker_combo.clear()

        operator_type = self._type_combo.currentText().lower()
        if operator_type == "llm":
            workers = _get_llm_workers()
        else:
            workers = _get_rl_workers()

        for worker in workers:
            self._worker_combo.addItem(worker.display_name, worker.worker_id)

        # Restore selection if possible
        if current_worker:
            idx = self._worker_combo.findData(current_worker)
            if idx >= 0:
                self._worker_combo.setCurrentIndex(idx)

        self._updating = False

    def _update_env_dropdown(self) -> None:
        """Update environment dropdown based on selected type."""
        self._updating = True
        current_env = self._env_combo.currentText()
        self._env_combo.clear()

        operator_type = self._type_combo.currentText().lower()
        if operator_type == "llm":
            envs = BARLOG_SUPPORTED_ENVS
        else:
            envs = RL_SUPPORTED_ENVS

        self._env_combo.addItems(envs)

        # Restore selection if possible
        if current_env:
            idx = self._env_combo.findText(current_env)
            if idx >= 0:
                self._env_combo.setCurrentIndex(idx)

        self._updating = False

    def _update_task_dropdown(self) -> None:
        """Update task dropdown based on selected environment."""
        self._updating = True
        current_task = self._task_combo.currentText()
        self._task_combo.clear()

        env_name = self._env_combo.currentText()
        operator_type = self._type_combo.currentText().lower()

        # Define tasks for each environment
        tasks: List[str] = []
        if operator_type == "llm":
            if env_name == "babyai":
                tasks = [
                    "BabyAI-GoToRedBall-v0",
                    "BabyAI-GoToRedBallGrey-v0",
                    "BabyAI-GoToRedBallNoDists-v0",
                    "BabyAI-GoToObj-v0",
                    "BabyAI-GoToLocal-v0",
                    "BabyAI-PutNextLocal-v0",
                ]
            elif env_name == "minigrid":
                tasks = [
                    "MiniGrid-Empty-5x5-v0",
                    "MiniGrid-DoorKey-5x5-v0",
                    "MiniGrid-DoorKey-6x6-v0",
                    "MiniGrid-DoorKey-8x8-v0",
                    "MiniGrid-LavaGapS5-v0",
                    "MiniGrid-LavaGapS7-v0",
                ]
            elif env_name == "minihack":
                tasks = [
                    "MiniHack-Room-5x5-v0",
                    "MiniHack-Room-15x15-v0",
                    "MiniHack-Corridor-R5-v0",
                    "MiniHack-Quest-Easy-v0",
                ]
            elif env_name == "crafter":
                tasks = ["crafter-reward-v1", "crafter-nonreward-v1"]
            elif env_name == "textworld":
                tasks = ["tw-simple", "tw-cooking"]
            else:
                tasks = [BARLOG_DEFAULT_TASK]
        else:
            # RL environments typically have a single variant
            tasks = [env_name]

        self._task_combo.addItems(tasks if tasks else ["default"])

        # Restore selection if possible
        if current_task:
            idx = self._task_combo.findText(current_task)
            if idx >= 0:
                self._task_combo.setCurrentIndex(idx)

        # Show/hide task dropdown based on whether there are options
        self._task_combo.setVisible(len(tasks) > 1 or operator_type == "llm")

        self._updating = False

    def _load_config(self, config: OperatorConfig) -> None:
        """Load configuration into UI elements."""
        self._updating = True

        self._name_edit.setText(config.display_name)

        # Set type
        type_idx = 0 if config.operator_type == "llm" else 1
        self._type_combo.setCurrentIndex(type_idx)

        # Update dropdowns for type
        self._update_worker_dropdown()
        self._update_env_dropdown()

        # Set worker
        worker_idx = self._worker_combo.findData(config.worker_id)
        if worker_idx >= 0:
            self._worker_combo.setCurrentIndex(worker_idx)

        # Set environment
        env_idx = self._env_combo.findText(config.env_name)
        if env_idx >= 0:
            self._env_combo.setCurrentIndex(env_idx)

        # Update and set task
        self._update_task_dropdown()
        task_idx = self._task_combo.findText(config.task)
        if task_idx >= 0:
            self._task_combo.setCurrentIndex(task_idx)

        self._updating = False

    def get_config(self) -> OperatorConfig:
        """Get current configuration from UI elements."""
        operator_type = self._type_combo.currentText().lower()
        worker_id = self._worker_combo.currentData() or ""
        display_name = self._name_edit.text() or f"Operator {self._operator_id[-1]}"
        env_name = self._env_combo.currentText() or "babyai"
        task = self._task_combo.currentText() or BARLOG_DEFAULT_TASK

        return OperatorConfig(
            operator_id=self._operator_id,
            operator_type=operator_type,
            worker_id=worker_id,
            display_name=display_name,
            env_name=env_name,
            task=task,
            settings={},
        )

    @property
    def operator_id(self) -> str:
        return self._operator_id


class OperatorConfigWidget(QtWidgets.QWidget):
    """Widget for managing N operator configurations.

    Provides:
    - List of OperatorConfigRow widgets
    - Add Operator button
    - Operator count limit (default MAX_OPERATORS)
    - Signals for operator list changes
    """

    operators_changed = pyqtSignal(list)  # List[OperatorConfig]

    def __init__(
        self,
        max_operators: int = MAX_OPERATORS,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._max_operators = max_operators
        self._rows: Dict[str, OperatorConfigRow] = {}
        self._next_index = 0

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header
        header = QtWidgets.QLabel("Configure Operators", self)
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        # Scroll area for operator rows
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(100)
        scroll.setMaximumHeight(250)

        self._rows_container = QtWidgets.QWidget(scroll)
        self._rows_layout = QtWidgets.QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch()

        scroll.setWidget(self._rows_container)
        layout.addWidget(scroll)

        # Add button
        self._add_btn = QtWidgets.QPushButton("+ Add Operator", self)
        self._add_btn.clicked.connect(self.add_operator)
        layout.addWidget(self._add_btn)

        # Info label
        self._info_label = QtWidgets.QLabel(f"0 / {self._max_operators} operators", self)
        self._info_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self._info_label)

    def add_operator(self, config: Optional[OperatorConfig] = None) -> Optional[str]:
        """Add a new operator row.

        Args:
            config: Optional initial configuration.

        Returns:
            The operator_id of the new row, or None if max reached.
        """
        if len(self._rows) >= self._max_operators:
            QtWidgets.QMessageBox.warning(
                self,
                "Maximum Operators",
                f"Maximum of {self._max_operators} operators allowed."
            )
            return None

        operator_id = f"operator_{self._next_index}"
        self._next_index += 1

        # Create default config if not provided
        if config is None:
            config = OperatorConfig(
                operator_id=operator_id,
                operator_type="llm",
                worker_id="barlog_worker",
                display_name=f"Operator {len(self._rows) + 1}",
                env_name="babyai",
                task="BabyAI-GoToRedBall-v0",
            )

        # Create row widget
        row = OperatorConfigRow(operator_id, config, self._rows_container)
        row.config_changed.connect(self._on_row_config_changed)
        row.remove_requested.connect(self.remove_operator)

        # Insert before stretch
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._rows[operator_id] = row

        self._update_ui_state()
        self._emit_operators_changed()

        return operator_id

    def remove_operator(self, operator_id: str) -> None:
        """Remove an operator row.

        Args:
            operator_id: ID of the operator to remove.
        """
        if operator_id not in self._rows:
            return

        row = self._rows.pop(operator_id)
        self._rows_layout.removeWidget(row)
        row.deleteLater()

        self._update_ui_state()
        self._emit_operators_changed()

    def get_operators(self) -> List[OperatorConfig]:
        """Get all current operator configurations."""
        return [row.get_config() for row in self._rows.values()]

    def set_operators(self, configs: List[OperatorConfig]) -> None:
        """Set operator configurations, replacing any existing ones."""
        # Clear existing
        for operator_id in list(self._rows.keys()):
            self.remove_operator(operator_id)

        # Add new
        for config in configs:
            self.add_operator(config)

    def clear(self) -> None:
        """Remove all operators."""
        for operator_id in list(self._rows.keys()):
            self.remove_operator(operator_id)

    def _on_row_config_changed(self, operator_id: str, config: OperatorConfig) -> None:
        """Handle config change from a row."""
        self._emit_operators_changed()

    def _emit_operators_changed(self) -> None:
        """Emit the operators_changed signal with current configs."""
        configs = self.get_operators()
        self.operators_changed.emit(configs)
        _LOGGER.debug(f"Operators changed: {len(configs)} operators")

    def _update_ui_state(self) -> None:
        """Update UI state based on current operator count."""
        count = len(self._rows)
        self._info_label.setText(f"{count} / {self._max_operators} operators")
        self._add_btn.setEnabled(count < self._max_operators)

        # Update index labels
        for i, (operator_id, row) in enumerate(self._rows.items()):
            row._index_label.setText(f"#{i + 1}")

    @property
    def operator_count(self) -> int:
        """Get the number of configured operators."""
        return len(self._rows)


__all__ = [
    "OperatorConfigRow",
    "OperatorConfigWidget",
    "MAX_OPERATORS",
]
