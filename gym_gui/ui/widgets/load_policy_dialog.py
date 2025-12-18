"""Load Policy Dialog for selecting trained checkpoints.

This dialog allows users to:
1. Browse discovered Ray RLlib and CleanRL checkpoints
2. Filter by environment, algorithm, or worker type
3. View checkpoint details (training metadata, policies)
4. Select a checkpoint for evaluation/inference
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PyQt6.QtCore import pyqtSignal
from qtpy import QtCore, QtWidgets

from gym_gui.policy_discovery.ray_policy_metadata import (
    RayRLlibCheckpoint,
    discover_ray_checkpoints,
)
from gym_gui.policy_discovery.cleanrl_policy_metadata import (
    CleanRlCheckpoint,
    discover_policies as discover_cleanrl_policies,
)

_LOGGER = logging.getLogger(__name__)

# Type alias for checkpoints
CheckpointType = Union[RayRLlibCheckpoint, CleanRlCheckpoint]


class LoadPolicyDialog(QtWidgets.QDialog):
    """Dialog for loading trained policy checkpoints.

    Signals:
        policy_selected: Emitted when a policy is selected and confirmed
    """

    policy_selected = pyqtSignal(object)  # Emits checkpoint object

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        filter_env: Optional[str] = None,
        filter_worker: Optional[str] = None,
    ) -> None:
        """Initialize the dialog.

        Args:
            parent: Parent widget
            filter_env: Optional environment ID to filter checkpoints
            filter_worker: Optional worker type to filter ("ray" or "cleanrl")
        """
        super().__init__(parent)
        self._filter_env = filter_env
        self._filter_worker = filter_worker
        self._checkpoints: List[CheckpointType] = []
        self._selected_checkpoint: Optional[CheckpointType] = None

        self.setWindowTitle("Load Trained Policy")
        self.setMinimumSize(700, 500)
        self.resize(800, 600)

        self._build_ui()
        self._connect_signals()
        self._refresh_checkpoints()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Filter section
        filter_group = QtWidgets.QGroupBox("Filters", self)
        filter_layout = QtWidgets.QHBoxLayout(filter_group)

        # Worker type filter
        filter_layout.addWidget(QtWidgets.QLabel("Type:", filter_group))
        self._type_filter = QtWidgets.QComboBox(filter_group)
        self._type_filter.addItem("All", "all")
        self._type_filter.addItem("Ray RLlib", "ray")
        self._type_filter.addItem("CleanRL", "cleanrl")
        if self._filter_worker:
            index = self._type_filter.findData(self._filter_worker)
            if index >= 0:
                self._type_filter.setCurrentIndex(index)
        filter_layout.addWidget(self._type_filter)

        # Environment filter
        filter_layout.addWidget(QtWidgets.QLabel("Environment:", filter_group))
        self._env_filter = QtWidgets.QComboBox(filter_group)
        self._env_filter.addItem("All Environments", "")
        filter_layout.addWidget(self._env_filter)

        # Refresh button
        self._refresh_btn = QtWidgets.QPushButton("Refresh", filter_group)
        self._refresh_btn.setMaximumWidth(80)
        filter_layout.addWidget(self._refresh_btn)

        filter_layout.addStretch(1)
        layout.addWidget(filter_group)

        # Splitter for list and details
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)

        # Checkpoint list
        list_widget = QtWidgets.QWidget(splitter)
        list_layout = QtWidgets.QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)

        list_label = QtWidgets.QLabel("Available Checkpoints:", list_widget)
        list_label.setStyleSheet("font-weight: bold;")
        list_layout.addWidget(list_label)

        self._checkpoint_list = QtWidgets.QListWidget(list_widget)
        self._checkpoint_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        list_layout.addWidget(self._checkpoint_list)

        splitter.addWidget(list_widget)

        # Details panel
        details_widget = QtWidgets.QWidget(splitter)
        details_layout = QtWidgets.QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)

        details_label = QtWidgets.QLabel("Checkpoint Details:", details_widget)
        details_label.setStyleSheet("font-weight: bold;")
        details_layout.addWidget(details_label)

        self._details_text = QtWidgets.QTextEdit(details_widget)
        self._details_text.setReadOnly(True)
        self._details_text.setStyleSheet(
            "background-color: #f8f8f8; font-family: monospace; font-size: 11px;"
        )
        details_layout.addWidget(self._details_text)

        splitter.addWidget(details_widget)
        splitter.setSizes([350, 350])

        layout.addWidget(splitter, 1)

        # Status bar
        self._status_label = QtWidgets.QLabel("", self)
        self._status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._status_label)

        # Dialog buttons
        button_layout = QtWidgets.QHBoxLayout()

        self._load_btn = QtWidgets.QPushButton("Load Policy", self)
        self._load_btn.setEnabled(False)
        self._load_btn.setDefault(True)
        button_layout.addWidget(self._load_btn)

        cancel_btn = QtWidgets.QPushButton("Cancel", self)
        button_layout.addWidget(cancel_btn)

        button_layout.addStretch(1)

        layout.addLayout(button_layout)

        # Connect cancel button
        cancel_btn.clicked.connect(self.reject)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._type_filter.currentIndexChanged.connect(self._apply_filters)
        self._env_filter.currentIndexChanged.connect(self._apply_filters)
        self._refresh_btn.clicked.connect(self._refresh_checkpoints)
        self._checkpoint_list.currentRowChanged.connect(self._on_selection_changed)
        self._checkpoint_list.itemDoubleClicked.connect(self._on_load)
        self._load_btn.clicked.connect(self._on_load)

    def _refresh_checkpoints(self) -> None:
        """Refresh the list of available checkpoints."""
        self._checkpoints = []
        env_ids = set()

        # Discover Ray checkpoints
        ray_checkpoints = discover_ray_checkpoints()
        for ckpt in ray_checkpoints:
            self._checkpoints.append(ckpt)
            env_ids.add(ckpt.env_id)

        # Discover CleanRL checkpoints
        cleanrl_checkpoints = discover_cleanrl_policies()
        for ckpt in cleanrl_checkpoints:
            self._checkpoints.append(ckpt)
            if ckpt.env_id:
                env_ids.add(ckpt.env_id)

        # Update environment filter
        current_env = self._env_filter.currentData()
        self._env_filter.clear()
        self._env_filter.addItem("All Environments", "")
        for env_id in sorted(env_ids):
            self._env_filter.addItem(env_id, env_id)

        # Restore selection
        if current_env:
            index = self._env_filter.findData(current_env)
            if index >= 0:
                self._env_filter.setCurrentIndex(index)

        # Apply pre-set env filter if provided
        if self._filter_env:
            index = self._env_filter.findData(self._filter_env)
            if index >= 0:
                self._env_filter.setCurrentIndex(index)
                self._filter_env = None  # Only apply once

        self._apply_filters()

        _LOGGER.info(
            "Refreshed checkpoints: %d Ray, %d CleanRL",
            len(ray_checkpoints),
            len(cleanrl_checkpoints),
        )

    def _apply_filters(self) -> None:
        """Apply filters and update the list."""
        self._checkpoint_list.clear()

        type_filter = self._type_filter.currentData()
        env_filter = self._env_filter.currentData()

        filtered = []
        for ckpt in self._checkpoints:
            # Type filter
            if type_filter != "all":
                if type_filter == "ray" and not isinstance(ckpt, RayRLlibCheckpoint):
                    continue
                if type_filter == "cleanrl" and not isinstance(ckpt, CleanRlCheckpoint):
                    continue

            # Environment filter
            if env_filter:
                if isinstance(ckpt, RayRLlibCheckpoint) and ckpt.env_id != env_filter:
                    continue
                if isinstance(ckpt, CleanRlCheckpoint) and ckpt.env_id != env_filter:
                    continue

            filtered.append(ckpt)

        # Populate list
        for ckpt in filtered:
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.UserRole, ckpt)

            if isinstance(ckpt, RayRLlibCheckpoint):
                text = f"[Ray] {ckpt.run_id[:12]} - {ckpt.env_id} ({ckpt.algorithm})"
                if ckpt.paradigm:
                    text += f" [{ckpt.paradigm}]"
            else:
                # CleanRL
                env_name = ckpt.env_id or "unknown"
                algo_name = ckpt.algo or "unknown"
                text = f"[CleanRL] {ckpt.run_id[:12]} - {env_name} ({algo_name})"

            item.setText(text)
            self._checkpoint_list.addItem(item)

        # Update status
        self._status_label.setText(
            f"{len(filtered)} checkpoint(s) found "
            f"(total: {len(self._checkpoints)})"
        )

        # Clear selection
        self._selected_checkpoint = None
        self._load_btn.setEnabled(False)
        self._details_text.clear()

    def _on_selection_changed(self, row: int) -> None:
        """Handle selection change in the checkpoint list."""
        if row < 0:
            self._selected_checkpoint = None
            self._load_btn.setEnabled(False)
            self._details_text.clear()
            return

        item = self._checkpoint_list.item(row)
        if item is None:
            return

        self._selected_checkpoint = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self._load_btn.setEnabled(True)
        self._update_details()

    def _update_details(self) -> None:
        """Update the details panel with selected checkpoint info."""
        if self._selected_checkpoint is None:
            self._details_text.clear()
            return

        ckpt = self._selected_checkpoint
        lines = []

        if isinstance(ckpt, RayRLlibCheckpoint):
            lines.append("=== Ray RLlib Checkpoint ===\n")
            lines.append(f"Run ID:          {ckpt.run_id}")
            lines.append(f"Environment:     {ckpt.env_id}")
            lines.append(f"Family:          {ckpt.env_family}")
            lines.append(f"Algorithm:       {ckpt.algorithm}")
            lines.append(f"Paradigm:        {ckpt.paradigm}")
            lines.append(f"Policy IDs:      {', '.join(ckpt.policy_ids)}")
            lines.append(f"Ray Version:     {ckpt.ray_version}")
            lines.append(f"Checkpoint Ver:  {ckpt.checkpoint_version}")
            lines.append(f"\nPath: {ckpt.checkpoint_path}")
            if ckpt.config_path:
                lines.append(f"Config: {ckpt.config_path}")
        else:
            # CleanRL
            lines.append("=== CleanRL Checkpoint ===\n")
            lines.append(f"Run ID:          {ckpt.run_id}")
            lines.append(f"CleanRL Run:     {ckpt.cleanrl_run_name or 'N/A'}")
            lines.append(f"Environment:     {ckpt.env_id or 'N/A'}")
            lines.append(f"Algorithm:       {ckpt.algo or 'N/A'}")
            lines.append(f"Seed:            {ckpt.seed or 'N/A'}")
            lines.append(f"Num Envs:        {ckpt.num_envs or 'N/A'}")
            lines.append(f"FastLane Only:   {ckpt.fastlane_only}")
            lines.append(f"\nPath: {ckpt.policy_path}")
            if ckpt.config_path:
                lines.append(f"Config: {ckpt.config_path}")

        self._details_text.setText("\n".join(lines))

    def _on_load(self) -> None:
        """Handle load button click or double-click."""
        if self._selected_checkpoint is not None:
            self.policy_selected.emit(self._selected_checkpoint)
            self.accept()

    def get_selected_checkpoint(self) -> Optional[CheckpointType]:
        """Get the selected checkpoint.

        Returns:
            Selected checkpoint or None
        """
        return self._selected_checkpoint


class QuickLoadPolicyWidget(QtWidgets.QWidget):
    """Compact widget for quickly loading a policy.

    Shows a dropdown with recent checkpoints and a browse button.

    Signals:
        policy_selected: Emitted when a policy is selected
    """

    policy_selected = pyqtSignal(object)  # Emits checkpoint object

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        max_recent: int = 5,
    ) -> None:
        super().__init__(parent)
        self._max_recent = max_recent
        self._checkpoints: List[CheckpointType] = []

        self._build_ui()
        self._connect_signals()
        self.refresh()

    def _build_ui(self) -> None:
        """Build the widget UI."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Policy dropdown
        self._policy_combo = QtWidgets.QComboBox(self)
        self._policy_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self._policy_combo.setPlaceholderText("Select a trained policy...")
        layout.addWidget(self._policy_combo)

        # Browse button
        self._browse_btn = QtWidgets.QPushButton("Browse...", self)
        self._browse_btn.setMaximumWidth(80)
        layout.addWidget(self._browse_btn)

    def _connect_signals(self) -> None:
        """Connect signals."""
        self._policy_combo.currentIndexChanged.connect(self._on_combo_changed)
        self._browse_btn.clicked.connect(self._on_browse)

    def refresh(self) -> None:
        """Refresh the list of checkpoints."""
        self._checkpoints = []

        # Get recent Ray checkpoints
        ray_ckpts = discover_ray_checkpoints()[:self._max_recent]
        self._checkpoints.extend(ray_ckpts)

        # Get recent CleanRL checkpoints
        cleanrl_ckpts = discover_cleanrl_policies()[:self._max_recent]
        self._checkpoints.extend(cleanrl_ckpts)

        # Update combo
        current_data = self._policy_combo.currentData()
        self._policy_combo.clear()
        self._policy_combo.addItem("(No policy - random actions)", None)

        for ckpt in self._checkpoints:
            if isinstance(ckpt, RayRLlibCheckpoint):
                text = f"[Ray] {ckpt.run_id[:8]} - {ckpt.env_id}"
            else:
                text = f"[CleanRL] {ckpt.run_id[:8]} - {ckpt.env_id or 'unknown'}"
            self._policy_combo.addItem(text, ckpt)

        # Restore selection
        if current_data:
            for i in range(self._policy_combo.count()):
                if self._policy_combo.itemData(i) == current_data:
                    self._policy_combo.setCurrentIndex(i)
                    break

    def _on_combo_changed(self, index: int) -> None:
        """Handle combo selection change."""
        ckpt = self._policy_combo.currentData()
        if ckpt is not None:
            self.policy_selected.emit(ckpt)

    def _on_browse(self) -> None:
        """Open the full browse dialog."""
        dialog = LoadPolicyDialog(self)
        dialog.policy_selected.connect(self._on_dialog_selected)
        dialog.exec()

    def _on_dialog_selected(self, ckpt: CheckpointType) -> None:
        """Handle selection from browse dialog."""
        # Add to combo if not present
        found = False
        for i in range(self._policy_combo.count()):
            if self._policy_combo.itemData(i) == ckpt:
                self._policy_combo.setCurrentIndex(i)
                found = True
                break

        if not found:
            if isinstance(ckpt, RayRLlibCheckpoint):
                text = f"[Ray] {ckpt.run_id[:8]} - {ckpt.env_id}"
            else:
                text = f"[CleanRL] {ckpt.run_id[:8]} - {ckpt.env_id or 'unknown'}"
            self._policy_combo.addItem(text, ckpt)
            self._policy_combo.setCurrentIndex(self._policy_combo.count() - 1)

        self.policy_selected.emit(ckpt)

    def get_selected_checkpoint(self) -> Optional[CheckpointType]:
        """Get currently selected checkpoint."""
        return self._policy_combo.currentData()


__all__ = [
    "LoadPolicyDialog",
    "QuickLoadPolicyWidget",
]
