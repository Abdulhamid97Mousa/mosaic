"""Dialog for loading trained Ray RLlib policies."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtWidgets

from gym_gui.config.paths import VAR_TRAINER_DIR
from gym_gui.ui.forms import get_worker_form_factory

_LOGGER = logging.getLogger(__name__)


class RayPolicyForm(QtWidgets.QDialog):
    """Dialog for loading trained Ray RLlib checkpoints.

    Provides UI for:
    - Browsing for checkpoint directories
    - Selecting which policy to use
    - Configuring evaluation settings
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        default_game: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load Ray RLlib Policy")
        self.resize(600, 400)

        self._default_game = default_game
        self._checkpoint_path: Optional[Path] = None
        self._result_config: Optional[Dict[str, Any]] = None

        self._build_ui()
        self._scan_checkpoints()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Instructions
        intro = QtWidgets.QLabel(self)
        intro.setWordWrap(True)
        intro.setText(
            "Select a Ray RLlib checkpoint to load for evaluation. "
            "Checkpoints are stored in var/trainer/runs/{run_id}/checkpoints/."
        )
        layout.addWidget(intro)

        # Checkpoint table
        self._table = QtWidgets.QTableWidget(self)
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Run ID", "Environment", "Paradigm", "Path"])
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table, 2)

        # Browse button
        browse_layout = QtWidgets.QHBoxLayout()
        self._path_label = QtWidgets.QLabel("No checkpoint selected", self)
        self._path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        browse_layout.addWidget(self._path_label, 1)
        browse_btn = QtWidgets.QPushButton("Browse...", self)
        browse_btn.clicked.connect(self._on_browse)
        browse_layout.addWidget(browse_btn)
        layout.addLayout(browse_layout)

        # Policy selection
        policy_group = QtWidgets.QGroupBox("Policy Settings", self)
        policy_layout = QtWidgets.QFormLayout(policy_group)

        self._policy_combo = QtWidgets.QComboBox(self)
        self._policy_combo.addItem("shared (parameter sharing)", "shared")
        self._policy_combo.addItem("main (self-play)", "main")
        policy_layout.addRow("Policy ID:", self._policy_combo)

        self._deterministic_checkbox = QtWidgets.QCheckBox(
            "Deterministic actions (no exploration)", self
        )
        self._deterministic_checkbox.setChecked(True)
        policy_layout.addRow("", self._deterministic_checkbox)

        layout.addWidget(policy_group)

        # Dialog buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _scan_checkpoints(self) -> None:
        """Scan var/trainer/runs for Ray checkpoints."""
        runs_dir = VAR_TRAINER_DIR / "runs"
        if not runs_dir.exists():
            return

        self._table.setRowCount(0)

        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            # Look for Ray checkpoint indicators
            checkpoint_dir = run_dir / "checkpoints"
            analytics_file = run_dir / "analytics.json"

            # Check if this looks like a Ray run
            if not checkpoint_dir.exists():
                continue

            # Check for Ray checkpoint structure
            ray_checkpoints = list(checkpoint_dir.glob("checkpoint_*"))
            if not ray_checkpoints:
                continue

            # Read analytics to get metadata
            env_id = "unknown"
            paradigm = "unknown"
            if analytics_file.exists():
                try:
                    import json
                    data = json.loads(analytics_file.read_text())
                    ray_meta = data.get("ray_metadata", {})
                    env_id = ray_meta.get("env_id", "unknown")
                    paradigm = ray_meta.get("paradigm", "unknown")
                except Exception:
                    pass

            # Add to table
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(run_dir.name))
            self._table.setItem(row, 1, QtWidgets.QTableWidgetItem(env_id))
            self._table.setItem(row, 2, QtWidgets.QTableWidgetItem(paradigm))
            self._table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(ray_checkpoints[-1])))

        self._table.resizeColumnsToContents()

    def _on_selection_changed(self) -> None:
        """Handle table selection change."""
        selected = self._table.selectedItems()
        if not selected:
            self._checkpoint_path = None
            self._path_label.setText("No checkpoint selected")
            return

        row = selected[0].row()
        path_item = self._table.item(row, 3)
        if path_item:
            self._checkpoint_path = Path(path_item.text())
            self._path_label.setText(str(self._checkpoint_path))

    def _on_browse(self) -> None:
        """Handle browse button click."""
        start_dir = str(VAR_TRAINER_DIR / "runs")
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Ray RLlib Checkpoint",
            start_dir,
        )
        if path:
            self._checkpoint_path = Path(path)
            self._path_label.setText(str(self._checkpoint_path))

    def _on_accept(self) -> None:
        """Handle OK button click."""
        if self._checkpoint_path is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Checkpoint Selected",
                "Please select a checkpoint to load.",
            )
            return

        self._result_config = self._build_config()
        self.accept()

    def _build_config(self) -> Dict[str, Any]:
        """Build the evaluation configuration."""
        policy_id = self._policy_combo.currentData() or "shared"
        deterministic = self._deterministic_checkbox.isChecked()

        return {
            "mode": "evaluate",
            "checkpoint_path": str(self._checkpoint_path),
            "policy_id": policy_id,
            "deterministic": deterministic,
            "metadata": {
                "ui": {
                    "worker_id": "ray_worker",
                    "mode": "evaluate",
                },
                "worker": {
                    "checkpoint_path": str(self._checkpoint_path),
                    "policy_id": policy_id,
                    "deterministic": deterministic,
                },
            },
        }

    def get_config(self) -> Optional[Dict[str, Any]]:
        """Return the configuration if dialog was accepted."""
        return self._result_config


# Register policy form with factory at module load
_factory = get_worker_form_factory()
if not _factory.has_policy_form("ray_worker"):
    _factory.register_policy_form(
        "ray_worker",
        lambda parent=None, **kwargs: RayPolicyForm(parent=parent, **kwargs),
    )


__all__ = ["RayPolicyForm"]
