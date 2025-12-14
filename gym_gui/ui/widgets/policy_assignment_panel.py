"""Policy Assignment Panel for multi-agent evaluation.

This widget allows users to:
1. See detected agents for the current environment
2. Assign trained policies (checkpoints) to each agent
3. Run evaluation with mixed policies (e.g., Policy A vs Policy B)

Checkpoints are scanned from var/trainer/runs/{run_id}/checkpoints/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import pyqtSignal
from qtpy import QtCore, QtWidgets

from gym_gui.config.paths import VAR_TRAINER_DIR

_LOGGER = logging.getLogger(__name__)


class PolicyAssignmentPanel(QtWidgets.QGroupBox):
    """Panel for assigning policies to agents in multi-agent environments.

    Signals:
        policies_changed: Emitted when policy assignments change
        evaluate_requested: Emitted when user clicks Evaluate button
    """

    policies_changed = pyqtSignal(dict)  # {agent_id: policy_path}
    evaluate_requested = pyqtSignal(dict)  # Full evaluation config

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Policy Assignment", parent)
        self._agent_ids: List[str] = []
        self._policy_combos: Dict[str, QtWidgets.QComboBox] = {}
        self._available_checkpoints: List[Tuple[str, str, str]] = []  # (display, path, run_id)

        self._build_ui()
        self._scan_checkpoints()

    def _build_ui(self) -> None:
        """Build the panel UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(8)

        # Info label
        self._info_label = QtWidgets.QLabel(
            "Assign trained policies to agents for evaluation.\n"
            "Train first to generate checkpoints.",
            self,
        )
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self._info_label)

        # Agents container (will be populated dynamically)
        self._agents_container = QtWidgets.QWidget(self)
        self._agents_layout = QtWidgets.QVBoxLayout(self._agents_container)
        self._agents_layout.setContentsMargins(0, 0, 0, 0)
        self._agents_layout.setSpacing(4)
        layout.addWidget(self._agents_container)

        # Placeholder when no agents
        self._no_agents_label = QtWidgets.QLabel(
            "No agents detected.\nLoad an environment first.",
            self,
        )
        self._no_agents_label.setStyleSheet("color: #888; font-style: italic;")
        self._no_agents_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._agents_layout.addWidget(self._no_agents_label)

        # Refresh button
        refresh_layout = QtWidgets.QHBoxLayout()
        self._refresh_btn = QtWidgets.QPushButton("Refresh Checkpoints", self)
        self._refresh_btn.setToolTip("Scan for new training checkpoints")
        self._refresh_btn.clicked.connect(self._scan_checkpoints)
        refresh_layout.addWidget(self._refresh_btn)
        refresh_layout.addStretch(1)
        layout.addLayout(refresh_layout)

        # Evaluate button
        self._evaluate_btn = QtWidgets.QPushButton("Evaluate Policies", self)
        self._evaluate_btn.setEnabled(False)
        self._evaluate_btn.setToolTip("Run evaluation with assigned policies")
        self._evaluate_btn.clicked.connect(self._on_evaluate)
        layout.addWidget(self._evaluate_btn)

        # Checkpoint count label
        self._checkpoint_count_label = QtWidgets.QLabel("", self)
        self._checkpoint_count_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self._checkpoint_count_label)

    def _scan_checkpoints(self) -> None:
        """Scan var/trainer/runs for available checkpoints."""
        self._available_checkpoints = []

        # Add "Random" option
        self._available_checkpoints.append(("Random (no policy)", "", ""))

        runs_dir = VAR_TRAINER_DIR / "runs"
        if not runs_dir.exists():
            self._update_checkpoint_count()
            self._update_combos()
            return

        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            checkpoint_dir = run_dir / "checkpoints"
            if not checkpoint_dir.exists():
                continue

            # Look for Ray checkpoints
            checkpoints = list(checkpoint_dir.glob("checkpoint_*"))
            if not checkpoints:
                continue

            # Get latest checkpoint
            latest = sorted(checkpoints)[-1]

            # Try to get metadata from analytics.json
            env_id = "unknown"
            paradigm = ""
            analytics_file = run_dir / "analytics.json"
            if analytics_file.exists():
                try:
                    data = json.loads(analytics_file.read_text())
                    ray_meta = data.get("ray_metadata", {})
                    env_id = ray_meta.get("env_id", "unknown")
                    paradigm = ray_meta.get("paradigm", "")
                except Exception:
                    pass

            # Create display name
            display = f"{run_dir.name} ({env_id})"
            if paradigm:
                display += f" [{paradigm}]"

            self._available_checkpoints.append((display, str(latest), run_dir.name))

        self._update_checkpoint_count()
        self._update_combos()
        _LOGGER.debug("Scanned %d checkpoints", len(self._available_checkpoints) - 1)

    def _update_checkpoint_count(self) -> None:
        """Update the checkpoint count label."""
        count = len(self._available_checkpoints) - 1  # Exclude "Random"
        if count == 0:
            self._checkpoint_count_label.setText("No checkpoints found. Train first!")
        else:
            self._checkpoint_count_label.setText(f"{count} checkpoint(s) available")

    def _update_combos(self) -> None:
        """Update all policy combos with current checkpoints."""
        for agent_id, combo in self._policy_combos.items():
            current_path = combo.currentData()
            combo.clear()
            for display, path, run_id in self._available_checkpoints:
                combo.addItem(display, path)
            # Restore selection if still available
            if current_path:
                for i in range(combo.count()):
                    if combo.itemData(i) == current_path:
                        combo.setCurrentIndex(i)
                        break

    def set_agents(self, agent_ids: List[str]) -> None:
        """Set the list of agents for the current environment.

        Args:
            agent_ids: List of agent identifiers
        """
        self._agent_ids = agent_ids
        self._rebuild_agent_rows()

    def _rebuild_agent_rows(self) -> None:
        """Rebuild the agent assignment rows."""
        # Clear existing rows
        while self._agents_layout.count() > 0:
            item = self._agents_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._policy_combos.clear()

        if not self._agent_ids:
            # Show placeholder
            self._no_agents_label = QtWidgets.QLabel(
                "No agents detected.\nLoad an environment first.",
                self,
            )
            self._no_agents_label.setStyleSheet("color: #888; font-style: italic;")
            self._no_agents_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._agents_layout.addWidget(self._no_agents_label)
            self._evaluate_btn.setEnabled(False)
            return

        # Create row for each agent
        for agent_id in self._agent_ids:
            row = self._create_agent_row(agent_id)
            self._agents_layout.addWidget(row)

        self._evaluate_btn.setEnabled(True)
        self._info_label.setText(
            f"{len(self._agent_ids)} agent(s) detected.\n"
            "Assign policies for evaluation."
        )

    def _create_agent_row(self, agent_id: str) -> QtWidgets.QWidget:
        """Create a row widget for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Widget containing agent label and policy combo
        """
        row = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Agent label
        label = QtWidgets.QLabel(f"{agent_id}:", row)
        label.setMinimumWidth(80)
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)

        # Policy combo
        combo = QtWidgets.QComboBox(row)
        combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        for display, path, run_id in self._available_checkpoints:
            combo.addItem(display, path)
        combo.currentIndexChanged.connect(self._on_policy_changed)
        layout.addWidget(combo)

        self._policy_combos[agent_id] = combo
        return row

    def _on_policy_changed(self, index: int) -> None:
        """Handle policy selection change."""
        assignments = self.get_policy_assignments()
        self.policies_changed.emit(assignments)

    def _on_evaluate(self) -> None:
        """Handle evaluate button click."""
        assignments = self.get_policy_assignments()

        config = {
            "mode": "evaluate",
            "agent_policies": assignments,
            "agents": self._agent_ids,
        }

        self.evaluate_requested.emit(config)
        _LOGGER.info("Evaluation requested with policies: %s", assignments)

    def get_policy_assignments(self) -> Dict[str, str]:
        """Get current policy assignments.

        Returns:
            Dict mapping agent_id to checkpoint path (empty string = random)
        """
        assignments = {}
        for agent_id, combo in self._policy_combos.items():
            path = combo.currentData() or ""
            assignments[agent_id] = path
        return assignments

    def set_policy_assignments(self, assignments: Dict[str, str]) -> None:
        """Set policy assignments programmatically.

        Args:
            assignments: Dict mapping agent_id to checkpoint path
        """
        for agent_id, path in assignments.items():
            combo = self._policy_combos.get(agent_id)
            if combo is None:
                continue
            for i in range(combo.count()):
                if combo.itemData(i) == path:
                    combo.setCurrentIndex(i)
                    break


class PolicyAssignmentDialog(QtWidgets.QDialog):
    """Dialog version of the policy assignment panel for modal use."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        agent_ids: Optional[List[str]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Assign Policies to Agents")
        self.setMinimumSize(500, 400)

        self._result_config: Optional[Dict[str, Any]] = None
        self._build_ui()

        if agent_ids:
            self._panel.set_agents(agent_ids)

    def _build_ui(self) -> None:
        """Build dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)

        # Policy assignment panel
        self._panel = PolicyAssignmentPanel(self)
        self._panel.evaluate_requested.connect(self._on_evaluate)
        layout.addWidget(self._panel)

        # Dialog buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_evaluate(self, config: Dict[str, Any]) -> None:
        """Handle evaluate request from panel."""
        self._result_config = config
        self.accept()

    def get_config(self) -> Optional[Dict[str, Any]]:
        """Get the evaluation configuration."""
        return self._result_config

    def set_agents(self, agent_ids: List[str]) -> None:
        """Set agents for the panel."""
        self._panel.set_agents(agent_ids)


__all__ = ["PolicyAssignmentPanel", "PolicyAssignmentDialog"]
