"""Ray RLlib policy evaluation form.

This form provides configuration options for evaluating trained multi-agent policies
using Ray RLlib. Allows users to configure:
- Number of evaluation episodes
- Episode length limits
- Deterministic vs stochastic actions
- Metrics to display
- FastLane visualization settings

Based on Ray RLlib's evaluation API:
https://docs.ray.io/en/latest/rllib/rllib-advanced-api.html#customized-evaluation-during-training
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.pettingzoo_enums import (
    PettingZooEnvId,
    PettingZooFamily,
    get_display_name,
    get_envs_by_family,
)
from gym_gui.ui.forms.factory import get_worker_form_factory
from gym_gui.policy_discovery.ray_policy_metadata import (
    RayRLlibCheckpoint,
    discover_ray_checkpoints as discover_checkpoints,
)

_LOGGER = logging.getLogger(__name__)


class RayEvaluationForm(QtWidgets.QDialog):
    """Dialog for configuring Ray RLlib policy evaluation.

    Provides UI controls for:
    - Checkpoint selection
    - Evaluation parameters (episodes, steps, deterministic)
    - Metrics configuration
    - FastLane visualization settings

    Signals:
        evaluation_requested: Emitted when user accepts with config dict
    """

    evaluation_requested = QtCore.Signal(dict)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        default_game: Optional[Any] = None,
        current_game: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ray RLlib Policy Evaluation")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self._default_game = default_game or current_game
        self._checkpoints: List[RayRLlibCheckpoint] = []
        self._result_config: Optional[Dict[str, Any]] = None

        self._setup_ui()
        self._connect_signals()
        self._refresh_checkpoints()

    def _setup_ui(self) -> None:
        """Create the form UI."""
        layout = QtWidgets.QVBoxLayout(self)

        # Title
        title = QtWidgets.QLabel("Configure Policy Evaluation")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Description
        desc = QtWidgets.QLabel(
            "Evaluate trained policies by running episodes and collecting metrics. "
            "Visualization is streamed to FastLane for real-time display."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: gray; margin-bottom: 10px;")
        layout.addWidget(desc)

        # Form layout
        form_widget = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout(form_widget)
        form_layout.setSpacing(10)

        # === Checkpoint Selection ===
        checkpoint_group = QtWidgets.QGroupBox("Checkpoint Selection")
        checkpoint_layout = QtWidgets.QVBoxLayout(checkpoint_group)

        # Checkpoint combo
        self._checkpoint_combo = QtWidgets.QComboBox()
        self._checkpoint_combo.setMinimumWidth(400)
        checkpoint_layout.addWidget(self._checkpoint_combo)

        # Checkpoint info
        self._checkpoint_info = QtWidgets.QLabel()
        self._checkpoint_info.setStyleSheet("color: gray; font-size: 9pt;")
        self._checkpoint_info.setWordWrap(True)
        checkpoint_layout.addWidget(self._checkpoint_info)

        # Refresh button
        refresh_btn = QtWidgets.QPushButton("Refresh Checkpoints")
        refresh_btn.clicked.connect(self._refresh_checkpoints)
        checkpoint_layout.addWidget(refresh_btn)

        layout.addWidget(checkpoint_group)

        # === Evaluation Parameters ===
        eval_group = QtWidgets.QGroupBox("Evaluation Parameters")
        eval_layout = QtWidgets.QFormLayout(eval_group)

        # Number of episodes
        self._num_episodes_spin = QtWidgets.QSpinBox()
        self._num_episodes_spin.setRange(1, 1000)
        self._num_episodes_spin.setValue(10)
        self._num_episodes_spin.setToolTip(
            "Number of evaluation episodes to run. "
            "More episodes = more accurate metrics but takes longer."
        )
        eval_layout.addRow("Number of Episodes:", self._num_episodes_spin)

        # Max steps per episode
        self._max_steps_spin = QtWidgets.QSpinBox()
        self._max_steps_spin.setRange(100, 100000)
        self._max_steps_spin.setValue(1000)
        self._max_steps_spin.setSingleStep(100)
        self._max_steps_spin.setToolTip(
            "Maximum steps per episode. Episode terminates if this limit is reached."
        )
        eval_layout.addRow("Max Steps per Episode:", self._max_steps_spin)

        # Deterministic actions
        self._deterministic_check = QtWidgets.QCheckBox("Use deterministic actions")
        self._deterministic_check.setChecked(True)
        self._deterministic_check.setToolTip(
            "If checked, use greedy/deterministic actions (no exploration). "
            "Unchecked allows stochastic sampling from policy distribution."
        )
        eval_layout.addRow("Action Mode:", self._deterministic_check)

        # Random seed
        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        self._seed_spin.setToolTip(
            "Random seed for reproducibility. Same seed = same episode outcomes."
        )
        eval_layout.addRow("Random Seed:", self._seed_spin)

        layout.addWidget(eval_group)

        # === Visualization Settings ===
        viz_group = QtWidgets.QGroupBox("Visualization Settings")
        viz_layout = QtWidgets.QFormLayout(viz_group)

        # FastLane enabled
        self._fastlane_check = QtWidgets.QCheckBox("Stream to FastLane")
        self._fastlane_check.setChecked(True)
        self._fastlane_check.setToolTip(
            "Stream rendered frames to FastLane for real-time visualization."
        )
        viz_layout.addRow("Live Visualization:", self._fastlane_check)

        # Render FPS
        self._render_fps_spin = QtWidgets.QSpinBox()
        self._render_fps_spin.setRange(10, 60)
        self._render_fps_spin.setValue(30)
        self._render_fps_spin.setToolTip("Target frames per second for rendering.")
        viz_layout.addRow("Render FPS:", self._render_fps_spin)

        layout.addWidget(viz_group)

        # === Metrics Display ===
        metrics_group = QtWidgets.QGroupBox("Metrics to Collect")
        metrics_layout = QtWidgets.QVBoxLayout(metrics_group)

        self._metrics_info = QtWidgets.QLabel(
            "The following metrics will be collected during evaluation:\n"
            "  - Episode Return (mean, min, max)\n"
            "  - Episode Length (mean)\n"
            "  - Per-Agent Rewards\n"
            "  - Per-Policy Rewards (for multi-policy setups)\n"
            "  - Episode Duration (seconds)"
        )
        self._metrics_info.setStyleSheet("font-size: 9pt;")
        metrics_layout.addWidget(self._metrics_info)

        layout.addWidget(metrics_group)

        # Spacer
        layout.addStretch()

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self._start_btn = QtWidgets.QPushButton("Start Evaluation")
        self._start_btn.setDefault(True)
        self._start_btn.clicked.connect(self._on_start_clicked)
        button_layout.addWidget(self._start_btn)

        layout.addLayout(button_layout)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._checkpoint_combo.currentIndexChanged.connect(
            self._on_checkpoint_changed
        )
        self._fastlane_check.toggled.connect(self._on_fastlane_toggled)

    def _refresh_checkpoints(self) -> None:
        """Refresh the list of available checkpoints."""
        # Temporarily block signals to prevent multiple _on_checkpoint_changed calls
        self._checkpoint_combo.blockSignals(True)
        self._checkpoint_combo.clear()
        self._checkpoints = []

        try:
            self._checkpoints = discover_checkpoints()

            if not self._checkpoints:
                self._checkpoint_combo.addItem("No checkpoints found")
                self._start_btn.setEnabled(False)
                self._checkpoint_combo.blockSignals(False)
                return

            for cp in self._checkpoints:
                # Format: [Algorithm] env_id - run_id (paradigm)
                label = f"[{cp.algorithm}] {cp.env_id} - {cp.run_id} ({cp.paradigm})"
                self._checkpoint_combo.addItem(label)

            # Enable button after all checkpoints are loaded
            self._start_btn.setEnabled(True)

        except Exception as e:
            _LOGGER.error("Failed to discover checkpoints: %s", e)
            self._checkpoint_combo.addItem(f"Error: {e}")
            self._start_btn.setEnabled(False)

        finally:
            # Always unblock signals and update display
            self._checkpoint_combo.blockSignals(False)
            # Manually trigger checkpoint changed to update info
            if self._checkpoints:
                self._on_checkpoint_changed(0)

    def _on_checkpoint_changed(self, index: int) -> None:
        """Update checkpoint info display."""
        if index < 0 or index >= len(self._checkpoints):
            self._checkpoint_info.setText("")
            return

        cp = self._checkpoints[index]
        info_lines = [
            f"Environment: {cp.env_family}/{cp.env_id}",
            f"Algorithm: {cp.algorithm}",
            f"Paradigm: {cp.paradigm}",
            f"Policies: {', '.join(cp.policy_ids) if cp.policy_ids else 'N/A'}",
            f"Training Steps: {cp.training_iteration or 'N/A'}",
            f"Path: {cp.checkpoint_path}",
        ]
        self._checkpoint_info.setText("\n".join(info_lines))

    def _on_fastlane_toggled(self, checked: bool) -> None:
        """Enable/disable visualization settings based on FastLane toggle."""
        self._render_fps_spin.setEnabled(checked)

    def _on_start_clicked(self) -> None:
        """Handle start evaluation button click."""
        if not self._checkpoints:
            QtWidgets.QMessageBox.warning(
                self,
                "No Checkpoint",
                "Please select a checkpoint to evaluate.",
            )
            return

        self._result_config = self._build_config()
        self.accept()

    def _build_config(self) -> Dict[str, Any]:
        """Build the evaluation configuration dictionary."""
        index = self._checkpoint_combo.currentIndex()
        if index < 0 or index >= len(self._checkpoints):
            return {}

        checkpoint = self._checkpoints[index]

        # Generate unique run ID for this evaluation session
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        eval_run_id = f"eval_{checkpoint.run_id}_{timestamp}"

        # Determine policy ID to use
        policy_id = "shared"
        if checkpoint.paradigm == "parameter_sharing":
            policy_id = "shared"
        elif checkpoint.policy_ids:
            policy_id = checkpoint.policy_ids[0]

        return {
            "mode": "evaluate",
            "run_name": f"Eval-{checkpoint.env_id}-{eval_run_id}",

            # Checkpoint info
            "checkpoint_path": str(checkpoint.checkpoint_path),
            "policy_id": policy_id,

            # Evaluation parameters
            "num_episodes": self._num_episodes_spin.value(),
            "max_steps_per_episode": self._max_steps_spin.value(),
            "deterministic": self._deterministic_check.isChecked(),
            "seed": self._seed_spin.value(),

            # Visualization
            "fastlane_enabled": self._fastlane_check.isChecked(),
            "render_fps": self._render_fps_spin.value(),

            # Metadata for UI and worker
            "metadata": {
                "ui": {
                    "worker_id": "ray_worker",
                    "mode": "evaluate",
                },
                "worker": {
                    "checkpoint_path": str(checkpoint.checkpoint_path),
                    "policy_id": policy_id,
                    "deterministic": self._deterministic_check.isChecked(),
                },
                "evaluation": {
                    "run_id": eval_run_id,
                    "num_episodes": self._num_episodes_spin.value(),
                    "max_steps_per_episode": self._max_steps_spin.value(),
                    "seed": self._seed_spin.value(),
                    "fastlane_enabled": self._fastlane_check.isChecked(),
                    "render_fps": self._render_fps_spin.value(),
                },
                "ray_checkpoint": {
                    "run_id": checkpoint.run_id,
                    "env_id": checkpoint.env_id,
                    "env_family": checkpoint.env_family,
                    "algorithm": checkpoint.algorithm,
                    "paradigm": checkpoint.paradigm,
                    "policy_ids": checkpoint.policy_ids,
                    "training_iteration": checkpoint.training_iteration,
                },
            },
        }

    def get_config(self) -> Optional[Dict[str, Any]]:
        """Return the evaluation configuration if dialog was accepted."""
        return self._result_config

    def exec(self) -> int:
        """Execute the dialog."""
        return super().exec()


# Register evaluation form with factory at module load
_factory = get_worker_form_factory()
if not _factory.has_evaluation_form("ray_worker"):
    # Check if factory supports evaluation forms
    if hasattr(_factory, "register_evaluation_form"):
        _factory.register_evaluation_form(
            "ray_worker",
            lambda parent=None, **kwargs: RayEvaluationForm(parent=parent, **kwargs),
        )
    else:
        _LOGGER.debug(
            "WorkerFormFactory does not support evaluation forms yet. "
            "RayEvaluationForm can be used directly."
        )


__all__ = ["RayEvaluationForm"]
