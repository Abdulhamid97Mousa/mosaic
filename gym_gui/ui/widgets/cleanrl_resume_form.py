"""Dialog for resuming CleanRL training from a checkpoint.

Resume training uses TensorBoard directly (no SQLite telemetry) and provides
the same algorithm parameters as the Train Agent form.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import sys

from qtpy import QtCore, QtWidgets

from gym_gui.config.paths import VAR_TRAINER_DIR
from gym_gui.core.enums import GameId
from gym_gui.fastlane.worker_helpers import apply_fastlane_environment
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_POLICY_FORM_INFO,
)
from gym_gui.telemetry.semconv import VideoModes, VIDEO_MODE_DESCRIPTORS
from gym_gui.ui.widgets.cleanrl_train_form import _generate_run_id as generate_run_id
from gym_gui.policy_discovery.cleanrl_policy_metadata import (
    CleanRlCheckpoint,
    discover_policies,
    load_metadata_for_policy,
)


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AlgoParamSpec:
    """Specification for an algorithm parameter input widget."""
    key: str
    label: str
    default: Any
    field_type: type
    tooltip: str | None = None


# Algorithm parameters - same as Train Agent form
_ALGO_PARAM_SPECS: dict[str, tuple[_AlgoParamSpec, ...]] = {
    "ppo": (
        _AlgoParamSpec("learning_rate", "Learning Rate", 2.5e-4, float, "Adam optimizer learning rate"),
        _AlgoParamSpec("num_envs", "Parallel Envs", 16, int, "Number of vectorized environments"),
        _AlgoParamSpec("num_steps", "Steps per Update", 2048, int, "Rollout length before each update"),
        _AlgoParamSpec("gamma", "Gamma (Discount)", 0.99, float, "Discount factor for rewards"),
        _AlgoParamSpec("gae_lambda", "GAE Lambda", 0.95, float, "Lambda for generalized advantage estimation"),
        _AlgoParamSpec("clip_coef", "PPO Clip Coef", 0.2, float, "PPO clipping coefficient"),
        _AlgoParamSpec("ent_coef", "Entropy Coef", 0.01, float, "Entropy coefficient for exploration"),
        _AlgoParamSpec("vf_coef", "Value Function Coef", 0.5, float, "Value function loss coefficient"),
    ),
    "ppo_continuous_action": (
        _AlgoParamSpec("learning_rate", "Learning Rate", 3e-4, float, "Adam optimizer learning rate"),
        _AlgoParamSpec("num_envs", "Parallel Envs", 1, int, "Number of vectorized environments"),
        _AlgoParamSpec("num_steps", "Steps per Update", 2048, int, "Rollout length before each update"),
        _AlgoParamSpec("gamma", "Gamma (Discount)", 0.99, float, "Discount factor for rewards"),
        _AlgoParamSpec("gae_lambda", "GAE Lambda", 0.95, float, "Lambda for generalized advantage estimation"),
    ),
}


class CleanRlResumeForm(QtWidgets.QDialog, LogConstantMixin):
    """Dialog for resuming training from a CleanRL checkpoint.

    Uses TensorBoard directly for logging (no SQLite telemetry).
    Provides the same algorithm parameters as Train Agent.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        current_game: Optional[GameId] = None,
    ) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self.setWindowTitle("Resume CleanRL Training")
        self.resize(800, 650)
        self._current_game = current_game
        self._checkpoints = discover_policies()
        self._selected: Optional[CleanRlCheckpoint] = None
        self._result_config: Optional[Dict[str, Any]] = None
        self._algo_param_inputs: Dict[str, QtWidgets.QWidget] = {}
        self._build_ui()
        self._populate_table()
        if self._checkpoints:
            self._table.selectRow(0)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        runs_root = (VAR_TRAINER_DIR / "runs").resolve()
        intro = QtWidgets.QLabel(self)
        intro.setWordWrap(True)
        intro.setTextFormat(QtCore.Qt.TextFormat.RichText)
        if self._checkpoints:
            intro.setText(
                f"Select a checkpoint from <code>{runs_root}</code> to resume training.<br>"
                "<small>Resume training uses TensorBoard directly (no Live-Agent telemetry).</small>"
            )
        else:
            intro.setText(
                f"No CleanRL checkpoints found under <code>{runs_root}</code>."
            )
        layout.addWidget(intro)

        # Checkpoint table
        self._table = QtWidgets.QTableWidget(self)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Env", "Algo", "Seed", "Run", "Path"])
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        layout.addWidget(self._table, 2)

        # Browse button
        browse_layout = QtWidgets.QHBoxLayout()
        self._path_label = QtWidgets.QLabel("No checkpoint selected", self)
        self._path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        browse_layout.addWidget(self._path_label, 1)
        browse_btn = QtWidgets.QPushButton("Browse...", self)
        browse_btn.clicked.connect(self._on_browse)
        browse_layout.addWidget(browse_btn)
        layout.addLayout(browse_layout)

        # Create horizontal layout for two columns
        columns_layout = QtWidgets.QHBoxLayout()

        # Left column: Training configuration
        left_column = QtWidgets.QVBoxLayout()

        train_group = QtWidgets.QGroupBox("Resume Training Configuration", self)
        train_layout = QtWidgets.QGridLayout(train_group)

        train_layout.addWidget(QtWidgets.QLabel("Additional Timesteps:", train_group), 0, 0)
        self._timesteps_spin = QtWidgets.QSpinBox(train_group)
        self._timesteps_spin.setRange(1000, 100_000_000)
        self._timesteps_spin.setValue(100_000)
        self._timesteps_spin.setSingleStep(10000)
        self._timesteps_spin.setToolTip("Number of additional timesteps to train")
        train_layout.addWidget(self._timesteps_spin, 0, 1)

        train_layout.addWidget(QtWidgets.QLabel("Seed:", train_group), 1, 0)
        self._seed_spin = QtWidgets.QSpinBox(train_group)
        self._seed_spin.setRange(0, 2_147_483_647)
        self._seed_spin.setValue(1)
        self._seed_spin.setToolTip("Random seed (0 = use checkpoint's seed)")
        train_layout.addWidget(self._seed_spin, 1, 1)

        # Save Model checkbox
        self._save_model_checkbox = QtWidgets.QCheckBox("Save model after training", train_group)
        self._save_model_checkbox.setChecked(True)
        self._save_model_checkbox.setToolTip(
            "Save the trained model checkpoint when training completes.\n"
            "The model will be saved in the run directory."
        )
        train_layout.addWidget(self._save_model_checkbox, 2, 0, 1, 2)

        left_column.addWidget(train_group)

        # Algorithm Parameters group (dynamic based on algo)
        self._algo_param_group = QtWidgets.QGroupBox("Algorithm Parameters", self)
        self._algo_param_layout = QtWidgets.QGridLayout(self._algo_param_group)
        self._algo_param_layout.setContentsMargins(12, 12, 12, 12)
        self._algo_param_layout.setHorizontalSpacing(12)
        self._algo_param_layout.setVerticalSpacing(8)
        left_column.addWidget(self._algo_param_group)

        left_column.addStretch(1)
        columns_layout.addLayout(left_column, 1)

        # Right column: Visualization (optional)
        right_column = QtWidgets.QVBoxLayout()

        vis_group = QtWidgets.QGroupBox("Visualization (Optional)", self)
        vis_layout = QtWidgets.QVBoxLayout(vis_group)

        self._enable_fastlane_checkbox = QtWidgets.QCheckBox("Enable FastLane real-time view", vis_group)
        self._enable_fastlane_checkbox.setChecked(False)  # OFF by default for resume
        self._enable_fastlane_checkbox.setToolTip(
            "Show the agent playing in real-time.\n"
            "Not required - TensorBoard is the primary output for training progress."
        )
        self._enable_fastlane_checkbox.stateChanged.connect(self._on_fastlane_toggled)
        vis_layout.addWidget(self._enable_fastlane_checkbox)

        # FastLane options container (initially hidden)
        self._fastlane_options = QtWidgets.QWidget(vis_group)
        fastlane_layout = QtWidgets.QGridLayout(self._fastlane_options)
        fastlane_layout.setContentsMargins(20, 0, 0, 0)

        self._fastlane_only_checkbox = QtWidgets.QCheckBox("Fast Lane only (no durable path)", self._fastlane_options)
        self._fastlane_only_checkbox.setChecked(True)
        fastlane_layout.addWidget(self._fastlane_only_checkbox, 0, 0, 1, 2)

        fastlane_layout.addWidget(QtWidgets.QLabel("Video Mode:", self._fastlane_options), 1, 0)
        self._video_mode_combo = QtWidgets.QComboBox(self._fastlane_options)
        for mode, descriptor in VIDEO_MODE_DESCRIPTORS.items():
            self._video_mode_combo.addItem(descriptor.label, mode)
        fastlane_layout.addWidget(self._video_mode_combo, 1, 1)

        self._fastlane_options.setVisible(False)  # Hidden by default
        vis_layout.addWidget(self._fastlane_options)

        vis_layout.addStretch(1)
        right_column.addWidget(vis_group)
        right_column.addStretch(1)
        columns_layout.addLayout(right_column, 1)

        layout.addLayout(columns_layout)

        # Dialog buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._checkpoints))
        for row, checkpoint in enumerate(self._checkpoints):
            self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(checkpoint.env_id or ""))
            self._table.setItem(row, 1, QtWidgets.QTableWidgetItem(checkpoint.algo or ""))
            self._table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(checkpoint.seed) if checkpoint.seed else ""))
            self._table.setItem(row, 3, QtWidgets.QTableWidgetItem(checkpoint.run_id or ""))
            self._table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(checkpoint.policy_path)))
        self._table.resizeColumnsToContents()

    def _on_table_selection(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._selected = None
            self._path_label.setText("No checkpoint selected")
            return
        row = rows[0].row()
        if row < 0 or row >= len(self._checkpoints):
            return
        checkpoint = self._checkpoints[row]
        self._apply_checkpoint(checkpoint)

    def _apply_checkpoint(self, checkpoint: CleanRlCheckpoint) -> None:
        self._selected = checkpoint
        self._path_label.setText(str(checkpoint.policy_path))

        # Set seed from checkpoint
        if checkpoint.seed:
            self._seed_spin.setValue(checkpoint.seed)

        # Rebuild algorithm parameters based on checkpoint's algorithm
        self._rebuild_algo_params(checkpoint.algo or "ppo")

    def _rebuild_algo_params(self, algo: str) -> None:
        """Rebuild the algorithm parameters widget based on the algorithm."""
        # Clear existing widgets
        while self._algo_param_layout.count():
            item = self._algo_param_layout.takeAt(0)
            if item and item.widget():
                item.widget().setParent(None)
        self._algo_param_inputs.clear()

        specs = _ALGO_PARAM_SPECS.get(algo, _ALGO_PARAM_SPECS.get("ppo", ()))
        if not specs:
            self._algo_param_group.setVisible(False)
            return

        self._algo_param_group.setVisible(True)

        for idx, spec in enumerate(specs):
            widget: QtWidgets.QWidget
            if spec.field_type is int:
                spin = QtWidgets.QSpinBox(self._algo_param_group)
                spin.setRange(1, 1_000_000_000)
                spin.setValue(int(spec.default))
                if spec.tooltip:
                    spin.setToolTip(spec.tooltip)
                widget = spin
            elif spec.field_type is float:
                spin = QtWidgets.QDoubleSpinBox(self._algo_param_group)
                spin.setDecimals(6)
                spin.setRange(0.0, 1e9)
                step = abs(spec.default) / 10 if isinstance(spec.default, (int, float)) and spec.default else 0.01
                spin.setSingleStep(step)
                spin.setValue(float(spec.default))
                if spec.tooltip:
                    spin.setToolTip(spec.tooltip)
                widget = spin
            elif spec.field_type is bool:
                checkbox = QtWidgets.QCheckBox(spec.label, self._algo_param_group)
                checkbox.setChecked(bool(spec.default))
                if spec.tooltip:
                    checkbox.setToolTip(spec.tooltip)
                widget = checkbox
            else:
                line = QtWidgets.QLineEdit(self._algo_param_group)
                line.setText(str(spec.default))
                if spec.tooltip:
                    line.setToolTip(spec.tooltip)
                widget = line

            self._algo_param_inputs[spec.key] = widget

            # Create cell with label
            row = idx // 2
            col = idx % 2
            cell = QtWidgets.QWidget(self._algo_param_group)
            cell_layout = QtWidgets.QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)

            if not isinstance(widget, QtWidgets.QCheckBox):
                label_widget = QtWidgets.QLabel(spec.label, cell)
                label_widget.setStyleSheet("font-weight: 600;")
                cell_layout.addWidget(label_widget)
            cell_layout.addWidget(widget)

            self._algo_param_layout.addWidget(cell, row, col)

    def _on_fastlane_toggled(self, state: int) -> None:
        """Show/hide FastLane options based on checkbox state."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self._fastlane_options.setVisible(enabled)

    def _on_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select CleanRL Checkpoint",
            str(VAR_TRAINER_DIR / "runs"),
            "CleanRL Models (*.cleanrl_model)",
        )
        if not path:
            return
        policy = Path(path)
        metadata = load_metadata_for_policy(policy)
        if metadata is None:
            metadata = CleanRlCheckpoint(
                policy_path=policy,
                run_id=generate_run_id("cleanrl-resume", "manual"),
                cleanrl_run_name=None,
                env_id=None,
                algo=None,
                seed=None,
                num_envs=None,
                fastlane_only=True,
                fastlane_video_mode=VideoModes.SINGLE,
                fastlane_grid_limit=1,
                config_path=None,
            )
        self._apply_checkpoint(metadata)

    def _on_accept(self) -> None:
        if self._selected is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Checkpoint Required",
                "Select a checkpoint before resuming training.",
            )
            return

        checkpoint = self._selected
        if not checkpoint.env_id:
            QtWidgets.QMessageBox.warning(
                self,
                "Environment Unknown",
                "Cannot determine environment from checkpoint. Please select a different checkpoint.",
            )
            return

        config = self._build_config(checkpoint)
        self._result_config = config
        self.log_constant(
            LOG_UI_POLICY_FORM_INFO,
            extra={"run_id": checkpoint.run_id, "policy_path": str(checkpoint.policy_path)},
        )
        self.accept()

    def _collect_algo_params(self) -> Dict[str, Any]:
        """Collect algorithm parameters from the UI widgets."""
        params: Dict[str, Any] = {}
        for key, widget in self._algo_param_inputs.items():
            if isinstance(widget, QtWidgets.QSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QtWidgets.QCheckBox):
                params[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QLineEdit):
                params[key] = widget.text()
        return params

    def _build_config(self, checkpoint: CleanRlCheckpoint) -> Dict[str, Any]:
        run_id = generate_run_id("cleanrl-resume", checkpoint.algo or "policy")
        additional_timesteps = int(self._timesteps_spin.value())
        algo_params = self._collect_algo_params()

        # Save model based on checkbox
        algo_params["save_model"] = self._save_model_checkbox.isChecked()

        # FastLane is optional (off by default)
        fastlane_enabled = self._enable_fastlane_checkbox.isChecked()
        fastlane_only = self._fastlane_only_checkbox.isChecked() if fastlane_enabled else False
        video_mode = self._video_mode_combo.currentData() if fastlane_enabled else VideoModes.SINGLE

        # TensorBoard path - this is where CleanRL writes directly
        tensorboard_dir = "tensorboard"
        tensorboard_abs = (VAR_TRAINER_DIR / "runs" / run_id / tensorboard_dir).resolve()

        extras: Dict[str, Any] = {
            "mode": "resume_training",
            "checkpoint_path": str(checkpoint.policy_path),
            "tensorboard_dir": tensorboard_dir,
            "algo_params": algo_params,
            # FastLane only if enabled
            "fastlane_enabled": fastlane_enabled,
            "fastlane_only": fastlane_only,
            "fastlane_slot": 0,
            "fastlane_video_mode": video_mode,
            "fastlane_grid_limit": 1,
        }

        seed_value = int(self._seed_spin.value())
        worker_config: Dict[str, Any] = {
            "run_id": run_id,
            "algo": checkpoint.algo or "ppo",
            "env_id": checkpoint.env_id,
            "total_timesteps": additional_timesteps,
            "seed": seed_value if seed_value > 0 else (checkpoint.seed or 1),
            "extras": extras,
        }

        # Metadata structure for TensorBoard-only mode (no Live-Agent telemetry)
        metadata = {
            "ui": {
                "worker_id": "cleanrl_worker",
                "algo": worker_config["algo"],
                "env_id": checkpoint.env_id,
                "dry_run": False,
                "run_mode": "resume_training",
                "checkpoint_path": str(checkpoint.policy_path),
                "additional_timesteps": additional_timesteps,
                "fastlane_enabled": fastlane_enabled,
                "fastlane_only": fastlane_only,
                "fastlane_slot": 0,
                "fastlane_video_mode": video_mode,
                "fastlane_grid_limit": 1,
            },
            "worker": {
                "worker_id": "cleanrl_worker",
                "module": "cleanrl_worker.cli",
                "use_grpc": False,  # No gRPC telemetry for resume training
                "grpc_target": "127.0.0.1:50055",
                "arguments": [],
                "config": worker_config,
            },
            "artifacts": {
                "tensorboard": {
                    "enabled": True,
                    "relative_path": tensorboard_dir,
                    "log_dir": str(tensorboard_abs),
                },
                "wandb": {
                    "enabled": False,
                    "run_path": None,
                },
            },
        }

        # Environment variables - TensorBoard-only mode
        environment: Dict[str, Any] = {
            "CLEANRL_RUN_ID": run_id,
            "CLEANRL_RESUME_PATH": str(checkpoint.policy_path),
            # Enable TensorBoard tracking (CleanRL native)
            "TRACK_TENSORBOARD": "1",
            "CLEANRL_TENSORBOARD_DIR": str(tensorboard_abs),
            # Disable WANDB
            "TRACK_WANDB": "0",
            "WANDB_MODE": "offline",
            "WANDB_DISABLE_GYM": "true",
            # Algorithm params
            "CLEANRL_NUM_ENVS": str(algo_params.get("num_envs", checkpoint.num_envs or 1)),
        }

        # Pass learning rate if overridden
        if "learning_rate" in algo_params:
            environment["CLEANRL_LEARNING_RATE"] = str(algo_params["learning_rate"])

        # Only apply FastLane environment when enabled
        if fastlane_enabled:
            apply_fastlane_environment(
                environment,
                fastlane_only=fastlane_only,
                fastlane_slot=0,
                video_mode=video_mode,
                grid_limit=1,
            )
        else:
            # Explicitly disable FastLane
            environment["MOSAIC_FASTLANE_ENABLED"] = "0"

        config: Dict[str, Any] = {
            "run_name": run_id,
            "entry_point": sys.executable,
            "arguments": ["-m", "cleanrl_worker.cli"],
            "environment": environment,
            "resources": {
                "cpus": 4,
                "memory_mb": 2048,
                "gpus": {"requested": 0, "mandatory": False},
            },
            "metadata": metadata,
            "artifacts": {
                "output_prefix": f"runs/{run_id}",
                "persist_logs": True,
                "keep_checkpoints": True,
            },
        }
        return config

    def get_config(self) -> Optional[Dict[str, Any]]:
        return self._result_config


# Register with factory
from gym_gui.ui.forms import get_worker_form_factory  # noqa: E402

_factory = get_worker_form_factory()
if not _factory.has_resume_form("cleanrl_worker"):
    _factory.register_resume_form(
        "cleanrl_worker",
        lambda parent=None, **kwargs: CleanRlResumeForm(parent=parent, **kwargs),
    )
