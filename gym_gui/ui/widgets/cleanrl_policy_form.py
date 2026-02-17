"""Dialog for loading trained CleanRL policies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import sys

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.config.paths import VAR_TRAINER_DIR
from gym_gui.config.cleanrl_eval_presets import get_eval_preset
from gym_gui.core.enums import EnvironmentFamily, GameId
from gym_gui.fastlane.worker_helpers import apply_fastlane_environment
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_POLICY_FORM_TRACE,
    LOG_UI_POLICY_FORM_INFO,
    LOG_UI_POLICY_FORM_ERROR,
)
from gym_gui.telemetry.semconv import VideoModes, VIDEO_MODE_DESCRIPTORS
from gym_gui.ui.widgets.cleanrl_train_form import (
    CLEANRL_ENVIRONMENT_FAMILY_INDEX,
    get_cleanrl_environment_family_index,
)
from gym_gui.ui.widgets.cleanrl_train_form import _generate_run_id as generate_run_id
from gym_gui.ui.widgets.cleanrl_train_form import _format_cleanrl_family_label as format_family_label
from gym_gui.policy_discovery.cleanrl_policy_metadata import (
    CleanRlCheckpoint,
    discover_policies,
    load_metadata_for_policy,
)


_LOGGER = logging.getLogger(__name__)

@dataclass
class _PolicyState:
    checkpoint: CleanRlCheckpoint
    env_id: Optional[str]
    family: Optional[EnvironmentFamily]
    fastlane_only: bool
    video_mode: str
    grid_limit: int
    eval_capture_video: bool
    seed: Optional[int]


class CleanRlPolicyForm(QtWidgets.QDialog, LogConstantMixin):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        current_game: Optional[GameId] = None,
    ) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self.setWindowTitle("Load CleanRL Policy")
        self.resize(840, 540)
        self._current_game = current_game
        self._checkpoints = discover_policies()
        self._selected: Optional[CleanRlCheckpoint] = None
        self._result_config: Optional[Dict[str, Any]] = None
        self._build_ui()
        self._populate_table()
        if self._checkpoints:
            self._table.selectRow(0)

    # ------------------------------------------------------------------
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
                f"Select a CleanRL checkpoint from <code>{runs_root}</code> or browse for an external model."
            )
        else:
            intro.setText(
                f"No CleanRL checkpoints found under <code>{runs_root}</code>."
            )
        layout.addWidget(intro)

        self._table = QtWidgets.QTableWidget(self)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Env", "Algo", "Seed", "Run", "Path"])
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        layout.addWidget(self._table, 2)

        browse_layout = QtWidgets.QHBoxLayout()
        self._path_label = QtWidgets.QLabel("No policy selected", self)
        self._path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        browse_layout.addWidget(self._path_label, 1)
        browse_btn = QtWidgets.QPushButton("Browse…", self)
        browse_btn.clicked.connect(self._on_browse)
        browse_layout.addWidget(browse_btn)
        layout.addLayout(browse_layout)

        form_layout = QtWidgets.QGridLayout()
        form_layout.setColumnStretch(1, 1)

        # Environment overrides
        self._override_env_checkbox = QtWidgets.QCheckBox("Override environment", self)
        self._override_env_checkbox.stateChanged.connect(self._on_override_toggled)
        form_layout.addWidget(self._override_env_checkbox, 0, 0, 1, 2)

        self._family_combo = QtWidgets.QComboBox(self)
        self._env_combo = QtWidgets.QComboBox(self)
        form_layout.addWidget(QtWidgets.QLabel("Family", self), 1, 0)
        form_layout.addWidget(self._family_combo, 1, 1)
        form_layout.addWidget(QtWidgets.QLabel("Environment", self), 2, 0)
        form_layout.addWidget(self._env_combo, 2, 1)
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)

        # Visualization options (FastLane is optional for evaluation)
        vis_group = QtWidgets.QGroupBox("Visualization (Optional)", self)
        vis_layout = QtWidgets.QGridLayout(vis_group)

        self._enable_fastlane_checkbox = QtWidgets.QCheckBox("Enable FastLane real-time view", vis_group)
        self._enable_fastlane_checkbox.setChecked(False)  # Off by default for evaluation
        self._enable_fastlane_checkbox.setToolTip(
            "Show the agent playing in real-time. Not required for evaluation - TensorBoard results are the primary output."
        )
        self._enable_fastlane_checkbox.stateChanged.connect(self._on_fastlane_toggled)
        vis_layout.addWidget(self._enable_fastlane_checkbox, 0, 0, 1, 2)

        # FastLane options (initially hidden)
        self._fastlane_only_checkbox = QtWidgets.QCheckBox("Fast Lane only (no durable path)", vis_group)
        self._fastlane_only_checkbox.setChecked(True)
        vis_layout.addWidget(self._fastlane_only_checkbox, 1, 0, 1, 2)
        vis_layout.addWidget(QtWidgets.QLabel("Video Mode", vis_group), 2, 0)
        self._video_mode_combo = QtWidgets.QComboBox(vis_group)
        for mode, descriptor in VIDEO_MODE_DESCRIPTORS.items():
            self._video_mode_combo.addItem(descriptor.label, mode)
        vis_layout.addWidget(self._video_mode_combo, 2, 1)
        vis_layout.addWidget(QtWidgets.QLabel("Grid Limit", vis_group), 3, 0)
        self._grid_spin = QtWidgets.QSpinBox(vis_group)
        self._grid_spin.setRange(1, 64)
        vis_layout.addWidget(self._grid_spin, 3, 1)
        self._video_mode_combo.currentIndexChanged.connect(self._sync_video_mode_controls)

        self._eval_video_checkbox = QtWidgets.QCheckBox("Capture evaluation video to disk", vis_group)
        vis_layout.addWidget(self._eval_video_checkbox, 4, 0, 1, 2)

        # Initially hide FastLane options
        self._fastlane_only_checkbox.setVisible(False)
        self._video_mode_combo.setVisible(False)
        self._grid_spin.setVisible(False)

        form_layout.addWidget(vis_group, 3, 0, 1, 2)

        # Seed override
        form_layout.addWidget(QtWidgets.QLabel("Seed", self), 4, 0)
        self._seed_spin = QtWidgets.QSpinBox(self)
        self._seed_spin.setRange(1, 2_147_483_647)
        form_layout.addWidget(self._seed_spin, 4, 1)

        form_layout.addWidget(QtWidgets.QLabel("Eval episodes per batch", self), 5, 0)
        self._episode_spin = QtWidgets.QSpinBox(self)
        self._episode_spin.setRange(1, 1_000_000)
        self._episode_spin.setValue(50)
        form_layout.addWidget(self._episode_spin, 5, 1)

        self._repeat_checkbox = QtWidgets.QCheckBox("Repeat evaluation until stopped", self)
        form_layout.addWidget(self._repeat_checkbox, 6, 0, 1, 2)

        advanced_group = QtWidgets.QGroupBox("Advanced Evaluation Controls", self)
        advanced_layout = QtWidgets.QGridLayout(advanced_group)
        advanced_layout.addWidget(QtWidgets.QLabel("Discount (gamma)", advanced_group), 0, 0)
        self._gamma_spin = QtWidgets.QDoubleSpinBox(advanced_group)
        self._gamma_spin.setDecimals(4)
        self._gamma_spin.setRange(0.0, 0.9999)
        self._gamma_spin.setSingleStep(0.01)
        self._gamma_spin.setValue(0.99)
        advanced_layout.addWidget(self._gamma_spin, 0, 1)

        advanced_layout.addWidget(QtWidgets.QLabel("Max episode steps (0 = default)", advanced_group), 1, 0)
        self._max_steps_spin = QtWidgets.QSpinBox(advanced_group)
        self._max_steps_spin.setRange(0, 1_000_000)
        self._max_steps_spin.setValue(0)
        advanced_layout.addWidget(self._max_steps_spin, 1, 1)

        advanced_layout.addWidget(QtWidgets.QLabel("Max episode seconds (0 = default)", advanced_group), 2, 0)
        self._max_seconds_spin = QtWidgets.QDoubleSpinBox(advanced_group)
        self._max_seconds_spin.setRange(0.0, 10_000.0)
        self._max_seconds_spin.setDecimals(2)
        self._max_seconds_spin.setSingleStep(1.0)
        self._max_seconds_spin.setValue(0.0)
        advanced_layout.addWidget(self._max_seconds_spin, 2, 1)

        preset_hint = QtWidgets.QLabel(
            "Presets load from metadata/cleanrl/eval_presets.json; adjust if needed.", advanced_group
        )
        preset_hint.setWordWrap(True)
        advanced_layout.addWidget(preset_hint, 3, 0, 1, 2)

        form_layout.addWidget(advanced_group, 7, 0, 1, 2)

        layout.addLayout(form_layout)

        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        self._ok_button = self._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if self._ok_button is not None:
            self._ok_button.setEnabled(False)
        layout.addWidget(self._button_box)

        self._populate_family_combo()
        self._toggle_env_controls(False)

    # ------------------------------------------------------------------
    def _populate_table(self) -> None:
        ordered = list(self._checkpoints)
        if self._current_game is not None:
            preferred = self._current_game.value
            ordered.sort(key=lambda ckpt: 0 if ckpt.env_id == preferred else 1)
        self._table.setRowCount(0)
        for checkpoint in ordered:
            row = self._table.rowCount()
            self._table.insertRow(row)
            values = [
                checkpoint.env_id or "?",
                checkpoint.algo or "?",
                str(checkpoint.seed or "?"),
                checkpoint.run_id,
                str(checkpoint.policy_path),
            ]
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, checkpoint)
                self._table.setItem(row, col, item)

    def _populate_family_combo(self, preferred: Optional[EnvironmentFamily] = None) -> None:
        self._family_combo.blockSignals(True)
        self._family_combo.clear()
        for family in CLEANRL_ENVIRONMENT_FAMILY_INDEX:
            self._family_combo.addItem(format_family_label(family), family)
        index = 0
        if preferred is not None:
            idx = self._family_combo.findData(preferred)
            if idx >= 0:
                index = idx
        self._family_combo.setCurrentIndex(index)
        self._family_combo.blockSignals(False)
        self._on_family_changed(index)

    def _on_family_changed(self, index: int) -> None:
        family = self._family_combo.itemData(index)
        self._env_combo.blockSignals(True)
        self._env_combo.clear()
        options = get_cleanrl_environment_family_index().get(family, [])
        for label, env_id in options:
            self._env_combo.addItem(label, env_id)
        self._env_combo.blockSignals(False)
        self._on_env_changed(self._env_combo.currentIndex())

    def _on_env_changed(self, index: int) -> None:
        env_id = self._env_combo.itemData(index)
        if isinstance(env_id, str) and env_id:
            self._apply_eval_presets(env_id)

    def _toggle_env_controls(self, enabled: bool) -> None:
        self._family_combo.setEnabled(enabled)
        self._env_combo.setEnabled(enabled)

    def _on_override_toggled(self, state: int) -> None:
        self._toggle_env_controls(state == QtCore.Qt.CheckState.Checked)

    def _on_table_selection(self) -> None:
        selection = self._table.selectionModel()
        if selection is None or not selection.hasSelection():
            if self._ok_button is not None:
                self._ok_button.setEnabled(False)
            return
        row = selection.selectedRows()[0].row()
        item = self._table.item(row, 0)
        checkpoint = item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
        if isinstance(checkpoint, CleanRlCheckpoint):
            self._apply_checkpoint(checkpoint)

    def _apply_checkpoint(self, checkpoint: CleanRlCheckpoint) -> None:
        self._selected = checkpoint
        self._path_label.setText(str(checkpoint.policy_path))
        env_id = checkpoint.env_id
        family = None
        if env_id:
            for fam, mappings in CLEANRL_ENVIRONMENT_FAMILY_INDEX.items():
                if any(eid == env_id for _, eid in mappings):
                    family = fam
                    break
        if family is not None:
            self._populate_family_combo(family)
            env_index = self._env_combo.findData(env_id)
            if env_index >= 0:
                self._env_combo.setCurrentIndex(env_index)
            self._override_env_checkbox.setChecked(False)
            self._toggle_env_controls(False)
        else:
            self._override_env_checkbox.setChecked(True)
            self._toggle_env_controls(True)
        self._fastlane_only_checkbox.setChecked(checkpoint.fastlane_only)
        video_mode = checkpoint.fastlane_video_mode or VideoModes.SINGLE
        idx = self._video_mode_combo.findData(video_mode)
        if idx >= 0:
            self._video_mode_combo.setCurrentIndex(idx)
        grid_limit = checkpoint.fastlane_grid_limit or max(1, checkpoint.num_envs or 1)
        self._grid_spin.setValue(max(1, grid_limit))
        self._seed_spin.setValue(checkpoint.seed or 1)
        self._eval_video_checkbox.setChecked(False)
        self._repeat_checkbox.setChecked(False)
        self._apply_eval_presets(checkpoint.env_id)
        if self._ok_button is not None:
            self._ok_button.setEnabled(True)
        self.log_constant(
            LOG_UI_POLICY_FORM_TRACE,
            extra={"policy_path": str(checkpoint.policy_path), "run_id": checkpoint.run_id},
        )

    def _apply_eval_presets(self, env_id: Optional[str]) -> None:
        preset = get_eval_preset(env_id)
        batch_size = preset.get("eval_batch_size")
        if isinstance(batch_size, int) and batch_size > 0:
            self._episode_spin.setValue(batch_size)
        repeat = preset.get("eval_repeat")
        if isinstance(repeat, bool):
            self._repeat_checkbox.setChecked(repeat)
        capture_video = preset.get("capture_video")
        if isinstance(capture_video, bool):
            self._eval_video_checkbox.setChecked(capture_video)
        gamma = preset.get("gamma")
        if isinstance(gamma, (int, float)):
            self._gamma_spin.setValue(float(gamma))
        max_steps = preset.get("max_episode_steps")
        if isinstance(max_steps, (int, float)) and max_steps > 0:
            self._max_steps_spin.setValue(int(max_steps))
        else:
            self._max_steps_spin.setValue(0)
        max_seconds = preset.get("max_episode_seconds")
        if isinstance(max_seconds, (int, float)) and max_seconds > 0:
            self._max_seconds_spin.setValue(float(max_seconds))
        else:
            self._max_seconds_spin.setValue(0.0)

    def _sync_video_mode_controls(self) -> None:
        mode = self._video_mode_combo.currentData()
        if mode == VideoModes.SINGLE:
            self._grid_spin.setValue(1)
            self._grid_spin.setEnabled(False)
        elif mode == VideoModes.GRID:
            self._grid_spin.setEnabled(True)
        else:
            self._grid_spin.setEnabled(False)

    def _on_fastlane_toggled(self, state: int) -> None:
        """Show/hide FastLane options based on checkbox state."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self._fastlane_only_checkbox.setVisible(enabled)
        self._video_mode_combo.setVisible(enabled)
        self._grid_spin.setVisible(enabled)
        # Find and toggle the labels too
        parent = self._video_mode_combo.parent()
        if parent:
            layout = parent.layout()
            if layout:
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item and item.widget():
                        widget = item.widget()
                        if isinstance(widget, QtWidgets.QLabel):
                            text = widget.text()
                            if text in ("Video Mode", "Grid Limit"):
                                widget.setVisible(enabled)

    def _on_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select CleanRL Model",
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
                run_id=generate_run_id("cleanrl-eval", "manual"),
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
            return
        env_id = self._env_combo.currentData()
        if env_id is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Environment Required",
                "Select or override the environment before running evaluation.",
            )
            return
        metadata = self._selected
        config = self._build_config(metadata, env_id=str(env_id))
        self._result_config = config
        self.log_constant(
            LOG_UI_POLICY_FORM_INFO,
            extra={"run_id": metadata.run_id, "policy_path": str(metadata.policy_path)},
        )
        self.accept()

    def _build_config(self, checkpoint: CleanRlCheckpoint, *, env_id: str) -> Dict[str, Any]:
        run_id = generate_run_id("cleanrl-eval", checkpoint.algo or "policy")
        episodes_per_batch = int(self._episode_spin.value())
        repeat_eval = self._repeat_checkbox.isChecked()

        # FastLane is optional for evaluation - only enable if user checks the box
        fastlane_enabled = self._enable_fastlane_checkbox.isChecked()
        fastlane_only = self._fastlane_only_checkbox.isChecked() if fastlane_enabled else False
        video_mode = self._video_mode_combo.currentData() if fastlane_enabled else VideoModes.SINGLE
        grid_limit = int(self._grid_spin.value()) if fastlane_enabled else 1

        extras: Dict[str, Any] = {
            "mode": "policy_eval",
            "policy_path": str(checkpoint.policy_path),
            "fastlane_enabled": fastlane_enabled,
            "fastlane_only": fastlane_only,
            "fastlane_slot": 0,
            "fastlane_video_mode": video_mode,
            "fastlane_grid_limit": grid_limit,
            "eval_capture_video": self._eval_video_checkbox.isChecked(),
            "eval_episodes": episodes_per_batch,
            "eval_batch_size": episodes_per_batch,
            "eval_repeat": repeat_eval,
            "tensorboard_dir": "tensorboard",
            "eval_gamma": float(self._gamma_spin.value()),
            "eval_max_episode_steps": int(self._max_steps_spin.value()) if self._max_steps_spin.value() > 0 else None,
            "eval_max_episode_seconds": float(self._max_seconds_spin.value()) if self._max_seconds_spin.value() > 0 else None,
        }
        agent_id = "cleanrl_eval"
        worker_config: Dict[str, Any] = {
            "run_id": run_id,
            "algo": checkpoint.algo or "ppo_continuous_action",
            "env_id": env_id,
            "total_timesteps": max(1, episodes_per_batch),
            "extras": extras,
        }
        worker_config["agent_id"] = agent_id
        seed_value = int(self._seed_spin.value())
        if seed_value > 0:
            worker_config["seed"] = seed_value

        metadata = {
            "ui": {
                "worker_id": "cleanrl_worker",
                "agent_id": agent_id,
                "algo": worker_config["algo"],
                "env_id": env_id,
                "dry_run": False,
                "fastlane_enabled": fastlane_enabled,
                "fastlane_only": fastlane_only,
                "fastlane_slot": 0,
                "fastlane_video_mode": video_mode,
                "fastlane_grid_limit": grid_limit,
                "run_mode": "policy_eval",
                "eval_episodes": episodes_per_batch,
                "eval_batch_size": episodes_per_batch,
                "eval_repeat": repeat_eval,
                "tensorboard_dir": extras["tensorboard_dir"],
                "eval_gamma": extras["eval_gamma"],
                "eval_max_episode_steps": extras["eval_max_episode_steps"],
                "eval_max_episode_seconds": extras["eval_max_episode_seconds"],
            },
            "worker": {
                "worker_id": "cleanrl_worker",
                "module": "cleanrl_worker.cli",
                "use_grpc": True,
                "grpc_target": "127.0.0.1:50055",
                "arguments": [],
                "config": worker_config,
            },
        }

        tensorboard_relpath = f"var/trainer/evals/{run_id}/{extras['tensorboard_dir']}"
        tensorboard_abs = (VAR_TRAINER_DIR / "evals" / run_id / extras["tensorboard_dir"]).resolve()
        metadata["artifacts"] = {
            "tensorboard": {
                "enabled": True,
                "relative_path": tensorboard_relpath,
                "log_dir": str(tensorboard_abs),
            },
            "wandb": {
                "enabled": False,
                "run_path": None,
            },
        }

        environment: Dict[str, Any] = {
            "CLEANRL_RUN_ID": run_id,
            "CLEANRL_AGENT_ID": agent_id,
            "MOSAIC_RUN_DIR": str((VAR_TRAINER_DIR / "evals" / run_id).resolve()),
            "TRACK_TENSORBOARD": "0",
            "TRACK_WANDB": "0",
            "WANDB_MODE": "offline",
            "WANDB_DISABLE_GYM": "true",
            "CLEANRL_NUM_ENVS": str(checkpoint.num_envs or 1),
        }

        # Only apply FastLane environment when enabled
        if fastlane_enabled:
            apply_fastlane_environment(
                environment,
                fastlane_only=fastlane_only,
                fastlane_slot=0,
                video_mode=video_mode,
                grid_limit=grid_limit,
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
                "keep_checkpoints": False,
            },
        }
        return config

    def get_config(self) -> Optional[Dict[str, Any]]:
        return self._result_config


# Late import to avoid cycles with the train form registrations.
from gym_gui.ui.forms.factory import get_worker_form_factory  # noqa: E402  # isort:skip

_factory = get_worker_form_factory()
if not _factory.has_policy_form("cleanrl_worker"):
    _factory.register_policy_form(
        "cleanrl_worker",
        lambda parent=None, **kwargs: CleanRlPolicyForm(parent=parent, **kwargs),
    )
