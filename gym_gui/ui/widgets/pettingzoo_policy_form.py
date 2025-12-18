"""Dialog for loading trained PettingZoo multi-agent policies.

This form provides UI for selecting and configuring evaluation of
trained PettingZoo policies across AEC and Parallel environments.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtWidgets

from gym_gui.config.paths import VAR_TRAINER_DIR
from gym_gui.core.pettingzoo_enums import (
    PETTINGZOO_ENV_METADATA,
    PettingZooAPIType,
    PettingZooEnvId,
    PettingZooFamily,
    get_api_type,
    get_description,
    get_display_name,
    get_envs_by_family,
    is_aec_env,
)
from gym_gui.policy_discovery.pettingzoo_policy_metadata import (
    PettingZooCheckpoint,
    discover_policies,
    load_metadata_for_policy,
)

_LOGGER = logging.getLogger(__name__)


class PettingZooPolicyForm(QtWidgets.QDialog):
    """Policy evaluation form for PettingZoo multi-agent environments.

    Provides UI for:
    - Discovering and selecting trained policies
    - Browsing for external policy files
    - Configuring evaluation parameters
    - Overriding environment settings
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        current_game: Optional[Any] = None,
    ) -> None:
        """Initialize the policy form.

        Args:
            parent: Parent widget
            current_game: Currently selected game (may be PettingZooEnvId)
        """
        super().__init__(parent)
        self.setWindowTitle("Load PettingZoo Policy")
        self.resize(800, 600)

        self._current_game = current_game
        self._checkpoints = discover_policies()
        self._selected: Optional[PettingZooCheckpoint] = None
        self._result_config: Optional[Dict[str, Any]] = None

        self._build_ui()
        self._populate_table()

        if self._checkpoints:
            self._table.selectRow(0)

    def _build_ui(self) -> None:
        """Build the form UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Introduction label
        runs_root = (VAR_TRAINER_DIR / "runs").resolve()
        intro = QtWidgets.QLabel(self)
        intro.setWordWrap(True)
        intro.setTextFormat(QtCore.Qt.TextFormat.RichText)

        if self._checkpoints:
            intro.setText(
                f"Select a PettingZoo checkpoint from <code>{runs_root}</code> "
                f"or browse for an external model."
            )
        else:
            intro.setText(
                f"No PettingZoo checkpoints found under <code>{runs_root}</code>. "
                f"Browse for an external model or train a policy first."
            )
        layout.addWidget(intro)

        # Policy table
        self._table = QtWidgets.QTableWidget(self)
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["Environment", "Family", "Algorithm", "API", "Seed", "Path"]
        )
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        layout.addWidget(self._table, 2)

        # Browse section
        browse_layout = QtWidgets.QHBoxLayout()
        self._path_label = QtWidgets.QLabel("No policy selected", self)
        self._path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        browse_layout.addWidget(self._path_label, 1)

        browse_btn = QtWidgets.QPushButton("Browse...", self)
        browse_btn.clicked.connect(self._on_browse)
        browse_layout.addWidget(browse_btn)
        layout.addLayout(browse_layout)

        # Configuration form
        form_layout = QtWidgets.QGridLayout()
        form_layout.setColumnStretch(1, 1)

        # Environment override
        self._override_env_checkbox = QtWidgets.QCheckBox(
            "Override environment", self
        )
        self._override_env_checkbox.stateChanged.connect(self._on_override_toggled)
        form_layout.addWidget(self._override_env_checkbox, 0, 0, 1, 2)

        self._family_combo = QtWidgets.QComboBox(self)
        self._env_combo = QtWidgets.QComboBox(self)
        form_layout.addWidget(QtWidgets.QLabel("Family:", self), 1, 0)
        form_layout.addWidget(self._family_combo, 1, 1)
        form_layout.addWidget(QtWidgets.QLabel("Environment:", self), 2, 0)
        form_layout.addWidget(self._env_combo, 2, 1)
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)

        # Evaluation settings group
        eval_group = QtWidgets.QGroupBox("Evaluation Settings", self)
        eval_layout = QtWidgets.QGridLayout(eval_group)

        # Number of episodes
        eval_layout.addWidget(QtWidgets.QLabel("Episodes:", eval_group), 0, 0)
        self._episodes_spin = QtWidgets.QSpinBox(eval_group)
        self._episodes_spin.setRange(1, 10000)
        self._episodes_spin.setValue(10)
        eval_layout.addWidget(self._episodes_spin, 0, 1)

        # Seed
        eval_layout.addWidget(QtWidgets.QLabel("Seed:", eval_group), 0, 2)
        self._seed_spin = QtWidgets.QSpinBox(eval_group)
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        self._seed_spin.setSpecialValueText("Random")
        eval_layout.addWidget(self._seed_spin, 0, 3)

        # Render mode
        eval_layout.addWidget(QtWidgets.QLabel("Render Mode:", eval_group), 1, 0)
        self._render_combo = QtWidgets.QComboBox(eval_group)
        self._render_combo.addItem("RGB Array", "rgb_array")
        self._render_combo.addItem("Human (window)", "human")
        self._render_combo.addItem("ANSI (text)", "ansi")
        self._render_combo.addItem("None", "none")
        eval_layout.addWidget(self._render_combo, 1, 1)

        # Capture video
        self._capture_video_checkbox = QtWidgets.QCheckBox(
            "Capture evaluation video", eval_group
        )
        eval_layout.addWidget(self._capture_video_checkbox, 1, 2, 1, 2)

        # Deterministic actions
        self._deterministic_checkbox = QtWidgets.QCheckBox(
            "Deterministic actions (no exploration)", eval_group
        )
        self._deterministic_checkbox.setChecked(True)
        eval_layout.addWidget(self._deterministic_checkbox, 2, 0, 1, 2)

        # Show agent stats
        self._show_stats_checkbox = QtWidgets.QCheckBox(
            "Show per-agent statistics", eval_group
        )
        self._show_stats_checkbox.setChecked(True)
        eval_layout.addWidget(self._show_stats_checkbox, 2, 2, 1, 2)

        form_layout.addWidget(eval_group, 3, 0, 1, 2)

        # Human control option (for AEC environments)
        self._human_control_group = QtWidgets.QGroupBox(
            "Human Control (AEC only)", self
        )
        human_layout = QtWidgets.QVBoxLayout(self._human_control_group)

        self._human_play_checkbox = QtWidgets.QCheckBox(
            "Play as human vs. trained policy", self._human_control_group
        )
        human_layout.addWidget(self._human_play_checkbox)

        agent_select_layout = QtWidgets.QHBoxLayout()
        agent_select_layout.addWidget(
            QtWidgets.QLabel("Human plays as:", self._human_control_group)
        )
        self._human_agent_combo = QtWidgets.QComboBox(self._human_control_group)
        self._human_agent_combo.addItem("Player 1", "player_0")
        self._human_agent_combo.addItem("Player 2", "player_1")
        agent_select_layout.addWidget(self._human_agent_combo)
        agent_select_layout.addStretch(1)
        human_layout.addLayout(agent_select_layout)

        self._human_control_group.setEnabled(False)
        form_layout.addWidget(self._human_control_group, 4, 0, 1, 2)

        layout.addLayout(form_layout)

        # Dialog buttons
        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)

        self._ok_button = self._button_box.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        if self._ok_button is not None:
            self._ok_button.setEnabled(False)

        layout.addWidget(self._button_box)

        # Initialize state
        self._populate_family_combo()
        self._toggle_env_controls(False)

    def _populate_table(self) -> None:
        """Populate the policy table with discovered checkpoints."""
        # Sort by current game preference
        ordered = list(self._checkpoints)
        if self._current_game is not None:
            current_env = (
                self._current_game.value
                if isinstance(self._current_game, PettingZooEnvId)
                else str(self._current_game)
            )
            ordered.sort(
                key=lambda ckpt: 0 if ckpt.env_id == current_env else 1
            )

        self._table.setRowCount(0)

        for checkpoint in ordered:
            row = self._table.rowCount()
            self._table.insertRow(row)

            values = [
                checkpoint.env_id or "?",
                checkpoint.family or "?",
                checkpoint.algorithm or "?",
                checkpoint.api_type or "?",
                str(checkpoint.seed or "?"),
                str(checkpoint.policy_path),
            ]

            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, checkpoint)
                self._table.setItem(row, col, item)

    def _populate_family_combo(
        self, preferred: Optional[PettingZooFamily] = None
    ) -> None:
        """Populate the family dropdown."""
        self._family_combo.blockSignals(True)
        self._family_combo.clear()

        for family in PettingZooFamily:
            self._family_combo.addItem(family.value.title(), family.value)

        index = 0
        if preferred is not None:
            idx = self._family_combo.findData(preferred.value)
            if idx >= 0:
                index = idx

        self._family_combo.setCurrentIndex(index)
        self._family_combo.blockSignals(False)
        self._on_family_changed(index)

    def _on_family_changed(self, index: int) -> None:
        """Handle family selection change."""
        family_value = self._family_combo.currentData()
        if not family_value:
            return

        try:
            family = PettingZooFamily(family_value)
        except ValueError:
            return

        self._env_combo.blockSignals(True)
        self._env_combo.clear()

        envs = get_envs_by_family(family)
        for env_id in envs:
            display_name = get_display_name(env_id)
            self._env_combo.addItem(display_name, env_id.value)

        self._env_combo.blockSignals(False)

        if envs:
            self._env_combo.setCurrentIndex(0)
            self._update_human_control_state()

    def _toggle_env_controls(self, enabled: bool) -> None:
        """Toggle environment override controls."""
        self._family_combo.setEnabled(enabled)
        self._env_combo.setEnabled(enabled)

    def _on_override_toggled(self, state: int) -> None:
        """Handle environment override checkbox change."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self._toggle_env_controls(enabled)

    def _on_table_selection(self) -> None:
        """Handle table row selection."""
        selection = self._table.selectionModel()
        if selection is None or not selection.hasSelection():
            if self._ok_button is not None:
                self._ok_button.setEnabled(False)
            return

        row = selection.selectedRows()[0].row()
        item = self._table.item(row, 0)
        checkpoint = (
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            if item is not None
            else None
        )

        if isinstance(checkpoint, PettingZooCheckpoint):
            self._apply_checkpoint(checkpoint)

    def _apply_checkpoint(self, checkpoint: PettingZooCheckpoint) -> None:
        """Apply a selected checkpoint to the form."""
        self._selected = checkpoint
        self._path_label.setText(str(checkpoint.policy_path))

        # Find and select family/env
        family = None
        if checkpoint.family:
            try:
                family = PettingZooFamily(checkpoint.family)
            except ValueError:
                pass

        if family is not None:
            self._populate_family_combo(family)

            if checkpoint.env_id:
                env_index = self._env_combo.findData(checkpoint.env_id)
                if env_index >= 0:
                    self._env_combo.setCurrentIndex(env_index)

            self._override_env_checkbox.setChecked(False)
            self._toggle_env_controls(False)
        else:
            self._override_env_checkbox.setChecked(True)
            self._toggle_env_controls(True)

        # Apply seed if available
        if checkpoint.seed:
            self._seed_spin.setValue(checkpoint.seed)

        # Update human control availability
        self._update_human_control_state()

        if self._ok_button is not None:
            self._ok_button.setEnabled(True)

        _LOGGER.debug(
            "Applied checkpoint: run=%s, env=%s",
            checkpoint.run_id,
            checkpoint.env_id,
        )

    def _update_human_control_state(self) -> None:
        """Update human control group based on current env selection."""
        env_value = self._env_combo.currentData()
        if not env_value:
            self._human_control_group.setEnabled(False)
            return

        try:
            env_id = PettingZooEnvId(env_value)
            is_aec = is_aec_env(env_id)
            self._human_control_group.setEnabled(is_aec)
        except ValueError:
            self._human_control_group.setEnabled(False)

    def _on_browse(self) -> None:
        """Handle browse button click."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select PettingZoo Model",
            str(VAR_TRAINER_DIR / "runs"),
            "Model Files (*.pt *.zip *.pkl);;All Files (*)",
        )

        if not path:
            return

        policy = Path(path)
        metadata = load_metadata_for_policy(policy)

        if metadata is None:
            # Create minimal checkpoint
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            metadata = PettingZooCheckpoint(
                policy_path=policy,
                run_id=f"pettingzoo-eval-{timestamp}",
                env_id=None,
                family=None,
                algorithm=None,
                api_type=None,
                seed=None,
                num_agents=None,
                total_timesteps=None,
                config_path=None,
            )

        self._apply_checkpoint(metadata)

    def _on_accept(self) -> None:
        """Handle OK button click."""
        if self._selected is None:
            return

        env_id = self._env_combo.currentData()
        if not env_id:
            QtWidgets.QMessageBox.warning(
                self,
                "Environment Required",
                "Select or override the environment before running evaluation.",
            )
            return

        config = self._build_config(env_id=str(env_id))
        self._result_config = config

        _LOGGER.info(
            "Built evaluation config: env=%s, policy=%s",
            env_id,
            self._selected.policy_path,
        )

        self.accept()

    def _build_config(self, *, env_id: str) -> Dict[str, Any]:
        """Build the evaluation configuration dictionary."""
        if self._selected is None:
            raise ValueError("No checkpoint selected")

        checkpoint = self._selected
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"pettingzoo-eval-{env_id.replace('_', '-')}-{timestamp}"

        # Determine family and API type
        family = self._family_combo.currentData() or "classic"
        is_parallel = False

        try:
            pz_env_id = PettingZooEnvId(env_id)
            is_parallel = not is_aec_env(pz_env_id)
        except ValueError:
            pass

        # Human control settings
        human_play = self._human_play_checkbox.isChecked()
        human_agent = self._human_agent_combo.currentData() if human_play else None

        config: Dict[str, Any] = {
            "run_name": run_name,
            "entry_point": "python",
            "arguments": ["-m", "pettingzoo_worker.cli", "evaluate"],
            "environment": {
                "PETTINGZOO_ENV_ID": env_id,
                "PETTINGZOO_FAMILY": family,
                "PETTINGZOO_API_TYPE": "parallel" if is_parallel else "aec",
                "POLICY_PATH": str(checkpoint.policy_path),
                "EVAL_EPISODES": str(self._episodes_spin.value()),
                "SEED": (
                    str(self._seed_spin.value())
                    if self._seed_spin.value() > 0
                    else ""
                ),
                "RENDER_MODE": self._render_combo.currentData(),
                "DETERMINISTIC": "1" if self._deterministic_checkbox.isChecked() else "0",
                "CAPTURE_VIDEO": "1" if self._capture_video_checkbox.isChecked() else "0",
                "HUMAN_PLAY": "1" if human_play else "0",
                "HUMAN_AGENT": human_agent or "",
            },
            "resources": {
                "cpus": 2,
                "memory_mb": 2048,
                "gpus": {"requested": 0, "mandatory": False},
            },
            "artifacts": {
                "output_prefix": f"runs/{run_name}",
                "persist_logs": True,
                "keep_checkpoints": False,
            },
            "metadata": {
                "ui": {
                    "worker_id": "pettingzoo_worker",
                    "env_id": env_id,
                    "family": family,
                    "is_parallel": is_parallel,
                    "mode": "evaluation",
                    "policy_path": str(checkpoint.policy_path),
                    "human_play": human_play,
                    "human_agent": human_agent,
                },
                "worker": {
                    "module": "pettingzoo_worker.cli",
                    "use_grpc": True,
                    "grpc_target": "127.0.0.1:50055",
                    "config": {
                        "policy_path": str(checkpoint.policy_path),
                        "env_id": env_id,
                        "family": family,
                        "api_type": "parallel" if is_parallel else "aec",
                        "num_episodes": self._episodes_spin.value(),
                        "seed": (
                            self._seed_spin.value()
                            if self._seed_spin.value() > 0
                            else None
                        ),
                        "render_mode": self._render_combo.currentData(),
                        "deterministic": self._deterministic_checkbox.isChecked(),
                        "capture_video": self._capture_video_checkbox.isChecked(),
                        "show_stats": self._show_stats_checkbox.isChecked(),
                        "human_play": human_play,
                        "human_agent": human_agent,
                    },
                },
                "artifacts": {
                    "tensorboard": {
                        "enabled": False,
                        "log_dir": f"runs/{run_name}/tensorboard",
                    },
                    "wandb": {
                        "enabled": False,
                    },
                },
            },
        }

        return config

    def get_config(self) -> Optional[Dict[str, Any]]:
        """Return the evaluation configuration.

        Returns:
            Configuration dictionary for TrainerClient submission, or None
        """
        return self._result_config


# Register form with factory at module load (late import to avoid cycles)
from gym_gui.ui.forms import get_worker_form_factory  # noqa: E402

_factory = get_worker_form_factory()
if not _factory.has_policy_form("pettingzoo_worker"):
    _factory.register_policy_form(
        "pettingzoo_worker",
        lambda parent=None, **kwargs: PettingZooPolicyForm(parent=parent, **kwargs),
    )


__all__ = ["PettingZooPolicyForm"]
