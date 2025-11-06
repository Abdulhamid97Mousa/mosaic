"""CleanRL worker training form."""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.enums import (
    EnvironmentFamily,
    GameId,
    ENVIRONMENT_FAMILY_BY_GAME,
)
from gym_gui.validations.validation_cleanrl_worker_form import run_cleanrl_dry_run
from gym_gui.Algo_docs.cleanrl_worker import get_algo_doc


_DEFAULT_ALGOS: tuple[str, ...] = (
    "ppo",
    "ppo_continuous_action",
    "ppo_atari",
    "dqn",
    "c51",
    "ppg_procgen",
    "ppo_rnd_envpool",
)


@dataclass(frozen=True)
class _AlgoParamSpec:
    key: str
    label: str
    default: Any
    field_type: type
    tooltip: str | None = None


_ALGO_PARAM_SPECS: dict[str, tuple[_AlgoParamSpec, ...]] = {
    "ppo": (
        _AlgoParamSpec("learning_rate", "Learning Rate", 2.5e-4, float, "Adam optimizer learning rate"),
        _AlgoParamSpec("num_envs", "Parallel Envs", 16, int, "Number of vectorized environments"),
        _AlgoParamSpec("num_steps", "Steps per Update", 2048, int, "Rollout length before each update"),
        _AlgoParamSpec("capture_video", "Capture Video", False, bool, "Record episode videos (first environment only)"),
    ),
    "ppo_atari": (
        _AlgoParamSpec("learning_rate", "Learning Rate", 2.5e-4, float, "Adam optimizer learning rate"),
        _AlgoParamSpec("total_frames", "Total Frames", 10_000_000, int, "Total Atari frames to collect"),
        _AlgoParamSpec("capture_video", "Capture Video", False, bool, "Record gameplay videos when supported"),
    ),
    "dqn": (
        _AlgoParamSpec("learning_rate", "Learning Rate", 1e-4, float, "Adam optimizer learning rate"),
        _AlgoParamSpec("batch_size", "Batch Size", 128, int, "Mini-batch size from replay buffer"),
        _AlgoParamSpec("buffer_size", "Replay Buffer", 100_000, int, "Maximum replay buffer size"),
        _AlgoParamSpec("capture_video", "Capture Video", False, bool, "Record evaluation videos (first environment only)"),
    ),
}


_SUPPORTED_FAMILIES: set[EnvironmentFamily] = {
    EnvironmentFamily.CLASSIC_CONTROL,
    EnvironmentFamily.BOX2D,
    EnvironmentFamily.MUJOCO,
    EnvironmentFamily.TOY_TEXT,
    EnvironmentFamily.MINIGRID,
    EnvironmentFamily.ATARI,
    EnvironmentFamily.ALE,
    EnvironmentFamily.OTHER,
}

_ADDITIONAL_SUPPORTED_GAMES: set[GameId] = {
    GameId.PONG_NO_FRAMESKIP,
    GameId.BREAKOUT_NO_FRAMESKIP,
    GameId.PROCGEN_COINRUN,
    GameId.PROCGEN_MAZE,
}

_PREFERRED_GAME_ORDER: Sequence[GameId] = (
    GameId.FROZEN_LAKE,
    GameId.FROZEN_LAKE_V2,
    GameId.CLIFF_WALKING,
    GameId.TAXI,
    GameId.BLACKJACK,
    GameId.CART_POLE,
    GameId.ACROBOT,
    GameId.MOUNTAIN_CAR,
    GameId.LUNAR_LANDER,
    GameId.CAR_RACING,
    GameId.BIPEDAL_WALKER,
    GameId.ANT,
    GameId.HALF_CHEETAH,
    GameId.HOPPER,
    GameId.WALKER2D,
    GameId.HUMANOID,
    GameId.HUMANOID_STANDUP,
    GameId.INVERTED_PENDULUM,
    GameId.INVERTED_DOUBLE_PENDULUM,
    GameId.REACHER,
    GameId.PUSHER,
    GameId.SWIMMER,
    GameId.PONG_NO_FRAMESKIP,
    GameId.BREAKOUT_NO_FRAMESKIP,
    GameId.PROCGEN_COINRUN,
    GameId.PROCGEN_MAZE,
)


def _format_family_label(family: EnvironmentFamily | None) -> str:
    if family is None:
        return "General"
    if family == EnvironmentFamily.OTHER:
        return "Other"
    return family.value.replace("_", " ").title()


def _build_environment_choices() -> tuple[tuple[str, str], ...]:
    supported: list[GameId] = []
    for game in GameId:
        family = ENVIRONMENT_FAMILY_BY_GAME.get(game)
        if family in _SUPPORTED_FAMILIES or game in _ADDITIONAL_SUPPORTED_GAMES:
            supported.append(game)

    ordered: list[GameId] = []
    for game in _PREFERRED_GAME_ORDER:
        if game in supported and game not in ordered:
            ordered.append(game)
    for game in sorted(supported, key=lambda g: g.value):
        if game not in ordered:
            ordered.append(game)

    choices: list[tuple[str, str]] = []
    for game in ordered:
        family = ENVIRONMENT_FAMILY_BY_GAME.get(game)
        label = f"{game.value} ({_format_family_label(family)})"
        # Special-case for Procgen and Atari where get_game_display_name already includes prefix
        if family == EnvironmentFamily.ATARI:
            label = f"{game.value} (Atari)"
        elif game in _ADDITIONAL_SUPPORTED_GAMES and family == EnvironmentFamily.OTHER:
            if "procgen" in game.value:
                label = f"{game.value} (Procgen)"
        choices.append((label, game.value))
    return tuple(choices)


_ENVIRONMENT_CHOICES: tuple[tuple[str, str], ...] = _build_environment_choices()


def _generate_run_id(prefix: str, algo: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = algo.replace("_", "-")
    return f"{prefix}-{slug}-{timestamp}"


@dataclass(frozen=True)
class _FormState:
    algo: str
    env_id: str
    total_timesteps: int
    seed: Optional[int]
    agent_id: Optional[str]
    worker_id: Optional[str]
    use_gpu: bool
    track_tensorboard: bool
    track_wandb: bool
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_run_name: Optional[str]
    wandb_api_key: Optional[str]
    wandb_email: Optional[str]
    wandb_http_proxy: Optional[str]
    wandb_https_proxy: Optional[str]
    use_wandb_vpn: bool
    notes: Optional[str]
    validate_only: bool
    algo_params: Dict[str, Any]


class CleanRlTrainForm(QtWidgets.QDialog):
    """Minimal training configuration dialog for CleanRL worker."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, default_game: Optional[GameId] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CleanRL Worker – Configure Training Run")
        self.setModal(True)
        self.resize(720, 420)

        self._last_config: Optional[Dict[str, Any]] = None
        self._last_validation_output: str = ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        intro = QtWidgets.QLabel(
            "Configure a CleanRL training run. Fields in the left column control the "
            "algorithm and rollout parameters, while the right column manages analytics "
            "exports and descriptive metadata."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        columns_widget = QtWidgets.QWidget(self)
        columns_layout = QtWidgets.QHBoxLayout(columns_widget)
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(16)

        left_container = QtWidgets.QWidget(columns_widget)
        left_column = QtWidgets.QVBoxLayout(left_container)
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(8)

        right_container = QtWidgets.QWidget(columns_widget)
        right_column = QtWidgets.QVBoxLayout(right_container)
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(8)

        columns_layout.addWidget(left_container, 1)
        columns_layout.addWidget(right_container, 1)
        layout.addWidget(columns_widget)

        self._algo_combo = QtWidgets.QComboBox(self)
        for algo in _DEFAULT_ALGOS:
            self._algo_combo.addItem(algo)
        self._algo_combo.setToolTip("Select the CleanRL algorithm entry point to invoke.")
        left_column.addWidget(self._labeled("Algorithm", self._algo_combo))

        env_widget = QtWidgets.QWidget(self)
        env_layout = QtWidgets.QHBoxLayout(env_widget)
        env_layout.setContentsMargins(0, 0, 0, 0)
        self._env_combo = QtWidgets.QComboBox(self)
        for label, env_id in _ENVIRONMENT_CHOICES:
            self._env_combo.addItem(label, env_id)
        env_layout.addWidget(self._env_combo, 1)
        self._custom_env_checkbox = QtWidgets.QCheckBox("Custom", self)
        self._custom_env_checkbox.setToolTip("Toggle to supply a Gymnasium environment id manually.")
        env_layout.addWidget(self._custom_env_checkbox)
        self._env_custom_input = QtWidgets.QLineEdit(self)
        self._env_custom_input.setPlaceholderText("e.g. procgen:procgen-bossfight-v0")
        self._env_custom_input.setEnabled(False)
        env_layout.addWidget(self._env_custom_input, 1)
        self._custom_env_checkbox.toggled.connect(self._on_custom_env_toggled)
        if default_game is not None:
            index = self._env_combo.findData(default_game.value)
            if index >= 0:
                self._env_combo.setCurrentIndex(index)
            else:
                self._custom_env_checkbox.setChecked(True)
                self._env_custom_input.setText(default_game.value)
        left_column.addWidget(self._labeled("Environment", env_widget))

        self._timesteps_spin = QtWidgets.QSpinBox(self)
        self._timesteps_spin.setRange(1_024, 1_000_000_000)
        self._timesteps_spin.setSingleStep(1_024)
        self._timesteps_spin.setValue(2048)
        self._timesteps_spin.setToolTip("Total timesteps (or frames) CleanRL will train before exiting.")
        left_column.addWidget(self._labeled("Total Timesteps", self._timesteps_spin))

        self._seed_spin = QtWidgets.QSpinBox(self)
        self._seed_spin.setRange(0, 1_000_000_000)
        self._seed_spin.setValue(1)
        self._seed_spin.setToolTip("Algorithm seed forwarded to CleanRL (use 0 to leave the seed unspecified).")
        left_column.addWidget(self._labeled("Seed (optional)", self._seed_spin))

        self._agent_id_input = QtWidgets.QLineEdit(self)
        self._agent_id_input.setPlaceholderText("Optional agent identifier")
        self._agent_id_input.setToolTip("Label used in analytics manifests and WANDB/TensorBoard tabs.")
        left_column.addWidget(self._labeled("Agent ID", self._agent_id_input))

        self._worker_id_input = QtWidgets.QLineEdit(self)
        self._worker_id_input.setPlaceholderText("Optional worker override (e.g. cleanrl-gpu-01)")
        self._worker_id_input.setToolTip("Override the worker id reported to the trainer daemon.")
        left_column.addWidget(self._labeled("Worker ID", self._worker_id_input))

        self._use_gpu_checkbox = QtWidgets.QCheckBox("Enable CUDA (GPU)", self)
        self._use_gpu_checkbox.setChecked(True)
        self._use_gpu_checkbox.setToolTip("Toggle CleanRL's --cuda flag; disable if the host lacks a GPU.")
        left_column.addWidget(self._labeled("GPU", self._use_gpu_checkbox))

        self._dry_run_checkbox = QtWidgets.QCheckBox("Validate only (skip training)", self)
        self._dry_run_checkbox.setToolTip(
            "When enabled the dialog performs the dry-run check but does not launch the training run."
        )
        self._dry_run_checkbox.setChecked(True)
        left_column.addWidget(self._dry_run_checkbox)

        help_box = QtWidgets.QGroupBox("Algorithm Notes", self)
        help_layout = QtWidgets.QVBoxLayout(help_box)
        help_layout.setContentsMargins(8, 8, 8, 8)

        self._algo_help_text = QtWidgets.QTextEdit(help_box)
        self._algo_help_text.setReadOnly(True)
        self._algo_help_text.setMinimumHeight(160)
        self._algo_help_text.setWordWrapMode(QtGui.QTextOption.WrapMode.WordWrap)
        help_layout.addWidget(self._algo_help_text)
        left_column.addWidget(help_box)
        left_column.addStretch(1)

        self._tensorboard_checkbox = QtWidgets.QCheckBox("Track TensorBoard", self)
        self._tensorboard_checkbox.setToolTip(
            "Write TensorBoard event files to var/trainer/runs/<run_id>/tensorboard."
        )
        right_column.addWidget(self._tensorboard_checkbox)

        self._track_wandb_checkbox = QtWidgets.QCheckBox("Track WANDB", self)
        self._track_wandb_checkbox.setToolTip("Requires wandb login on the trainer host.")
        self._track_wandb_checkbox.toggled.connect(self._on_track_wandb_toggled)
        right_column.addWidget(self._track_wandb_checkbox)

        wandb_container = QtWidgets.QWidget(self)
        wandb_layout = QtWidgets.QFormLayout(wandb_container)
        wandb_layout.setContentsMargins(0, 0, 0, 0)
        self._wandb_project_input = QtWidgets.QLineEdit(self)
        self._wandb_project_input.setPlaceholderText("e.g. MOSAIC")
        self._wandb_project_input.setToolTip("Project name inside wandb.ai where runs will be grouped.")
        wandb_layout.addRow("WANDB Project", self._wandb_project_input)
        self._wandb_entity_input = QtWidgets.QLineEdit(self)
        self._wandb_entity_input.setPlaceholderText("e.g. abdulhamid97mousa")
        self._wandb_entity_input.setToolTip("WANDB entity (team or user namespace) to publish to.")
        wandb_layout.addRow("WANDB Entity", self._wandb_entity_input)
        self._wandb_run_name_input = QtWidgets.QLineEdit(self)
        self._wandb_run_name_input.setPlaceholderText("Optional custom run name")
        wandb_layout.addRow("WANDB Run Name", self._wandb_run_name_input)
        self._wandb_api_key_input = QtWidgets.QLineEdit(self)
        self._wandb_api_key_input.setPlaceholderText("Optional API key override")
        self._wandb_api_key_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        wandb_layout.addRow("WANDB API Key", self._wandb_api_key_input)
        self._wandb_email_input = QtWidgets.QLineEdit(self)
        self._wandb_email_input.setPlaceholderText("Optional WANDB account email")
        wandb_layout.addRow("WANDB Email", self._wandb_email_input)
        self._wandb_use_vpn_checkbox = QtWidgets.QCheckBox("Route WANDB traffic through VPN proxy", self)
        self._wandb_use_vpn_checkbox.toggled.connect(self._on_wandb_vpn_toggled)
        wandb_layout.addRow("Use WANDB VPN", self._wandb_use_vpn_checkbox)
        self._wandb_http_proxy_input = QtWidgets.QLineEdit(self)
        self._wandb_http_proxy_input.setPlaceholderText("Optional HTTP proxy (e.g. http://127.0.0.1:7890)")
        wandb_layout.addRow("WANDB HTTP Proxy", self._wandb_http_proxy_input)
        self._wandb_https_proxy_input = QtWidgets.QLineEdit(self)
        self._wandb_https_proxy_input.setPlaceholderText("Optional HTTPS proxy (e.g. https://127.0.0.1:7890)")
        wandb_layout.addRow("WANDB HTTPS Proxy", self._wandb_https_proxy_input)
        right_column.addWidget(self._labeled("WANDB", wandb_container))

        self._notes_edit = QtWidgets.QPlainTextEdit(self)
        self._notes_edit.setPlaceholderText("Optional notes for analytics manifests and logs.")
        self._notes_edit.setToolTip("Notes appear alongside analytics artifacts for later reference.")
        right_column.addWidget(self._labeled("Notes", self._notes_edit))

        right_column.addStretch(1)

        self._update_wandb_controls()

        self._algo_param_group = QtWidgets.QGroupBox("Algorithm Parameters", self)
        self._algo_param_form = QtWidgets.QFormLayout(self._algo_param_group)
        layout.addWidget(self._algo_param_group)
        self._algo_param_inputs: Dict[str, QtWidgets.QWidget] = {}
        self._algo_combo.currentTextChanged.connect(self._on_algorithm_changed)
        self._on_algorithm_changed(self._algo_combo.currentText())

        self._validation_status_label = QtWidgets.QLabel(
            "Dry-run validation has not been executed yet."
        )
        self._validation_status_label.setStyleSheet("color: #666666;")
        layout.addWidget(self._validation_status_label)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        buttons.accepted.connect(self._handle_accept)
        buttons.rejected.connect(self.reject)
        validate_btn = buttons.addButton("Validate", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        if validate_btn is not None:
            validate_btn.clicked.connect(self._on_validate_clicked)
        layout.addWidget(buttons)

    def _labeled(self, label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget(self)
        form_layout = QtWidgets.QFormLayout(container)
        form_layout.addRow(label, widget)
        return container

    def _update_wandb_controls(self) -> None:
        track_enabled = self._track_wandb_checkbox.isChecked()
        base_fields = (
            self._wandb_project_input,
            self._wandb_entity_input,
            self._wandb_run_name_input,
            self._wandb_api_key_input,
            self._wandb_email_input,
        )
        for field in base_fields:
            field.setEnabled(track_enabled)
        self._wandb_use_vpn_checkbox.setEnabled(track_enabled)
        if not track_enabled:
            self._wandb_use_vpn_checkbox.setChecked(False)
        vpn_enabled = track_enabled and self._wandb_use_vpn_checkbox.isChecked()
        for field in (self._wandb_http_proxy_input, self._wandb_https_proxy_input):
            field.setEnabled(vpn_enabled)

    def _on_track_wandb_toggled(self, checked: bool) -> None:
        _ = checked
        self._update_wandb_controls()

    def _on_wandb_vpn_toggled(self, checked: bool) -> None:
        _ = checked
        self._update_wandb_controls()

    def _collect_state(self) -> _FormState:
        algo = self._algo_combo.currentText().strip()
        if self._custom_env_checkbox.isChecked():
            env_id = self._env_custom_input.text().strip()
        else:
            env_id = str(self._env_combo.currentData())
        seed_value = int(self._seed_spin.value())
        selected_seed: Optional[int] = seed_value if seed_value > 0 else None
        notes = self._notes_edit.toPlainText().strip() or None
        worker_id_value = self._worker_id_input.text().strip() or None
        agent_id_value = self._agent_id_input.text().strip() or None
        wandb_project = self._wandb_project_input.text().strip() or None
        wandb_entity = self._wandb_entity_input.text().strip() or None
        wandb_run_name = self._wandb_run_name_input.text().strip() or None
        wandb_api_key = self._wandb_api_key_input.text().strip() or None
        wandb_email = self._wandb_email_input.text().strip() or None
        use_wandb_vpn = self._wandb_use_vpn_checkbox.isChecked()
        raw_http_proxy = self._wandb_http_proxy_input.text().strip()
        raw_https_proxy = self._wandb_https_proxy_input.text().strip()
        wandb_http_proxy = None
        wandb_https_proxy = None
        if use_wandb_vpn:
            wandb_http_proxy = raw_http_proxy or os.environ.get("WANDB_VPN_HTTP_PROXY", "").strip() or None
            wandb_https_proxy = raw_https_proxy or os.environ.get("WANDB_VPN_HTTPS_PROXY", "").strip() or None

        algo_params: Dict[str, Any] = {}
        for key, widget in self._algo_param_inputs.items():
            if isinstance(widget, QtWidgets.QSpinBox):
                algo_params[key] = int(widget.value())
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                algo_params[key] = float(widget.value())
            elif isinstance(widget, QtWidgets.QCheckBox):
                algo_params[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QLineEdit):
                algo_params[key] = widget.text().strip()

        return _FormState(
            algo=algo,
            env_id=env_id,
            total_timesteps=int(self._timesteps_spin.value()),
            seed=selected_seed,
            agent_id=agent_id_value or None,
            worker_id=worker_id_value,
            use_gpu=self._use_gpu_checkbox.isChecked(),
            track_tensorboard=self._tensorboard_checkbox.isChecked(),
            track_wandb=self._track_wandb_checkbox.isChecked(),
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_api_key=wandb_api_key,
            wandb_email=wandb_email,
            wandb_http_proxy=wandb_http_proxy,
            wandb_https_proxy=wandb_https_proxy,
            use_wandb_vpn=use_wandb_vpn,
            notes=notes,
            validate_only=self._dry_run_checkbox.isChecked(),
            algo_params=algo_params,
        )

    def _build_config(self, state: _FormState, *, run_id: Optional[str] = None) -> Dict[str, Any]:
        run_id = run_id or _generate_run_id("cleanrl", state.algo)

        extras: Dict[str, Any] = {"cuda": state.use_gpu}
        if state.track_tensorboard:
            extras["tensorboard_dir"] = "tensorboard"
        if state.track_wandb:
            extras["track_wandb"] = True
        if state.wandb_project:
            extras["wandb_project_name"] = state.wandb_project
        if state.wandb_entity:
            extras["wandb_entity"] = state.wandb_entity
        if state.wandb_run_name:
            extras["wandb_run_name"] = state.wandb_run_name
        if state.wandb_email:
            extras["wandb_email"] = state.wandb_email
        if state.use_wandb_vpn:
            extras["wandb_use_vpn_proxy"] = True
        if state.wandb_api_key:
            extras["wandb_api_key"] = state.wandb_api_key
        if state.use_wandb_vpn and state.wandb_http_proxy:
            extras["wandb_http_proxy"] = state.wandb_http_proxy
        if state.use_wandb_vpn and state.wandb_https_proxy:
            extras["wandb_https_proxy"] = state.wandb_https_proxy
        if state.notes:
            extras["notes"] = state.notes
        if state.algo_params:
            extras["algo_params"] = state.algo_params

        worker_config: Dict[str, Any] = {
            "run_id": run_id,
            "algo": state.algo,
            "env_id": state.env_id,
            "total_timesteps": state.total_timesteps,
            "extras": extras,
        }
        if state.seed is not None:
            worker_config["seed"] = state.seed
        if state.worker_id:
            worker_config["worker_id"] = state.worker_id
        if state.agent_id:
            worker_config["agent_id"] = state.agent_id
            worker_config.setdefault("extras", extras).setdefault("agent_id", state.agent_id)

        arguments: list[str] = []
        if state.validate_only:
            arguments.extend(["--dry-run", "--emit-summary"])

        metadata = {
            "ui": {
                "worker_id": state.worker_id or "cleanrl_worker",
                "agent_id": state.agent_id or "cleanrl_agent",
                "algo": state.algo,
                "env_id": state.env_id,
                "dry_run": state.validate_only,
            },
            "worker": {
                "worker_id": state.worker_id or "cleanrl_worker",
                "module": "cleanrl_worker.cli",
                "use_grpc": True,
                "grpc_target": "127.0.0.1:50055",
                "arguments": arguments,
                "config": worker_config,
            },
        }

        tensorboard_relpath = None
        if state.track_tensorboard:
            tensorboard_relpath = f"var/trainer/runs/{run_id}/tensorboard"

        metadata["artifacts"] = {
            "tensorboard": {
                "enabled": state.track_tensorboard,
                "relative_path": tensorboard_relpath,
            },
            "wandb": {
                "enabled": state.track_wandb,
                "run_path": None,
                "use_vpn_proxy": state.use_wandb_vpn,
                "http_proxy": state.wandb_http_proxy,
                "https_proxy": state.wandb_https_proxy,
            },
            "notes": state.notes,
        }

        environment: Dict[str, Any] = {
            "CLEANRL_RUN_ID": run_id,
            "CLEANRL_AGENT_ID": state.agent_id or "cleanrl_agent",
            "TRACK_TENSORBOARD": "1" if state.track_tensorboard else "0",
            "TRACK_WANDB": "1" if state.track_wandb else "0",
        }
        if state.wandb_api_key:
            environment["WANDB_API_KEY"] = state.wandb_api_key
        if state.wandb_email:
            environment["WANDB_EMAIL"] = state.wandb_email
        if state.use_wandb_vpn and state.wandb_http_proxy:
            environment.update(
                {
                    "WANDB_HTTP_PROXY": state.wandb_http_proxy,
                    "HTTP_PROXY": state.wandb_http_proxy,
                    "http_proxy": state.wandb_http_proxy,
                }
            )
        if state.use_wandb_vpn and state.wandb_https_proxy:
            environment.update(
                {
                    "WANDB_HTTPS_PROXY": state.wandb_https_proxy,
                    "HTTPS_PROXY": state.wandb_https_proxy,
                    "https_proxy": state.wandb_https_proxy,
                }
            )

        config: Dict[str, Any] = {
            "run_name": run_id,
            "entry_point": sys.executable,
            "arguments": ["-m", "cleanrl_worker.cli"],
            "environment": environment,
            "resources": {
                "cpus": 4,
                "memory_mb": 4096,
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

    def _run_validation(
        self,
        state: _FormState,
        *,
        run_id: str,
        persist_config: bool,
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        config = self._build_config(state, run_id=run_id)
        self._validation_status_label.setText("Running CleanRL dry-run validation…")
        self._validation_status_label.setStyleSheet("color: #1565c0;")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            success, output = run_cleanrl_dry_run(config)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._set_validation_result(success, output)
        self._append_validation_notes(success, output)

        if persist_config and success:
            self._last_config = copy.deepcopy(config)
        elif not persist_config:
            self._last_config = None

        return success, (copy.deepcopy(config) if success else None)

    def _set_validation_result(self, success: bool, output: str) -> None:
        self._last_validation_output = output or ""
        if success:
            self._validation_status_label.setText("✔ Dry-run validation succeeded.")
            self._validation_status_label.setStyleSheet("color: #2e7d32;")
        else:
            self._validation_status_label.setText(
                "✖ Dry-run validation failed. Check the details returned by cleanrl_worker."
            )
            self._validation_status_label.setStyleSheet("color: #c62828;")

    def _append_validation_notes(self, success: bool, output: str) -> None:
        status = "SUCCESS" if success else "FAILED"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        details = output.strip() if output else ("Dry-run completed." if success else "Dry-run failed without output.")
        entry = (
            f"[Dry-Run {status} — {timestamp}]\n"
            f"{details}\n"
            f"{'-' * 40}\n"
        )
        self._notes_edit.appendPlainText(entry)

    def _on_algorithm_changed(self, algo: str) -> None:
        self._rebuild_algo_params(algo)
        self._algo_help_text.setHtml(get_algo_doc(algo))

    def _handle_accept(self) -> None:
        state = self._collect_state()
        if not state.algo:
            QtWidgets.QMessageBox.warning(
                self,
                "Algorithm Required",
                "Select a CleanRL algorithm before launching.",
            )
            return
        if not state.env_id:
            QtWidgets.QMessageBox.warning(
                self,
                "Environment Required",
                "Specify a Gymnasium environment id (e.g. CartPole-v1).",
            )
            return
        if self._custom_env_checkbox.isChecked() and not state.env_id:
            QtWidgets.QMessageBox.warning(
                self,
                "Custom Environment",
                "Provide an environment id when using custom mode.",
            )
            return
        run_id = _generate_run_id("cleanrl", state.algo)
        success, config = self._run_validation(state, run_id=run_id, persist_config=not state.validate_only)
        if not success:
            self._last_config = None
            return

        if state.validate_only:
            self._last_config = None
            return

        self._last_config = config
        self.accept()

    def _on_validate_clicked(self) -> None:
        state = self._collect_state()
        if not state.algo or not state.env_id:
            QtWidgets.QMessageBox.warning(
                self,
                "Incomplete Configuration",
                "Select an algorithm and environment before running validation.",
            )
            return
        run_id = _generate_run_id("cleanrl", state.algo)
        self._run_validation(state, run_id=run_id, persist_config=False)

    def get_config(self) -> Dict[str, Any]:
        """Return the trainer payload generated by the form."""

        if self._last_config is not None:
            return copy.deepcopy(self._last_config)

        state = self._collect_state()
        return self._build_config(state)

    def _on_custom_env_toggled(self, checked: bool) -> None:
        self._env_combo.setEnabled(not checked)
        self._env_custom_input.setEnabled(checked)
        if not checked:
            self._env_custom_input.clear()

    def _rebuild_algo_params(self, algo: str) -> None:
        while self._algo_param_form.rowCount():
            self._algo_param_form.removeRow(0)
        self._algo_param_inputs.clear()

        specs = _ALGO_PARAM_SPECS.get(algo, ())
        if not specs:
            self._algo_param_group.setVisible(False)
            return

        self._algo_param_group.setVisible(True)

        for spec in specs:
            if spec.field_type is int:
                spin = QtWidgets.QSpinBox(self)
                spin.setRange(-1_000_000_000, 1_000_000_000)
                spin.setValue(int(spec.default))
                if spec.tooltip:
                    spin.setToolTip(spec.tooltip)
                self._algo_param_form.addRow(spec.label + ":", spin)
                self._algo_param_inputs[spec.key] = spin
            elif spec.field_type is float:
                spin = QtWidgets.QDoubleSpinBox(self)
                spin.setDecimals(6)
                spin.setRange(-1e9, 1e9)
                spin.setSingleStep(abs(spec.default) / 10 if isinstance(spec.default, (int, float)) and spec.default else 0.1)
                spin.setValue(float(spec.default))
                if spec.tooltip:
                    spin.setToolTip(spec.tooltip)
                self._algo_param_form.addRow(spec.label + ":", spin)
                self._algo_param_inputs[spec.key] = spin
            elif spec.field_type is bool:
                checkbox = QtWidgets.QCheckBox(spec.label, self)
                checkbox.setChecked(bool(spec.default))
                if spec.tooltip:
                    checkbox.setToolTip(spec.tooltip)
                self._algo_param_form.addRow("", checkbox)
                self._algo_param_inputs[spec.key] = checkbox
            else:
                line = QtWidgets.QLineEdit(self)
                line.setText(str(spec.default))
                if spec.tooltip:
                    line.setToolTip(spec.tooltip)
                self._algo_param_form.addRow(spec.label + ":", line)
                self._algo_param_inputs[spec.key] = line


__all__ = ["CleanRlTrainForm"]


# Late import to avoid circular registration at module import time.
from gym_gui.ui.forms import get_worker_form_factory

_factory = get_worker_form_factory()
if not _factory.has_train_form("cleanrl_worker"):
    _factory.register_train_form(
        "cleanrl_worker",
        lambda parent=None, **kwargs: CleanRlTrainForm(parent=parent, **kwargs),
    )
