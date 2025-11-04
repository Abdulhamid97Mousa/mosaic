"""CleanRL worker training form."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from qtpy import QtCore, QtWidgets

from gym_gui.core.enums import (
    EnvironmentFamily,
    GameId,
    ENVIRONMENT_FAMILY_BY_GAME,
)


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
    ),
    "ppo_atari": (
        _AlgoParamSpec("learning_rate", "Learning Rate", 2.5e-4, float, "Adam optimizer learning rate"),
        _AlgoParamSpec("total_frames", "Total Frames", 10_000_000, int, "Total Atari frames to collect"),
    ),
    "dqn": (
        _AlgoParamSpec("learning_rate", "Learning Rate", 1e-4, float, "Adam optimizer learning rate"),
        _AlgoParamSpec("batch_size", "Batch Size", 128, int, "Mini-batch size from replay buffer"),
        _AlgoParamSpec("buffer_size", "Replay Buffer", 100_000, int, "Maximum replay buffer size"),
    ),
}


_SUPPORTED_FAMILIES: set[EnvironmentFamily] = {
    EnvironmentFamily.CLASSIC_CONTROL,
    EnvironmentFamily.BOX2D,
    EnvironmentFamily.MUJOCO,
}

_ADDITIONAL_SUPPORTED_GAMES: set[GameId] = {
    GameId.PONG_NO_FRAMESKIP,
    GameId.BREAKOUT_NO_FRAMESKIP,
    GameId.PROCGEN_COINRUN,
    GameId.PROCGEN_MAZE,
}

_PREFERRED_GAME_ORDER: Sequence[GameId] = (
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
    notes: Optional[str]
    dry_run: bool
    algo_params: Dict[str, Any]


class CleanRlTrainForm(QtWidgets.QDialog):
    """Minimal training configuration dialog for CleanRL worker."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, default_game: Optional[GameId] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CleanRL Worker – Configure Training Run")
        self.setModal(True)
        self.resize(420, 360)

        layout = QtWidgets.QVBoxLayout(self)

        self._algo_combo = QtWidgets.QComboBox(self)
        for algo in _DEFAULT_ALGOS:
            self._algo_combo.addItem(algo)
        layout.addWidget(self._labeled("Algorithm", self._algo_combo))

        env_widget = QtWidgets.QWidget(self)
        env_layout = QtWidgets.QHBoxLayout(env_widget)
        env_layout.setContentsMargins(0, 0, 0, 0)
        self._env_combo = QtWidgets.QComboBox(self)
        for label, env_id in _ENVIRONMENT_CHOICES:
            self._env_combo.addItem(label, env_id)
        env_layout.addWidget(self._env_combo, 1)
        self._custom_env_checkbox = QtWidgets.QCheckBox("Custom", self)
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
        layout.addWidget(self._labeled("Environment", env_widget))

        self._timesteps_spin = QtWidgets.QSpinBox(self)
        self._timesteps_spin.setRange(1_024, 1_000_000_000)
        self._timesteps_spin.setSingleStep(1_024)
        self._timesteps_spin.setValue(2048)
        self._timesteps_spin.setSuffix(" steps")
        self._timesteps_spin.setToolTip("Total timesteps (or frames) CleanRL will train before exiting")
        layout.addWidget(self._labeled("Total Timesteps", self._timesteps_spin))

        self._seed_spin = QtWidgets.QSpinBox(self)
        self._seed_spin.setRange(0, 1_000_000_000)
        self._seed_spin.setValue(0)
        self._seed_spin.setToolTip("Seed forwarded to CleanRL (0 = leave unspecified)")
        layout.addWidget(self._labeled("Seed (optional)", self._seed_spin))

        self._agent_id_input = QtWidgets.QLineEdit(self)
        self._agent_id_input.setPlaceholderText("Optional agent identifier")
        layout.addWidget(self._labeled("Agent ID", self._agent_id_input))

        self._worker_id_input = QtWidgets.QLineEdit(self)
        self._worker_id_input.setPlaceholderText("Optional worker override (e.g. cleanrl-gpu-01)")
        layout.addWidget(self._labeled("Worker ID", self._worker_id_input))

        self._use_gpu_checkbox = QtWidgets.QCheckBox("Enable CUDA (GPU)", self)
        self._use_gpu_checkbox.setChecked(True)
        self._use_gpu_checkbox.setToolTip("Toggle CleanRL's --cuda flag; disable if the host lacks a GPU.")
        layout.addWidget(self._labeled("GPU", self._use_gpu_checkbox))

        self._tensorboard_checkbox = QtWidgets.QCheckBox("Track TensorBoard", self)
        self._tensorboard_checkbox.setToolTip("Write TensorBoard event files to var/trainer/runs/<run_id>/tensorboard")
        layout.addWidget(self._tensorboard_checkbox)

        self._track_wandb_checkbox = QtWidgets.QCheckBox("Track Weights & Biases", self)
        self._track_wandb_checkbox.setToolTip("Requires wandb login on the trainer host")
        layout.addWidget(self._track_wandb_checkbox)
        wandb_container = QtWidgets.QWidget(self)
        wandb_layout = QtWidgets.QFormLayout(wandb_container)
        wandb_layout.setContentsMargins(0, 0, 0, 0)
        self._wandb_project_input = QtWidgets.QLineEdit(self)
        self._wandb_project_input.setPlaceholderText("e.g. MOSAIC")
        wandb_layout.addRow("W&B Project", self._wandb_project_input)
        self._wandb_entity_input = QtWidgets.QLineEdit(self)
        self._wandb_entity_input.setPlaceholderText("e.g. abdulhamid97mousa")
        wandb_layout.addRow("W&B Entity", self._wandb_entity_input)
        self._wandb_run_name_input = QtWidgets.QLineEdit(self)
        self._wandb_run_name_input.setPlaceholderText("Optional custom run name")
        wandb_layout.addRow("W&B Run Name", self._wandb_run_name_input)
        self._wandb_api_key_input = QtWidgets.QLineEdit(self)
        self._wandb_api_key_input.setPlaceholderText("Optional API key override")
        self._wandb_api_key_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        wandb_layout.addRow("W&B API Key", self._wandb_api_key_input)
        self._wandb_email_input = QtWidgets.QLineEdit(self)
        self._wandb_email_input.setPlaceholderText("Optional W&B account email")
        wandb_layout.addRow("W&B Email", self._wandb_email_input)
        layout.addWidget(self._labeled("Weights & Biases", wandb_container))

        self._notes_edit = QtWidgets.QPlainTextEdit(self)
        self._notes_edit.setPlaceholderText("Optional notes for analytics manifest.")
        layout.addWidget(self._labeled("Notes", self._notes_edit))

        self._dry_run_checkbox = QtWidgets.QCheckBox("Dry run only (validate configuration)", self)
        self._dry_run_checkbox.setChecked(True)
        layout.addWidget(self._dry_run_checkbox)

        self._algo_param_group = QtWidgets.QGroupBox("Algorithm Parameters", self)
        self._algo_param_form = QtWidgets.QFormLayout(self._algo_param_group)
        layout.addWidget(self._algo_param_group)
        self._algo_param_inputs: Dict[str, QtWidgets.QWidget] = {}
        self._algo_combo.currentTextChanged.connect(self._rebuild_algo_params)
        self._rebuild_algo_params(self._algo_combo.currentText())

        layout.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        buttons.accepted.connect(self._handle_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _labeled(self, label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget(self)
        form_layout = QtWidgets.QFormLayout(container)
        form_layout.addRow(label, widget)
        return container

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
            notes=notes,
            dry_run=self._dry_run_checkbox.isChecked(),
            algo_params=algo_params,
        )

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
        self.accept()

    def get_config(self) -> Dict[str, Any]:
        """Build the trainer payload from the form state."""

        state = self._collect_state()
        run_id = _generate_run_id("cleanrl", state.algo)

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
        if state.dry_run:
            arguments.extend(["--dry-run", "--emit-summary"])

        metadata = {
            "ui": {
                "worker_id": state.worker_id or "cleanrl_worker",
                "agent_id": state.agent_id or "cleanrl_agent",
                "algo": state.algo,
                "env_id": state.env_id,
                "dry_run": state.dry_run,
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
            },
            "notes": state.notes,
        }

        config: Dict[str, Any] = {
            "run_name": run_id,
            "entry_point": sys.executable,
            "arguments": ["-m", "cleanrl_worker.cli"],
            "environment": {
                "CLEANRL_RUN_ID": run_id,
                "CLEANRL_AGENT_ID": state.agent_id or "cleanrl_agent",
                "TRACK_TENSORBOARD": "1" if state.track_tensorboard else "0",
                "TRACK_WANDB": "1" if state.track_wandb else "0",
                **({"WANDB_API_KEY": state.wandb_api_key} if state.wandb_api_key else {}),
                **({"WANDB_EMAIL": state.wandb_email} if state.wandb_email else {}),
            },
            "resources": {
                "cpus": 4,
                "memory_mb": 4096,
                "gpus": {"requested": 0, "mandatory": False},
            },
            "metadata": metadata,
        }

        return config

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
