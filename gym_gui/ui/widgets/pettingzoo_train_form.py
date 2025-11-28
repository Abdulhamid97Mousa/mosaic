"""PettingZoo worker training form.

This form provides configuration options for training multi-agent policies
on PettingZoo environments. Supports both AEC (turn-based) and Parallel
(simultaneous) environments across all PettingZoo families.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtWidgets

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
from gym_gui.ui.forms import get_worker_form_factory

_LOGGER = logging.getLogger(__name__)


# Supported training algorithms for multi-agent RL
_MARL_ALGORITHMS = (
    ("PPO", "Proximal Policy Optimization - works well for most environments"),
    ("A2C", "Advantage Actor-Critic - synchronous updates"),
    ("DQN", "Deep Q-Network - for discrete action spaces"),
    ("MAPPO", "Multi-Agent PPO - parameter sharing for cooperative tasks"),
    ("QMIX", "Q-Mixing Network - value decomposition for cooperative tasks"),
    ("VDN", "Value Decomposition Network - simple value factorization"),
    ("IQL", "Independent Q-Learning - each agent learns independently"),
)


class PettingZooTrainForm(QtWidgets.QDialog):
    """Training configuration form for PettingZoo multi-agent environments.

    Provides UI for:
    - Environment family and game selection
    - Training algorithm selection
    - Hyperparameter configuration
    - Resource allocation (CPU, GPU, memory)
    - Logging and checkpoint settings
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        initial_env_id: Optional[str] = None,
        initial_family: Optional[str] = None,
    ) -> None:
        """Initialize the training form.

        Args:
            parent: Parent widget
            initial_env_id: Pre-selected environment ID
            initial_family: Pre-selected environment family
        """
        super().__init__(parent)
        self.setWindowTitle("PettingZoo Training Configuration")
        self.setMinimumSize(600, 700)

        self._initial_env_id = initial_env_id
        self._initial_family = initial_family
        self._last_config: Optional[Dict[str, Any]] = None

        self._build_ui()
        self._connect_signals()
        self._populate_families()

        # Apply initial selections if provided
        if initial_family:
            self._select_family(initial_family)
        if initial_env_id:
            self._select_env(initial_env_id)

    def _build_ui(self) -> None:
        """Build the form UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Scroll area for main content
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(16)

        # Environment Selection Group
        content_layout.addWidget(self._create_env_group())

        # Algorithm Group
        content_layout.addWidget(self._create_algorithm_group())

        # Training Parameters Group
        content_layout.addWidget(self._create_training_params_group())

        # Resources Group
        content_layout.addWidget(self._create_resources_group())

        # Logging Group
        content_layout.addWidget(self._create_logging_group())

        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        # Dialog buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_env_group(self) -> QtWidgets.QGroupBox:
        """Create environment selection group."""
        group = QtWidgets.QGroupBox("Environment Selection", self)
        layout = QtWidgets.QFormLayout(group)
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        # Family dropdown
        self._family_combo = QtWidgets.QComboBox(group)
        self._family_combo.setMinimumWidth(250)
        layout.addRow("Family:", self._family_combo)

        # Environment dropdown
        self._env_combo = QtWidgets.QComboBox(group)
        self._env_combo.setMinimumWidth(250)
        layout.addRow("Environment:", self._env_combo)

        # Environment info
        self._env_info_label = QtWidgets.QLabel("", group)
        self._env_info_label.setWordWrap(True)
        self._env_info_label.setStyleSheet(
            "color: #666; font-size: 11px; padding: 4px;"
        )
        layout.addRow("", self._env_info_label)

        # API type indicator
        self._api_type_label = QtWidgets.QLabel("", group)
        self._api_type_label.setStyleSheet("font-weight: bold;")
        layout.addRow("API Type:", self._api_type_label)

        return group

    def _create_algorithm_group(self) -> QtWidgets.QGroupBox:
        """Create algorithm selection group."""
        group = QtWidgets.QGroupBox("Training Algorithm", self)
        layout = QtWidgets.QFormLayout(group)

        # Algorithm dropdown
        self._algo_combo = QtWidgets.QComboBox(group)
        for algo_name, algo_desc in _MARL_ALGORITHMS:
            self._algo_combo.addItem(f"{algo_name} - {algo_desc}", algo_name)
        layout.addRow("Algorithm:", self._algo_combo)

        # Shared parameters checkbox
        self._shared_params_checkbox = QtWidgets.QCheckBox(
            "Share parameters across agents (cooperative)", group
        )
        self._shared_params_checkbox.setChecked(True)
        self._shared_params_checkbox.setToolTip(
            "When enabled, all agents share the same neural network weights. "
            "Recommended for cooperative environments."
        )
        layout.addRow("", self._shared_params_checkbox)

        return group

    def _create_training_params_group(self) -> QtWidgets.QGroupBox:
        """Create training parameters group."""
        group = QtWidgets.QGroupBox("Training Parameters", self)
        layout = QtWidgets.QGridLayout(group)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)

        # Total timesteps
        layout.addWidget(QtWidgets.QLabel("Total Timesteps:"), 0, 0)
        self._timesteps_spin = QtWidgets.QSpinBox(group)
        self._timesteps_spin.setRange(1000, 100_000_000)
        self._timesteps_spin.setValue(100_000)
        self._timesteps_spin.setSingleStep(10000)
        layout.addWidget(self._timesteps_spin, 0, 1)

        # Learning rate
        layout.addWidget(QtWidgets.QLabel("Learning Rate:"), 0, 2)
        self._lr_spin = QtWidgets.QDoubleSpinBox(group)
        self._lr_spin.setDecimals(6)
        self._lr_spin.setRange(1e-8, 1.0)
        self._lr_spin.setValue(3e-4)
        self._lr_spin.setSingleStep(1e-5)
        layout.addWidget(self._lr_spin, 0, 3)

        # Batch size
        layout.addWidget(QtWidgets.QLabel("Batch Size:"), 1, 0)
        self._batch_spin = QtWidgets.QSpinBox(group)
        self._batch_spin.setRange(8, 65536)
        self._batch_spin.setValue(256)
        self._batch_spin.setSingleStep(32)
        layout.addWidget(self._batch_spin, 1, 1)

        # Gamma (discount factor)
        layout.addWidget(QtWidgets.QLabel("Discount (gamma):"), 1, 2)
        self._gamma_spin = QtWidgets.QDoubleSpinBox(group)
        self._gamma_spin.setDecimals(4)
        self._gamma_spin.setRange(0.0, 1.0)
        self._gamma_spin.setValue(0.99)
        self._gamma_spin.setSingleStep(0.01)
        layout.addWidget(self._gamma_spin, 1, 3)

        # Seed
        layout.addWidget(QtWidgets.QLabel("Random Seed:"), 2, 0)
        self._seed_spin = QtWidgets.QSpinBox(group)
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        self._seed_spin.setSpecialValueText("Random")
        layout.addWidget(self._seed_spin, 2, 1)

        # Max cycles (for parallel envs)
        layout.addWidget(QtWidgets.QLabel("Max Cycles:"), 2, 2)
        self._max_cycles_spin = QtWidgets.QSpinBox(group)
        self._max_cycles_spin.setRange(100, 100000)
        self._max_cycles_spin.setValue(500)
        self._max_cycles_spin.setSingleStep(100)
        self._max_cycles_spin.setToolTip(
            "Maximum steps per episode (for parallel environments)"
        )
        layout.addWidget(self._max_cycles_spin, 2, 3)

        return group

    def _create_resources_group(self) -> QtWidgets.QGroupBox:
        """Create resources configuration group."""
        group = QtWidgets.QGroupBox("Resources", self)
        layout = QtWidgets.QGridLayout(group)

        # CPUs
        layout.addWidget(QtWidgets.QLabel("CPUs:"), 0, 0)
        self._cpu_spin = QtWidgets.QSpinBox(group)
        self._cpu_spin.setRange(1, 64)
        self._cpu_spin.setValue(4)
        layout.addWidget(self._cpu_spin, 0, 1)

        # Memory
        layout.addWidget(QtWidgets.QLabel("Memory (MB):"), 0, 2)
        self._memory_spin = QtWidgets.QSpinBox(group)
        self._memory_spin.setRange(512, 65536)
        self._memory_spin.setValue(4096)
        self._memory_spin.setSingleStep(512)
        layout.addWidget(self._memory_spin, 0, 3)

        # GPU
        self._gpu_checkbox = QtWidgets.QCheckBox("Use GPU", group)
        self._gpu_checkbox.setChecked(False)
        layout.addWidget(self._gpu_checkbox, 1, 0, 1, 2)

        return group

    def _create_logging_group(self) -> QtWidgets.QGroupBox:
        """Create logging configuration group."""
        group = QtWidgets.QGroupBox("Logging & Checkpoints", self)
        layout = QtWidgets.QVBoxLayout(group)

        # Checkboxes in horizontal layout
        checkbox_layout = QtWidgets.QHBoxLayout()

        self._tensorboard_checkbox = QtWidgets.QCheckBox("TensorBoard", group)
        self._tensorboard_checkbox.setChecked(True)
        checkbox_layout.addWidget(self._tensorboard_checkbox)

        self._wandb_checkbox = QtWidgets.QCheckBox("Weights & Biases", group)
        self._wandb_checkbox.setChecked(False)
        checkbox_layout.addWidget(self._wandb_checkbox)

        self._save_model_checkbox = QtWidgets.QCheckBox("Save Model", group)
        self._save_model_checkbox.setChecked(True)
        checkbox_layout.addWidget(self._save_model_checkbox)

        self._render_checkbox = QtWidgets.QCheckBox("Render Video", group)
        self._render_checkbox.setChecked(False)
        checkbox_layout.addWidget(self._render_checkbox)

        checkbox_layout.addStretch(1)
        layout.addLayout(checkbox_layout)

        # Log interval
        interval_layout = QtWidgets.QHBoxLayout()
        interval_layout.addWidget(QtWidgets.QLabel("Log Interval:"))
        self._log_interval_spin = QtWidgets.QSpinBox(group)
        self._log_interval_spin.setRange(1, 10000)
        self._log_interval_spin.setValue(100)
        interval_layout.addWidget(self._log_interval_spin)
        interval_layout.addStretch(1)
        layout.addLayout(interval_layout)

        return group

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)

    def _populate_families(self) -> None:
        """Populate the family dropdown."""
        self._family_combo.clear()
        self._family_combo.addItem("Classic (Turn-based)", PettingZooFamily.CLASSIC.value)
        self._family_combo.addItem("MPE (Particle)", PettingZooFamily.MPE.value)
        self._family_combo.addItem("SISL (Continuous)", PettingZooFamily.SISL.value)
        self._family_combo.addItem("Butterfly (Visual)", PettingZooFamily.BUTTERFLY.value)
        self._family_combo.addItem("Atari (2-Player)", PettingZooFamily.ATARI.value)

    def _select_family(self, family: str) -> None:
        """Select a family by value."""
        for i in range(self._family_combo.count()):
            if self._family_combo.itemData(i) == family:
                self._family_combo.setCurrentIndex(i)
                break

    def _select_env(self, env_id: str) -> None:
        """Select an environment by ID."""
        for i in range(self._env_combo.count()):
            if self._env_combo.itemData(i) == env_id:
                self._env_combo.setCurrentIndex(i)
                break

    def _on_family_changed(self, index: int) -> None:
        """Handle family selection change."""
        family_value = self._family_combo.currentData()
        if not family_value:
            return

        try:
            family = PettingZooFamily(family_value)
        except ValueError:
            return

        # Populate environments
        self._env_combo.clear()
        envs = get_envs_by_family(family)

        for env_id in envs:
            display_name = get_display_name(env_id)
            self._env_combo.addItem(display_name, env_id.value)

        if envs:
            self._env_combo.setCurrentIndex(0)
            self._on_env_changed(0)

    def _on_env_changed(self, index: int) -> None:
        """Handle environment selection change."""
        env_value = self._env_combo.currentData()
        if not env_value:
            self._env_info_label.setText("")
            self._api_type_label.setText("")
            return

        try:
            env_id = PettingZooEnvId(env_value)
        except ValueError:
            return

        # Update description
        description = get_description(env_id)
        self._env_info_label.setText(description)

        # Update API type
        api_type = get_api_type(env_id)
        if api_type == PettingZooAPIType.AEC:
            self._api_type_label.setText("AEC (Turn-based)")
            self._api_type_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        else:
            self._api_type_label.setText("Parallel (Simultaneous)")
            self._api_type_label.setStyleSheet("font-weight: bold; color: #4CAF50;")

    def _on_accept(self) -> None:
        """Handle OK button click."""
        self._last_config = self._build_config()
        self.accept()

    def _build_config(self) -> Dict[str, Any]:
        """Build the training configuration dictionary."""
        env_id = self._env_combo.currentData() or "tictactoe_v3"
        family = self._family_combo.currentData() or "classic"
        algorithm = self._algo_combo.currentData() or "PPO"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"pettingzoo-{env_id.replace('_', '-')}-{timestamp}"

        # Determine API type
        is_parallel = False
        try:
            pz_env_id = PettingZooEnvId(env_id)
            is_parallel = not is_aec_env(pz_env_id)
        except ValueError:
            pass

        config = {
            "run_name": run_name,
            "entry_point": "python",
            "arguments": ["-m", "pettingzoo_worker.cli", "train"],
            "environment": {
                "PETTINGZOO_ENV_ID": env_id,
                "PETTINGZOO_FAMILY": family,
                "PETTINGZOO_API_TYPE": "parallel" if is_parallel else "aec",
                "ALGORITHM": algorithm,
                "TOTAL_TIMESTEPS": str(self._timesteps_spin.value()),
                "LEARNING_RATE": str(self._lr_spin.value()),
                "BATCH_SIZE": str(self._batch_spin.value()),
                "GAMMA": str(self._gamma_spin.value()),
                "SEED": str(self._seed_spin.value()) if self._seed_spin.value() > 0 else "",
                "MAX_CYCLES": str(self._max_cycles_spin.value()),
                "SHARED_PARAMS": "1" if self._shared_params_checkbox.isChecked() else "0",
                "RENDER_MODE": "rgb_array" if self._render_checkbox.isChecked() else "none",
            },
            "resources": {
                "cpus": self._cpu_spin.value(),
                "memory_mb": self._memory_spin.value(),
                "gpus": {
                    "requested": 1 if self._gpu_checkbox.isChecked() else 0,
                    "mandatory": False,
                },
            },
            "artifacts": {
                "output_prefix": f"runs/{run_name}",
                "persist_logs": True,
                "keep_checkpoints": self._save_model_checkbox.isChecked(),
            },
            "metadata": {
                "ui": {
                    "worker_id": "pettingzoo_worker",
                    "env_id": env_id,
                    "family": family,
                    "algorithm": algorithm,
                    "is_parallel": is_parallel,
                    "mode": "training",
                },
                "worker": {
                    "module": "pettingzoo_worker.cli",
                    "use_grpc": True,
                    "grpc_target": "127.0.0.1:50055",
                    "config": {
                        "algorithm": algorithm,
                        "total_timesteps": self._timesteps_spin.value(),
                        "learning_rate": self._lr_spin.value(),
                        "batch_size": self._batch_spin.value(),
                        "gamma": self._gamma_spin.value(),
                        "seed": self._seed_spin.value() if self._seed_spin.value() > 0 else None,
                        "max_cycles": self._max_cycles_spin.value(),
                        "shared_params": self._shared_params_checkbox.isChecked(),
                        "render": self._render_checkbox.isChecked(),
                        "save_model": self._save_model_checkbox.isChecked(),
                        "log_interval": self._log_interval_spin.value(),
                    },
                },
                "artifacts": {
                    "tensorboard": {
                        "enabled": self._tensorboard_checkbox.isChecked(),
                        "log_dir": f"runs/{run_name}/tensorboard",
                    },
                    "wandb": {
                        "enabled": self._wandb_checkbox.isChecked(),
                        "project": "pettingzoo-training",
                    },
                },
            },
        }

        _LOGGER.info(
            "Built PettingZoo training config: env=%s, algo=%s, timesteps=%d",
            env_id,
            algorithm,
            self._timesteps_spin.value(),
        )

        return config

    def get_config(self) -> Dict[str, Any]:
        """Return the training configuration.

        Returns:
            dict: Configuration dictionary for TrainerClient submission
        """
        if self._last_config is not None:
            return copy.deepcopy(self._last_config)
        return self._build_config()


# Register form with factory at module load
_factory = get_worker_form_factory()
if not _factory.has_train_form("pettingzoo_worker"):
    _factory.register_train_form(
        "pettingzoo_worker",
        lambda parent=None, **kwargs: PettingZooTrainForm(parent=parent, **kwargs),
    )


__all__ = ["PettingZooTrainForm"]
