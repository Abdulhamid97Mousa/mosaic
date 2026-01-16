"""MCTX Worker training form for Monte Carlo Tree Search on board games.

This form provides configuration options for training MCTS-based policies
using mctx + PGX for turn-based games like Chess, Go, Connect Four, etc.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtGui, QtWidgets


_LOGGER = logging.getLogger(__name__)

# PGX Environment definitions
_PGX_ENVIRONMENTS = {
    "Board Games": [
        ("chess", "Chess", "Classic chess - 64 squares, full rules"),
        ("go_9x9", "Go 9x9", "Go on 9x9 board - faster training"),
        ("go_19x19", "Go 19x19", "Full-size Go board - very complex"),
        ("shogi", "Shogi", "Japanese chess with drops"),
        ("othello", "Othello", "Reversi - flip opponent pieces"),
        ("backgammon", "Backgammon", "Dice-based racing game"),
    ],
    "Simple Games": [
        ("connect_four", "Connect Four", "4-in-a-row on 7x6 grid"),
        ("tic_tac_toe", "Tic-Tac-Toe", "3-in-a-row on 3x3 grid"),
        ("hex", "Hex", "Connection game on hexagonal grid"),
    ],
    "Card Games": [
        ("kuhn_poker", "Kuhn Poker", "Simplified 3-card poker"),
        ("leduc_holdem", "Leduc Hold'em", "Simplified Texas Hold'em"),
    ],
    "Single Player": [
        ("2048", "2048", "Sliding tile puzzle game"),
        ("minatar-asterix", "MinAtar Asterix", "Simplified Atari Asterix"),
        ("minatar-breakout", "MinAtar Breakout", "Simplified Atari Breakout"),
        ("minatar-freeway", "MinAtar Freeway", "Simplified Atari Freeway"),
        ("minatar-seaquest", "MinAtar Seaquest", "Simplified Atari Seaquest"),
        ("minatar-space_invaders", "MinAtar Space Invaders", "Simplified Atari Space Invaders"),
    ],
}

# MCTS Algorithm definitions
_MCTX_ALGORITHMS = {
    "gumbel_muzero": {
        "name": "Gumbel MuZero",
        "description": (
            "<b>Gumbel MuZero</b><br><br>"
            "State-of-the-art MCTS algorithm combining MuZero with Gumbel sampling. "
            "Uses Sequential Halving for efficient action selection with fewer simulations.<br><br>"
            "<b>Best for:</b> Complex games like Chess, Go<br>"
            "<b>Strengths:</b> Sample efficient, state-of-the-art performance<br>"
            "<b>Recommended simulations:</b> 50-800 per move"
        ),
    },
    "alphazero": {
        "name": "AlphaZero",
        "description": (
            "<b>AlphaZero</b><br><br>"
            "Classic MCTS with learned policy/value networks. Uses PUCT for tree search "
            "and Dirichlet noise for exploration at root.<br><br>"
            "<b>Best for:</b> All turn-based games<br>"
            "<b>Strengths:</b> Well-understood, stable training<br>"
            "<b>Recommended simulations:</b> 400-1600 per move"
        ),
    },
    "muzero": {
        "name": "MuZero",
        "description": (
            "<b>MuZero</b><br><br>"
            "Model-based MCTS that learns environment dynamics. Can work without "
            "knowing game rules, learns internal representation.<br><br>"
            "<b>Best for:</b> Complex games, unknown dynamics<br>"
            "<b>Strengths:</b> No need for perfect simulator<br>"
            "<b>Recommended simulations:</b> 200-800 per move"
        ),
    },
    "stochastic_muzero": {
        "name": "Stochastic MuZero",
        "description": (
            "<b>Stochastic MuZero</b><br><br>"
            "MuZero variant for stochastic environments with chance nodes. "
            "Handles games with random elements like dice or card draws.<br><br>"
            "<b>Best for:</b> Backgammon, poker variants<br>"
            "<b>Strengths:</b> Handles stochasticity properly<br>"
            "<b>Recommended simulations:</b> 200-800 per move"
        ),
    },
}

# Recommended settings per game
_GAME_RECOMMENDATIONS = {
    "chess": {"num_simulations": 800, "num_res_blocks": 8, "channels": 128},
    "go_9x9": {"num_simulations": 400, "num_res_blocks": 6, "channels": 64},
    "go_19x19": {"num_simulations": 1600, "num_res_blocks": 20, "channels": 256},
    "connect_four": {"num_simulations": 100, "num_res_blocks": 4, "channels": 32},
    "tic_tac_toe": {"num_simulations": 50, "num_res_blocks": 2, "channels": 16},
    "othello": {"num_simulations": 200, "num_res_blocks": 6, "channels": 64},
    "hex": {"num_simulations": 200, "num_res_blocks": 6, "channels": 64},
    "backgammon": {"num_simulations": 200, "num_res_blocks": 4, "channels": 64},
    "kuhn_poker": {"num_simulations": 50, "num_res_blocks": 2, "channels": 16},
    "leduc_holdem": {"num_simulations": 100, "num_res_blocks": 4, "channels": 32},
    "2048": {"num_simulations": 100, "num_res_blocks": 4, "channels": 32},
}


def _generate_run_id(env_id: str, algo: str) -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"mctx-{env_id}-{algo}-{timestamp}"


class MCTXTrainForm(QtWidgets.QDialog):
    """Training configuration dialog for MCTX worker."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        default_env_id: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("MCTX Training Configuration")
        self.setModal(True)
        self.resize(800, 700)

        self._last_config: Optional[Dict[str, Any]] = None

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(12)

        # Scroll area for form content
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        main_layout.addWidget(scroll, 1)

        # Form container
        form_widget = QtWidgets.QWidget()
        form_layout = QtWidgets.QVBoxLayout(form_widget)
        form_layout.setSpacing(12)
        scroll.setWidget(form_widget)

        # Introduction
        intro = QtWidgets.QLabel(
            "Configure MCTS-based training for turn-based games using JAX-accelerated "
            "Monte Carlo Tree Search (mctx) with PGX environments."
        )
        intro.setWordWrap(True)
        form_layout.addWidget(intro)

        # Add form groups
        form_layout.addWidget(self._create_environment_group(default_env_id))
        form_layout.addWidget(self._create_algorithm_group())
        form_layout.addWidget(self._create_mcts_params_group())
        form_layout.addWidget(self._create_network_params_group())
        form_layout.addWidget(self._create_training_params_group())
        form_layout.addWidget(self._create_logging_group())
        form_layout.addWidget(self._create_checkpoint_group())

        form_layout.addStretch(1)

        # Button row
        button_layout = QtWidgets.QHBoxLayout()

        # Progress indicator (hidden by default)
        self._progress_label = QtWidgets.QLabel("")
        self._progress_label.setStyleSheet("color: #666; font-style: italic;")
        button_layout.addWidget(self._progress_label)

        button_layout.addStretch(1)

        # Buttons
        self._validate_button = QtWidgets.QPushButton("Validate")
        self._validate_button.setToolTip("Check if JAX and dependencies are available")
        self._validate_button.clicked.connect(self._on_validate_clicked)
        button_layout.addWidget(self._validate_button)

        self._cancel_button = QtWidgets.QPushButton("Cancel")
        self._cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_button)

        self._start_button = QtWidgets.QPushButton("Start Training")
        self._start_button.setDefault(True)
        self._start_button.clicked.connect(self._on_start_clicked)
        button_layout.addWidget(self._start_button)

        main_layout.addLayout(button_layout)

        # Connect signals
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)
        self._algo_combo.currentIndexChanged.connect(self._on_algo_changed)

        # Initialize
        self._update_algo_notes()
        if default_env_id:
            self._select_env_by_id(default_env_id)
        self._apply_game_recommendations()

    def _create_environment_group(self, default_env_id: Optional[str]) -> QtWidgets.QGroupBox:
        """Create environment selection group."""
        group = QtWidgets.QGroupBox("Environment (PGX)", self)
        layout = QtWidgets.QFormLayout(group)

        # Family selector
        self._family_combo = QtWidgets.QComboBox(group)
        for family in _PGX_ENVIRONMENTS.keys():
            self._family_combo.addItem(family)
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        layout.addRow("Category:", self._family_combo)

        # Environment selector
        self._env_combo = QtWidgets.QComboBox(group)
        self._populate_env_combo()
        layout.addRow("Game:", self._env_combo)

        # Environment description
        self._env_desc_label = QtWidgets.QLabel("")
        self._env_desc_label.setStyleSheet("color: #666; font-size: 11px;")
        self._env_desc_label.setWordWrap(True)
        layout.addRow("", self._env_desc_label)

        return group

    def _populate_env_combo(self) -> None:
        """Populate environment combo based on selected family."""
        family = self._family_combo.currentText()
        self._env_combo.clear()
        for env_id, display_name, desc in _PGX_ENVIRONMENTS.get(family, []):
            self._env_combo.addItem(f"{display_name} ({env_id})", env_id)
        self._update_env_description()

    def _on_family_changed(self, _index: int) -> None:
        """Handle family selection change."""
        self._populate_env_combo()
        self._apply_game_recommendations()

    def _on_env_changed(self, _index: int) -> None:
        """Handle environment selection change."""
        self._update_env_description()
        self._apply_game_recommendations()

    def _update_env_description(self) -> None:
        """Update environment description label."""
        family = self._family_combo.currentText()
        idx = self._env_combo.currentIndex()
        envs = _PGX_ENVIRONMENTS.get(family, [])
        if 0 <= idx < len(envs):
            _, _, desc = envs[idx]
            self._env_desc_label.setText(desc)

    def _select_env_by_id(self, env_id: str) -> None:
        """Select environment by ID."""
        for family, envs in _PGX_ENVIRONMENTS.items():
            for i, (eid, _, _) in enumerate(envs):
                if eid == env_id:
                    family_idx = self._family_combo.findText(family)
                    if family_idx >= 0:
                        self._family_combo.setCurrentIndex(family_idx)
                        self._env_combo.setCurrentIndex(i)
                    return

    def _apply_game_recommendations(self) -> None:
        """Apply recommended settings for selected game."""
        env_id = self._env_combo.currentData()
        if env_id and env_id in _GAME_RECOMMENDATIONS:
            recs = _GAME_RECOMMENDATIONS[env_id]
            self._simulations_spin.setValue(recs.get("num_simulations", 400))
            self._res_blocks_spin.setValue(recs.get("num_res_blocks", 8))
            self._channels_spin.setValue(recs.get("channels", 128))

    def _create_algorithm_group(self) -> QtWidgets.QGroupBox:
        """Create algorithm selection group."""
        group = QtWidgets.QGroupBox("MCTS Algorithm", self)
        layout = QtWidgets.QVBoxLayout(group)

        # Algorithm selector
        algo_layout = QtWidgets.QFormLayout()
        self._algo_combo = QtWidgets.QComboBox(group)
        for algo_id, algo_info in _MCTX_ALGORITHMS.items():
            self._algo_combo.addItem(f"{algo_info['name']}", algo_id)
        # Default to Gumbel MuZero (state-of-the-art)
        self._algo_combo.setCurrentIndex(0)
        algo_layout.addRow("Algorithm:", self._algo_combo)
        layout.addLayout(algo_layout)

        # Algorithm notes
        self._algo_notes_text = QtWidgets.QTextEdit(group)
        self._algo_notes_text.setReadOnly(True)
        self._algo_notes_text.setMinimumHeight(100)
        self._algo_notes_text.setMaximumHeight(120)
        layout.addWidget(self._algo_notes_text)

        return group

    def _on_algo_changed(self, _index: int) -> None:
        """Handle algorithm selection change."""
        self._update_algo_notes()

    def _update_algo_notes(self) -> None:
        """Update algorithm notes display."""
        algo_id = self._algo_combo.currentData()
        if algo_id and algo_id in _MCTX_ALGORITHMS:
            desc = _MCTX_ALGORITHMS[algo_id]["description"]
            self._algo_notes_text.setHtml(desc)

    def _create_mcts_params_group(self) -> QtWidgets.QGroupBox:
        """Create MCTS parameters group."""
        group = QtWidgets.QGroupBox("MCTS Parameters", self)
        layout = QtWidgets.QGridLayout(group)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)

        row = 0

        # Number of simulations
        layout.addWidget(QtWidgets.QLabel("Simulations per move:"), row, 0)
        self._simulations_spin = QtWidgets.QSpinBox(group)
        self._simulations_spin.setRange(10, 10000)
        self._simulations_spin.setValue(800)
        self._simulations_spin.setSingleStep(50)
        self._simulations_spin.setToolTip(
            "Number of MCTS simulations per move. "
            "More = stronger play but slower training. "
            "Chess: 400-1600, Tic-tac-toe: 50-100"
        )
        layout.addWidget(self._simulations_spin, row, 1)

        # Max considered actions (Gumbel)
        layout.addWidget(QtWidgets.QLabel("Max actions (Gumbel):"), row, 2)
        self._max_actions_spin = QtWidgets.QSpinBox(group)
        self._max_actions_spin.setRange(2, 64)
        self._max_actions_spin.setValue(16)
        self._max_actions_spin.setToolTip(
            "For Gumbel MuZero: max actions to consider. "
            "Lower = faster, higher = better coverage."
        )
        layout.addWidget(self._max_actions_spin, row, 3)

        row += 1

        # Dirichlet alpha
        layout.addWidget(QtWidgets.QLabel("Dirichlet alpha:"), row, 0)
        self._dirichlet_alpha_spin = QtWidgets.QDoubleSpinBox(group)
        self._dirichlet_alpha_spin.setRange(0.01, 1.0)
        self._dirichlet_alpha_spin.setValue(0.3)
        self._dirichlet_alpha_spin.setSingleStep(0.05)
        self._dirichlet_alpha_spin.setToolTip(
            "Dirichlet noise parameter for exploration. "
            "Lower = more exploration. Chess: 0.3, Go: 0.03"
        )
        layout.addWidget(self._dirichlet_alpha_spin, row, 1)

        # Dirichlet fraction
        layout.addWidget(QtWidgets.QLabel("Dirichlet fraction:"), row, 2)
        self._dirichlet_frac_spin = QtWidgets.QDoubleSpinBox(group)
        self._dirichlet_frac_spin.setRange(0.0, 1.0)
        self._dirichlet_frac_spin.setValue(0.25)
        self._dirichlet_frac_spin.setSingleStep(0.05)
        self._dirichlet_frac_spin.setToolTip(
            "Fraction of root policy replaced by Dirichlet noise."
        )
        layout.addWidget(self._dirichlet_frac_spin, row, 3)

        row += 1

        # Temperature
        layout.addWidget(QtWidgets.QLabel("Temperature:"), row, 0)
        self._temperature_spin = QtWidgets.QDoubleSpinBox(group)
        self._temperature_spin.setRange(0.0, 2.0)
        self._temperature_spin.setValue(1.0)
        self._temperature_spin.setSingleStep(0.1)
        self._temperature_spin.setToolTip(
            "Action selection temperature. "
            "1.0 = stochastic, 0.0 = greedy (deterministic)."
        )
        layout.addWidget(self._temperature_spin, row, 1)

        # Temperature drop step
        layout.addWidget(QtWidgets.QLabel("Temp drop step:"), row, 2)
        self._temp_drop_spin = QtWidgets.QSpinBox(group)
        self._temp_drop_spin.setRange(0, 1000)
        self._temp_drop_spin.setValue(30)
        self._temp_drop_spin.setToolTip(
            "Move number to drop temperature to 0 for evaluation."
        )
        layout.addWidget(self._temp_drop_spin, row, 3)

        return group

    def _create_network_params_group(self) -> QtWidgets.QGroupBox:
        """Create neural network parameters group."""
        group = QtWidgets.QGroupBox("Neural Network Architecture", self)
        layout = QtWidgets.QGridLayout(group)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)

        row = 0

        # Residual blocks
        layout.addWidget(QtWidgets.QLabel("Residual blocks:"), row, 0)
        self._res_blocks_spin = QtWidgets.QSpinBox(group)
        self._res_blocks_spin.setRange(1, 40)
        self._res_blocks_spin.setValue(8)
        self._res_blocks_spin.setToolTip(
            "Number of residual blocks. More = stronger but slower. "
            "Chess: 8-20, Simple games: 2-6"
        )
        layout.addWidget(self._res_blocks_spin, row, 1)

        # Channels
        layout.addWidget(QtWidgets.QLabel("Channels:"), row, 2)
        self._channels_spin = QtWidgets.QSpinBox(group)
        self._channels_spin.setRange(16, 512)
        self._channels_spin.setValue(128)
        self._channels_spin.setSingleStep(16)
        self._channels_spin.setToolTip(
            "CNN channel width. More = more capacity but more VRAM."
        )
        layout.addWidget(self._channels_spin, row, 3)

        row += 1

        # Hidden dimensions
        layout.addWidget(QtWidgets.QLabel("Hidden dims:"), row, 0)
        self._hidden_dims_input = QtWidgets.QLineEdit(group)
        self._hidden_dims_input.setText("256, 256")
        self._hidden_dims_input.setToolTip(
            "Comma-separated hidden layer dimensions for MLP heads."
        )
        layout.addWidget(self._hidden_dims_input, row, 1)

        # Use ResNet
        self._use_resnet_checkbox = QtWidgets.QCheckBox("Use ResNet", group)
        self._use_resnet_checkbox.setChecked(True)
        self._use_resnet_checkbox.setToolTip(
            "Use ResNet architecture. Uncheck for MLP (non-image observations)."
        )
        layout.addWidget(self._use_resnet_checkbox, row, 2, 1, 2)

        return group

    def _create_training_params_group(self) -> QtWidgets.QGroupBox:
        """Create training parameters group."""
        group = QtWidgets.QGroupBox("Training Parameters", self)
        layout = QtWidgets.QGridLayout(group)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)

        row = 0

        # Max steps
        layout.addWidget(QtWidgets.QLabel("Max training steps:"), row, 0)
        self._max_steps_spin = QtWidgets.QSpinBox(group)
        self._max_steps_spin.setRange(0, 100_000_000)
        self._max_steps_spin.setValue(100_000)
        self._max_steps_spin.setSingleStep(10000)
        self._max_steps_spin.setSpecialValueText("Unlimited")
        self._max_steps_spin.setToolTip("Maximum training steps. 0 = unlimited.")
        layout.addWidget(self._max_steps_spin, row, 1)

        # Seed
        layout.addWidget(QtWidgets.QLabel("Random seed:"), row, 2)
        self._seed_spin = QtWidgets.QSpinBox(group)
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        self._seed_spin.setToolTip("Random seed for reproducibility.")
        layout.addWidget(self._seed_spin, row, 3)

        row += 1

        # Learning rate
        layout.addWidget(QtWidgets.QLabel("Learning rate:"), row, 0)
        self._lr_spin = QtWidgets.QDoubleSpinBox(group)
        self._lr_spin.setRange(1e-6, 0.1)
        self._lr_spin.setValue(2e-4)
        self._lr_spin.setDecimals(6)
        self._lr_spin.setSingleStep(1e-5)
        self._lr_spin.setToolTip("Adam optimizer learning rate.")
        layout.addWidget(self._lr_spin, row, 1)

        # Batch size
        layout.addWidget(QtWidgets.QLabel("Batch size:"), row, 2)
        self._batch_size_spin = QtWidgets.QSpinBox(group)
        self._batch_size_spin.setRange(16, 4096)
        self._batch_size_spin.setValue(256)
        self._batch_size_spin.setSingleStep(32)
        self._batch_size_spin.setToolTip("Training batch size.")
        layout.addWidget(self._batch_size_spin, row, 3)

        row += 1

        # Replay buffer size
        layout.addWidget(QtWidgets.QLabel("Replay buffer size:"), row, 0)
        self._buffer_size_spin = QtWidgets.QSpinBox(group)
        self._buffer_size_spin.setRange(1000, 10_000_000)
        self._buffer_size_spin.setValue(100_000)
        self._buffer_size_spin.setSingleStep(10000)
        self._buffer_size_spin.setToolTip("Maximum replay buffer capacity.")
        layout.addWidget(self._buffer_size_spin, row, 1)

        # Games per iteration
        layout.addWidget(QtWidgets.QLabel("Games/iteration:"), row, 2)
        self._games_per_iter_spin = QtWidgets.QSpinBox(group)
        self._games_per_iter_spin.setRange(1, 1024)
        self._games_per_iter_spin.setValue(128)
        self._games_per_iter_spin.setToolTip(
            "Self-play games to generate per training iteration."
        )
        layout.addWidget(self._games_per_iter_spin, row, 3)

        row += 1

        # Number of actors
        layout.addWidget(QtWidgets.QLabel("Self-play actors:"), row, 0)
        self._num_actors_spin = QtWidgets.QSpinBox(group)
        self._num_actors_spin.setRange(1, 128)
        self._num_actors_spin.setValue(8)
        self._num_actors_spin.setToolTip(
            "Number of parallel self-play actors (batched on GPU)."
        )
        layout.addWidget(self._num_actors_spin, row, 1)

        # Device
        layout.addWidget(QtWidgets.QLabel("Device:"), row, 2)
        self._device_combo = QtWidgets.QComboBox(group)
        self._device_combo.addItems(["gpu", "cpu", "tpu"])
        self._device_combo.setToolTip("JAX device for training.")
        layout.addWidget(self._device_combo, row, 3)

        return group

    def _create_logging_group(self) -> QtWidgets.QGroupBox:
        """Create logging configuration group."""
        group = QtWidgets.QGroupBox("Logging & Monitoring", self)
        layout = QtWidgets.QVBoxLayout(group)

        # Checkbox row
        checkbox_layout = QtWidgets.QHBoxLayout()

        self._tensorboard_checkbox = QtWidgets.QCheckBox("TensorBoard", group)
        self._tensorboard_checkbox.setChecked(True)
        self._tensorboard_checkbox.setToolTip("Log metrics to TensorBoard.")
        checkbox_layout.addWidget(self._tensorboard_checkbox)

        self._wandb_checkbox = QtWidgets.QCheckBox("Weights & Biases", group)
        self._wandb_checkbox.setChecked(False)
        self._wandb_checkbox.setToolTip("Log metrics to W&B (requires wandb login).")
        checkbox_layout.addWidget(self._wandb_checkbox)

        checkbox_layout.addStretch(1)
        layout.addLayout(checkbox_layout)

        # Logging interval
        interval_layout = QtWidgets.QHBoxLayout()
        interval_layout.addWidget(QtWidgets.QLabel("Log interval (steps):"))
        self._log_interval_spin = QtWidgets.QSpinBox(group)
        self._log_interval_spin.setRange(1, 10000)
        self._log_interval_spin.setValue(100)
        self._log_interval_spin.setToolTip("Steps between progress log outputs.")
        interval_layout.addWidget(self._log_interval_spin)
        interval_layout.addStretch(1)
        layout.addLayout(interval_layout)

        return group

    def _create_checkpoint_group(self) -> QtWidgets.QGroupBox:
        """Create checkpoint configuration group."""
        group = QtWidgets.QGroupBox("Model Checkpoints", self)
        layout = QtWidgets.QVBoxLayout(group)

        # Checkbox row
        checkbox_layout = QtWidgets.QHBoxLayout()

        self._save_checkpoint_checkbox = QtWidgets.QCheckBox("Save checkpoints", group)
        self._save_checkpoint_checkbox.setChecked(True)
        self._save_checkpoint_checkbox.setToolTip("Save model checkpoints during training.")
        checkbox_layout.addWidget(self._save_checkpoint_checkbox)

        checkbox_layout.addStretch(1)
        layout.addLayout(checkbox_layout)

        # Checkpoint interval
        interval_layout = QtWidgets.QHBoxLayout()
        interval_layout.addWidget(QtWidgets.QLabel("Checkpoint interval:"))
        self._checkpoint_interval_spin = QtWidgets.QSpinBox(group)
        self._checkpoint_interval_spin.setRange(0, 100000)
        self._checkpoint_interval_spin.setValue(1000)
        self._checkpoint_interval_spin.setSpecialValueText("At end only")
        self._checkpoint_interval_spin.setToolTip(
            "Iterations between checkpoints. 0 = only save at end."
        )
        interval_layout.addWidget(self._checkpoint_interval_spin)
        interval_layout.addStretch(1)
        layout.addLayout(interval_layout)

        # Policy output info
        output_label = QtWidgets.QLabel(
            "Policy checkpoints saved to: <code>var/trainer/runs/&lt;run_id&gt;/checkpoints/</code>"
        )
        output_label.setStyleSheet("color: #666; font-size: 10px;")
        output_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(output_label)

        return group

    def _on_validate_clicked(self) -> None:
        """Validate JAX and dependencies are available."""
        self._progress_label.setText("Validating dependencies...")
        QtWidgets.QApplication.processEvents()

        errors = []

        # Check JAX
        try:
            import jax
            devices = jax.devices()
            device_str = ", ".join(str(d) for d in devices)
            _LOGGER.info(f"JAX available with devices: {device_str}")
        except ImportError:
            errors.append("JAX not installed. Install with: pip install jax jaxlib")
        except Exception as e:
            errors.append(f"JAX error: {e}")

        # Check pgx
        try:
            import pgx
            _LOGGER.info("PGX available")
        except ImportError:
            errors.append("PGX not installed. Install with: pip install pgx")

        # Check mctx
        try:
            import mctx
            _LOGGER.info("mctx available")
        except ImportError:
            errors.append("mctx not installed. Install with: pip install mctx")

        # Check flax
        try:
            import flax
            _LOGGER.info("Flax available")
        except ImportError:
            errors.append("Flax not installed. Install with: pip install flax")

        # Check optax
        try:
            import optax
            _LOGGER.info("Optax available")
        except ImportError:
            errors.append("Optax not installed. Install with: pip install optax")

        if errors:
            self._progress_label.setText("")
            QtWidgets.QMessageBox.warning(
                self,
                "Validation Failed",
                "Missing dependencies:\n\n" + "\n".join(f"- {e}" for e in errors),
            )
        else:
            self._progress_label.setText("Validation successful!")
            QtWidgets.QMessageBox.information(
                self,
                "Validation Successful",
                "All dependencies available. Ready to train!",
            )

    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        self._last_config = self._build_config()
        self.accept()

    def _parse_hidden_dims(self) -> tuple:
        """Parse hidden dimensions from text input."""
        text = self._hidden_dims_input.text().strip()
        if not text:
            return (256, 256)
        try:
            dims = [int(d.strip()) for d in text.split(",") if d.strip()]
            return tuple(dims) if dims else (256, 256)
        except ValueError:
            return (256, 256)

    def _build_config(self) -> Dict[str, Any]:
        """Build the configuration dictionary."""
        env_id = self._env_combo.currentData() or "chess"
        algo_id = self._algo_combo.currentData() or "gumbel_muzero"
        run_id = _generate_run_id(env_id, algo_id)

        config = {
            "metadata": {
                "worker": {
                    "type": "mctx",
                    "config": {
                        "run_id": run_id,
                        "seed": self._seed_spin.value(),
                        "env_id": env_id,
                        "algorithm": algo_id,
                        "max_steps": self._max_steps_spin.value(),
                        "max_episodes": 0,
                        "mode": "train",
                        "device": self._device_combo.currentText(),
                        "logging_interval": self._log_interval_spin.value(),
                        "verbose": 1,
                        "mcts": {
                            "num_simulations": self._simulations_spin.value(),
                            "max_num_considered_actions": self._max_actions_spin.value(),
                            "dirichlet_alpha": self._dirichlet_alpha_spin.value(),
                            "dirichlet_fraction": self._dirichlet_frac_spin.value(),
                            "temperature": self._temperature_spin.value(),
                            "temperature_drop_step": self._temp_drop_spin.value(),
                        },
                        "network": {
                            "num_res_blocks": self._res_blocks_spin.value(),
                            "channels": self._channels_spin.value(),
                            "hidden_dims": list(self._parse_hidden_dims()),
                            "use_resnet": self._use_resnet_checkbox.isChecked(),
                        },
                        "training": {
                            "learning_rate": self._lr_spin.value(),
                            "batch_size": self._batch_size_spin.value(),
                            "replay_buffer_size": self._buffer_size_spin.value(),
                            "games_per_iteration": self._games_per_iter_spin.value(),
                            "num_actors": self._num_actors_spin.value(),
                            "checkpoint_interval": self._checkpoint_interval_spin.value(),
                        },
                    },
                },
            },
            "tensorboard": self._tensorboard_checkbox.isChecked(),
            "wandb": self._wandb_checkbox.isChecked(),
            "run_id": run_id,
        }

        return config

    def get_config(self) -> Dict[str, Any]:
        """Return the training configuration."""
        if self._last_config is not None:
            return copy.deepcopy(self._last_config)
        return self._build_config()


# Register form with factory at module load
def _register_mctx_train_form() -> None:
    """Register MCTX train form with factory (deferred to avoid circular import)."""
    try:
        from gym_gui.ui.forms import get_worker_form_factory
        factory = get_worker_form_factory()
        if not factory.has_train_form("mctx_worker"):
            factory.register_train_form(
                "mctx_worker",
                lambda parent=None, **kwargs: MCTXTrainForm(parent=parent, **kwargs),
            )
    except Exception as e:
        _LOGGER.warning(f"Failed to register MCTX train form: {e}")


_register_mctx_train_form()


__all__ = ["MCTXTrainForm"]
