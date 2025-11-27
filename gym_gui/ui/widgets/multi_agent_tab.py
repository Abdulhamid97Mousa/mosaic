"""Multi-Agent Mode tab widget with game mode subtabs.

This widget provides three game modes for multi-agent environments:
1. Human-Vs-Agent: Play against trained AI agents (AEC turn-based games)
2. Multi-Agent Cooperation: Train/evaluate cooperative multi-agent teams
3. Multi-Agent Competition: Train/evaluate competitive agents
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal

from gym_gui.core.pettingzoo_enums import (
    PETTINGZOO_ENV_METADATA,
    PettingZooAPIType,
    PettingZooEnvId,
    PettingZooFamily,
    PettingZooGameType,
    get_api_type,
    get_competitive_envs,
    get_cooperative_envs,
    get_description,
    get_display_name,
    get_envs_by_family,
    get_game_type,
    get_human_vs_agent_envs,
    is_aec_env,
)
from gym_gui.ui.workers import WorkerDefinition, get_worker_catalog

_LOGGER = logging.getLogger(__name__)


class HumanVsAgentTab(QtWidgets.QWidget):
    """Tab for Human vs Agent gameplay in turn-based games.

    Supports AEC (Agent Environment Cycle) games where humans can
    play against trained AI policies.
    """

    # Signals
    load_environment_requested = pyqtSignal(str, int)  # env_id, seed
    load_policy_requested = pyqtSignal(str)  # env_id
    start_game_requested = pyqtSignal(str, str, int)  # env_id, human_agent, seed
    reset_game_requested = pyqtSignal(int)  # seed
    action_submitted = pyqtSignal(int)  # action_id

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._selected_env: Optional[PettingZooEnvId] = None
        self._default_seed = 42
        self._allow_seed_reuse = False
        self._environment_loaded = False
        self._policy_loaded = False
        self._build_ui()
        self._connect_signals()
        self._populate_environments()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Info label
        info = QtWidgets.QLabel(
            "Play against trained AI agents in turn-based games. "
            "Load an environment, select a policy, choose your player, and start!",
            self,
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info)

        # Environment selection group (matches Human Control structure)
        env_group = QtWidgets.QGroupBox("Environment", self)
        env_layout = QtWidgets.QGridLayout(env_group)
        env_layout.setColumnStretch(1, 1)

        # Family selection
        env_layout.addWidget(QtWidgets.QLabel("Family", env_group), 0, 0)
        self._family_combo = QtWidgets.QComboBox(env_group)
        self._family_combo.setMaxVisibleItems(10)
        env_layout.addWidget(self._family_combo, 0, 1, 1, 2)

        # Game selection
        env_layout.addWidget(QtWidgets.QLabel("Game", env_group), 1, 0)
        self._env_combo = QtWidgets.QComboBox(env_group)
        self._env_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._env_combo.setMaxVisibleItems(10)
        env_layout.addWidget(self._env_combo, 1, 1, 1, 2)

        # Seed
        env_layout.addWidget(QtWidgets.QLabel("Seed", env_group), 2, 0)
        self._seed_spin = QtWidgets.QSpinBox(env_group)
        self._seed_spin.setRange(1, 10_000_000)
        self._seed_spin.setValue(self._default_seed)
        self._seed_spin.setToolTip(
            "Seed for environment randomization. Increment after each game for variety."
        )
        env_layout.addWidget(self._seed_spin, 2, 1)

        self._seed_reuse_checkbox = QtWidgets.QCheckBox("Allow seed reuse", env_group)
        self._seed_reuse_checkbox.setChecked(self._allow_seed_reuse)
        env_layout.addWidget(self._seed_reuse_checkbox, 2, 2)

        # Load Environment button
        self._load_env_btn = QtWidgets.QPushButton("Load Environment", env_group)
        env_layout.addWidget(self._load_env_btn, 3, 0, 1, 3)

        layout.addWidget(env_group)

        # Environment info
        self._env_info = QtWidgets.QLabel("", self)
        self._env_info.setWordWrap(True)
        self._env_info.setStyleSheet(
            "color: #666; font-size: 11px; padding: 4px; "
            "background-color: #f8f8f8; border-radius: 4px;"
        )
        layout.addWidget(self._env_info)

        # AI Policy group
        policy_group = QtWidgets.QGroupBox("AI Policy", self)
        policy_layout = QtWidgets.QVBoxLayout(policy_group)

        policy_btn_layout = QtWidgets.QHBoxLayout()
        self._load_policy_btn = QtWidgets.QPushButton(
            "Load Trained Policy...", policy_group
        )
        self._load_policy_btn.setEnabled(False)  # Enable after environment loaded
        policy_btn_layout.addWidget(self._load_policy_btn)
        policy_btn_layout.addStretch(1)
        policy_layout.addLayout(policy_btn_layout)

        self._policy_label = QtWidgets.QLabel("No policy loaded", policy_group)
        self._policy_label.setStyleSheet("color: #888;")
        policy_layout.addWidget(self._policy_label)

        layout.addWidget(policy_group)

        # Player Assignment group
        player_group = QtWidgets.QGroupBox("Player Assignment", self)
        player_layout = QtWidgets.QFormLayout(player_group)

        self._human_player_combo = QtWidgets.QComboBox(player_group)
        self._human_player_combo.addItem("Player 1 (First)", "player_0")
        self._human_player_combo.addItem("Player 2 (Second)", "player_1")
        player_layout.addRow("You play as:", self._human_player_combo)

        layout.addWidget(player_group)

        # Game Controls group
        control_group = QtWidgets.QGroupBox("Game Controls", self)
        control_layout = QtWidgets.QHBoxLayout(control_group)

        self._start_btn = QtWidgets.QPushButton("Start Game", control_group)
        self._start_btn.setEnabled(False)
        control_layout.addWidget(self._start_btn)

        self._reset_btn = QtWidgets.QPushButton("Reset", control_group)
        self._reset_btn.setEnabled(False)
        control_layout.addWidget(self._reset_btn)

        control_layout.addStretch(1)

        layout.addWidget(control_group)

        # Game Status group
        status_group = QtWidgets.QGroupBox("Game Status", self)
        status_layout = QtWidgets.QFormLayout(status_group)

        self._turn_label = QtWidgets.QLabel("—", status_group)
        status_layout.addRow("Current Turn:", self._turn_label)

        self._score_label = QtWidgets.QLabel("—", status_group)
        status_layout.addRow("Score:", self._score_label)

        self._result_label = QtWidgets.QLabel("—", status_group)
        status_layout.addRow("Result:", self._result_label)

        layout.addWidget(status_group)

        layout.addStretch(1)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)
        self._seed_reuse_checkbox.stateChanged.connect(self._on_seed_reuse_changed)
        self._load_env_btn.clicked.connect(self._on_load_environment)
        self._load_policy_btn.clicked.connect(self._on_load_policy)
        self._start_btn.clicked.connect(self._on_start_game)
        self._reset_btn.clicked.connect(self._on_reset_game)

    def _populate_environments(self) -> None:
        """Populate environment dropdowns with human-controllable games."""
        self._family_combo.clear()
        self._family_combo.addItem("Classic (Board Games)", PettingZooFamily.CLASSIC.value)

        # Populate initial environments
        self._on_family_changed(0)

    def _on_family_changed(self, index: int) -> None:
        """Handle family selection change."""
        self._env_combo.clear()

        # Get human-controllable envs
        human_envs = get_human_vs_agent_envs()

        for env_id in human_envs:
            display_name = get_display_name(env_id)
            self._env_combo.addItem(display_name, env_id.value)

        if human_envs:
            self._env_combo.setCurrentIndex(0)
            self._on_env_changed(0)

    def _on_env_changed(self, index: int) -> None:
        """Handle environment selection change."""
        env_value = self._env_combo.currentData()
        if not env_value:
            self._env_info.setText("")
            return

        try:
            env_id = PettingZooEnvId(env_value)
            self._selected_env = env_id
            description = get_description(env_id)
            api_type = get_api_type(env_id)
            self._env_info.setText(
                f"<b>{get_display_name(env_id)}</b><br/>"
                f"{description}<br/>"
                f"<i>API: {api_type.value.upper()} (Turn-based)</i>"
            )
            # Reset loaded state when environment changes
            self._environment_loaded = False
            self._policy_loaded = False
            self._load_policy_btn.setEnabled(False)
            self._start_btn.setEnabled(False)
            self._policy_label.setText("No policy loaded")
        except ValueError:
            self._env_info.setText("")

    def _on_seed_reuse_changed(self, state: int) -> None:
        """Handle seed reuse checkbox change."""
        self._allow_seed_reuse = state == QtCore.Qt.CheckState.Checked.value
        if self._allow_seed_reuse:
            self._seed_spin.setToolTip(
                "Seed can be reused. Adjust before loading to replay same game."
            )
        else:
            self._seed_spin.setToolTip(
                "Seed increments automatically after each game."
            )

    def _on_load_environment(self) -> None:
        """Handle load environment button click."""
        if self._selected_env:
            seed = self._seed_spin.value()
            self.load_environment_requested.emit(self._selected_env.value, seed)
            self._environment_loaded = True
            self._load_policy_btn.setEnabled(True)
            self._update_button_states()
            _LOGGER.info(
                "Environment load requested: %s (seed=%d)",
                self._selected_env.value,
                seed,
            )

    def _on_load_policy(self) -> None:
        """Handle load policy button click."""
        if self._selected_env:
            self.load_policy_requested.emit(self._selected_env.value)

    def _on_start_game(self) -> None:
        """Handle start game button click."""
        if self._selected_env:
            human_agent = self._human_player_combo.currentData()
            seed = self._seed_spin.value()
            self.start_game_requested.emit(self._selected_env.value, human_agent, seed)
            self._reset_btn.setEnabled(True)
            # Auto-increment seed if not allowing reuse
            if not self._allow_seed_reuse:
                self._seed_spin.setValue(seed + 1)

    def _on_reset_game(self) -> None:
        """Handle reset button click."""
        seed = self._seed_spin.value()
        self.reset_game_requested.emit(seed)
        self._result_label.setText("—")
        self._turn_label.setText("—")
        self._score_label.setText("—")
        # Auto-increment seed if not allowing reuse
        if not self._allow_seed_reuse:
            self._seed_spin.setValue(seed + 1)

    def _update_button_states(self) -> None:
        """Update button enabled states based on current state."""
        self._load_policy_btn.setEnabled(self._environment_loaded)
        self._start_btn.setEnabled(self._environment_loaded and self._policy_loaded)

    def set_policy_loaded(self, policy_path: str) -> None:
        """Update UI when a policy is loaded."""
        self._policy_label.setText(f"Loaded: {policy_path}")
        self._policy_loaded = True
        self._update_button_states()

    def set_environment_loaded(self, env_id: str, seed: int) -> None:
        """Update UI when environment is loaded."""
        self._environment_loaded = True
        self._load_policy_btn.setEnabled(True)
        _LOGGER.debug("Environment loaded: %s (seed=%d)", env_id, seed)

    def update_game_status(
        self,
        *,
        current_turn: str,
        score: str,
        result: Optional[str] = None,
    ) -> None:
        """Update game status display."""
        self._turn_label.setText(current_turn)
        self._score_label.setText(score)
        if result:
            self._result_label.setText(result)
            self._reset_btn.setEnabled(True)
        else:
            self._result_label.setText("In Progress")

    def current_seed(self) -> int:
        """Get current seed value."""
        return self._seed_spin.value()

    def current_env_id(self) -> Optional[str]:
        """Get current environment ID."""
        return self._selected_env.value if self._selected_env else None


class MultiAgentCooperationTab(QtWidgets.QWidget):
    """Tab for cooperative multi-agent training and evaluation.

    Supports environments where agents must work together to achieve
    a common goal.
    """

    # Signals
    worker_changed = pyqtSignal(str)  # worker_id
    train_requested = pyqtSignal(str, str)  # worker_id, env_id
    evaluate_requested = pyqtSignal(str, str)  # worker_id, env_id

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._workers: List[WorkerDefinition] = []
        self._build_ui()
        self._connect_signals()
        self._populate_workers()
        self._populate_environments()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Info label
        info = QtWidgets.QLabel(
            "Train and evaluate cooperative multi-agent teams. "
            "Agents work together to achieve shared objectives.",
            self,
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info)

        # Worker selection
        worker_group = QtWidgets.QGroupBox("Worker Integration", self)
        worker_layout = QtWidgets.QFormLayout(worker_group)

        self._worker_combo = QtWidgets.QComboBox(worker_group)
        worker_layout.addRow("Worker:", self._worker_combo)

        self._worker_info = QtWidgets.QLabel("", worker_group)
        self._worker_info.setWordWrap(True)
        self._worker_info.setStyleSheet("color: #666; font-size: 11px;")
        worker_layout.addRow("", self._worker_info)

        layout.addWidget(worker_group)

        # Environment selection
        env_group = QtWidgets.QGroupBox("Cooperative Environment", self)
        env_layout = QtWidgets.QFormLayout(env_group)

        self._family_combo = QtWidgets.QComboBox(env_group)
        env_layout.addRow("Family:", self._family_combo)

        self._env_combo = QtWidgets.QComboBox(env_group)
        env_layout.addRow("Environment:", self._env_combo)

        self._env_info = QtWidgets.QLabel("", env_group)
        self._env_info.setWordWrap(True)
        self._env_info.setStyleSheet("color: #666; font-size: 11px;")
        env_layout.addRow("", self._env_info)

        layout.addWidget(env_group)

        # Action buttons
        action_group = QtWidgets.QGroupBox("Actions", self)
        action_layout = QtWidgets.QHBoxLayout(action_group)

        self._train_btn = QtWidgets.QPushButton("Train Agents", action_group)
        action_layout.addWidget(self._train_btn)

        self._eval_btn = QtWidgets.QPushButton("Load Policy", action_group)
        action_layout.addWidget(self._eval_btn)

        action_layout.addStretch(1)

        layout.addWidget(action_group)

        layout.addStretch(1)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._worker_combo.currentIndexChanged.connect(self._on_worker_changed)
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)
        self._train_btn.clicked.connect(self._on_train)
        self._eval_btn.clicked.connect(self._on_evaluate)

    def _populate_workers(self) -> None:
        """Populate worker dropdown."""
        self._worker_combo.clear()
        catalog = get_worker_catalog()

        for worker in catalog:
            if worker.supports_training:
                self._worker_combo.addItem(worker.display_name, worker.worker_id)
                self._workers.append(worker)

        if self._workers:
            self._on_worker_changed(0)

    def _populate_environments(self) -> None:
        """Populate environment dropdowns with cooperative games."""
        self._family_combo.clear()

        # Add families that have cooperative games
        families_with_coop = set()
        for env_id in get_cooperative_envs():
            if env_id in PETTINGZOO_ENV_METADATA:
                family = PETTINGZOO_ENV_METADATA[env_id][0]
                families_with_coop.add(family)

        for family in PettingZooFamily:
            if family in families_with_coop:
                label = family.value.replace("_", " ").title()
                self._family_combo.addItem(label, family.value)

        if families_with_coop:
            self._on_family_changed(0)

    def _on_worker_changed(self, index: int) -> None:
        """Handle worker selection change."""
        worker_id = self._worker_combo.currentData()
        if not worker_id:
            return

        for worker in self._workers:
            if worker.worker_id == worker_id:
                self._worker_info.setText(worker.description)
                break

        self.worker_changed.emit(worker_id)

    def _on_family_changed(self, index: int) -> None:
        """Handle family selection change."""
        family_value = self._family_combo.currentData()
        if not family_value:
            return

        try:
            family = PettingZooFamily(family_value)
        except ValueError:
            return

        self._env_combo.clear()

        # Get cooperative envs for this family
        coop_envs = get_cooperative_envs()
        family_envs = get_envs_by_family(family)

        for env_id in family_envs:
            if env_id in coop_envs or get_game_type(env_id) == PettingZooGameType.MIXED:
                display_name = get_display_name(env_id)
                self._env_combo.addItem(display_name, env_id.value)

        if self._env_combo.count() > 0:
            self._env_combo.setCurrentIndex(0)
            self._on_env_changed(0)

    def _on_env_changed(self, index: int) -> None:
        """Handle environment selection change."""
        env_value = self._env_combo.currentData()
        if not env_value:
            self._env_info.setText("")
            return

        try:
            env_id = PettingZooEnvId(env_value)
            description = get_description(env_id)
            api_type = get_api_type(env_id)
            self._env_info.setText(f"{description}\n(API: {api_type.value.upper()})")
        except ValueError:
            self._env_info.setText("")

    def _on_train(self) -> None:
        """Handle train button click."""
        worker_id = self._worker_combo.currentData()
        env_id = self._env_combo.currentData()
        if worker_id and env_id:
            self.train_requested.emit(worker_id, env_id)

    def _on_evaluate(self) -> None:
        """Handle evaluate button click."""
        worker_id = self._worker_combo.currentData()
        env_id = self._env_combo.currentData()
        if worker_id and env_id:
            self.evaluate_requested.emit(worker_id, env_id)


class MultiAgentCompetitionTab(QtWidgets.QWidget):
    """Tab for competitive multi-agent training and evaluation.

    Supports environments where agents compete against each other,
    including self-play and tournament modes.
    """

    # Signals
    worker_changed = pyqtSignal(str)  # worker_id
    train_requested = pyqtSignal(str, str)  # worker_id, env_id
    evaluate_requested = pyqtSignal(str, str)  # worker_id, env_id

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._workers: List[WorkerDefinition] = []
        self._build_ui()
        self._connect_signals()
        self._populate_workers()
        self._populate_environments()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Info label
        info = QtWidgets.QLabel(
            "Train and evaluate competitive agents. "
            "Agents compete against each other through self-play or tournaments.",
            self,
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info)

        # Worker selection
        worker_group = QtWidgets.QGroupBox("Worker Integration", self)
        worker_layout = QtWidgets.QFormLayout(worker_group)

        self._worker_combo = QtWidgets.QComboBox(worker_group)
        worker_layout.addRow("Worker:", self._worker_combo)

        self._worker_info = QtWidgets.QLabel("", worker_group)
        self._worker_info.setWordWrap(True)
        self._worker_info.setStyleSheet("color: #666; font-size: 11px;")
        worker_layout.addRow("", self._worker_info)

        layout.addWidget(worker_group)

        # Environment selection
        env_group = QtWidgets.QGroupBox("Competition Environment", self)
        env_layout = QtWidgets.QFormLayout(env_group)

        self._family_combo = QtWidgets.QComboBox(env_group)
        env_layout.addRow("Family:", self._family_combo)

        self._env_combo = QtWidgets.QComboBox(env_group)
        env_layout.addRow("Environment:", self._env_combo)

        self._env_info = QtWidgets.QLabel("", env_group)
        self._env_info.setWordWrap(True)
        self._env_info.setStyleSheet("color: #666; font-size: 11px;")
        env_layout.addRow("", self._env_info)

        layout.addWidget(env_group)

        # Training mode
        mode_group = QtWidgets.QGroupBox("Training Mode", self)
        mode_layout = QtWidgets.QVBoxLayout(mode_group)

        self._selfplay_radio = QtWidgets.QRadioButton("Self-Play", mode_group)
        self._selfplay_radio.setChecked(True)
        self._selfplay_radio.setToolTip("Agent plays against copies of itself")
        mode_layout.addWidget(self._selfplay_radio)

        self._population_radio = QtWidgets.QRadioButton("Population-Based", mode_group)
        self._population_radio.setToolTip("Train a population of diverse agents")
        mode_layout.addWidget(self._population_radio)

        self._league_radio = QtWidgets.QRadioButton("League Training", mode_group)
        self._league_radio.setToolTip("AlphaStar-style league training")
        mode_layout.addWidget(self._league_radio)

        layout.addWidget(mode_group)

        # Action buttons
        action_group = QtWidgets.QGroupBox("Actions", self)
        action_layout = QtWidgets.QHBoxLayout(action_group)

        self._train_btn = QtWidgets.QPushButton("Train Agents", action_group)
        action_layout.addWidget(self._train_btn)

        self._eval_btn = QtWidgets.QPushButton("Load Policy", action_group)
        action_layout.addWidget(self._eval_btn)

        self._tournament_btn = QtWidgets.QPushButton("Run Tournament", action_group)
        self._tournament_btn.setEnabled(False)  # Future feature
        action_layout.addWidget(self._tournament_btn)

        action_layout.addStretch(1)

        layout.addWidget(action_group)

        layout.addStretch(1)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._worker_combo.currentIndexChanged.connect(self._on_worker_changed)
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)
        self._train_btn.clicked.connect(self._on_train)
        self._eval_btn.clicked.connect(self._on_evaluate)

    def _populate_workers(self) -> None:
        """Populate worker dropdown."""
        self._worker_combo.clear()
        catalog = get_worker_catalog()

        for worker in catalog:
            if worker.supports_training:
                self._worker_combo.addItem(worker.display_name, worker.worker_id)
                self._workers.append(worker)

        if self._workers:
            self._on_worker_changed(0)

    def _populate_environments(self) -> None:
        """Populate environment dropdowns with competitive games."""
        self._family_combo.clear()

        # Add families that have competitive games
        families_with_competitive = set()
        for env_id in get_competitive_envs():
            if env_id in PETTINGZOO_ENV_METADATA:
                family = PETTINGZOO_ENV_METADATA[env_id][0]
                families_with_competitive.add(family)

        for family in PettingZooFamily:
            if family in families_with_competitive:
                label = family.value.replace("_", " ").title()
                self._family_combo.addItem(label, family.value)

        if families_with_competitive:
            self._on_family_changed(0)

    def _on_worker_changed(self, index: int) -> None:
        """Handle worker selection change."""
        worker_id = self._worker_combo.currentData()
        if not worker_id:
            return

        for worker in self._workers:
            if worker.worker_id == worker_id:
                self._worker_info.setText(worker.description)
                break

        self.worker_changed.emit(worker_id)

    def _on_family_changed(self, index: int) -> None:
        """Handle family selection change."""
        family_value = self._family_combo.currentData()
        if not family_value:
            return

        try:
            family = PettingZooFamily(family_value)
        except ValueError:
            return

        self._env_combo.clear()

        # Get competitive envs for this family
        competitive_envs = get_competitive_envs()
        family_envs = get_envs_by_family(family)

        for env_id in family_envs:
            if env_id in competitive_envs or get_game_type(env_id) == PettingZooGameType.MIXED:
                display_name = get_display_name(env_id)
                self._env_combo.addItem(display_name, env_id.value)

        if self._env_combo.count() > 0:
            self._env_combo.setCurrentIndex(0)
            self._on_env_changed(0)

    def _on_env_changed(self, index: int) -> None:
        """Handle environment selection change."""
        env_value = self._env_combo.currentData()
        if not env_value:
            self._env_info.setText("")
            return

        try:
            env_id = PettingZooEnvId(env_value)
            description = get_description(env_id)
            api_type = get_api_type(env_id)
            self._env_info.setText(f"{description}\n(API: {api_type.value.upper()})")
        except ValueError:
            self._env_info.setText("")

    def _on_train(self) -> None:
        """Handle train button click."""
        worker_id = self._worker_combo.currentData()
        env_id = self._env_combo.currentData()
        if worker_id and env_id:
            self.train_requested.emit(worker_id, env_id)

    def _on_evaluate(self) -> None:
        """Handle evaluate button click."""
        worker_id = self._worker_combo.currentData()
        env_id = self._env_combo.currentData()
        if worker_id and env_id:
            self.evaluate_requested.emit(worker_id, env_id)


class MultiAgentTab(QtWidgets.QWidget):
    """Main Multi-Agent Mode tab with subtabs for different game modes.

    Contains:
    - Human vs Agent: Play against trained AI
    - Cooperation: Train cooperative teams
    - Competition: Train competitive agents
    """

    # Forwarded signals
    worker_changed = pyqtSignal(str)
    train_requested = pyqtSignal(str, str)  # worker_id, env_id
    evaluate_requested = pyqtSignal(str, str)  # worker_id, env_id
    load_policy_requested = pyqtSignal(str)  # env_id
    load_environment_requested = pyqtSignal(str, int)  # env_id, seed
    start_game_requested = pyqtSignal(str, str, int)  # env_id, human_agent, seed
    reset_game_requested = pyqtSignal(int)  # seed

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Subtab widget
        self._subtabs = QtWidgets.QTabWidget(self)
        self._subtabs.setTabPosition(QtWidgets.QTabWidget.TabPosition.North)

        # Create subtabs
        self._human_vs_agent_tab = HumanVsAgentTab(self)
        self._cooperation_tab = MultiAgentCooperationTab(self)
        self._competition_tab = MultiAgentCompetitionTab(self)

        # Add subtabs
        self._subtabs.addTab(self._human_vs_agent_tab, "Human vs Agent")
        self._subtabs.addTab(self._cooperation_tab, "Cooperation")
        self._subtabs.addTab(self._competition_tab, "Competition")

        layout.addWidget(self._subtabs)

    def _connect_signals(self) -> None:
        """Connect signals from subtabs."""
        # Human vs Agent
        self._human_vs_agent_tab.load_environment_requested.connect(
            self.load_environment_requested
        )
        self._human_vs_agent_tab.load_policy_requested.connect(
            self.load_policy_requested
        )
        self._human_vs_agent_tab.start_game_requested.connect(
            self.start_game_requested
        )
        self._human_vs_agent_tab.reset_game_requested.connect(
            self.reset_game_requested
        )

        # Cooperation
        self._cooperation_tab.worker_changed.connect(self.worker_changed)
        self._cooperation_tab.train_requested.connect(self.train_requested)
        self._cooperation_tab.evaluate_requested.connect(self.evaluate_requested)

        # Competition
        self._competition_tab.worker_changed.connect(self.worker_changed)
        self._competition_tab.train_requested.connect(self.train_requested)
        self._competition_tab.evaluate_requested.connect(self.evaluate_requested)

    @property
    def human_vs_agent(self) -> HumanVsAgentTab:
        """Get the Human vs Agent subtab."""
        return self._human_vs_agent_tab

    @property
    def cooperation(self) -> MultiAgentCooperationTab:
        """Get the Cooperation subtab."""
        return self._cooperation_tab

    @property
    def competition(self) -> MultiAgentCompetitionTab:
        """Get the Competition subtab."""
        return self._competition_tab


__all__ = [
    "MultiAgentTab",
    "HumanVsAgentTab",
    "MultiAgentCooperationTab",
    "MultiAgentCompetitionTab",
]
