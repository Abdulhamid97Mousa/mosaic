"""Multi-Agent Mode tab widget with game mode subtabs.

This widget provides three game modes for multi-agent environments:
1. Human-Vs-Agent: Play against trained AI agents (AEC turn-based games)
2. Multi-Agent Cooperation: Train/evaluate cooperative multi-agent teams
3. Multi-Agent Competition: Train/evaluate competitive agents
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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
from gym_gui.ui.worker_catalog import WorkerDefinition, get_worker_catalog
from gym_gui.ui.widgets.human_vs_agent_config_form import (
    HumanVsAgentConfigForm,
    HumanVsAgentConfig,
    DIFFICULTY_PRESETS,
)
from gym_gui.ui.widgets.policy_assignment_panel import PolicyAssignmentPanel
from gym_gui.ui.widgets.load_policy_dialog import LoadPolicyDialog

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
    ai_opponent_changed = pyqtSignal(str, str)  # opponent_type, difficulty

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._selected_env: Optional[PettingZooEnvId] = None
        self._selected_env_value: Optional[str] = None  # For non-PettingZoo envs like OpenSpiel Checkers
        self._default_seed = 42
        self._allow_seed_reuse = False
        self._environment_loaded = False
        self._policy_loaded = False
        # AI opponent configuration (default to Stockfish medium)
        self._ai_config = HumanVsAgentConfig()
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

        # Environment Configuration group
        config_group = QtWidgets.QGroupBox("Environment Configuration", self)
        config_layout = QtWidgets.QVBoxLayout(config_group)

        # Configure button
        config_btn_layout = QtWidgets.QHBoxLayout()
        self._configure_btn = QtWidgets.QPushButton("Configure AI Opponent...", config_group)
        self._configure_btn.setToolTip(
            "Open configuration dialog to set AI opponent type, "
            "difficulty level, and advanced settings."
        )
        config_btn_layout.addWidget(self._configure_btn)
        config_btn_layout.addStretch(1)
        config_layout.addLayout(config_btn_layout)

        # Current configuration summary
        self._config_summary = QtWidgets.QLabel(self)
        self._config_summary.setWordWrap(True)
        self._config_summary.setStyleSheet(
            "color: #555; font-size: 11px; padding: 8px; "
            "background-color: #f5f5f5; border-radius: 4px;"
        )
        self._update_config_summary()
        config_layout.addWidget(self._config_summary)

        layout.addWidget(config_group)

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

        # Active AI indicator - shows what AI is ACTUALLY being used
        self._active_ai_label = QtWidgets.QLabel("—", status_group)
        self._active_ai_label.setStyleSheet(
            "font-weight: bold; padding: 2px 6px; border-radius: 3px;"
        )
        status_layout.addRow("Active AI:", self._active_ai_label)

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
        self._start_btn.clicked.connect(self._on_start_game)
        self._reset_btn.clicked.connect(self._on_reset_game)
        self._configure_btn.clicked.connect(self._on_configure_clicked)

    def _populate_environments(self) -> None:
        """Populate environment dropdowns with human-controllable games."""
        self._family_combo.clear()
        self._family_combo.addItem("Classic (Board Games)", PettingZooFamily.CLASSIC.value)

        # Populate initial environments
        self._on_family_changed(0)

    def _on_family_changed(self, index: int) -> None:
        """Handle family selection change."""
        self._env_combo.clear()

        # Get human-controllable envs from PettingZoo
        human_envs = get_human_vs_agent_envs()

        for env_id in human_envs:
            display_name = get_display_name(env_id)
            self._env_combo.addItem(display_name, env_id.value)

        # Add OpenSpiel Checkers (not a PettingZoo env, but supported via Shimmy)
        self._env_combo.addItem("Checkers (OpenSpiel)", "checkers")

        if self._env_combo.count() > 0:
            self._env_combo.setCurrentIndex(0)
            self._on_env_changed(0)

    def _on_env_changed(self, index: int) -> None:
        """Handle environment selection change."""
        env_value = self._env_combo.currentData()
        if not env_value:
            self._env_info.setText("")
            return

        # Handle OpenSpiel Checkers specially (not a PettingZoo env)
        if env_value == "checkers":
            self._selected_env = None  # Not a PettingZooEnvId
            self._selected_env_value = "checkers"
            self._env_info.setText(
                f"<b>Checkers (OpenSpiel)</b><br/>"
                f"Classic checkers/draughts board game via OpenSpiel<br/>"
                f"<i>API: AEC (Turn-based)</i>"
            )
            # Reset loaded state when environment changes
            self._environment_loaded = False
            self._policy_loaded = False
            self._start_btn.setEnabled(False)
            self._update_config_summary()
            return

        try:
            env_id = PettingZooEnvId(env_value)
            self._selected_env = env_id
            self._selected_env_value = env_value
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
            self._start_btn.setEnabled(False)
            self._update_config_summary()
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
        env_value = self._selected_env_value
        if env_value:
            seed = self._seed_spin.value()
            self.load_environment_requested.emit(env_value, seed)
            self._environment_loaded = True
            # For non-custom AI, policy is implicitly "loaded"
            if self._ai_config.opponent_type != "custom":
                self._policy_loaded = True
            self._update_button_states()
            _LOGGER.info(
                "Environment load requested: %s (seed=%d)",
                env_value,
                seed,
            )

    def _on_start_game(self) -> None:
        """Handle start game button click."""
        env_value = self._selected_env_value
        if env_value:
            human_agent = self._human_player_combo.currentData()
            seed = self._seed_spin.value()
            self.start_game_requested.emit(env_value, human_agent, seed)
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
        self._start_btn.setEnabled(self._environment_loaded and self._policy_loaded)

    def set_policy_loaded(self, policy_path: str) -> None:
        """Update UI when a policy is loaded."""
        self._policy_loaded = True
        self._update_button_states()
        self._update_config_summary()

    def set_environment_loaded(self, env_id: str, seed: int) -> None:
        """Update UI when environment is loaded."""
        self._environment_loaded = True
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

    def set_active_ai(self, ai_name: str, is_fallback: bool = False) -> None:
        """Update the active AI indicator to show what AI is actually being used.

        Args:
            ai_name: Display name of the AI (e.g., "Stockfish (Medium)", "Random AI")
            is_fallback: True if this is a fallback from the requested AI
        """
        self._active_ai_label.setText(ai_name)

        if is_fallback:
            # Yellow/orange warning style - fallback from requested AI
            self._active_ai_label.setStyleSheet(
                "font-weight: bold; padding: 2px 6px; border-radius: 3px; "
                "background-color: #fff3cd; color: #856404; border: 1px solid #ffc107;"
            )
            self._active_ai_label.setToolTip(
                "Fallback: The requested AI (Stockfish) was not available. "
                "Install with: sudo apt install stockfish"
            )
        elif "stockfish" in ai_name.lower():
            # Green success style - Stockfish is running
            self._active_ai_label.setStyleSheet(
                "font-weight: bold; padding: 2px 6px; border-radius: 3px; "
                "background-color: #d4edda; color: #155724; border: 1px solid #28a745;"
            )
            self._active_ai_label.setToolTip("Stockfish engine is active")
        elif "katago" in ai_name.lower():
            # Green success style - KataGo is running
            self._active_ai_label.setStyleSheet(
                "font-weight: bold; padding: 2px 6px; border-radius: 3px; "
                "background-color: #d4edda; color: #155724; border: 1px solid #28a745;"
            )
            self._active_ai_label.setToolTip("KataGo engine is active (superhuman strength)")
        elif "gnu go" in ai_name.lower() or "gnugo" in ai_name.lower():
            # Blue info style - GNU Go is running (weaker than KataGo but still good)
            self._active_ai_label.setStyleSheet(
                "font-weight: bold; padding: 2px 6px; border-radius: 3px; "
                "background-color: #cce5ff; color: #004085; border: 1px solid #007bff;"
            )
            self._active_ai_label.setToolTip("GNU Go engine is active (amateur dan level)")
        else:
            # Neutral style - Random AI (as selected)
            self._active_ai_label.setStyleSheet(
                "font-weight: bold; padding: 2px 6px; border-radius: 3px; "
                "background-color: #e2e3e5; color: #383d41; border: 1px solid #6c757d;"
            )
            self._active_ai_label.setToolTip("Random move selection")

    def current_seed(self) -> int:
        """Get current seed value."""
        return self._seed_spin.value()

    def current_env_id(self) -> Optional[str]:
        """Get current environment ID."""
        return self._selected_env_value

    def _on_configure_clicked(self) -> None:
        """Handle configure button click - open the configuration dialog."""
        # Determine game type from selected environment
        game_type = self._get_game_type()
        dialog = HumanVsAgentConfigForm(self, self._ai_config, game_type=game_type)
        dialog.config_accepted.connect(self._on_config_accepted)
        dialog.exec()

    def _get_game_type(self) -> str:
        """Get game type ('chess', 'go', or 'checkers') from selected environment."""
        env_id = self._selected_env_value
        if env_id is None:
            return "chess"  # Default

        if env_id in ("go_v5", "go"):
            return "go"
        elif env_id == "checkers":
            return "checkers"
        else:
            return "chess"  # Chess, Connect Four, Tic-Tac-Toe all use similar config

    def _on_config_accepted(self, config: HumanVsAgentConfig) -> None:
        """Handle configuration dialog accepted."""
        self._ai_config = config
        self._update_config_summary()

        # Mark policy as loaded for non-custom opponents
        if config.opponent_type != "custom":
            self._policy_loaded = True
        else:
            self._policy_loaded = config.custom_policy_path is not None

        self._update_button_states()

        # Emit signal for main window to update AI provider
        self.ai_opponent_changed.emit(config.opponent_type, config.difficulty)
        _LOGGER.info(
            "AI config updated: type=%s, difficulty=%s",
            config.opponent_type,
            config.difficulty,
        )

    def _update_config_summary(self) -> None:
        """Update the configuration summary label.

        Shows appropriate AI opponent info based on the selected game:
        - Chess: Stockfish engine available with difficulty options
        - Go: KataGo/GNU Go available with difficulty options
        - Connect Four/Tic-Tac-Toe: Random AI only (no dedicated engine)
        """
        config = self._ai_config

        # Check which game is selected
        is_chess = (
            self._selected_env is not None
            and self._selected_env.value == "chess_v6"
        )
        is_go = (
            self._selected_env is not None
            and self._selected_env.value == "go_v5"
        )

        # For games without dedicated AI engines
        if not is_chess and not is_go and self._selected_env is not None:
            summary = (
                "<b>AI Opponent:</b> Random AI<br>"
                "Makes random legal moves. (No dedicated AI engine for this game.)"
            )
            self._config_summary.setText(summary)
            return

        # Go-specific AI summary
        if is_go:
            if config.opponent_type == "katago":
                summary = (
                    f"<b>AI Opponent:</b> KataGo ({config.difficulty.capitalize()})<br>"
                    "Superhuman-strength Go AI (requires katago + model)."
                )
            elif config.opponent_type == "gnugo":
                summary = (
                    f"<b>AI Opponent:</b> GNU Go ({config.difficulty.capitalize()})<br>"
                    "Classical Go AI (amateur dan level). Install: sudo apt install gnugo"
                )
            else:
                summary = (
                    "<b>AI Opponent:</b> Random AI<br>"
                    "Makes random legal moves. Install gnugo or katago for stronger AI."
                )
            self._config_summary.setText(summary)
            return

        # Chess-specific AI summary
        if config.opponent_type == "random":
            summary = (
                "<b>AI Opponent:</b> Random<br>"
                "Makes random legal moves (for testing)."
            )
        elif config.opponent_type == "stockfish":
            preset = DIFFICULTY_PRESETS.get(config.difficulty)
            summary = (
                f"<b>AI Opponent:</b> Stockfish ({config.difficulty.capitalize()})<br>"
                f"Skill: {config.stockfish.skill_level}, "
                f"Depth: {config.stockfish.depth}, "
                f"Time: {config.stockfish.time_limit_ms}ms"
            )
        elif config.opponent_type == "custom":
            if config.custom_policy_path:
                summary = (
                    f"<b>AI Opponent:</b> Custom Policy<br>"
                    f"Path: {config.custom_policy_path}"
                )
            else:
                summary = (
                    "<b>AI Opponent:</b> Custom Policy<br>"
                    "<i>No policy loaded yet.</i>"
                )
        else:
            summary = "<i>Click 'Configure AI Opponent' to set up.</i>"

        self._config_summary.setText(summary)

    def get_ai_config(self) -> HumanVsAgentConfig:
        """Get the current AI opponent configuration.

        Returns:
            Full HumanVsAgentConfig object
        """
        return self._ai_config

    def get_ai_opponent_config(self) -> tuple[str, str]:
        """Get the current AI opponent configuration (legacy interface).

        Returns:
            Tuple of (opponent_type, difficulty)
        """
        return (self._ai_config.opponent_type, self._ai_config.difficulty)


class MultiAgentCooperationTab(QtWidgets.QWidget):
    """Tab for cooperative multi-agent training and evaluation.

    Supports environments where agents must work together to achieve
    a common goal.
    """

    # Signals
    worker_changed = pyqtSignal(str)  # worker_id
    train_requested = pyqtSignal(str, str)  # worker_id, env_id
    evaluate_requested = pyqtSignal(str, str)  # worker_id, env_id
    policy_evaluate_requested = pyqtSignal(dict)  # Full evaluation config with policies

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._workers: List[WorkerDefinition] = []
        self._current_agent_ids: List[str] = []
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

        # Policy Assignment panel
        self._policy_panel = PolicyAssignmentPanel(self)
        layout.addWidget(self._policy_panel)

        layout.addStretch(1)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._worker_combo.currentIndexChanged.connect(self._on_worker_changed)
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)
        self._train_btn.clicked.connect(self._on_train)
        self._eval_btn.clicked.connect(self._on_evaluate)
        self._policy_panel.evaluate_requested.connect(self._on_policy_evaluate)

    def _on_policy_evaluate(self, config: dict) -> None:
        """Handle policy evaluation request from the panel."""
        # Add environment info to config
        config["env_id"] = self._env_combo.currentData()
        config["env_family"] = self._family_combo.currentData()
        config["worker_id"] = self._worker_combo.currentData()
        self.policy_evaluate_requested.emit(config)

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
            self._policy_panel.set_agents([])
            return

        try:
            env_id = PettingZooEnvId(env_value)
            description = get_description(env_id)
            api_type = get_api_type(env_id)
            self._env_info.setText(f"{description}\n(API: {api_type.value.upper()})")

            # Detect agents for this environment
            agent_ids = self._detect_agents(env_value)
            self._current_agent_ids = agent_ids
            self._policy_panel.set_agents(agent_ids)

        except ValueError:
            self._env_info.setText("")
            self._policy_panel.set_agents([])

    def _detect_agents(self, env_id: str) -> List[str]:
        """Detect agent IDs for an environment.

        Args:
            env_id: PettingZoo environment ID

        Returns:
            List of agent IDs
        """
        family = self._family_combo.currentData()
        if not family:
            return []

        try:
            # Try to create environment and get agents
            from gym_gui.core.adapters.pettingzoo import PettingZooAdapter, PettingZooConfig
            from gym_gui.core.pettingzoo_enums import PettingZooEnvId, PettingZooFamily as PZFamily

            # Create config for the environment
            try:
                pz_env_id = PettingZooEnvId(env_id)
                pz_family = PZFamily(family)
            except ValueError:
                return ["agent_0", "agent_1"]

            config = PettingZooConfig(env_id=pz_env_id, family=pz_family)
            adapter = PettingZooAdapter(config=config)
            adapter.load()

            # Get possible agents from the underlying environment
            if hasattr(adapter, '_pz_env') and adapter._pz_env is not None:
                agent_ids = list(adapter._pz_env.possible_agents)
            else:
                agent_ids = []
            adapter.close()
            return agent_ids
        except Exception as e:
            _LOGGER.warning("Could not detect agents for %s: %s", env_id, e)
            # Return default agent pattern based on env
            if "waterworld" in env_id:
                return [f"pursuer_{i}" for i in range(5)]
            elif "multiwalker" in env_id:
                return [f"walker_{i}" for i in range(3)]
            elif "pursuit" in env_id:
                return [f"pursuer_{i}" for i in range(8)]
            return ["agent_0", "agent_1"]

    def _on_train(self) -> None:
        """Handle train button click."""
        worker_id = self._worker_combo.currentData()
        env_id = self._env_combo.currentData()
        if worker_id and env_id:
            self.train_requested.emit(worker_id, env_id)

    def _on_evaluate(self) -> None:
        """Handle evaluate/load policy button click."""
        env_id = self._env_combo.currentData()

        # Open the LoadPolicyDialog with environment filter
        dialog = LoadPolicyDialog(
            self,
            filter_env=env_id,
            filter_worker="ray",  # Default to Ray for multi-agent
        )
        dialog.policy_selected.connect(self._on_policy_loaded)
        dialog.exec()

    def _on_policy_loaded(self, checkpoint: Any) -> None:
        """Handle policy loaded from dialog."""
        worker_id = self._worker_combo.currentData()
        env_id = self._env_combo.currentData()
        if worker_id and env_id:
            # Emit with checkpoint info
            self.evaluate_requested.emit(worker_id, env_id)
            _LOGGER.info(
                "Policy loaded for evaluation: %s (env=%s)",
                getattr(checkpoint, 'run_id', 'unknown'),
                env_id,
            )


class MultiAgentCompetitionTab(QtWidgets.QWidget):
    """Tab for competitive multi-agent training and evaluation.

    Supports environments where agents compete against each other,
    including self-play and tournament modes.
    """

    # Signals
    worker_changed = pyqtSignal(str)  # worker_id
    train_requested = pyqtSignal(str, str)  # worker_id, env_id
    evaluate_requested = pyqtSignal(str, str)  # worker_id, env_id
    policy_evaluate_requested = pyqtSignal(dict)  # Full evaluation config with policies

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._workers: List[WorkerDefinition] = []
        self._current_agent_ids: List[str] = []
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

        # Policy Assignment panel
        self._policy_panel = PolicyAssignmentPanel(self)
        layout.addWidget(self._policy_panel)

        layout.addStretch(1)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._worker_combo.currentIndexChanged.connect(self._on_worker_changed)
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)
        self._train_btn.clicked.connect(self._on_train)
        self._eval_btn.clicked.connect(self._on_evaluate)
        self._policy_panel.evaluate_requested.connect(self._on_policy_evaluate)

    def _on_policy_evaluate(self, config: dict) -> None:
        """Handle policy evaluation request from the panel."""
        config["env_id"] = self._env_combo.currentData()
        config["env_family"] = self._family_combo.currentData()
        config["worker_id"] = self._worker_combo.currentData()
        self.policy_evaluate_requested.emit(config)

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
            self._policy_panel.set_agents([])
            return

        try:
            env_id = PettingZooEnvId(env_value)
            description = get_description(env_id)
            api_type = get_api_type(env_id)
            self._env_info.setText(f"{description}\n(API: {api_type.value.upper()})")

            # Detect agents for this environment
            agent_ids = self._detect_agents(env_value)
            self._current_agent_ids = agent_ids
            self._policy_panel.set_agents(agent_ids)

        except ValueError:
            self._env_info.setText("")
            self._policy_panel.set_agents([])

    def _detect_agents(self, env_id: str) -> List[str]:
        """Detect agent IDs for an environment."""
        family = self._family_combo.currentData()
        if not family:
            return []

        try:
            from gym_gui.core.adapters.pettingzoo import PettingZooAdapter, PettingZooConfig
            from gym_gui.core.pettingzoo_enums import PettingZooEnvId, PettingZooFamily as PZFamily

            # Create config for the environment
            try:
                pz_env_id = PettingZooEnvId(env_id)
                pz_family = PZFamily(family)
            except ValueError:
                return ["agent_0", "agent_1"]

            config = PettingZooConfig(env_id=pz_env_id, family=pz_family)
            adapter = PettingZooAdapter(config=config)
            adapter.load()

            # Get possible agents from the underlying environment
            if hasattr(adapter, '_pz_env') and adapter._pz_env is not None:
                agent_ids = list(adapter._pz_env.possible_agents)
            else:
                agent_ids = []
            adapter.close()
            return agent_ids
        except Exception as e:
            _LOGGER.warning("Could not detect agents for %s: %s", env_id, e)
            # Return default agents for competitive games
            if "chess" in env_id:
                return ["player_0", "player_1"]
            elif "go" in env_id:
                return ["black_0", "white_0"]
            elif "connect_four" in env_id:
                return ["player_0", "player_1"]
            elif "tictactoe" in env_id:
                return ["player_1", "player_2"]
            return ["agent_0", "agent_1"]

    def _on_train(self) -> None:
        """Handle train button click."""
        worker_id = self._worker_combo.currentData()
        env_id = self._env_combo.currentData()
        if worker_id and env_id:
            self.train_requested.emit(worker_id, env_id)

    def _on_evaluate(self) -> None:
        """Handle evaluate/load policy button click."""
        env_id = self._env_combo.currentData()

        # Open the LoadPolicyDialog with environment filter
        dialog = LoadPolicyDialog(
            self,
            filter_env=env_id,
            filter_worker="ray",  # Default to Ray for multi-agent
        )
        dialog.policy_selected.connect(self._on_policy_loaded)
        dialog.exec()

    def _on_policy_loaded(self, checkpoint: Any) -> None:
        """Handle policy loaded from dialog."""
        worker_id = self._worker_combo.currentData()
        env_id = self._env_combo.currentData()
        if worker_id and env_id:
            # Emit with checkpoint info
            self.evaluate_requested.emit(worker_id, env_id)
            _LOGGER.info(
                "Policy loaded for evaluation: %s (env=%s)",
                getattr(checkpoint, 'run_id', 'unknown'),
                env_id,
            )


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
    policy_evaluate_requested = pyqtSignal(dict)  # Full evaluation config with policies
    load_policy_requested = pyqtSignal(str)  # env_id
    load_environment_requested = pyqtSignal(str, int)  # env_id, seed
    start_game_requested = pyqtSignal(str, str, int)  # env_id, human_agent, seed
    reset_game_requested = pyqtSignal(int)  # seed
    ai_opponent_changed = pyqtSignal(str, str)  # opponent_type, difficulty

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
        self._human_vs_agent_tab.ai_opponent_changed.connect(
            self.ai_opponent_changed
        )

        # Cooperation
        self._cooperation_tab.worker_changed.connect(self.worker_changed)
        self._cooperation_tab.train_requested.connect(self.train_requested)
        self._cooperation_tab.evaluate_requested.connect(self.evaluate_requested)
        self._cooperation_tab.policy_evaluate_requested.connect(self.policy_evaluate_requested)

        # Competition
        self._competition_tab.worker_changed.connect(self.worker_changed)
        self._competition_tab.train_requested.connect(self.train_requested)
        self._competition_tab.evaluate_requested.connect(self.evaluate_requested)
        self._competition_tab.policy_evaluate_requested.connect(self.policy_evaluate_requested)

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
