from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]

from gym_gui.config.game_configs import (
    CliffWalkingConfig,
    CarRacingConfig,
    BipedalWalkerConfig,
    FrozenLakeConfig,
    LunarLanderConfig,
    TaxiConfig,
    DEFAULT_FROZEN_LAKE_V2_CONFIG,
)
from gym_gui.core.enums import ControlMode, GameId
from gym_gui.services.actor import ActorDescriptor


@dataclass(frozen=True)
class ControlPanelConfig:
    """Configuration for the control panel with game-specific configs."""
    
    available_modes: Dict[GameId, Iterable[ControlMode]]
    default_mode: ControlMode
    frozen_lake_config: FrozenLakeConfig
    taxi_config: TaxiConfig
    cliff_walking_config: CliffWalkingConfig
    lunar_lander_config: LunarLanderConfig
    car_racing_config: CarRacingConfig
    bipedal_walker_config: BipedalWalkerConfig
    default_seed: int
    allow_seed_reuse: bool
    actors: tuple[ActorDescriptor, ...]
    default_actor_id: Optional[str] = None


class ControlPanelWidget(QtWidgets.QWidget):
    control_mode_changed = pyqtSignal(ControlMode)
    game_changed = pyqtSignal(GameId)
    load_requested = pyqtSignal(GameId, ControlMode, int)
    reset_requested = pyqtSignal(int)
    agent_form_requested = pyqtSignal()
    slippery_toggled = pyqtSignal(bool)
    frozen_v2_config_changed = pyqtSignal(str, object)  # (param_name, value)
    taxi_config_changed = pyqtSignal(str, bool)  # (param_name, value)
    cliff_config_changed = pyqtSignal(str, bool)  # (param_name, value)
    lunar_config_changed = pyqtSignal(str, object)  # (param_name, value)
    car_config_changed = pyqtSignal(str, object)  # (param_name, value)
    bipedal_config_changed = pyqtSignal(str, object)  # (param_name, value)
    start_game_requested = pyqtSignal()
    pause_game_requested = pyqtSignal()
    continue_game_requested = pyqtSignal()
    terminate_game_requested = pyqtSignal()
    agent_step_requested = pyqtSignal()
    actor_changed = pyqtSignal(str)
    train_agent_requested = pyqtSignal()  # New signal for headless training
    trained_agent_requested = pyqtSignal()  # Load trained policy/evaluation

    def __init__(
        self,
        *,
        config: ControlPanelConfig,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._available_modes = config.available_modes
        self._default_seed = max(1, config.default_seed)
        self._allow_seed_reuse = config.allow_seed_reuse
        self._game_overrides: Dict[GameId, Dict[str, object]] = {
            GameId.FROZEN_LAKE: {
                "is_slippery": config.frozen_lake_config.is_slippery,
                "success_rate": config.frozen_lake_config.success_rate,
                "reward_schedule": config.frozen_lake_config.reward_schedule,
            },
            GameId.FROZEN_LAKE_V2: {
                "is_slippery": DEFAULT_FROZEN_LAKE_V2_CONFIG.is_slippery,
                "success_rate": DEFAULT_FROZEN_LAKE_V2_CONFIG.success_rate,
                "reward_schedule": DEFAULT_FROZEN_LAKE_V2_CONFIG.reward_schedule,
                "grid_height": DEFAULT_FROZEN_LAKE_V2_CONFIG.grid_height,
                "grid_width": DEFAULT_FROZEN_LAKE_V2_CONFIG.grid_width,
                "start_position": DEFAULT_FROZEN_LAKE_V2_CONFIG.start_position,
                "goal_position": DEFAULT_FROZEN_LAKE_V2_CONFIG.goal_position,
                "hole_count": DEFAULT_FROZEN_LAKE_V2_CONFIG.hole_count,
                "random_holes": DEFAULT_FROZEN_LAKE_V2_CONFIG.random_holes,
            },
            GameId.TAXI: {
                "is_raining": config.taxi_config.is_raining,
                "fickle_passenger": config.taxi_config.fickle_passenger,
            },
            GameId.CLIFF_WALKING: {"is_slippery": config.cliff_walking_config.is_slippery},
            GameId.LUNAR_LANDER: {
                "continuous": config.lunar_lander_config.continuous,
                "gravity": config.lunar_lander_config.gravity,
                "enable_wind": config.lunar_lander_config.enable_wind,
                "wind_power": config.lunar_lander_config.wind_power,
                "turbulence_power": config.lunar_lander_config.turbulence_power,
                "max_episode_steps": config.lunar_lander_config.max_episode_steps,
            },
            GameId.CAR_RACING: {
                "continuous": config.car_racing_config.continuous,
                "domain_randomize": config.car_racing_config.domain_randomize,
                "lap_complete_percent": config.car_racing_config.lap_complete_percent,
                "max_episode_steps": config.car_racing_config.max_episode_steps,
                "max_episode_seconds": config.car_racing_config.max_episode_seconds,
            },
            GameId.BIPEDAL_WALKER: {
                "hardcore": config.bipedal_walker_config.hardcore,
                "max_episode_steps": config.bipedal_walker_config.max_episode_steps,
                "max_episode_seconds": config.bipedal_walker_config.max_episode_seconds,
            },
        }

        self._current_game: Optional[GameId] = None
        self._current_mode: ControlMode = self._load_mode_preference(config.default_mode)
        self._awaiting_human: bool = False
        self._auto_running: bool = False
        self._game_started: bool = False
        self._game_paused: bool = False
        self._actor_descriptors: Dict[str, ActorDescriptor] = {
            descriptor.actor_id: descriptor for descriptor in config.actors
        }
        self._actor_order: tuple[ActorDescriptor, ...] = config.actors
        default_actor = config.default_actor_id
        if default_actor is None and self._actor_order:
            default_actor = self._actor_order[0].actor_id
        self._active_actor_id: Optional[str] = default_actor

        # Store game configurations
        self._game_configs: Dict[GameId, Dict[str, object]] = {
            GameId.FROZEN_LAKE: {
                "is_slippery": config.frozen_lake_config.is_slippery
            },
            GameId.TAXI: {
                "is_raining": config.taxi_config.is_raining,
                "fickle_passenger": config.taxi_config.fickle_passenger,
            },
            GameId.CLIFF_WALKING: {
                "is_slippery": config.cliff_walking_config.is_slippery,
            },
            GameId.LUNAR_LANDER: {
                "continuous": config.lunar_lander_config.continuous,
                "gravity": config.lunar_lander_config.gravity,
                "enable_wind": config.lunar_lander_config.enable_wind,
                "wind_power": config.lunar_lander_config.wind_power,
                "turbulence_power": config.lunar_lander_config.turbulence_power,
                "max_episode_steps": config.lunar_lander_config.max_episode_steps,
            },
            GameId.CAR_RACING: {
                "continuous": config.car_racing_config.continuous,
                "domain_randomize": config.car_racing_config.domain_randomize,
                "lap_complete_percent": config.car_racing_config.lap_complete_percent,
                "max_episode_steps": config.car_racing_config.max_episode_steps,
                "max_episode_seconds": config.car_racing_config.max_episode_seconds,
            },
            GameId.BIPEDAL_WALKER: {
                "hardcore": config.bipedal_walker_config.hardcore,
                "max_episode_steps": config.bipedal_walker_config.max_episode_steps,
                "max_episode_seconds": config.bipedal_walker_config.max_episode_seconds,
            },
        }

        self._build_ui()
        self._apply_current_mode_selection()
        self._connect_signals()
        self._update_control_states()
        self._populate_actor_combo()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def populate_games(self, games: Iterable[GameId], *, default: Optional[GameId] = None) -> None:
        games_tuple = tuple(games)
        self._game_combo.blockSignals(True)
        self._game_combo.clear()
        for game in games_tuple:
            self._game_combo.addItem(game.value, game)
        self._game_combo.blockSignals(False)

        if not games_tuple:
            return

        chosen = default if (default is not None and default in games_tuple) else games_tuple[0]
        index = self._game_combo.findData(chosen)
        if index < 0:
            index = 0
        self._game_combo.setCurrentIndex(index)
        chosen_game = self._game_combo.itemData(index)
        if isinstance(chosen_game, GameId):
            self._emit_game_changed(chosen_game)

    def current_actor(self) -> Optional[str]:
        return self._active_actor_id

    def set_active_actor(self, actor_id: str) -> None:
        if actor_id == self._active_actor_id:
            return
        index = self._actor_combo.findData(actor_id)
        if index < 0:
            return
        self._actor_combo.blockSignals(True)
        self._actor_combo.setCurrentIndex(index)
        self._actor_combo.blockSignals(False)
        self._active_actor_id = actor_id
        self._update_actor_description()

    def update_modes(self, game_id: GameId) -> None:
        supported = tuple(self._available_modes.get(game_id, ()))
        if not supported:
            self._mode_combo.setEnabled(False)
            return

        self._mode_combo.blockSignals(True)
        self._mode_combo.clear()
        for mode in supported:
            label = mode.value.replace("_", " ").title()
            self._mode_combo.addItem(label, mode)
        self._mode_combo.blockSignals(False)
        self._mode_combo.setEnabled(bool(supported))

        if self._current_mode not in supported:
            self._current_mode = supported[0]
            self._persist_mode_preference(self._current_mode)
            index = self._mode_combo.findData(self._current_mode)
            if index >= 0:
                self._mode_combo.blockSignals(True)
                self._mode_combo.setCurrentIndex(index)
                self._mode_combo.blockSignals(False)
            self._emit_mode_changed(self._current_mode)
            return

        index = self._mode_combo.findData(self._current_mode)
        if index >= 0:
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentIndex(index)
            self._mode_combo.blockSignals(False)
        self._update_control_states()

    def set_status(
        self,
        *,
        step: int,
        reward: float,
        total_reward: float,
        terminated: bool,
        truncated: bool,
        turn: str,
        awaiting_human: bool,
        session_time: str,
        active_time: str,
        episode_duration: str,
        outcome_time: str = "—",
        outcome_wall_clock: str | None = None,
    ) -> None:
        self._step_label.setText(str(step))
        self._reward_label.setText(f"{reward:.2f}")
        self._total_reward_label.setText(f"{total_reward:.2f}")
        self._terminated_label.setText(self._format_bool(terminated))
        self._truncated_label.setText(self._format_bool(truncated))
        self._turn_label.setText(turn)
        self.set_awaiting_human(awaiting_human)
        self.set_time_labels(
            session_time,
            active_time,
            outcome_time,
            outcome_timestamp=outcome_wall_clock,
        )

    def set_turn(self, turn: str) -> None:
        self._turn_label.setText(turn)

    def set_mode(self, mode: ControlMode) -> None:
        index = self._mode_combo.findData(mode)
        if index < 0:
            return
        self._current_mode = mode
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(index)
        self._mode_combo.blockSignals(False)
        self._emit_mode_changed(mode)

    def set_game(self, game: GameId) -> None:
        index = self._game_combo.findData(game)
        if index >= 0:
            self._game_combo.blockSignals(True)
            self._game_combo.setCurrentIndex(index)
            self._game_combo.blockSignals(False)
            self._emit_game_changed(game)

    def override_slippery(self, enabled: bool) -> None:
        overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE, {})
        overrides["is_slippery"] = enabled
        if self._frozen_slippery_checkbox is not None:
            self._frozen_slippery_checkbox.blockSignals(True)
            self._frozen_slippery_checkbox.setChecked(enabled)
            self._frozen_slippery_checkbox.blockSignals(False)

    def current_seed(self) -> int:
        return int(self._seed_spin.value())

    def current_mode(self) -> ControlMode:
        return self._current_mode

    def current_game(self) -> Optional[GameId]:
        return self._current_game

    def get_overrides(self, game_id: GameId) -> Dict[str, object]:
        return dict(self._game_overrides.get(game_id, {}))

    def set_auto_running(self, running: bool) -> None:
        self._auto_running = running
        self._update_control_states()

    def set_game_started(self, started: bool) -> None:
        """Set whether the game has been started."""
        self._game_started = started
        if not started:
            self._game_paused = False
        self._update_control_states()

    def set_game_paused(self, paused: bool) -> None:
        """Set whether the game is paused."""
        self._game_paused = paused
        self._update_control_states()

    def set_slippery_visible(self, visible: bool) -> None:
        if self._frozen_slippery_checkbox is not None:
            self._frozen_slippery_checkbox.setVisible(visible)

    def set_awaiting_human(self, awaiting: bool) -> None:
        self._awaiting_human = awaiting
        self._awaiting_label.setText("Yes" if awaiting else "No")
        self._update_control_states()

    def set_time_labels(
        self,
        session_time: str,
        active_time: str,
        outcome_time: str,
        *,
        outcome_timestamp: str | None = None,
    ) -> None:
        self._session_time_label.setText(session_time)
        self._active_time_label.setText(active_time)
        self._outcome_time_label.setText(outcome_time)
        tooltip = "Elapsed time between the first move and the recorded outcome."
        if outcome_timestamp and outcome_timestamp != "—":
            tooltip += f"\nOutcome recorded at {outcome_timestamp}."
        elif outcome_timestamp == "—":
            tooltip += "\nOutcome not recorded yet."
        self._outcome_time_label.setToolTip(tooltip)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Environment picker
        env_group = QtWidgets.QGroupBox("Environment", self)
        env_layout = QtWidgets.QVBoxLayout(env_group)
        self._game_combo = QtWidgets.QComboBox(env_group)
        self._seed_spin = QtWidgets.QSpinBox(env_group)
        self._seed_spin.setRange(1, 10_000_000)
        self._seed_spin.setValue(self._default_seed)
        if self._allow_seed_reuse:
            self._seed_spin.setToolTip(
                "Seeds auto-increment by default. Adjust before loading to reuse a previous seed."
            )
        else:
            self._seed_spin.setToolTip("Seed increments automatically after each episode.")
        self._load_button = QtWidgets.QPushButton("Load", env_group)
        env_layout.addWidget(QtWidgets.QLabel("Select environment", env_group))
        env_layout.addWidget(self._game_combo)
        env_layout.addWidget(QtWidgets.QLabel("Seed", env_group))
        env_layout.addWidget(self._seed_spin)
        env_layout.addWidget(self._load_button)
        layout.addWidget(env_group)

        # Game configuration placeholder
        self._config_group = QtWidgets.QGroupBox("Game Configuration", self)
        self._config_layout = QtWidgets.QFormLayout(self._config_group)
        layout.addWidget(self._config_group)
        self._frozen_slippery_checkbox: Optional[QtWidgets.QCheckBox] = None

        # Mode selector (QComboBox)
        mode_group = QtWidgets.QGroupBox("Control Mode", self)
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self._mode_combo = QtWidgets.QComboBox(self)
        self._mode_combo.setEnabled(False)
        
        # Populate combo box with all control modes
        modes_tuple = tuple(ControlMode)
        for mode in modes_tuple:
            label = mode.value.replace("_", " ").title()
            self._mode_combo.addItem(label, mode)
        
        mode_layout.addWidget(self._mode_combo)
        layout.addWidget(mode_group)

        # Actor selector
        actor_group = QtWidgets.QGroupBox("Active Actor", self)
        actor_layout = QtWidgets.QVBoxLayout(actor_group)
        self._actor_combo = QtWidgets.QComboBox(actor_group)
        self._actor_combo.setEnabled(bool(self._actor_order))
        actor_layout.addWidget(self._actor_combo)
        self._actor_description = QtWidgets.QLabel("—", actor_group)
        self._actor_description.setWordWrap(True)
        actor_layout.addWidget(self._actor_description)
        layout.addWidget(actor_group)

        # Train Agent button (headless mode)
        train_group = QtWidgets.QGroupBox("Headless Training", self)
        train_layout = QtWidgets.QVBoxLayout(train_group)
        self._configure_agent_button = QtWidgets.QPushButton("🚀 Configure Agent…", train_group)
        self._configure_agent_button.setToolTip(
            "Open the agent training form to configure the backend used for headless training."
        )
        self._configure_agent_button.setEnabled(False)
        self._configure_agent_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 8px; background-color: #455a64; color: white; }"
            "QPushButton:hover { background-color: #37474f; }"
            "QPushButton:pressed { background-color: #263238; }"
            "QPushButton:disabled { background-color: #9ea7aa; color: #ECEFF1; }"
        )
        train_layout.addWidget(self._configure_agent_button)
        self._train_agent_button = QtWidgets.QPushButton("🤖 Train Agent", train_group)
        self._train_agent_button.setToolTip(
            "Submit a headless training run to the trainer daemon.\n"
            "Training will run in the background with live telemetry streaming."
        )
        self._train_agent_button.setEnabled(False)
        self._train_agent_button.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 8px; background-color: #1976d2; color: white; }"
            "QPushButton:hover { background-color: #1565c0; }"
            "QPushButton:pressed { background-color: #0d47a1; }"
            "QPushButton:disabled { background-color: #90caf9; color: #E3F2FD; }"
        )
        train_layout.addWidget(self._train_agent_button)
        self._trained_agent_button = QtWidgets.QPushButton("📦 Load Trained Policy", train_group)
        self._trained_agent_button.setToolTip(
            "Select an existing policy or checkpoint to evaluate inside the GUI."
        )
        self._trained_agent_button.setEnabled(False)
        self._trained_agent_button.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; background-color: #388e3c; color: white; }"
            "QPushButton:hover { background-color: #2e7d32; }"
            "QPushButton:pressed { background-color: #1b5e20; }"
            "QPushButton:disabled { background-color: #a5d6a7; color: #E8F5E9; }"
        )
        train_layout.addWidget(self._trained_agent_button)
        layout.addWidget(train_group)

        # Control buttons - renamed group
        controls_group = QtWidgets.QGroupBox("Game Control Flow", self)
        controls_layout = QtWidgets.QVBoxLayout(controls_group)
        
        # First row: Start, Pause, Continue
        row1_layout = QtWidgets.QHBoxLayout()
        self._start_button = QtWidgets.QPushButton("Start Game", controls_group)
        self._pause_button = QtWidgets.QPushButton("Pause Game", controls_group)
        self._continue_button = QtWidgets.QPushButton("Continue Game", controls_group)
        row1_layout.addWidget(self._start_button)
        row1_layout.addWidget(self._pause_button)
        row1_layout.addWidget(self._continue_button)
        controls_layout.addLayout(row1_layout)
        
        # Second row: Terminate, Agent Step, Reset
        row2_layout = QtWidgets.QHBoxLayout()
        self._terminate_button = QtWidgets.QPushButton("Terminate Game", controls_group)
        self._step_button = QtWidgets.QPushButton("Agent Step", controls_group)
        self._reset_button = QtWidgets.QPushButton("Reset", controls_group)
        row2_layout.addWidget(self._terminate_button)
        row2_layout.addWidget(self._step_button)
        row2_layout.addWidget(self._reset_button)
        controls_layout.addLayout(row2_layout)
        
        layout.addWidget(controls_group)

        # Status group
        status_group = QtWidgets.QGroupBox("Status", self)
        status_layout = QtWidgets.QGridLayout(status_group)
        self._step_label = QtWidgets.QLabel("0", status_group)
        self._reward_label = QtWidgets.QLabel("0.0", status_group)
        self._total_reward_label = QtWidgets.QLabel("0.00", status_group)
        self._terminated_label = QtWidgets.QLabel("No", status_group)
        self._truncated_label = QtWidgets.QLabel("No", status_group)
        self._turn_label = QtWidgets.QLabel("human", status_group)
        self._awaiting_label = QtWidgets.QLabel("–", status_group)
        self._session_time_label = QtWidgets.QLabel("00:00:00", status_group)
        self._active_time_label = QtWidgets.QLabel("—", status_group)
        self._outcome_time_label = QtWidgets.QLabel("—", status_group)

        fields: list[tuple[str, QtWidgets.QLabel]] = [
            ("Step", self._step_label),
            ("Reward", self._reward_label),
            ("Total Reward", self._total_reward_label),
            ("Episode Finished", self._terminated_label),
            ("Episode Aborted", self._truncated_label),
            ("Turn", self._turn_label),
            ("Awaiting Input", self._awaiting_label),
            ("Session Uptime", self._session_time_label),
            ("Active Play Time", self._active_time_label),
            ("Outcome Time", self._outcome_time_label),
        ]

        midpoint = (len(fields) + 1) // 2
        columns = [fields[:midpoint], fields[midpoint:]]
        for col_index, column_fields in enumerate(columns):
            for row_index, (title, value_label) in enumerate(column_fields):
                title_label = QtWidgets.QLabel(f"{title}", status_group)
                title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
                base_col = col_index * 2
                status_layout.addWidget(title_label, row_index, base_col)
                status_layout.addWidget(value_label, row_index, base_col + 1)

        status_layout.setColumnStretch(1, 1)
        status_layout.setColumnStretch(3, 1)
        layout.addWidget(status_group)

        layout.addStretch(1)

    def _connect_signals(self) -> None:
        self._game_combo.currentIndexChanged.connect(self._on_game_changed)
        self._seed_spin.valueChanged.connect(lambda _: self._update_control_states())
        self._wire_mode_combo()
        self._configure_agent_button.clicked.connect(self.agent_form_requested.emit)

        self._load_button.clicked.connect(self._on_load_clicked)
        self._train_agent_button.clicked.connect(self.train_agent_requested.emit)
        self._trained_agent_button.clicked.connect(self.trained_agent_requested.emit)
        self._start_button.clicked.connect(self._on_start_clicked)
        self._pause_button.clicked.connect(self._on_pause_clicked)
        self._continue_button.clicked.connect(self._on_continue_clicked)
        self._terminate_button.clicked.connect(self._on_terminate_clicked)
        self._step_button.clicked.connect(self._on_step_clicked)
        self._reset_button.clicked.connect(self._on_reset_clicked)
        self._actor_combo.currentIndexChanged.connect(self._on_actor_selection_changed)

    def set_seed_value(self, seed: int) -> None:
        clamped = max(1, min(seed, self._seed_spin.maximum()))
        if self._seed_spin.value() == clamped:
            return
        self._seed_spin.blockSignals(True)
        self._seed_spin.setValue(clamped)
        self._seed_spin.blockSignals(False)
        self._update_control_states()

    # ------------------------------------------------------------------
    # Signal emitters
    # ------------------------------------------------------------------
    def _emit_mode_changed(self, mode: ControlMode) -> None:
        if self._current_mode != mode:
            self._current_mode = mode
            self._persist_mode_preference(mode)
        self.control_mode_changed.emit(mode)
        self._update_control_states()

    def _emit_game_changed(self, game: GameId | None) -> None:
        if game is None:
            return
        if self._current_game != game:
            self._current_game = game
        self.game_changed.emit(game)
        self.update_modes(game)
        self._refresh_game_config_ui()
        self.set_slippery_visible(game == GameId.FROZEN_LAKE)

    # ------------------------------------------------------------------
    # Qt slots
    # ------------------------------------------------------------------
    def _on_game_changed(self, index: int) -> None:
        game = self._game_combo.itemData(index)
        if isinstance(game, GameId):
            self._emit_game_changed(game)

    def _on_load_clicked(self) -> None:
        if self._current_game is None:
            return
        self.load_requested.emit(self._current_game, self._current_mode, self.current_seed())

    def _on_start_clicked(self) -> None:
        self._game_started = True
        self._game_paused = False
        self._update_control_states()
        self.start_game_requested.emit()

    def _on_pause_clicked(self) -> None:
        self._game_paused = True
        self._update_control_states()
        self.pause_game_requested.emit()

    def _on_continue_clicked(self) -> None:
        self._game_paused = False
        self._update_control_states()
        self.continue_game_requested.emit()

    def _on_terminate_clicked(self) -> None:
        self._game_started = False
        self._game_paused = False
        self._update_control_states()
        self.terminate_game_requested.emit()

    def _on_step_clicked(self) -> None:
        self.agent_step_requested.emit()

    def _on_reset_clicked(self) -> None:
        self._game_started = False
        self._game_paused = False
        self._update_control_states()
        self.reset_requested.emit(self.current_seed())

    def _on_slippery_toggled(self, state: int) -> None:
        try:
            state_enum = QtCore.Qt.CheckState(state)
        except ValueError:
            state_enum = QtCore.Qt.CheckState.Unchecked
        enabled = state_enum == QtCore.Qt.CheckState.Checked
        overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE, {})
        overrides["is_slippery"] = enabled
        self.slippery_toggled.emit(enabled)

    def _on_taxi_config_changed(self, param_name: str, value: bool) -> None:
        """Handle changes to Taxi configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.TAXI, {})
        overrides[param_name] = value
        self.taxi_config_changed.emit(param_name, value)

    def _on_cliff_config_changed(self, param_name: str, value: bool) -> None:
        """Handle changes to CliffWalking configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.CLIFF_WALKING, {})
        overrides[param_name] = value
        self.cliff_config_changed.emit(param_name, value)

    def _on_lunar_config_changed(self, param_name: str, value: object) -> None:
        """Handle changes to LunarLander configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.LUNAR_LANDER, {})
        overrides[param_name] = value
        self.lunar_config_changed.emit(param_name, value)

    def _on_car_config_changed(self, param_name: str, value: object) -> None:
        """Handle changes to CarRacing configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.CAR_RACING, {})
        overrides[param_name] = value
        self.car_config_changed.emit(param_name, value)

    def _on_bipedal_config_changed(self, param_name: str, value: object) -> None:
        """Handle changes to BipedalWalker configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.BIPEDAL_WALKER, {})
        overrides[param_name] = value
        self.bipedal_config_changed.emit(param_name, value)

    def _on_frozen_v2_config_changed(self, param_name: str, value: object) -> None:
        """Handle changes to FrozenLake-v2 configuration parameters."""
        overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE_V2, {})
        overrides[param_name] = value
        self.frozen_v2_config_changed.emit(param_name, value)

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------
    def _wire_mode_combo(self) -> None:
        self._mode_combo.currentIndexChanged.connect(self._on_mode_selection_changed)

    def _on_mode_selection_changed(self, index: int) -> None:
        mode = self._mode_combo.itemData(index)
        if not isinstance(mode, ControlMode):
            return
        if mode == self._current_mode:
            return
        self._current_mode = mode
        self._persist_mode_preference(mode)
        self._emit_mode_changed(mode)

    def _update_control_states(self) -> None:
        """Update button states based on game flow."""
        is_human = self._current_mode == ControlMode.HUMAN_ONLY
        
        # Start button: enabled only if game not started and environment loaded
        self._start_button.setEnabled(not self._game_started)
        
        # Pause button: enabled only if game started and not paused
        self._pause_button.setEnabled(self._game_started and not self._game_paused)
        
        # Continue button: enabled only if game paused
        self._continue_button.setEnabled(self._game_paused)
        
        # Terminate button: enabled only if game started
        self._terminate_button.setEnabled(self._game_started)
        
        # Agent Step: enabled only if game started, not paused, not human-only, and not auto-running
        self._step_button.setEnabled(
            self._game_started and not self._game_paused and not is_human and not self._auto_running
        )
        
        # Reset: always enabled (can reset even during active game)
        self._reset_button.setEnabled(True)

        # Enable actor selector only when an agent can participate
        has_agent_component = self._current_mode != ControlMode.HUMAN_ONLY
        agent_only_mode = self._current_mode == ControlMode.AGENT_ONLY
        self._actor_combo.setEnabled(has_agent_component and self._actor_combo.count() > 0)
        self._configure_agent_button.setEnabled(agent_only_mode)
        self._train_agent_button.setEnabled(agent_only_mode)
        self._trained_agent_button.setEnabled(agent_only_mode)

        self._update_actor_description()

    def _apply_current_mode_selection(self) -> None:
        index = self._mode_combo.findData(self._current_mode)
        if index < 0:
            return
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(index)
        self._mode_combo.blockSignals(False)

    def _persist_mode_preference(self, mode: ControlMode) -> None:
        """Persist control mode preference to environment variable."""
        os.environ["GYM_CONTROL_MODE"] = mode.name.lower()

    def _load_mode_preference(self, fallback: ControlMode) -> ControlMode:
        """Load control mode preference from environment variable."""
        stored = os.environ.get("GYM_CONTROL_MODE")
        if stored is None:
            return fallback
        stored = stored.strip().upper()
        try:
            return ControlMode[stored]
        except KeyError:
            try:
                return ControlMode(stored.lower())
            except ValueError:
                return fallback

    def _populate_actor_combo(self) -> None:
        self._actor_combo.blockSignals(True)
        self._actor_combo.clear()
        for descriptor in self._actor_order:
            self._actor_combo.addItem(descriptor.display_name, descriptor.actor_id)
        self._actor_combo.blockSignals(False)

        if not self._actor_order:
            self._actor_description.setText("No actors registered")
            return

        default_id = self._active_actor_id or self._actor_order[0].actor_id
        index = self._actor_combo.findData(default_id)
        if index < 0:
            index = 0
        self._actor_combo.setCurrentIndex(index)
        current_data = self._actor_combo.currentData()
        self._active_actor_id = current_data if isinstance(current_data, str) else None
        self._update_actor_description()

    def _on_actor_selection_changed(self, index: int) -> None:
        actor_id = self._actor_combo.itemData(index)
        if not isinstance(actor_id, str):
            return
        if actor_id == self._active_actor_id:
            return
        self._active_actor_id = actor_id
        self._update_actor_description()
        self.actor_changed.emit(actor_id)

    def _update_actor_description(self) -> None:
        if self._active_actor_id is None:
            self._actor_description.setText("—")
            return
        descriptor = self._actor_descriptors.get(self._active_actor_id)
        if descriptor is None or descriptor.description is None:
            self._actor_description.setText("—")
            return
        self._actor_description.setText(descriptor.description)

    @staticmethod
    def _format_bool(value: bool) -> str:
        return "Yes" if value else "No"

    def _clear_config_layout(self) -> None:
        while self._config_layout.count():
            item = self._config_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            layout = item.layout()
            if layout is not None:
                layout.deleteLater()

    def _refresh_game_config_ui(self) -> None:
        self._clear_config_layout()
        if self._frozen_slippery_checkbox is not None:
            try:
                self._frozen_slippery_checkbox.deleteLater()
            except RuntimeError:
                pass
            self._frozen_slippery_checkbox = None

        if self._current_game == GameId.FROZEN_LAKE:
            overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE, {})
            value = overrides.get(
                "is_slippery",
                self._config.frozen_lake_config.is_slippery,
            )
            overrides["is_slippery"] = value
            checkbox = QtWidgets.QCheckBox("Enable slippery ice (stochastic)", self._config_group)
            checkbox.setChecked(bool(value))
            checkbox.stateChanged.connect(self._on_slippery_toggled)
            self._frozen_slippery_checkbox = checkbox
            self._config_layout.addRow("Slippery ice", checkbox)
        elif self._current_game == GameId.FROZEN_LAKE_V2:
            # FrozenLake-v2 configuration
            overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE_V2, {})
            from gym_gui.config.game_configs import DEFAULT_FROZEN_LAKE_V2_CONFIG
            defaults = DEFAULT_FROZEN_LAKE_V2_CONFIG
            
            # Slippery checkbox
            is_slippery = bool(overrides.get("is_slippery", defaults.is_slippery))
            overrides["is_slippery"] = is_slippery
            slippery_checkbox = QtWidgets.QCheckBox("Enable slippery ice (stochastic)", self._config_group)
            slippery_checkbox.setChecked(is_slippery)
            slippery_checkbox.stateChanged.connect(
                lambda state: self._on_frozen_v2_config_changed("is_slippery", state == QtCore.Qt.CheckState.Checked.value)
            )
            self._config_layout.addRow("Slippery ice", slippery_checkbox)
            
            # Grid height
            grid_height_raw = overrides.get("grid_height", defaults.grid_height)
            grid_height = int(grid_height_raw) if isinstance(grid_height_raw, (int, float)) else defaults.grid_height
            overrides["grid_height"] = grid_height
            height_spin = QtWidgets.QSpinBox(self._config_group)
            height_spin.setRange(4, 20)
            height_spin.setValue(grid_height)
            height_spin.valueChanged.connect(
                lambda value: self._on_frozen_v2_config_changed("grid_height", int(value))
            )
            height_spin.setToolTip("Number of rows in the grid (4-20)")
            self._config_layout.addRow("Grid Height", height_spin)
            
            # Grid width
            grid_width_raw = overrides.get("grid_width", defaults.grid_width)
            grid_width = int(grid_width_raw) if isinstance(grid_width_raw, (int, float)) else defaults.grid_width
            overrides["grid_width"] = grid_width
            width_spin = QtWidgets.QSpinBox(self._config_group)
            width_spin.setRange(4, 20)
            width_spin.setValue(grid_width)
            width_spin.valueChanged.connect(
                lambda value: self._on_frozen_v2_config_changed("grid_width", int(value))
            )
            width_spin.setToolTip("Number of columns in the grid (4-20)")
            self._config_layout.addRow("Grid Width", width_spin)
            
            # Start position dropdown
            start_combo = QtWidgets.QComboBox(self._config_group)
            start_positions = [(r, c) for r in range(grid_height) for c in range(grid_width)]
            start_pos = overrides.get("start_position", defaults.start_position or (0, 0))
            overrides["start_position"] = start_pos
            
            for pos in start_positions:
                start_combo.addItem(f"({pos[0]}, {pos[1]})", pos)
            
            start_idx = start_combo.findData(start_pos)
            if start_idx >= 0:
                start_combo.setCurrentIndex(start_idx)
            
            start_combo.currentIndexChanged.connect(
                lambda idx: self._on_frozen_v2_config_changed("start_position", start_combo.itemData(idx))
            )
            start_combo.setToolTip("Starting position for the agent")
            self._config_layout.addRow("Start Position", start_combo)
            
            # Goal position dropdown (excludes start position)
            goal_combo = QtWidgets.QComboBox(self._config_group)
            goal_positions = [pos for pos in start_positions if pos != start_pos]
            
            # Get goal position from overrides, fallback to defaults, finally to bottom-right
            goal_pos = overrides.get("goal_position")
            if goal_pos is None:
                goal_pos = defaults.goal_position if defaults.goal_position is not None else (grid_height - 1, grid_width - 1)
            overrides["goal_position"] = goal_pos
            
            for pos in goal_positions:
                goal_combo.addItem(f"({pos[0]}, {pos[1]})", pos)
            
            goal_idx = goal_combo.findData(goal_pos)
            if goal_idx >= 0:
                goal_combo.setCurrentIndex(goal_idx)
            else:
                # If goal position not found (shouldn't happen), select last item (bottom-right)
                goal_combo.setCurrentIndex(goal_combo.count() - 1)
            
            goal_combo.currentIndexChanged.connect(
                lambda idx: self._on_frozen_v2_config_changed("goal_position", goal_combo.itemData(idx))
            )
            goal_combo.setToolTip("Goal position (excludes start position)")
            self._config_layout.addRow("Goal Position", goal_combo)
            
            # Hole count spinner
            hole_count_raw = overrides.get("hole_count", defaults.hole_count or 10)
            hole_count = int(hole_count_raw) if isinstance(hole_count_raw, (int, float)) else 10
            overrides["hole_count"] = hole_count
            hole_spin = QtWidgets.QSpinBox(self._config_group)
            max_holes = (grid_height * grid_width) - 2  # Exclude start and goal
            hole_spin.setRange(0, max_holes)
            hole_spin.setValue(hole_count)
            hole_spin.valueChanged.connect(
                lambda value: self._on_frozen_v2_config_changed("hole_count", int(value))
            )
            hole_spin.setToolTip(f"Number of holes in the grid (0-{max_holes})")
            self._config_layout.addRow("Hole Count", hole_spin)
            
            # Random holes checkbox
            random_holes = bool(overrides.get("random_holes", defaults.random_holes))
            overrides["random_holes"] = random_holes
            random_holes_checkbox = QtWidgets.QCheckBox("Random hole placement", self._config_group)
            random_holes_checkbox.setChecked(random_holes)
            random_holes_checkbox.stateChanged.connect(
                lambda state: self._on_frozen_v2_config_changed("random_holes", state == QtCore.Qt.CheckState.Checked.value)
            )
            random_holes_checkbox.setToolTip(
                "If checked, holes are placed randomly. "
                "If unchecked (default), uses fixed Gymnasium map patterns for 4×4 and 8×8 grids."
            )
            self._config_layout.addRow("Random Holes", random_holes_checkbox)
        elif self._current_game == GameId.LUNAR_LANDER:
            overrides = self._game_overrides.setdefault(GameId.LUNAR_LANDER, {})
            defaults = self._config.lunar_lander_config

            continuous = bool(overrides.get("continuous", defaults.continuous))
            overrides["continuous"] = continuous
            continuous_checkbox = QtWidgets.QCheckBox(
                "Continuous control (Box actions)", self._config_group
            )
            continuous_checkbox.setChecked(continuous)
            continuous_checkbox.toggled.connect(
                lambda checked: self._on_lunar_config_changed("continuous", bool(checked))
            )
            continuous_checkbox.setToolTip(
                "Continuous control uses analog thrusters. Human-only mode will rely on passive actions."
            )
            self._config_layout.addRow("Action space", continuous_checkbox)

            gravity_raw = overrides.get("gravity", defaults.gravity)
            gravity_value = float(gravity_raw) if isinstance(gravity_raw, (int, float)) else defaults.gravity
            overrides["gravity"] = gravity_value
            gravity_spin = QtWidgets.QDoubleSpinBox(self._config_group)
            gravity_spin.setRange(-12.0, 0.0)
            gravity_spin.setSingleStep(0.1)
            gravity_spin.setDecimals(2)
            gravity_spin.setValue(gravity_value)
            gravity_spin.valueChanged.connect(
                lambda value: self._on_lunar_config_changed("gravity", float(value))
            )
            gravity_spin.setToolTip("Gravity constant applied to the lander (0 to -12).")
            self._config_layout.addRow("Gravity", gravity_spin)

            enable_wind = bool(overrides.get("enable_wind", defaults.enable_wind))
            overrides["enable_wind"] = enable_wind
            wind_checkbox = QtWidgets.QCheckBox("Enable wind", self._config_group)
            wind_checkbox.setChecked(enable_wind)
            self._config_layout.addRow("Wind", wind_checkbox)

            wind_power_raw = overrides.get("wind_power", defaults.wind_power)
            wind_power_value = float(wind_power_raw) if isinstance(wind_power_raw, (int, float)) else defaults.wind_power
            overrides["wind_power"] = wind_power_value
            wind_spin = QtWidgets.QDoubleSpinBox(self._config_group)
            wind_spin.setRange(0.0, 20.0)
            wind_spin.setSingleStep(0.5)
            wind_spin.setDecimals(2)
            wind_spin.setValue(wind_power_value)
            wind_spin.setEnabled(enable_wind)
            wind_spin.valueChanged.connect(
                lambda value: self._on_lunar_config_changed("wind_power", float(value))
            )
            wind_spin.setToolTip("Maximum horizontal wind magnitude (0-20).")
            self._config_layout.addRow("Wind power", wind_spin)

            turbulence_raw = overrides.get("turbulence_power", defaults.turbulence_power)
            turbulence_value = (
                float(turbulence_raw)
                if isinstance(turbulence_raw, (int, float))
                else defaults.turbulence_power
            )
            overrides["turbulence_power"] = turbulence_value
            turbulence_spin = QtWidgets.QDoubleSpinBox(self._config_group)
            turbulence_spin.setRange(0.0, 5.0)
            turbulence_spin.setSingleStep(0.1)
            turbulence_spin.setDecimals(2)
            turbulence_spin.setValue(turbulence_value)
            turbulence_spin.setEnabled(enable_wind)
            turbulence_spin.valueChanged.connect(
                lambda value: self._on_lunar_config_changed("turbulence_power", float(value))
            )
            turbulence_spin.setToolTip("Maximum rotational gust strength (0-5).")
            self._config_layout.addRow("Turbulence", turbulence_spin)

            steps_raw = overrides.get("max_episode_steps", defaults.max_episode_steps)
            steps_value = (
                int(steps_raw)
                if isinstance(steps_raw, (int, float)) and int(steps_raw) > 0
                else 0
            )
            overrides["max_episode_steps"] = None if steps_value == 0 else steps_value
            steps_spin = QtWidgets.QSpinBox(self._config_group)
            steps_spin.setRange(0, 20000)
            steps_spin.setSpecialValueText("Use Gym default (1000)")
            steps_spin.setValue(steps_value)
            steps_spin.valueChanged.connect(
                lambda value: self._on_lunar_config_changed(
                    "max_episode_steps",
                    None if value == 0 else int(value),
                )
            )
            steps_spin.setToolTip(
                "Maximum number of steps before truncation (0 keeps Gymnasium's default of 1000)."
            )
            self._config_layout.addRow("Max steps", steps_spin)

            def _update_wind_controls(enabled: bool, *, emit: bool = True) -> None:
                wind_spin.setEnabled(enabled)
                turbulence_spin.setEnabled(enabled)
                if emit:
                    self._on_lunar_config_changed("enable_wind", bool(enabled))

            wind_checkbox.toggled.connect(lambda checked: _update_wind_controls(checked, emit=True))
            _update_wind_controls(enable_wind, emit=False)
        elif self._current_game == GameId.TAXI:
            # Add Taxi-specific configuration options
            overrides = self._game_overrides.setdefault(GameId.TAXI, {})
            
            # Is Raining checkbox
            is_raining = overrides.get("is_raining", self._config.taxi_config.is_raining)
            overrides["is_raining"] = is_raining
            raining_checkbox = QtWidgets.QCheckBox("Enable rain (80% move success)", self._config_group)
            raining_checkbox.setChecked(bool(is_raining))
            raining_checkbox.stateChanged.connect(
                lambda state: self._on_taxi_config_changed("is_raining", state == QtCore.Qt.CheckState.Checked.value)
            )
            self._config_layout.addRow("Rain", raining_checkbox)
            
            # Fickle Passenger checkbox
            fickle = overrides.get("fickle_passenger", self._config.taxi_config.fickle_passenger)
            overrides["fickle_passenger"] = fickle
            fickle_checkbox = QtWidgets.QCheckBox("Fickle passenger (30% dest change)", self._config_group)
            fickle_checkbox.setChecked(bool(fickle))
            fickle_checkbox.stateChanged.connect(
                lambda state: self._on_taxi_config_changed("fickle_passenger", state == QtCore.Qt.CheckState.Checked.value)
            )
            self._config_layout.addRow("Fickle", fickle_checkbox)
        elif self._current_game == GameId.CLIFF_WALKING:
            # Add CliffWalking-specific configuration options
            overrides = self._game_overrides.setdefault(GameId.CLIFF_WALKING, {})
            
            # Is Slippery checkbox
            is_slippery = overrides.get("is_slippery", self._config.cliff_walking_config.is_slippery)
            overrides["is_slippery"] = is_slippery
            slippery_checkbox = QtWidgets.QCheckBox("Enable slippery cliff (stochastic)", self._config_group)
            slippery_checkbox.setChecked(bool(is_slippery))
            slippery_checkbox.stateChanged.connect(
                lambda state: self._on_cliff_config_changed("is_slippery", state == QtCore.Qt.CheckState.Checked.value)
            )
            self._config_layout.addRow("Slippery", slippery_checkbox)
        elif self._current_game == GameId.CAR_RACING:
            overrides = self._game_overrides.setdefault(GameId.CAR_RACING, {})
            defaults = self._config.car_racing_config

            continuous = bool(overrides.get("continuous", defaults.continuous))
            overrides["continuous"] = continuous
            continuous_checkbox = QtWidgets.QCheckBox("Continuous control (Box actions)", self._config_group)
            continuous_checkbox.setChecked(continuous)
            continuous_checkbox.toggled.connect(
                lambda checked: self._on_car_config_changed("continuous", bool(checked))
            )
            continuous_checkbox.setToolTip(
                "Continuous control exposes steering, gas, and brake as float actions."
            )
            self._config_layout.addRow("Action space", continuous_checkbox)

            domain_randomize = bool(overrides.get("domain_randomize", defaults.domain_randomize))
            overrides["domain_randomize"] = domain_randomize
            domain_checkbox = QtWidgets.QCheckBox("Enable domain randomization", self._config_group)
            domain_checkbox.setChecked(domain_randomize)
            domain_checkbox.toggled.connect(
                lambda checked: self._on_car_config_changed("domain_randomize", bool(checked))
            )
            domain_checkbox.setToolTip("Randomize track and background colours on reset.")
            self._config_layout.addRow("Domain randomize", domain_checkbox)

            lap_raw = overrides.get("lap_complete_percent", defaults.lap_complete_percent)
            lap_value = float(lap_raw) if isinstance(lap_raw, (int, float)) else defaults.lap_complete_percent
            overrides["lap_complete_percent"] = lap_value
            lap_spin = QtWidgets.QDoubleSpinBox(self._config_group)
            lap_spin.setRange(0.50, 1.00)
            lap_spin.setSingleStep(0.01)
            lap_spin.setDecimals(2)
            lap_spin.setValue(lap_value)
            lap_spin.valueChanged.connect(
                lambda value: self._on_car_config_changed("lap_complete_percent", float(value))
            )
            lap_spin.setToolTip("Percentage of tiles required to complete a lap (0.50 - 1.00).")
            self._config_layout.addRow("Lap completion", lap_spin)

            steps_raw = overrides.get("max_episode_steps", defaults.max_episode_steps)
            steps_value = int(steps_raw) if isinstance(steps_raw, (int, float)) and int(steps_raw) > 0 else 0
            overrides["max_episode_steps"] = None if steps_value == 0 else steps_value
            steps_spin = QtWidgets.QSpinBox(self._config_group)
            steps_spin.setRange(0, 20000)
            steps_spin.setSpecialValueText("Disabled (unlimited)")
            steps_spin.setValue(steps_value)
            steps_spin.valueChanged.connect(
                lambda value: self._on_car_config_changed("max_episode_steps", None if value == 0 else int(value))
            )
            steps_spin.setToolTip("Maximum number of steps before truncation (0 disables the step limit).")
            self._config_layout.addRow("Max steps", steps_spin)

            seconds_raw = overrides.get("max_episode_seconds", defaults.max_episode_seconds)
            seconds_value = (
                float(seconds_raw)
                if isinstance(seconds_raw, (int, float)) and float(seconds_raw) > 0
                else 0.0
            )
            overrides["max_episode_seconds"] = None if seconds_value == 0 else seconds_value
            seconds_spin = QtWidgets.QDoubleSpinBox(self._config_group)
            seconds_spin.setRange(0.0, 3600.0)
            seconds_spin.setSingleStep(5.0)
            seconds_spin.setDecimals(1)
            seconds_spin.setSpecialValueText("Use Gym default (disabled)")
            seconds_spin.setValue(seconds_value)
            seconds_spin.valueChanged.connect(
                lambda value: self._on_car_config_changed(
                    "max_episode_seconds", None if value == 0 else float(value)
                )
            )
            seconds_spin.setToolTip("Maximum wall-clock seconds before truncation (0 disables the limit).")
            self._config_layout.addRow("Time limit (s)", seconds_spin)
        elif self._current_game == GameId.BIPEDAL_WALKER:
            overrides = self._game_overrides.setdefault(GameId.BIPEDAL_WALKER, {})
            defaults = self._config.bipedal_walker_config

            hardcore = bool(overrides.get("hardcore", defaults.hardcore))
            overrides["hardcore"] = hardcore
            hardcore_checkbox = QtWidgets.QCheckBox("Enable hardcore terrain", self._config_group)
            hardcore_checkbox.setChecked(hardcore)
            hardcore_checkbox.toggled.connect(
                lambda checked: self._on_bipedal_config_changed("hardcore", bool(checked))
            )
            hardcore_checkbox.setToolTip("Adds ladders, stumps, and pits to the terrain.")
            self._config_layout.addRow("Hardcore mode", hardcore_checkbox)

            steps_raw = overrides.get("max_episode_steps", defaults.max_episode_steps)
            steps_value = int(steps_raw) if isinstance(steps_raw, (int, float)) and int(steps_raw) > 0 else 0
            overrides["max_episode_steps"] = None if steps_value == 0 else steps_value
            steps_spin = QtWidgets.QSpinBox(self._config_group)
            steps_spin.setRange(0, 20000)
            default_steps = 2000 if defaults.hardcore else 1600
            steps_spin.setSpecialValueText(f"Use Gym default ({default_steps})")
            steps_spin.setValue(steps_value)
            steps_spin.valueChanged.connect(
                lambda value: self._on_bipedal_config_changed(
                    "max_episode_steps", None if value == 0 else int(value)
                )
            )
            steps_spin.setToolTip("Maximum steps before truncation (0 keeps Gym default).")
            self._config_layout.addRow("Max steps", steps_spin)

            seconds_raw = overrides.get("max_episode_seconds", defaults.max_episode_seconds)
            seconds_value = (
                float(seconds_raw)
                if isinstance(seconds_raw, (int, float)) and float(seconds_raw) > 0
                else 0.0
            )
            overrides["max_episode_seconds"] = None if seconds_value == 0 else seconds_value
            seconds_spin = QtWidgets.QDoubleSpinBox(self._config_group)
            seconds_spin.setRange(0.0, 3600.0)
            seconds_spin.setSingleStep(5.0)
            seconds_spin.setDecimals(1)
            seconds_spin.setSpecialValueText("Use Gym default (disabled)")
            seconds_spin.setValue(seconds_value)
            seconds_spin.valueChanged.connect(
                lambda value: self._on_bipedal_config_changed(
                    "max_episode_seconds", None if value == 0 else float(value)
                )
            )
            seconds_spin.setToolTip("Maximum wall-clock seconds before truncation (0 disables limit).")
            self._config_layout.addRow("Time limit (s)", seconds_spin)
        else:
            placeholder = QtWidgets.QLabel("No overrides available for this game.")
            placeholder.setWordWrap(True)
            self._config_layout.addRow(placeholder)
