from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from qtpy import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal as Signal

from gym_gui.config.game_configs import CliffWalkingConfig, FrozenLakeConfig, TaxiConfig
from gym_gui.core.enums import ControlMode, GameId


@dataclass(frozen=True)
class ControlPanelConfig:
    """Configuration for the control panel with game-specific configs."""
    
    available_modes: Dict[GameId, Iterable[ControlMode]]
    default_mode: ControlMode
    frozen_lake_config: FrozenLakeConfig
    taxi_config: TaxiConfig
    cliff_walking_config: CliffWalkingConfig


class ControlPanelWidget(QtWidgets.QWidget):
    control_mode_changed = Signal(ControlMode)
    game_changed = Signal(GameId)
    load_requested = Signal(GameId, ControlMode, int)
    reset_requested = Signal(int)
    slippery_toggled = Signal(bool)
    taxi_config_changed = Signal(str, bool)  # (param_name, value)
    cliff_config_changed = Signal(str, bool)  # (param_name, value)
    play_requested = Signal()
    pause_requested = Signal()
    agent_step_requested = Signal()

    def __init__(
        self,
        *,
        config: ControlPanelConfig,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._available_modes = config.available_modes
        self._game_overrides: Dict[GameId, Dict[str, object]] = {
            GameId.FROZEN_LAKE: {"is_slippery": config.frozen_lake_config.is_slippery},
            GameId.TAXI: {
                "is_raining": config.taxi_config.is_raining,
                "fickle_passenger": config.taxi_config.fickle_passenger,
            },
            GameId.CLIFF_WALKING: {"is_slippery": config.cliff_walking_config.is_slippery},
        }

        self._current_game: Optional[GameId] = None
        self._current_mode: ControlMode = config.default_mode
        self._awaiting_human: bool = False
        self._auto_running: bool = False

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
        }

        self._build_ui()
        self._connect_signals()

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

    def update_modes(self, game_id: GameId) -> None:
        supported = tuple(self._available_modes.get(game_id, ()))
        if not supported:
            for button in self._mode_buttons.values():
                button.setEnabled(False)
            return

        for mode, button in self._mode_buttons.items():
            button.blockSignals(True)
            button.setEnabled(mode in supported)
            button.blockSignals(False)

        if self._current_mode not in supported:
            self._current_mode = supported[0]
        self._mode_buttons[self._current_mode].setChecked(True)
        self._update_control_states()

    def set_status(
        self,
        *,
        step: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        turn: str,
        awaiting_human: bool,
        session_time: str,
        active_time: str,
        episode_duration: str,
        outcome_time: str = "—",
    ) -> None:
        self._step_label.setText(str(step))
        self._reward_label.setText(f"{reward:.2f}")
        self._terminated_label.setText(self._format_bool(terminated))
        self._truncated_label.setText(self._format_bool(truncated))
        self._turn_label.setText(turn)
        self.set_awaiting_human(awaiting_human)
        self.set_time_labels(session_time, active_time, outcome_time)

    def set_turn(self, turn: str) -> None:
        self._turn_label.setText(turn)

    def set_mode(self, mode: ControlMode) -> None:
        if mode not in self._mode_buttons:
            return
        self._current_mode = mode
        self._mode_buttons[mode].blockSignals(True)
        self._mode_buttons[mode].setChecked(True)
        self._mode_buttons[mode].blockSignals(False)
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

    def set_slippery_visible(self, visible: bool) -> None:
        if self._frozen_slippery_checkbox is not None:
            self._frozen_slippery_checkbox.setVisible(visible)

    def set_awaiting_human(self, awaiting: bool) -> None:
        self._awaiting_human = awaiting
        self._awaiting_label.setText("Yes" if awaiting else "No")
        self._update_control_states()

    def set_time_labels(self, session_time: str, active_time: str, outcome_time: str) -> None:
        self._session_time_label.setText(session_time)
        self._active_time_label.setText(active_time)
        self._outcome_time_label.setText(outcome_time)

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
        self._seed_spin.setRange(0, 10_000_000)
        self._seed_spin.setValue(0)
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

        # Mode selector
        mode_group = QtWidgets.QGroupBox("Control Mode", self)
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self._mode_buttons: Dict[ControlMode, QtWidgets.QRadioButton] = {}
        mode_button_group = QtWidgets.QButtonGroup(mode_group)
        for mode in ControlMode:
            button = QtWidgets.QRadioButton(
                mode.value.replace("_", " ").title(), mode_group
            )
            button.setEnabled(False)
            mode_button_group.addButton(button)
            mode_layout.addWidget(button)
            self._mode_buttons[mode] = button
        layout.addWidget(mode_group)

        # Control buttons
        controls_group = QtWidgets.QGroupBox("Controls", self)
        controls_layout = QtWidgets.QHBoxLayout(controls_group)
        self._play_button = QtWidgets.QPushButton("Play", controls_group)
        self._pause_button = QtWidgets.QPushButton("Pause", controls_group)
        self._step_button = QtWidgets.QPushButton("Agent Step", controls_group)
        self._reset_button = QtWidgets.QPushButton("Reset", controls_group)
        controls_layout.addWidget(self._play_button)
        controls_layout.addWidget(self._pause_button)
        controls_layout.addWidget(self._step_button)
        controls_layout.addWidget(self._reset_button)
        layout.addWidget(controls_group)

        # Status group
        status_group = QtWidgets.QGroupBox("Status", self)
        status_layout = QtWidgets.QFormLayout(status_group)
        self._step_label = QtWidgets.QLabel("0", status_group)
        self._reward_label = QtWidgets.QLabel("0.0", status_group)
        self._terminated_label = QtWidgets.QLabel("No", status_group)
        self._truncated_label = QtWidgets.QLabel("No", status_group)
        self._turn_label = QtWidgets.QLabel("human", status_group)
        self._awaiting_label = QtWidgets.QLabel("–", status_group)
        self._session_time_label = QtWidgets.QLabel("00:00:00", status_group)
        self._active_time_label = QtWidgets.QLabel("—", status_group)
        self._outcome_time_label = QtWidgets.QLabel("—", status_group)
        status_layout.addRow("Step", self._step_label)
        status_layout.addRow("Reward", self._reward_label)
        status_layout.addRow("Episode Finished", self._terminated_label)
        status_layout.addRow("Episode Aborted", self._truncated_label)
        status_layout.addRow("Turn", self._turn_label)
        status_layout.addRow("Awaiting Input", self._awaiting_label)
        status_layout.addRow("Session Uptime", self._session_time_label)
        status_layout.addRow("Active Play Time", self._active_time_label)
        status_layout.addRow("Outcome Time", self._outcome_time_label)
        layout.addWidget(status_group)

        layout.addStretch(1)

    def _connect_signals(self) -> None:
        self._game_combo.currentIndexChanged.connect(self._on_game_changed)
        self._seed_spin.valueChanged.connect(lambda _: self._update_control_states())
        self._wire_mode_buttons()

        self._load_button.clicked.connect(self._on_load_clicked)
        self._play_button.clicked.connect(self._on_play_clicked)
        self._pause_button.clicked.connect(self._on_pause_clicked)
        self._step_button.clicked.connect(self._on_step_clicked)
        self._reset_button.clicked.connect(self._on_reset_clicked)

    # ------------------------------------------------------------------
    # Signal emitters
    # ------------------------------------------------------------------
    def _emit_mode_changed(self, mode: ControlMode) -> None:
        if self._current_mode != mode:
            self._current_mode = mode
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

    def _on_mode_toggled(self, checked: bool, mode: ControlMode) -> None:
        if checked:
            self._emit_mode_changed(mode)

    def _on_load_clicked(self) -> None:
        if self._current_game is None:
            return
        self.load_requested.emit(self._current_game, self._current_mode, self.current_seed())

    def _on_play_clicked(self) -> None:
        self.play_requested.emit()

    def _on_pause_clicked(self) -> None:
        self.pause_requested.emit()

    def _on_step_clicked(self) -> None:
        self.agent_step_requested.emit()

    def _on_reset_clicked(self) -> None:
        self.reset_requested.emit(self.current_seed())

    def _on_slippery_toggled(self, state: int) -> None:
        enabled = state == QtCore.Qt.CheckState.Checked
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

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------
    def _wire_mode_buttons(self) -> None:
        for mode, button in self._mode_buttons.items():
            button.toggled.connect(lambda checked, m=mode: self._on_mode_toggled(checked, m))

    def _update_control_states(self) -> None:
        is_human = self._current_mode == ControlMode.HUMAN_ONLY
        self._play_button.setEnabled(not self._auto_running and not is_human)
        self._pause_button.setEnabled(self._auto_running)
        self._step_button.setEnabled(not self._auto_running and not is_human)

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
        else:
            placeholder = QtWidgets.QLabel("No overrides available for this game.")
            placeholder.setWordWrap(True)
            self._config_layout.addRow(placeholder)