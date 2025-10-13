from __future__ import annotations

"""Main Qt window for the Gym GUI application."""

import logging
from datetime import datetime
from typing import Any, Dict, List

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.config.game_configs import (
    CliffWalkingConfig,
    CarRacingConfig,
    BipedalWalkerConfig,
    FrozenLakeConfig,
    LunarLanderConfig,
    TaxiConfig,
)
from gym_gui.config.settings import Settings
from gym_gui.core.enums import ControlMode, GameId
from gym_gui.core.factories.adapters import available_games
from gym_gui.controllers.human_input import HumanInputController
from gym_gui.controllers.session import SessionController
from gym_gui.ui.logging_bridge import LogRecordPayload, QtLogHandler
from gym_gui.ui.presenters.main_window_presenter import MainWindowPresenter, MainWindowView
from gym_gui.ui.widgets.control_panel import ControlPanelConfig, ControlPanelWidget
from gym_gui.ui.widgets.busy_indicator import modal_busy_indicator
from gym_gui.ui.widgets.render_tabs import RenderTabs
from gym_gui.docs.game_info import get_game_info
from gym_gui.services.actor import ActorService
from gym_gui.services.service_locator import get_service_locator
from gym_gui.services.telemetry import TelemetryService


class MainWindow(QtWidgets.QMainWindow):
    """Primary window that orchestrates the Gym session."""

    LOG_FILTER_OPTIONS: Dict[str, str | None] = {
        "All": None,
        "Controllers": "gym_gui.controllers",
        "Adapters": "gym_gui.core.adapters",
        "Agents": "gym_gui.agents",
    }

    CONTROL_MODE_LABELS: Dict[ControlMode, str] = {
        ControlMode.HUMAN_ONLY: "Human Only",
        ControlMode.AGENT_ONLY: "Agent Only",
        ControlMode.HYBRID_TURN_BASED: "Hybrid (Turn-Based)",
        ControlMode.HYBRID_HUMAN_AGENT: "Hybrid (Human + Agent)",
        ControlMode.MULTI_AGENT_COOP: "Multi-Agent (Cooperation)",
        ControlMode.MULTI_AGENT_COMPETITIVE: "Multi-Agent (Competition)",
    }

    _HUMAN_INPUT_MODES = {
        ControlMode.HUMAN_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    }

    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._settings = settings
        self._session = SessionController(settings, self)
        self._log_handler = QtLogHandler(parent=self)
        self._log_records: List[LogRecordPayload] = []
        self._auto_running = False
        self._episode_finished = False  # Track episode termination state
        self._game_started = False
        self._game_paused = False
        self._awaiting_human = False
        self._human_input = HumanInputController(self, self._session)
        locator = get_service_locator()
        telemetry_service = locator.resolve(TelemetryService)
        actor_service = locator.resolve(ActorService)
        if telemetry_service is None or actor_service is None:
            raise RuntimeError("Required services are not registered in the locator")
        self._telemetry_service: TelemetryService = telemetry_service
        self._actor_service: ActorService = actor_service

        # Build control panel config
        available_modes = {}
        for game in available_games():
            available_modes[game] = SessionController.supported_control_modes(game)

        actor_descriptors = self._actor_service.describe_actors()
        default_actor_id = self._actor_service.get_active_actor_id()
        
        control_config = ControlPanelConfig(
            available_modes=available_modes,
            default_mode=settings.default_control_mode,
            frozen_lake_config=FrozenLakeConfig(is_slippery=False),
            taxi_config=TaxiConfig(is_raining=False, fickle_passenger=False),
            cliff_walking_config=CliffWalkingConfig(is_slippery=False),
            lunar_lander_config=LunarLanderConfig(),
            car_racing_config=CarRacingConfig.from_env(),
            bipedal_walker_config=BipedalWalkerConfig.from_env(),
            default_seed=settings.default_seed,
            allow_seed_reuse=settings.allow_seed_reuse,
            actors=actor_descriptors,
            default_actor_id=default_actor_id,
        )
        
        self._control_panel = ControlPanelWidget(config=control_config, parent=self)
        if default_actor_id is not None:
            self._control_panel.set_active_actor(default_actor_id)
        
        # Create presenter to coordinate
        self._presenter = MainWindowPresenter(self._session, self._human_input, parent=self)
        
        status_bar = self.statusBar()
        if status_bar is None:
            status_bar = QtWidgets.QStatusBar(self)
            self.setStatusBar(status_bar)
        self._status_bar: QtWidgets.QStatusBar = status_bar

        self._configure_logging()
        self._build_ui()
        self._connect_signals()
        self._populate_environments()
        self._status_bar.showMessage("Select an environment to begin")
        self._time_refresh_timer = QtCore.QTimer(self)
        self._time_refresh_timer.setInterval(1000)
        self._time_refresh_timer.timeout.connect(self._refresh_time_labels)
        self._time_refresh_timer.start()
        self._refresh_time_labels()
        QtCore.QTimer.singleShot(0, self._render_tabs.refresh_replays)
        
        # Bind presenter to view
        self._wire_presenter()

    def _wire_presenter(self) -> None:
        """Wire the MainWindowPresenter to coordinate SessionController signals."""
        view = MainWindowView(
            control_panel=self._control_panel,
            status_message_sink=lambda msg, timeout: self._status_bar.showMessage(msg, timeout or 0),
            awaiting_label_setter=lambda waiting: self._on_awaiting_human(waiting, ""),
            turn_label_setter=lambda turn: self._control_panel.set_turn(turn),
            render_adapter=lambda payload: self._render_tabs.display_payload(payload),
            time_refresher=self._refresh_time_labels,
            game_info_setter=self._set_game_info,
        )
        self._presenter.bind_view(view)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _configure_logging(self) -> None:
        root_logger = logging.getLogger()
        self._log_handler.setLevel(logging.NOTSET)
        formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        self._log_handler.setFormatter(formatter)
        root_logger.addHandler(self._log_handler)

    def _build_ui(self) -> None:
        self.setWindowTitle("Gym GUI – Qt Shell")
        self.resize(1200, 800)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
        layout.addWidget(splitter)

        # Use the ControlPanelWidget created in __init__
        splitter.addWidget(self._control_panel)

        right_panel = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, central)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)

        self._render_group = QtWidgets.QGroupBox("Render View", right_panel)
        render_layout = QtWidgets.QVBoxLayout(self._render_group)
        self._render_tabs = RenderTabs(
            self._render_group,
            telemetry_service=self._telemetry_service,
        )
        render_layout.addWidget(self._render_tabs)
        right_panel.addWidget(self._render_group)

        # Game information panel (right-most column)
        self._info_group = QtWidgets.QGroupBox("Game Info", self)
        info_layout = QtWidgets.QVBoxLayout(self._info_group)
        self._game_info = QtWidgets.QTextBrowser(self._info_group)
        self._game_info.setReadOnly(True)
        self._game_info.setOpenExternalLinks(True)
        info_layout.addWidget(self._game_info, 1)

        self._log_group = QtWidgets.QGroupBox("Runtime Log", self._info_group)
        log_layout = QtWidgets.QVBoxLayout(self._log_group)
        filter_row = QtWidgets.QHBoxLayout()
        filter_label = QtWidgets.QLabel("Filter:")
        self._log_filter = QtWidgets.QComboBox()
        self._log_filter.addItems(self.LOG_FILTER_OPTIONS.keys())
        filter_row.addWidget(filter_label)
        filter_row.addWidget(self._log_filter, 1)
        log_layout.addLayout(filter_row)

        self._log_console = QtWidgets.QPlainTextEdit()
        self._log_console.setReadOnly(True)
        self._log_console.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self._log_console, 1)
        info_layout.addWidget(self._log_group, 1)
        splitter.addWidget(self._info_group)

        # Configure splitter stretch: control panel (left) small, right_panel (middle) large, info (right) small
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)

    def _connect_signals(self) -> None:
        # Connect control panel signals to session controller
        self._control_panel.load_requested.connect(self._on_load_requested)
        self._control_panel.reset_requested.connect(self._on_reset_requested)
        self._control_panel.start_game_requested.connect(self._on_start_game)
        self._control_panel.pause_game_requested.connect(self._on_pause_game)
        self._control_panel.continue_game_requested.connect(self._on_continue_game)
        self._control_panel.terminate_game_requested.connect(self._on_terminate_game)
        self._control_panel.agent_step_requested.connect(self._session.perform_agent_step)
        self._control_panel.game_changed.connect(self._on_game_changed)
        self._control_panel.control_mode_changed.connect(self._on_mode_changed)
        self._control_panel.actor_changed.connect(self._on_actor_changed)
        self._control_panel.slippery_toggled.connect(self._on_slippery_toggled)
        self._control_panel.taxi_config_changed.connect(self._on_taxi_config_changed)
        self._control_panel.cliff_config_changed.connect(self._on_cliff_config_changed)
        self._control_panel.lunar_config_changed.connect(self._on_lunar_config_changed)
        self._control_panel.car_config_changed.connect(self._on_car_config_changed)
        self._control_panel.bipedal_config_changed.connect(self._on_bipedal_config_changed)
        self._session.seed_applied.connect(self._on_seed_applied)
        
        # Connect log filter
        self._log_filter.currentTextChanged.connect(self._on_log_filter_changed)

        self._session.session_initialized.connect(self._on_session_initialized)
        self._session.step_processed.connect(self._on_step_processed)
        self._session.episode_finished.connect(self._on_episode_finished)
        self._session.status_message.connect(self._on_status_message)
        # Note: awaiting_human is handled by MainWindowPresenter, not directly here
        self._session.turn_changed.connect(self._on_turn_changed)
        self._session.error_occurred.connect(self._on_error)
        self._session.auto_play_state_changed.connect(self._on_auto_play_state)

        self._log_handler.emitter.record_emitted.connect(self._append_log_record)

    def _populate_environments(self) -> None:
        """Populate control panel with available games."""
        games = sorted(available_games(), key=lambda g: g.value)
        default = GameId(self._settings.gym_default_env) if games else None
        self._control_panel.populate_games(games, default=default)

    # ------------------------------------------------------------------
    # Slots - Control Panel Signal Handlers
    # ------------------------------------------------------------------
    def _on_game_changed(self, game_id: GameId) -> None:
        """Handle game selection from control panel."""
        self._status_bar.showMessage(f"Selected {game_id.value}. Load to begin.")
        # Update game info panel with a short description from centralized docs
        desc = get_game_info(game_id)
        if desc:
            self._set_game_info(desc)
        # HumanInputController will be configured when environment loads

    def _on_mode_changed(self, mode: ControlMode) -> None:
        """Handle control mode change from control panel."""
        label = self.CONTROL_MODE_LABELS.get(mode, mode.value)
        self._status_bar.showMessage(f"Mode set to {label}")
        self._human_input.update_for_mode(mode)

    def _on_actor_changed(self, actor_id: str) -> None:
        """Handle active actor selection from the control panel."""
        try:
            self._actor_service.set_active_actor(actor_id)
        except KeyError:
            self.logger.error("Attempted to activate unknown actor '%s'", actor_id)
            self._status_bar.showMessage(f"Unknown actor '{actor_id}'", 5000)
            return

        descriptor = self._actor_service.get_actor_descriptor(actor_id)
        label = descriptor.display_name if descriptor is not None else actor_id
        self._status_bar.showMessage(f"Active actor set to {label}", 4000)

    def _on_load_requested(self, game_id: GameId, mode: ControlMode, seed: int) -> None:
        """Handle load request from control panel."""
        self._episode_finished = False  # Reset episode state on new load
        overrides = self._control_panel.get_overrides(game_id)
        game_config = self._build_game_config(game_id, overrides)
        self._session.load_environment(
            game_id,
            mode,
            seed=seed,
            game_config=game_config,
        )

    def _on_reset_requested(self, seed: int) -> None:
        """Handle reset request from control panel."""
        self._episode_finished = False  # Reset episode state on reset

        self._session.reset_environment(seed=seed)

    def _on_start_game(self) -> None:
        """Handle Start Game button."""
        status = "Game started"
        if self._episode_finished:
            seed = self._control_panel.current_seed()
            self._session.reset_environment(seed=seed)
            self._episode_finished = False
            status = f"Loaded new episode with seed {seed}. Game started"

        self._session.start_game()
        self._game_started = True
        self._game_paused = False
        self._control_panel.set_game_started(True)
        self._control_panel.set_game_paused(False)
        self._update_input_state()
        self._status_bar.showMessage(status, 3000)

    def _on_pause_game(self) -> None:
        """Handle Pause Game button."""
        self._session.pause_game()
        self._game_paused = True
        self._control_panel.set_game_paused(True)
        self._update_input_state()
        self._status_bar.showMessage("Game paused", 3000)

    def _on_continue_game(self) -> None:
        """Handle Continue Game button."""
        self._session.resume_game()
        self._game_paused = False
        self._control_panel.set_game_paused(False)
        self._update_input_state()
        self._status_bar.showMessage("Game continued", 3000)

    def _on_terminate_game(self) -> None:
        """Handle Terminate Game button."""
        with modal_busy_indicator(
            self,
            title="Terminating episode",
            message="Finalizing telemetry and stopping the environment…",
        ):
            self._session.terminate_game()
        self._game_started = False
        self._game_paused = False
        self._control_panel.set_game_started(False)
        self._control_panel.set_game_paused(False)
        self._update_input_state()
        self._episode_finished = True
        self._status_bar.showMessage("Game terminated", 3000)

    def _on_slippery_toggled(self, enabled: bool) -> None:
        """Handle slippery ice toggle from control panel."""
        status = "enabled" if enabled else "disabled"
        current_game = self._control_panel.current_game()
        
        if current_game == GameId.FROZEN_LAKE and self._session.game_id == GameId.FROZEN_LAKE:
            # Reload environment with new setting
            self._status_bar.showMessage(
                f"Frozen Lake slippery ice {status}. Reloading environment...",
                5000,
            )
            mode = self._control_panel.current_mode()
            seed = self._control_panel.current_seed()
            overrides = self._control_panel.get_overrides(GameId.FROZEN_LAKE)
            game_config = self._build_game_config(GameId.FROZEN_LAKE, overrides)
            self._session.load_environment(
                GameId.FROZEN_LAKE,
                mode,
                seed=seed,
                game_config=game_config,
            )
        else:
            # Just update the setting for next load
            self._status_bar.showMessage(
                f"Frozen Lake slippery ice {status}. Reload to apply.",
                5000,
            )

    def _on_taxi_config_changed(self, param_name: str, value: bool) -> None:
        """Handle Taxi configuration changes from control panel."""
        status = "enabled" if value else "disabled"
        param_label = "rain" if param_name == "is_raining" else "fickle passenger"
        current_game = self._control_panel.current_game()
        
        if current_game == GameId.TAXI and self._session.game_id == GameId.TAXI:
            # Reload environment with new setting
            self._status_bar.showMessage(
                f"Taxi {param_label} {status}. Reloading environment...",
                5000,
            )
            mode = self._control_panel.current_mode()
            seed = self._control_panel.current_seed()
            overrides = self._control_panel.get_overrides(GameId.TAXI)
            game_config = self._build_game_config(GameId.TAXI, overrides)
            self._session.load_environment(
                GameId.TAXI,
                mode,
                seed=seed,
                game_config=game_config,
            )
        else:
            # Just update the setting for next load
            self._status_bar.showMessage(
                f"Taxi {param_label} {status}. Reload to apply.",
                5000,
            )

    def _on_cliff_config_changed(self, param_name: str, value: bool) -> None:
        """Handle CliffWalking configuration changes from control panel."""
        status = "enabled" if value else "disabled"
        param_label = "slippery cliff"
        current_game = self._control_panel.current_game()
        
        if current_game == GameId.CLIFF_WALKING and self._session.game_id == GameId.CLIFF_WALKING:
            # Reload environment with new setting
            self._status_bar.showMessage(
                f"Cliff Walking {param_label} {status}. Reloading environment...",
                5000,
            )
            mode = self._control_panel.current_mode()
            seed = self._control_panel.current_seed()
            overrides = self._control_panel.get_overrides(GameId.CLIFF_WALKING)
            game_config = self._build_game_config(GameId.CLIFF_WALKING, overrides)
            self._session.load_environment(
                GameId.CLIFF_WALKING,
                mode,
                seed=seed,
                game_config=game_config,
            )
        else:
            # Just update the setting for next load
            self._status_bar.showMessage(
                f"Cliff Walking {param_label} {status}. Reload to apply.",
                5000,
            )

    def _on_lunar_config_changed(self, param_name: str, value: object) -> None:
        """Handle LunarLander configuration changes from control panel."""
        label_map = {
            "continuous": "continuous control",
            "gravity": "gravity",
            "enable_wind": "wind",
            "wind_power": "wind power",
            "turbulence_power": "turbulence",
        }
        current_game = self._control_panel.current_game()
        descriptor = label_map.get(param_name, param_name)
        if value is None:
            value_str = "default"
        elif isinstance(value, bool):
            value_str = "enabled" if value else "disabled"
        elif isinstance(value, (int, float)):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        reloading = current_game == GameId.LUNAR_LANDER and self._session.game_id == GameId.LUNAR_LANDER
        message = f"Lunar Lander {descriptor} updated to {value_str}."
        if reloading:
            message += " Reloading to apply..."
        self._status_bar.showMessage(message, 5000 if reloading else 4000)

        if reloading:
            mode = self._control_panel.current_mode()
            seed = self._control_panel.current_seed()
            overrides = self._control_panel.get_overrides(GameId.LUNAR_LANDER)
            game_config = self._build_game_config(GameId.LUNAR_LANDER, overrides)
            self._session.load_environment(
                GameId.LUNAR_LANDER,
                mode,
                seed=seed,
                game_config=game_config,
            )

    def _on_car_config_changed(self, param_name: str, value: object) -> None:
        """Handle CarRacing configuration changes from control panel."""
        label_map = {
            "continuous": "continuous control",
            "domain_randomize": "domain randomization",
            "lap_complete_percent": "lap completion requirement",
            "max_episode_steps": "episode step limit",
            "max_episode_seconds": "episode time limit",
        }
        current_game = self._control_panel.current_game()
        descriptor = label_map.get(param_name, param_name)
        if isinstance(value, bool):
            value_str = "enabled" if value else "disabled"
        elif isinstance(value, (int, float)):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        reloading = current_game == GameId.CAR_RACING and self._session.game_id == GameId.CAR_RACING
        message = f"Car Racing {descriptor} updated to {value_str}."
        if reloading:
            message += " Reloading to apply..."
        self._status_bar.showMessage(message, 5000 if reloading else 4000)

        if reloading:
            mode = self._control_panel.current_mode()
            seed = self._control_panel.current_seed()
            overrides = self._control_panel.get_overrides(GameId.CAR_RACING)
            game_config = self._build_game_config(GameId.CAR_RACING, overrides)
            self._session.load_environment(
                GameId.CAR_RACING,
                mode,
                seed=seed,
                game_config=game_config,
            )

    def _on_bipedal_config_changed(self, param_name: str, value: object) -> None:
        """Handle BipedalWalker configuration changes from control panel."""
        label_map = {
            "hardcore": "hardcore terrain",
            "max_episode_steps": "episode step limit",
            "max_episode_seconds": "episode time limit",
        }
        current_game = self._control_panel.current_game()
        descriptor = label_map.get(param_name, param_name)
        if value is None:
            value_str = "default"
        elif isinstance(value, bool):
            value_str = "enabled" if value else "disabled"
        elif isinstance(value, (int, float)):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        reloading = current_game == GameId.BIPEDAL_WALKER and self._session.game_id == GameId.BIPEDAL_WALKER
        message = f"Bipedal Walker {descriptor} {value_str}."
        if reloading:
            message += " Reloading to apply..."
        self._status_bar.showMessage(message, 5000 if reloading else 4000)

        if reloading:
            mode = self._control_panel.current_mode()
            seed = self._control_panel.current_seed()
            overrides = self._control_panel.get_overrides(GameId.BIPEDAL_WALKER)
            game_config = self._build_game_config(GameId.BIPEDAL_WALKER, overrides)
            self._session.load_environment(
                GameId.BIPEDAL_WALKER,
                mode,
                seed=seed,
                game_config=game_config,
            )

    def _build_game_config(
        self,
        game_id: GameId,
        overrides: dict[str, Any],
    ) -> (
        FrozenLakeConfig
        | TaxiConfig
        | CliffWalkingConfig
        | LunarLanderConfig
        | CarRacingConfig
        | BipedalWalkerConfig
        | None
    ):
        """Build game configuration from control panel overrides."""
        if game_id == GameId.FROZEN_LAKE:
            is_slippery = bool(overrides.get("is_slippery", False))
            return FrozenLakeConfig(is_slippery=is_slippery)
        elif game_id == GameId.TAXI:
            is_raining = bool(overrides.get("is_raining", False))
            fickle_passenger = bool(overrides.get("fickle_passenger", False))
            return TaxiConfig(is_raining=is_raining, fickle_passenger=fickle_passenger)
        elif game_id == GameId.CLIFF_WALKING:
            is_slippery = bool(overrides.get("is_slippery", False))
            return CliffWalkingConfig(is_slippery=is_slippery)
        elif game_id == GameId.LUNAR_LANDER:
            continuous = bool(overrides.get("continuous", False))
            gravity = overrides.get("gravity", -10.0)
            enable_wind = bool(overrides.get("enable_wind", False))
            wind_power = overrides.get("wind_power", 15.0)
            turbulence_power = overrides.get("turbulence_power", 1.5)
            try:
                gravity_value = float(gravity)
            except (TypeError, ValueError):
                gravity_value = -10.0
            try:
                wind_power_value = float(wind_power)
            except (TypeError, ValueError):
                wind_power_value = 15.0
            try:
                turbulence_value = float(turbulence_power)
            except (TypeError, ValueError):
                turbulence_value = 1.5
            return LunarLanderConfig(
                continuous=continuous,
                gravity=gravity_value,
                enable_wind=enable_wind,
                wind_power=wind_power_value,
                turbulence_power=turbulence_value,
            )
        elif game_id == GameId.CAR_RACING:
            continuous = bool(overrides.get("continuous", False))
            domain_randomize = bool(overrides.get("domain_randomize", False))
            lap_percent = overrides.get("lap_complete_percent", 0.95)
            try:
                lap_value = float(lap_percent)
            except (TypeError, ValueError):
                lap_value = 0.95
            steps_override = overrides.get("max_episode_steps")
            seconds_override = overrides.get("max_episode_seconds")
            max_steps: int | None = None
            max_seconds: float | None = None
            if isinstance(steps_override, (int, float)) and int(steps_override) > 0:
                max_steps = int(steps_override)
            if isinstance(seconds_override, (int, float)) and float(seconds_override) > 0:
                max_seconds = float(seconds_override)
            return CarRacingConfig(
                continuous=continuous,
                domain_randomize=domain_randomize,
                lap_complete_percent=lap_value,
                max_episode_steps=max_steps,
                max_episode_seconds=max_seconds,
            )
        elif game_id == GameId.BIPEDAL_WALKER:
            hardcore = bool(overrides.get("hardcore", False))
            steps_override = overrides.get("max_episode_steps")
            seconds_override = overrides.get("max_episode_seconds")
            max_steps: int | None = None
            max_seconds: float | None = None
            if isinstance(steps_override, (int, float)) and int(steps_override) > 0:
                max_steps = int(steps_override)
            if isinstance(seconds_override, (int, float)) and float(seconds_override) > 0:
                max_seconds = float(seconds_override)
            return BipedalWalkerConfig(
                hardcore=hardcore,
                max_episode_steps=max_steps,
                max_episode_seconds=max_seconds,
            )
        else:
            return None

    def _on_session_initialized(self, game_id: str, mode: str, step: object) -> None:
        try:
            mode_label = self.CONTROL_MODE_LABELS[ControlMode(mode)]
        except Exception:
            mode_label = mode
        self._status_bar.showMessage(f"Loaded {game_id} in {mode_label} mode - Click 'Start Game' to begin")
        self.logger.info("Loaded %s (%s)", game_id, mode_label)
        self._auto_running = False
        self._game_started = False
        self._game_paused = False
        self._awaiting_human = False
        self._human_input.configure(self._session.game_id, self._session.action_space)
        self._control_panel.set_auto_running(False)
        self._control_panel.set_game_started(False)  # Reset game state on new load
        self._control_panel.set_game_paused(False)
        self._update_input_state()
        self._refresh_time_labels()
        
        # Notify render view of current game for asset selection
        if self._session.game_id is not None:
            self._render_tabs.set_current_game(self._session.game_id)
            # Update game info with a slightly more detailed description (include mode)
            try:
                gid = GameId(self._session.game_id)
            except Exception:
                gid = None
            if gid is not None:
                desc = get_game_info(gid)
                if desc:
                    self._set_game_info(desc + f"<p><b>Mode:</b> {mode}</p>")

    def _set_game_info(self, html: str) -> None:
        """Set the HTML content of the Game Info panel."""
        if not hasattr(self, "_game_info"):
            return
        if not html:
            html = "<p>Select an environment to begin. The Game Info panel will show rules, controls, and rewards for the chosen environment.</p>"
        self._game_info.setHtml(html)

    def _on_step_processed(self, step: object, index: int) -> None:
        """Handle step processed from session controller."""
        if not hasattr(step, "reward"):
            return
        reward = getattr(step, "reward", 0.0)
        terminated = getattr(step, "terminated", False)
        truncated = getattr(step, "truncated", False)
        render_payload = getattr(step, "render_payload", None)

        # Update control panel status (awaiting_human and turn updated via separate signals)
        self._control_panel.set_status(
            step=index,
            reward=reward,
            total_reward=self._session.current_episode_reward,
            terminated=terminated,
            truncated=truncated,
            turn=self._session._turn,
            awaiting_human=False,  # Will be updated via awaiting_human signal
            session_time=self._session._timers.launch_elapsed_formatted(),
            active_time=self._session._timers.first_move_elapsed_formatted(),
            episode_duration=self._session._timers.episode_duration_formatted(),
            outcome_time=self._session._timers.outcome_elapsed_formatted(),
            outcome_wall_clock=self._session._timers.outcome_wall_clock_formatted(),
        )
        
        self._render_tabs.display_payload(render_payload)

    def _on_episode_finished(self, finished: bool) -> None:
        self._episode_finished = finished
        if finished:
            # Disable shortcuts when episode terminates
            self._game_started = False
            self._game_paused = False
            self._control_panel.set_game_started(False)  # Reset game state
            self._control_panel.set_game_paused(False)
            self._update_input_state()
            self._render_tabs.on_episode_finished()
            next_seed = self._session.next_seed
            self._control_panel.set_seed_value(next_seed)
            self._status_bar.showMessage(
                f"Episode finished. Next seed prepared: {next_seed}", 4000
            )

    def _on_seed_applied(self, seed: int) -> None:
        self._control_panel.set_seed_value(seed)
        if self._settings.allow_seed_reuse:
            message = (
                f"Seed {seed} applied. Override by adjusting the seed before the next run."
            )
        else:
            message = "Seed applied. Episode will reuse this value until it finishes."
        self._status_bar.showMessage(message, 4000)

    def _on_status_message(self, message: str) -> None:
        self._status_bar.showMessage(message, 5000)

    def _on_awaiting_human(self, waiting: bool, message: str) -> None:
        """
        Handle awaiting_human signal to update UI and keyboard shortcuts.
        
        Shortcuts are disabled if the episode has finished OR if the game hasn't been started.
        In HUMAN_ONLY mode, shortcuts stay enabled during active started episodes.
        In hybrid modes, shortcuts are only enabled when waiting for human input.
        """
        self._awaiting_human = waiting
        self._control_panel.set_awaiting_human(waiting)
        if message:
            self._status_bar.showMessage(message, 5000)
        self._update_input_state()

    def _on_turn_changed(self, turn: str) -> None:
        self._control_panel.set_turn(turn)

    def _on_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Session Error", message)
        self._status_bar.showMessage(message, 5000)

    def _on_auto_play_state(self, running: bool) -> None:
        self._auto_running = running
        self._control_panel.set_auto_running(running)
        self._update_input_state()

    def _append_log_record(self, payload: LogRecordPayload) -> None:
        self._log_records.append(payload)
        if self._passes_filter(payload):
            self._log_console.appendPlainText(self._format_log(payload))
            scrollbar = self._log_console.verticalScrollBar()
            if scrollbar is not None:
                scrollbar.setValue(scrollbar.maximum())

    def _on_log_filter_changed(self, _: str) -> None:
        self._log_console.clear()
        for record in self._log_records:
            if self._passes_filter(record):
                self._log_console.appendPlainText(self._format_log(record))
        scrollbar = self._log_console.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _passes_filter(self, payload: LogRecordPayload) -> bool:
        selected = self._log_filter.currentText()
        prefix = self.LOG_FILTER_OPTIONS.get(selected)
        if not prefix:
            return True
        return payload.name.startswith(prefix)

    def _update_input_state(self) -> None:
        """Synchronize keyboard input enablement with session state."""
        mode = self._control_panel.current_mode()

        enable_input = False
        if self._episode_finished:
            enable_input = False
        elif not self._game_started:
            enable_input = False
        elif self._game_paused:
            enable_input = False
        elif mode == ControlMode.HUMAN_ONLY:
            enable_input = True
        elif mode in self._HUMAN_INPUT_MODES:
            enable_input = self._awaiting_human

        self._human_input.set_enabled(enable_input)

    @staticmethod
    def _format_log(payload: LogRecordPayload) -> str:
        ts = datetime.fromtimestamp(payload.created).strftime("%H:%M:%S")
        return f"{ts} | {payload.level:<7} | {payload.name} | {payload.message}"

    @staticmethod
    def _format_bool(value: bool) -> str:
        return "Yes" if value else "No"

    def _refresh_time_labels(self) -> None:
        """Update time labels in control panel."""
        timers = self._session._timers
        self._control_panel.set_time_labels(
            session_time=timers.launch_elapsed_formatted(),
            active_time=timers.first_move_elapsed_formatted(),
            outcome_time=timers.outcome_elapsed_formatted(),
            outcome_timestamp=timers.outcome_wall_clock_formatted(),
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI only
        logging.getLogger().removeHandler(self._log_handler)
        self._session.shutdown()
        if hasattr(self, "_time_refresh_timer") and self._time_refresh_timer.isActive():
            self._time_refresh_timer.stop()
        super().closeEvent(event)
__all__ = ["MainWindow"]
