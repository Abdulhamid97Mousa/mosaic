from __future__ import annotations

"""Main Qt window for the Gym GUI application."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, cast
import json
import threading
import grpc

from PyQt6.QtCore import pyqtSlot, pyqtSignal  # type: ignore[attr-defined]
from qtpy import QtCore, QtGui, QtWidgets  # type: ignore[attr-defined]
try:
    from qtpy.QtGui import QAction
except ImportError:
    from qtpy.QtWidgets import QAction  # type: ignore[attr-defined]

from gym_gui.config.game_configs import (
    CliffWalkingConfig,
    CarRacingConfig,
    BipedalWalkerConfig,
    FrozenLakeConfig,
    LunarLanderConfig,
    TaxiConfig,
    DEFAULT_MINIGRID_EMPTY_5x5_CONFIG,
    DEFAULT_MINIGRID_DOORKEY_5x5_CONFIG,
    DEFAULT_MINIGRID_DOORKEY_6x6_CONFIG,
    DEFAULT_MINIGRID_DOORKEY_8x8_CONFIG,
    DEFAULT_MINIGRID_DOORKEY_16x16_CONFIG,
    DEFAULT_MINIGRID_LAVAGAP_S7_CONFIG,
    DEFAULT_MINIGRID_REDBLUE_DOORS_6x6_CONFIG,  # Added new config
    DEFAULT_MINIGRID_REDBLUE_DOORS_8x8_CONFIG,  # Added new config
)
from gym_gui.constants import UI_DEFAULTS, TRAINER_DEFAULTS
from gym_gui.config.game_config_builder import GameConfigBuilder
from gym_gui.config.paths import VAR_TRAINER_DIR
from gym_gui.config.settings import Settings
from gym_gui.core.enums import ControlMode, GameId
from gym_gui.rendering import RendererRegistry
from gym_gui.core.factories.adapters import available_games
from gym_gui.controllers.human_input import HumanInputController
from gym_gui.controllers.session import SessionController
from gym_gui.controllers.live_telemetry_controllers import LiveTelemetryController
from gym_gui.logging_config.log_constants import (
    LOG_UI_MAINWINDOW_TRACE,
    LOG_UI_MAINWINDOW_INFO,
    LOG_UI_MAINWINDOW_WARNING,
    LOG_UI_MAINWINDOW_ERROR,
    LOG_UI_MAINWINDOW_INVALID_CONFIG,
    LOG_TELEMETRY_SUBSCRIBE_ERROR,
    LOG_LIVE_CONTROLLER_RUN_COMPLETED,
    LOG_UI_WORKER_TABS_WARNING,
    LOG_UI_WORKER_TABS_ERROR,
    LOG_UI_WORKER_TABS_INFO,
    LOG_UI_MAIN_WINDOW_SHUTDOWN_WARNING,
    LOG_UI_MULTI_AGENT_ENV_LOAD_REQUESTED,
)
from gym_gui.constants import DEFAULT_RENDER_DELAY_MS
from gym_gui.logging_config.logger import list_known_components
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.ui.logging_bridge import LogRecordPayload, QtLogHandler
from gym_gui.ui.presenters.main_window_presenter import MainWindowPresenter, MainWindowView
from gym_gui.ui.widgets.control_panel import ControlPanelConfig, ControlPanelWidget
from gym_gui.ui.indicators.busy_indicator import modal_busy_indicator
from gym_gui.ui.widgets.render_tabs import RenderTabs
from gym_gui.game_docs import get_game_info
from gym_gui.game_docs.mosaic_welcome import MOSAIC_WELCOME_HTML
from gym_gui.services.actor import ActorService
from gym_gui.services.service_locator import get_service_locator
from gym_gui.services.telemetry import TelemetryService
from gym_gui.services.trainer import TrainerClient, TrainerClientRunner, RunStatus
from gym_gui.services.trainer.streams import TelemetryAsyncHub
from gym_gui.services.trainer.client_runner import TrainerWatchStopped
from gym_gui.ui.widgets.live_telemetry_tab import LiveTelemetryTab
from gym_gui.ui.widgets.fastlane_tab import FastLaneTab
from gym_gui.ui.presenters.workers import (
    get_worker_presenter_registry,
)
from gym_gui.ui.widgets.spade_bdi_worker_tabs import (
    AgentReplayTab,
)
from gym_gui.ui.panels.analytics_tabs import AnalyticsTabManager
from gym_gui.ui.forms import get_worker_form_factory
from gym_gui.ui.handlers import (
    GameConfigHandler,
    LogHandler,
    MPCHandler,
    GodotHandler,
    ChessHandler,
    ConnectFourHandler,
    GoHandler,
    HumanVsAgentHandler,
    ChessEnvLoader,
    ConnectFourEnvLoader,
    GoEnvLoader,
    TicTacToeEnvLoader,
    VizdoomEnvLoader,
)
from gym_gui.ui.widgets.advanced_config import LaunchConfig, RunMode

from gym_gui.constants.optional_deps import get_mjpc_launcher, get_godot_launcher

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gym_gui.ui.widgets.spade_bdi_train_form import SpadeBdiTrainForm
    from gym_gui.ui.widgets.spade_bdi_policy_selection_form import SpadeBdiPolicySelectionForm
from gym_gui.services.trainer.signals import get_trainer_signals

TRAINER_SUBMIT_DEADLINE_MULTIPLIER = 6


def _training_submit_deadline_seconds() -> float:
    """Return the gRPC deadline used for SubmitRun requests."""

    return TRAINER_DEFAULTS.client.deadline_s * TRAINER_SUBMIT_DEADLINE_MULTIPLIER


_LOGGER = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow, LogConstantMixin):
    """Primary window that orchestrates the Gym session."""

    _auto_subscribe_requested = pyqtSignal(str)

    # Severity-level filters for log viewer
    LOG_SEVERITY_OPTIONS: Dict[str, str | None] = {
        "All": None,
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
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

    _WATCHED_RUN_STATUSES = [
        RunStatus.INIT,
        RunStatus.HANDSHAKE,
        RunStatus.READY,
        RunStatus.EXECUTING,
        RunStatus.PAUSED,
        RunStatus.FAULTED,
        RunStatus.TERMINATED,
    ]

    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self._settings = settings
        self._session = SessionController(settings, self)
        self._log_handler = QtLogHandler(parent=self)
        self._component_filter_options: List[str] = ["All", *list_known_components()]
        self._auto_running = False
        self._episode_finished = False  # Track episode termination state
        self._game_started = False
        self._game_paused = False
        self._awaiting_human = False
        self._latest_fps: float | None = None
        self._human_input = HumanInputController(self, self._session)
        locator = get_service_locator()
        telemetry_service = locator.resolve(TelemetryService)
        actor_service = locator.resolve(ActorService)
        trainer_client = locator.resolve(TrainerClient)
        telemetry_hub = locator.resolve(TelemetryAsyncHub)
        if telemetry_service is None or actor_service is None or trainer_client is None or telemetry_hub is None:
            raise RuntimeError("Required services are not registered in the locator")
        self._telemetry_service: TelemetryService = telemetry_service
        self._actor_service: ActorService = actor_service
        self._trainer_client: TrainerClient = trainer_client
        self._telemetry_hub: TelemetryAsyncHub = telemetry_hub
        
        # Track dynamic agent tabs by (run_id, agent_id)
        self._agent_tab_index: set[tuple[str, str]] = set()
        self._run_watch_stop = threading.Event()
        self._run_watch_thread: Optional[threading.Thread] = None
        self._run_watch_subscription = None
        self._selected_policy_path: Optional[Path] = None
        self._run_metadata: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._fastlane_tabs_open: set[tuple[str, str]] = set()
        self._trainer_daemon_ready: bool = False
        self._trainer_poll_failures: int = 0
        self._trainer_poll_quiet_logged: bool = False

        # MuJoCo MPC launcher
        self._mjpc_launcher = get_mjpc_launcher()

        # Godot game engine launcher
        self._godot_launcher = get_godot_launcher()

        # Environment loaders (initialized in _init_handlers after UI components are created)
        self._chess_env_loader: ChessEnvLoader
        self._connect_four_env_loader: ConnectFourEnvLoader
        self._go_env_loader: GoEnvLoader
        self._tictactoe_env_loader: TicTacToeEnvLoader
        self._vizdoom_env_loader: VizdoomEnvLoader

        # Get live telemetry controller from service locator
        live_controller = locator.resolve(LiveTelemetryController)
        if live_controller is None:
            raise RuntimeError("LiveTelemetryController is not registered in the locator")
        self._live_controller: LiveTelemetryController = live_controller

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
            minigrid_empty_config=DEFAULT_MINIGRID_EMPTY_5x5_CONFIG,
            minigrid_doorkey_5x5_config=DEFAULT_MINIGRID_DOORKEY_5x5_CONFIG,
            minigrid_doorkey_6x6_config=DEFAULT_MINIGRID_DOORKEY_6x6_CONFIG,
            minigrid_doorkey_8x8_config=DEFAULT_MINIGRID_DOORKEY_8x8_CONFIG,
            minigrid_doorkey_16x16_config=DEFAULT_MINIGRID_DOORKEY_16x16_CONFIG,
            minigrid_lavagap_config=DEFAULT_MINIGRID_LAVAGAP_S7_CONFIG,
            minigrid_redbluedoors_6x6_config=DEFAULT_MINIGRID_REDBLUE_DOORS_6x6_CONFIG,  # Added new config
            minigrid_redbluedoors_8x8_config=DEFAULT_MINIGRID_REDBLUE_DOORS_8x8_CONFIG,  # Added new config
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

        # Ensure auto-subscribe dispatches cross-thread via Qt's signal delivery
        self._auto_subscribe_requested.connect(self._auto_subscribe_run_main_thread)
        
        status_bar = self.statusBar()
        if status_bar is None:
            status_bar = QtWidgets.QStatusBar(self)
            self.setStatusBar(status_bar)
        self._status_bar: QtWidgets.QStatusBar = status_bar

        self._configure_logging()
        self._build_ui()
        self._create_view_toolbar()
        self._init_handlers()
        self._connect_signals()
        self._populate_environments()
        self._status_bar.showMessage("Select an environment to begin")
        self._time_refresh_timer = QtCore.QTimer(self)
        self._time_refresh_timer.setInterval(1000)
        self._time_refresh_timer.timeout.connect(self._refresh_time_labels)
        self._time_refresh_timer.start()
        self._refresh_time_labels()
        self._session.set_slow_lane_enabled(not self._control_panel.fastlane_only_enabled())
        self._render_tabs.set_human_replay_enabled(not self._control_panel.fastlane_only_enabled())
        QtCore.QTimer.singleShot(0, self._render_tabs.refresh_replays)
        
        # Poll for new training runs and auto-subscribe
        self._known_runs: set[str] = set()
        self._run_poll_timer = QtCore.QTimer(self)
        self._run_poll_timer.setInterval(2000)  # Poll every 2 seconds
        self._run_poll_timer.timeout.connect(self._poll_for_new_runs)
        self._run_poll_timer.start()

        self._start_run_watch()

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
        # Qt handler emits raw log messages; timestamps and metadata are added when rendering.
        formatter = logging.Formatter("%(message)s")
        self._log_handler.setFormatter(formatter)
        root_logger.addHandler(self._log_handler)

    def _build_ui(self) -> None:
        self.setWindowTitle("Gym GUI – Qt Shell")
        self.resize(800, 600)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, central)
        splitter.setChildrenCollapsible(True)
        layout.addWidget(splitter)

        layout_defaults = UI_DEFAULTS.layout

        # Use the ControlPanelWidget created in __init__ and wrap it in a scroll area
        self._control_panel_scroll = QtWidgets.QScrollArea(splitter)
        self._control_panel_scroll.setWidgetResizable(True)
        self._control_panel_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._control_panel_scroll.setWidget(self._control_panel)
        self._control_panel_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        splitter.addWidget(self._control_panel_scroll)
        self._control_panel_scroll.setMinimumWidth(layout_defaults.control_panel_min_width)
        self._control_panel.setMinimumWidth(layout_defaults.control_panel_min_width)

        right_panel = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, central)
        right_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)
        right_panel.setChildrenCollapsible(True)
        right_panel.setMinimumWidth(layout_defaults.render_min_width)
        if layout_defaults.render_max_width:
            right_panel.setMaximumWidth(layout_defaults.render_max_width)

        self._render_group = QtWidgets.QGroupBox("Render View", right_panel)
        self._render_group.setMinimumWidth(layout_defaults.render_min_width)
        render_layout = QtWidgets.QVBoxLayout(self._render_group)
        self._render_tabs = RenderTabs(
            self._render_group,
            telemetry_service=self._telemetry_service,
        )
        # Note: Chess move signal connected in _connect_signals after handlers init
        self._analytics_tabs = AnalyticsTabManager(self._render_tabs, self)
        render_layout.addWidget(self._render_tabs)
        right_panel.addWidget(self._render_group)

        # Game information panel (right-most column)
        self._info_group = QtWidgets.QGroupBox("Game Info", self)
        info_layout = QtWidgets.QVBoxLayout(self._info_group)
        self._game_info = QtWidgets.QTextBrowser(self._info_group)
        self._game_info.setReadOnly(True)
        self._game_info.setOpenExternalLinks(True)
        # Set MOSAIC welcome message as default
        self._game_info.setHtml(MOSAIC_WELCOME_HTML)
        info_layout.addWidget(self._game_info, 1)

        # Runtime Log panel (far-right column)
        self._log_group = QtWidgets.QGroupBox("Runtime Log", self)
        log_layout = QtWidgets.QVBoxLayout(self._log_group)
        filter_row = QtWidgets.QHBoxLayout()

        # Component filter
        component_label = QtWidgets.QLabel("Component:")
        self._log_filter = QtWidgets.QComboBox()
        self._log_filter.addItems(self._component_filter_options)
        filter_row.addWidget(component_label)
        filter_row.addWidget(self._log_filter, 1)

        # Severity filter
        severity_label = QtWidgets.QLabel("Severity:")
        self._log_severity_filter = QtWidgets.QComboBox()
        self._log_severity_filter.addItems(self.LOG_SEVERITY_OPTIONS.keys())
        filter_row.addWidget(severity_label)
        filter_row.addWidget(self._log_severity_filter, 1)

        log_layout.addLayout(filter_row)

        self._log_console = QtWidgets.QPlainTextEdit()
        self._log_console.setReadOnly(True)
        self._log_console.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self._log_console, 1)
        self._log_group.setMinimumWidth(layout_defaults.log_min_width)

        info_log_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, central)
        info_log_splitter.setChildrenCollapsible(True)
        self._info_group.setMinimumWidth(layout_defaults.info_min_width)
        info_log_splitter.addWidget(self._info_group)
        info_log_splitter.addWidget(self._log_group)
        info_log_splitter.setStretchFactor(0, 2)
        info_log_splitter.setStretchFactor(1, 1)
        info_log_splitter.setMinimumWidth(layout_defaults.info_min_width + layout_defaults.log_min_width)
        info_log_splitter.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        info_log_splitter.setSizes(
            [
                layout_defaults.info_default_width,
                layout_defaults.log_default_width,
            ]
        )
        splitter.addWidget(info_log_splitter)

        splitter.setSizes(
            [
                layout_defaults.control_panel_default_width,
                layout_defaults.render_default_width,
                layout_defaults.info_default_width + layout_defaults.log_default_width,
            ]
        )

        # Configure splitter stretch: control panel (left) small, render view large, info/log column medium
        splitter.setStretchFactor(0, 1)  # Control Panel
        splitter.setStretchFactor(1, 3)  # Render View
        splitter.setStretchFactor(2, 2)  # Game Info + Runtime Log

    def _create_view_toolbar(self) -> None:
        """Create toolbar with quick toggles for key panels."""
        self._view_toolbar = QtWidgets.QToolBar("View", self)
        self._view_toolbar.setObjectName("view-toolbar")
        self._view_toolbar.setMovable(False)
        self._view_toolbar.setFloatable(False)
        self._view_toolbar.setIconSize(QtCore.QSize(16, 16))
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self._view_toolbar)

        self._view_actions: dict[str, QAction] = {}
        self._add_view_toggle("Control Panel", self._control_panel_scroll)
        self._add_view_toggle("Render View", self._render_group)
        self._add_view_toggle("Game Info", self._info_group)
        self._add_view_toggle("Runtime Log", self._log_group)

    def _add_view_toggle(self, label: str, widget: QtWidgets.QWidget) -> None:
        """Insert a checkable action to show/hide a widget."""
        action = QAction(label, self)
        action.setCheckable(True)
        action.setChecked(widget.isVisible())

        def handle_toggle(checked: bool, target: QtWidgets.QWidget = widget, name: str = label) -> None:
            target.setVisible(checked)
            self._status_bar.showMessage(f"{name} {'shown' if checked else 'hidden'}", 2000)

        action.toggled.connect(handle_toggle)
        self._view_toolbar.addAction(action)
        self._view_actions[label] = action

    def _init_handlers(self) -> None:
        """Initialize composed handlers for delegated functionality."""
        # Game configuration handler
        self._game_config_handler = GameConfigHandler(
            control_panel=self._control_panel,
            session=self._session,
            status_bar=self._status_bar,
        )

        # MPC handler
        self._mpc_handler = MPCHandler(
            mjpc_launcher=self._mjpc_launcher,
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )

        # Godot handler
        self._godot_handler = GodotHandler(
            godot_launcher=self._godot_launcher,
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )

        # Log handler
        self._log_handler_composed = LogHandler(
            log_filter=self._log_filter,
            log_severity_filter=self._log_severity_filter,
            log_console=self._log_console,
            severity_options=self.LOG_SEVERITY_OPTIONS,
            initial_components=self._component_filter_options,
        )

        # Board game handlers for Human Control Mode
        # These handle moves from the BoardGameRendererStrategy in the Grid tab
        self._chess_handler = ChessHandler(
            session=self._session,
            render_tabs=self._render_tabs,
            status_bar=self._status_bar,
        )
        self._connect_four_handler = ConnectFourHandler(
            session=self._session,
            render_tabs=self._render_tabs,
            status_bar=self._status_bar,
        )
        self._go_handler = GoHandler(
            session=self._session,
            render_tabs=self._render_tabs,
            status_bar=self._status_bar,
        )

        # Environment loaders (for Human vs Agent mode and environment-specific setup)
        self._chess_env_loader = ChessEnvLoader(
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )
        self._connect_four_env_loader = ConnectFourEnvLoader(
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )
        self._go_env_loader = GoEnvLoader(
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )
        self._tictactoe_env_loader = TicTacToeEnvLoader(
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )
        self._vizdoom_env_loader = VizdoomEnvLoader(
            render_tabs=self._render_tabs,
        )

    def _connect_signals(self) -> None:
        # Connect control panel signals to session controller
        self._control_panel.load_requested.connect(self._on_load_requested)
        self._control_panel.reset_requested.connect(self._on_reset_requested)
        self._control_panel.train_agent_requested.connect(self._on_train_agent_requested)
        self._control_panel.trained_agent_requested.connect(self._on_trained_agent_requested)
        self._control_panel.resume_training_requested.connect(self._on_resume_training_requested)
        self._control_panel.start_game_requested.connect(self._on_start_game)
        self._control_panel.pause_game_requested.connect(self._on_pause_game)
        self._control_panel.continue_game_requested.connect(self._on_continue_game)
        self._control_panel.terminate_game_requested.connect(self._on_terminate_game)
        self._control_panel.agent_step_requested.connect(self._session.perform_agent_step)
        self._control_panel._fastlane_only_checkbox.toggled.connect(self._on_fastlane_only_toggled)
        self._control_panel.game_changed.connect(self._on_game_changed)
        self._control_panel.control_mode_changed.connect(self._on_mode_changed)
        self._control_panel.actor_changed.connect(self._on_actor_changed)
        # Game configuration handlers (delegated)
        self._control_panel.slippery_toggled.connect(self._game_config_handler.on_slippery_toggled)
        self._control_panel.frozen_v2_config_changed.connect(self._game_config_handler.on_frozen_v2_config_changed)
        self._control_panel.taxi_config_changed.connect(self._game_config_handler.on_taxi_config_changed)
        self._control_panel.cliff_config_changed.connect(self._game_config_handler.on_cliff_config_changed)
        self._control_panel.lunar_config_changed.connect(self._game_config_handler.on_lunar_config_changed)
        self._control_panel.car_config_changed.connect(self._game_config_handler.on_car_config_changed)
        self._control_panel.bipedal_config_changed.connect(self._game_config_handler.on_bipedal_config_changed)
        self._control_panel.vizdoom_config_changed.connect(self._game_config_handler.on_vizdoom_config_changed)

        # MPC handlers (delegated)
        self._control_panel.mpc_launch_requested.connect(self._mpc_handler.on_launch_requested)
        self._control_panel.mpc_stop_all_requested.connect(self._mpc_handler.on_stop_all_requested)

        # Godot handlers (delegated)
        self._control_panel.godot_launch_requested.connect(self._godot_handler.on_launch_requested)
        self._control_panel.godot_editor_requested.connect(self._godot_handler.on_editor_requested)
        self._control_panel.godot_stop_all_requested.connect(self._godot_handler.on_stop_all_requested)

        # Multi-Agent Mode handlers
        self._control_panel.multi_agent_load_requested.connect(self._on_multi_agent_load_requested)
        self._control_panel.multi_agent_start_requested.connect(self._on_multi_agent_start_requested)
        self._control_panel.multi_agent_reset_requested.connect(self._on_multi_agent_reset_requested)
        self._control_panel.multi_agent_tab.ai_opponent_changed.connect(self._on_ai_opponent_changed)

        # Advanced Configuration Tab handlers
        self._control_panel.advanced_launch_requested.connect(self._on_advanced_launch)
        self._control_panel.advanced_env_load_requested.connect(self._on_advanced_env_load)

        # Board game handlers (Human Control Mode)
        # These signals come from BoardGameRendererStrategy in the Grid tab
        self._render_tabs.chess_move_made.connect(self._chess_handler.on_chess_move)
        self._render_tabs.connect_four_column_clicked.connect(self._connect_four_handler.on_column_clicked)
        self._render_tabs.go_intersection_clicked.connect(self._go_handler.on_intersection_clicked)
        self._render_tabs.go_pass_requested.connect(self._go_handler.on_pass_requested)

        self._session.seed_applied.connect(self._on_seed_applied)

        # Log filters (delegated)
        self._log_filter.currentTextChanged.connect(self._log_handler_composed.on_filter_changed)
        self._log_severity_filter.currentTextChanged.connect(self._log_handler_composed.on_filter_changed)

        self._session.session_initialized.connect(self._on_session_initialized)
        self._session.step_processed.connect(self._on_step_processed)
        self._session.episode_finished.connect(self._on_episode_finished)
        self._session.status_message.connect(self._on_status_message)
        self._session.fps_updated.connect(self._on_fps_updated)
        # Note: awaiting_human is handled by MainWindowPresenter, not directly here
        self._session.turn_changed.connect(self._on_turn_changed)
        self._session.error_occurred.connect(self._on_error)
        self._session.auto_play_state_changed.connect(self._on_auto_play_state)

        self._log_handler.emitter.record_emitted.connect(self._log_handler_composed.append_log_record)

        # Connect live telemetry controller signals
        # The controller owns tab creation and routing; main window only handles cleanup
        self._live_controller.run_tab_requested.connect(self._on_live_telemetry_tab_requested)
        self._live_controller.run_completed.connect(self._on_run_completed)

        # Connect trainer lifecycle signals
        try:
            trainer_signals = get_trainer_signals()
            trainer_signals.training_finished.connect(self._on_training_finished)
            self.log_constant( 
                LOG_UI_MAINWINDOW_TRACE,
                message="Connected to trainer lifecycle signals",
            )
        except Exception as e:
            self.log_constant( 
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Failed to connect trainer signals: {e}",
                extra={"exception": type(e).__name__},
                exc_info=e,
            )

        # Ensure status reflects persisted mode on startup
        self._on_mode_changed(self._control_panel.current_mode())

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

    def _on_fastlane_only_toggled(self, enabled: bool) -> None:
        self._session.set_slow_lane_enabled(not enabled)
        self._render_tabs.set_human_replay_enabled(not enabled)
        if enabled:
            self._status_bar.showMessage("Fast lane only: telemetry persistence disabled")
        else:
            self._status_bar.showMessage("Slow lane re-enabled")

    def _on_actor_changed(self, actor_id: str) -> None:
        """Handle active actor selection from the control panel."""
        try:
            self._actor_service.set_active_actor(actor_id)
        except KeyError:
            self.log_constant( 
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Attempted to activate unknown actor '{actor_id}'",
                extra={"actor_id": actor_id},
            )
            self._status_bar.showMessage(f"Unknown actor '{actor_id}'", 5000)
            return

        descriptor = self._actor_service.get_actor_descriptor(actor_id)
        label = descriptor.display_name if descriptor is not None else actor_id
        self._status_bar.showMessage(f"Active actor set to {label}", 4000)

    def _on_load_requested(self, game_id: GameId, mode: ControlMode, seed: int) -> None:
        """Handle load request from control panel."""
        self._episode_finished = False  # Reset episode state on new load
        overrides = self._control_panel.get_overrides(game_id)
        game_config = GameConfigBuilder.build_config(game_id, overrides)
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

    def _on_multi_agent_load_requested(self, env_id: str, seed: int) -> None:
        """Handle load request from Multi-Agent tab (Human vs Agent mode).

        This creates a PettingZoo environment for human vs agent play.
        Supports Chess, Connect Four, and Go with interactive board UI.
        """
        _LOGGER.info(
            f"{LOG_UI_MULTI_AGENT_ENV_LOAD_REQUESTED.code} {LOG_UI_MULTI_AGENT_ENV_LOAD_REQUESTED.message} | "
            f"env_id={env_id} seed={seed}"
        )

        if env_id == "chess_v6":
            self._load_chess_game(seed)
        elif env_id == "connect_four_v3":
            self._load_connect_four_game(seed)
        elif env_id == "go_v5":
            self._load_go_game(seed)
        elif env_id == "tictactoe_v3":
            self._load_tictactoe_game(seed)
        else:
            # Other PettingZoo environments not yet implemented
            self._status_bar.showMessage(
                f"Multi-agent environment not yet supported: {env_id}",
                5000
            )

    def _load_chess_game(self, seed: int) -> None:
        """Load and initialize the Chess game with interactive board.

        Delegates to ChessEnvLoader for all chess-specific setup.

        Args:
            seed: Random seed for game initialization
        """
        self._chess_env_loader.load(seed, parent=self)
        # Update game info panel
        desc = get_game_info(GameId.CHESS)
        if desc:
            self._set_game_info(desc)

    def _load_connect_four_game(self, seed: int) -> None:
        """Load and initialize the Connect Four game with interactive board.

        Delegates to ConnectFourEnvLoader for all Connect Four-specific setup.

        Args:
            seed: Random seed for game initialization
        """
        self._connect_four_env_loader.load(seed, parent=self)
        # Update game info panel
        desc = get_game_info(GameId.CONNECT_FOUR)
        if desc:
            self._set_game_info(desc)

    def _load_go_game(self, seed: int) -> None:
        """Load and initialize the Go game with interactive board.

        Delegates to GoEnvLoader for all Go-specific setup.

        Args:
            seed: Random seed for game initialization
        """
        self._go_env_loader.load(seed, parent=self)
        # Update game info panel
        desc = get_game_info(GameId.GO)
        if desc:
            self._set_game_info(desc)

    def _load_tictactoe_game(self, seed: int) -> None:
        """Load and initialize the Tic-Tac-Toe game with interactive board.

        Delegates to TicTacToeEnvLoader for all Tic-Tac-Toe-specific setup.

        Args:
            seed: Random seed for game initialization
        """
        self._tictactoe_env_loader.load(seed, parent=self)
        # Update game info panel
        desc = get_game_info(GameId.TIC_TAC_TOE)
        if desc:
            self._set_game_info(desc)

    def _on_multi_agent_start_requested(self, env_id: str, human_agent: str, seed: int) -> None:
        """Handle start game request from Multi-Agent tab.

        Args:
            env_id: Environment ID (e.g., "chess", "chess_v6")
            human_agent: Which agent the human plays ("player_0" or "player_1")
            seed: Random seed
        """
        if env_id in ("chess", "chess_v6") and self._chess_env_loader.is_loaded:
            self._chess_env_loader.on_start_requested(human_agent, seed)
        elif env_id in ("connect_four", "connect_four_v3") and self._connect_four_env_loader.is_loaded:
            self._connect_four_env_loader.on_start_requested(human_agent, seed)
        elif env_id in ("go", "go_v5") and self._go_env_loader.is_loaded:
            self._go_env_loader.on_start_requested(human_agent, seed)
        elif env_id in ("tictactoe", "tictactoe_v3") and self._tictactoe_env_loader.is_loaded:
            self._tictactoe_env_loader.on_start_requested(human_agent, seed)
        else:
            self._status_bar.showMessage(f"Start game not supported for: {env_id}", 3000)

    def _on_multi_agent_reset_requested(self, seed: int) -> None:
        """Handle reset game request from Multi-Agent tab.

        Resets the currently active game.

        Args:
            seed: New random seed for reset
        """
        # Try to reset whichever game is currently loaded
        if self._chess_env_loader.is_loaded:
            self._chess_env_loader.on_reset_requested(seed)
        elif self._connect_four_env_loader.is_loaded:
            self._connect_four_env_loader.on_reset_requested(seed)
        elif self._go_env_loader.is_loaded:
            self._go_env_loader.on_reset_requested(seed)
        elif self._tictactoe_env_loader.is_loaded:
            self._tictactoe_env_loader.on_reset_requested(seed)
        else:
            self._status_bar.showMessage("No active game to reset", 3000)

    def _on_ai_opponent_changed(self, opponent_type: str, difficulty: str) -> None:
        """Handle AI opponent selection change.

        If a game is currently running, update the AI provider via loader.

        Args:
            opponent_type: Type of AI opponent ("random", "stockfish", "custom")
            difficulty: Difficulty level for engines like Stockfish
        """
        self._chess_env_loader.on_ai_config_changed(opponent_type, difficulty)

    # ------------------------------------------------------------------
    # Advanced Configuration Tab Handlers
    # ------------------------------------------------------------------

    def _on_advanced_env_load(self, env_id: str, seed: int) -> None:
        """Handle environment load request from Advanced Configuration tab.

        This is triggered when the user clicks 'Load Environment' in Step 1.
        Currently logs the request; full environment loading will be implemented
        as part of Phase 2.2 (SessionController integration).

        Args:
            env_id: Environment identifier (e.g., "CartPole-v1", "chess_v6")
            seed: Random seed for environment initialization
        """
        _LOGGER.info(
            "Advanced tab: Environment load requested | env_id=%s seed=%d",
            env_id,
            seed,
        )
        self._status_bar.showMessage(
            f"Environment preview: {env_id} (seed={seed})", 3000
        )

    def _on_advanced_launch(self, config: LaunchConfig) -> None:
        """Handle launch request from Advanced Configuration tab.

        Dispatches to the appropriate handler based on run mode:
        - INTERACTIVE: Start rendered session via SessionController
        - HEADLESS: Submit to trainer daemon for background training
        - EVALUATION: Load policy and run evaluation with rendering

        Args:
            config: Complete launch configuration from the Advanced tab
        """
        _LOGGER.info(
            "Advanced tab: Launch requested | env=%s mode=%s paradigm=%s agents=%d",
            config.env_id,
            config.run_mode.name,
            config.paradigm.value,
            len(config.agent_bindings),
        )

        if config.run_mode == RunMode.INTERACTIVE:
            self._launch_interactive_session(config)
        elif config.run_mode == RunMode.HEADLESS:
            self._launch_headless_training(config)
        elif config.run_mode == RunMode.EVALUATION:
            self._launch_evaluation_session(config)
        else:
            _LOGGER.warning("Unknown run mode: %s", config.run_mode)
            self._status_bar.showMessage(
                f"Unknown run mode: {config.run_mode.name}", 5000
            )

    def _launch_interactive_session(self, config: LaunchConfig) -> None:
        """Launch an interactive session with rendering.

        Uses the existing SessionController infrastructure.

        Args:
            config: Launch configuration
        """
        _LOGGER.info(
            "Launching interactive session: env=%s seed=%d",
            config.env_id,
            config.seed,
        )

        # TODO (Phase 2.2): Full integration with SessionController
        # For now, provide a status message indicating the configuration is ready
        # The PolicyMappingService is already configured by AdvancedConfigTab._configure_policy_mapping()

        human_agents = [
            aid for aid, b in config.agent_bindings.items()
            if b.actor_id == "human_keyboard"
        ]
        ai_agents = [
            aid for aid, b in config.agent_bindings.items()
            if b.actor_id != "human_keyboard"
        ]

        self._status_bar.showMessage(
            f"Interactive: {config.env_id} | {len(human_agents)} human, {len(ai_agents)} AI agents",
            5000,
        )

        # Log the full configuration for debugging
        _LOGGER.debug(
            "Interactive session config: paradigm=%s bindings=%s",
            config.paradigm.value,
            {aid: (b.actor_id, b.worker_id) for aid, b in config.agent_bindings.items()},
        )

    def _launch_headless_training(self, config: LaunchConfig) -> None:
        """Launch headless training via the trainer daemon.

        Builds a training configuration from the LaunchConfig and submits
        it to the TrainerClientRunner.

        Args:
            config: Launch configuration
        """
        _LOGGER.info(
            "Launching headless training: env=%s seed=%d",
            config.env_id,
            config.seed,
        )

        # Build training config from LaunchConfig
        training_config = self._build_training_config_from_launch(config)
        if training_config is None:
            return

        self._status_bar.showMessage(
            f"Submitting training: {config.env_id}", 5000
        )
        self._submit_training_config(training_config)

    def _launch_evaluation_session(self, config: LaunchConfig) -> None:
        """Launch evaluation session with a trained policy.

        Loads the trained policy and runs in evaluation mode with rendering.

        Args:
            config: Launch configuration
        """
        _LOGGER.info(
            "Launching evaluation session: env=%s seed=%d",
            config.env_id,
            config.seed,
        )

        # TODO (Phase 2.2): Full integration with policy loading
        # For now, provide status and log the configuration

        self._status_bar.showMessage(
            f"Evaluation: {config.env_id} (policy loading not yet implemented)",
            5000,
        )

    def _build_training_config_from_launch(
        self, config: LaunchConfig
    ) -> Optional[dict[str, object]]:
        """Build a training configuration dict from LaunchConfig.

        Converts the unified LaunchConfig into the format expected by
        the trainer daemon.

        Args:
            config: Launch configuration from Advanced tab

        Returns:
            Training config dict, or None if configuration fails
        """
        # Determine primary worker
        primary_worker = config.primary_worker_id
        if not primary_worker:
            QtWidgets.QMessageBox.warning(
                self,
                "No Worker Selected",
                "Headless training requires at least one agent with a worker "
                "(e.g., CleanRL, RLlib). Please configure a non-local worker.",
            )
            return None

        # Get worker-specific config
        worker_config = config.worker_configs.get(primary_worker, {})

        # Build training config
        training_config: dict[str, object] = {
            "env_id": config.env_id,
            "seed": config.seed,
            "worker_id": primary_worker,
            "paradigm": config.paradigm.value,
            "agent_bindings": {
                aid: {
                    "actor_id": b.actor_id,
                    "worker_id": b.worker_id,
                    "mode": b.mode,
                }
                for aid, b in config.agent_bindings.items()
            },
            **worker_config,
        }

        _LOGGER.debug(
            "Built training config: %s",
            {k: v for k, v in training_config.items() if k != "agent_bindings"},
        )

        return training_config

    def _on_start_game(self) -> None:
        """Handle Start Game button."""
        status = "Game started"
        if self._episode_finished:
            seed = self._control_panel.current_seed()
            self._session.reset_environment(seed=seed)
            self._episode_finished = False
            status = f"Loaded new episode with seed {seed}. Game started"

        self._session.set_slow_lane_enabled(not self._control_panel.fastlane_only_enabled())

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

    def _on_session_initialized(self, game_id: str, mode: str, _step: object) -> None:
        try:
            mode_label = self.CONTROL_MODE_LABELS[ControlMode(mode)]
        except Exception:
            mode_label = mode
        self._status_bar.showMessage(f"Loaded {game_id} in {mode_label} mode - Click 'Start Game' to begin")
        self.log_constant( 
            LOG_UI_MAINWINDOW_INFO,
            message=f"Loaded {game_id} ({mode_label})",
            extra={"game_id": getattr(game_id, "value", str(game_id)), "mode": mode_label},
        )
        self._auto_running = False
        self._game_started = False
        self._game_paused = False
        self._awaiting_human = False
        self._latest_fps = None
        self._human_input.configure(self._session.game_id, self._session.action_space)
        self._configure_mouse_capture()  # Configure FPS-style mouse capture for ViZDoom
        self._control_panel.set_auto_running(False)
        self._control_panel.set_game_started(False)  # Reset game state on new load
        self._control_panel.set_game_paused(False)
        self._control_panel.set_fps(None)
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
            html = MOSAIC_WELCOME_HTML
        self._game_info.setHtml(html)

    def _configure_mouse_capture(self) -> None:
        """Configure FPS-style mouse capture for ViZDoom games.

        Delegates to VizdoomEnvLoader for all ViZDoom-specific mouse capture setup.
        Click on the Video tab to capture the mouse, ESC or focus-loss to release.
        """
        self._vizdoom_env_loader.configure_mouse_capture(self._session)

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

    def _on_fps_updated(self, fps: float) -> None:
        self._latest_fps = fps if fps > 0 else None
        self._control_panel.set_fps(self._latest_fps)

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


    def _on_trained_agent_requested(self, worker_id: str) -> None:
        """Handle the 'Load Trained Policy' button for a specific worker."""
        if not worker_id:
            QtWidgets.QMessageBox.information(
                self,
                "Worker Required",
                "Select a worker integration before loading a trained policy.",
            )
            return

        factory = get_worker_form_factory()
        try:
            dialog = factory.create_policy_form(
                worker_id,
                parent=self,
                current_game=self._control_panel.current_game(),
            )
        except KeyError:
            QtWidgets.QMessageBox.warning(
                self,
                "Policy form unavailable",
                f"No policy selection form registered for worker '{worker_id}'.",
            )
            return

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        config_builder = getattr(dialog, "get_config", None)
        if callable(config_builder):
            config = config_builder()
            if not config:
                return
            self.log_constant(
                LOG_UI_MAINWINDOW_INFO,
                message="CleanRL policy evaluation submitted",
                extra={"worker_id": worker_id},
            )
            self._status_bar.showMessage("Launching evaluation run...", 5000)
            self._submit_training_config(cast(Dict[str, Any], config))
            return

        policy_path = getattr(dialog, "selected_path", None)
        if policy_path is None:
            return

        self._selected_policy_path = policy_path
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message="Selected policy for evaluation",
            extra={
                "policy_path": str(policy_path),
                "worker_id": worker_id,
            },
        )

        config = self._build_policy_evaluation_config(worker_id, policy_path)
        if config is None:
            return

        self._status_bar.showMessage(
            f"Launching evaluation run for {policy_path.name}",
            5000,
        )
        self._submit_training_config(config)

    def _on_train_agent_requested(self, worker_id: str) -> None:
        """Handle the 'Train Agent' button - opens the training configuration form."""
        if not worker_id:
            QtWidgets.QMessageBox.information(
                self,
                "Worker Required",
                "Select a worker integration before configuring training.",
            )
            return

        factory = get_worker_form_factory()
        form_kwargs: dict[str, Any] = {
            "parent": self,
            "default_game": self._control_panel.current_game(),
        }
        if worker_id == "cleanrl_worker":
            env_id = self._control_panel.cleanrl_environment_id()
            if env_id:
                form_kwargs["default_env_id"] = env_id
        try:
            dialog = factory.create_train_form(
                worker_id,
                **form_kwargs,
            )
        except KeyError:
            QtWidgets.QMessageBox.warning(
                self,
                "Train form unavailable",
                f"No train form registered for worker '{worker_id}'.",
            )
            return

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        get_config = getattr(dialog, "get_config", None)
        if not callable(get_config):
            QtWidgets.QMessageBox.warning(
                self,
                "Unsupported Form",
                "Selected worker form does not provide a configuration payload.",
            )
            return

        config = get_config()
        if config is None:
            return
        if not isinstance(config, dict):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Configuration",
                "Worker form returned an unexpected payload. Expected a dictionary.",
            )
            return

        config_payload = cast(dict[str, object], config)
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message="Agent training configuration submitted from dialog",
            extra={"worker_id": worker_id},
        )
        self._status_bar.showMessage("Launching training run...", 5000)
        self._submit_training_config(config_payload)

    def _on_resume_training_requested(self, worker_id: str) -> None:
        """Handle the 'Resume Training' button - loads checkpoint and continues training."""
        if not worker_id:
            QtWidgets.QMessageBox.information(
                self,
                "Worker Required",
                "Select a worker integration before resuming training.",
            )
            return

        factory = get_worker_form_factory()
        try:
            dialog = factory.create_resume_form(
                worker_id,
                parent=self,
                current_game=self._control_panel.current_game(),
            )
        except KeyError:
            QtWidgets.QMessageBox.warning(
                self,
                "Resume form unavailable",
                f"Resume training for '{worker_id}' is not yet implemented.",
            )
            return

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        get_config = getattr(dialog, "get_config", None)
        if not callable(get_config):
            QtWidgets.QMessageBox.warning(
                self,
                "Unsupported Form",
                "Selected worker form does not provide a configuration payload.",
            )
            return

        config = get_config()
        if config is None:
            return
        if not isinstance(config, dict):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Configuration",
                "Worker form returned an unexpected payload. Expected a dictionary.",
            )
            return

        config_payload = cast(dict[str, object], config)
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message="Resume training configuration submitted from dialog",
            extra={"worker_id": worker_id},
        )
        self._status_bar.showMessage("Resuming training run...", 5000)
        self._submit_training_config(config_payload)

    def _build_policy_evaluation_config(
        self, worker_id: str, policy_path: Path
    ) -> Optional[dict[str, object]]:
        """Build training config using the worker presenter registry.
        
        Delegates configuration composition to the appropriate worker presenter,
        which handles worker-specific logic for config building, metadata composition, etc.
        """
        try:
            registry = get_worker_presenter_registry()
            presenter = registry.get(worker_id)
            
            if presenter is None:
                raise ValueError(f"Worker presenter '{worker_id}' not found in registry")
            
            config = presenter.build_train_request(
                policy_path=policy_path,
                current_game=self._control_panel.current_game(),
            )
            return config
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(
                self,
                "Policy Not Found",
                f"Could not read policy file:\n{policy_path}",
            )
            return None
        except ValueError as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Configuration",
                f"Configuration error:\n{e}",
            )
            return None
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Policy Load Failed",
                f"Could not prepare training request:\n{e}",
            )
            return None

    def _submit_training_config(self, config: dict) -> None:
        """Submit a training configuration to the trainer daemon."""
        try:
            self.log_constant( 
                LOG_UI_MAINWINDOW_TRACE,
                message="_submit_training_config: START",
            )
            config_json = json.dumps(config)
            self.log_constant( 
                LOG_UI_MAINWINDOW_TRACE,
                message=f"_submit_training_config: Config JSON length={len(config_json)}",
                extra={"config_length": len(config_json)},
            )

            locator = get_service_locator()
            runner = locator.resolve(TrainerClientRunner)
            if runner is None:
                raise RuntimeError("TrainerClientRunner not registered")

            self.log_constant( 
                LOG_UI_MAINWINDOW_TRACE,
                message="_submit_training_config: TrainerClientRunner resolved",
            )
            self._status_bar.showMessage("Submitting training run...", 3000)

            # Submit returns a Future
            future = runner.submit_run(config_json, deadline=_training_submit_deadline_seconds())
            self.log_constant( 
                LOG_UI_MAINWINDOW_TRACE,
                message="_submit_training_config: submit_run() called, future created",
            )

            # Add callback to handle result
            def on_done(fut):
                self.log_constant( 
                    LOG_UI_MAINWINDOW_TRACE,
                    message="_submit_training_config: on_done callback called",
                )
                try:
                    response = fut.result()
                    self.log_constant( 
                        LOG_UI_MAINWINDOW_TRACE,
                        message=f"_submit_training_config: Got response with run_id={response.run_id}",
                        extra={"run_id": str(response.run_id)},
                    )
                except Exception as error:
                    if isinstance(error, grpc.aio.AioRpcError) and error.code() == grpc.StatusCode.INVALID_ARGUMENT:
                        self.log_constant( 
                            LOG_UI_MAINWINDOW_INVALID_CONFIG,
                            message="Trainer rejected training config",
                            extra={
                                "exception": type(error).__name__,
                                "details": error.details() if hasattr(error, "details") else "",
                            },
                            exc_info=error,
                        )
                    else:
                        self.log_constant( 
                            LOG_UI_MAINWINDOW_ERROR,
                            message="_submit_training_config: on_done got exception",
                            extra={"exception": type(error).__name__},
                            exc_info=error,
                        )
                    QtCore.QTimer.singleShot(0, lambda: self._on_training_submit_failed(error))
                    return
                run_id = str(response.run_id)
                self.log_constant( 
                    LOG_UI_MAINWINDOW_INFO,
                    message=f"_submit_training_config: Scheduling _on_training_submitted with run_id={run_id}",
                    extra={"run_id": run_id},
                )
                QtCore.QTimer.singleShot(0, lambda: self._on_training_submitted(run_id, config))

            future.add_done_callback(on_done)
            self.log_constant( 
                LOG_UI_MAINWINDOW_TRACE,
                message="_submit_training_config: Callback added to future",
            )

        except Exception as e:
            self.log_constant( 
                LOG_UI_MAINWINDOW_ERROR,
                message="Failed to prepare training submission",
                extra={"exception": type(e).__name__},
                exc_info=e,
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Training Preparation Failed",
                f"Could not prepare training request:\n{e}",
            )
    
    def _on_training_submitted(self, run_id: str, config: dict) -> None:
        """Handle successful training submission (called on main thread)."""
        self._status_bar.showMessage(f"Training run submitted: {run_id[:12]}...", 5000)
        self.log_constant( 
            LOG_UI_MAINWINDOW_INFO,
            message="Submitted training run",
            extra={"run_id": run_id, "config": config},
        )

        metadata = config.get("metadata", {})
        environment = config.get("environment", {})
        environment_dict = environment if isinstance(environment, dict) else {}

        # Extract buffer sizes from config and set them in the controller
        try:
            ui_config = metadata.get("ui", {})
            step_buffer_size = ui_config.get("telemetry_buffer_size", 100)
            episode_buffer_size = ui_config.get("episode_buffer_size", 100)
            hub_buffer_size = ui_config.get("hub_buffer_size")  # Hub buffer from training config
            
            self._live_controller.set_buffer_sizes_for_run(run_id, step_buffer_size, episode_buffer_size)
            
            # Set hub buffer size if provided
            if hub_buffer_size is not None:
                self._telemetry_hub.set_run_buffer_size(run_id, hub_buffer_size)
                self.log_constant( 
                    LOG_UI_MAINWINDOW_TRACE,
                    message="Set hub buffer size for run",
                    extra={
                        "run_id": run_id,
                        "hub_buffer_size": hub_buffer_size,
                    },
                )
            
            self.log_constant( 
                LOG_UI_MAINWINDOW_TRACE,
                message="Set buffer sizes for run",
                extra={
                    "run_id": run_id,
                    "step_buffer_size": step_buffer_size,
                    "episode_buffer_size": episode_buffer_size,
                    "hub_buffer_size": hub_buffer_size,
                },
            )
        except Exception as e:
            self.log_constant( 
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Failed to set buffer sizes: {e}",
                extra={"exception": type(e).__name__},
                exc_info=e,
            )

        # Extract game_id from environment and store in controller
        try:
            game_id = environment_dict.get("GYM_ENV_ID", "")
            if game_id:
                self._live_controller.set_game_id_for_run(run_id, game_id)
                self.log_constant( 
                    LOG_UI_MAINWINDOW_TRACE,
                    message="Set game_id for run",
                    extra={"run_id": run_id, "game_id": game_id},
                )
        except Exception as e:
            self.log_constant( 
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Failed to set game_id: {e}",
                extra={"exception": type(e).__name__},
                exc_info=e,
            )

        self.log_constant( 
            LOG_UI_MAINWINDOW_INFO,
            message="Waiting for live telemetry to create dynamic agent tabs",
            extra={
                "run_id": run_id,
                "expected_tabs": [
                    "Agent-{agent_id}-Replay",
                    "Agent-{agent_id}-Online-Grid",
                    "Agent-{agent_id}-Online-Raw",
                    "Agent-{agent_id}-Online-Video",
                ],
            },
        )

        # Extract UI rendering throttle from environment variables and set it on the controller
        if environment_dict:
            try:
                throttle_str = environment_dict.get("TELEMETRY_SAMPLING_INTERVAL", "2")
                throttle_interval = int(throttle_str)
                self._live_controller.set_render_throttle_for_run(run_id, throttle_interval)
                self.log_constant( 
                    LOG_UI_MAINWINDOW_TRACE,
                    message="Set render throttle for run",
                    extra={"run_id": run_id, "throttle": throttle_interval},
                )
            except (ValueError, TypeError):
                self.log_constant( 
                    LOG_UI_MAINWINDOW_WARNING,
                    message="Failed to parse TELEMETRY_SAMPLING_INTERVAL",
                    extra={
                        "run_id": run_id,
                        "value": environment_dict.get("TELEMETRY_SAMPLING_INTERVAL"),
                    },
                )

        # Apply render delay and enable flag from metadata (defaulting to enabled)
        ui_config = metadata.get("ui", {}) if isinstance(metadata, dict) else {}
        render_delay_ms = int(environment_dict.get("UI_RENDER_DELAY_MS", ui_config.get("render_delay_ms", 100)))
        live_rendering_enabled = ui_config.get("live_rendering_enabled", True)
        self._live_controller.set_render_delay_for_run(run_id, int(render_delay_ms))
        self._live_controller.set_live_render_enabled_for_run(run_id, bool(live_rendering_enabled))

        # Persist metadata keyed by (run_id, agent_id)
        worker_meta = metadata.get("worker", {}) if isinstance(metadata, dict) else {}
        worker_config = worker_meta.get("config", {}) if isinstance(worker_meta, dict) else {}
        agent_id_key = worker_meta.get("agent_id") or worker_config.get("agent_id") or "default"
        self._run_metadata[(run_id, agent_id_key)] = metadata

        # Attempt to provision FastLane tab immediately (fastlane-only runs may never emit telemetry)
        self._maybe_open_fastlane_tab(run_id, agent_id_key)

        tb_ready = self._analytics_tabs.ensure_tensorboard_tab(run_id, agent_id_key, metadata)
        wb_ready = self._analytics_tabs.ensure_wandb_tab(run_id, agent_id_key, metadata)

        if not wb_ready:
            # Schedule retries so that WANDB manifest written after initialization triggers the tab.
            self._analytics_tabs.load_and_create_tabs(run_id, agent_id_key)

        if not tb_ready:
            # TensorBoard manifests usually exist up front; if they do not, reuse the same retry path.
            self._analytics_tabs.load_and_create_tabs(run_id, agent_id_key)

        # Subscribe to telemetry
        # NOTE: Do NOT call _create_agent_tabs_for() here!
        # The LiveTelemetryController will create the LiveTelemetryTab dynamically
        # when the first telemetry event arrives. This ensures proper naming and
        # routing of telemetry data to the correct tab.
        self._live_controller.subscribe_to_run(run_id)
        self._render_group.setTitle(f"Live Training - {run_id[:12]}...")

    def _on_training_submit_failed(self, error: Exception) -> None:
        """Handle training submission failure (called on main thread)."""
        self.log_constant( 
            LOG_UI_MAINWINDOW_ERROR,
            message="Failed to submit training run",
            extra={"exception": type(error).__name__},
            exc_info=error,
        )
        QtWidgets.QMessageBox.critical(
            self,
            "Training Submission Failed",
            f"Could not submit training run:\n{error}\n\n"
            "Make sure the trainer daemon is running:\n"
            "  python -m gym_gui.services.trainer_daemon",
        )
    
    def _on_live_telemetry_tab_requested(self, run_id: str, agent_id: str, tab_title: str) -> None:
        """Create and register a new live telemetry tab dynamically."""

        locator = get_service_locator()
        renderer_registry = locator.resolve(RendererRegistry)

        # Get buffer sizes from controller (set during training submission)
        step_buffer_size, episode_buffer_size = self._live_controller.get_buffer_sizes_for_run(run_id)

        # Get game_id from controller (set during training submission)
        game_id_str = self._live_controller.get_game_id_for_run(run_id)
        game_id = None
        if game_id_str:
            try:
                game_id = GameId(game_id_str)
            except (ValueError, KeyError):
                self.log_constant( 
                    LOG_UI_MAINWINDOW_WARNING,
                    message="Invalid game_id for run",
                    extra={"run_id": run_id, "game_id": game_id_str},
                )

        # Get render throttle interval from controller (set during training submission)
        render_throttle_interval = self._live_controller.get_render_throttle_for_run(run_id)
        render_delay_ms = self._live_controller.get_render_delay_for_run(run_id)
        live_render_enabled = self._live_controller.is_live_render_enabled(run_id)

        tab = LiveTelemetryTab(
            run_id,
            agent_id,
            game_id=game_id,
            buffer_size=step_buffer_size,
            episode_buffer_size=episode_buffer_size,
            render_throttle_interval=render_throttle_interval,
            render_delay_ms=render_delay_ms,
            live_render_enabled=live_render_enabled,
            renderer_registry=renderer_registry,
            parent=self._render_tabs,
        )
        self._live_controller.register_tab(run_id, agent_id, tab)

        # Add to render tabs widget using add_dynamic_tab to include close button
        self._render_tabs.add_dynamic_tab(run_id, tab_title, tab)

        self.log_constant( 
            LOG_UI_MAINWINDOW_INFO,
            message="Created live telemetry tab",
            extra={"run_id": run_id, "agent_id": agent_id, "title": tab_title, "game_id": game_id_str},
        )

        self._maybe_open_fastlane_tab(run_id, agent_id)

    # NOTE: Removed _on_live_step_received and _on_live_episode_received
    # The LiveTelemetryController now owns all routing (tab creation and step/episode delivery).
    # Main window only handles tab creation via run_tab_requested signal.

    def _create_agent_tabs_for(self, run_id: str, agent_id: str, first_payload: dict) -> None:
        """Create dynamic agent tabs using the worker presenter registry.

        For ToyText environments (FrozenLake, CliffWalking, Taxi):
        - Online tab shows grid rendering (primary view)
        - Replay tab shows episode browser

        For visual environments (Atari, etc.):
        - Online tab shows video rendering
        - Replay tab shows episode browser
        
        Delegates tab creation to the appropriate worker presenter based on
        the worker type, which handles environment detection and conditional
        tab instantiation.
        """
        try:
            # Get the presenter registry and resolve the worker presenter
            registry = get_worker_presenter_registry()
            worker_id = "spade_bdi_worker"  # TODO: Extract from config/payload if supporting multiple workers
            presenter = registry.get(worker_id)
            
            if presenter is None:
                self.log_constant( 
                    LOG_UI_WORKER_TABS_ERROR,
                    message="Worker presenter not found in registry",
                    extra={"run_id": run_id, "agent_id": agent_id, "worker_id": worker_id},
                )
                return
            
            tabs = presenter.create_tabs(run_id, agent_id, first_payload, parent=self)
            
            # Tab names and registration order must match presenter output
            tab_names = [
                f"Agent-{agent_id}-Online",
                f"Agent-{agent_id}-Replay",
                f"Agent-{agent_id}-Live – Grid",
                f"Agent-{agent_id}-Debug",
            ]
            
            # Determine if video tab was created (check if environment is visual)
            game_id_str = first_payload.get("game_id", "").lower()
            is_toytext = any(name in game_id_str for name in ["frozenlake", "cliffwalking", "taxi", "gridworld"])
            
            if not is_toytext:
                tab_names.append(f"Agent-{agent_id}-Live – Video")
            
            # Register tabs with the render container
            if len(tabs) != len(tab_names):
                self.log_constant( 
                    LOG_UI_WORKER_TABS_WARNING,
                    message="Tab count mismatch",
                    extra={
                        "run_id": run_id,
                        "agent_id": agent_id,
                        "expected": len(tab_names),
                        "actual": len(tabs),
                    },
                )
            
            for tab_name, tab_widget in zip(tab_names, tabs):
                self._render_tabs.add_dynamic_tab(run_id, tab_name, tab_widget)
            
            # Update metadata if available
            metadata = self._run_metadata.get((run_id, agent_id))
            if metadata:
                grid_tab = tabs[2] if len(tabs) > 2 else None
                if grid_tab and hasattr(grid_tab, "update_metadata"):
                    grid_tab.update_metadata(metadata)
            
            self.log_constant( 
                LOG_UI_WORKER_TABS_INFO,
                message="Created dynamic agent tabs via presenter registry",
                extra={
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "worker_id": worker_id,
                    "game_id": game_id_str,
                    "is_toytext": is_toytext,
                    "tabs": tab_names,
                },
            )
        except Exception as e:
            self.log_constant( 
                LOG_UI_WORKER_TABS_ERROR,
                message="Failed to create agent tabs",
                extra={"run_id": run_id, "agent_id": agent_id, "error": str(e)},
                exc_info=e,
            )

    def _resolve_run_metadata(self, run_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        meta = self._run_metadata.get((run_id, agent_id))
        if meta is not None:
            return meta
        for (stored_run_id, _stored_agent), stored_meta in self._run_metadata.items():
            if stored_run_id == run_id:
                return stored_meta
        return None

    def _maybe_open_fastlane_tab(self, run_id: str, agent_id: str) -> None:
        metadata = self._resolve_run_metadata(run_id, agent_id)
        if not metadata:
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="FastLane tab skipped; metadata missing",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            return
        if not self._metadata_supports_fastlane(metadata):
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="FastLane tab skipped; metadata does not advertise fastlane",
                extra={"run_id": run_id, "agent_id": agent_id},
            )
            return

        canonical_agent_id = self._canonical_agent_id(metadata, agent_id)
        run_mode = self._metadata_run_mode(metadata)
        key = (run_id, canonical_agent_id)
        if key in self._fastlane_tabs_open:
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="FastLane tab already tracked",
                extra={"run_id": run_id, "agent_id": canonical_agent_id},
            )
            return

        try:
            mode_label = "Fast lane (evaluation)" if run_mode == "policy_eval" else "Fast lane"
            tab = FastLaneTab(
                run_id,
                canonical_agent_id,
                mode_label=mode_label,
                run_mode=run_mode,
                parent=self._render_tabs,
            )
        except Exception as exc:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message="Failed to create FastLane tab",
                extra={"run_id": run_id, "agent_id": canonical_agent_id},
                exc_info=exc,
            )
            return

        if run_mode == "policy_eval":
            title = f"CleanRL-Eval-{canonical_agent_id or 'agent'}"
        else:
            title = f"CleanRL-Live-{canonical_agent_id or 'agent'}"
        self._render_tabs.add_dynamic_tab(run_id, title, tab)
        self._fastlane_tabs_open.add(key)
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message="Opened FastLane tab",
            extra={"run_id": run_id, "agent_id": canonical_agent_id, "title": title},
        )

    def _canonical_agent_id(self, metadata: Dict[str, Any], fallback: str) -> str:
        worker_meta = metadata.get("worker") if isinstance(metadata, dict) else None
        if isinstance(worker_meta, dict):
            meta_agent = worker_meta.get("agent_id")
            if isinstance(meta_agent, str) and meta_agent.strip():
                return meta_agent.strip()
            worker_config = worker_meta.get("config")
            if isinstance(worker_config, dict):
                config_agent = worker_config.get("agent_id")
                if isinstance(config_agent, str) and config_agent.strip():
                    return config_agent.strip()
        return fallback

    def _metadata_run_mode(self, metadata: Dict[str, Any]) -> str:
        ui_meta = metadata.get("ui") if isinstance(metadata, dict) else None
        if isinstance(ui_meta, dict):
            mode = ui_meta.get("run_mode")
            if isinstance(mode, str) and mode.strip():
                return mode.strip().lower()
        worker_meta = metadata.get("worker") if isinstance(metadata, dict) else None
        if isinstance(worker_meta, dict):
            config = worker_meta.get("config")
            if isinstance(config, dict):
                extras = config.get("extras")
                if isinstance(extras, dict):
                    mode = extras.get("mode")
                    if isinstance(mode, str) and mode.strip():
                        return mode.strip().lower()
        return "train"

    def _metadata_supports_fastlane(self, metadata: Dict[str, Any]) -> bool:
        """Return True if the run metadata indicates FastLane visuals are available."""

        def _is_truthy(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return False

        ui_meta = metadata.get("ui") if isinstance(metadata, dict) else None
        if isinstance(ui_meta, dict) and _is_truthy(ui_meta.get("fastlane_only")):
            return True

        worker_meta = metadata.get("worker") if isinstance(metadata, dict) else None
        if not isinstance(worker_meta, dict):
            return False

        module_name = str(worker_meta.get("module") or "").lower()
        worker_kind = str(worker_meta.get("worker_kind") or "").lower()
        worker_identifier = str(worker_meta.get("worker_id") or "").lower()
        if "cleanrl_worker" in module_name or worker_kind == "cleanrl" or worker_identifier == "cleanrl_worker":
            return True

        worker_config = worker_meta.get("config")
        if isinstance(worker_config, dict):
            extras = worker_config.get("extras")
            if isinstance(extras, dict):
                if _is_truthy(extras.get("fastlane_only")) or _is_truthy(extras.get("fastlane_enabled")):
                    return True

        return False

    def _backfill_run_metadata_from_disk(self, run_id: str) -> None:
        """Load run metadata for previously scheduled runs discovered outside the submission flow."""
        config_path = VAR_TRAINER_DIR / "configs" / f"config-{run_id}.json"
        if not config_path.exists():
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="Metadata backfill skipped; trainer config not found",
                extra={"run_id": run_id, "path": str(config_path)},
            )
            return

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config_payload = json.load(handle)
        except Exception as exc:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message="Failed to load trainer config for metadata backfill",
                extra={"run_id": run_id, "path": str(config_path)},
                exc_info=exc,
            )
            return

        metadata = config_payload.get("metadata")
        if not isinstance(metadata, dict):
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="Metadata backfill skipped; config payload missing metadata",
                extra={"run_id": run_id, "path": str(config_path)},
            )
            return

        candidate_agent_ids: List[str] = []
        worker_meta = metadata.get("worker")
        if isinstance(worker_meta, dict):
            worker_agent = worker_meta.get("agent_id")
            if worker_agent is not None:
                candidate_agent_ids.append(str(worker_agent))
            worker_config = worker_meta.get("config")
            if isinstance(worker_config, dict):
                config_agent = worker_config.get("agent_id")
                if config_agent is not None:
                    candidate_agent_ids.append(str(config_agent))

        env_payload = config_payload.get("environment")
        if isinstance(env_payload, dict):
            env_agent = env_payload.get("TRAIN_AGENT_ID")
            if env_agent is not None:
                candidate_agent_ids.append(str(env_agent))

        if not candidate_agent_ids:
            candidate_agent_ids.append("default")

        ordered_unique_agent_ids: List[str] = []
        for agent_id in candidate_agent_ids:
            if agent_id not in ordered_unique_agent_ids:
                ordered_unique_agent_ids.append(agent_id)

        artifacts_payload = metadata.get("artifacts") if isinstance(metadata, dict) else None

        for agent_id in ordered_unique_agent_ids:
            key = (run_id, agent_id)
            if key in self._run_metadata:
                existing_payload = self._run_metadata[key]
                if isinstance(existing_payload, dict):
                    current_artifacts = existing_payload.get("artifacts")
                    if not isinstance(current_artifacts, dict) and isinstance(artifacts_payload, dict):
                        existing_payload["artifacts"] = artifacts_payload
                        self.log_constant(
                            LOG_UI_MAINWINDOW_TRACE,
                            message="Backfilled tensorboard artifacts into existing metadata",
                            extra={"run_id": run_id, "agent_id": agent_id},
                        )
                meta_payload = self._run_metadata.get(key)
                self._analytics_tabs.ensure_tensorboard_tab(run_id, agent_id, meta_payload)
                self._analytics_tabs.ensure_wandb_tab(run_id, agent_id, meta_payload)
                continue

            self._run_metadata[key] = metadata
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="Backfilled run metadata from trainer config",
                extra={"run_id": run_id, "agent_id": agent_id, "path": str(config_path)},
            )
            self._analytics_tabs.ensure_tensorboard_tab(run_id, agent_id, metadata)
            self._analytics_tabs.ensure_wandb_tab(run_id, agent_id, metadata)
            self._maybe_open_fastlane_tab(run_id, agent_id)

    def _on_training_finished(self, run_id: str, outcome: str, failure_reason: str) -> None:
        """Handle training_finished signal - create/refresh replay tabs for all agents in this run."""
        self.log_constant( 
            LOG_UI_MAINWINDOW_INFO,
            message="Training finished signal received",
            extra={"run_id": run_id, "outcome": outcome, "failure_reason": failure_reason},
        )

        # Get all agents that participated in this run
        agent_tabs = self._render_tabs._agent_tabs.get(run_id, {})
        
        self.log_constant( 
            LOG_UI_MAINWINDOW_TRACE,
            message="_on_training_finished: agent_tabs for run",
            extra={"run_id": run_id, "tab_count": len(agent_tabs), "tab_names": list(agent_tabs.keys())},
        )

        # Extract unique agent IDs from tab names (e.g., "Agent-1-Online" -> agent_id="1")
        agent_ids_with_tabs = set()
        for tab_name in agent_tabs.keys():
            # Tab names follow pattern: "Agent-{agent_id}-*"
            if tab_name.startswith("Agent-"):
                parts = tab_name.split("-")
                if len(parts) >= 2:
                    agent_id = parts[1]
                    agent_ids_with_tabs.add(agent_id)

        if not agent_ids_with_tabs:
            # Analytics-only runs (Fast Path) may never instantiate live tabs. Fall back to
            # metadata captured at submission time so we can surface analytics tabs.
            for (meta_run_id, meta_agent_id), _metadata in self._run_metadata.items():
                if meta_run_id != run_id:
                    continue
                if not meta_agent_id:
                    continue
                agent_ids_with_tabs.add(meta_agent_id)

            if agent_ids_with_tabs:
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="_on_training_finished: using metadata agent ids",
                    extra={"run_id": run_id, "agent_ids": list(agent_ids_with_tabs)},
                )
            else:
                # Guarantee downstream logic executes at least once; analytics tabs will use
                # "default" which matches legacy emitter behaviour.
                agent_ids_with_tabs.add("default")
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="_on_training_finished: no agent tabs or metadata; defaulting",
                    extra={"run_id": run_id},
                )

        self.log_constant(
            LOG_UI_MAINWINDOW_TRACE,
            message="_on_training_finished: extracted agent IDs",
            extra={"run_id": run_id, "agent_ids": list(agent_ids_with_tabs)},
        )

        # Create or refresh replay tabs for each agent, and switch to the first one created
        first_replay_tab_index: int | None = None
        
        for agent_id in agent_ids_with_tabs:
            replay_tab_name = f"Agent-{agent_id}-Replay"
            
            self.log_constant( 
                LOG_UI_MAINWINDOW_TRACE,
                message="_on_training_finished: processing agent",
                extra={"run_id": run_id, "agent_id": agent_id, "replay_tab_name": replay_tab_name},
            )

            # Load analytics.json from disk and create/refresh analytics tabs (TensorBoard, WANDB)
            self._analytics_tabs.load_and_create_tabs(run_id, agent_id)

            # Check if replay tab already exists
            if replay_tab_name in agent_tabs:
                # Refresh existing replay tab
                try:
                    tab_widget = agent_tabs[replay_tab_name]
                    refresh = getattr(tab_widget, "refresh", None)
                    if callable(refresh):
                        self.log_constant( 
                            LOG_UI_MAINWINDOW_TRACE,
                            message="_on_training_finished: calling refresh on existing replay tab",
                            extra={"run_id": run_id, "agent_id": agent_id},
                        )
                        refresh()
                        self.log_constant( 
                            LOG_UI_MAINWINDOW_TRACE,
                            message="Refreshed replay tab",
                            extra={"run_id": run_id, "tab_name": replay_tab_name},
                        )
                    # Record the tab index for switching later
                    if first_replay_tab_index is None:
                        first_replay_tab_index = self._render_tabs.indexOf(tab_widget)
                        self.log_constant( 
                            LOG_UI_MAINWINDOW_TRACE,
                            message="_on_training_finished: recorded existing replay tab index",
                            extra={"run_id": run_id, "agent_id": agent_id, "tab_index": first_replay_tab_index},
                        )
                except Exception as e:
                    self.log_constant( 
                        LOG_UI_MAINWINDOW_WARNING,
                        message="Failed to refresh replay tab",
                        exc_info=e,
                        extra={"run_id": run_id, "tab_name": replay_tab_name},
                    )
            else:
                # Create new replay tab for this agent
                self.log_constant( 
                    LOG_UI_MAINWINDOW_TRACE,
                    message="_on_training_finished: replay tab does not exist, creating new",
                    extra={"run_id": run_id, "agent_id": agent_id, "replay_tab_name": replay_tab_name},
                )
                try:
                    replay = AgentReplayTab(run_id, agent_id, parent=self)
                    self._render_tabs.add_dynamic_tab(run_id, replay_tab_name, replay)
                    self.log_constant( 
                        LOG_UI_MAINWINDOW_INFO,
                        message="Created replay tab for agent",
                        extra={"run_id": run_id, "agent_id": agent_id, "tab_name": replay_tab_name},
                    )
                    # Record the tab index for switching later (add_dynamic_tab already switches, but we track it)
                    if first_replay_tab_index is None:
                        first_replay_tab_index = self._render_tabs.indexOf(replay)
                        self.log_constant( 
                            LOG_UI_MAINWINDOW_TRACE,
                            message="_on_training_finished: recorded new replay tab index",
                            extra={"run_id": run_id, "agent_id": agent_id, "tab_index": first_replay_tab_index},
                        )
                except Exception as e:
                    self.log_constant( 
                        LOG_UI_MAINWINDOW_WARNING,
                        message="Failed to create replay tab",
                        exc_info=e,
                        extra={"run_id": run_id, "agent_id": agent_id},
                    )
        
        self.log_constant( 
            LOG_UI_MAINWINDOW_TRACE,
            message="_on_training_finished: about to switch to replay tab",
            extra={"run_id": run_id, "first_replay_tab_index": first_replay_tab_index},
        )
        
        # Switch to the first replay tab created/refreshed so user can see results
        if first_replay_tab_index is not None and first_replay_tab_index >= 0:
            self._render_tabs.setCurrentIndex(first_replay_tab_index)
            self.log_constant( 
                LOG_UI_MAINWINDOW_INFO,
                message="Switched to replay tab after training completion",
                extra={"run_id": run_id, "tab_index": first_replay_tab_index},
            )
        else:
            self.log_constant( 
                LOG_UI_MAINWINDOW_WARNING,
                message="_on_training_finished: could not find valid replay tab to switch to",
                extra={"run_id": run_id, "first_replay_tab_index": first_replay_tab_index},
            )

    def _on_run_completed(self, run_id: str) -> None:
        """Handle run completion - keep Live-Agent tabs open, add Replay tabs."""
        self.log_constant( 
            LOG_LIVE_CONTROLLER_RUN_COMPLETED,
            message="Run completed signal received",
            extra={"run_id": run_id},
        )

        self._fastlane_tabs_open = {key for key in self._fastlane_tabs_open if key[0] != run_id}

        # Unsubscribe from telemetry (stops new events from arriving)
        if self._live_controller:
            try:
                self._live_controller.unsubscribe_from_run(run_id)
                self.log_constant( 
                    LOG_UI_MAINWINDOW_TRACE,
                    message="Unsubscribed from telemetry",
                    extra={"run_id": run_id},
                )
            except Exception as e:
                self.log_constant( 
                    LOG_UI_MAINWINDOW_WARNING,
                    message="Failed to unsubscribe from telemetry",
                    exc_info=e,
                    extra={"run_id": run_id},
                )

        # NOTE: Do NOT remove Live-Agent tabs - keep them open so user can review the training
        # The Live-Agent tab will remain visible with the final state
        # Replay tabs will be created by _on_training_finished() signal
        self.log_constant( 
            LOG_LIVE_CONTROLLER_RUN_COMPLETED,
            message="Run completed - Live-Agent tabs remain open for review",
            extra={"run_id": run_id},
        )

    def _poll_for_new_runs(self) -> None:
        """Poll daemon for new training runs and auto-subscribe to their telemetry."""
        try:
            
            locator = get_service_locator()
            runner = locator.resolve(TrainerClientRunner)
            if runner is None:
                self.log_constant( 
                    LOG_UI_MAINWINDOW_TRACE,
                    message="TrainerClientRunner not available, skipping poll",
                )
                return
            
            # List runs that should have tabs (active or recently completed)
            if self._trainer_daemon_ready:
                self.log_constant( 
                    LOG_UI_MAINWINDOW_TRACE,
                    message="Polling daemon for active training runs...",
                )
            future = runner.list_runs(
                statuses=self._WATCHED_RUN_STATUSES,
                deadline=3.0,
            )
            
            def on_done(fut):
                try:
                    response = fut.result(timeout=1.0)
                except Exception as exc:
                    self._trainer_poll_failures += 1
                    if self._trainer_daemon_ready:
                        self.log_constant( 
                            LOG_UI_MAINWINDOW_TRACE,
                            message=f"Run poll failed: {exc}",
                        )
                    else:
                        if not self._trainer_poll_quiet_logged:
                            self.log_constant(
                                LOG_UI_MAINWINDOW_TRACE,
                                message="Trainer daemon not yet reachable; suppressing poll failures until it responds",
                            )
                            self._trainer_poll_quiet_logged = True
                    return
                self._trainer_daemon_ready = True
                if self._trainer_poll_failures and self._trainer_poll_quiet_logged:
                    self.log_constant(
                        LOG_UI_MAINWINDOW_TRACE,
                        message="Trainer daemon responded; resuming poll logging",
                    )
                self._trainer_poll_failures = 0
                self._trainer_poll_quiet_logged = False
                self.log_constant( 
                    LOG_UI_MAINWINDOW_TRACE,
                    message=f"Received {len(response.runs)} active runs from daemon",
                )
                for record in response.runs:
                    run_id = str(record.run_id)
                    if run_id not in self._known_runs:
                        # Convert protobuf status integer to human-readable name
                        status_value = record.status
                        status_name = status_value
                        if isinstance(status_value, int):
                            from gym_gui.services.trainer import RunStatus
                            status_name = RunStatus.from_proto(status_value).value
                        self.log_constant( 
                            LOG_UI_MAINWINDOW_TRACE,
                            message=f"Discovered new run: {run_id} (status={status_name}, proto={status_value})",
                        )
                        self.log_constant( 
                            LOG_UI_MAINWINDOW_TRACE,
                            message=f"Calling auto-subscribe directly for run: {run_id}",
                        )
                        self._auto_subscribe_run(run_id)
                    else:
                        self.log_constant( 
                            LOG_UI_MAINWINDOW_TRACE,
                            message=f"Run {run_id[:12]} already known, skipping",
                        )
            
            future.add_done_callback(on_done)
            
        except Exception as e:
            self.log_constant( 
                LOG_UI_MAINWINDOW_TRACE,
                message="Failed to initiate run poll",
                exc_info=e,
            )
    
    def _auto_subscribe_run(self, run_id: str) -> None:
        """Ensure auto-subscribe logic executes on the GUI thread."""
        current_thread = QtCore.QThread.currentThread()
        widget_thread = self.thread()
        if current_thread != widget_thread:
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message="Queueing auto-subscribe on GUI thread",
                extra={
                    "run_id": run_id,
                    "current_thread": repr(current_thread),
                    "widget_thread": repr(widget_thread),
                },
            )
            self._auto_subscribe_requested.emit(run_id)
            return

        self._auto_subscribe_run_main_thread(run_id)

    @pyqtSlot(str)
    def _auto_subscribe_run_main_thread(self, run_id: str) -> None:
        """Auto-subscribe to a newly discovered run (always called on main thread)."""
        self.log_constant( 
            LOG_UI_MAINWINDOW_INFO,
            message="Auto-subscribing to new run",
            extra={"run_id": run_id},
        )
        self._known_runs.add(run_id)
        self._backfill_run_metadata_from_disk(run_id)
        try:
            self._live_controller.subscribe_to_run(run_id)
            self._render_group.setTitle(f"Live Training - {run_id[:12]}...")
            self._status_bar.showMessage(f"Detected new training run: {run_id[:12]}...", 5000)
            self.log_constant( 
                LOG_UI_MAINWINDOW_INFO,
                message="Subscribed to run - waiting for telemetry steps to create agent tabs",
                extra={"run_id": run_id},
            )
        except Exception as e:
            self.log_constant( 
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Failed to subscribe to run {run_id}",
                exc_info=e,
            )

    def _start_run_watch(self) -> None:
        locator = get_service_locator()
        runner: Optional[TrainerClientRunner] = locator.resolve(TrainerClientRunner)
        if runner is None:
            self.log_constant( 
                LOG_UI_MAINWINDOW_TRACE,
                message="TrainerClientRunner not available; skipping run watch subscription",
            )
            return

        subscription = runner.watch_runs(statuses=self._WATCHED_RUN_STATUSES, since_seq=0)
        self._run_watch_subscription = subscription

        def _watch_loop() -> None:
            self.log_constant( 
                LOG_UI_MAINWINDOW_INFO,
                message=f"Run watch thread started (statuses={','.join([status.name for status in self._WATCHED_RUN_STATUSES])})",
            )
            while not self._run_watch_stop.is_set():
                try:
                    record = subscription.get(timeout=1.0)
                except TrainerWatchStopped:
                    self.log_constant( 
                        LOG_UI_MAINWINDOW_INFO,
                        message="Run watch subscription closed by daemon",
                    )
                    break
                except TimeoutError:
                    continue
                except Exception as exc:
                    self.log_constant( 
                        LOG_UI_MAINWINDOW_TRACE,
                        message=f"Run watch error: {exc}",
                    )
                    continue

                run_id = getattr(record, "run_id", "")
                status_value = getattr(record, "status", None)
                # Convert protobuf status integer to human-readable name
                status_name = status_value
                if isinstance(status_value, int):
                    from gym_gui.services.trainer import RunStatus
                    status_name = RunStatus.from_proto(status_value).value
                self.log_constant( 
                    LOG_UI_MAINWINDOW_INFO,
                    message=f"Run watch update: run_id={run_id} status={status_name} (proto={status_value})",
                )
                QtCore.QTimer.singleShot(0, lambda rid=run_id: self._auto_subscribe_run(rid))

            subscription.close()
            self.log_constant( 
                LOG_UI_MAINWINDOW_INFO,
                message="Run watch thread exiting",
            )

        self._run_watch_thread = threading.Thread(
            target=_watch_loop,
            name="trainer-run-watch",
            daemon=True,
        )
        self._run_watch_thread.start()

    def _shutdown_run_watch(self) -> None:
        self._run_watch_stop.set()
        if self._run_watch_subscription is not None:
            try:
                self._run_watch_subscription.close()
            except Exception as exc:
                self.log_constant(
                    LOG_UI_MAIN_WINDOW_SHUTDOWN_WARNING,
                    message="Failed to close run watch subscription during shutdown",
                    extra={
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    exc_info=exc,
                )
            self._run_watch_subscription = None
        if self._run_watch_thread is not None:
            self._run_watch_thread.join(timeout=2.0)
            self._run_watch_thread = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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

    def closeEvent(self, a0: QtGui.QCloseEvent | None) -> None:
        logging.getLogger().removeHandler(self._log_handler)

        # Shutdown live telemetry controller
        if hasattr(self, "_live_controller"):
            self._live_controller.shutdown()

        self._shutdown_run_watch()

        # Clean up board games (via env loaders)
        if hasattr(self, "_chess_env_loader"):
            self._chess_env_loader.cleanup()
        if hasattr(self, "_connect_four_env_loader"):
            self._connect_four_env_loader.cleanup()
        if hasattr(self, "_go_env_loader"):
            self._go_env_loader.cleanup()
        if hasattr(self, "_tictactoe_env_loader"):
            self._tictactoe_env_loader.cleanup()

        # Clean up Stockfish service
        if hasattr(self, "_stockfish_service") and self._stockfish_service is not None:
            self._stockfish_service.stop()
            self._stockfish_service = None

        # Shutdown session
        self._session.shutdown()

        if hasattr(self, "_time_refresh_timer") and self._time_refresh_timer.isActive():
            self._time_refresh_timer.stop()

        super().closeEvent(a0)


__all__ = ["MainWindow"]
