from __future__ import annotations

"""Main Qt window for the Gym GUI application."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, cast
import json
import threading
import grpc
import numpy as np

from PyQt6.QtCore import pyqtSlot, pyqtSignal  # type: ignore[attr-defined]
from qtpy import QtCore, QtGui, QtWidgets  # type: ignore[attr-defined]
try:
    from qtpy.QtGui import QAction
except ImportError:
    from qtpy.QtWidgets import QAction  # type: ignore[attr-defined]

from gym_gui.config import game_configs
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
    LOG_OPERATOR_RESET_ALL_STARTED,
    LOG_OPERATOR_STEP_ALL_COMPLETED,
    LOG_OPERATOR_STOP_ALL_COMPLETED,
    LOG_OPERATOR_ENV_PREVIEW_STARTED,
    LOG_OPERATOR_ENV_PREVIEW_SUCCESS,
    LOG_OPERATOR_ENV_PREVIEW_IMPORT_ERROR,
    LOG_OPERATOR_ENV_PREVIEW_ERROR,
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
from gym_gui.services.operator import OperatorConfig, OperatorDescriptor, MultiOperatorService
from gym_gui.services.operator_launcher import OperatorLauncher, OperatorLaunchError
from gym_gui.services.service_locator import get_service_locator
from gym_gui.services.telemetry import TelemetryService
from gym_gui.services.trainer import (
    TrainerClient,
    TrainerClientRunner,
    TrainingRunManager,
    RunRegistry,
)
from gym_gui.services.trainer.streams import TelemetryAsyncHub
from gym_gui.ui.widgets.live_telemetry_tab import LiveTelemetryTab
from gym_gui.ui.widgets.fastlane_tab import FastLaneTab
from gym_gui.ui.widgets.ray_multi_worker_fastlane_tab import RayMultiWorkerFastLaneTab
from gym_gui.ui.presenters.workers import (
    get_worker_presenter_registry,
)
from gym_gui.services.llm import LLM_CHAT_AVAILABLE
from gym_gui.ui.themes import apply_theme, DARK_THEME, LIGHT_THEME

if LLM_CHAT_AVAILABLE:
    from gym_gui.ui.widgets.chat_panel import ChatPanel
else:
    ChatPanel = None  # type: ignore[misc, assignment]
from gym_gui.ui.widgets.settings import SettingsDialog
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
    SudokuHandler,
    CheckersHandler,
    HumanVsAgentHandler,
    CheckersEnvLoader,
    ChessEnvLoader,
    ConnectFourEnvLoader,
    GoEnvLoader,
    TicTacToeEnvLoader,
    VizdoomEnvLoader,
    # New composed handlers for extracted functionality
    MultiAgentGameHandler,
    TrainingFormHandler,
    PolicyEvaluationHandler,
    FastLaneTabHandler,
    TrainingMonitorHandler,
)

from gym_gui.constants.optional_deps import (
    get_mjpc_launcher,
    get_godot_launcher,
    OptionalDependencyError,
)

from gym_gui.services.trainer.signals import get_trainer_signals

TRAINER_SUBMIT_DEADLINE_MULTIPLIER = 6


def _training_submit_deadline_seconds() -> float:
    """Return the gRPC deadline used for SubmitRun requests."""

    return TRAINER_DEFAULTS.client.deadline_s * TRAINER_SUBMIT_DEADLINE_MULTIPLIER


_LOGGER = logging.getLogger(__name__)
# Dedicated operator logger for PettingZoo/multi-agent game operations → operators.log
_OP_LOGGER = logging.getLogger("gym_gui.operators.main_window")


class MainWindow(QtWidgets.QMainWindow, LogConstantMixin):
    """Primary window that orchestrates the Gym session."""

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
        self._session.set_input_controller(self._human_input)
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

        # Create TrainingRunManager for the Management tab
        run_registry = locator.resolve(RunRegistry)
        client_runner = locator.resolve(TrainerClientRunner)
        if run_registry is not None:
            self._run_manager: TrainingRunManager | None = TrainingRunManager(
                registry=run_registry,
                client_runner=client_runner,
                telemetry_service=telemetry_service,
            )
        else:
            _LOGGER.warning("RunRegistry not available; Management tab will be disabled")
            self._run_manager = None

        # Multi-operator service for parallel operator execution
        self._multi_operator_service = MultiOperatorService()
        self._operator_launcher = OperatorLauncher()

        # Shared PettingZoo environment for multi-agent games (LLM vs LLM)
        # When env_name == "pettingzoo", the GUI owns ONE shared environment
        # and coordinates turn-based action selection from multiple workers
        self._shared_pettingzoo_env: Any = None
        self._pettingzoo_multiagent_mode: bool = False
        self._pettingzoo_player_handles: Dict[str, Any] = {}  # player_id -> handle
        self._pettingzoo_current_seed: int = 42

        # Track dynamic agent tabs by (run_id, agent_id)
        self._agent_tab_index: set[tuple[str, str]] = set()
        self._selected_policy_path: Optional[Path] = None
        self._run_metadata: Dict[tuple[str, str], Dict[str, Any]] = {}
        # Note: FastLane tab tracking moved to FastLaneTabHandler
        # Note: Run watch/poll state moved to TrainingMonitorHandler

        # Settings dialog (created on demand, lazy initialization)
        self._settings_dialog: Optional[SettingsDialog] = None

        # Stockfish service (may be set by handlers, cleaned up on close)
        self._stockfish_service: Any = None

        # MuJoCo MPC launcher (optional)
        try:
            self._mjpc_launcher = get_mjpc_launcher()
        except OptionalDependencyError as e:
            _LOGGER.warning(f"MuJoCo MPC launcher not available: {e}")
            self._mjpc_launcher = None

        # Godot game engine launcher (optional)
        try:
            self._godot_launcher = get_godot_launcher()
        except OptionalDependencyError as e:
            _LOGGER.warning(f"Godot launcher not available: {e}")
            self._godot_launcher = None

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
        default_operator_id = self._actor_service.get_active_actor_id()

        # Convert ActorDescriptor to OperatorDescriptor for the UI migration
        operator_descriptors = tuple(
            OperatorDescriptor(
                operator_id=ad.actor_id,
                display_name=ad.display_name,
                description=ad.description,
                category="default",
            )
            for ad in actor_descriptors
        )

        control_config = ControlPanelConfig(
            available_modes=available_modes,
            default_mode=settings.default_control_mode,
            frozen_lake_config=game_configs.FrozenLakeConfig(is_slippery=False),
            taxi_config=game_configs.TaxiConfig(is_raining=False, fickle_passenger=False),
            cliff_walking_config=game_configs.CliffWalkingConfig(is_slippery=False),
            lunar_lander_config=game_configs.LunarLanderConfig(),
            car_racing_config=game_configs.CarRacingConfig.from_env(),
            bipedal_walker_config=game_configs.BipedalWalkerConfig.from_env(),
            minigrid_empty_config=game_configs.DEFAULT_MINIGRID_EMPTY_5x5_CONFIG,
            minigrid_doorkey_5x5_config=game_configs.DEFAULT_MINIGRID_DOORKEY_5x5_CONFIG,
            minigrid_doorkey_6x6_config=game_configs.DEFAULT_MINIGRID_DOORKEY_6x6_CONFIG,
            minigrid_doorkey_8x8_config=game_configs.DEFAULT_MINIGRID_DOORKEY_8x8_CONFIG,
            minigrid_doorkey_16x16_config=game_configs.DEFAULT_MINIGRID_DOORKEY_16x16_CONFIG,
            minigrid_lavagap_config=game_configs.DEFAULT_MINIGRID_LAVAGAP_S7_CONFIG,
            minigrid_redbluedoors_6x6_config=game_configs.DEFAULT_MINIGRID_REDBLUE_DOORS_6x6_CONFIG,
            minigrid_redbluedoors_8x8_config=game_configs.DEFAULT_MINIGRID_REDBLUE_DOORS_8x8_CONFIG,
            default_seed=settings.default_seed,
            allow_seed_reuse=settings.allow_seed_reuse,
            operators=operator_descriptors,
            default_operator_id=default_operator_id,
        )

        self._control_panel = ControlPanelWidget(config=control_config, parent=self)
        if default_operator_id is not None:
            self._control_panel.set_active_operator(default_operator_id)
        
        # Create presenter to coordinate
        self._presenter = MainWindowPresenter(self._session, self._human_input, parent=self)

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
        
        # Poll for new training runs and auto-subscribe (delegated to handler)
        self._run_poll_timer = QtCore.QTimer(self)
        self._run_poll_timer.setInterval(2000)  # Poll every 2 seconds
        self._run_poll_timer.timeout.connect(self._training_monitor_handler.poll_for_new_runs)
        self._run_poll_timer.start()

        self._training_monitor_handler.start_run_watch()

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
        self.setWindowTitle("MOSAIC - Qt Shell")
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
            run_manager=self._run_manager,
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

        # Chat panel (MOSAIC Assistant) - optional, requires [chat] extra
        self._chat_group: QtWidgets.QGroupBox | None = None
        self._chat_panel = None
        if LLM_CHAT_AVAILABLE and ChatPanel is not None:
            self._chat_group = QtWidgets.QGroupBox("Chat", self)
            chat_layout = QtWidgets.QVBoxLayout(self._chat_group)
            chat_layout.setContentsMargins(0, 0, 0, 0)
            self._chat_panel = ChatPanel(parent=self._chat_group)
            chat_layout.addWidget(self._chat_panel)

        info_log_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, central)
        info_log_splitter.setChildrenCollapsible(True)
        self._info_group.setMinimumWidth(layout_defaults.info_min_width)
        info_log_splitter.addWidget(self._info_group)
        info_log_splitter.addWidget(self._log_group)
        if self._chat_group is not None:
            info_log_splitter.addWidget(self._chat_group)

        # Adjust stretch factors based on whether chat panel is present
        if self._chat_group is not None:
            info_log_splitter.setStretchFactor(0, 2)  # Game Info
            info_log_splitter.setStretchFactor(1, 1)  # Runtime Log
            info_log_splitter.setStretchFactor(2, 2)  # Chat
            info_log_splitter.setSizes(
                [
                    layout_defaults.info_default_width // 2,
                    layout_defaults.log_default_width,
                    layout_defaults.info_default_width // 2,
                ]
            )
        else:
            info_log_splitter.setStretchFactor(0, 2)  # Game Info
            info_log_splitter.setStretchFactor(1, 1)  # Runtime Log
            info_log_splitter.setSizes(
                [
                    layout_defaults.info_default_width,
                    layout_defaults.log_default_width,
                ]
            )

        info_log_splitter.setMinimumWidth(layout_defaults.info_min_width + layout_defaults.log_min_width)
        info_log_splitter.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
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

        # Add Settings action as first item
        self._settings_action = QAction("Settings...", self)
        self._settings_action.triggered.connect(self._on_settings_clicked)
        self._view_toolbar.addAction(self._settings_action)
        self._view_toolbar.addSeparator()

        # Add panel view toggles
        self._add_view_toggle("Control Panel", self._control_panel_scroll)
        self._add_view_toggle("Render View", self._render_group)
        self._add_view_toggle("Game Info", self._info_group)
        self._add_view_toggle("Runtime Log", self._log_group)
        if self._chat_group is not None:
            self._add_view_toggle("Chat", self._chat_group)

        # Add spacer to push theme toggle to the right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._view_toolbar.addWidget(spacer)

        # Add theme toggle
        self._dark_mode = False
        self._theme_action = QAction("Dark Mode", self)
        self._theme_action.setCheckable(True)
        self._theme_action.setChecked(False)
        self._theme_action.toggled.connect(self._on_theme_toggled)
        self._view_toolbar.addAction(self._theme_action)

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

    def _on_theme_toggled(self, checked: bool) -> None:
        """Toggle between light and dark themes."""
        self._dark_mode = checked
        self._theme_action.setText("Light Mode" if checked else "Dark Mode")

        if checked:
            self._apply_dark_theme()
        else:
            self._apply_light_theme()

        self._status_bar.showMessage(
            f"{'Dark' if checked else 'Light'} theme applied", 2000
        )

    def _apply_dark_theme(self) -> None:
        """Apply dark theme from external QSS file."""
        apply_theme(DARK_THEME)

    def _apply_light_theme(self) -> None:
        """Apply light theme (reset to system default)."""
        apply_theme(LIGHT_THEME)

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
        self._sudoku_handler = SudokuHandler(
            session=self._session,
            render_tabs=self._render_tabs,
            status_bar=self._status_bar,
        )
        self._checkers_handler = CheckersHandler(
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
        self._checkers_env_loader = CheckersEnvLoader(
            render_tabs=self._render_tabs,
            control_panel=self._control_panel,
            status_bar=self._status_bar,
        )
        self._vizdoom_env_loader = VizdoomEnvLoader(
            render_tabs=self._render_tabs,
        )

        # Multi-agent game routing handler
        self._multi_agent_game_handler = MultiAgentGameHandler(
            status_bar=self._status_bar,
            chess_loader=self._chess_env_loader,
            connect_four_loader=self._connect_four_env_loader,
            go_loader=self._go_env_loader,
            tictactoe_loader=self._tictactoe_env_loader,
            checkers_loader=self._checkers_env_loader,
            set_game_info=self._set_game_info,
            get_game_info=get_game_info,
            parent=self,
        )

        # Training form handler (Train/Policy/Resume dialogs)
        self._training_form_handler = TrainingFormHandler(
            parent=self,
            get_form_factory=get_worker_form_factory,
            get_current_game=self._control_panel.current_game,
            get_cleanrl_env_id=self._control_panel.cleanrl_environment_id,
            submit_config=self._submit_training_config,
            build_policy_config=self._build_policy_evaluation_config,
            log_callback=lambda message=None, extra=None, exc_info=None: self.log_constant(
                LOG_UI_MAINWINDOW_INFO, message=message, extra=extra, exc_info=exc_info
            ),
            status_callback=self._status_bar.showMessage,
        )

        # FastLane tab handler (must be initialized before PolicyEvaluationHandler)
        self._fastlane_tab_handler = FastLaneTabHandler(
            render_tabs=self._render_tabs,
            log_callback=lambda message=None, extra=None, exc_info=None: self.log_constant(
                LOG_UI_WORKER_TABS_INFO, message=message, extra=extra, exc_info=exc_info
            ),
        )

        # Policy evaluation handler
        self._policy_evaluation_handler = PolicyEvaluationHandler(
            parent=self,
            status_bar=self._status_bar,
            open_ray_fastlane_tabs=self._fastlane_tab_handler.open_ray_fastlane_tabs,
        )

        # Training monitor handler
        # Note: fastlane_callback delegates to FastLaneTabHandler with metadata resolution
        self._training_monitor_handler = TrainingMonitorHandler(
            parent=self,
            live_controller=self._live_controller,
            analytics_tabs=self._analytics_tabs,
            render_tabs=self._render_tabs,
            run_metadata=self._run_metadata,
            trainer_dir=VAR_TRAINER_DIR,
            log_callback=lambda message=None, extra=None, exc_info=None: self.log_constant(
                LOG_UI_MAINWINDOW_INFO, message=message, extra=extra, exc_info=exc_info
            ),
            status_callback=self._status_bar.showMessage,
            title_callback=self._render_group.setTitle,
            fastlane_callback=lambda run_id, agent_id: self._fastlane_tab_handler.maybe_open_fastlane_tab(
                run_id, agent_id, self._resolve_run_metadata(run_id, agent_id)
            ),
        )

    def _connect_signals(self) -> None:
        # Connect control panel signals to session controller
        self._control_panel.load_requested.connect(self._on_load_requested)
        self._control_panel.reset_requested.connect(self._on_reset_requested)
        # Training form signals (delegated to handler)
        self._control_panel.train_agent_requested.connect(
            self._training_form_handler.on_train_agent_requested
        )
        self._control_panel.trained_agent_requested.connect(
            self._training_form_handler.on_trained_agent_requested
        )
        self._control_panel.resume_training_requested.connect(
            self._training_form_handler.on_resume_training_requested
        )
        self._control_panel.start_game_requested.connect(self._on_start_game)
        self._control_panel.pause_game_requested.connect(self._on_pause_game)
        self._control_panel.continue_game_requested.connect(self._on_continue_game)
        self._control_panel.terminate_game_requested.connect(self._on_terminate_game)
        self._control_panel.agent_step_requested.connect(self._session.perform_agent_step)
        self._control_panel._fastlane_only_checkbox.toggled.connect(self._on_fastlane_only_toggled)
        self._control_panel.game_changed.connect(self._on_game_changed)
        self._control_panel.control_mode_changed.connect(self._on_mode_changed)
        self._control_panel.operator_changed.connect(self._on_operator_changed)
        # Multi-operator signals - scientific execution for fair comparison
        self._control_panel.operators_changed.connect(self._on_operators_changed)
        self._control_panel.step_all_requested.connect(self._on_step_all_operators)
        self._control_panel.step_player_requested.connect(self._on_step_player)
        self._control_panel.reset_all_requested.connect(self._on_reset_all_operators)
        self._control_panel.stop_operators_requested.connect(self._on_stop_operators)
        self._control_panel.initialize_operator_requested.connect(self._on_initialize_operator)
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
        # Multi-agent game signals (delegated to handler)
        self._control_panel.multi_agent_load_requested.connect(
            self._multi_agent_game_handler.on_load_requested
        )
        self._control_panel.multi_agent_start_requested.connect(
            self._multi_agent_game_handler.on_start_requested
        )
        self._control_panel.multi_agent_reset_requested.connect(
            self._multi_agent_game_handler.on_reset_requested
        )
        self._control_panel.multi_agent_tab.ai_opponent_changed.connect(
            self._multi_agent_game_handler.on_ai_opponent_changed
        )
        # Policy evaluation (delegated to handler)
        self._control_panel.policy_evaluate_requested.connect(
            self._policy_evaluation_handler.handle_evaluate_request
        )

        # Board game handlers (Human Control Mode)
        # These signals come from BoardGameRendererStrategy in the Grid tab
        self._render_tabs.chess_move_made.connect(self._chess_handler.on_chess_move)
        self._render_tabs.connect_four_column_clicked.connect(self._connect_four_handler.on_column_clicked)
        self._render_tabs.go_intersection_clicked.connect(self._go_handler.on_intersection_clicked)
        self._render_tabs.go_pass_requested.connect(self._go_handler.on_pass_requested)
        # Sudoku handlers (Jumanji environment with mouse selection + keyboard digit entry)
        self._render_tabs.sudoku_cell_selected.connect(self._sudoku_handler.on_cell_selected)
        self._render_tabs.sudoku_digit_entered.connect(self._sudoku_handler.on_digit_entered)
        self._render_tabs.sudoku_cell_cleared.connect(self._sudoku_handler.on_cell_cleared)
        # Checkers handler (OpenSpiel via Shimmy - two-click selection)
        self._render_tabs.checkers_cell_clicked.connect(self._checkers_handler.on_checkers_cell_clicked)

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

    def _on_operator_changed(self, operator_id: str) -> None:
        """Handle active operator selection from the control panel."""
        try:
            self._actor_service.set_active_actor(operator_id)
        except KeyError:
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Attempted to activate unknown operator '{operator_id}'",
                extra={"operator_id": operator_id},
            )
            self._status_bar.showMessage(f"Unknown operator '{operator_id}'", 5000)
            return

        descriptor = self._actor_service.get_actor_descriptor(operator_id)
        label = descriptor.display_name if descriptor is not None else operator_id
        self._status_bar.showMessage(f"Active operator set to {label}", 4000)

    # ------------------------------------------------------------------
    # Multi-Operator Signal Handlers (Phase 6)
    # ------------------------------------------------------------------

    def _on_operators_changed(self, configs: list) -> None:
        """Handle operator configuration changes from the control panel.

        Syncs render view containers with operator configurations:
        - Removes containers for deleted operators
        - Adds containers for new operators
        - Updates existing operator configurations

        Args:
            configs: List of OperatorConfig instances representing current operator set.
        """
        current_ids = set(self._multi_operator_service.get_active_operators().keys())
        new_ids = {c.operator_id for c in configs}

        # Remove deleted operators
        for operator_id in current_ids - new_ids:
            self._render_tabs.remove_operator_view(operator_id)
            self._multi_operator_service.remove_operator(operator_id)
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message=f"Removed operator from multi-operator view: {operator_id}",
            )

        # Add or update operators
        for config in configs:
            if config.operator_id not in current_ids:
                # New operator
                self._render_tabs.add_operator_view(config)
                self._multi_operator_service.add_operator(config)
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message=f"Added operator to multi-operator view: {config.operator_id}",
                    extra={"operator_type": config.operator_type, "worker_id": config.worker_id},
                )
            else:
                # Existing operator - update config if needed
                self._multi_operator_service.add_operator(config)  # Will update if exists

        count = len(configs)
        self._status_bar.showMessage(
            f"Multi-operator configuration updated: {count} operator{'s' if count != 1 else ''}",
            3000
        )

    def _is_pettingzoo_multiagent(self) -> tuple[bool, Optional["OperatorConfig"]]:
        """Check if we're in PettingZoo multi-agent mode.

        Returns:
            Tuple of (is_multiagent, first_config) where first_config is used
            to get env_name and task for creating the shared environment.
        """
        active_operators = self._multi_operator_service.get_active_operators()
        if not active_operators:
            return False, None

        # Get first operator config to check env_name
        first_id = next(iter(active_operators.keys()))
        first_config = self._multi_operator_service.get_operator(first_id)
        if first_config is None:
            return False, None

        # PettingZoo multi-agent: env_name == "pettingzoo" and multiple workers per operator
        # or multiple operators assigned to different players
        if first_config.env_name == "pettingzoo":
            # Check if we have multiple workers (player assignments)
            if len(first_config.workers) > 1:
                return True, first_config
            # Or check if multiple operators exist (each controlling one player)
            if len(active_operators) > 1:
                return True, first_config

        return False, None

    def _create_pettingzoo_env(self, task: str, seed: int) -> Any:
        """Create a PettingZoo environment for multi-agent games.

        Args:
            task: The specific game (e.g., "chess_v6", "connect_four_v3").
            seed: Random seed for initialization.

        Returns:
            The PettingZoo AEC environment.
        """
        from pettingzoo.classic import (
            chess_v6,
            connect_four_v3,
            go_v5,
            tictactoe_v3,
        )

        env_factories = {
            "chess_v6": chess_v6.env,
            "connect_four_v3": connect_four_v3.env,
            "go_v5": go_v5.env,
            "tictactoe_v3": tictactoe_v3.env,
        }

        if task not in env_factories:
            raise ValueError(f"Unknown PettingZoo task: {task}")

        env = env_factories[task](render_mode="rgb_array")
        env.reset(seed=seed)
        return env

    def _get_chess_legal_moves(self, env: Any) -> list[str]:
        """Get legal moves for the current player in chess.

        Returns:
            List of UCI move strings (e.g., ["e2e4", "g1f3", ...]).
        """
        try:
            # PettingZoo chess uses python-chess internally
            board = env.board
            return [move.uci() for move in board.legal_moves]
        except Exception as e:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Failed to get legal moves: {e}",
            )
            return []

    def _convert_uci_to_action_index(self, env: Any, uci_move: str) -> Optional[int]:
        """Convert UCI move string to PettingZoo action index.

        Uses PettingZoo's chess_utils module for correct AlphaZero-style action encoding.

        Args:
            env: The PettingZoo chess environment.
            uci_move: UCI move string (e.g., "e2e4").

        Returns:
            Action index for env.step(), or None if invalid.
        """
        try:
            import chess
            from pettingzoo.classic.chess import chess_utils

            board = env.board
            move = chess.Move.from_uci(uci_move)

            if move not in board.legal_moves:
                _OP_LOGGER.debug("_convert_uci_to_action_index: %s not in legal_moves", uci_move)
                return None

            # Determine current player (0 = white, 1 = black)
            current_agent = env.agent_selection
            current_player = 0 if current_agent == "player_0" else 1

            # For black, we need to mirror the move since PettingZoo encodes from white's perspective
            if current_player == 1:
                # Mirror the move for black's encoding
                move_for_encoding = chess_utils.mirror_move(move)
            else:
                move_for_encoding = move

            # Get the UCI string for the (possibly mirrored) move
            move_uci = move_for_encoding.uci()

            # Use PettingZoo's encoding: action = (col * 8 + row) * 73 + plane
            source = move_for_encoding.from_square
            coord = chess_utils.square_to_coord(source)
            panel = chess_utils.get_move_plane(move_for_encoding)
            action = (coord[0] * 8 + coord[1]) * 73 + panel

            _OP_LOGGER.debug(
                "_convert_uci_to_action_index: %s -> action=%s, current_player=%s, coord=%s, panel=%s",
                uci_move, action, current_player, coord, panel,
            )

            # Verify this action is legal
            obs = env.observe(current_agent)
            if isinstance(obs, dict) and "action_mask" in obs:
                if obs["action_mask"][action] == 1:
                    return action
                else:
                    _OP_LOGGER.debug("_convert_uci_to_action_index: action %s not in action_mask", action)
                    return None
            else:
                # No action mask, return the computed action
                return action

        except Exception as e:
            _OP_LOGGER.debug("_convert_uci_to_action_index EXCEPTION: %s", e, exc_info=True)
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Failed to convert UCI move '{uci_move}': {e}",
            )
            return None

    def _on_reset_all_operators(self, seed: int) -> None:
        """Reset all configured operators with shared seed.

        Scientific Execution Model (inspired by BALROG):
        - All environments reset with identical seed for fair comparison
        - Launches worker subprocesses in interactive mode if not already running
        - Sends reset command with seed to each subprocess
        - Switches to Multi-Operator tab for viewing

        For PettingZoo multi-agent games (chess, etc.):
        - GUI creates ONE shared environment
        - Workers are initialized in action_selector mode
        - Turn-based coordination handled by _on_step_all_operators
        """
        active_operators = self._multi_operator_service.get_active_operators()
        if not active_operators:
            self._status_bar.showMessage("No operators configured to reset", 3000)
            return

        # Check if this is PettingZoo multi-agent mode
        is_multiagent, first_config = self._is_pettingzoo_multiagent()
        _OP_LOGGER.debug(
            "_on_reset_all_operators: is_multiagent=%s, first_config=%s",
            is_multiagent, first_config,
        )
        if first_config:
            _OP_LOGGER.debug(
                "_on_reset_all_operators: env_name=%s, workers=%s",
                first_config.env_name, list(first_config.workers.keys()),
            )

        if is_multiagent and first_config is not None:
            # PettingZoo multi-agent: GUI owns the shared environment
            _OP_LOGGER.debug("_on_reset_all_operators: Taking PettingZoo path")
            self._on_reset_pettingzoo_multiagent(seed, first_config)
            return

        # Standard single-agent flow: each worker owns its own environment
        # Get operators that are pending (not yet started)
        pending_ids = self._multi_operator_service.start_all()

        # Launch subprocess workers for each operator that needs to be started
        started_ids = []
        failed_ids = []
        for operator_id in pending_ids:
            config = self._multi_operator_service.get_operator(operator_id)
            if config is None:
                continue

            try:
                # Launch the subprocess in interactive mode for step-by-step control
                handle = self._operator_launcher.launch_operator(
                    config,
                    interactive=True,  # Enable step-by-step control
                )

                # Send reset command with seed
                handle.send_reset(seed)

                # Assign run_id to the service for telemetry routing
                self._multi_operator_service.assign_run_id(operator_id, handle.run_id)
                self._multi_operator_service.set_operator_state(operator_id, "running")

                started_ids.append(operator_id)

                self.log_constant(
                    LOG_UI_MAINWINDOW_INFO,
                    message=f"Launched interactive operator subprocess with seed",
                    extra={
                        "operator_id": operator_id,
                        "seed": seed,
                        "run_id": handle.run_id,
                        "pid": handle.pid,
                        "log_path": str(handle.log_path),
                        "interactive": True,
                    },
                )
            except OperatorLaunchError as e:
                self._multi_operator_service.set_operator_state(operator_id, "error")
                failed_ids.append(operator_id)
                self.log_constant(
                    LOG_UI_MAINWINDOW_ERROR,
                    message=f"Failed to launch operator: {e}",
                    extra={"operator_id": operator_id, "seed": seed},
                )

        # Update status indicators and set container display sizes
        for operator_id in started_ids:
            self._render_tabs.set_operator_status(operator_id, "running")
            # Ensure container display size is set from config
            op_config = self._multi_operator_service.get_operator(operator_id)
            if op_config:
                container_size = op_config.settings.get("container_size", 0)
                if container_size and container_size > 0:
                    self._render_tabs.set_operator_display_size(
                        operator_id, container_size, container_size
                    )
        for operator_id in failed_ids:
            self._render_tabs.set_operator_status(operator_id, "error")

        # Switch to Multi-Operator tab
        self._render_tabs.switch_to_multi_operator_tab()

        count = len(started_ids)
        if failed_ids:
            self._status_bar.showMessage(
                f"Reset {count} operator{'s' if count != 1 else ''} (seed={seed}), {len(failed_ids)} failed",
                5000
            )
        else:
            self._status_bar.showMessage(
                f"Reset all operators with seed={seed}",
                3000
            )
        self.log_constant(
            LOG_OPERATOR_RESET_ALL_STARTED,
            message=f"Reset {count} operators with shared seed",
            extra={"operator_ids": started_ids, "failed_ids": failed_ids, "seed": seed},
        )

    def _on_reset_pettingzoo_multiagent(self, seed: int, config: "OperatorConfig") -> None:
        """Reset for PettingZoo multi-agent mode.

        In this mode:
        1. GUI creates ONE shared PettingZoo environment
        2. Each worker is initialized in action_selector mode (no env ownership)
        3. Workers provide actions when asked, GUI executes them

        Args:
            seed: Random seed for environment.
            config: The operator config with task and worker assignments.
        """
        task = config.task  # e.g., "chess_v6"

        # Close existing shared environment if any
        if self._shared_pettingzoo_env is not None:
            try:
                self._shared_pettingzoo_env.close()
            except Exception:
                pass

        # Create the shared environment in the GUI
        try:
            self._shared_pettingzoo_env = self._create_pettingzoo_env(task, seed)
            self._pettingzoo_multiagent_mode = True
            self._pettingzoo_current_seed = seed
            self._pettingzoo_player_handles.clear()

            self.log_constant(
                LOG_UI_MAINWINDOW_INFO,
                message=f"Created shared PettingZoo environment: {task}",
                extra={"task": task, "seed": seed},
            )
        except Exception as e:
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Failed to create PettingZoo environment: {e}",
                extra={"task": task, "seed": seed},
            )
            self._status_bar.showMessage(f"Failed to create {task}: {e}", 5000)
            return

        # Launch workers in action_selector mode
        # Each worker controls one player (e.g., player_0 = White, player_1 = Black)
        started_ids = []
        failed_ids = []

        # Get all operators and their player assignments
        active_operators = self._multi_operator_service.get_active_operators()
        _OP_LOGGER.debug("active_operators count=%d", len(active_operators))
        pending_ids = self._multi_operator_service.start_all()
        _OP_LOGGER.debug("pending_ids=%s", pending_ids)

        for operator_id in pending_ids:
            _OP_LOGGER.debug("Processing operator_id=%s", operator_id)
            op_config = self._multi_operator_service.get_operator(operator_id)
            if op_config is None:
                _OP_LOGGER.debug("op_config is None for %s", operator_id)
                continue

            _OP_LOGGER.debug("op_config.workers=%s", list(op_config.workers.keys()))
            # Determine which player this operator controls
            # If multiple workers in one operator, each controls a player
            # If single worker per operator, operator controls one player
            for player_id, worker_assignment in op_config.workers.items():
                _OP_LOGGER.debug(
                    "Launching worker for player_id=%s, worker_id=%s",
                    player_id, worker_assignment.worker_id,
                )
                try:
                    # Create single-agent config for this player's worker
                    # (multiagent OperatorConfig has operator_type="multiagent" which
                    #  the launcher doesn't handle directly)
                    player_config = OperatorConfig.single_agent(
                        operator_id=f"{operator_id}_{player_id}",
                        display_name=f"{op_config.display_name} - {player_id}",
                        worker_id=worker_assignment.worker_id,
                        worker_type=worker_assignment.worker_type,
                        env_name="pettingzoo",  # Action-selector mode
                        task=task,
                        settings=worker_assignment.settings,
                    )

                    # Launch subprocess
                    handle = self._operator_launcher.launch_operator(
                        player_config,
                        interactive=True,
                    )

                    # Initialize in action_selector mode (not reset with env ownership)
                    handle.send_init_agent(
                        game_name=task,
                        player_id=player_id,
                    )

                    # Store mapping for step coordination
                    self._pettingzoo_player_handles[player_id] = handle
                    _OP_LOGGER.debug(
                        "Stored handle for %s, total handles=%d",
                        player_id, len(self._pettingzoo_player_handles),
                    )

                    # Assign run_id
                    self._multi_operator_service.assign_run_id(operator_id, handle.run_id)
                    self._multi_operator_service.set_operator_state(operator_id, "running")

                    started_ids.append(operator_id)

                    self.log_constant(
                        LOG_UI_MAINWINDOW_INFO,
                        message=f"Initialized worker in action_selector mode",
                        extra={
                            "operator_id": operator_id,
                            "player_id": player_id,
                            "game": task,
                            "run_id": handle.run_id,
                        },
                    )
                except OperatorLaunchError as e:
                    self._multi_operator_service.set_operator_state(operator_id, "error")
                    failed_ids.append(operator_id)
                    self.log_constant(
                        LOG_UI_MAINWINDOW_ERROR,
                        message=f"Failed to launch operator: {e}",
                        extra={"operator_id": operator_id, "player_id": player_id},
                    )

        # Update status indicators
        for operator_id in started_ids:
            self._render_tabs.set_operator_status(operator_id, "running")
        for operator_id in failed_ids:
            self._render_tabs.set_operator_status(operator_id, "error")

        # Set the container display size for ALL active operators
        # (In PettingZoo mode, all operators share the same environment rendering)
        # Use active_operators instead of started_ids because the render uses active_ops
        container_size = config.settings.get("container_size", 0)
        if container_size and container_size > 0:
            for op_id in active_operators.keys():
                self._render_tabs.set_operator_display_size(
                    op_id, container_size, container_size
                )

        # Render initial board state
        self._render_pettingzoo_frame()

        # Show turn indicator for first player
        current_player = self._shared_pettingzoo_env.agent_selection
        self._control_panel.set_turn_indicator(current_player, visible=True)

        # Enable PettingZoo mode: show player step buttons instead of Step All
        _OP_LOGGER.debug("Enabling PettingZoo mode, current_player=%s", current_player)
        self._control_panel.set_pettingzoo_mode(True)
        self._control_panel.set_current_player(current_player)
        _OP_LOGGER.debug("PettingZoo mode enabled")

        # Switch to Multi-Operator tab
        self._render_tabs.switch_to_multi_operator_tab()

        player_count = len(self._pettingzoo_player_handles)
        self._status_bar.showMessage(
            f"PettingZoo {task} ready: {player_count} players (seed={seed})",
            3000
        )
        self.log_constant(
            LOG_OPERATOR_RESET_ALL_STARTED,
            message=f"PettingZoo multi-agent reset complete",
            extra={
                "task": task,
                "seed": seed,
                "players": list(self._pettingzoo_player_handles.keys()),
            },
        )

    def _render_pettingzoo_frame(self) -> None:
        """Render the current state of the shared PettingZoo environment."""
        if self._shared_pettingzoo_env is None:
            _OP_LOGGER.debug("_render_pettingzoo_frame: No environment")
            return

        try:
            env = self._shared_pettingzoo_env
            active_ops = self._multi_operator_service.get_active_operators()
            if not active_ops:
                return

            first_id = next(iter(active_ops.keys()))
            first_config = active_ops[first_id]
            task = first_config.task

            # For chess, use board game renderer with FEN data
            if task == "chess_v6" and hasattr(env, "board"):
                import chess
                board: chess.Board = env.board
                _OP_LOGGER.debug("_render_pettingzoo_frame: Chess board FEN=%s", board.fen())

                legal_moves = [move.uci() for move in board.legal_moves]
                current_player = "white" if board.turn == chess.WHITE else "black"

                # Build chess-specific payload for BoardGameRendererStrategy
                # Note: "render_payload" key is required for _extract_render_payload
                payload = {
                    "step_index": 0,
                    "episode_index": 0,
                    "render_payload": {
                        "chess": {
                            "fen": board.fen(),
                            "legal_moves": legal_moves,
                            "current_player": current_player,
                            "is_check": board.is_check(),
                        },
                        "game_id": "chess",
                    },
                }
                _OP_LOGGER.debug(
                    "_render_pettingzoo_frame: Sending payload to operator_id=%s, keys=%s",
                    first_id, list(payload.keys()),
                )
                self._render_tabs.display_operator_payload(first_id, payload)
                _OP_LOGGER.debug("_render_pettingzoo_frame: Payload sent")
            else:
                # Fallback to RGB rendering for other games
                rgb_frame = env.render()
                if rgb_frame is not None and isinstance(rgb_frame, np.ndarray):
                    payload = {
                        "step_index": 0,
                        "episode_index": 0,
                        "render_payload": {
                            "mode": "rgb",
                            "rgb": rgb_frame.tolist(),
                            "width": rgb_frame.shape[1],
                            "height": rgb_frame.shape[0],
                        },
                    }
                    self._render_tabs.display_operator_payload(first_id, payload)
        except Exception as e:
            _OP_LOGGER.debug("_render_pettingzoo_frame EXCEPTION: %s", e, exc_info=True)
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Failed to render PettingZoo frame: {e}",
            )

    def _on_step_all_operators(self, seed: int) -> None:
        """Step all running operators by exactly one step.

        Scientific Execution Model (inspired by BALROG):
        - Lock-step execution: each operator's agent selects one action
        - This ensures scientifically fair side-by-side comparison
        - No arbitrary timing delays between operators

        For PettingZoo multi-agent games:
        - GUI owns the shared environment
        - Gets current player from env.agent_selection
        - Sends observation to that player's worker via select_action
        - Executes returned action on shared environment
        """
        active_operators = self._multi_operator_service.get_active_operators()
        if not active_operators:
            self._status_bar.showMessage("No operators to step", 3000)
            return

        # Check if we're in PettingZoo multi-agent mode
        if self._pettingzoo_multiagent_mode and self._shared_pettingzoo_env is not None:
            self._on_step_pettingzoo_multiagent()
            return

        # Standard single-agent flow: each worker owns its own environment
        # Send step command to each running operator subprocess
        stepped_count = 0
        stepped_handles = []  # Track handles that received step command
        for operator_id in active_operators:
            handle = self._operator_launcher.get_handle(operator_id)
            if handle is None:
                self.log_constant(
                    LOG_UI_MAINWINDOW_WARNING,
                    message=f"No process handle for operator",
                    extra={"operator_id": operator_id},
                )
                continue

            if not handle.is_running:
                self.log_constant(
                    LOG_UI_MAINWINDOW_WARNING,
                    message=f"Operator process not running",
                    extra={"operator_id": operator_id, "return_code": handle.return_code},
                )
                self._multi_operator_service.set_operator_state(operator_id, "stopped")
                continue

            # Send step command
            if handle.send_step():
                stepped_count += 1
                stepped_handles.append((operator_id, handle))
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message=f"Sent step command to operator",
                    extra={"operator_id": operator_id},
                )
            else:
                self.log_constant(
                    LOG_UI_MAINWINDOW_WARNING,
                    message=f"Failed to send step command to operator",
                    extra={"operator_id": operator_id},
                )

        # Read responses from operators and update render view
        # Use a short delay to allow LLM inference to complete
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, lambda: self._poll_operator_responses(stepped_handles))

        self.log_constant(
            LOG_OPERATOR_STEP_ALL_COMPLETED,
            message=f"Step all operators completed",
            extra={"stepped_count": stepped_count, "total_active": len(active_operators)},
        )
        self._status_bar.showMessage(
            f"Stepped {stepped_count} operator{'s' if stepped_count != 1 else ''}",
            2000
        )

    def _on_step_player(self, player_id: str, seed: int) -> None:
        """Handle step request for a specific player (PettingZoo mode).

        Called when user clicks one of the player-specific step buttons.

        Args:
            player_id: Which player to step (e.g., "player_0", "player_1").
            seed: Random seed (currently unused for PettingZoo steps).
        """
        _OP_LOGGER.debug("_on_step_player: player_id=%s, seed=%s", player_id, seed)
        env = self._shared_pettingzoo_env
        if env is None:
            _OP_LOGGER.debug("_on_step_player: No environment")
            self._status_bar.showMessage("No PettingZoo environment active", 3000)
            return

        # Validate it's actually this player's turn
        current_player = env.agent_selection
        _OP_LOGGER.debug("_on_step_player: current_player from env=%s", current_player)
        if current_player != player_id:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Attempted to step wrong player",
                extra={"requested": player_id, "current": current_player},
            )
            self._status_bar.showMessage(
                f"Not {player_id}'s turn! Current: {current_player}",
                3000
            )
            # Fix button states
            self._control_panel.set_current_player(current_player)
            return

        # Delegate to existing step logic
        self._on_step_pettingzoo_multiagent()

    def _on_step_pettingzoo_multiagent(self) -> None:
        """Step the shared PettingZoo environment with turn-based coordination.

        Flow:
        1. Get current player from env.agent_selection
        2. Get observation and legal moves for that player
        3. Send select_action to that player's worker
        4. Wait for action response
        5. Execute action on shared environment
        6. Render updated board
        """
        _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: Starting")
        env = self._shared_pettingzoo_env
        if env is None:
            _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: No environment")
            return

        # Check if game is over
        if env.terminations.get(env.agent_selection, False) or \
           env.truncations.get(env.agent_selection, False):
            _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: Game over")
            self._status_bar.showMessage("Game over! Use Reset to start new game.", 3000)
            return

        # Get current player
        current_player = env.agent_selection  # e.g., "player_0" or "player_1"
        _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: current_player=%s", current_player)

        # Get handle for this player's worker
        _OP_LOGGER.debug(
            "_on_step_pettingzoo_multiagent: Available handles=%s",
            list(self._pettingzoo_player_handles.keys()),
        )
        handle = self._pettingzoo_player_handles.get(current_player)
        if handle is None:
            _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: NO HANDLE for %s", current_player)
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"No worker handle for player: {current_player}",
                extra={"player": current_player, "available": list(self._pettingzoo_player_handles.keys())},
            )
            self._status_bar.showMessage(f"No worker for {current_player}", 3000)
            return

        _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: Got handle, is_running=%s", handle.is_running)
        if not handle.is_running:
            _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: Handle not running")
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Worker not running for player: {current_player}",
            )
            self._status_bar.showMessage(f"Worker stopped for {current_player}", 3000)
            return

        # Get observation for current player
        obs = env.observe(current_player)

        # Get legal moves (for chess, convert to UCI strings)
        legal_moves = self._get_chess_legal_moves(env)
        _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: legal_moves count=%d", len(legal_moves))

        # Build observation string for LLM
        obs_str = f"Current player: {current_player}\n"
        obs_str += f"Board state:\n{env.board}\n" if hasattr(env, "board") else str(obs)

        # Send select_action command to worker
        info = {"legal_moves": legal_moves}
        _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: Sending select_action to worker...")
        if handle.send_select_action(obs_str, current_player, info):
            _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: select_action sent successfully")
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message=f"Sent select_action to {current_player}",
                extra={"player": current_player, "legal_moves_count": len(legal_moves)},
            )

            # Poll for response with timeout
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, lambda: self._poll_pettingzoo_action(handle, current_player))
        else:
            _OP_LOGGER.debug("_on_step_pettingzoo_multiagent: Failed to send select_action")
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Failed to send select_action to {current_player}",
            )

    def _poll_pettingzoo_action(
        self,
        handle: Any,
        player_id: str,
        attempts: int = 0,
        max_attempts: int = 300,  # 30 seconds at 100ms intervals
    ) -> None:
        """Poll for action response from worker and execute on shared env.

        Args:
            handle: The worker process handle.
            player_id: Which player we're waiting for.
            attempts: Current attempt number.
            max_attempts: Maximum polling attempts before timeout.
        """
        from PyQt6.QtCore import QTimer

        if attempts >= max_attempts:
            _OP_LOGGER.debug("_poll_pettingzoo_action: TIMEOUT for %s after %d attempts", player_id, max_attempts)
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Timeout waiting for action from {player_id}",
            )
            self._status_bar.showMessage(f"Timeout: {player_id} didn't respond", 5000)
            # Re-enable the current player's button after timeout
            self._control_panel.set_current_player(player_id)
            return

        # Try to read response
        response = handle.try_read_response(timeout=0.1)

        if response is None:
            # No response yet, poll again
            QTimer.singleShot(100, lambda: self._poll_pettingzoo_action(
                handle, player_id, attempts + 1, max_attempts
            ))
            return

        response_type = response.get("type", "")

        if response_type == "action_selected":
            # Got the action!
            action_str = response.get("action_str", "")
            action_index = response.get("action")
            _OP_LOGGER.debug(
                "_poll_pettingzoo_action: Got action_selected from %s: %s (index=%s)",
                player_id, action_str, action_index,
            )

            self.log_constant(
                LOG_UI_MAINWINDOW_INFO,
                message=f"Received action from {player_id}: {action_str}",
                extra={"player": player_id, "action": action_str, "index": action_index},
            )

            # Execute action on shared environment
            self._execute_pettingzoo_action(player_id, action_str, action_index)

        elif response_type == "agent_ready":
            # Worker just initialized, poll again for the actual action
            QTimer.singleShot(100, lambda: self._poll_pettingzoo_action(
                handle, player_id, attempts + 1, max_attempts
            ))

        elif response_type == "error":
            error_msg = response.get("message", "Unknown error")
            _OP_LOGGER.debug("_poll_pettingzoo_action: Got error from %s: %s", player_id, error_msg)
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Worker error for {player_id}: {error_msg}",
            )
            self._status_bar.showMessage(f"Error: {error_msg}", 5000)
            # Re-enable the current player's button after error
            self._control_panel.set_current_player(player_id)

        else:
            # Unknown response, log and poll again
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Unexpected response type from {player_id}: {response_type}",
            )
            QTimer.singleShot(100, lambda: self._poll_pettingzoo_action(
                handle, player_id, attempts + 1, max_attempts
            ))

    def _execute_pettingzoo_action(
        self,
        player_id: str,
        action_str: str,
        action_index: Optional[int] = None,
    ) -> None:
        """Execute an action on the shared PettingZoo environment.

        Args:
            player_id: Which player made the move.
            action_str: The action as a string (e.g., UCI move "e2e4").
            action_index: Optional pre-computed action index.
        """
        _OP_LOGGER.debug(
            "_execute_pettingzoo_action: player_id=%s, action_str=%s, action_index=%s",
            player_id, action_str, action_index,
        )
        env = self._shared_pettingzoo_env
        if env is None:
            _OP_LOGGER.debug("_execute_pettingzoo_action: No environment")
            return

        # Convert action string to index if needed
        if action_index is None or not isinstance(action_index, int):
            _OP_LOGGER.debug("action_index is not int (%s), converting UCI to index", type(action_index))
            action_index = self._convert_uci_to_action_index(env, action_str)
            _OP_LOGGER.debug("Converted action_index = %s", action_index)

        if action_index is None:
            # Invalid move - try to pick a random legal move using action mask
            _OP_LOGGER.debug("Invalid move '%s', picking random legal move from action_mask", action_str)
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Invalid move '{action_str}' from {player_id}, selecting random",
            )
            # Get legal actions from the environment's action mask
            try:
                # PettingZoo AEC environments provide action_mask
                obs = env.observe(player_id)
                if isinstance(obs, dict) and "action_mask" in obs:
                    action_mask = obs["action_mask"]
                else:
                    # Try getting mask directly
                    action_mask = env.action_mask(player_id) if hasattr(env, "action_mask") else None

                if action_mask is not None:
                    legal_action_indices = np.where(action_mask == 1)[0]
                    _OP_LOGGER.debug("legal_action_indices count = %d", len(legal_action_indices))
                    if len(legal_action_indices) > 0:
                        import random
                        action_index = int(random.choice(legal_action_indices))
                        _OP_LOGGER.debug("Random legal action_index = %s", action_index)
                    else:
                        _OP_LOGGER.debug("No legal actions in action_mask")
                        self._status_bar.showMessage(f"No legal moves available", 3000)
                        return
                else:
                    _OP_LOGGER.debug("No action_mask available")
                    self._status_bar.showMessage(f"Cannot determine legal moves", 3000)
                    return
            except Exception as mask_err:
                _OP_LOGGER.debug("Error getting action_mask: %s", mask_err)
                self._status_bar.showMessage(f"Error: {mask_err}", 3000)
                return

        # Execute the action
        try:
            _OP_LOGGER.debug("Executing env.step(%s)", action_index)
            env.step(action_index)
            _OP_LOGGER.debug("env.step completed successfully")

            # Check for game end - in PettingZoo AEC, check if ALL agents are terminated
            # or if there are no more agents to act
            next_player = env.agent_selection
            _OP_LOGGER.debug("After step, agent_selection=%s", next_player)

            # Check if game is truly over (all agents terminated or no agents left)
            all_terminated = all(env.terminations.values())
            all_truncated = all(env.truncations.values())
            no_agents_left = len(env.agents) == 0
            game_over = all_terminated or all_truncated or no_agents_left
            _OP_LOGGER.debug(
                "all_terminated=%s, all_truncated=%s, no_agents=%s, game_over=%s",
                all_terminated, all_truncated, no_agents_left, game_over,
            )

            # For backward compat with the rest of the code
            terminated = game_over
            truncated = False

            # Render updated board
            _OP_LOGGER.debug("Rendering frame...")
            self._render_pettingzoo_frame()
            _OP_LOGGER.debug("Frame rendered")

            if terminated or truncated:
                _OP_LOGGER.debug("Game over path")
                # Game over
                rewards = env.rewards
                winner = "Draw"
                for p, r in rewards.items():
                    if r > 0:
                        winner = p
                        break
                self._status_bar.showMessage(f"Game over! Winner: {winner}", 5000)
                self.log_constant(
                    LOG_UI_MAINWINDOW_INFO,
                    message=f"PettingZoo game ended",
                    extra={"winner": winner, "rewards": rewards},
                )
                # Disable both player step buttons when game is over
                self._control_panel.set_current_player("")  # Empty disables both
            else:
                _OP_LOGGER.debug("Game continues path")
                _OP_LOGGER.debug(
                    "_execute_pettingzoo_action: player_id=%s, action=%s, next_player=%s",
                    player_id, action_str, next_player,
                )
                # Update turn indicator for next player
                self._control_panel.set_turn_indicator(next_player, visible=True)
                # Toggle player step buttons for next turn
                _OP_LOGGER.debug("Calling set_current_player(%s)", next_player)
                self._control_panel.set_current_player(next_player)
                self._status_bar.showMessage(
                    f"{player_id} played {action_str}. Next: {next_player}",
                    2000
                )

            self.log_constant(
                LOG_OPERATOR_STEP_ALL_COMPLETED,
                message=f"PettingZoo step completed",
                extra={
                    "player": player_id,
                    "action": action_str,
                    "next_player": env.agent_selection,
                },
            )
            _OP_LOGGER.debug("_execute_pettingzoo_action completed successfully")

        except Exception as e:
            _OP_LOGGER.debug("_execute_pettingzoo_action EXCEPTION: %s", e, exc_info=True)
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Failed to execute action: {e}",
                extra={"player": player_id, "action": action_str, "index": action_index},
            )
            self._status_bar.showMessage(f"Move failed: {e}", 5000)
            # Re-enable the current player's button after error
            self._control_panel.set_current_player(player_id)

    def _poll_operator_responses(self, handles: list, max_wait_ms: int = 30000) -> None:
        """Poll for responses from operator subprocesses and update render view.

        Args:
            handles: List of (operator_id, handle) tuples to poll.
            max_wait_ms: Maximum time to wait for responses in milliseconds.
        """
        from PyQt6.QtCore import QTimer

        pending = list(handles)
        start_time = datetime.now()

        def poll_once():
            nonlocal pending
            if not pending:
                return

            # Check if we've exceeded max wait time
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            if elapsed > max_wait_ms:
                self.log_constant(
                    LOG_UI_MAINWINDOW_WARNING,
                    message=f"Timeout waiting for operator responses",
                    extra={"pending_count": len(pending)},
                )
                return

            still_pending = []
            for operator_id, handle in pending:
                # Try to read response (non-blocking)
                response = handle.try_read_response(timeout=0.1)
                if response is not None:
                    response_type = response.get("type", "unknown")
                    self._handle_operator_response(operator_id, response)
                    # If we got a non-step response (e.g., "ready" from reset), keep polling
                    if response_type != "step" and handle.is_running:
                        still_pending.append((operator_id, handle))
                else:
                    # Check if process is still running
                    if handle.is_running:
                        still_pending.append((operator_id, handle))
                    else:
                        self.log_constant(
                            LOG_UI_MAINWINDOW_WARNING,
                            message=f"Operator process terminated while waiting for response",
                            extra={"operator_id": operator_id},
                        )
                        self._multi_operator_service.set_operator_state(operator_id, "stopped")
                        self._render_tabs.set_operator_status(operator_id, "stopped")

            pending = still_pending
            if pending:
                # Schedule another poll
                QTimer.singleShot(200, poll_once)

        poll_once()

    def _handle_operator_response(self, operator_id: str, response: dict) -> None:
        """Handle a response from an operator subprocess.

        Args:
            operator_id: The operator that sent the response.
            response: The parsed JSON response dict.
        """
        response_type = response.get("type", "unknown")

        if response_type == "step":
            # Build render payload from step response
            payload = {
                "step_index": response.get("step_index", 0),
                "episode_index": response.get("episode_index", 0),
                "reward": response.get("reward", 0.0),
                "total_reward": response.get("total_reward", 0.0),
                "terminated": response.get("terminated", False),
                "truncated": response.get("truncated", False),
                "action": response.get("action", ""),
                "observation": response.get("observation", ""),
                # Include render payload if available (format: {"mode": "rgb", "rgb": [...], "width": N, "height": N})
                "render_payload": response.get("render_payload"),
            }
            self._render_tabs.display_operator_payload(operator_id, payload)
            self.log_constant(
                LOG_UI_MAINWINDOW_TRACE,
                message=f"Received step response from operator",
                extra={
                    "operator_id": operator_id,
                    "step_index": payload["step_index"],
                    "reward": payload["reward"],
                },
            )

        elif response_type == "ready":
            # Build payload to reset stats and display initial render
            payload = {
                "step_index": response.get("step_index", 0),
                "episode_index": response.get("episode_index", 0),
                "reward": 0.0,
                "episode_reward": response.get("episode_reward", 0.0),
                "render_payload": response.get("render_payload"),
            }
            self._render_tabs.display_operator_payload(operator_id, payload)
            self.log_constant(
                LOG_UI_MAINWINDOW_INFO,
                message=f"Operator ready",
                extra={"operator_id": operator_id, "seed": response.get("seed")},
            )

        elif response_type == "error":
            self.log_constant(
                LOG_UI_MAINWINDOW_ERROR,
                message=f"Operator error: {response.get('message', 'Unknown error')}",
                extra={"operator_id": operator_id},
            )
            self._render_tabs.set_operator_status(operator_id, "error")

        elif response_type == "stopped":
            self._multi_operator_service.set_operator_state(operator_id, "stopped")
            self._render_tabs.set_operator_status(operator_id, "stopped")

        else:
            self.log_constant(
                LOG_UI_MAINWINDOW_WARNING,
                message=f"Unknown response type from operator",
                extra={"operator_id": operator_id, "type": response_type},
            )

    def _on_stop_operators(self) -> None:
        """Stop all running operators.

        Terminates all worker subprocesses and updates status indicators.
        """
        # First stop via the launcher (actually terminates subprocesses)
        stopped_launcher_ids = self._operator_launcher.stop_all()

        # Then update the service state
        stopped_service_ids = self._multi_operator_service.stop_all()

        # Combine stopped IDs (may differ if launcher had extras)
        all_stopped = set(stopped_launcher_ids) | set(stopped_service_ids)

        if not all_stopped:
            self._status_bar.showMessage("No operators running", 3000)
            return

        # Update status indicators
        for operator_id in all_stopped:
            self._render_tabs.set_operator_status(operator_id, "stopped")

        count = len(all_stopped)
        self._status_bar.showMessage(
            f"Stopped {count} operator{'s' if count != 1 else ''}",
            3000
        )
        self.log_constant(
            LOG_OPERATOR_STOP_ALL_COMPLETED,
            message=f"Stopped {count} operators",
            extra={"operator_ids": list(all_stopped)},
        )

        # Disable PettingZoo mode if it was active
        if self._shared_pettingzoo_env is not None:
            self._control_panel.set_pettingzoo_mode(False)
            self._control_panel.set_turn_indicator("", visible=False)
            self._shared_pettingzoo_env = None
            self._pettingzoo_player_handles.clear()

    def _on_initialize_operator(self, operator_id: str, config: OperatorConfig, seed: int) -> None:
        """Initialize environment for operator preview with shared seed.

        Creates the environment, resets it with the shared seed, and displays
        the initial observation. Using the same seed for all operators ensures
        identical environments for controlled scientific comparison.

        Args:
            operator_id: The operator's unique ID
            config: Operator configuration with env_name and task
            seed: Shared seed for reproducible environment initialization
        """
        env_name = config.env_name
        task = config.task

        self._status_bar.showMessage(f"Initializing {task} with seed={seed}...", 2000)
        self.log_constant(
            LOG_OPERATOR_ENV_PREVIEW_STARTED,
            message=f"Loading environment preview for {env_name}/{task}",
            extra={"operator_id": operator_id, "env_name": env_name, "task": task, "seed": seed},
        )

        try:
            env = None
            rgb_frame = None
            board_game_payload: Dict[str, Any] | None = None

            if env_name in ("babyai", "minigrid"):
                # Use MiniGrid/BabyAI via gymnasium
                try:
                    import gymnasium as gym
                    import minigrid
                    # Only register if not already in registry
                    if "MiniGrid-Empty-5x5-v0" not in gym.envs.registry:
                        minigrid.register_minigrid_envs()
                except ImportError:
                    self._status_bar.showMessage(
                        f"MiniGrid not installed - cannot preview {task}",
                        5000
                    )
                    return

                env = gym.make(task, render_mode="rgb_array")
                env.reset(seed=seed)
                rgb_frame = env.render()
                env.close()

            elif env_name == "crafter":
                # Use Crafter environment with high resolution from config
                # Note: Crafter takes seed in __init__, not reset()
                try:
                    import crafter
                    cfg = game_configs.CrafterConfig()
                    env = crafter.Env(size=cfg.size, seed=seed)
                    env.reset()
                    rgb_frame = env.render()
                    env.close()
                except ImportError:
                    self._status_bar.showMessage(
                        "Crafter not installed - cannot preview",
                        5000
                    )
                    return

            elif env_name == "nle":
                # NLE (NetHack) uses TTY rendering - convert to RGB via nle_render
                try:
                    import nle  # noqa: F401
                    import gymnasium as gym
                    from gym_gui.core.adapters.nle_render import render_tty_to_rgb

                    # NLE doesn't support rgb_array mode - use default and get tty_chars
                    env = gym.make(
                        task,
                        observation_keys=("tty_chars", "tty_colors", "blstats"),
                    )
                    obs, _ = env.reset(seed=seed)
                    tty_chars = obs.get("tty_chars")
                    tty_colors = obs.get("tty_colors")
                    env.close()

                    # Convert TTY to RGB using the existing renderer
                    if tty_chars is not None:
                        rgb_frame = render_tty_to_rgb(tty_chars, tty_colors)
                        # Scale up for better visibility (3x)
                        rgb_frame = np.repeat(np.repeat(rgb_frame, 3, axis=0), 3, axis=1)
                except ImportError:
                    self._status_bar.showMessage(
                        "NLE not installed - cannot preview",
                        5000
                    )
                    return
                except Exception as e:
                    self._status_bar.showMessage(f"Cannot preview NLE: {e}", 5000)
                    return

            elif env_name == "minihack":
                # MiniHack supports 'rgb_array' mode
                try:
                    import minihack  # noqa: F401
                    import gymnasium as gym
                    env = gym.make(task, render_mode="rgb_array")
                    env.reset(seed=seed)
                    rgb_frame = env.render()
                    env.close()
                except ImportError:
                    self._status_bar.showMessage(
                        "MiniHack not installed - cannot preview",
                        5000
                    )
                    return

            elif env_name == "pettingzoo":
                # PettingZoo classic games (chess, go, connect_four, etc.)
                # These use their own factory functions, not gymnasium.make()
                try:
                    from pettingzoo.classic import (
                        chess_v6,
                        connect_four_v3,
                        go_v5,
                        tictactoe_v3,
                    )

                    # Map task names to environment factories
                    pz_env_factories = {
                        "chess_v6": chess_v6.env,
                        "connect_four_v3": connect_four_v3.env,
                        "go_v5": go_v5.env,
                        "tictactoe_v3": tictactoe_v3.env,
                    }

                    if task not in pz_env_factories:
                        self._status_bar.showMessage(
                            f"Unknown PettingZoo game: {task}",
                            5000
                        )
                        return

                    # Create environment with rgb_array rendering
                    env = pz_env_factories[task](render_mode="rgb_array")
                    env.reset(seed=seed)
                    # PettingZoo AEC envs render() returns the board
                    rgb_frame = env.render()

                    # Build game-specific payload for BoardGameRendererStrategy
                    board_game_payload: Dict[str, Any] | None = None
                    if task == "chess_v6" and hasattr(env, "board"):
                        # Extract chess-specific data for BoardGameRendererStrategy
                        import chess
                        board: chess.Board = env.board
                        legal_moves = [move.uci() for move in board.legal_moves]
                        current_player = "white" if board.turn == chess.WHITE else "black"
                        board_game_payload = {
                            "chess": {
                                "fen": board.fen(),
                                "legal_moves": legal_moves,
                                "current_player": current_player,
                                "is_check": board.is_check(),
                            },
                            "game_id": "chess",
                        }
                    elif task == "connect_four_v3" and hasattr(env, "board"):
                        board_game_payload = {
                            "connect_four": {
                                "board": env.board.tolist() if hasattr(env.board, "tolist") else list(env.board),
                                "current_player": getattr(env, "agent_selection", "player_0"),
                            },
                            "game_id": "connect_four",
                        }
                    elif task == "tictactoe_v3" and hasattr(env, "board"):
                        board_game_payload = {
                            "board": env.board.tolist() if hasattr(env.board, "tolist") else list(env.board),
                            "current_player": getattr(env, "agent_selection", "player_1"),
                            "game_id": "tictactoe",
                        }

                    env.close()

                except ImportError as e:
                    self._status_bar.showMessage(
                        f"PettingZoo classic games not installed: {e}",
                        5000
                    )
                    self.log_constant(
                        LOG_OPERATOR_ENV_PREVIEW_IMPORT_ERROR,
                        message=f"PettingZoo classic games not installed: {e}",
                        extra={"operator_id": operator_id, "env_name": env_name, "task": task, "error": str(e)},
                    )
                    return
                except Exception as e:
                    self._status_bar.showMessage(
                        f"Cannot preview PettingZoo {task}: {e}",
                        5000
                    )
                    self.log_constant(
                        LOG_OPERATOR_ENV_PREVIEW_ERROR,
                        message=f"Cannot preview PettingZoo {task}: {e}",
                        extra={"operator_id": operator_id, "env_name": env_name, "task": task, "error": str(e)},
                    )
                    return

            else:
                # Generic gymnasium environment
                try:
                    import gymnasium as gym
                    env = gym.make(task, render_mode="rgb_array")
                    env.reset(seed=seed)
                    rgb_frame = env.render()
                    env.close()
                except Exception as e:
                    self._status_bar.showMessage(
                        f"Cannot preview {env_name}/{task}: {e}",
                        5000
                    )
                    return

            if rgb_frame is not None:
                # Build payload for the render container
                # Extract dimensions safely for type checker
                if isinstance(rgb_frame, np.ndarray) and rgb_frame.ndim >= 2:
                    # Check if we need to scale image to target resolution
                    image_scale = config.settings.get("image_scale", 0)
                    if image_scale and image_scale > 0:
                        # Scale image to target resolution (square)
                        from PIL import Image
                        pil_image = Image.fromarray(rgb_frame)
                        pil_image = pil_image.resize(
                            (image_scale, image_scale),
                            Image.Resampling.LANCZOS
                        )
                        rgb_frame = np.array(pil_image)

                    shape = cast(tuple[int, ...], rgb_frame.shape)
                    frame_height, frame_width = shape[0], shape[1]
                    frame_data = rgb_frame.tolist()
                else:
                    frame_height, frame_width = 0, 0
                    frame_data = rgb_frame
                # Build payload - use board game payload for board games, RGB for others
                if board_game_payload is not None:
                    # Use structured board game payload for BoardGameRendererStrategy
                    payload = {
                        "render_payload": board_game_payload,
                        "episode_index": 0,
                        "step_index": 0,
                        "reward": 0.0,
                    }
                else:
                    # Use RGB payload for generic environments
                    payload = {
                        "render_payload": {
                            "mode": "rgb",
                            "rgb": frame_data,
                            "width": frame_width,
                            "height": frame_height,
                        },
                        "episode_index": 0,
                        "step_index": 0,
                        "reward": 0.0,
                    }

                # Update the operator's config (updates header: name, type badge, env/task)
                self._render_tabs.update_operator_view(config)

                # Set the container display size based on selected container size
                container_size = config.settings.get("container_size", 0)
                if container_size and container_size > 0:
                    self._render_tabs.set_operator_display_size(
                        operator_id, container_size, container_size
                    )

                # Display in the operator's container
                self._render_tabs.display_operator_payload(operator_id, payload)
                # Update status to "loaded" to indicate environment is ready
                self._render_tabs.set_operator_status(operator_id, "loaded")
                self._render_tabs.switch_to_multi_operator_tab()

                # Update the environment size in the operator config widget
                if frame_width > 0 and frame_height > 0:
                    self._control_panel.set_operator_environment_size(
                        operator_id, frame_width, frame_height,
                        container_size if container_size and container_size > 0 else None
                    )

                self._status_bar.showMessage(
                    f"Previewing {task} - ready to start",
                    3000
                )
                self.log_constant(
                    LOG_OPERATOR_ENV_PREVIEW_SUCCESS,
                    message=f"Environment preview loaded for {env_name}/{task}",
                    extra={
                        "operator_id": operator_id,
                        "env_name": env_name,
                        "task": task,
                        "width": frame_width,
                        "height": frame_height,
                    },
                )
            else:
                self._status_bar.showMessage(
                    f"No render available for {task}",
                    3000
                )

        except Exception as e:
            self.log_constant(
                LOG_OPERATOR_ENV_PREVIEW_ERROR,
                message=f"Failed to initialize operator {operator_id}: {e}",
                extra={"operator_id": operator_id, "env_name": env_name, "task": task, "error": str(e)},
            )
            self._status_bar.showMessage(
                f"Failed to initialize: {e}",
                5000
            )

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

    # Multi-agent game methods delegated to MultiAgentGameHandler:
    # - on_load_requested
    # - on_start_requested
    # - on_reset_requested
    # - on_ai_opponent_changed

    # Policy evaluation methods delegated to PolicyEvaluationHandler:
    # - handle_evaluate_request
    # - _launch_evaluation (uses EvaluationWorker QThread)

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
        # Get overrides to pass input_mode setting to human input controller
        current_game = self._session.game_id
        overrides = None
        if current_game is not None:
            try:
                gid = GameId(current_game) if isinstance(current_game, str) else current_game
                overrides = self._control_panel.get_overrides(gid)
            except Exception:
                pass
        self._human_input.configure(
            self._session.game_id,
            self._session.action_space,
            overrides=overrides,
        )
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

    # Training form methods delegated to TrainingFormHandler:
    # - on_trained_agent_requested
    # - on_train_agent_requested
    # - on_resume_training_requested

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
        self.log_constant(
            LOG_UI_MAINWINDOW_TRACE,
            message="Calling maybe_open_fastlane_tab",
            extra={
                "run_id": run_id,
                "agent_id_key": agent_id_key,
                "metadata_keys": list(metadata.keys()) if metadata else None,
                "ui_fastlane_only": metadata.get("ui", {}).get("fastlane_only") if metadata else None,
                "worker_module": metadata.get("worker", {}).get("module") if metadata else None,
            },
        )
        self._fastlane_tab_handler.maybe_open_fastlane_tab(run_id, agent_id_key, metadata)

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

        self._fastlane_tab_handler.maybe_open_fastlane_tab(
            run_id, agent_id, self._resolve_run_metadata(run_id, agent_id)
        )

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
            worker_id = "cleanrl_worker"  # TODO: Extract from config/payload if supporting multiple workers
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

    # FastLane tab methods delegated to FastLaneTabHandler:
    # - maybe_open_fastlane_tab
    # - open_ray_fastlane_tabs
    # - open_single_fastlane_tab
    # - get_num_workers, get_canonical_agent_id, get_worker_id, get_env_id
    # - get_run_mode, metadata_supports_fastlane, clear_tabs_for_run

    # Training monitor methods delegated to TrainingMonitorHandler:
    # - poll_for_new_runs, start_run_watch, shutdown_run_watch
    # - backfill_run_metadata_from_disk, auto_subscribe_run

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
                # TODO: Replay tab creation needs to be reimplemented for current workers
                # Workers should implement their own replay tab via their presenter's create_tabs method
                self.log_constant(
                    LOG_UI_MAINWINDOW_TRACE,
                    message="_on_training_finished: replay tab not available for this worker",
                    extra={"run_id": run_id, "agent_id": agent_id, "replay_tab_name": replay_tab_name},
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

        # Clear FastLane tab tracking for this run (delegated to handler)
        self._fastlane_tab_handler.clear_tabs_for_run(run_id)

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

    def _on_settings_clicked(self) -> None:
        """Handle Settings toolbar action click.

        Creates dialog on first use (lazy initialization).
        Shows non-modal dialog allowing interaction with main window.
        """
        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(parent=self)
            self._settings_dialog.setting_changed.connect(self._on_setting_changed)
            self._settings_dialog.settings_reset.connect(self._on_settings_reset)

        # Show non-modal dialog (user can interact with main window)
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def _on_setting_changed(self, key: str, value: str) -> None:
        """Handle individual setting change from Settings dialog.

        Logs the change but does not reload settings (requires restart).

        Args:
            key: Setting key
            value: New value
        """
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message=f"Setting changed: {key}",
            extra={"setting_key": key, "setting_value_length": len(value)},
        )
        self._status_bar.showMessage(f"Setting saved: {key}", 3000)

    def _on_settings_reset(self) -> None:
        """Handle reset of all settings to defaults."""
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message="All settings reset to defaults",
        )
        self._status_bar.showMessage("All settings reset to defaults (restart required)", 5000)

    def closeEvent(self, a0: QtGui.QCloseEvent | None) -> None:
        logging.getLogger().removeHandler(self._log_handler)

        # Shutdown live telemetry controller
        if hasattr(self, "_live_controller"):
            self._live_controller.shutdown()

        self._training_monitor_handler.shutdown_run_watch()

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

        # Clean up chat panel (optional - requires [chat] extra)
        if hasattr(self, "_chat_panel") and self._chat_panel is not None:
            self._chat_panel.cleanup()

        # Close settings dialog if open
        if hasattr(self, "_settings_dialog") and self._settings_dialog is not None:
            self._settings_dialog.close()
            self._settings_dialog = None

        # Stop all operator subprocesses
        if hasattr(self, "_operator_launcher"):
            stopped = self._operator_launcher.stop_all()
            if stopped:
                self.log_constant(
                    LOG_UI_MAINWINDOW_INFO,
                    message=f"Stopped {len(stopped)} operator(s) on shutdown",
                    extra={"operator_ids": stopped},
                )

        # Shutdown session
        self._session.shutdown()

        if hasattr(self, "_time_refresh_timer") and self._time_refresh_timer.isActive():
            self._time_refresh_timer.stop()

        super().closeEvent(a0)


__all__ = ["MainWindow"]
