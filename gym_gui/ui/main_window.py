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
from gym_gui.services.operator import OperatorConfig, OperatorDescriptor, MultiOperatorService
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
from gym_gui.ui.widgets.spade_bdi_worker_tabs import (
    AgentReplayTab,
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
    HumanVsAgentHandler,
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
from gym_gui.ui.widgets.advanced_config import LaunchConfig, RunMode

from gym_gui.constants.optional_deps import (
    get_mjpc_launcher,
    get_godot_launcher,
    OptionalDependencyError,
)

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

        # Multi-agent game routing handler
        self._multi_agent_game_handler = MultiAgentGameHandler(
            status_bar=self._status_bar,
            chess_loader=self._chess_env_loader,
            connect_four_loader=self._connect_four_env_loader,
            go_loader=self._go_env_loader,
            tictactoe_loader=self._tictactoe_env_loader,
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
        # Multi-operator signals (Phase 6)
        self._control_panel.operators_changed.connect(self._on_operators_changed)
        self._control_panel.start_operators_requested.connect(self._on_start_operators)
        self._control_panel.stop_operators_requested.connect(self._on_stop_operators)
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

    def _on_start_operators(self) -> None:
        """Start all configured operators.

        Launches worker subprocesses for each operator and switches
        to the Multi-Operator tab for viewing.
        """
        active_operators = self._multi_operator_service.get_active_operators()
        if not active_operators:
            self._status_bar.showMessage("No operators configured to start", 3000)
            return

        # Start all operators via the service
        started_ids = self._multi_operator_service.start_all()

        # Update status indicators
        for operator_id in started_ids:
            self._render_tabs.set_operator_status(operator_id, "running")

        # Switch to Multi-Operator tab
        self._render_tabs.switch_to_multi_operator_tab()

        count = len(started_ids)
        self._status_bar.showMessage(
            f"Started {count} operator{'s' if count != 1 else ''}",
            3000
        )
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message=f"Started {count} operators",
            extra={"operator_ids": started_ids},
        )

    def _on_stop_operators(self) -> None:
        """Stop all running operators.

        Terminates all worker subprocesses and updates status indicators.
        """
        active_operators = self._multi_operator_service.get_active_operators()
        if not active_operators:
            self._status_bar.showMessage("No operators running", 3000)
            return

        # Stop all operators via the service
        stopped_ids = self._multi_operator_service.stop_all()

        # Update status indicators
        for operator_id in stopped_ids:
            self._render_tabs.set_operator_status(operator_id, "stopped")

        count = len(stopped_ids)
        self._status_bar.showMessage(
            f"Stopped {count} operator{'s' if count != 1 else ''}",
            3000
        )
        self.log_constant(
            LOG_UI_MAINWINDOW_INFO,
            message=f"Stopped {count} operators",
            extra={"operator_ids": stopped_ids},
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

        # Shutdown session
        self._session.shutdown()

        if hasattr(self, "_time_refresh_timer") and self._time_refresh_timer.isActive():
            self._time_refresh_timer.stop()

        super().closeEvent(a0)


__all__ = ["MainWindow"]
