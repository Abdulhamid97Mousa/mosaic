from __future__ import annotations

"""Main Qt window for the Gym GUI application."""

import logging
from datetime import datetime
from typing import Any, Dict, List

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.config.settings import Settings
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.core.factories.adapters import available_games
from gym_gui.controllers.human_input import HumanInputController
from gym_gui.controllers.session import SessionController
from gym_gui.ui.logging_bridge import LogRecordPayload, QtLogHandler


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

    def __init__(self, settings: Settings, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._session = SessionController(settings, self)
        self._log_handler = QtLogHandler(parent=self)
        self._log_records: List[LogRecordPayload] = []
        self._current_game: GameId | None = None
        self._current_mode: ControlMode = settings.default_control_mode
        self._auto_running = False
        self._human_input = HumanInputController(self, self._session)
        self._game_overrides: Dict[GameId, Dict[str, Any]] = {
            GameId.FROZEN_LAKE: {"frozen_lake_is_slippery": settings.frozen_lake_is_slippery}
        }
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

        self._control_panel = self._build_control_panel()
        splitter.addWidget(self._control_panel)

        right_panel = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, central)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)

        self._render_group = QtWidgets.QGroupBox("Render View", right_panel)
        render_layout = QtWidgets.QVBoxLayout(self._render_group)
        self._render_view = RenderView(self._render_group)
        render_layout.addWidget(self._render_view)
        right_panel.addWidget(self._render_group)

        self._log_group = QtWidgets.QGroupBox("Runtime Log", right_panel)
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
        right_panel.addWidget(self._log_group)
        right_panel.setStretchFactor(0, 2)
        right_panel.setStretchFactor(1, 1)

    def _build_control_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setSpacing(12)

        # Environment picker
        env_group = QtWidgets.QGroupBox("Environment")
        env_layout = QtWidgets.QVBoxLayout(env_group)
        self._game_combo = QtWidgets.QComboBox()
        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(0, 10_000_000)
        self._seed_spin.setValue(0)
        load_button = QtWidgets.QPushButton("Load")
        load_button.clicked.connect(self._on_load_clicked)
        env_layout.addWidget(QtWidgets.QLabel("Select environment"))
        env_layout.addWidget(self._game_combo)
        env_layout.addWidget(QtWidgets.QLabel("Seed"))
        env_layout.addWidget(self._seed_spin)
        env_layout.addWidget(load_button)
        panel_layout.addWidget(env_group)

        # Game configuration placeholder (populated per environment)
        self._config_group = QtWidgets.QGroupBox("Game Configuration")
        self._config_layout = QtWidgets.QFormLayout(self._config_group)
        panel_layout.addWidget(self._config_group)

        self._frozen_slippery_checkbox: QtWidgets.QCheckBox | None = None

        # Mode selector
        mode_group = QtWidgets.QGroupBox("Control Mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self._mode_buttons: Dict[ControlMode, QtWidgets.QRadioButton] = {}
        mode_button_group = QtWidgets.QButtonGroup(mode_group)
        for mode in ControlMode:
            label = self.CONTROL_MODE_LABELS.get(mode, mode.name.replace("_", " ").title())
            button = QtWidgets.QRadioButton(label)
            button.setEnabled(False)
            mode_button_group.addButton(button)
            mode_layout.addWidget(button)
            self._mode_buttons[mode] = button
            button.toggled.connect(self._on_mode_toggled)
        panel_layout.addWidget(mode_group)

        # Control buttons
        controls_group = QtWidgets.QGroupBox("Controls")
        controls_layout = QtWidgets.QHBoxLayout(controls_group)
        self._play_button = QtWidgets.QPushButton("Play")
        self._pause_button = QtWidgets.QPushButton("Pause")
        self._step_button = QtWidgets.QPushButton("Agent Step")
        self._reset_button = QtWidgets.QPushButton("Reset")
        controls_layout.addWidget(self._play_button)
        controls_layout.addWidget(self._pause_button)
        controls_layout.addWidget(self._step_button)
        controls_layout.addWidget(self._reset_button)
        panel_layout.addWidget(controls_group)

        self._play_button.clicked.connect(self._session.start_auto_play)
        self._pause_button.clicked.connect(self._session.stop_auto_play)
        self._step_button.clicked.connect(self._session.perform_agent_step)
        self._reset_button.clicked.connect(self._on_reset_clicked)

        # Status group
        status_group = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QFormLayout(status_group)
        self._step_label = QtWidgets.QLabel("0")
        self._reward_label = QtWidgets.QLabel("0.0")
        self._terminated_label = QtWidgets.QLabel("No")
        self._truncated_label = QtWidgets.QLabel("No")
        self._turn_label = QtWidgets.QLabel("human")
        self._awaiting_label = QtWidgets.QLabel("–")
        self._session_time_label = QtWidgets.QLabel("00:00:00")
        self._active_time_label = QtWidgets.QLabel("—")
        self._outcome_time_label = QtWidgets.QLabel("—")
        status_layout.addRow("Step", self._step_label)
        status_layout.addRow("Reward", self._reward_label)
        status_layout.addRow("Episode Finished", self._terminated_label)
        status_layout.addRow("Episode Aborted", self._truncated_label)
        status_layout.addRow("Turn", self._turn_label)
        status_layout.addRow("Awaiting Input", self._awaiting_label)
        status_layout.addRow("Session Uptime", self._session_time_label)
        status_layout.addRow("Active Play Time", self._active_time_label)
        status_layout.addRow("Outcome Logged At", self._outcome_time_label)
        panel_layout.addWidget(status_group)

        panel_layout.addStretch(1)
        return panel

    def _connect_signals(self) -> None:
        self._game_combo.currentIndexChanged.connect(self._on_game_changed)
        self._log_filter.currentTextChanged.connect(self._on_log_filter_changed)

        self._session.session_initialized.connect(self._on_session_initialized)
        self._session.step_processed.connect(self._on_step_processed)
        self._session.episode_finished.connect(self._on_episode_finished)
        self._session.status_message.connect(self._on_status_message)
        self._session.awaiting_human.connect(self._on_awaiting_human)
        self._session.turn_changed.connect(self._on_turn_changed)
        self._session.error_occurred.connect(self._on_error)
        self._session.auto_play_state_changed.connect(self._on_auto_play_state)

        self._log_handler.emitter.record_emitted.connect(self._append_log_record)

    def _populate_environments(self) -> None:
        games = sorted(available_games(), key=lambda g: g.value)
        for game in games:
            self._game_combo.addItem(game.value, game)
        if games:
            default = GameId(self._settings.gym_default_env)
            index = self._game_combo.findData(default)
            if index >= 0:
                self._game_combo.setCurrentIndex(index)
            else:
                self._game_combo.setCurrentIndex(0)
            self._on_game_changed(self._game_combo.currentIndex())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_game_changed(self, index: int) -> None:
        game = self._game_combo.itemData(index)
        if not isinstance(game, GameId):
            return
        self._current_game = game
        supported = SessionController.supported_control_modes(game)
        for mode, button in self._mode_buttons.items():
            button.blockSignals(True)
            button.setEnabled(mode in supported)
            button.blockSignals(False)
        # Select default mode if current one unsupported
        if self._current_mode not in supported:
            self._current_mode = supported[0]
        self._mode_buttons[self._current_mode].setChecked(True)
        self._status_bar.showMessage(f"Selected {game.value}. Load to begin.")
        self._update_control_states()
        self._refresh_game_config_ui()

    def _on_mode_toggled(self, checked: bool) -> None:
        if not checked:
            return
        sender = self.sender()
        for mode, button in self._mode_buttons.items():
            if button is sender:
                self._current_mode = mode
                label = self.CONTROL_MODE_LABELS.get(mode, mode.value)
                self._status_bar.showMessage(f"Mode set to {label}")
                self._update_control_states()
                break

    def _on_load_clicked(self) -> None:
        if self._current_game is None:
            return
        seed = int(self._seed_spin.value())
        overrides = dict(self._game_overrides.get(self._current_game, {}))
        self._session.load_environment(
            self._current_game,
            self._current_mode,
            seed=seed,
            settings_overrides=overrides or None,
        )

    def _on_reset_clicked(self) -> None:
        seed = int(self._seed_spin.value())
        self._session.reset_environment(seed=seed)

    def _on_frozen_slippery_toggled(self, state: int) -> None:
        enabled = state == QtCore.Qt.CheckState.Checked
        overrides = self._game_overrides.setdefault(GameId.FROZEN_LAKE, {})
        overrides["frozen_lake_is_slippery"] = enabled
        status = "enabled" if enabled else "disabled"
        if self._current_game == GameId.FROZEN_LAKE and self._session.game_id == GameId.FROZEN_LAKE:
            self._status_bar.showMessage(
                f"Frozen Lake slippery ice {status}. Reloading environment...",
                5000,
            )
            seed = int(self._seed_spin.value())
            self._session.load_environment(
                self._current_game,
                self._current_mode,
                seed=seed,
                settings_overrides=dict(overrides),
            )
        else:
            if self._frozen_slippery_checkbox is not None:
                self._frozen_slippery_checkbox.blockSignals(True)
                self._frozen_slippery_checkbox.setChecked(enabled)
                self._frozen_slippery_checkbox.blockSignals(False)
            self._status_bar.showMessage(
                f"Frozen Lake slippery ice {status}. Reload to apply.",
                5000,
            )

    def _on_session_initialized(self, game_id: str, mode: str, step: object) -> None:
        try:
            mode_label = self.CONTROL_MODE_LABELS[ControlMode(mode)]
        except Exception:
            mode_label = mode
        self._status_bar.showMessage(f"Loaded {game_id} in {mode_label} mode")
        self._log_console.appendPlainText(f"Loaded {game_id} ({mode_label})")
        self._auto_running = False
        self._human_input.configure(self._session.game_id, self._session.action_space)
        self._update_control_states()
        self._refresh_time_labels()

    def _on_step_processed(self, step: object, index: int) -> None:
        if not hasattr(step, "reward"):
            return
        reward = getattr(step, "reward", 0.0)
        terminated = getattr(step, "terminated", False)
        truncated = getattr(step, "truncated", False)
        render_payload = getattr(step, "render_payload", None)

        self._step_label.setText(str(index))
        self._reward_label.setText(f"{reward:.2f}")
        self._terminated_label.setText(self._format_bool(terminated))
        self._truncated_label.setText(self._format_bool(truncated))
        self._render_view.display(render_payload)
        self._refresh_time_labels()

    def _on_episode_finished(self, finished: bool) -> None:
        if finished:
            self._status_bar.showMessage("Episode finished")

    def _on_status_message(self, message: str) -> None:
        self._status_bar.showMessage(message, 5000)

    def _on_awaiting_human(self, waiting: bool, message: str) -> None:
        self._awaiting_label.setText("Yes" if waiting else "No")
        if message:
            self._status_bar.showMessage(message, 5000)
        enable_shortcuts = waiting or self._current_mode == ControlMode.HUMAN_ONLY
        self._human_input.set_enabled(enable_shortcuts)

    def _on_turn_changed(self, turn: str) -> None:
        self._turn_label.setText(turn)

    def _on_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Session Error", message)
        self._status_bar.showMessage(message, 5000)

    def _on_auto_play_state(self, running: bool) -> None:
        self._auto_running = running
        self._update_control_states()

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

    @staticmethod
    def _format_log(payload: LogRecordPayload) -> str:
        ts = datetime.fromtimestamp(payload.created).strftime("%H:%M:%S")
        return f"{ts} | {payload.level:<7} | {payload.name} | {payload.message}"

    @staticmethod
    def _format_bool(value: bool) -> str:
        return "Yes" if value else "No"

    def _refresh_time_labels(self) -> None:
        timers = self._session.timers
        self._session_time_label.setText(timers.launch_elapsed_formatted())
        self._active_time_label.setText(timers.first_move_elapsed_formatted())
        self._outcome_time_label.setText(timers.outcome_timestamp_formatted())

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
                "frozen_lake_is_slippery",
                self._settings.frozen_lake_is_slippery,
            )
            overrides["frozen_lake_is_slippery"] = value
            checkbox = QtWidgets.QCheckBox("Enable slippery ice (stochastic)", self._config_group)
            checkbox.setChecked(bool(value))
            checkbox.stateChanged.connect(self._on_frozen_slippery_toggled)
            self._frozen_slippery_checkbox = checkbox
            self._config_layout.addRow("Slippery ice", checkbox)
        else:
            placeholder = QtWidgets.QLabel("No overrides available for this game.")
            placeholder.setWordWrap(True)
            self._config_layout.addRow(placeholder)

    def _update_control_states(self) -> None:
        is_human = self._current_mode == ControlMode.HUMAN_ONLY
        if self._auto_running:
            self._play_button.setEnabled(False)
            self._pause_button.setEnabled(True)
        else:
            self._play_button.setEnabled(not is_human)
            self._pause_button.setEnabled(False)
        self._step_button.setEnabled((not is_human) and not self._auto_running)
        self._human_input.update_for_mode(self._current_mode)
        base_enabled = (
            self._current_mode in {ControlMode.HUMAN_ONLY, ControlMode.HYBRID_HUMAN_AGENT}
            or self._awaiting_label.text().lower() == "yes"
        )
        self._human_input.set_enabled(base_enabled)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI only
        logging.getLogger().removeHandler(self._log_handler)
        self._session.shutdown()
        if hasattr(self, "_time_refresh_timer") and self._time_refresh_timer.isActive():
            self._time_refresh_timer.stop()
        super().closeEvent(event)


class RenderView(QtWidgets.QTabWidget):
    """Render pane supporting grid and raw text displays."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid_table = QtWidgets.QTableWidget()
        self._grid_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        horizontal_header = self._grid_table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        vertical_header = self._grid_table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._grid_table.setAlternatingRowColors(True)
        self._grid_table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self._raw_text = QtWidgets.QPlainTextEdit()
        self._raw_text.setReadOnly(True)
        self._raw_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)

        self.addTab(self._grid_table, "Grid")
        self.addTab(self._raw_text, "Raw")
        self.setTabEnabled(0, False)

    def display(self, payload: object) -> None:
        if isinstance(payload, dict):
            mode = payload.get("mode")
            if mode == RenderMode.GRID.value and "grid" in payload:
                agent_pos = payload.get("agent_position")
                self._render_grid(payload["grid"], agent_pos)
            else:
                text = payload.get("ansi") or payload.get("text") or str(payload)
                self._raw_text.setPlainText(text)
                self.setTabEnabled(0, False)
                self.setCurrentWidget(self._raw_text)
        elif payload is None:
            self._raw_text.setPlainText("No render payload yet.")
            self.setTabEnabled(0, False)
            self.setCurrentWidget(self._raw_text)
        else:
            self._raw_text.setPlainText(str(payload))
            self.setTabEnabled(0, False)
            self.setCurrentWidget(self._raw_text)

    def _render_grid(self, grid: List[List[str]], agent_position: tuple[int, int] | None = None) -> None:
        # Support legacy payloads that provide a list of strings
        if grid and isinstance(grid[0], str):
            grid = [list(row) for row in grid]

        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        self._grid_table.clearContents()
        self._grid_table.setRowCount(rows)
        self._grid_table.setColumnCount(cols)
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self._grid_table.setItem(r, c, item)
        if agent_position is not None:
            r, c = agent_position
            item = self._grid_table.item(r, c)
            if item is not None:
                item.setBackground(QtGui.QColor("#4CAF50"))
                item.setForeground(QtGui.QColor("#ffffff"))
        self.setTabEnabled(0, True)
        self.setCurrentWidget(self._grid_table)
        # Ensure raw tab mirrors latest ansi if available
        self._raw_text.setPlainText("\n".join("".join(row) for row in grid))


__all__ = ["MainWindow", "RenderView"]
