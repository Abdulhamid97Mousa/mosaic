"""Composite widget that presents live render streams alongside telemetry replays."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, List, Mapping, Optional, Tuple

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.rendering import (
    RendererContext,
    RendererRegistry,
    RendererStrategy,
    create_default_renderer_registry,
)
from gym_gui.replays import EpisodeReplay, EpisodeReplayLoader
from gym_gui.services.service_locator import get_service_locator
from gym_gui.services.telemetry import TelemetryService
from gym_gui.ui.indicators.busy_indicator import modal_busy_indicator
from gym_gui.ui.indicators import RunSummary, TabClosureChoice, TabClosureDialog
from gym_gui.ui.widgets.live_telemetry_tab import LiveTelemetryTab
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_RENDER_TABS_TRACE,
    LOG_UI_RENDER_TABS_INFO,
    LOG_UI_RENDER_TABS_WARNING,
    LOG_UI_RENDER_TABS_ERROR,
    LOG_UI_RENDER_TABS_DELETE_REQUESTED,
    LOG_UI_RENDER_TABS_EVENT_FOR_DELETED_RUN,
    LOG_UI_RENDER_TABS_TAB_ADDED,
)


_LOGGER = logging.getLogger(__name__)


class RenderTabs(QtWidgets.QTabWidget, LogConstantMixin):
    """Tab widget combining grid, raw text, video, and replay views."""

    _current_game: GameId | None

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        telemetry_service: TelemetryService | None = None,
    ) -> None:
        super().__init__(parent)
        self._logger = _LOGGER

        locator = get_service_locator()
        registry = locator.resolve(RendererRegistry)
        if registry is None:
            registry = create_default_renderer_registry()
            locator.register(RendererRegistry, registry)
        self._registry = registry

        self._mode_hosts: dict[RenderMode, _RendererHost] = {}
        self._telemetry_service = telemetry_service
        self._current_game = None

        self._raw_tab = _RawTab(parent=self)

        self._grid_host = self._create_host(RenderMode.GRID, parent=self)
        if self._grid_host is not None:
            self._grid_tab_index = self.addTab(self._grid_host.widget, "Grid")
            self.setTabEnabled(self._grid_tab_index, False)
        else:
            self._grid_tab_index = -1

        self._raw_tab_index = self.addTab(self._raw_tab.widget, "Raw")
        self.setTabEnabled(self._raw_tab_index, True)

        self._video_host = self._create_host(RenderMode.RGB_ARRAY, parent=self)
        if self._video_host is not None:
            self._video_tab_index = self.addTab(self._video_host.widget, "Video")
            self.setTabEnabled(self._video_tab_index, False)
        else:
            self._video_tab_index = -1

        self._replay_tab = _ReplayTab(
            parent=self,
            telemetry_service=telemetry_service,
            renderer_registry=registry,
        )
        self._replay_tab_index = self.addTab(self._replay_tab, "Human Replay")
        self.setTabToolTip(self._replay_tab_index, "Review episodes from manual gameplay sessions only")

        # Dynamic agent tabs state
        self.init_dynamic_tab_state()

    def init_dynamic_tab_state(self) -> None:
        """Initialize state for dynamically created agent tabs."""
        # keyed by run_id → dict(name → QWidget)
        self._agent_tabs: dict[str, dict[str, QtWidgets.QWidget]] = {}

    def add_dynamic_tab(self, run_id: str, name: str, widget: QtWidgets.QWidget) -> None:
        """Add (or focus) a dynamic tab; stable across re-emits."""
        if run_id not in self._agent_tabs:
            self._agent_tabs[run_id] = {}
        if name in self._agent_tabs[run_id]:
            # Tab already exists, just focus it
            idx = self.indexOf(self._agent_tabs[run_id][name])
            if idx >= 0:
                self.setCurrentIndex(idx)
            return
        # Create new tab
        self._agent_tabs[run_id][name] = widget
        idx = self.addTab(widget, name)
        self.setTabToolTip(idx, f"{name} - Live training telemetry")
        
        # Add close button to the tab
        self._add_close_button_to_tab(idx, run_id, name)
        
        self.setCurrentIndex(idx)
    
    def _add_close_button_to_tab(self, tab_index: int, run_id: str, tab_name: str) -> None:
        """Add a close button to a dynamic tab."""
        close_button = QtWidgets.QPushButton(self)
        close_button.setText("✕")
        close_button.setMaximumWidth(30)
        close_button.setMaximumHeight(20)
        close_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        close_button.setToolTip("Close this tab")
        close_button.setStyleSheet("""
            QPushButton {
                border: none;
                background-color: transparent;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #dd4444;
                color: white;
                border-radius: 2px;
            }
        """)
        # Create a lambda to capture the parameters
        close_button.clicked.connect(
            lambda: self._close_dynamic_tab(run_id, tab_name, tab_index)
        )
        tab_bar = self.tabBar()
        if tab_bar is not None:
            tab_bar.setTabButton(tab_index, QtWidgets.QTabBar.ButtonPosition.RightSide, close_button)
    
    def _close_dynamic_tab(self, run_id: str, tab_name: str, tab_index: int) -> None:
        """Close a dynamic tab and cleanup resources."""
        try:
            self.log_constant(LOG_UI_RENDER_TABS_TRACE, message=f"_close_dynamic_tab: START (run_id={run_id}, tab_name={tab_name}, tab_index={tab_index})")

            # Get the widget before removing the tab
            widget = self.widget(tab_index)
            self.log_constant(LOG_UI_RENDER_TABS_TRACE, message=f"_close_dynamic_tab: Got widget: {widget}")

            apply_to_all = False

            if self._should_prompt_tab_closure(widget, tab_name):
                if widget is None:
                    return
                choice, apply_to_all = self._prompt_tab_closure(run_id, widget)
                if choice == TabClosureChoice.CANCEL:
                    self.log_constant(
                        LOG_UI_RENDER_TABS_INFO,
                        message="_close_dynamic_tab: User cancelled tab closure",
                        extra={"run_id": run_id, "tab_name": tab_name},
                    )
                    return
                self._execute_closure_choice(run_id, choice)

            if apply_to_all:
                self.log_constant(
                    LOG_UI_RENDER_TABS_INFO,
                    message="_close_dynamic_tab: Applying decision to all tabs",
                    extra={"run_id": run_id, "tab_name": tab_name},
                )
                self.remove_dynamic_tabs_for_run(run_id)
                return

            # Remove from tracking
            if run_id in self._agent_tabs and tab_name in self._agent_tabs[run_id]:
                del self._agent_tabs[run_id][tab_name]
                self.log_constant(LOG_UI_RENDER_TABS_TRACE, message="_close_dynamic_tab: Removed from tracking")

                # If this was the last tab for this run, clean up the run entry
                if not self._agent_tabs[run_id]:
                    del self._agent_tabs[run_id]
                    self.log_constant(LOG_UI_RENDER_TABS_TRACE, message="_close_dynamic_tab: Removed run entry")

            # Remove the tab from the widget
            self.log_constant(LOG_UI_RENDER_TABS_TRACE, message=f"_close_dynamic_tab: Removing tab at index {tab_index}")
            self.removeTab(tab_index)

            # Cleanup the widget
            if widget is not None:
                self.log_constant(LOG_UI_RENDER_TABS_TRACE, message="_close_dynamic_tab: Cleaning up widget")
                # Call cleanup on LiveTelemetryTab to prevent segfaults from pending QTimer callbacks
                if hasattr(widget, 'cleanup'):
                    try:
                        self.log_constant(LOG_UI_RENDER_TABS_TRACE, message="_close_dynamic_tab: Calling widget.cleanup()")
                        # Type: ignore because cleanup is a custom method on LiveTelemetryTab
                        widget.cleanup()  # type: ignore[attr-defined]
                        self.log_constant(LOG_UI_RENDER_TABS_TRACE, message="_close_dynamic_tab: widget.cleanup() completed")
                    except Exception as e:
                        self.log_constant(LOG_UI_RENDER_TABS_ERROR, message=f"_close_dynamic_tab: Error cleaning up tab: {e}", exc_info=e)
                self.log_constant(LOG_UI_RENDER_TABS_TRACE, message="_close_dynamic_tab: Calling widget.deleteLater()")
                widget.deleteLater()

            self.log_constant(LOG_UI_RENDER_TABS_TRACE, message="_close_dynamic_tab: COMPLETE")
        except Exception as e:
            self.log_constant(LOG_UI_RENDER_TABS_ERROR, message=f"_close_dynamic_tab: FATAL ERROR - {e}", exc_info=e)

    def _should_prompt_tab_closure(self, widget: QtWidgets.QWidget | None, tab_name: str) -> bool:
        """Determine whether we should show the tab closure dialog."""

        if widget is None:
            return False
        if isinstance(widget, LiveTelemetryTab):
            return True
        # Support legacy naming for live agent tabs even if widget type changes
        return tab_name.lower().startswith("live-agent-")

    def _prompt_tab_closure(self, run_id: str, widget: QtWidgets.QWidget) -> Tuple[TabClosureChoice, bool]:
        """Display the tab closure dialog and return the user's decision."""

        dialog = TabClosureDialog(self)
        summary = self._build_run_summary(run_id, widget)
        dialog.set_run_summary(summary)
        choice = dialog.exec()
        apply_to_all = dialog.is_batch_apply()
        self.log_constant(
            LOG_UI_RENDER_TABS_TRACE,
            message="_prompt_tab_closure: Dialog completed",
            extra={
                "run_id": run_id,
                "choice": getattr(choice, "value", str(choice)),
                "apply_to_all": apply_to_all,
            },
        )
        return choice, apply_to_all

    def _build_run_summary(self, run_id: str, widget: QtWidgets.QWidget) -> RunSummary:
        """Construct a RunSummary for the dialog, using stored stats when available."""

        agent_id = getattr(widget, "agent_id", "unknown")
        dropped_steps = int(getattr(widget, "_dropped_steps", 0))
        dropped_episodes = int(getattr(widget, "_dropped_episodes", 0))

        telemetry_stats = {
            "episodes": 0,
            "steps": 0,
            "total_reward": 0.0,
            "last_update": "",
            "status": "unknown",
            "agent_id": agent_id,
        }

        if self._telemetry_service is not None:
            try:
                telemetry_stats = self._telemetry_service.get_run_summary(run_id)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.log_constant(
                    LOG_UI_RENDER_TABS_WARNING,
                    message=f"_build_run_summary: Failed to fetch summary ({exc})",
                    extra={"run_id": run_id},
                )

        episodes_collected = int(telemetry_stats.get("episodes", 0) or 0)
        steps_collected = int(telemetry_stats.get("steps", 0) or 0)
        total_reward = float(telemetry_stats.get("total_reward", 0.0) or 0.0)
        last_update = telemetry_stats.get("last_update", "") or ""
        status = (telemetry_stats.get("status") or "unknown").lower()

        if isinstance(widget, LiveTelemetryTab):
            buffer_episodes = len(getattr(widget, "_episode_buffer", []))
            buffer_steps = len(getattr(widget, "_step_buffer", []))
            current_episode_index = getattr(widget, "_current_episode_index", -1)
            current_step_index = getattr(widget, "_current_step_in_episode", -1)

            # Convert indices to counts (1-based) when activity observed
            live_episode_count = buffer_episodes
            if current_episode_index >= 0 and (buffer_steps > 0 or current_step_index >= 0):
                live_episode_count = max(live_episode_count, current_episode_index + 1)

            live_step_count = buffer_steps
            if current_step_index >= 0 and (buffer_steps > 0):
                live_step_count = max(live_step_count, current_step_index + 1)

            steps_collected = max(steps_collected, live_step_count)
            episodes_collected = max(episodes_collected, live_episode_count)

        is_active = status == "active"

        return RunSummary(
            run_id=run_id,
            agent_id=agent_id,
            episodes_collected=episodes_collected,
            steps_collected=steps_collected,
            dropped_episodes=dropped_episodes,
            dropped_steps=dropped_steps,
            total_reward=total_reward,
            is_active=is_active,
            last_update_timestamp=str(last_update),
        )

    def _execute_closure_choice(self, run_id: str, choice: TabClosureChoice) -> None:
        """Execute the side effects associated with the user's choice."""

        if choice == TabClosureChoice.KEEP_AND_CLOSE:
            return

        if self._telemetry_service is None:
            self.log_constant(
                LOG_UI_RENDER_TABS_WARNING,
                message="_execute_closure_choice: No telemetry service available",
                extra={"run_id": run_id, "choice": choice.value},
            )
            return

        if choice == TabClosureChoice.DELETE:
            self.log_constant(
                LOG_UI_RENDER_TABS_INFO,
                message="_execute_closure_choice: Deleting run",
                extra={"run_id": run_id},
            )
            self._telemetry_service.delete_run(run_id)
        elif choice == TabClosureChoice.ARCHIVE:
            self.log_constant(
                LOG_UI_RENDER_TABS_INFO,
                message="_execute_closure_choice: Archiving run",
                extra={"run_id": run_id},
            )
            self._telemetry_service.archive_run(run_id)

    def remove_dynamic_tabs_for_run(self, run_id: str) -> None:
        """Remove all dynamic tabs associated with a run_id."""
        try:
            self.log_constant(LOG_UI_RENDER_TABS_INFO, message=f"remove_dynamic_tabs_for_run: START (run_id={run_id})")

            tabs = self._agent_tabs.pop(run_id, {})
            self.log_constant(LOG_UI_RENDER_TABS_TRACE, message=f"remove_dynamic_tabs_for_run: Found {len(tabs)} tabs to remove")

            for i, widget in enumerate(tabs.values()):
                try:
                    self.log_constant(LOG_UI_RENDER_TABS_TRACE, message=f"remove_dynamic_tabs_for_run: Processing tab {i+1}/{len(tabs)}: {widget}")

                    # Call cleanup on LiveTelemetryTab to prevent segfaults from pending QTimer callbacks
                    if hasattr(widget, 'cleanup'):
                        try:
                            self.log_constant(LOG_UI_RENDER_TABS_TRACE, message="remove_dynamic_tabs_for_run: Calling widget.cleanup()")
                            # Type: ignore because cleanup is a custom method on LiveTelemetryTab
                            widget.cleanup()  # type: ignore[attr-defined]
                            self.log_constant(LOG_UI_RENDER_TABS_TRACE, message="remove_dynamic_tabs_for_run: widget.cleanup() completed")
                        except Exception as e:
                            self.log_constant(LOG_UI_RENDER_TABS_ERROR, message=f"remove_dynamic_tabs_for_run: Error cleaning up tab: {e}", exc_info=e)

                    idx = self.indexOf(widget)
                    self.log_constant(LOG_UI_RENDER_TABS_TRACE, message=f"remove_dynamic_tabs_for_run: Widget index: {idx}")

                    if idx >= 0:
                        self.log_constant(LOG_UI_RENDER_TABS_TRACE, message=f"remove_dynamic_tabs_for_run: Removing tab at index {idx}")
                        self.removeTab(idx)

                    self.log_constant(LOG_UI_RENDER_TABS_TRACE, message="remove_dynamic_tabs_for_run: Calling widget.deleteLater()")
                    widget.deleteLater()
                    self.log_constant(LOG_UI_RENDER_TABS_TRACE, message=f"remove_dynamic_tabs_for_run: Tab {i+1} cleanup complete")
                except Exception as e:
                    self.log_constant(LOG_UI_RENDER_TABS_ERROR, message=f"remove_dynamic_tabs_for_run: Error processing tab {i+1}: {e}", exc_info=e)

            self.log_constant(LOG_UI_RENDER_TABS_INFO, message="remove_dynamic_tabs_for_run: COMPLETE")
        except Exception as e:
            self.log_constant(LOG_UI_RENDER_TABS_ERROR, message=f"remove_dynamic_tabs_for_run: FATAL ERROR - {e}", exc_info=e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_current_game(self, game_id: GameId) -> None:
        self._current_game = game_id
        for host in self._mode_hosts.values():
            host.set_current_game(game_id)
        self._replay_tab.set_current_game(game_id)

    def display_payload(self, payload: object) -> None:
        if isinstance(payload, Mapping):
            self._update_game_from_payload(payload)
            self._raw_tab.display_from_payload(payload)

            mode = _coerce_render_mode(payload.get("mode"))
            host = self._mode_hosts.get(mode) if mode is not None else None
            if host is not None and host.supports(payload):
                host.render(payload)
                self._activate_tab(host.widget)
                return

            text = payload.get("ansi") or payload.get("text") or str(payload)
            self._raw_tab.display_plain_text(text)
            self._activate_tab(self._raw_tab.widget, enable_only=True)
            return

        if payload is None:
            self._reset_hosts()
            self._raw_tab.display_plain_text("No render payload yet.")
            self._activate_tab(self._raw_tab.widget, enable_only=True)
            return

        self._reset_hosts()
        self._raw_tab.display_plain_text(str(payload))
        self._activate_tab(self._raw_tab.widget, enable_only=True)

    def refresh_replays(self) -> None:
        self._replay_tab.refresh()

    def on_episode_finished(self) -> None:
        self._replay_tab.refresh()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _activate_tab(self, widget: QtWidgets.QWidget, *, enable_only: bool = False) -> None:
        index = self.indexOf(widget)
        if index == -1:
            return
        self.setTabEnabled(index, True)
        if not enable_only:
            self.setCurrentIndex(index)

    def _create_host(
        self,
        mode: RenderMode,
        *,
        parent: QtWidgets.QWidget | None,
    ) -> _RendererHost | None:
        if not self._registry.is_registered(mode):
            return None
        strategy = self._registry.create(mode, parent)
        host = _RendererHost(strategy)
        self._mode_hosts[mode] = host
        if self._current_game is not None:
            host.set_current_game(self._current_game)
        return host

    def _update_game_from_payload(self, payload: Mapping[str, object]) -> None:
        raw_game = payload.get("game_id")
        if raw_game is None:
            return
        try:
            game_id = raw_game if isinstance(raw_game, GameId) else GameId(str(raw_game))
        except ValueError:
            return
        self._current_game = game_id
        for host in self._mode_hosts.values():
            host.set_current_game(game_id)
        self._replay_tab.set_current_game(game_id)

    def _reset_hosts(self) -> None:
        for host in self._mode_hosts.values():
            host.reset()


@dataclass(slots=True)
class _RawTab:
    widget: QtWidgets.QPlainTextEdit

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        editor = QtWidgets.QPlainTextEdit(parent)
        editor.setReadOnly(True)
        editor.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.widget = editor

    def display_from_payload(self, payload: Mapping[str, Any]) -> None:
        ansi = payload.get("ansi")
        if ansi:
            self.widget.setPlainText(_strip_ansi_codes(ansi))

    def display_plain_text(self, text: str) -> None:
        self.widget.setPlainText(text)


class _RendererHost:
    """Wrapper that binds a renderer strategy to the owning widget."""

    def __init__(self, strategy: RendererStrategy) -> None:
        self._strategy = strategy
        self.widget = strategy.widget
        self.mode = strategy.mode
        self._game_id: GameId | None = None

    def set_current_game(self, game_id: GameId | None) -> None:
        self._game_id = game_id

    def supports(self, payload: Mapping[str, object]) -> bool:
        return self._strategy.supports(payload)

    def render(self, payload: Mapping[str, object], *, game_id: GameId | None = None) -> None:
        if game_id is not None:
            self._game_id = game_id
        context = RendererContext(game_id=self._game_id)
        self._strategy.render(payload, context=context)

    def reset(self) -> None:
        self._strategy.reset()


class _ReplayPreview(QtWidgets.QStackedWidget):
    """Mini viewer that mirrors replay frames for quick inspection."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        registry: RendererRegistry | None = None,
    ) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self._registry = registry or create_default_renderer_registry()
        self._mode_hosts: dict[RenderMode, _RendererHost] = {}
        self._mode_indices: dict[RenderMode, int] = {}
        self._current_game: GameId | None = None

        self._text_view = QtWidgets.QPlainTextEdit(self)
        self._text_view.setReadOnly(True)
        self._text_view.setMinimumHeight(120)
        self._placeholder = QtWidgets.QLabel("No render data available for this step.", self)
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)

        self._placeholder_index = self.addWidget(self._placeholder)

        grid_host = self._create_host(RenderMode.GRID)
        if grid_host is not None:
            self._grid_index = self.addWidget(grid_host.widget)
            self._mode_indices[RenderMode.GRID] = self._grid_index
        else:
            self._grid_index = -1

        video_host = self._create_host(RenderMode.RGB_ARRAY)
        if video_host is not None:
            self._video_index = self.addWidget(video_host.widget)
            self._mode_indices[RenderMode.RGB_ARRAY] = self._video_index
        else:
            self._video_index = -1

        self._text_index = self.addWidget(self._text_view)

        self.setCurrentIndex(self._placeholder_index)

    def set_current_game(self, game: GameId | None) -> None:
        self._current_game = game
        for host in self._mode_hosts.values():
            host.set_current_game(game)

    def clear(self) -> None:
        for host in self._mode_hosts.values():
            host.reset()
        self.setCurrentIndex(self._placeholder_index)
        self._text_view.clear()

    def display(self, payload: Any, game: GameId | None = None) -> None:
        if game is not None:
            self.set_current_game(game)
        if payload is None:
            self.clear()
            return
        if isinstance(payload, Mapping):
            mode = _coerce_render_mode(payload.get("mode"))
            host = self._mode_hosts.get(mode) if mode is not None else None
            if host is not None and host.supports(payload):
                host.render(payload, game_id=game or self._current_game)
                index = self._mode_indices.get(host.mode, -1)
                if index != -1:
                    self.setCurrentIndex(index)
                return
            ansi = payload.get("ansi")
            if isinstance(ansi, str) and ansi:
                self._text_view.setPlainText(_strip_ansi_codes(ansi))
                self.setCurrentIndex(self._text_index)
                return
        try:
            self._text_view.setPlainText(str(payload))
        except Exception:
            self._text_view.setPlainText("Unsupported render payload")
        self.setCurrentIndex(self._text_index)

    def _create_host(self, mode: RenderMode) -> _RendererHost | None:
        if not self._registry.is_registered(mode):
            return None
        strategy = self._registry.create(mode, self)
        host = _RendererHost(strategy)
        self._mode_hosts[mode] = host
        if self._current_game is not None:
            host.set_current_game(self._current_game)
        return host


class _ReplayTab(QtWidgets.QWidget):
    """Displays recent episodes from telemetry for quick replay selection."""

    _telemetry: TelemetryService | None
    _loader: EpisodeReplayLoader | None
    _current_game: GameId | None
    _load_button: QtWidgets.QPushButton
    _delete_button: QtWidgets.QPushButton
    _clear_button: QtWidgets.QPushButton
    _order_button: QtWidgets.QPushButton
    _episodes: QtWidgets.QTableWidget
    _placeholder: QtWidgets.QLabel
    _details_group: QtWidgets.QGroupBox
    _episode_summary: QtWidgets.QLabel
    _preview: _ReplayPreview
    _slider: QtWidgets.QSlider
    _step_label: QtWidgets.QLabel
    _step_view: QtWidgets.QPlainTextEdit
    _current_replay: Optional[EpisodeReplay]
    _current_replay_game: GameId | None

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        telemetry_service: TelemetryService | None = None,
        renderer_registry: RendererRegistry | None = None,
    ) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self._telemetry = telemetry_service
        self._loader = EpisodeReplayLoader(telemetry_service) if telemetry_service else None
        self._current_game = None
        self._sort_descending = True
        self._renderer_registry = renderer_registry or create_default_renderer_registry()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._details_group = QtWidgets.QGroupBox("Episode Playback", self)
        details_layout = QtWidgets.QVBoxLayout(self._details_group)
        details_layout.setContentsMargins(8, 8, 8, 8)

        self._episode_summary = QtWidgets.QLabel("Select an episode to load its replay.")
        self._episode_summary.setWordWrap(True)
        details_layout.addWidget(self._episode_summary)

        self._preview = _ReplayPreview(self._details_group, registry=self._renderer_registry)
        self._preview.setMinimumHeight(220)
        details_layout.addWidget(self._preview)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self._details_group)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(1)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        details_layout.addWidget(self._slider)

        self._step_label = QtWidgets.QLabel("Step 0 / 0")
        details_layout.addWidget(self._step_label)

        self._step_view = QtWidgets.QPlainTextEdit(self._details_group)
        self._step_view.setReadOnly(True)
        self._step_view.setMinimumHeight(160)
        details_layout.addWidget(self._step_view)

        layout.addWidget(self._details_group)

        footer = QtWidgets.QHBoxLayout()
        footer_label = QtWidgets.QLabel("Recent Episodes")
        footer.addWidget(footer_label)
        self._order_button = QtWidgets.QPushButton(self)
        self._order_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._order_button.clicked.connect(self._toggle_sort_order)
        footer.addWidget(self._order_button)
        footer.addStretch(1)
        self._load_button = QtWidgets.QPushButton("Load Replay")
        self._load_button.clicked.connect(self._load_selected_episode)
        self._load_button.setEnabled(False)
        footer.addWidget(self._load_button)
        self._copy_button = QtWidgets.QPushButton("Copy to Clipboard")
        self._copy_button.clicked.connect(self._copy_table_to_clipboard)
        self._copy_button.setEnabled(False)
        footer.addWidget(self._copy_button)
        self._delete_button = QtWidgets.QPushButton("Delete Selected")
        self._delete_button.clicked.connect(self._delete_selected_episode)
        self._delete_button.setEnabled(False)
        footer.addWidget(self._delete_button)
        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        footer.addWidget(refresh_button)
        self._clear_button = QtWidgets.QPushButton("Clear All")
        self._clear_button.clicked.connect(self._clear_all_episodes)
        self._clear_button.setEnabled(False)
        footer.addWidget(self._clear_button)
        layout.addLayout(footer)

        self._update_order_button()

        self._episodes = QtWidgets.QTableWidget(0, 7, self)
        self._episodes.setHorizontalHeaderLabels([
            "Seed",
            "Episode",
            "Game",
            "Steps",
            "Reward",
            "Outcome",
            "Timestamp",
        ])
        header = self._episodes.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self._episodes.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._episodes.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._episodes.itemSelectionChanged.connect(self._on_episode_selection_changed)
        self._episodes.itemDoubleClicked.connect(lambda *_: self._load_selected_episode())
        layout.addWidget(self._episodes, 1)

        self._placeholder = QtWidgets.QLabel(
            "Telemetry playback data will appear here once episodes are recorded.",
            self,
        )
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        layout.addWidget(self._placeholder)

        self._update_placeholder_visibility()
        self._current_replay = None
        self._current_replay_game = None

    def set_current_game(self, game_id: GameId | None) -> None:
        self._current_game = game_id
        self._preview.set_current_game(game_id)

    def refresh(self) -> None:
        records = self._fetch_recent_episodes()
        records.sort(
            key=lambda record: record.get("timestamp_sort", datetime.min),
            reverse=self._sort_descending,
        )
        self._episodes.setRowCount(0)
        for display_index, record in enumerate(records, start=1):
            row = self._episodes.rowCount()
            self._episodes.insertRow(row)
            episode_label = record.get("episode_index")
            episode_display = (
                str(episode_label)
                if episode_label is not None
                else str(display_index)
            )
            display_values = [
                str(record["seed"]),
                episode_display,
                record.get("game", "—"),
                str(record["steps"]),
                str(record["reward"]),
                str(record["terminated"]),
                str(record["timestamp"]),
            ]
            for column, value in enumerate(display_values):
                item = QtWidgets.QTableWidgetItem(value)
                if column == 1:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, record["episode_id"])
                if column == 4:  # reward column alignment
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self._episodes.setItem(row, column, item)
        self._update_placeholder_visibility()
        self._on_episode_selection_changed()
        if self._episodes.rowCount() == 0:
            self._clear_replay_details()

    def _copy_table_to_clipboard(self) -> None:
        if self._episodes.rowCount() == 0:
            return
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is None:
            return
        headers: List[str] = []
        for column_index in range(self._episodes.columnCount()):
            header_item = self._episodes.horizontalHeaderItem(column_index)
            headers.append(header_item.text() if header_item is not None else "")
        rows: List[str] = ["\t".join(headers)]
        for row_index in range(self._episodes.rowCount()):
            row_values: List[str] = []
            for column_index in range(self._episodes.columnCount()):
                item = self._episodes.item(row_index, column_index)
                row_values.append(item.text() if item is not None else "")
            rows.append("\t".join(row_values))
        clipboard.setText("\n".join(rows))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fetch_recent_episodes(self) -> List[dict[str, Any]]:
        if self._telemetry is None:
            return []
        episodes = [
            ep
            for ep in self._telemetry.recent_episodes()
            if self._is_human_episode(ep)
        ]
        return [self._format_episode_row(ep) for ep in episodes]

    def _format_episode_row(self, episode: EpisodeRollup) -> dict[str, Any]:
        seed_value, episode_index, game_label = self._parse_episode_metadata(episode.metadata)
        timestamp = episode.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        display_ts = timestamp.astimezone(timezone.utc)
        return {
            "episode_id": episode.episode_id,
            "episode_index": episode_index,
            "seed": seed_value,
            "game": game_label,
            "steps": str(episode.steps),
            "reward": f"{episode.total_reward:.2f}",
            "terminated": self._termination_label(episode),
            "timestamp": display_ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "timestamp_sort": display_ts,
        }

    def _parse_episode_metadata(
        self, metadata: Any
    ) -> tuple[str, int | None, str]:
        if not isinstance(metadata, Mapping):
            return "—", None, "—"
        seed_value = metadata.get("seed")
        episode_index_raw = metadata.get("episode_index")
        game_label = self._resolve_game_label(metadata.get("game_id"))
        seed_display = str(seed_value) if seed_value is not None else "—"
        episode_index = None
        if episode_index_raw is not None:
            try:
                episode_index = int(episode_index_raw)
            except (TypeError, ValueError):
                episode_index = None
        return seed_display, episode_index, game_label

    def _resolve_game_label(self, raw_game: Any) -> str:
        if isinstance(raw_game, GameId):
            return raw_game.value
        if isinstance(raw_game, str):
            try:
                return GameId(raw_game).value
            except ValueError:
                return raw_game
        return "—"

    @staticmethod
    def _is_human_episode(episode: EpisodeRollup) -> bool:
        metadata = episode.metadata
        if not isinstance(metadata, Mapping):
            return True
        mode = metadata.get("control_mode") or metadata.get("controlMode")
        if mode is None:
            return True
        if isinstance(mode, ControlMode):
            return mode is ControlMode.HUMAN_ONLY or mode.value == "human_only"
        normalized = str(mode).strip().lower()
        return normalized in {"human", "human_only"}

    @staticmethod
    def _termination_label(episode: EpisodeRollup) -> str:
        if episode.terminated:
            return "Yes"
        if episode.truncated:
            return "Aborted"
        return "No"

    def _update_placeholder_visibility(self) -> None:
        has_rows = self._episodes.rowCount() > 0
        self._episodes.setVisible(has_rows)
        self._placeholder.setVisible(not has_rows)
        self._details_group.setVisible(has_rows)
        self._clear_button.setEnabled(has_rows and self._telemetry is not None)
        self._copy_button.setEnabled(has_rows)
        if not has_rows:
            self._load_button.setEnabled(False)
            self._delete_button.setEnabled(False)

    def _update_order_button(self) -> None:
        if self._sort_descending:
            self._order_button.setText("Newest ↓")
            self._order_button.setToolTip("Show oldest episodes first")
        else:
            self._order_button.setText("Oldest ↑")
            self._order_button.setToolTip("Show newest episodes first")

    def _toggle_sort_order(self) -> None:
        self._sort_descending = not self._sort_descending
        self._update_order_button()
        self.refresh()

    def _on_episode_selection_changed(self) -> None:
        indexes = self._episodes.selectionModel()
        has_selection = bool(indexes and indexes.hasSelection())
        self._load_button.setEnabled(has_selection)
        self._delete_button.setEnabled(has_selection and self._telemetry is not None)

    def _load_selected_episode(self) -> None:
        if self._loader is None:
            return
        selection = self._episodes.selectionModel()
        if selection is None or not selection.hasSelection():
            return
        row = selection.selectedRows()[0].row()
        episode_id_item = self._episodes.item(row, 1)
        if episode_id_item is None:
            return
        episode_id = episode_id_item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(episode_id, str):
            episode_id = episode_id_item.text()
        replay = self._loader.load_episode(episode_id)
        if replay is None:
            self._episode_summary.setText("No telemetry data available for the selected episode.")
            self._slider.setEnabled(False)
            self._slider.setMaximum(0)
            self._step_view.setPlainText("")
            self._preview.clear()
            self._current_replay = None
            self._current_replay_game = None
            return
        self._current_replay = replay
        self._current_replay_game = self._resolve_game_from_metadata(replay.rollup.metadata)
        if self._current_replay_game is not None:
            self._preview.set_current_game(self._current_replay_game)
        self._episode_summary.setText(
            f"Episode {replay.episode_id}\nTotal reward: {replay.total_reward:.2f}\nSteps: {len(replay.steps)}"
        )
        self._slider.setEnabled(True)
        self._slider.blockSignals(True)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, len(replay.steps) - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._display_step(0)

    def _delete_selected_episode(self) -> None:
        if self._telemetry is None:
            return
        selection = self._episodes.selectionModel()
        if selection is None or not selection.hasSelection():
            return
        row = selection.selectedRows()[0].row()
        episode_item = self._episodes.item(row, 1)
        if episode_item is None:
            return
        episode_id = episode_item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(episode_id, str):
            episode_id = episode_item.text()
        with modal_busy_indicator(
            self,
            title="Deleting episode",
            message="Persisted data is being removed…",
        ):
            self._telemetry.delete_episode(episode_id)
        self.refresh()
        self._clear_replay_details()

    def _clear_all_episodes(self) -> None:
        if self._telemetry is None:
            return
        with modal_busy_indicator(
            self,
            title="Clearing telemetry",
            message="Flushing and deleting recorded episodes…",
        ):
            self._telemetry.clear_all_episodes()
        self.refresh()
        self._clear_replay_details()

    def _on_slider_changed(self, value: int) -> None:
        self._display_step(value)

    def _display_step(self, index: int) -> None:
        if self._current_replay is None or not self._current_replay.steps:
            self._step_label.setText("Step 0 / 0")
            self._step_view.setPlainText("")
            self._preview.clear()
            return
        index = max(0, min(index, len(self._current_replay.steps) - 1))
        step = self._current_replay.steps[index]
        self._step_label.setText(f"Step {index + 1} / {len(self._current_replay.steps)}")
        self._step_view.setPlainText(self._format_step(step))
        self._preview.display(step.render_payload, self._current_replay_game or self._current_game)

    def _clear_replay_details(self) -> None:
        self._episode_summary.setText("Select an episode to load its replay.")
        self._slider.setEnabled(False)
        self._slider.setMaximum(0)
        self._step_label.setText("Step 0 / 0")
        self._step_view.setPlainText("")
        self._preview.clear()
        self._current_replay = None
        self._current_replay_game = None

    def _format_step(self, step: StepRecord) -> str:
        observation_preview = self._summarise_value(step.observation)
        info_preview = self._summarise_value(step.info)
        return (
            f"Episode: {step.episode_id}\n"
            f"Step Index: {step.step_index}\n"
            f"Reward: {step.reward:.4f}\n"
            f"Terminated: {step.terminated}\n"
            f"Truncated: {step.truncated}\n"
            f"Timestamp: {step.timestamp.isoformat()}\n\n"
            f"Observation:\n{observation_preview}\n\n"
            f"Info:\n{info_preview}"
        )

    @staticmethod
    def _resolve_game_from_metadata(metadata: Any) -> GameId | None:
        if not isinstance(metadata, Mapping):
            return None
        raw_game = metadata.get("game_id")
        if isinstance(raw_game, GameId):
            return raw_game
        if isinstance(raw_game, str):
            try:
                return GameId(raw_game)
            except ValueError:
                return None
        return None

    @staticmethod
    def _summarise_value(value: Any, *, max_length: int = 800) -> str:
        try:
            representation = repr(value)
        except Exception:
            representation = str(type(value))
        if len(representation) > max_length:
            return representation[: max_length - 3] + "..."
        return representation


def _strip_ansi_codes(text: str) -> str:
    import re

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)
    action_indicator = re.compile(r"^\s*\([A-Za-z]+\)\s*\n?", re.MULTILINE)
    text = action_indicator.sub("", text)
    return text.strip()


def _coerce_render_mode(value: object) -> RenderMode | None:
    if isinstance(value, RenderMode):
        return value
    if isinstance(value, str):
        try:
            return RenderMode(value)
        except ValueError:
            return None
    return None


__all__ = ["RenderTabs"]
