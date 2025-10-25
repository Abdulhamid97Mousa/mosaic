# /home/hamid/Desktop/Projects/GUI_BDI_RL/gym_gui/ui/widgets/live_telemetry_tab.py
"""Live telemetry tab widget for displaying streamed run data."""

from __future__ import annotations

from collections import deque
import json
import logging
from typing import Any, Deque, Optional

from qtpy import QtCore, QtWidgets

from PyQt6.QtCore import pyqtSlot  # type: ignore[attr-defined]

_QUEUED_CONNECTION = QtCore.Qt.ConnectionType.QueuedConnection  # type: ignore[attr-defined]


from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_LIVE_TAB_TRACE,
    LOG_UI_LIVE_TAB_INFO,
    LOG_UI_LIVE_TAB_WARNING,
    LOG_UI_LIVE_TAB_ERROR,
)
from gym_gui.ui.widgets.base_telemetry_tab import BaseTelemetryTab
from gym_gui.rendering import RendererRegistry, create_default_renderer_registry, RendererContext
from gym_gui.core.enums import GameId, RenderMode
from gym_gui.telemetry.rendering_speed_regulator import RenderingSpeedRegulator


_LOGGER = logging.getLogger(__name__)


class LiveTelemetryTab(BaseTelemetryTab, LogConstantMixin):
    """Displays live telemetry stream for a specific (run_id, agent_id) pair with live rendering."""

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        game_id: Optional[GameId] = None,
        buffer_size: int = 100,
        episode_buffer_size: int = 100,
        render_throttle_interval: int = 1,
        render_delay_ms: int = 100,
        live_render_enabled: bool = True,
        renderer_registry: Optional[RendererRegistry] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        # Store render delay for later use in _build_ui()
        self._logger = _LOGGER
        self._render_delay_ms = render_delay_ms
        self._render_regulator: Optional[RenderingSpeedRegulator] = None
        self._live_render_enabled = live_render_enabled

        self._step_buffer: Deque[Any] = deque(maxlen=buffer_size)
        self._episode_buffer: Deque[Any] = deque(maxlen=episode_buffer_size)
        self._dropped_steps = 0
        self._dropped_episodes = 0
        self._renderer_registry = renderer_registry or create_default_renderer_registry()
        self._renderer_strategy: Optional[Any] = None
        self._current_game: Optional[GameId] = game_id  # Initialize from parameter
        self._render_throttle_counter = 0
        self._render_throttle_interval = max(1, render_throttle_interval)  # Render every Nth step (from config)
        self._last_render_payload: Optional[dict[str, Any]] = None
        self._is_destroyed = False  # Track if widget is being destroyed
        self._pending_render_timer_id: Optional[int] = None  # Track pending QTimer for cleanup
        
        # COUNTER TRACKING: Independent metrics display (not based on buffer size)
        self._current_episode_index: int = 0  # Actual episode being trained
        self._current_step_in_episode: int = 0  # Step within current episode
        self._previous_episode_index: int = -1  # Detect episode boundaries

        # CRITICAL: Call super().__init__() which calls _build_ui()
        # _build_ui() will create the regulator after self is a valid Qt object
        super().__init__(run_id, agent_id, parent=parent)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Use inherited header builder and extend it
        header = self._build_header()
        self._stats_label = QtWidgets.QLabel("Steps: 0 | Episodes: 0")
        header.addWidget(self._stats_label)
        layout.addLayout(header)

        # 1) LIVE RENDERING PANEL (at top)
        if self._live_render_enabled:
            render_group = QtWidgets.QGroupBox("Live Rendering", self)
            render_group.setMinimumHeight(300)
            render_layout = QtWidgets.QVBoxLayout(render_group)
            self._render_container = QtWidgets.QWidget(render_group)
            self._render_container.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding
            )
            self._render_layout = QtWidgets.QVBoxLayout(self._render_container)
            self._render_layout.setContentsMargins(0, 0, 0, 0)
            self._render_layout.setSpacing(0)
            self._render_placeholder = QtWidgets.QLabel("Waiting for render data...")
            self._render_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self._render_placeholder.setStyleSheet("color: #999; font-style: italic;")
            self._render_layout.addWidget(self._render_placeholder)
            render_layout.addWidget(self._render_container)
            render_layout.setContentsMargins(0, 0, 0, 0)
            render_layout.setSpacing(0)
            layout.addWidget(render_group, 3)
        else:
            disabled_group = QtWidgets.QGroupBox("Live Rendering", self)
            disabled_layout = QtWidgets.QVBoxLayout(disabled_group)
            notice = QtWidgets.QLabel(
                "Live rendering disabled for this run. Telemetry tables below remain active."
            )
            notice.setStyleSheet("color: #607D8B; font-style: italic;")
            notice.setWordWrap(True)
            disabled_layout.addWidget(notice)
            layout.addWidget(disabled_group)
            self._render_container = None
            self._render_layout = None
            self._render_placeholder = None

        # 2) TELEMETRY SECTIONS (vertical splitter for episodes and steps)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_build_ui: Vertical splitter created for episodes and steps")
        layout.addWidget(splitter, 1)  # Stretch factor 1 (decreased from 2)
        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_build_ui: Splitter added to main layout with stretch factor 1")

        # 2a) Telemetry Recent Episodes (above steps)
        episodes_group = QtWidgets.QGroupBox("Telemetry Recent Episodes", self)
        episodes_layout = QtWidgets.QVBoxLayout(episodes_group)

        # Episodes toolbar with Copy button
        episodes_toolbar = QtWidgets.QHBoxLayout()
        episodes_toolbar.addStretch()
        self._episodes_copy_button = QtWidgets.QPushButton("Copy to Clipboard")
        self._episodes_copy_button.clicked.connect(self._copy_episodes_table_to_clipboard)
        self._episodes_copy_button.setEnabled(False)
        episodes_toolbar.addWidget(self._episodes_copy_button)
        episodes_layout.addLayout(episodes_toolbar)

        self._episodes_table = QtWidgets.QTableWidget(0, 11, episodes_group)
        self._episodes_table.setHorizontalHeaderLabels([
            "Timestamp",
            "Episode",
            "Steps",
            "Reward",
            "Terminated",
            "Truncated",
            "Seed",
            "Episode Seed",
            "Mode",
            "Game",
            "Outcome",
        ])
        header = self._episodes_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        episodes_layout.addWidget(self._episodes_table)
        splitter.addWidget(episodes_group)

        # 2b) Telemetry Recent Steps (below episodes)
        steps_group = QtWidgets.QGroupBox("Telemetry Recent Steps", self)
        steps_layout = QtWidgets.QVBoxLayout(steps_group)

        # Steps toolbar with Copy button
        steps_toolbar = QtWidgets.QHBoxLayout()
        steps_toolbar.addStretch()
        self._steps_copy_button = QtWidgets.QPushButton("Copy to Clipboard")
        self._steps_copy_button.clicked.connect(self._copy_steps_table_to_clipboard)
        self._steps_copy_button.setEnabled(False)
        steps_toolbar.addWidget(self._steps_copy_button)
        steps_layout.addLayout(steps_toolbar)

        self._steps_table = QtWidgets.QTableWidget(0, 10, steps_group)
        self._steps_table.setHorizontalHeaderLabels([
            "Timestamp",
            "Episode",
            "Step",
            "Episode Seed",
            "Action",
            "Reward",
            "Terminated",
            "Truncated",
            "Observation",
            "Info",
        ])
        header = self._steps_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        steps_layout.addWidget(self._steps_table)
        splitter.addWidget(steps_group)

        splitter.setStretchFactor(0, 1)  # Episodes (reduced from 1)
        splitter.setStretchFactor(1, 1)  # Steps (reduced from 2)
        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_build_ui: Splitter stretch factors set - Episodes: 1, Steps: 1")

        # Footer with overflow stats
        self._overflow_label = QtWidgets.QLabel("")
        self._overflow_label.setStyleSheet("color: #d32f2f;")
        layout.addWidget(self._overflow_label)

        if self._live_render_enabled:
            self._render_regulator = RenderingSpeedRegulator(
                render_delay_ms=self._render_delay_ms,
                max_queue_size=32,
                parent=self
            )
            self._render_regulator.payload_ready.connect(self._try_render_visual)
            self._render_regulator.start()
        else:
            self._render_regulator = None

    def set_render_throttle_interval(self, interval: int) -> None:
        """Set the render throttle interval (render every Nth step).

        Args:
            interval: Render every Nth step (1=every step, 2=every 2nd step, etc.)
        """
        self._render_throttle_interval = max(1, interval)
        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"Render throttle interval set to {self._render_throttle_interval}")

    def set_render_delay(self, delay_ms: int) -> None:
        """Set the rendering delay (time between visual renders).

        Args:
            delay_ms: Delay in milliseconds (e.g., 100ms = 10 FPS, 50ms = 20 FPS)
        """
        if self._render_regulator is not None:
            self._render_regulator.set_render_delay(delay_ms)
            self.log_constant(LOG_UI_LIVE_TAB_INFO, message=f"Render delay set to {delay_ms}ms")

    def add_step(self, payload: Any) -> None:
        """Add a step to the buffer and update display lazily."""
        try:
            step_index = payload.get("step_index", "?") if isinstance(payload, dict) else getattr(payload, "step_index", "?")
            self.log_constant(LOG_UI_LIVE_TAB_INFO, message=f"add_step: START (step={step_index}, is_destroyed={self._is_destroyed})")

            if self._is_destroyed:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="add_step: Widget is destroyed, skipping")
                return

            # CRITICAL: Extract episode_index and step_index to track current episode/step
            # This is separate from buffer size tracking
            episode_index_raw = payload.get("episode_index", 0) if isinstance(payload, dict) else getattr(payload, "episode_index", 0)
            step_index_raw = payload.get("step_index", 0) if isinstance(payload, dict) else getattr(payload, "step_index", 0)
            
            try:
                episode_idx = int(episode_index_raw) if episode_index_raw is not None else 0
                step_idx = int(step_index_raw) if step_index_raw is not None else 0
            except (TypeError, ValueError):
                episode_idx = 0
                step_idx = 0
            
            # Detect episode boundary: when episode_index changes from previous
            if episode_idx != self._previous_episode_index:
                self._current_step_in_episode = 0  # Reset step counter at episode boundary
                self._previous_episode_index = episode_idx
                self.log_constant(LOG_UI_LIVE_TAB_INFO, message=f"add_step: Episode boundary detected (episode {episode_idx}), resetting step counter")
            
            # Update current metrics
            self._current_episode_index = episode_idx
            self._current_step_in_episode = step_idx

            self._step_buffer.append(payload)
            self._update_stats()

            # Schedule step rendering on the GUI thread using QMetaObject.invokeMethod.
            # This works from any thread (including background threads without Qt event loops).
            self._schedule_step_render(payload)
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="add_step: Table update scheduled (no throttle)")

            # Submit payload to rendering speed regulator for decoupled visual rendering
            # Visual rendering happens at configurable rate (e.g., 10 FPS) independent of table updates
            if self._render_regulator is not None:
                if isinstance(payload, dict):
                    self._render_regulator.submit_payload(payload)
                    self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"add_step: Payload submitted to regulator (queue_size={self._render_regulator.get_queue_size()})")
                else:
                    # Convert object to dict if needed
                    try:
                        payload_dict = dict(payload) if hasattr(payload, '__dict__') else {"payload": payload}
                        self._render_regulator.submit_payload(payload_dict)
                        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"add_step: Object payload converted and submitted to regulator")
                    except Exception as e:
                        self.log_constant(
                            LOG_UI_LIVE_TAB_WARNING,
                            message=f"add_step: Failed to submit payload to regulator: {e}",
                            exc_info=e,
                        )

            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="add_step: COMPLETE")
        except Exception as e:
            self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=f"add_step: ERROR - {e}", exc_info=e)

    def _safe_process_deferred_render(self) -> None:
        """Safe wrapper for deferred rendering that checks if widget is destroyed.

        This prevents segfaults when QTimer callbacks try to access a deleted widget.
        """
        try:
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_safe_process_deferred_render: START")
            # Check if widget is being destroyed BEFORE accessing any attributes
            if not hasattr(self, '_is_destroyed') or self._is_destroyed:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_safe_process_deferred_render: Widget is destroyed, skipping")
                return

            # Now safe to call the actual render method
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_safe_process_deferred_render: Calling _process_deferred_render")
            self._process_deferred_render()
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_safe_process_deferred_render: COMPLETE")
        except Exception as e:
            # Log errors during deferred render
            self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=f"_safe_process_deferred_render: ERROR - {e}", exc_info=e)

    def _process_deferred_render(self) -> None:
        """Process deferred rendering without lambda capture issues."""
        try:
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_process_deferred_render: START (is_destroyed={self._is_destroyed}, has_payload={self._last_render_payload is not None})")

            # Safety check: if widget is being destroyed, skip rendering
            if self._is_destroyed:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_process_deferred_render: Widget is destroyed, skipping")
                self._last_render_payload = None
                return

            if self._last_render_payload is None:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_process_deferred_render: No payload, skipping")
                return

            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_process_deferred_render: Calling _try_render_visual")
            self._try_render_visual(self._last_render_payload)
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_process_deferred_render: COMPLETE")
        except Exception as e:
            self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=f"_process_deferred_render: ERROR - {e}", exc_info=e)
        finally:
            self._last_render_payload = None

    def _preview(self, s: str, n: int = 50) -> str:
        """Safely preview string truncated to n chars."""
        if not s:
            return ""
        return (s[:n] + "…") if len(s) > n else s

    def _render_latest_step(self, payload: Any) -> None:
        """Render only the latest step to the steps table."""
        # Helper to extract fields from dict or object
        def _get_field(obj: Any, *field_names: str, default: Any = None) -> Any:
            for field in field_names:
                if isinstance(obj, dict):
                    val = obj.get(field)
                    if val is not None:
                        return val
                else:
                    val = getattr(obj, field, None)
                    if val is not None:
                        return val
            return default

        # Extract and convert to proper types
        episode_index_raw = _get_field(payload, "episode_index", "episode", default=0)
        step_index_raw = _get_field(payload, "step_index", "step", default=0)

        # Convert to int (handles both string and int values from protobuf)
        try:
            episode_index = int(episode_index_raw) if episode_index_raw is not None else 0
        except (TypeError, ValueError):
            episode_index = 0

        try:
            step_index = int(step_index_raw) if step_index_raw is not None else 0
        except (TypeError, ValueError):
            step_index = 0

        reward_raw = _get_field(payload, "reward", default=0.0)
        # Ensure reward is a float for formatting
        try:
            reward = float(reward_raw) if reward_raw is not None else 0.0
        except (TypeError, ValueError):
            reward = 0.0
        
        terminated = _get_field(payload, "terminated", default=False)
        truncated = _get_field(payload, "truncated", default=False)

        # Extract metadata to get seed for display_episode calculation
        metadata_json = _get_field(payload, "metadataJson", "metadata_json", default="")
        metadata: dict[str, Any] = {}
        if isinstance(metadata_json, str) and metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                metadata = {}
        elif isinstance(metadata_json, dict):
            metadata = metadata_json

        # CRITICAL FIX: Episode number should equal seed + episode_index
        # If seed=39 and episode_index=0, display episode as 39
        seed = metadata.get("seed", "—")
        episode_seed = metadata.get("episode_seed", "—")
        try:
            seed_int = int(seed) if seed != "—" else 0
            display_episode = seed_int + int(episode_index)
        except (TypeError, ValueError):
            display_episode = int(episode_index)

        # Extract timestamp - try multiple field names
        timestamp = _get_field(payload, "timestamp", "ts", default=None)

        # Format timestamp for display
        if timestamp:
            # If it's already a string, use it directly
            if isinstance(timestamp, str):
                ts_display = timestamp[:19]  # Truncate to seconds precision
            else:
                # If it's a protobuf Timestamp object
                ts_display = str(timestamp)[:19]
        else:
            ts_display = "—"

        # Get action display - action_json contains the serialized action
        action_json = _get_field(payload, "action_json", default="")
        if action_json:
            action_display = self._preview(action_json)
        else:
            action_display = "—"

        # Get observation display - observation_json contains the serialized observation
        # Use larger preview size (200 chars) for observation to show more detail
        observation_json = _get_field(payload, "observation_json", default="")
        if observation_json:
            observation_display = self._preview(observation_json, n=200)
        else:
            observation_display = "—"

        # Get info display - info is typically in the observation_json or render_hint_json
        # For now, try to extract from render_hint_json or info field
        # Use larger preview size (200 chars) for info to show more detail
        render_hint_json = _get_field(payload, "render_hint_json", default="")
        info_value = _get_field(payload, "info", default=None)

        if render_hint_json:
            info_display = self._preview(render_hint_json, n=200)
        elif info_value:
            try:
                info_display = self._preview(json.dumps(info_value), n=200)
            except (TypeError, ValueError):
                info_display = self._preview(str(info_value), n=200)
        else:
            info_display = "—"

        # Add row to steps table
        row = self._steps_table.rowCount()
        self._steps_table.insertRow(row)

        # Keep only last 100 rows
        if row > 100:
            self._steps_table.removeRow(0)
            row = self._steps_table.rowCount() - 1

        # Populate row
        self._steps_table.setItem(row, 0, QtWidgets.QTableWidgetItem(ts_display))
        # CRITICAL FIX: Use display_episode (seed + episode_index) instead of raw episode_index
        self._steps_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(display_episode)))
        self._steps_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(int(step_index))))
        self._steps_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(episode_seed)))
        self._steps_table.setItem(row, 4, QtWidgets.QTableWidgetItem(action_display))
        self._steps_table.setItem(row, 5, QtWidgets.QTableWidgetItem(f"{reward:+.3f}"))
        self._steps_table.setItem(row, 6, QtWidgets.QTableWidgetItem(str(terminated)))
        self._steps_table.setItem(row, 7, QtWidgets.QTableWidgetItem(str(truncated)))
        self._steps_table.setItem(row, 8, QtWidgets.QTableWidgetItem(observation_display))
        self._steps_table.setItem(row, 9, QtWidgets.QTableWidgetItem(info_display))

        # Scroll to bottom
        self._steps_table.scrollToBottom()

        # Enable Copy button when data is available
        if self._steps_table.rowCount() > 0:
            self._steps_copy_button.setEnabled(True)

    def _try_render_visual(self, payload: Any) -> None:
        """Try to render visual representation from payload (throttled)."""
        try:
            if not self._live_render_enabled:
                return
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_try_render_visual: START (is_destroyed={self._is_destroyed}, renderer_strategy={self._renderer_strategy is not None})")

            # Skip if renderer not initialized yet
            if self._renderer_strategy is None:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_try_render_visual: Renderer not initialized, attempting to create")
                # Try to initialize on first call
                if not self._renderer_registry.is_registered(RenderMode.GRID):
                    self.log_constant(LOG_UI_LIVE_TAB_WARNING, message="_try_render_visual: GRID renderer not registered in registry")
                    return
                try:
                    self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_try_render_visual: Creating GRID renderer...")
                    self._renderer_strategy = self._renderer_registry.create(RenderMode.GRID, self._render_container)
                    self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_try_render_visual: Renderer created: {self._renderer_strategy}")

                    # Safely remove placeholder and add renderer widget
                    if (
                        self._render_placeholder is not None
                        and self._render_placeholder.parent()
                        and self._render_layout is not None
                    ):
                        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_try_render_visual: Removing placeholder widget")
                        self._render_layout.removeWidget(self._render_placeholder)
                        self._render_placeholder.deleteLater()
                    if (
                        self._renderer_strategy
                        and hasattr(self._renderer_strategy, 'widget')
                        and self._render_layout is not None
                    ):
                        widget = self._renderer_strategy.widget
                        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_try_render_visual: Adding renderer widget to layout (widget={widget})")
                        # Ensure renderer widget expands to fill container
                        widget.setSizePolicy(
                            QtWidgets.QSizePolicy.Policy.Expanding,
                            QtWidgets.QSizePolicy.Policy.Expanding
                        )
                        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_try_render_visual: Renderer widget size policy set to Expanding/Expanding")
                        self._render_layout.addWidget(widget)
                        self.log_constant(LOG_UI_LIVE_TAB_INFO, message=f"_try_render_visual: Renderer widget successfully added to layout")
                except Exception as e:
                    self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=
                        f"_try_render_visual: Failed to initialize renderer: {e}",
                        exc_info=e,
                    )
                    return

            # Extract render_payload if available
            render_payload = None
            if isinstance(payload, dict):
                # Try render_payload first (from dict), then render_payload_json (from protobuf)
                render_payload = payload.get("render_payload")
                if render_payload is None:
                    render_payload_json = payload.get("render_payload_json")
                    if render_payload_json:
                        try:
                            if isinstance(render_payload_json, str):
                                render_payload = json.loads(render_payload_json)
                            else:
                                render_payload = render_payload_json
                        except (json.JSONDecodeError, TypeError) as e:
                            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_try_render_visual: Failed to parse render_payload_json: {e}")
            else:
                render_payload = getattr(payload, "render_payload", None)

            # If no render_payload, try to generate one from observation (lightweight)
            if render_payload is None:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_try_render_visual: No render_payload, generating from observation")
                render_payload = self._generate_render_payload_from_observation(payload)

            if render_payload is None:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_try_render_visual: No render_payload after generation, returning")
                return

            # Convert to dict if needed
            if not isinstance(render_payload, dict):
                try:
                    if isinstance(render_payload, str):
                        render_payload = json.loads(render_payload)
                    else:
                        render_payload = dict(render_payload)
                except (json.JSONDecodeError, TypeError):
                    self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_try_render_visual: Failed to convert render_payload to dict")
                    return

            # Extract game_id from payload (update on every render to handle game switching)
            game_id_raw = render_payload.get("game_id")
            if game_id_raw:
                try:
                    new_game = game_id_raw if isinstance(game_id_raw, GameId) else GameId(str(game_id_raw))
                    if new_game != self._current_game:
                        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_try_render_visual: Switching game from {self._current_game} to {new_game}")
                        self._current_game = new_game
                    # Always set current_game even if unchanged, to ensure renderer uses correct assets
                    self._current_game = new_game
                except (ValueError, KeyError):
                    pass

            # Render if supported
            if self._renderer_strategy and self._renderer_strategy.supports(render_payload):
                game_id_str = str(self._current_game) if self._current_game else "None"
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_try_render_visual: Renderer supports payload (game_id={game_id_str})")
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_try_render_visual: Calling renderer.render() with grid={render_payload.get('grid')}")
                context = RendererContext(game_id=self._current_game)
                self._renderer_strategy.render(render_payload, context=context)
                self.log_constant(LOG_UI_LIVE_TAB_INFO, message=f"_try_render_visual: Grid rendered successfully (grid size={len(render_payload.get('grid', []))}x{len(render_payload.get('grid', [[]])[0]) if render_payload.get('grid') else 0})")
            else:
                self.log_constant(LOG_UI_LIVE_TAB_WARNING, message=f"_try_render_visual: Renderer does not support payload or is None")
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"  - _renderer_strategy: {self._renderer_strategy}")
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"  - supports(): {self._renderer_strategy.supports(render_payload) if self._renderer_strategy else 'N/A'}")
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"  - payload keys: {list(render_payload.keys())}")
                game_id_str = str(self._current_game) if self._current_game else "None"
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"  - game_id: {game_id_str}")
        except Exception as e:
            self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=f"_try_render_visual: ERROR - {e}", exc_info=e)

    def add_episode(self, payload: Any) -> None:
        """Add an episode to the buffer and update table."""
        try:
            episode_index = payload.get("episode_index", "?") if isinstance(payload, dict) else getattr(payload, "episode_index", "?")
            self.log_constant(LOG_UI_LIVE_TAB_INFO, message=f"add_episode: START (episode={episode_index}, is_destroyed={self._is_destroyed})")

            if self._is_destroyed:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="add_episode: Widget is destroyed, skipping")
                return

            self._episode_buffer.append(payload)
            self._update_stats()

            # Schedule episode rendering on the GUI thread using QMetaObject.invokeMethod.
            # This works from any thread (including background threads without Qt event loops).
            self._schedule_episode_render(payload)
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="add_episode: COMPLETE")
        except Exception as e:
            self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=f"add_episode: ERROR - {e}", exc_info=e)
    
    
    def _schedule_step_render(self, payload: Any) -> None:
        """Schedule step rendering on the GUI thread using QMetaObject.invokeMethod.

        This method works from any thread (including background threads without event loops).
        It stores the payload and invokes the actual render method on the GUI thread.
        """
        try:
            # Store payload in a temporary slot so we can pass it to the render method
            self._pending_step_payload = payload
            # Use QMetaObject.invokeMethod to schedule on GUI thread
            # QtCore.Qt.ConnectionType.QueuedConnection ensures it runs on the GUI thread's event loop
            QtCore.QMetaObject.invokeMethod(
                self,
                "_render_latest_step_from_pending",
                _QUEUED_CONNECTION,
            )
        except Exception as e:
            self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=f"_schedule_step_render: ERROR - {e}", exc_info=e)


    def _schedule_episode_render(self, payload: Any) -> None:
        """Schedule episode rendering on the GUI thread using QMetaObject.invokeMethod.

        This method works from any thread (including background threads without event loops).
        It stores the payload and invokes the actual render method on the GUI thread.
        """
        try:
            # Store payload in a temporary slot so we can pass it to the render method
            self._pending_episode_payload = payload
            # Use QMetaObject.invokeMethod to schedule on GUI thread
            # QtCore.Qt.ConnectionType.QueuedConnection ensures it runs on the GUI thread's event loop
            QtCore.QMetaObject.invokeMethod(
                self,
                "_render_episode_row_from_pending",
                _QUEUED_CONNECTION,
            )
        except Exception as e:
            self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=f"_schedule_episode_render: ERROR - {e}", exc_info=e)
            
    @pyqtSlot()
    def _render_latest_step_from_pending(self) -> None:
        """Wrapper to render the pending step payload. Called via QMetaObject.invokeMethod."""
        try:
            if hasattr(self, '_pending_step_payload'):
                payload = self._pending_step_payload
                delattr(self, '_pending_step_payload')
                self._render_latest_step(payload)
        except Exception as e:
            self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=f"_render_latest_step_from_pending: ERROR - {e}", exc_info=e)
            
    @pyqtSlot()
    def _render_episode_row_from_pending(self) -> None:
        """Wrapper to render the pending episode payload. Called via QMetaObject.invokeMethod."""
        try:
            if hasattr(self, '_pending_episode_payload'):
                payload = self._pending_episode_payload
                delattr(self, '_pending_episode_payload')
                self._render_episode_row(payload)
        except Exception as e:
            self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=f"_render_episode_row_from_pending: ERROR - {e}", exc_info=e)



    def _render_episode_row(self, payload: Any) -> None:
        """Add a row to the episodes table."""
        try:
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_render_episode_row: START (is_destroyed={self._is_destroyed})")

            if self._is_destroyed:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_render_episode_row: Widget is destroyed, skipping")
                return

            # Helper function to handle both dict and object payloads
            def _get_field(obj: Any, *field_names: str, default: Any = None) -> Any:
                for field in field_names:
                    if isinstance(obj, dict):
                        val = obj.get(field)
                        if val is not None:
                            return val
                    else:
                        val = getattr(obj, field, None)
                        if val is not None:
                            return val
                return default

            # Try both camelCase (from protobuf) and snake_case field names
            episode_index_raw = _get_field(payload, "episodeIndex", "episode_index", "episode", default=0)
            steps_raw = _get_field(payload, "steps", default=0)
            total_reward_raw = _get_field(payload, "totalReward", "total_reward", "reward", default=0.0)
            terminated = _get_field(payload, "terminated", default=False)
            truncated = _get_field(payload, "truncated", default=False)

            # Convert to proper types (handles both string and int values from protobuf)
            try:
                episode_index = int(episode_index_raw) if episode_index_raw is not None else 0
            except (TypeError, ValueError):
                episode_index = 0

            try:
                steps = int(steps_raw) if steps_raw is not None else 0
            except (TypeError, ValueError):
                steps = 0

            # Ensure total_reward is a float for formatting and comparisons
            try:
                total_reward = float(total_reward_raw) if total_reward_raw is not None else 0.0
            except (TypeError, ValueError):
                total_reward = 0.0

            # Extract timestamp
            timestamp = _get_field(payload, "timestamp", "ts")
            if timestamp:
                if hasattr(timestamp, 'seconds'):
                    # Protobuf Timestamp
                    from datetime import datetime, timezone as tz
                    dt = datetime.fromtimestamp(timestamp.seconds + timestamp.nanos / 1e9, tz=tz.utc)
                    ts_display = dt.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(timestamp, str):
                    ts_display = timestamp[:19].replace('T', ' ')
                else:
                    ts_display = str(timestamp)[:19]
            else:
                ts_display = "—"

            metadata_json = _get_field(payload, "metadataJson", "metadata_json", default="")
            metadata: dict[str, Any] = {}
            if isinstance(metadata_json, str) and metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {}
            elif isinstance(metadata_json, dict):
                metadata = metadata_json

            seed = metadata.get("seed", "—")
            episode_seed = metadata.get("episode_seed", "—")
            control_mode = metadata.get("mode") or metadata.get("control_mode") or "—"
            game_id = metadata.get("game_id", "—")

            # CRITICAL FIX: Update current_game from episode metadata
            # This ensures the renderer uses the correct game_id for rendering
            if game_id != "—":
                try:
                    new_game_id = game_id if isinstance(game_id, GameId) else GameId(str(game_id))
                    if self._current_game != new_game_id:
                        self._current_game = new_game_id
                        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_render_episode_row: Updated current_game to {self._current_game} from episode metadata")
                except (ValueError, KeyError) as exc:
                    self.log_constant(
                        LOG_UI_LIVE_TAB_WARNING,
                        message=f"_render_episode_row: Invalid game_id in metadata: {game_id}",
                        exc_info=exc,
                    )

            # CRITICAL FIX: Display episode should equal seed + episode_index
            # episode_index is 0-based (0, 1, 2, 3...)
            # display_episode = seed + episode_index
            try:
                seed_int = int(seed) if seed != "—" else 0
                episode_idx_int = int(episode_index)
                display_episode = seed_int + episode_idx_int
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_render_episode_row: seed={seed} ({seed_int}), episode_index={episode_index} ({episode_idx_int}), display_episode={display_episode}")
            except (TypeError, ValueError) as e:
                self.log_constant(LOG_UI_LIVE_TAB_WARNING, message=f"_render_episode_row: Failed to calculate display_episode: {e}", exc_info=e)
                display_episode = int(episode_index) if episode_index else 0

            # Compute outcome display from episode state
            if terminated:
                if total_reward > 0:
                    outcome_display = "Success"
                elif total_reward < 0:
                    outcome_display = "Failed"
                else:
                    outcome_display = "Completed"
            elif truncated:
                outcome_display = "Timeout"
            else:
                outcome_display = "Running"

            row = self._episodes_table.rowCount()
            self._episodes_table.insertRow(row)
            # Column 0: Timestamp
            self._episodes_table.setItem(row, 0, QtWidgets.QTableWidgetItem(ts_display))
            # Column 1: Episode (display value = seed + episode_index)
            self._episodes_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(display_episode)))
            self._episodes_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(steps)))
            self._episodes_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{total_reward:.2f}"))
            self._episodes_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(terminated)))
            self._episodes_table.setItem(row, 5, QtWidgets.QTableWidgetItem(str(truncated)))
            self._episodes_table.setItem(row, 6, QtWidgets.QTableWidgetItem(str(seed)))
            self._episodes_table.setItem(row, 7, QtWidgets.QTableWidgetItem(str(episode_seed)))
            self._episodes_table.setItem(row, 8, QtWidgets.QTableWidgetItem(str(control_mode)))
            self._episodes_table.setItem(row, 9, QtWidgets.QTableWidgetItem(str(game_id)))
            self._episodes_table.setItem(row, 10, QtWidgets.QTableWidgetItem(str(outcome_display)))

            if metadata:
                tooltip = json.dumps(metadata, indent=2)
                for col in range(self._episodes_table.columnCount()):
                    item = self._episodes_table.item(row, col)
                    if item is not None:
                        item.setToolTip(tooltip)

            # Auto-scroll to latest
            self._episodes_table.scrollToBottom()

            # Enable Copy button when data is available
            if self._episodes_table.rowCount() > 0:
                self._episodes_copy_button.setEnabled(True)

            # Limit rows to prevent memory bloat (keep last 100 episodes, same as steps)
            if self._episodes_table.rowCount() > 100:
                self._episodes_table.removeRow(0)

            self.log_constant(LOG_UI_LIVE_TAB_INFO, message=f"_render_episode_row: COMPLETE (row_count={self._episodes_table.rowCount()})")
        except Exception as e:
            # NEW: Never let UI rendering die silently - log raw payload preview
            try:
                preview = (json.dumps(payload)[:400] + "…") if isinstance(payload, dict) else str(payload)[:400] + "…"
            except Exception:
                preview = "<uninspectable>"
            self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=
                "EPISODE ROW RENDER FAILED",
                extra={
                    "agent": self.agent_id,
                    "run": self.run_id,
                    "payload_preview": preview,
                    "error": str(e)
                }
            )

    def mark_overflow(self, stream_type: str, dropped: int) -> None:
        """Record dropped events and update overflow indicator."""
        if stream_type == "step":
            self._dropped_steps += dropped
        else:
            self._dropped_episodes += dropped
        self._update_overflow_label()

    def _update_stats(self) -> None:
        """Refresh step/episode counters.
        
        CRITICAL FIX: Display current episode/step indices, NOT buffer sizes.
        This ensures counter resets at episode boundaries and shows actual training progress.
        """
        # OLD (BROKEN): f"Steps: {len(self._step_buffer)} | Episodes: {len(self._episode_buffer)}"
        # Problem: Buffer maxes at 100, so counter gets stuck showing "100" for multiple episodes
        
        # NEW (FIXED): Show actual current episode and step indices
        self._stats_label.setText(
            f"Episode: {self._current_episode_index} Step: {self._current_step_in_episode}"
        )

    def _update_overflow_label(self) -> None:
        """Show overflow warnings if drops occurred."""
        if self._dropped_steps > 0 or self._dropped_episodes > 0:
            text = f"⚠ Dropped: {self._dropped_steps} steps, {self._dropped_episodes} episodes"
            self._overflow_label.setText(text)
            self._overflow_label.setToolTip(
                "GUI couldn't keep up with telemetry stream. Consider reducing sampling rate."
            )
        else:
            self._overflow_label.setText("")

    def get_buffer_stats(self) -> dict[str, Any]:
        """Return buffer statistics for tooltip/badge."""
        return {
            "steps_buffered": len(self._step_buffer),
            "episodes_buffered": len(self._episode_buffer),
            "dropped_steps": self._dropped_steps,
            "dropped_episodes": self._dropped_episodes,
        }

    def _copy_episodes_table_to_clipboard(self) -> None:
        """Copy episodes table contents to clipboard in TSV format."""
        if self._episodes_table.rowCount() == 0:
            return

        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is None:
            return

        # Extract headers
        headers: list[str] = []
        for column_index in range(self._episodes_table.columnCount()):
            header_item = self._episodes_table.horizontalHeaderItem(column_index)
            headers.append(header_item.text() if header_item is not None else "")

        # Extract rows
        rows: list[str] = ["\t".join(headers)]
        for row_index in range(self._episodes_table.rowCount()):
            row_values: list[str] = []
            for column_index in range(self._episodes_table.columnCount()):
                item = self._episodes_table.item(row_index, column_index)
                row_values.append(item.text() if item is not None else "")
            rows.append("\t".join(row_values))

        # Copy to clipboard
        clipboard.setText("\n".join(rows))
        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"Copied {len(rows)-1} episodes to clipboard")

    def _copy_steps_table_to_clipboard(self) -> None:
        """Copy steps table contents to clipboard in TSV format."""
        if self._steps_table.rowCount() == 0:
            return

        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is None:
            return

        # Extract headers
        headers: list[str] = []
        for column_index in range(self._steps_table.columnCount()):
            header_item = self._steps_table.horizontalHeaderItem(column_index)
            headers.append(header_item.text() if header_item is not None else "")

        # Extract rows
        rows: list[str] = ["\t".join(headers)]
        for row_index in range(self._steps_table.rowCount()):
            row_values: list[str] = []
            for column_index in range(self._steps_table.columnCount()):
                item = self._steps_table.item(row_index, column_index)
                row_values.append(item.text() if item is not None else "")
            rows.append("\t".join(row_values))

        # Copy to clipboard
        clipboard.setText("\n".join(rows))
        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"Copied {len(rows)-1} steps to clipboard")

    def cleanup(self) -> None:
        """Clean up resources before widget destruction.

        This method should be called before the widget is deleted to prevent
        segmentation faults from pending QTimer callbacks trying to access
        destroyed widgets.
        """
        try:
            self.log_constant(LOG_UI_LIVE_TAB_INFO, message=f"cleanup: START for {self.run_id}/{self.agent_id}")

            # Set destroyed flag FIRST to prevent any pending timers from accessing the widget
            self._is_destroyed = True
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="cleanup: Set _is_destroyed = True")

            # Clear pending render payload to prevent timer callbacks from using it
            self._last_render_payload = None
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="cleanup: Cleared _last_render_payload")

            # Clear buffers
            step_count = len(self._step_buffer)
            self._step_buffer.clear()
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"cleanup: Cleared step buffer (was {step_count} items)")

            episode_count = len(self._episode_buffer)
            self._episode_buffer.clear()
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"cleanup: Cleared episode buffer (was {episode_count} items)")

            # Stop rendering speed regulator
            if hasattr(self, '_render_regulator') and self._render_regulator is not None:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="cleanup: Stopping RenderingSpeedRegulator")
                try:
                    self._render_regulator.stop()
                    self._render_regulator.clear_queue()
                    self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="cleanup: RenderingSpeedRegulator stopped and queue cleared")
                except Exception as e:
                    self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=f"cleanup: Error stopping regulator: {e}", exc_info=e)

            # Clear renderer strategy
            if self._renderer_strategy is not None:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"cleanup: Cleaning up renderer strategy: {self._renderer_strategy}")
                try:
                    if hasattr(self._renderer_strategy, 'cleanup'):
                        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="cleanup: Calling renderer.cleanup()")
                        self._renderer_strategy.cleanup()
                except Exception as e:
                    self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=f"cleanup: Error cleaning up renderer: {e}", exc_info=e)
                self._renderer_strategy = None
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="cleanup: Set _renderer_strategy = None")

            self.log_constant(LOG_UI_LIVE_TAB_INFO, message=f"cleanup: COMPLETE for {self.run_id}/{self.agent_id}")
        except Exception as e:
            self.log_constant(LOG_UI_LIVE_TAB_ERROR, message=f"cleanup: FATAL ERROR - {e}", exc_info=e)

    def _generate_render_payload_from_observation(self, payload: Any) -> Optional[dict[str, Any]]:
        """Generate a render payload from observation data (lightweight fallback).

        This is a fallback when the worker doesn't send render_payload.
        Optimized to be fast and avoid blocking the event loop.
        """
        try:
            # Extract observation and metadata
            obs_json = None
            if isinstance(payload, dict):
                obs_json = payload.get("observation_json")
            else:
                obs_json = getattr(payload, "observation_json", None)

            if not obs_json:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_generate_render_payload_from_observation: No observation_json found")
                return None

            # Parse observation (fast path for simple cases)
            try:
                if isinstance(obs_json, str):
                    # Quick check: if it looks like a simple number, skip parsing
                    if obs_json.isdigit():
                        self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_generate_render_payload_from_observation: observation_json is just a number, skipping")
                        return None  # Can't render without position info
                    obs_data = json.loads(obs_json)
                else:
                    obs_data = obs_json
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_generate_render_payload_from_observation: Failed to parse observation_json: {e}")
                return None

            # For FrozenLake: observation is typically {"state": N, "position": {"row": r, "col": c}, ...}
            if not isinstance(obs_data, dict):
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_generate_render_payload_from_observation: obs_data is not a dict: {type(obs_data)}")
                return None

            state = obs_data.get("state")
            position = obs_data.get("position")
            holes = obs_data.get("holes")

            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"_generate_render_payload_from_observation: state={state}, position={position}, holes={holes}")

            if state is None or position is None:
                self.log_constant(LOG_UI_LIVE_TAB_TRACE, message="_generate_render_payload_from_observation: Missing state or position")
                return None

            # Extract position (fast path)
            try:
                row = int(position.get("row", 0)) if isinstance(position, dict) else 0
                col = int(position.get("col", 0)) if isinstance(position, dict) else 0
            except (TypeError, ValueError):
                return None

            # Extract grid size from observation (respect actual environment dimensions)
            # Default to 4x4 for FrozenLake-v1, but respect what the observation reports
            grid_size_info = obs_data.get("grid_size")
            if isinstance(grid_size_info, dict):
                # grid_size is {"height": h, "width": w}
                grid_height = int(grid_size_info.get("height", 4))
                grid_width = int(grid_size_info.get("width", 4))
            elif isinstance(grid_size_info, int):
                # grid_size is a single number (square grid)
                grid_height = grid_width = int(grid_size_info)
            else:
                # Default to 4x4 for FrozenLake-v1
                grid_height = grid_width = 4

            # Create grid representation with correct dimensions
            grid = [['F' for _ in range(grid_width)] for _ in range(grid_height)]
            grid_size = max(grid_height, grid_width)  # Use max for bounds checking

            # Mark holes from observation data
            holes = obs_data.get("holes", [])
            if isinstance(holes, list):
                for hole in holes:
                    if isinstance(hole, dict):
                        try:
                            hole_row = int(hole.get("row", -1))
                            hole_col = int(hole.get("col", -1))
                            if 0 <= hole_row < grid_size and 0 <= hole_col < grid_size:
                                grid[hole_row][hole_col] = 'H'
                        except (TypeError, ValueError):
                            pass

            # Mark goal (bottom-right by default)
            goal = obs_data.get("goal", {})
            if isinstance(goal, dict):
                try:
                    goal_row = int(goal.get("row", grid_height - 1))
                    goal_col = int(goal.get("col", grid_width - 1))
                    if 0 <= goal_row < grid_height and 0 <= goal_col < grid_width:
                        grid[goal_row][goal_col] = 'G'
                except (TypeError, ValueError):
                    pass

            # Mark start (top-left)
            if grid_height > 0 and grid_width > 0:
                grid[0][0] = 'S'

            # Create render payload
            render_payload = {
                "mode": RenderMode.GRID.value,
                "grid": grid,
                "agent_position": (row, col),
                "game_id": self._current_game.value if self._current_game else "FrozenLake-v1",
                "terminated": getattr(payload, "terminated", False) if not isinstance(payload, dict) else payload.get("terminated", False),
            }
            return render_payload
        except Exception as e:
            self.log_constant(LOG_UI_LIVE_TAB_TRACE, message=f"Failed to generate render payload: {e}")
            return None


__all__ = ["LiveTelemetryTab"]
