"""Live telemetry tab widget for displaying streamed run data."""

from __future__ import annotations

from collections import deque
import json
import logging
from typing import Any, Deque, Optional

from qtpy import QtCore, QtWidgets

from gym_gui.ui.widgets.base_telemetry_tab import BaseTelemetryTab
from gym_gui.rendering import RendererRegistry, create_default_renderer_registry, RendererContext
from gym_gui.core.enums import GameId, RenderMode

_LOGGER = logging.getLogger(__name__)


class LiveTelemetryTab(BaseTelemetryTab):
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
        renderer_registry: Optional[RendererRegistry] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
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
        render_group = QtWidgets.QGroupBox("Live Rendering", self)
        render_group.setMinimumHeight(300)  # Ensure rendering area has minimum height
        render_layout = QtWidgets.QVBoxLayout(render_group)
        self._render_container = QtWidgets.QWidget(render_group)
        # Set size policy to expand and fill available space
        self._render_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self._render_layout = QtWidgets.QVBoxLayout(self._render_container)
        self._render_layout.setContentsMargins(0, 0, 0, 0)
        self._render_placeholder = QtWidgets.QLabel("Waiting for render data...")
        self._render_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._render_placeholder.setStyleSheet("color: #999; font-style: italic;")
        self._render_layout.addWidget(self._render_placeholder)
        render_layout.addWidget(self._render_container)
        layout.addWidget(render_group, 3)  # Stretch factor 3 (increased from 1)

        # 2) TELEMETRY SECTIONS (vertical splitter for episodes and steps)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        layout.addWidget(splitter, 1)  # Stretch factor 1 (decreased from 2)

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

        # Footer with overflow stats
        self._overflow_label = QtWidgets.QLabel("")
        self._overflow_label.setStyleSheet("color: #d32f2f;")
        layout.addWidget(self._overflow_label)

    def set_render_throttle_interval(self, interval: int) -> None:
        """Set the render throttle interval (render every Nth step).
        
        Args:
            interval: Render every Nth step (1=every step, 2=every 2nd step, etc.)
        """
        self._render_throttle_interval = max(1, interval)
        _LOGGER.debug(f"Render throttle interval set to {self._render_throttle_interval}")

    def add_step(self, payload: Any) -> None:
        """Add a step to the buffer and update display lazily."""
        try:
            step_index = payload.get("step_index", "?") if isinstance(payload, dict) else getattr(payload, "step_index", "?")
            _LOGGER.info(f"add_step: START (step={step_index}, is_destroyed={self._is_destroyed})")

            if self._is_destroyed:
                _LOGGER.debug("add_step: Widget is destroyed, skipping")
                return

            self._step_buffer.append(payload)
            self._update_stats()

            # Schedule table update on main Qt thread (safe from any thread)
            # Use singleShot to defer to main event loop
            QtCore.QTimer.singleShot(0, lambda: self._render_latest_step(payload))

            # Throttle visual rendering to avoid blocking event loop
            # Only render every N steps to keep UI responsive during high-frequency telemetry
            self._render_throttle_counter += 1
            _LOGGER.debug(f"add_step: Throttle counter={self._render_throttle_counter}, interval={self._render_throttle_interval}")
            if self._render_throttle_counter >= self._render_throttle_interval:
                self._render_throttle_counter = 0
                # Store payload for deferred rendering (avoid lambda capture issues)
                self._last_render_payload = payload
                # Schedule rendering on main Qt thread (CRITICAL: prevents QBasicTimer threading errors)
                _LOGGER.debug(f"add_step: Throttle check passed, scheduling _try_render_visual on main thread")
                QtCore.QTimer.singleShot(0, lambda p=payload: self._try_render_visual(p))
            else:
                _LOGGER.debug(f"add_step: Throttle check failed, skipping render (counter={self._render_throttle_counter} < interval={self._render_throttle_interval})")

            _LOGGER.debug("add_step: COMPLETE")
        except Exception as e:
            _LOGGER.exception(f"add_step: ERROR - {e}")

    def _safe_process_deferred_render(self) -> None:
        """Safe wrapper for deferred rendering that checks if widget is destroyed.

        This prevents segfaults when QTimer callbacks try to access a deleted widget.
        """
        try:
            _LOGGER.debug(f"_safe_process_deferred_render: START")
            # Check if widget is being destroyed BEFORE accessing any attributes
            if not hasattr(self, '_is_destroyed') or self._is_destroyed:
                _LOGGER.debug(f"_safe_process_deferred_render: Widget is destroyed, skipping")
                return

            # Now safe to call the actual render method
            _LOGGER.debug(f"_safe_process_deferred_render: Calling _process_deferred_render")
            self._process_deferred_render()
            _LOGGER.debug(f"_safe_process_deferred_render: COMPLETE")
        except Exception as e:
            # Log errors during deferred render
            _LOGGER.exception(f"_safe_process_deferred_render: ERROR - {e}")

    def _process_deferred_render(self) -> None:
        """Process deferred rendering without lambda capture issues."""
        try:
            _LOGGER.debug(f"_process_deferred_render: START (is_destroyed={self._is_destroyed}, has_payload={self._last_render_payload is not None})")

            # Safety check: if widget is being destroyed, skip rendering
            if self._is_destroyed:
                _LOGGER.debug("_process_deferred_render: Widget is destroyed, skipping")
                self._last_render_payload = None
                return

            if self._last_render_payload is None:
                _LOGGER.debug("_process_deferred_render: No payload, skipping")
                return

            _LOGGER.debug("_process_deferred_render: Calling _try_render_visual")
            self._try_render_visual(self._last_render_payload)
            _LOGGER.debug("_process_deferred_render: COMPLETE")
        except Exception as e:
            _LOGGER.exception(f"_process_deferred_render: ERROR - {e}")
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

        reward = _get_field(payload, "reward", default=0.0)
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
            _LOGGER.debug(f"_try_render_visual: START (is_destroyed={self._is_destroyed}, renderer_strategy={self._renderer_strategy is not None})")

            # Skip if renderer not initialized yet
            if self._renderer_strategy is None:
                _LOGGER.debug("_try_render_visual: Renderer not initialized, attempting to create")
                # Try to initialize on first call
                if not self._renderer_registry.is_registered(RenderMode.GRID):
                    _LOGGER.warning("_try_render_visual: GRID renderer not registered in registry")
                    return
                try:
                    _LOGGER.debug("_try_render_visual: Creating GRID renderer...")
                    self._renderer_strategy = self._renderer_registry.create(RenderMode.GRID, self._render_container)
                    _LOGGER.debug(f"_try_render_visual: Renderer created: {self._renderer_strategy}")

                    # Safely remove placeholder and add renderer widget
                    if self._render_placeholder and self._render_placeholder.parent():
                        _LOGGER.debug("_try_render_visual: Removing placeholder widget")
                        self._render_layout.removeWidget(self._render_placeholder)
                        self._render_placeholder.deleteLater()
                    if self._renderer_strategy and hasattr(self._renderer_strategy, 'widget'):
                        _LOGGER.debug("_try_render_visual: Adding renderer widget to layout")
                        self._render_layout.addWidget(self._renderer_strategy.widget)
                        _LOGGER.debug(f"_try_render_visual: Renderer widget added, widget={self._renderer_strategy.widget}")
                except Exception as e:
                    _LOGGER.exception(f"_try_render_visual: Failed to initialize renderer: {e}")
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
                            _LOGGER.debug(f"_try_render_visual: Failed to parse render_payload_json: {e}")
            else:
                render_payload = getattr(payload, "render_payload", None)

            # If no render_payload, try to generate one from observation (lightweight)
            if render_payload is None:
                _LOGGER.debug("_try_render_visual: No render_payload, generating from observation")
                render_payload = self._generate_render_payload_from_observation(payload)

            if render_payload is None:
                _LOGGER.debug("_try_render_visual: No render_payload after generation, returning")
                return

            # Convert to dict if needed
            if not isinstance(render_payload, dict):
                try:
                    if isinstance(render_payload, str):
                        render_payload = json.loads(render_payload)
                    else:
                        render_payload = dict(render_payload)
                except (json.JSONDecodeError, TypeError):
                    _LOGGER.debug("_try_render_visual: Failed to convert render_payload to dict")
                    return

            # Extract game_id from payload (cache it)
            if self._current_game is None:
                game_id_raw = render_payload.get("game_id")
                if game_id_raw:
                    try:
                        self._current_game = game_id_raw if isinstance(game_id_raw, GameId) else GameId(str(game_id_raw))
                        _LOGGER.debug(f"_try_render_visual: Set current_game to {self._current_game}")
                    except (ValueError, KeyError):
                        pass

            # Render if supported
            if self._renderer_strategy and self._renderer_strategy.supports(render_payload):
                game_id_str = str(self._current_game) if self._current_game else "None"
                _LOGGER.debug(f"_try_render_visual: Renderer supports payload (game_id={game_id_str})")
                _LOGGER.debug(f"_try_render_visual: Calling renderer.render() with grid={render_payload.get('grid')}")
                context = RendererContext(game_id=self._current_game)
                self._renderer_strategy.render(render_payload, context=context)
                _LOGGER.info(f"_try_render_visual: Grid rendered successfully (grid size={len(render_payload.get('grid', []))}x{len(render_payload.get('grid', [[]])[0]) if render_payload.get('grid') else 0})")
            else:
                _LOGGER.warning(f"_try_render_visual: Renderer does not support payload or is None")
                _LOGGER.debug(f"  - _renderer_strategy: {self._renderer_strategy}")
                _LOGGER.debug(f"  - supports(): {self._renderer_strategy.supports(render_payload) if self._renderer_strategy else 'N/A'}")
                _LOGGER.debug(f"  - payload keys: {list(render_payload.keys())}")
                game_id_str = str(self._current_game) if self._current_game else "None"
                _LOGGER.debug(f"  - game_id: {game_id_str}")
        except Exception as e:
            _LOGGER.exception(f"_try_render_visual: ERROR - {e}")

    def add_episode(self, payload: Any) -> None:
        """Add an episode to the buffer and update table."""
        try:
            episode_index = payload.get("episode_index", "?") if isinstance(payload, dict) else getattr(payload, "episode_index", "?")
            _LOGGER.info(f"add_episode: START (episode={episode_index}, is_destroyed={self._is_destroyed})")

            if self._is_destroyed:
                _LOGGER.debug("add_episode: Widget is destroyed, skipping")
                return

            self._episode_buffer.append(payload)
            self._update_stats()

            # Schedule episode row rendering on main Qt thread (safe from any thread)
            # Use lambda to defer the call to the main event loop
            _LOGGER.debug(f"add_episode: Scheduling _render_episode_row on main thread")
            QtCore.QTimer.singleShot(0, lambda p=payload: self._render_episode_row(p))
            _LOGGER.debug("add_episode: COMPLETE")
        except Exception as e:
            _LOGGER.exception(f"add_episode: ERROR - {e}")

    def _render_episode_row(self, payload: Any) -> None:
        """Add a row to the episodes table."""
        try:
            _LOGGER.debug(f"_render_episode_row: START (is_destroyed={self._is_destroyed})")

            if self._is_destroyed:
                _LOGGER.debug("_render_episode_row: Widget is destroyed, skipping")
                return
        except Exception as e:
            _LOGGER.exception(f"_render_episode_row: ERROR checking destroyed state - {e}")
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
        total_reward = _get_field(payload, "totalReward", "total_reward", "reward", default=0.0)
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
                    _LOGGER.debug(f"_render_episode_row: Updated current_game to {self._current_game} from episode metadata")
            except (ValueError, KeyError):
                _LOGGER.warning(f"_render_episode_row: Invalid game_id in metadata: {game_id}")

        # CRITICAL FIX: Display episode should equal seed + episode_index
        # episode_index is 0-based (0, 1, 2, 3...)
        # display_episode = seed + episode_index
        try:
            seed_int = int(seed) if seed != "—" else 0
            episode_idx_int = int(episode_index)
            display_episode = seed_int + episode_idx_int
            _LOGGER.debug(f"_render_episode_row: seed={seed} ({seed_int}), episode_index={episode_index} ({episode_idx_int}), display_episode={display_episode}")
        except (TypeError, ValueError) as e:
            _LOGGER.warning(f"_render_episode_row: Failed to calculate display_episode: {e}")
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
        self._episodes_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{float(total_reward):.2f}"))
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

        _LOGGER.info(f"_render_episode_row: COMPLETE (row_count={self._episodes_table.rowCount()})")

    def mark_overflow(self, stream_type: str, dropped: int) -> None:
        """Record dropped events and update overflow indicator."""
        if stream_type == "step":
            self._dropped_steps += dropped
        else:
            self._dropped_episodes += dropped
        self._update_overflow_label()

    def _update_stats(self) -> None:
        """Refresh step/episode counters."""
        self._stats_label.setText(
            f"Steps: {len(self._step_buffer)} | Episodes: {len(self._episode_buffer)}"
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
        _LOGGER.debug(f"Copied {len(rows)-1} episodes to clipboard")

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
        _LOGGER.debug(f"Copied {len(rows)-1} steps to clipboard")

    def cleanup(self) -> None:
        """Clean up resources before widget destruction.

        This method should be called before the widget is deleted to prevent
        segmentation faults from pending QTimer callbacks trying to access
        destroyed widgets.
        """
        try:
            _LOGGER.info(f"cleanup: START for {self.run_id}/{self.agent_id}")

            # Set destroyed flag FIRST to prevent any pending timers from accessing the widget
            self._is_destroyed = True
            _LOGGER.debug("cleanup: Set _is_destroyed = True")

            # Clear pending render payload to prevent timer callbacks from using it
            self._last_render_payload = None
            _LOGGER.debug("cleanup: Cleared _last_render_payload")

            # Clear buffers
            step_count = len(self._step_buffer)
            self._step_buffer.clear()
            _LOGGER.debug(f"cleanup: Cleared step buffer (was {step_count} items)")

            episode_count = len(self._episode_buffer)
            self._episode_buffer.clear()
            _LOGGER.debug(f"cleanup: Cleared episode buffer (was {episode_count} items)")

            # Clear renderer strategy
            if self._renderer_strategy is not None:
                _LOGGER.debug(f"cleanup: Cleaning up renderer strategy: {self._renderer_strategy}")
                try:
                    if hasattr(self._renderer_strategy, 'cleanup'):
                        _LOGGER.debug("cleanup: Calling renderer.cleanup()")
                        self._renderer_strategy.cleanup()
                except Exception as e:
                    _LOGGER.exception(f"cleanup: Error cleaning up renderer: {e}")
                self._renderer_strategy = None
                _LOGGER.debug("cleanup: Set _renderer_strategy = None")

            _LOGGER.info(f"cleanup: COMPLETE for {self.run_id}/{self.agent_id}")
        except Exception as e:
            _LOGGER.exception(f"cleanup: FATAL ERROR - {e}")

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
                _LOGGER.debug("_generate_render_payload_from_observation: No observation_json found")
                return None

            # Parse observation (fast path for simple cases)
            try:
                if isinstance(obs_json, str):
                    # Quick check: if it looks like a simple number, skip parsing
                    if obs_json.isdigit():
                        _LOGGER.debug("_generate_render_payload_from_observation: observation_json is just a number, skipping")
                        return None  # Can't render without position info
                    obs_data = json.loads(obs_json)
                else:
                    obs_data = obs_json
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                _LOGGER.debug(f"_generate_render_payload_from_observation: Failed to parse observation_json: {e}")
                return None

            # For FrozenLake: observation is typically {"state": N, "position": {"row": r, "col": c}, ...}
            if not isinstance(obs_data, dict):
                _LOGGER.debug(f"_generate_render_payload_from_observation: obs_data is not a dict: {type(obs_data)}")
                return None

            state = obs_data.get("state")
            position = obs_data.get("position")
            holes = obs_data.get("holes")

            _LOGGER.debug(f"_generate_render_payload_from_observation: state={state}, position={position}, holes={holes}")

            if state is None or position is None:
                _LOGGER.debug("_generate_render_payload_from_observation: Missing state or position")
                return None

            # Extract position (fast path)
            try:
                row = int(position.get("row", 0)) if isinstance(position, dict) else 0
                col = int(position.get("col", 0)) if isinstance(position, dict) else 0
            except (TypeError, ValueError):
                return None

            # Create minimal grid representation (8x8 default)
            grid_size = 8
            grid = [['F' for _ in range(grid_size)] for _ in range(grid_size)]

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

            # Mark goal (bottom-right)
            goal = obs_data.get("goal", {})
            if isinstance(goal, dict):
                try:
                    goal_row = int(goal.get("row", grid_size - 1))
                    goal_col = int(goal.get("col", grid_size - 1))
                    if 0 <= goal_row < grid_size and 0 <= goal_col < grid_size:
                        grid[goal_row][goal_col] = 'G'
                except (TypeError, ValueError):
                    pass

            # Mark start (top-left)
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
            _LOGGER.debug(f"Failed to generate render payload: {e}")
            return None


__all__ = ["LiveTelemetryTab"]
