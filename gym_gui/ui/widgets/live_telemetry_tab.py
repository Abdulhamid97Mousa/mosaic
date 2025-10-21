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
        buffer_size: int = 100,
        episode_buffer_size: int = 100,
        renderer_registry: Optional[RendererRegistry] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        self._step_buffer: Deque[Any] = deque(maxlen=buffer_size)
        self._episode_buffer: Deque[Any] = deque(maxlen=episode_buffer_size)
        self._dropped_steps = 0
        self._dropped_episodes = 0
        self._renderer_registry = renderer_registry or create_default_renderer_registry()
        self._renderer_strategy: Optional[Any] = None
        self._current_game: Optional[GameId] = None
        self._render_throttle_counter = 0
        self._render_throttle_interval = 20  # Render every 20th step to reduce GUI load during high-frequency training
        self._last_render_payload: Optional[dict[str, Any]] = None
        self._is_destroyed = False  # Track if widget is being destroyed
        super().__init__(run_id, agent_id, parent=parent)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Use inherited header builder and extend it
        header = self._build_header()
        self._stats_label = QtWidgets.QLabel("Steps: 0 | Episodes: 0")
        header.addWidget(self._stats_label)
        layout.addLayout(header)

        # Main horizontal splitter: rendering on left, telemetry on right
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        layout.addWidget(main_splitter)

        # Left side: Rendering panel
        render_group = QtWidgets.QGroupBox("Live Rendering", self)
        render_layout = QtWidgets.QVBoxLayout(render_group)
        self._render_container = QtWidgets.QWidget(render_group)
        self._render_layout = QtWidgets.QVBoxLayout(self._render_container)
        self._render_layout.setContentsMargins(0, 0, 0, 0)
        self._render_placeholder = QtWidgets.QLabel("Waiting for render data...")
        self._render_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._render_placeholder.setStyleSheet("color: #999; font-style: italic;")
        self._render_layout.addWidget(self._render_placeholder)
        render_layout.addWidget(self._render_container)
        main_splitter.addWidget(render_group)

        # Right side: Telemetry (vertical splitter for steps and episodes)
        telemetry_widget = QtWidgets.QWidget(self)
        telemetry_layout = QtWidgets.QVBoxLayout(telemetry_widget)
        telemetry_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, telemetry_widget)
        telemetry_layout.addWidget(splitter)

        # Steps view
        steps_group = QtWidgets.QGroupBox("Recent Steps", self)
        steps_layout = QtWidgets.QVBoxLayout(steps_group)
        self._steps_view = QtWidgets.QPlainTextEdit(steps_group)
        self._steps_view.setReadOnly(True)
        self._steps_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._steps_view.setMaximumBlockCount(200)
        steps_layout.addWidget(self._steps_view)
        splitter.addWidget(steps_group)

        # Episodes view
        episodes_group = QtWidgets.QGroupBox("Recent Episodes", self)
        episodes_layout = QtWidgets.QVBoxLayout(episodes_group)
        self._episodes_table = QtWidgets.QTableWidget(0, 10, episodes_group)
        self._episodes_table.setHorizontalHeaderLabels([
            "Timestamp",
            "Episode",
            "Steps",
            "Reward",
            "Terminated",
            "Truncated",
            "Seed",
            "Mode",
            "Game",
            "Outcome",
        ])
        header = self._episodes_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        episodes_layout.addWidget(self._episodes_table)
        splitter.addWidget(episodes_group)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        main_splitter.addWidget(telemetry_widget)
        main_splitter.setStretchFactor(0, 1)  # Rendering panel
        main_splitter.setStretchFactor(1, 2)  # Telemetry panel

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
            _LOGGER.debug(f"add_step: START (is_destroyed={self._is_destroyed})")

            if self._is_destroyed:
                _LOGGER.debug("add_step: Widget is destroyed, skipping")
                return

            self._step_buffer.append(payload)
            self._update_stats()
            # Only render the most recent step to keep UI responsive
            self._render_latest_step(payload)

            # Throttle visual rendering to avoid blocking event loop
            # Only render every N steps to keep UI responsive during high-frequency telemetry
            self._render_throttle_counter += 1
            if self._render_throttle_counter >= self._render_throttle_interval:
                self._render_throttle_counter = 0
                # Store payload for deferred rendering (avoid lambda capture issues)
                self._last_render_payload = payload
                # Schedule rendering on next event loop iteration
                _LOGGER.debug(f"add_step: Scheduling deferred render (counter={self._render_throttle_counter})")
                QtCore.QTimer.singleShot(0, self._process_deferred_render)

            _LOGGER.debug("add_step: COMPLETE")
        except Exception as e:
            _LOGGER.exception(f"add_step: ERROR - {e}")

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
        """Render only the latest step without parsing entire buffer."""
        episode_index = getattr(payload, "episode_index", 0)
        step_index = getattr(payload, "step_index", 0)
        reward = getattr(payload, "reward", 0.0)
        terminated = getattr(payload, "terminated", False)
        truncated = getattr(payload, "truncated", False)

        # Extract timestamp - try multiple field names
        timestamp = None
        if isinstance(payload, dict):
            timestamp = payload.get("timestamp") or payload.get("ts")
        else:
            timestamp = getattr(payload, "timestamp", None) or getattr(payload, "ts", None)

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

        # Get action display - try multiple sources for robustness
        action_json = getattr(payload, "action_json", "") or ""
        action_value = getattr(payload, "action", None)

        if action_value is not None:
            action_display = str(action_value)
        elif isinstance(payload, dict) and "action" in payload:
            action_display = str(payload.get("action"))
        elif action_json:
            action_display = self._preview(action_json)
        else:
            action_display = "—"

        # Get observation display - try multiple sources for robustness
        observation_json = getattr(payload, "observation_json", "") or ""
        observation_value = getattr(payload, "observation", None)

        if observation_json:
            observation_display = self._preview(observation_json)
        elif observation_value is not None:
            try:
                observation_display = self._preview(json.dumps(observation_value))
            except (TypeError, ValueError):
                observation_display = self._preview(str(observation_value))
        elif isinstance(payload, dict):
            obs_value = payload.get("observation") or payload.get("state")
            if obs_value is not None:
                try:
                    observation_display = self._preview(json.dumps(obs_value))
                except (TypeError, ValueError):
                    observation_display = self._preview(str(obs_value))
            else:
                observation_display = "—"
        else:
            observation_display = "—"

        line = (
            f"[{ts_display}] "
            f"[ep{episode_index:04d} #{int(step_index):04d}] "
            f"r={reward:+.3f} term={terminated} trunc={truncated} "
            f"a={action_display} o={observation_display}"
        )
        self._steps_view.appendPlainText(line)

    def _try_render_visual(self, payload: Any) -> None:
        """Try to render visual representation from payload (throttled)."""
        try:
            _LOGGER.debug(f"_try_render_visual: START (is_destroyed={self._is_destroyed}, renderer_strategy={self._renderer_strategy is not None})")

            # Skip if renderer not initialized yet
            if self._renderer_strategy is None:
                _LOGGER.debug("_try_render_visual: Renderer not initialized, attempting to create")
                # Try to initialize on first call
                if not self._renderer_registry.is_registered(RenderMode.GRID):
                    _LOGGER.debug("_try_render_visual: GRID renderer not registered")
                    return
                try:
                    _LOGGER.debug("_try_render_visual: Creating GRID renderer")
                    self._renderer_strategy = self._renderer_registry.create(RenderMode.GRID, self._render_container)
                    _LOGGER.debug(f"_try_render_visual: Renderer created: {self._renderer_strategy}")

                    # Safely remove placeholder and add renderer widget
                    if self._render_placeholder and self._render_placeholder.parent():
                        _LOGGER.debug("_try_render_visual: Removing placeholder")
                        self._render_layout.removeWidget(self._render_placeholder)
                        self._render_placeholder.deleteLater()
                    if self._renderer_strategy and hasattr(self._renderer_strategy, 'widget'):
                        _LOGGER.debug("_try_render_visual: Adding renderer widget to layout")
                        self._render_layout.addWidget(self._renderer_strategy.widget)
                except Exception as e:
                    _LOGGER.exception(f"_try_render_visual: Failed to initialize renderer: {e}")
                    return

            # Extract render_payload if available
            render_payload = None
            if isinstance(payload, dict):
                render_payload = payload.get("render_payload")
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
                _LOGGER.debug("_try_render_visual: Calling renderer.render()")
                context = RendererContext(game_id=self._current_game)
                self._renderer_strategy.render(render_payload, context=context)
                _LOGGER.debug("_try_render_visual: COMPLETE")
            else:
                _LOGGER.debug(f"_try_render_visual: Renderer doesn't support payload or is None")
        except Exception as e:
            _LOGGER.exception(f"_try_render_visual: ERROR - {e}")

    def add_episode(self, payload: Any) -> None:
        """Add an episode to the buffer and update table."""
        self._episode_buffer.append(payload)
        self._update_stats()
        self._render_episode_row(payload)

    def _render_episode_row(self, payload: Any) -> None:
        """Add a row to the episodes table."""
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
        episode_index = _get_field(payload, "episodeIndex", "episode_index", "episode", default=-1)
        steps = _get_field(payload, "steps", default=0)
        total_reward = _get_field(payload, "totalReward", "total_reward", "reward", default=0.0)
        terminated = _get_field(payload, "terminated", default=False)
        truncated = _get_field(payload, "truncated", default=False)

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
        control_mode = metadata.get("mode") or metadata.get("control_mode") or "—"
        game_id = metadata.get("game_id", "—")
        
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
        # Column 0: Timestamp (NEW)
        self._episodes_table.setItem(row, 0, QtWidgets.QTableWidgetItem(ts_display))
        # Shift all other columns by 1
        self._episodes_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(episode_index)))
        self._episodes_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(steps)))
        self._episodes_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{float(total_reward):.2f}"))
        self._episodes_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(terminated)))
        self._episodes_table.setItem(row, 5, QtWidgets.QTableWidgetItem(str(truncated)))
        self._episodes_table.setItem(row, 6, QtWidgets.QTableWidgetItem(str(seed)))
        self._episodes_table.setItem(row, 7, QtWidgets.QTableWidgetItem(str(control_mode)))
        self._episodes_table.setItem(row, 8, QtWidgets.QTableWidgetItem(str(game_id)))
        self._episodes_table.setItem(row, 9, QtWidgets.QTableWidgetItem(str(outcome_display)))

        if metadata:
            tooltip = json.dumps(metadata, indent=2)
            for col in range(self._episodes_table.columnCount()):
                item = self._episodes_table.item(row, col)
                if item is not None:
                    item.setToolTip(tooltip)

        # Auto-scroll to latest
        self._episodes_table.scrollToBottom()

        # Limit rows to prevent memory bloat
        if self._episodes_table.rowCount() > 50:
            self._episodes_table.removeRow(0)

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

    def cleanup(self) -> None:
        """Clean up resources before widget destruction.

        This method should be called before the widget is deleted to prevent
        segmentation faults from pending QTimer callbacks trying to access
        destroyed widgets.
        """
        try:
            _LOGGER.info(f"cleanup: START for {self.run_id}/{self.agent_id}")
            self._is_destroyed = True
            _LOGGER.debug("cleanup: Set _is_destroyed = True")

            self._last_render_payload = None
            _LOGGER.debug("cleanup: Cleared _last_render_payload")

            self._step_buffer.clear()
            _LOGGER.debug(f"cleanup: Cleared step buffer (was {len(self._step_buffer)} items)")

            self._episode_buffer.clear()
            _LOGGER.debug(f"cleanup: Cleared episode buffer (was {len(self._episode_buffer)} items)")

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
                return None

            # Parse observation (fast path for simple cases)
            try:
                if isinstance(obs_json, str):
                    # Quick check: if it looks like a simple number, skip parsing
                    if obs_json.isdigit():
                        return None  # Can't render without position info
                    obs_data = json.loads(obs_json)
                else:
                    obs_data = obs_json
            except (json.JSONDecodeError, TypeError, ValueError):
                return None

            # For FrozenLake: observation is typically {"state": N, "position": {"row": r, "col": c}, ...}
            if not isinstance(obs_data, dict):
                return None

            state = obs_data.get("state")
            position = obs_data.get("position")

            if state is None or position is None:
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
