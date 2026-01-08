"""Render container for a single operator in multi-operator mode.

Each operator gets its own render container with:
- Header showing operator name and type badge
- Render area (Grid, Video, or Text)
- Compact telemetry stats (step, episode, reward)
- Status indicator (pending, running, stopped)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]
from qtpy import QtCore, QtWidgets

from gym_gui.services.operator import OperatorConfig
from gym_gui.rendering import RendererRegistry, create_default_renderer_registry, RendererContext
from gym_gui.rendering.strategies.board_game import BoardGameRendererStrategy
from gym_gui.core.enums import RenderMode

# Use operators namespace for dedicated operators.log routing
_LOGGER = logging.getLogger("gym_gui.operators.render_container")


class OperatorRenderContainer(QtWidgets.QFrame):
    """Render container for a single operator.

    Displays:
    - Header with operator name, type badge, and status
    - Render area for visual output (grid/video)
    - Stats bar with step/episode/reward counters
    """

    status_changed = pyqtSignal(str, str)  # operator_id, new_status

    # Status colors
    STATUS_COLORS = {
        "pending": "#9E9E9E",    # Gray
        "loaded": "#2196F3",     # Blue - environment loaded/ready
        "running": "#4CAF50",    # Green
        "stopped": "#F44336",    # Red
        "error": "#FF9800",      # Orange
    }

    # Type badge colors
    TYPE_COLORS = {
        "llm": "#2196F3",  # Blue
        "vlm": "#00BCD4",  # Cyan
        "rl": "#9C27B0",   # Purple
    }

    def __init__(
        self,
        config: OperatorConfig,
        renderer_registry: Optional[RendererRegistry] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._status = "pending"
        self._renderer_registry = renderer_registry or create_default_renderer_registry()
        # Support multiple renderer types (GRID for text environments, RGB for visual)
        self._grid_renderer: Optional[Any] = None
        self._rgb_renderer: Optional[Any] = None
        self._board_game_renderer: Optional[BoardGameRendererStrategy] = None
        self._renderer_strategy: Optional[Any] = None  # Currently active renderer
        self._active_render_mode: Optional[RenderMode] = None
        self._is_board_game: bool = False  # Track if current payload is a board game

        # Stats tracking
        self._current_step = 0
        self._current_episode = 0
        self._total_reward = 0.0
        self._episode_reward = 0.0

        self._build_ui()
        self._update_header()

    def _log_extra(self) -> dict:
        """Return extra dict for correlated logging with run_id and operator_id."""
        return {
            "run_id": self._config.run_id or "unknown",
            "agent_id": self._config.operator_id,
        }

    def _build_ui(self) -> None:
        self.setFrameStyle(QtWidgets.QFrame.Shape.Box | QtWidgets.QFrame.Shadow.Raised)
        self.setLineWidth(1)
        self.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 4px; }")

        # Container should expand to fill grid cell (Qt6 best practice)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        # Start with a large minimum size for better initial display
        self.setMinimumSize(400, 350)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header with name, type badge, and status
        self._header = QtWidgets.QWidget(self)
        header_layout = QtWidgets.QHBoxLayout(self._header)
        header_layout.setContentsMargins(4, 2, 4, 2)
        header_layout.setSpacing(8)

        # Operator name
        self._name_label = QtWidgets.QLabel(self._config.display_name, self._header)
        self._name_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        header_layout.addWidget(self._name_label)

        # Type badge (LLM / VLM / RL)
        self._type_badge = QtWidgets.QLabel(self._config.operator_type.upper(), self._header)
        type_color = self.TYPE_COLORS.get(self._config.operator_type, "#666")
        self._type_badge.setStyleSheet(
            f"background-color: {type_color}; color: white; "
            f"padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold;"
        )
        header_layout.addWidget(self._type_badge)

        # Worker name (e.g., "BALROG LLM Worker")
        worker_name = self._get_worker_display_name()
        self._worker_label = QtWidgets.QLabel(worker_name, self._header)
        self._worker_label.setStyleSheet("color: #333; font-size: 10px;")
        header_layout.addWidget(self._worker_label)

        # Model/Policy info (e.g., "GPT-4o Mini" for LLM, policy name for RL)
        model_info = self._get_model_display_name()
        self._model_label = QtWidgets.QLabel(model_info, self._header)
        self._model_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        header_layout.addWidget(self._model_label)

        # Environment/Task info
        env_task = f"{self._config.env_name}/{self._config.task}"
        self._env_label = QtWidgets.QLabel(env_task, self._header)
        self._env_label.setStyleSheet("color: #666; font-size: 10px;")
        self._env_label.setToolTip(f"Environment: {self._config.env_name}\nTask: {self._config.task}")
        header_layout.addWidget(self._env_label)

        header_layout.addStretch()

        # Status indicator
        self._status_indicator = QtWidgets.QLabel("PENDING", self._header)
        self._update_status_indicator()
        header_layout.addWidget(self._status_indicator)

        layout.addWidget(self._header)

        # Render area - should expand to fill available space
        self._render_container = QtWidgets.QWidget(self)
        self._render_container.setMinimumSize(350, 280)  # Larger minimum for better display
        self._render_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self._render_layout = QtWidgets.QVBoxLayout(self._render_container)
        self._render_layout.setContentsMargins(2, 2, 2, 2)
        self._render_layout.setSpacing(0)

        # Placeholder label
        self._placeholder = QtWidgets.QLabel("Waiting for data...", self._render_container)
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #999; font-style: italic;")
        self._render_layout.addWidget(self._placeholder)

        layout.addWidget(self._render_container, 1)

        # Stats bar
        self._stats_bar = QtWidgets.QWidget(self)
        stats_layout = QtWidgets.QHBoxLayout(self._stats_bar)
        stats_layout.setContentsMargins(4, 2, 4, 2)
        stats_layout.setSpacing(16)

        # Episode counter
        self._episode_label = QtWidgets.QLabel("Episode: 0", self._stats_bar)
        self._episode_label.setStyleSheet("font-size: 10px;")
        stats_layout.addWidget(self._episode_label)

        # Step counter
        self._step_label = QtWidgets.QLabel("Step: 0", self._stats_bar)
        self._step_label.setStyleSheet("font-size: 10px;")
        stats_layout.addWidget(self._step_label)

        # Episode reward
        self._reward_label = QtWidgets.QLabel("Reward: 0.00", self._stats_bar)
        self._reward_label.setStyleSheet("font-size: 10px;")
        stats_layout.addWidget(self._reward_label)

        stats_layout.addStretch()

        layout.addWidget(self._stats_bar)

    def _update_header(self) -> None:
        """Update header with current config."""
        self._name_label.setText(self._config.display_name)
        self._type_badge.setText(self._config.operator_type.upper())
        type_color = self.TYPE_COLORS.get(self._config.operator_type, "#666")
        self._type_badge.setStyleSheet(
            f"background-color: {type_color}; color: white; "
            f"padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold;"
        )
        # Update worker and model labels
        self._worker_label.setText(self._get_worker_display_name())
        self._model_label.setText(self._get_model_display_name())
        env_task = f"{self._config.env_name}/{self._config.task}"
        self._env_label.setText(env_task)

    def _update_status_indicator(self) -> None:
        """Update status indicator appearance."""
        status_color = self.STATUS_COLORS.get(self._status, "#666")
        self._status_indicator.setText(self._status.upper())
        self._status_indicator.setStyleSheet(
            f"background-color: {status_color}; color: white; "
            f"padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold;"
        )

    def _get_worker_display_name(self) -> str:
        """Get a user-friendly worker name from the config.

        Returns:
            Worker display name (e.g., "BALROG Worker", "CleanRL Worker").
        """
        worker_id = self._config.worker_id
        # Convert worker_id to display name
        # e.g., "balrog_worker" -> "BALROG", "cleanrl_worker" -> "CleanRL"
        if worker_id:
            name = worker_id.replace("_worker", "").replace("_", " ")
            return name.upper() if len(name) <= 6 else name.title()
        return ""

    def _get_model_display_name(self) -> str:
        """Get a user-friendly model/policy name from the config.

        Returns:
            For LLM/VLM: Model name (e.g., "GPT-4o Mini", "Llama 3.3 70B")
            For RL: Policy filename or "No policy"
        """
        settings = self._config.settings
        op_type = self._config.operator_type

        if op_type in ("llm", "vlm"):
            # Get model_id from settings
            model_id = settings.get("model_id", "")
            if model_id:
                # Extract display name from model_id
                # e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini"
                # e.g., "meta-llama/llama-3.3-70b-instruct:free" -> "llama-3.3-70b"
                if "/" in model_id:
                    model_id = model_id.split("/")[-1]
                # Remove common suffixes
                for suffix in (":free", "-instruct", "-chat"):
                    model_id = model_id.replace(suffix, "")
                return model_id
            return ""
        elif op_type == "rl":
            # Get policy path
            policy_path = settings.get("policy_path", "")
            if policy_path:
                # Extract filename from path
                import os
                return os.path.basename(policy_path)
            return ""
        return ""

    def set_config(self, config: OperatorConfig) -> None:
        """Update the operator configuration."""
        self._config = config
        self._update_header()

    def set_status(self, status: str) -> None:
        """Set the operator status.

        Args:
            status: One of "pending", "running", "stopped", "error".
        """
        if status not in self.STATUS_COLORS:
            _LOGGER.warning(f"Unknown status: {status}")
            return

        old_status = self._status
        self._status = status
        self._update_status_indicator()

        if old_status != status:
            self.status_changed.emit(self._config.operator_id, status)

    def set_display_size(self, width: int, height: int) -> None:
        """Set the display size of the render area.

        This controls the size of the render container widget,
        which affects how large the environment is displayed.

        Args:
            width: Width in pixels
            height: Height in pixels
        """
        # Update render container to fixed size
        self._render_container.setMinimumSize(width, height)
        self._render_container.setFixedSize(width, height)

        # Also update the overall widget size (add padding for header/stats)
        total_height = height + 80  # header + stats bar
        total_width = width + 20
        self.setMinimumSize(total_width, total_height)
        self.setFixedSize(total_width, total_height)

        # Force layout update
        self.updateGeometry()
        if self.parent():
            self.parent().updateGeometry()

        _LOGGER.debug(f"Set display size to {width}x{height} for {self._config.operator_id}")

    def display_payload(self, payload: Dict[str, Any]) -> None:
        """Display a render payload from telemetry.

        Args:
            payload: Telemetry payload containing render data.
        """
        _LOGGER.debug(
            "display_payload: received payload keys=%s",
            list(payload.keys()),
            extra=self._log_extra(),
        )
        try:
            # Update stats from payload
            self._update_stats_from_payload(payload)

            # Extract render data
            render_payload = self._extract_render_payload(payload)
            if render_payload is None:
                _LOGGER.debug(
                    "display_payload: render_payload is None, returning",
                    extra=self._log_extra(),
                )
                return

            # Debug: check render_payload structure
            rp_keys = list(render_payload.keys()) if isinstance(render_payload, dict) else "not_dict"
            rp_width = render_payload.get("width") if isinstance(render_payload, dict) else None
            rp_height = render_payload.get("height") if isinstance(render_payload, dict) else None
            rgb = render_payload.get("rgb") if isinstance(render_payload, dict) else None
            rgb_info = f"len={len(rgb)}" if rgb else "None"
            _LOGGER.debug(
                "render_payload: keys=%s, width=%s, height=%s, rgb=%s",
                rp_keys, rp_width, rp_height, rgb_info,
                extra=self._log_extra(),
            )

            # Check if this is a board game payload (chess, go, connect_four, etc.)
            is_board_game = self._is_board_game_payload(render_payload)
            _LOGGER.debug(
                "display_payload: is_board_game=%s",
                is_board_game,
                extra=self._log_extra(),
            )

            if is_board_game:
                # Use BoardGameRendererStrategy for board games
                if not self._board_game_renderer:
                    self._board_game_renderer = BoardGameRendererStrategy(self._render_container)
                    self._render_layout.addWidget(self._board_game_renderer.widget)

                # Switch to board game renderer if needed
                if not self._is_board_game:
                    self._is_board_game = True
                    # Hide other renderers
                    if self._grid_renderer:
                        self._grid_renderer.widget.hide()
                    if self._rgb_renderer:
                        self._rgb_renderer.widget.hide()
                    self._board_game_renderer.widget.show()
                    self._renderer_strategy = self._board_game_renderer

                # Render the board game
                if self._board_game_renderer.supports(render_payload):
                    context = RendererContext()
                    self._board_game_renderer.render(render_payload, context=context)
            else:
                # Use standard RGB or GRID renderer
                if self._is_board_game:
                    self._is_board_game = False
                    if self._board_game_renderer:
                        self._board_game_renderer.widget.hide()

                # Detect payload type and initialize appropriate renderer
                required_mode = self._detect_render_mode(render_payload)
                if required_mode != self._active_render_mode:
                    self._init_renderer(required_mode)

                # Render using the active strategy
                if self._renderer_strategy and self._renderer_strategy.supports(render_payload):
                    context = RendererContext()
                    self._renderer_strategy.render(render_payload, context=context)

        except Exception as e:
            _LOGGER.error(f"Error displaying payload: {e}")

    def _is_board_game_payload(self, payload: Dict[str, Any]) -> bool:
        """Check if the payload is for a board game (chess, go, connect_four, etc.).

        Args:
            payload: The render payload to analyze.

        Returns:
            True if this is a board game payload that should use BoardGameRendererStrategy.
        """
        # Check for board game specific keys
        if "chess" in payload or "fen" in payload:
            return True
        if "connect_four" in payload:
            return True
        if "go" in payload:
            return True
        if "sudoku" in payload:
            return True
        # Check game_id
        game_id = payload.get("game_id")
        if game_id in ("chess", "connect_four", "go", "tictactoe", "sudoku"):
            return True
        return False

    def _detect_render_mode(self, payload: Dict[str, Any]) -> RenderMode:
        """Detect which render mode is needed for this payload.

        Args:
            payload: The render payload to analyze.

        Returns:
            RenderMode.RGB_ARRAY for RGB frames, RenderMode.GRID otherwise.
        """
        # Check for RGB payload (used by BabyAI, MiniHack, Crafter, etc.)
        if "rgb" in payload or "frame" in payload:
            return RenderMode.RGB_ARRAY
        # Check for grid payload (used by FrozenLake, Taxi, etc.)
        if "grid" in payload:
            return RenderMode.GRID
        # Default to GRID (most basic)
        return RenderMode.GRID

    def _init_renderer(self, mode: RenderMode = RenderMode.GRID) -> None:
        """Initialize the renderer strategy.

        Args:
            mode: Which render mode to initialize (GRID or RGB_ARRAY).
        """
        try:
            # Check if already initialized with this mode
            if mode == RenderMode.GRID and self._grid_renderer:
                self._switch_to_renderer(self._grid_renderer, mode)
                return
            if mode == RenderMode.RGB_ARRAY and self._rgb_renderer:
                self._switch_to_renderer(self._rgb_renderer, mode)
                return

            # Create new renderer
            if not self._renderer_registry.is_registered(mode):
                _LOGGER.warning(f"{mode} renderer not registered")
                return

            new_renderer = self._renderer_registry.create(mode, self._render_container)

            # Store in appropriate slot
            if mode == RenderMode.GRID:
                self._grid_renderer = new_renderer
            elif mode == RenderMode.RGB_ARRAY:
                self._rgb_renderer = new_renderer

            self._switch_to_renderer(new_renderer, mode)
            _LOGGER.info(f"Renderer {mode} initialized for {self._config.operator_id}")

        except Exception as e:
            _LOGGER.error(f"Failed to initialize renderer {mode}: {e}")

    def _switch_to_renderer(self, renderer: Any, mode: RenderMode) -> None:
        """Switch the active renderer displayed in the container."""
        # Hide current renderer widget if any
        if self._renderer_strategy and hasattr(self._renderer_strategy, 'widget'):
            self._renderer_strategy.widget.hide()

        # Remove placeholder if present
        if self._placeholder and self._placeholder.parent():
            self._render_layout.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None

        # Set new active renderer
        self._renderer_strategy = renderer
        self._active_render_mode = mode

        if renderer and hasattr(renderer, 'widget'):
            widget = renderer.widget
            # Qt6 best practice: Use Expanding policy for widgets that should fill space
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding
            )
            # Add widget with stretch factor=1 to ensure it expands (Qt6 layout docs)
            if widget.parent() != self._render_container:
                self._render_layout.addWidget(widget, 1)
            widget.show()

            # Qt6 best practice: Schedule deferred resize after layout settles
            # This ensures proper geometry before scaling pixmap
            QtCore.QTimer.singleShot(100, lambda: self._trigger_renderer_resize(widget))

    def _trigger_renderer_resize(self, widget: QtWidgets.QWidget) -> None:
        """Trigger a resize on the renderer widget after layout settles.

        This follows Qt6 documentation recommendation for handling initial geometry.
        """
        try:
            # Force the widget to update its geometry
            widget.updateGeometry()
            # If the renderer has a resize handler, trigger it
            if hasattr(widget, 'resizeEvent'):
                from qtpy.QtGui import QResizeEvent
                event = QResizeEvent(widget.size(), widget.size())
                widget.resizeEvent(event)
        except Exception as e:
            _LOGGER.debug(f"Renderer resize trigger: {e}")

    def _extract_render_payload(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract render payload from telemetry payload."""
        # Try render_payload first
        render_payload = payload.get("render_payload")
        if render_payload:
            return render_payload

        # Try render_payload_json
        import json
        render_payload_json = payload.get("render_payload_json")
        if render_payload_json:
            try:
                if isinstance(render_payload_json, str):
                    return json.loads(render_payload_json)
                return render_payload_json
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _update_stats_from_payload(self, payload: Dict[str, Any]) -> None:
        """Update stats labels from payload data."""
        # Extract episode/step info
        episode_index = payload.get("episode_index", payload.get("episodeIndex", 0))
        step_index = payload.get("step_index", payload.get("stepIndex", 0))

        # Use episode_reward if available (cumulative from worker), else accumulate step reward
        episode_reward = payload.get("episode_reward")
        step_reward = payload.get("reward", 0.0)

        try:
            new_episode = int(episode_index) + 1
            new_step = int(step_index) + 1

            # Reset episode reward when episode changes
            if new_episode != self._current_episode:
                self._episode_reward = 0.0

            self._current_episode = new_episode
            self._current_step = new_step

            # Prefer episode_reward from worker (already cumulative)
            if episode_reward is not None:
                self._episode_reward = float(episode_reward)
            else:
                self._episode_reward += float(step_reward)
        except (TypeError, ValueError):
            pass

        # Update labels
        self._episode_label.setText(f"Episode: {self._current_episode}")
        self._step_label.setText(f"Step: {self._current_step}")
        self._reward_label.setText(f"Reward: {self._episode_reward:.2f}")

    def reset_stats(self) -> None:
        """Reset stats for a new episode."""
        self._current_step = 0
        self._episode_reward = 0.0
        self._episode_label.setText(f"Episode: {self._current_episode}")
        self._step_label.setText("Step: 0")
        self._reward_label.setText("Reward: 0.00")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._renderer_strategy:
            if hasattr(self._renderer_strategy, 'cleanup'):
                self._renderer_strategy.cleanup()
            self._renderer_strategy = None

    @property
    def config(self) -> OperatorConfig:
        return self._config

    @property
    def status(self) -> str:
        return self._status

    @property
    def operator_id(self) -> str:
        return self._config.operator_id


__all__ = ["OperatorRenderContainer"]
