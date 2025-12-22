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
from gym_gui.core.enums import RenderMode

_LOGGER = logging.getLogger(__name__)


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
        "running": "#4CAF50",    # Green
        "stopped": "#F44336",    # Red
        "error": "#FF9800",      # Orange
    }

    # Type badge colors
    TYPE_COLORS = {
        "llm": "#2196F3",  # Blue
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
        self._renderer_strategy: Optional[Any] = None  # Currently active renderer
        self._active_render_mode: Optional[RenderMode] = None

        # Stats tracking
        self._current_step = 0
        self._current_episode = 0
        self._total_reward = 0.0
        self._episode_reward = 0.0

        self._build_ui()
        self._update_header()

    def _build_ui(self) -> None:
        self.setFrameStyle(QtWidgets.QFrame.Shape.Box | QtWidgets.QFrame.Shadow.Raised)
        self.setLineWidth(1)
        self.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 4px; }")

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

        # Type badge (LLM / RL)
        self._type_badge = QtWidgets.QLabel(self._config.operator_type.upper(), self._header)
        type_color = self.TYPE_COLORS.get(self._config.operator_type, "#666")
        self._type_badge.setStyleSheet(
            f"background-color: {type_color}; color: white; "
            f"padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold;"
        )
        header_layout.addWidget(self._type_badge)

        # Environment/Task info
        env_task = f"{self._config.env_name} / {self._config.task}"
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
        self._render_container.setMinimumSize(200, 200)
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
        env_task = f"{self._config.env_name} / {self._config.task}"
        self._env_label.setText(env_task)

    def _update_status_indicator(self) -> None:
        """Update status indicator appearance."""
        status_color = self.STATUS_COLORS.get(self._status, "#666")
        self._status_indicator.setText(self._status.upper())
        self._status_indicator.setStyleSheet(
            f"background-color: {status_color}; color: white; "
            f"padding: 2px 6px; border-radius: 3px; font-size: 9px; font-weight: bold;"
        )

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

    def display_payload(self, payload: Dict[str, Any]) -> None:
        """Display a render payload from telemetry.

        Args:
            payload: Telemetry payload containing render data.
        """
        try:
            # Update stats from payload
            self._update_stats_from_payload(payload)

            # Extract render data
            render_payload = self._extract_render_payload(payload)
            if render_payload is None:
                return

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
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding
            )
            # Add widget if not already in layout
            if widget.parent() != self._render_container:
                self._render_layout.addWidget(widget)
            widget.show()

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
        reward = payload.get("reward", 0.0)

        try:
            self._current_episode = int(episode_index) + 1
            self._current_step = int(step_index) + 1
            self._episode_reward += float(reward)
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
