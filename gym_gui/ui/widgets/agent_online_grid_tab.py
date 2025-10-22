"""Live grid visualization tab for agent training runs."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from qtpy import QtCore, QtWidgets

from gym_gui.core.enums import GameId, RenderMode
from gym_gui.rendering import RendererContext, RendererRegistry, RendererStrategy, create_default_renderer_registry
from gym_gui.ui.widgets.base_telemetry_tab import BaseTelemetryTab

_LOGGER = logging.getLogger(__name__)


class AgentOnlineGridTab(BaseTelemetryTab):
    """Displays live grid rendering + episode stats for a specific agent run."""

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        game_id: Optional[GameId] = None,
        renderer_registry: Optional[RendererRegistry] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        self._episodes = 0
        self._steps = 0
        self._total_reward = 0.0
        self._current_episode_reward = 0.0
        self._game_id: Optional[GameId] = game_id
        self._renderer_strategy: Optional[RendererStrategy] = None
        self._renderer_registry = renderer_registry or create_default_renderer_registry()
        self._seed: Optional[int] = None
        self._control_mode: Optional[str] = None

        super().__init__(run_id, agent_id, parent=parent)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Use inherited header builder
        layout.addLayout(self._build_header())

        # Use inherited stats group builder
        stats_group, stats_layout = self._build_stats_group()
        
        stats_layout.addWidget(QtWidgets.QLabel("<b>Episodes:</b>"), 0, 0)
        self._episodes_label = QtWidgets.QLabel("0")
        stats_layout.addWidget(self._episodes_label, 0, 1)
        
        stats_layout.addWidget(QtWidgets.QLabel("<b>Total Steps:</b>"), 0, 2)
        self._steps_label = QtWidgets.QLabel("0")
        stats_layout.addWidget(self._steps_label, 0, 3)
        
        stats_layout.addWidget(QtWidgets.QLabel("<b>Episode Reward:</b>"), 1, 0)
        self._episode_reward_label = QtWidgets.QLabel("0.00")
        stats_layout.addWidget(self._episode_reward_label, 1, 1)
        
        stats_layout.addWidget(QtWidgets.QLabel("<b>Total Reward:</b>"), 1, 2)
        self._total_reward_label = QtWidgets.QLabel("0.00")
        stats_layout.addWidget(self._total_reward_label, 1, 3)

        stats_layout.addWidget(QtWidgets.QLabel("<b>Seed:</b>"), 2, 0)
        self._seed_label = QtWidgets.QLabel("—")
        stats_layout.addWidget(self._seed_label, 2, 1)

        stats_layout.addWidget(QtWidgets.QLabel("<b>Mode:</b>"), 2, 2)
        self._mode_label = QtWidgets.QLabel("—")
        stats_layout.addWidget(self._mode_label, 2, 3)
        
        layout.addWidget(stats_group)

        # Grid renderer placeholder
        self._grid_container = QtWidgets.QWidget(self)
        self._grid_layout = QtWidgets.QVBoxLayout(self._grid_container)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder_label = QtWidgets.QLabel("Waiting for grid telemetry…", self._grid_container)
        self._placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._grid_layout.addWidget(self._placeholder_label)
        self._pending = QtWidgets.QLabel("Waiting for live telemetry…", self)
        self._pending.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._pending)
        layout.addWidget(self._grid_container, 1)

    def on_step(self, step: Dict[str, Any], *, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update stats and render grid from incoming step."""
        # DEBUG: Log step keys
        step_keys = list(step.keys()) if isinstance(step, dict) else "not_dict"
        has_render = "render_payload_json" in step if isinstance(step, dict) else False
        _LOGGER.debug(f"[on_step] step keys: {step_keys}, has_render_payload_json={has_render}")

        # Update counters
        if self._pending.isVisible():
            self._pending.hide()
        self._steps += 1
        reward = float(step.get("reward", 0.0))
        self._current_episode_reward += reward
        self._total_reward += reward
        
        episode_index = step.get("episode_index")
        if episode_index is not None:
            self._episodes = max(self._episodes, int(episode_index) + 1)
        
        # Reset episode reward on new episode
        terminated = step.get("terminated", False)
        truncated = step.get("truncated", False)
        if terminated or truncated:
            self._current_episode_reward = 0.0
        
        # Update labels
        self._episodes_label.setText(str(self._episodes))
        self._steps_label.setText(str(self._steps))
        self._episode_reward_label.setText(f"{self._current_episode_reward:.2f}")
        self._total_reward_label.setText(f"{self._total_reward:.2f}")

        if metadata:
            self.update_metadata(metadata)

        # Render grid if available
        self._render_grid(step)

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        seed = metadata.get("seed")
        if seed is not None:
            try:
                self._seed = int(seed)
                self._seed_label.setText(str(self._seed))
            except (TypeError, ValueError):
                self._seed_label.setText(str(seed))
        control_mode = metadata.get("control_mode") or metadata.get("mode")
        if control_mode:
            self._control_mode = str(control_mode)
            self._mode_label.setText(self._control_mode)
        game_id = metadata.get("game_id")
        if game_id:
            try:
                self._game_id = GameId(str(game_id))
            except ValueError:
                self._game_id = None
            if self._game_id is not None:
                self._run_label.setText(f"<b>Run:</b> {self.run_id[:12]}… • Game: {self._game_id.value}")
            else:
                self._run_label.setText(f"<b>Run:</b> {self.run_id[:12]}…")

    def _render_grid(self, payload: Dict[str, Any]) -> None:
        """Render grid visualization using renderer registry."""
        if self._renderer_registry is None:
            _LOGGER.debug(f"[_render_grid] renderer_registry is None, returning")
            return

        # DEBUG: Log what keys are in the payload
        _LOGGER.debug(f"[_render_grid] payload keys: {list(payload.keys())}")

        # CRITICAL: Extract render_payload from step payload
        # Try render_payload_json first (from protobuf), then render_payload (from dict)
        render_payload = payload.get("render_payload")

        # If render_payload_json exists (from protobuf), parse it
        if render_payload is None:
            render_payload_json = payload.get("render_payload_json")
            _LOGGER.debug(f"[_render_grid] render_payload_json present: {render_payload_json is not None}")
            if render_payload_json:
                try:
                    if isinstance(render_payload_json, str):
                        render_payload = json.loads(render_payload_json)
                    else:
                        render_payload = render_payload_json
                    _LOGGER.debug(f"[_render_grid] Parsed render_payload_json, keys: {list(render_payload.keys()) if isinstance(render_payload, dict) else 'not_dict'}")
                except (json.JSONDecodeError, TypeError) as e:
                    _LOGGER.debug(f"[_render_grid] Failed to parse render_payload_json: {e}")
                    render_payload = None

        if render_payload is None:
            _LOGGER.debug(f"[_render_grid] render_payload is None, returning")
            return

        # Check if render_payload has grid mode
        mode = render_payload.get("mode")
        _LOGGER.debug(f"[_render_grid] mode: {mode}")
        # Accept both "ansi" string and GRID render mode
        if mode not in ("ansi", "grid", RenderMode.GRID):
            _LOGGER.debug(f"[_render_grid] mode not supported: {mode}")
            return

        # Initialize renderer on first grid payload
        if self._renderer_strategy is None:
            _LOGGER.debug(f"[_render_grid] Initializing renderer strategy")
            if not self._renderer_registry.is_registered(RenderMode.GRID):
                _LOGGER.debug(f"[_render_grid] GRID renderer not registered")
                return
            self._renderer_strategy = self._renderer_registry.create(RenderMode.GRID, self._grid_container)
            _LOGGER.debug(f"[_render_grid] Renderer strategy created: {self._renderer_strategy}")
            self._grid_layout.removeWidget(self._placeholder_label)
            self._placeholder_label.deleteLater()
            self._grid_layout.addWidget(self._renderer_strategy.widget)
            _LOGGER.debug(f"[_render_grid] Renderer widget added to layout")

        # Extract game_id from render_payload
        raw_game = render_payload.get("game_id")
        if raw_game and self._game_id is None:
            try:
                self._game_id = raw_game if isinstance(raw_game, GameId) else GameId(str(raw_game))
                _LOGGER.debug(f"[_render_grid] Set game_id to: {self._game_id}")
            except ValueError:
                _LOGGER.debug(f"[_render_grid] Failed to parse game_id: {raw_game}")
                pass

        # Render using the render_payload (not the step payload)
        if self._renderer_strategy and self._renderer_strategy.supports(render_payload):
            _LOGGER.debug(f"[_render_grid] Calling renderer.render()")
            context = RendererContext(game_id=self._game_id)
            self._renderer_strategy.render(render_payload, context=context)
            _LOGGER.debug(f"[_render_grid] Render complete")
        else:
            _LOGGER.debug(f"[_render_grid] Renderer doesn't support payload or is None")


__all__ = ["AgentOnlineGridTab"]
