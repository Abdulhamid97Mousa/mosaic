"""Live grid visualization tab for agent training runs."""

from __future__ import annotations

from typing import Any, Dict, Optional

from qtpy import QtCore, QtWidgets

from gym_gui.core.enums import GameId, RenderMode
from gym_gui.rendering import RendererContext, RendererRegistry, RendererStrategy


class AgentOnlineGridTab(QtWidgets.QWidget):
    """Displays live grid rendering + episode stats for a specific agent run."""

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        renderer_registry: Optional[RendererRegistry] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.run_id = run_id
        self.agent_id = agent_id
        self._episodes = 0
        self._steps = 0
        self._total_reward = 0.0
        self._current_episode_reward = 0.0
        self._game_id: Optional[GameId] = None
        self._renderer_strategy: Optional[RendererStrategy] = None
        self._renderer_registry = renderer_registry

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header with run/agent info
        header = QtWidgets.QHBoxLayout()
        self._run_label = QtWidgets.QLabel(f"<b>Run:</b> {self.run_id[:12]}...")
        self._agent_label = QtWidgets.QLabel(f"<b>Agent:</b> {self.agent_id}")
        header.addWidget(self._run_label)
        header.addWidget(self._agent_label)
        header.addStretch()
        layout.addLayout(header)

        # Stats panel
        stats_group = QtWidgets.QGroupBox("Training Statistics", self)
        stats_layout = QtWidgets.QGridLayout(stats_group)
        
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

    def on_step(self, step: Dict[str, Any]) -> None:
        """Update stats and render grid from incoming step."""
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

        # Render grid if available
        self._render_grid(step)

    def _render_grid(self, payload: Dict[str, Any]) -> None:
        """Render grid visualization using renderer registry."""
        if self._renderer_registry is None:
            return
        
        mode = payload.get("mode")
        # Accept both "ansi" string and GRID render mode
        if mode not in ("ansi", "grid", RenderMode.GRID):
            return
        
        # Initialize renderer on first grid payload
        if self._renderer_strategy is None:
            if not self._renderer_registry.is_registered(RenderMode.GRID):
                return
            self._renderer_strategy = self._renderer_registry.create(RenderMode.GRID, self._grid_container)
            self._grid_layout.removeWidget(self._placeholder_label)
            self._placeholder_label.deleteLater()
            self._grid_layout.addWidget(self._renderer_strategy.widget)
        
        # Extract game_id from payload
        raw_game = payload.get("game_id")
        if raw_game and self._game_id is None:
            try:
                self._game_id = raw_game if isinstance(raw_game, GameId) else GameId(str(raw_game))
            except ValueError:
                pass
        
        # Render
        if self._renderer_strategy and self._renderer_strategy.supports(payload):
            context = RendererContext(game_id=self._game_id)
            self._renderer_strategy.render(payload, context=context)


__all__ = ["AgentOnlineGridTab"]
