"""Default real-time view for agent training (combines grid + stats)."""

from typing import Any, Dict, Optional

from qtpy import QtCore, QtWidgets

from gym_gui.ui.widgets.base_telemetry_tab import BaseTelemetryTab


class AgentOnlineTab(BaseTelemetryTab):
    """Default real-time view for agent training.
    
    Combines grid rendering (for ToyText) or video (for visual envs) with live stats.
    This is the primary tab users see during training.
    """

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(run_id, agent_id, parent=parent)
        self._episodes = 0
        self._steps = 0
        self._total_reward = 0.0
        self._current_episode_reward = 0.0
        self._last_episode_index = -1

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Stats header group
        stats_group = self._create_stats_group()
        layout.addWidget(stats_group)

        # Main display area (grid or video)
        self._display_container = QtWidgets.QWidget()
        self._display_layout = QtWidgets.QVBoxLayout(self._display_container)
        self._display_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._display_container, 1)

        # Status bar
        self._status_label = QtWidgets.QLabel("Waiting for telemetry...")
        self._status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._status_label)

        self.setLayout(layout)

    def _create_stats_group(self) -> QtWidgets.QGroupBox:
        """Create stats display group."""
        group = QtWidgets.QGroupBox("Training Statistics")
        layout = QtWidgets.QGridLayout(group)

        # Row 1: Episode, Mode
        self._episode_label = QtWidgets.QLabel("Episode #: 0")
        self._mode_label = QtWidgets.QLabel("Mode: --")
        layout.addWidget(self._episode_label, 0, 0)
        layout.addWidget(self._mode_label, 0, 1)

        # Row 2: Episode Reward, Total Reward
        self._episode_reward_label = QtWidgets.QLabel("Episode Reward: 0.00")
        self._total_reward_label = QtWidgets.QLabel("Total Reward: 0.00")
        layout.addWidget(self._episode_reward_label, 1, 0)
        layout.addWidget(self._total_reward_label, 1, 1)

        # Row 3: Total Steps, Seed
        self._steps_label = QtWidgets.QLabel("Total Steps: 0")
        self._seed_label = QtWidgets.QLabel("Seed: --")
        layout.addWidget(self._steps_label, 2, 0)
        layout.addWidget(self._seed_label, 2, 1)

        return group

    def on_step(
        self,
        step: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update stats and render display from incoming step."""
        # Extract step data
        episode_index = step.get("episode_index", 0)
        reward = float(step.get("reward", 0.0))
        seed = step.get("seed")
        mode = metadata.get("mode", "--") if metadata else "--"

        # Update counters
        self._steps += 1
        self._current_episode_reward += reward
        self._total_reward += reward

        # Detect episode boundary
        if episode_index != self._last_episode_index:
            self._episodes = max(self._episodes, int(episode_index) + 1)
            self._last_episode_index = episode_index
            self._current_episode_reward = reward  # Reset for new episode

        # Update labels
        self._episode_label.setText(f"Episode #: {self._episodes}")
        self._mode_label.setText(f"Mode: {mode}")
        self._episode_reward_label.setText(f"Episode Reward: {self._current_episode_reward:.2f}")
        self._total_reward_label.setText(f"Total Reward: {self._total_reward:.2f}")
        self._steps_label.setText(f"Total Steps: {self._steps}")
        if seed is not None:
            self._seed_label.setText(f"Seed: {seed}")

        # Update status
        self._status_label.setText(f"Live: {self._steps} steps, {self._episodes} episodes")

    def on_episode(
        self,
        episode: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle episode completion."""
        # Episode data is already reflected in on_step calls
        # This is called when episode is finalized
        pass

