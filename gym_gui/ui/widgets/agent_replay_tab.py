"""Historical replay tab for completed agent training runs."""

from __future__ import annotations

from typing import Optional

from qtpy import QtWidgets


class AgentReplayTab(QtWidgets.QWidget):
    """Displays historical per-run replay data filtered by run_id and agent_id."""

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.run_id = run_id
        self.agent_id = agent_id

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # Header
        header = QtWidgets.QLabel(
            f"<h3>Agent Training Replay</h3>"
            f"<p><b>Run ID:</b> {self.run_id[:16]}...<br>"
            f"<b>Agent ID:</b> {self.agent_id}</p>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Placeholder for future replay integration
        info = QtWidgets.QLabel(
            "This tab will display historical replay data for this training run.\n\n"
            "Features to be implemented:\n"
            "• Per-episode metrics (reward, steps, loss)\n"
            "• Episode selector with timeline\n"
            "• Step-by-step replay slider\n"
            "• Learning curve visualization\n"
            "• Policy checkpoint metadata\n\n"
            "Data will be loaded from SQLite telemetry store filtered by run_id."
        )
        info.setWordWrap(True)
        info.setStyleSheet("background-color: #f0f0f0; padding: 12px; border-radius: 4px;")
        layout.addWidget(info)

        layout.addStretch(1)

        # Future: integrate with EpisodeReplayLoader filtered by run_id + agent_id
        # Example:
        # self._loader = EpisodeReplayLoader(telemetry_service)
        # episodes = self._loader.load_episodes_for_run(run_id, agent_id)
        # self._build_episode_selector(episodes)


__all__ = ["AgentReplayTab"]
