"""Live telemetry tab widget for displaying streamed run data."""

from __future__ import annotations

from collections import deque
import json
from typing import Any, Deque, Optional

from qtpy import QtCore, QtWidgets


class LiveTelemetryTab(QtWidgets.QWidget):
    """Displays live telemetry stream for a specific (run_id, agent_id) pair."""

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        buffer_size: int = 100,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.run_id = run_id
        self.agent_id = agent_id
        self._step_buffer: Deque[Any] = deque(maxlen=buffer_size)
        self._episode_buffer: Deque[Any] = deque(maxlen=10)
        self._dropped_steps = 0
        self._dropped_episodes = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header with run/agent info
        header = QtWidgets.QHBoxLayout()
        self._run_label = QtWidgets.QLabel(f"<b>Run:</b> {self.run_id[:12]}...")
        self._agent_label = QtWidgets.QLabel(f"<b>Agent:</b> {self.agent_id}")
        self._stats_label = QtWidgets.QLabel("Steps: 0 | Episodes: 0")
        header.addWidget(self._run_label)
        header.addWidget(self._agent_label)
        header.addStretch()
        header.addWidget(self._stats_label)
        layout.addLayout(header)

        # Splitter for steps and episodes
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        layout.addWidget(splitter)

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
        self._episodes_table = QtWidgets.QTableWidget(0, 5, episodes_group)
        self._episodes_table.setHorizontalHeaderLabels([
            "Episode",
            "Steps",
            "Reward",
            "Terminated",
            "Truncated",
        ])
        header = self._episodes_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        episodes_layout.addWidget(self._episodes_table)
        splitter.addWidget(episodes_group)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        # Footer with overflow stats
        self._overflow_label = QtWidgets.QLabel("")
        self._overflow_label.setStyleSheet("color: #d32f2f;")
        layout.addWidget(self._overflow_label)

    def add_step(self, payload: Any) -> None:
        """Add a step to the buffer and update display lazily."""
        self._step_buffer.append(payload)
        self._update_stats()
        # Only render the most recent step to keep UI responsive
        self._render_latest_step(payload)

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
        
        # Store raw JSON, parse only when needed
        action_json = getattr(payload, "action_json", "")
        observation_json = getattr(payload, "observation_json", "")
        
        line = (
            f"[ep{episode_index:04d} #{int(step_index):04d}] "
            f"r={reward:+.3f} term={terminated} trunc={truncated} "
            f"a={self._preview(action_json)} o={self._preview(observation_json)}"
        )
        self._steps_view.appendPlainText(line)

    def add_episode(self, payload: Any) -> None:
        """Add an episode to the buffer and update table."""
        self._episode_buffer.append(payload)
        self._update_stats()
        self._render_episode_row(payload)

    def _render_episode_row(self, payload: Any) -> None:
        """Add a row to the episodes table."""
        episode_index = getattr(payload, "episode_index", -1)
        steps = getattr(payload, "steps", 0)
        total_reward = getattr(payload, "total_reward", 0.0)
        terminated = getattr(payload, "terminated", False)
        truncated = getattr(payload, "truncated", False)

        row = self._episodes_table.rowCount()
        self._episodes_table.insertRow(row)
        self._episodes_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(episode_index)))
        self._episodes_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(steps)))
        self._episodes_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{total_reward:.2f}"))
        self._episodes_table.setItem(row, 3, QtWidgets.QTableWidgetItem("✓" if terminated else "✗"))
        self._episodes_table.setItem(row, 4, QtWidgets.QTableWidgetItem("✓" if truncated else "✗"))

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


__all__ = ["LiveTelemetryTab"]
