"""Historical replay tab for completed agent training runs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from qtpy import QtCore, QtWidgets

from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.services.service_locator import get_service_locator
from gym_gui.telemetry import TelemetrySQLiteStore


class AgentReplayTab(QtWidgets.QWidget):
    """Display telemetry captured for a specific training run and agent."""

    _MAX_EPISODES = 500

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        telemetry_store: Optional[TelemetrySQLiteStore] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.run_id = run_id
        self.agent_id = agent_id

        locator = get_service_locator()
        self._store: Optional[TelemetrySQLiteStore] = (
            telemetry_store or locator.resolve(TelemetrySQLiteStore)
        )

        self._episodes: List[EpisodeRollup] = []
        self._steps_cache: Dict[str, Sequence[StepRecord]] = {}

        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QLabel(
            f"<h3>Agent Replay</h3>"
            f"<p><b>Run:</b> {self.run_id}<br>"
            f"<b>Agent:</b> {self.agent_id}</p>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Summary + actions
        actions_layout = QtWidgets.QHBoxLayout()
        self._summary_label = QtWidgets.QLabel("No telemetry available.")
        self._summary_label.setWordWrap(True)
        actions_layout.addWidget(self._summary_label, 1)

        self._refresh_button = QtWidgets.QPushButton("Refresh")
        self._refresh_button.clicked.connect(self.refresh)
        actions_layout.addWidget(self._refresh_button, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addLayout(actions_layout)

        # Episode table
        self._episode_table = QtWidgets.QTableWidget(self)
        self._episode_table.setColumnCount(8)
        self._episode_table.setHorizontalHeaderLabels(
            [
                "Episode",
                "Reward",
                "Steps",
                "Result",
                "Seed",
                "Control Mode",
                "Game",
                "Timestamp (UTC)",
            ]
        )
        self._episode_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._episode_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._episode_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        header_view = self._episode_table.horizontalHeader()
        if header_view is not None:
            header_view.setStretchLastSection(True)
        self._episode_table.itemSelectionChanged.connect(self._on_episode_selected)
        layout.addWidget(self._episode_table, 2)

        # Step detail pane
        self._step_summary = QtWidgets.QLabel("Select an episode to inspect the recorded steps.")
        self._step_summary.setWordWrap(True)
        layout.addWidget(self._step_summary)

        self._step_view = QtWidgets.QPlainTextEdit(self)
        self._step_view.setReadOnly(True)
        self._step_view.setMinimumHeight(200)
        layout.addWidget(self._step_view, 1)

        # Placeholder shown when no data
        self._placeholder = QtWidgets.QLabel(
            "No telemetry data has been recorded for this training run yet."
        )
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet("color: #607D8B; font-style: italic;")
        layout.addWidget(self._placeholder)

    # ------------------------------------------------------------------
    # Data loading & rendering
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Reload telemetry from persistent storage."""

        if self._store is None:
            self._summary_label.setText("Telemetry store is unavailable.")
            self._episode_table.setRowCount(0)
            self._step_view.clear()
            self._toggle_placeholder(True)
            return

        episodes = self._store.episodes_for_run(
            self.run_id,
            agent_id=self.agent_id,
            limit=self._MAX_EPISODES,
            order_desc=False,
        )
        self._episodes = list(episodes)
        self._steps_cache.clear()
        self._populate_episode_table()

    def _populate_episode_table(self) -> None:
        self._episode_table.setRowCount(0)

        if not self._episodes:
            self._summary_label.setText("No telemetry episodes have been recorded yet.")
            self._toggle_placeholder(True)
            self._step_view.clear()
            return

        total_reward = sum(episode.total_reward for episode in self._episodes)
        success_count = sum(
            1 for episode in self._episodes if self._episode_success(episode)
        )
        summary = (
            f"{len(self._episodes)} episode(s) • "
            f"Σ reward = {total_reward:.2f} • "
            f"successes = {success_count}"
        )
        self._summary_label.setText(summary)

        self._episode_table.setRowCount(len(self._episodes))
        for row, episode in enumerate(self._episodes):
            episode_index = self._episode_index(episode, fallback=row + 1)
            reward_item = QtWidgets.QTableWidgetItem(f"{episode.total_reward:.2f}")
            reward_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            steps_item = QtWidgets.QTableWidgetItem(str(episode.steps))
            steps_item.setTextAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            )

            timestamp = episode.timestamp if isinstance(episode.timestamp, datetime) else None
            timestamp_display = timestamp.isoformat(timespec="seconds") if timestamp else "—"

            metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
            seed_display = metadata.get("seed", "—")
            control_mode = metadata.get("control_mode") or metadata.get("mode") or "—"
            game_label = metadata.get("game_id") or "—"

            cells = [
                QtWidgets.QTableWidgetItem(str(episode_index)),
                reward_item,
                steps_item,
                QtWidgets.QTableWidgetItem(self._episode_result_label(episode)),
                QtWidgets.QTableWidgetItem(str(seed_display)),
                QtWidgets.QTableWidgetItem(str(control_mode)),
                QtWidgets.QTableWidgetItem(str(game_label)),
                QtWidgets.QTableWidgetItem(timestamp_display),
            ]
            for column, item in enumerate(cells):
                item.setData(QtCore.Qt.ItemDataRole.UserRole, episode.episode_id)
                if metadata:
                    item.setToolTip(json.dumps(metadata, indent=2))
                self._episode_table.setItem(row, column, item)

        self._toggle_placeholder(False)
        self._episode_table.selectRow(0)

    def _on_episode_selected(self) -> None:
        row = self._episode_table.currentRow()
        if row < 0 or row >= len(self._episodes):
            self._step_view.clear()
            self._step_summary.setText("Select an episode to inspect the recorded steps.")
            return

        episode = self._episodes[row]
        if episode.episode_id not in self._steps_cache:
            steps = self._store.episode_steps(episode.episode_id) if self._store else ()
            self._steps_cache[episode.episode_id] = steps
        steps = self._steps_cache[episode.episode_id]
        self._render_step_details(episode, steps)

    def _render_step_details(
        self,
        episode: EpisodeRollup,
        steps: Sequence[StepRecord],
    ) -> None:
        metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
        episode_index = self._episode_index(episode, fallback=0)
        header = (
            f"Episode {episode_index} • Reward {episode.total_reward:.2f} • "
            f"Steps {episode.steps} • Success: {self._episode_success(episode)}"
        )
        seed = metadata.get("seed")
        if seed is not None:
            header += f" • Seed {seed}"
        control_mode = metadata.get("control_mode") or metadata.get("mode")
        if control_mode:
            header += f" • Mode {control_mode}"
        game = metadata.get("game_id")
        if game:
            header += f" • Game {game}"
        self._step_summary.setText(header)

        if not steps:
            self._step_view.setPlainText("No step telemetry recorded.")
            return

        lines: List[str] = []
        for step in steps:
            line = (
                f"Step {step.step_index:03d}: action={step.action} "
                f"reward={step.reward:.3f} "
                f"terminated={'Y' if step.terminated else 'N'} "
                f"truncated={'Y' if step.truncated else 'N'}"
            )
            lines.append(line)

            info_payload = step.info or {}
            if info_payload:
                info_preview = json.dumps(info_payload, separators=(",", ":"), default=str)
                if len(info_preview) > 160:
                    info_preview = info_preview[:157] + "…"
                lines.append(f"  info={info_preview}")

        self._step_view.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _toggle_placeholder(self, visible: bool) -> None:
        self._placeholder.setVisible(visible)
        self._episode_table.setVisible(not visible)
        self._step_view.setVisible(not visible)
        self._step_summary.setVisible(not visible)

    @staticmethod
    def _episode_index(episode: EpisodeRollup, fallback: int) -> int:
        metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
        try:
            return int(metadata.get("episode_index", fallback))
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _episode_success(episode: EpisodeRollup) -> bool:
        metadata = episode.metadata if isinstance(episode.metadata, dict) else {}
        if "success" in metadata:
            return bool(metadata["success"])
        return episode.terminated and not episode.truncated

    @staticmethod
    def _episode_result_label(episode: EpisodeRollup) -> str:
        if episode.terminated and not episode.truncated:
            return "Success"
        if episode.truncated:
            return "Aborted"
        return "Incomplete" if episode.steps else "—"


__all__ = ["AgentReplayTab"]
