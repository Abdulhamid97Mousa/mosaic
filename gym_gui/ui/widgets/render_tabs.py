"""Composite widget that presents live render streams alongside telemetry replays."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, List, Mapping, Optional

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.core.enums import GameId, RenderMode
from gym_gui.rendering.grid_renderer import GridRenderer
from gym_gui.replays import EpisodeReplay, EpisodeReplayLoader
from gym_gui.services.telemetry import TelemetryService


class RenderTabs(QtWidgets.QTabWidget):
    """Tab widget combining grid, raw text, video, and replay views."""

    _current_game: GameId | None

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        telemetry_service: TelemetryService | None = None,
    ) -> None:
        super().__init__(parent)

        self._grid_tab = _GridTab(parent=self)
        self._raw_tab = _RawTab(parent=self)
        self._video_tab = _VideoTab(parent=self)
        self._replay_tab = _ReplayTab(parent=self, telemetry_service=telemetry_service)

        self.addTab(self._grid_tab.widget, "Grid")
        self.addTab(self._raw_tab.widget, "Raw")
        self.addTab(self._video_tab.widget, "Video")
        self.addTab(self._replay_tab, "Replay")

        # Disable grid/video tabs until data arrives
        self.setTabEnabled(self.indexOf(self._grid_tab.widget), False)
        self.setTabEnabled(self.indexOf(self._video_tab.widget), False)
        self._current_game = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_current_game(self, game_id: GameId) -> None:
        self._current_game = game_id
        self._grid_tab.set_current_game(game_id)
        self._replay_tab.set_current_game(game_id)

    def display_payload(self, payload: object) -> None:
        if isinstance(payload, Mapping):
            mode = payload.get("mode")
            if "game_id" in payload and payload["game_id"]:
                try:
                    self._current_game = GameId(payload["game_id"])
                    self._grid_tab.set_current_game(self._current_game)
                    self._replay_tab.set_current_game(self._current_game)
                except Exception:
                    pass

            if mode == RenderMode.GRID.value and "grid" in payload:
                self._grid_tab.render_grid(payload)
                self._raw_tab.display_from_payload(payload)
                self._activate_tab(self._grid_tab.widget)
            elif mode == RenderMode.RGB_ARRAY.value and "rgb" in payload:
                self._video_tab.render_frame(payload["rgb"], ansi=payload.get("ansi"))
                self._raw_tab.display_from_payload(payload)
                self._activate_tab(self._video_tab.widget)
            else:
                text = payload.get("ansi") or payload.get("text") or str(payload)
                self._raw_tab.display_plain_text(text)
                self._activate_tab(self._raw_tab.widget, enable_only=True)
        elif payload is None:
            self._raw_tab.display_plain_text("No render payload yet.")
            self._activate_tab(self._raw_tab.widget, enable_only=True)
        else:
            self._raw_tab.display_plain_text(str(payload))
            self._activate_tab(self._raw_tab.widget, enable_only=True)

    def refresh_replays(self) -> None:
        self._replay_tab.refresh()

    def on_episode_finished(self) -> None:
        self._replay_tab.refresh()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _activate_tab(self, widget: QtWidgets.QWidget, *, enable_only: bool = False) -> None:
        index = self.indexOf(widget)
        if index == -1:
            return
        self.setTabEnabled(index, True)
        if not enable_only:
            self.setCurrentIndex(index)


@dataclass(slots=True)
class _GridTab:
    widget: QtWidgets.QGraphicsView
    _renderer: GridRenderer
    _current_game: GameId | None = None

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        view = QtWidgets.QGraphicsView(parent)
        view.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        view.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(240, 240, 240)))
        self.widget = view
        self._renderer = GridRenderer(self.widget)
        self._current_game = None

    def set_current_game(self, game_id: GameId | None) -> None:
        self._current_game = game_id

    def render_grid(self, payload: Mapping[str, Any]) -> None:
        raw_grid = payload.get("grid")
        if raw_grid is None:
            return
        game_id = self._current_game or GameId.FROZEN_LAKE
        agent_pos = payload.get("agent_position")
        taxi_state = payload.get("taxi_state")
        terminated = payload.get("terminated", False)

        rows: List[List[str]] = []
        for row in list(raw_grid):
            if isinstance(row, str):
                rows.append(list(row))
            else:
                rows.append([str(cell) for cell in list(row)])

        self._renderer.render(rows, game_id, agent_pos, taxi_state, terminated, dict(payload))


@dataclass(slots=True)
class _RawTab:
    widget: QtWidgets.QPlainTextEdit

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        editor = QtWidgets.QPlainTextEdit(parent)
        editor.setReadOnly(True)
        editor.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.widget = editor

    def display_from_payload(self, payload: Mapping[str, Any]) -> None:
        ansi = payload.get("ansi")
        if ansi:
            self.widget.setPlainText(_strip_ansi_codes(ansi))

    def display_plain_text(self, text: str) -> None:
        self.widget.setPlainText(text)


class _VideoTab(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        container = QtWidgets.QScrollArea(self)
        container.setWidgetResizable(True)
        container.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        label = QtWidgets.QLabel(container)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(320, 240)
        label.setStyleSheet("background-color: #111; color: #eee;")

        container.setWidget(label)
        layout.addWidget(container)

        self._label = label
        self._container = container
        self._current_pixmap: QtGui.QPixmap | None = None

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self

    def render_frame(self, frame: Any, *, ansi: str | None = None) -> None:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            self._label.setText("Unsupported RGB frame format")
            return

        array = np.ascontiguousarray(array)
        height, width, channels = array.shape
        if channels == 3:
            fmt = QtGui.QImage.Format.Format_RGB888
        else:
            fmt = QtGui.QImage.Format.Format_RGBA8888

        qimage = QtGui.QImage(array.data, width, height, width * channels, fmt).copy()
        self._current_pixmap = QtGui.QPixmap.fromImage(qimage)
        self._scale_pixmap()

        if ansi:
            # Keep textual representation in sync by showing it as tooltip
            self._label.setToolTip(_strip_ansi_codes(ansi))
        else:
            self._label.setToolTip("")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # pragma: no cover - GUI only
        super().resizeEvent(event)
        self._scale_pixmap()

    def _scale_pixmap(self) -> None:
        if self._current_pixmap is None:
            return
        if self._label.width() <= 0 or self._label.height() <= 0:
            self._label.setPixmap(self._current_pixmap)
            return
        scaled = self._current_pixmap.scaled(
            self._label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)


class _ReplayPreview(QtWidgets.QStackedWidget):
    """Mini viewer that mirrors replay frames for quick inspection."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid_tab = _GridTab(parent=self)
        self._video_tab = _VideoTab(parent=self)
        self._text_view = QtWidgets.QPlainTextEdit(self)
        self._text_view.setReadOnly(True)
        self._text_view.setMinimumHeight(120)
        self._placeholder = QtWidgets.QLabel("No render data available for this step.", self)
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)

        self._placeholder_index = self.addWidget(self._placeholder)
        self._grid_index = self.addWidget(self._grid_tab.widget)
        self._video_index = self.addWidget(self._video_tab)
        self._text_index = self.addWidget(self._text_view)

        self._current_game: GameId | None = None
        self.setCurrentIndex(self._placeholder_index)

    def set_current_game(self, game: GameId | None) -> None:
        self._current_game = game
        self._grid_tab.set_current_game(game)

    def clear(self) -> None:
        self.setCurrentIndex(self._placeholder_index)
        self._text_view.clear()

    def display(self, payload: Any, game: GameId | None = None) -> None:
        if game is not None:
            self.set_current_game(game)
        if payload is None:
            self.clear()
            return
        if isinstance(payload, Mapping):
            mode = payload.get("mode")
            if mode == RenderMode.GRID.value and "grid" in payload:
                self._grid_tab.render_grid(payload)
                self.setCurrentIndex(self._grid_index)
                return
            if mode == RenderMode.RGB_ARRAY.value and "rgb" in payload:
                self._video_tab.render_frame(payload["rgb"], ansi=payload.get("ansi"))
                self.setCurrentIndex(self._video_index)
                return
            ansi = payload.get("ansi")
            if ansi:
                self._text_view.setPlainText(_strip_ansi_codes(ansi))
                self.setCurrentIndex(self._text_index)
                return
        try:
            self._text_view.setPlainText(str(payload))
        except Exception:
            self._text_view.setPlainText("Unsupported render payload")
        self.setCurrentIndex(self._text_index)


class _ReplayTab(QtWidgets.QWidget):
    """Displays recent episodes from telemetry for quick replay selection."""

    _telemetry: TelemetryService | None
    _loader: EpisodeReplayLoader | None
    _current_game: GameId | None
    _load_button: QtWidgets.QPushButton
    _delete_button: QtWidgets.QPushButton
    _clear_button: QtWidgets.QPushButton
    _order_button: QtWidgets.QPushButton
    _episodes: QtWidgets.QTableWidget
    _placeholder: QtWidgets.QLabel
    _details_group: QtWidgets.QGroupBox
    _episode_summary: QtWidgets.QLabel
    _preview: _ReplayPreview
    _slider: QtWidgets.QSlider
    _step_label: QtWidgets.QLabel
    _step_view: QtWidgets.QPlainTextEdit
    _current_replay: Optional[EpisodeReplay]
    _current_replay_game: GameId | None

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        telemetry_service: TelemetryService | None = None,
    ) -> None:
        super().__init__(parent)
        self._telemetry = telemetry_service
        self._loader = EpisodeReplayLoader(telemetry_service) if telemetry_service else None
        self._current_game = None
        self._sort_descending = True

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._details_group = QtWidgets.QGroupBox("Episode Playback", self)
        details_layout = QtWidgets.QVBoxLayout(self._details_group)
        details_layout.setContentsMargins(8, 8, 8, 8)

        self._episode_summary = QtWidgets.QLabel("Select an episode to load its replay.")
        self._episode_summary.setWordWrap(True)
        details_layout.addWidget(self._episode_summary)

        self._preview = _ReplayPreview(self._details_group)
        self._preview.setMinimumHeight(220)
        details_layout.addWidget(self._preview)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self._details_group)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(1)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        details_layout.addWidget(self._slider)

        self._step_label = QtWidgets.QLabel("Step 0 / 0")
        details_layout.addWidget(self._step_label)

        self._step_view = QtWidgets.QPlainTextEdit(self._details_group)
        self._step_view.setReadOnly(True)
        self._step_view.setMinimumHeight(160)
        details_layout.addWidget(self._step_view)

        layout.addWidget(self._details_group)

        footer = QtWidgets.QHBoxLayout()
        footer_label = QtWidgets.QLabel("Recent Episodes")
        footer.addWidget(footer_label)
        self._order_button = QtWidgets.QPushButton(self)
        self._order_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._order_button.clicked.connect(self._toggle_sort_order)
        footer.addWidget(self._order_button)
        footer.addStretch(1)
        self._load_button = QtWidgets.QPushButton("Load Replay")
        self._load_button.clicked.connect(self._load_selected_episode)
        self._load_button.setEnabled(False)
        footer.addWidget(self._load_button)
        self._copy_button = QtWidgets.QPushButton("Copy to Clipboard")
        self._copy_button.clicked.connect(self._copy_table_to_clipboard)
        self._copy_button.setEnabled(False)
        footer.addWidget(self._copy_button)
        self._delete_button = QtWidgets.QPushButton("Delete Selected")
        self._delete_button.clicked.connect(self._delete_selected_episode)
        self._delete_button.setEnabled(False)
        footer.addWidget(self._delete_button)
        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        footer.addWidget(refresh_button)
        self._clear_button = QtWidgets.QPushButton("Clear All")
        self._clear_button.clicked.connect(self._clear_all_episodes)
        self._clear_button.setEnabled(False)
        footer.addWidget(self._clear_button)
        layout.addLayout(footer)

        self._update_order_button()

        self._episodes = QtWidgets.QTableWidget(0, 7, self)
        self._episodes.setHorizontalHeaderLabels([
            "Seed",
            "Episode",
            "Game",
            "Steps",
            "Reward",
            "Outcome",
            "Timestamp",
        ])
        header = self._episodes.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self._episodes.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._episodes.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._episodes.itemSelectionChanged.connect(self._on_episode_selection_changed)
        self._episodes.itemDoubleClicked.connect(lambda *_: self._load_selected_episode())
        layout.addWidget(self._episodes, 1)

        self._placeholder = QtWidgets.QLabel(
            "Telemetry playback data will appear here once episodes are recorded.",
            self,
        )
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        layout.addWidget(self._placeholder)

        self._update_placeholder_visibility()
        self._current_replay = None
        self._current_replay_game = None

    def set_current_game(self, game_id: GameId | None) -> None:
        self._current_game = game_id
        self._preview.set_current_game(game_id)

    def refresh(self) -> None:
        records = self._fetch_recent_episodes()
        records.sort(
            key=lambda record: record.get("timestamp_sort", datetime.min),
            reverse=self._sort_descending,
        )
        self._episodes.setRowCount(0)
        for display_index, record in enumerate(records, start=1):
            row = self._episodes.rowCount()
            self._episodes.insertRow(row)
            episode_label = record.get("episode_index")
            episode_display = (
                str(episode_label)
                if episode_label is not None
                else str(display_index)
            )
            display_values = [
                str(record["seed"]),
                episode_display,
                record.get("game", "—"),
                str(record["steps"]),
                str(record["reward"]),
                str(record["terminated"]),
                str(record["timestamp"]),
            ]
            for column, value in enumerate(display_values):
                item = QtWidgets.QTableWidgetItem(value)
                if column == 1:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, record["episode_id"])
                if column == 4:  # reward column alignment
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self._episodes.setItem(row, column, item)
        self._update_placeholder_visibility()
        self._on_episode_selection_changed()
        if self._episodes.rowCount() == 0:
            self._clear_replay_details()

    def _copy_table_to_clipboard(self) -> None:
        if self._episodes.rowCount() == 0:
            return
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is None:
            return
        headers: List[str] = []
        for column_index in range(self._episodes.columnCount()):
            header_item = self._episodes.horizontalHeaderItem(column_index)
            headers.append(header_item.text() if header_item is not None else "")
        rows: List[str] = ["\t".join(headers)]
        for row_index in range(self._episodes.rowCount()):
            row_values: List[str] = []
            for column_index in range(self._episodes.columnCount()):
                item = self._episodes.item(row_index, column_index)
                row_values.append(item.text() if item is not None else "")
            rows.append("\t".join(row_values))
        clipboard.setText("\n".join(rows))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fetch_recent_episodes(self) -> List[dict[str, Any]]:
        if self._telemetry is None:
            return []
        episodes = list(self._telemetry.recent_episodes())
        return [self._format_episode_row(ep) for ep in episodes]

    def _format_episode_row(self, episode: EpisodeRollup) -> dict[str, Any]:
        seed_value, episode_index, game_label = self._parse_episode_metadata(episode.metadata)
        return {
            "episode_id": episode.episode_id,
            "episode_index": episode_index,
            "seed": seed_value,
            "game": game_label,
            "steps": str(episode.steps),
            "reward": f"{episode.total_reward:.2f}",
            "terminated": self._termination_label(episode),
            "timestamp": episode.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_sort": episode.timestamp,
        }

    def _parse_episode_metadata(
        self, metadata: Any
    ) -> tuple[str, int | None, str]:
        if not isinstance(metadata, Mapping):
            return "—", None, "—"
        seed_value = metadata.get("seed")
        episode_index_raw = metadata.get("episode_index")
        game_label = self._resolve_game_label(metadata.get("game_id"))
        seed_display = str(seed_value) if seed_value is not None else "—"
        episode_index = None
        if episode_index_raw is not None:
            try:
                episode_index = int(episode_index_raw)
            except (TypeError, ValueError):
                episode_index = None
        return seed_display, episode_index, game_label

    def _resolve_game_label(self, raw_game: Any) -> str:
        if isinstance(raw_game, GameId):
            return raw_game.value
        if isinstance(raw_game, str):
            try:
                return GameId(raw_game).value
            except ValueError:
                return raw_game
        return "—"

    @staticmethod
    def _termination_label(episode: EpisodeRollup) -> str:
        if episode.terminated:
            return "Yes"
        if episode.truncated:
            return "Aborted"
        return "No"

    def _update_placeholder_visibility(self) -> None:
        has_rows = self._episodes.rowCount() > 0
        self._episodes.setVisible(has_rows)
        self._placeholder.setVisible(not has_rows)
        self._details_group.setVisible(has_rows)
        self._clear_button.setEnabled(has_rows and self._telemetry is not None)
        self._copy_button.setEnabled(has_rows)
        if not has_rows:
            self._load_button.setEnabled(False)
            self._delete_button.setEnabled(False)

    def _update_order_button(self) -> None:
        if self._sort_descending:
            self._order_button.setText("Newest ↓")
            self._order_button.setToolTip("Show oldest episodes first")
        else:
            self._order_button.setText("Oldest ↑")
            self._order_button.setToolTip("Show newest episodes first")

    def _toggle_sort_order(self) -> None:
        self._sort_descending = not self._sort_descending
        self._update_order_button()
        self.refresh()

    def _on_episode_selection_changed(self) -> None:
        indexes = self._episodes.selectionModel()
        has_selection = bool(indexes and indexes.hasSelection())
        self._load_button.setEnabled(has_selection)
        self._delete_button.setEnabled(has_selection and self._telemetry is not None)

    def _load_selected_episode(self) -> None:
        if self._loader is None:
            return
        selection = self._episodes.selectionModel()
        if selection is None or not selection.hasSelection():
            return
        row = selection.selectedRows()[0].row()
        episode_id_item = self._episodes.item(row, 1)
        if episode_id_item is None:
            return
        episode_id = episode_id_item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(episode_id, str):
            episode_id = episode_id_item.text()
        replay = self._loader.load_episode(episode_id)
        if replay is None:
            self._episode_summary.setText("No telemetry data available for the selected episode.")
            self._slider.setEnabled(False)
            self._slider.setMaximum(0)
            self._step_view.setPlainText("")
            self._preview.clear()
            self._current_replay = None
            self._current_replay_game = None
            return
        self._current_replay = replay
        self._current_replay_game = self._resolve_game_from_metadata(replay.rollup.metadata)
        if self._current_replay_game is not None:
            self._preview.set_current_game(self._current_replay_game)
        self._episode_summary.setText(
            f"Episode {replay.episode_id}\nTotal reward: {replay.total_reward:.2f}\nSteps: {len(replay.steps)}"
        )
        self._slider.setEnabled(True)
        self._slider.blockSignals(True)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, len(replay.steps) - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._display_step(0)

    def _delete_selected_episode(self) -> None:
        if self._telemetry is None:
            return
        selection = self._episodes.selectionModel()
        if selection is None or not selection.hasSelection():
            return
        row = selection.selectedRows()[0].row()
        episode_item = self._episodes.item(row, 1)
        if episode_item is None:
            return
        episode_id = episode_item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(episode_id, str):
            episode_id = episode_item.text()
        self._telemetry.delete_episode(episode_id)
        self.refresh()
        self._clear_replay_details()

    def _clear_all_episodes(self) -> None:
        if self._telemetry is None:
            return
        self._telemetry.clear_all_episodes()
        self.refresh()
        self._clear_replay_details()

    def _on_slider_changed(self, value: int) -> None:
        self._display_step(value)

    def _display_step(self, index: int) -> None:
        if self._current_replay is None or not self._current_replay.steps:
            self._step_label.setText("Step 0 / 0")
            self._step_view.setPlainText("")
            self._preview.clear()
            return
        index = max(0, min(index, len(self._current_replay.steps) - 1))
        step = self._current_replay.steps[index]
        self._step_label.setText(f"Step {index + 1} / {len(self._current_replay.steps)}")
        self._step_view.setPlainText(self._format_step(step))
        self._preview.display(step.render_payload, self._current_replay_game or self._current_game)

    def _clear_replay_details(self) -> None:
        self._episode_summary.setText("Select an episode to load its replay.")
        self._slider.setEnabled(False)
        self._slider.setMaximum(0)
        self._step_label.setText("Step 0 / 0")
        self._step_view.setPlainText("")
        self._preview.clear()
        self._current_replay = None
        self._current_replay_game = None

    def _format_step(self, step: StepRecord) -> str:
        observation_preview = self._summarise_value(step.observation)
        info_preview = self._summarise_value(step.info)
        return (
            f"Episode: {step.episode_id}\n"
            f"Step Index: {step.step_index}\n"
            f"Reward: {step.reward:.4f}\n"
            f"Terminated: {step.terminated}\n"
            f"Truncated: {step.truncated}\n"
            f"Timestamp: {step.timestamp.isoformat()}\n\n"
            f"Observation:\n{observation_preview}\n\n"
            f"Info:\n{info_preview}"
        )

    @staticmethod
    def _resolve_game_from_metadata(metadata: Any) -> GameId | None:
        if not isinstance(metadata, Mapping):
            return None
        raw_game = metadata.get("game_id")
        if isinstance(raw_game, GameId):
            return raw_game
        if isinstance(raw_game, str):
            try:
                return GameId(raw_game)
            except ValueError:
                return None
        return None

    @staticmethod
    def _summarise_value(value: Any, *, max_length: int = 800) -> str:
        try:
            representation = repr(value)
        except Exception:
            representation = str(type(value))
        if len(representation) > max_length:
            return representation[: max_length - 3] + "..."
        return representation


def _strip_ansi_codes(text: str) -> str:
    import re

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)
    action_indicator = re.compile(r"^\s*\([A-Za-z]+\)\s*\n?", re.MULTILINE)
    text = action_indicator.sub("", text)
    return text.strip()


__all__ = ["RenderTabs"]
