"""FastLane tab for board game training visualization.

This tab displays live training frames with interactive board game rendering
when metadata is available (e.g., chess FEN, legal moves).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtWidgets, QtGui

from gym_gui.config.paths import VAR_TRAINER_DIR
from gym_gui.core.enums import GameId
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_UI_FASTLANE_EVAL_SUMMARY_UPDATE,
    LOG_UI_FASTLANE_EVAL_SUMMARY_WARNING,
)
from gym_gui.rendering.strategies.board_game import BoardGameRendererStrategy
from gym_gui.rendering.interfaces import RendererContext
from gym_gui.ui.fastlane_consumer import FastLaneConsumer, FastLaneFrameEvent


_LOGGER = logging.getLogger(__name__)


class BoardGameFastLaneTab(QtWidgets.QWidget):
    """FastLane tab with board game rendering support.

    When metadata contains board game state (chess FEN, etc.), renders
    an interactive board. Otherwise falls back to RGB frame display.
    """

    # Signals for game interactions (forwarded from BoardGameRendererStrategy)
    chess_move_made = QtCore.pyqtSignal(str, str)  # from_sq, to_sq

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        game_id: Optional[GameId] = None,
        mode_label: str | None = None,
        run_mode: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._run_id = run_id
        self._agent_id = agent_id
        self._game_id = game_id
        self._mode_label = mode_label or "Fast lane"
        self._run_mode = (run_mode or "train").lower()
        self._summary_text = ""
        self._summary_path: Path | None = None
        self._summary_timer: QtCore.QTimer | None = None
        self._has_board_game = False
        self._last_metadata_hash: int = 0

        # FastLane consumer for receiving frames
        # Use agent_id as channel name (includes worker suffix, e.g., {run_id}_W0)
        self._consumer = FastLaneConsumer(agent_id, parent=self)
        self._consumer.frame_ready.connect(self._on_frame_ready)
        self._consumer.status_changed.connect(self._on_status_changed)

        # Build UI
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Status bar
        self._status_label = QtWidgets.QLabel(f"{self._mode_label}: connecting...", self)
        layout.addWidget(self._status_label)

        # Stacked widget to switch between RGB and board game views
        self._stack = QtWidgets.QStackedWidget(self)
        layout.addWidget(self._stack, 1)

        # RGB frame view (fallback)
        self._rgb_label = QtWidgets.QLabel(self._stack)
        self._rgb_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._rgb_label.setScaledContents(False)
        self._rgb_label.setStyleSheet("background-color: #1a1a1a;")
        self._stack.addWidget(self._rgb_label)  # Index 0

        # Board game view
        self._board_strategy = BoardGameRendererStrategy(parent=self._stack)
        self._stack.addWidget(self._board_strategy.widget)  # Index 1

        # Connect board game signals
        self._board_strategy.chess_move_made.connect(self.chess_move_made)

        # HUD overlay
        self._hud_label = QtWidgets.QLabel(self)
        self._hud_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: #ffcc00;
                padding: 4px 8px;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        self._hud_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self._hud_label.hide()

        # Eval summary polling
        if self._run_mode == "policy_eval":
            self._bootstrap_eval_summary()

    def _on_status_changed(self, status: str) -> None:
        self._status_label.setText(f"{self._mode_label}: {status}")

    def _on_frame_ready(self, event: FastLaneFrameEvent) -> None:
        """Handle incoming frame from FastLane."""
        # Update HUD
        hud_text = event.hud_text
        if self._summary_text:
            hud_text = f"{hud_text}\n{self._summary_text}"
        self._hud_label.setText(hud_text)
        if not self._hud_label.isVisible():
            self._hud_label.show()
            self._hud_label.raise_()

        # Check for board game metadata
        if event.metadata:
            self._handle_board_game_metadata(event.metadata)
        else:
            # No metadata - show RGB frame
            self._show_rgb_frame(event.image)

    def _handle_board_game_metadata(self, metadata_bytes: bytes) -> None:
        """Parse and render board game state from metadata."""
        # Check if metadata changed (avoid unnecessary re-renders)
        metadata_hash = hash(metadata_bytes)
        if metadata_hash == self._last_metadata_hash:
            return
        self._last_metadata_hash = metadata_hash

        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            _LOGGER.debug(f"Failed to parse board game metadata: {e}")
            return

        game_type = metadata.get("game_type")
        if game_type == "chess":
            self._render_chess(metadata)
        else:
            _LOGGER.debug(f"Unknown game type in metadata: {game_type}")

    def _render_chess(self, metadata: dict) -> None:
        """Render chess board from metadata."""
        # Build payload for BoardGameRendererStrategy
        payload = {
            "game_id": GameId.CHESS,
            "fen": metadata.get("fen"),
            "legal_moves": metadata.get("legal_moves", []),
            "current_player": metadata.get("current_player"),
            "is_check": metadata.get("is_check", False),
            "is_checkmate": metadata.get("is_checkmate", False),
            "is_stalemate": metadata.get("is_stalemate", False),
            "is_game_over": metadata.get("is_game_over", False),
        }

        # Create renderer context
        context = RendererContext(
            game_id=GameId.CHESS,
            run_id=self._run_id,
            episode_index=0,
            step_index=0,
        )

        # Render using board game strategy
        self._board_strategy.render(payload, context=context)

        # Switch to board game view
        if self._stack.currentIndex() != 1:
            self._stack.setCurrentIndex(1)
            self._has_board_game = True

    def _show_rgb_frame(self, image: QtGui.QImage) -> None:
        """Display RGB frame in the fallback label."""
        if image.isNull():
            return

        # Scale to fit while maintaining aspect ratio
        pixmap = QtGui.QPixmap.fromImage(image)
        label_size = self._rgb_label.size()
        scaled = pixmap.scaled(
            label_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._rgb_label.setPixmap(scaled)

        # Switch to RGB view
        if self._stack.currentIndex() != 0:
            self._stack.setCurrentIndex(0)
            self._has_board_game = False

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Reposition HUD on resize."""
        super().resizeEvent(event)
        # Position HUD in top-left corner
        self._hud_label.adjustSize()
        self._hud_label.move(8, self._status_label.height() + 12)

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._summary_timer is not None:
            self._summary_timer.stop()
            self._summary_timer.deleteLater()
            self._summary_timer = None
        self._consumer.stop()
        self._board_strategy.cleanup()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.cleanup()
        super().closeEvent(event)

    def _bootstrap_eval_summary(self) -> None:
        summary_path = (VAR_TRAINER_DIR / "runs" / self._run_id / "eval_summary.json").resolve()
        self._summary_path = summary_path
        self._summary_timer = QtCore.QTimer(self)
        self._summary_timer.setInterval(1000)
        self._summary_timer.timeout.connect(self._refresh_eval_summary)
        self._summary_timer.start()
        self._refresh_eval_summary()

    def _refresh_eval_summary(self) -> None:
        path = self._summary_path
        if path is None:
            return
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except Exception as exc:
            log_constant(
                _LOGGER,
                LOG_UI_FASTLANE_EVAL_SUMMARY_WARNING,
                extra={"run_id": self._run_id, "path": str(path)},
                exc_info=exc,
            )
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            log_constant(
                _LOGGER,
                LOG_UI_FASTLANE_EVAL_SUMMARY_WARNING,
                extra={"run_id": self._run_id, "path": str(path)},
                exc_info=exc,
            )
            return

        batch = payload.get("batch_index", 0)
        episodes = payload.get("episodes", 0)
        avg_value = float(payload.get("avg_return", 0.0) or 0.0)
        min_value = float(payload.get("min_return", 0.0) or 0.0)
        max_value = float(payload.get("max_return", 0.0) or 0.0)
        summary_text = (
            f"eval batch {batch} | episodes={episodes} avg={avg_value:.2f} "
            f"min={min_value:.2f} max={max_value:.2f}"
        )
        if summary_text == self._summary_text:
            return
        self._summary_text = summary_text
        log_constant(
            _LOGGER,
            LOG_UI_FASTLANE_EVAL_SUMMARY_UPDATE,
            extra={"run_id": self._run_id, "text": summary_text},
        )
