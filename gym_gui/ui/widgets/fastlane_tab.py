from __future__ import annotations

import json
import logging
from pathlib import Path

from PyQt6 import QtCore, QtWidgets, QtQuickWidgets, QtGui

from gym_gui.config.paths import VAR_TRAINER_DIR
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_UI_FASTLANE_EVAL_SUMMARY_UPDATE,
    LOG_UI_FASTLANE_EVAL_SUMMARY_WARNING,
)
from gym_gui.ui.fastlane_consumer import FastLaneConsumer, FastLaneFrameEvent
from gym_gui.ui.renderers.fastlane_item import FastLaneItem  # ensures type registered


_LOGGER = logging.getLogger(__name__)


class FastLaneTab(QtWidgets.QWidget):
    """Qt Quick-based view that renders frames from the fast lane."""

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        mode_label: str | None = None,
        run_mode: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._run_id = run_id
        self._agent_id = agent_id
        self._mode_label = mode_label or "Fast lane"
        self._run_mode = (run_mode or "train").lower()
        self._summary_text = ""
        self._summary_path: Path | None = None
        self._summary_timer: QtCore.QTimer | None = None
        self._consumer = FastLaneConsumer(run_id, parent=self)
        self._consumer.frame_ready.connect(self._on_frame_ready)
        self._consumer.status_changed.connect(self._on_status_changed)
        self._status_label = QtWidgets.QLabel(f"{self._mode_label}: connecting…", self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._status_label)

        self._quick = QtQuickWidgets.QQuickWidget(self)
        self._quick.setResizeMode(QtQuickWidgets.QQuickWidget.ResizeMode.SizeRootObjectToView)
        qml_path = Path(__file__).resolve().parent.parent / "qml" / "FastLaneView.qml"
        self._quick.engine().addImportPath(str(qml_path.parent))
        self._quick.setSource(QtCore.QUrl.fromLocalFile(str(qml_path)))
        layout.addWidget(self._quick, 1)

        self._root_obj = self._quick.rootObject()
        if self._run_mode == "policy_eval":
            self._bootstrap_eval_summary()

    def _on_status_changed(self, status: str) -> None:
        self._status_label.setText(f"{self._mode_label}: {status}")

    def _on_frame_ready(self, event: FastLaneFrameEvent) -> None:
        if self._root_obj is None:
            self._root_obj = self._quick.rootObject()
        if self._root_obj is None:
            return
        hud_text = event.hud_text
        if self._summary_text:
            hud_text = f"{hud_text}\n{self._summary_text}"
        self._root_obj.setProperty("hudText", hud_text)
        canvas = self._root_obj.findChild(QtCore.QObject, "fastlaneCanvas")
        if canvas is None:
            return
        QtCore.QMetaObject.invokeMethod(
            canvas,
            "setFrame",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(QtGui.QImage, event.image),
        )

    def cleanup(self) -> None:
        if self._summary_timer is not None:
            self._summary_timer.stop()
            self._summary_timer.deleteLater()
            self._summary_timer = None
        self._consumer.stop()
        self._quick.setSource(QtCore.QUrl())

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
        except Exception as exc:  # pragma: no cover - IO race
            log_constant(
                _LOGGER,
                LOG_UI_FASTLANE_EVAL_SUMMARY_WARNING,
                extra={"run_id": self._run_id, "path": str(path)},
                exc_info=exc,
            )
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - partial write
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
