from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

from PyQt6 import QtCore, QtGui

from gym_gui.fastlane import FastLaneReader, FastLaneFrame
from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import (
    LOG_FASTLANE_CONNECTED,
    LOG_FASTLANE_UNAVAILABLE,
    LOG_FASTLANE_HEADER_INVALID,
    LOG_FASTLANE_FRAME_READ_ERROR,
)


@dataclass(slots=True)
class FastLaneFrameEvent:
    image: QtGui.QImage
    hud_text: str
    metadata: Optional[bytes] = None  # JSON metadata bytes (for board games, etc.)


_LOGGER = logging.getLogger(__name__)


class FastLaneConsumer(QtCore.QObject):
    """Background reader that polls FastLane shared memory and emits frames."""

    frame_ready = QtCore.pyqtSignal(object)
    status_changed = QtCore.pyqtSignal(str)

    def __init__(self, run_id: str, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._run_id = run_id
        self._reader: FastLaneReader | None = None
        self._header_warning_emitted = False
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._poll)
        self._attach_reader()
        self._timer.start()

    def _attach_reader(self) -> None:
        try:
            self._reader = FastLaneReader.attach(self._run_id)
            self.status_changed.emit("connected")
            self._header_warning_emitted = False
            log_constant(_LOGGER, LOG_FASTLANE_CONNECTED, extra={"run_id": self._run_id})
        except FileNotFoundError:
            self._reader = None
            self.status_changed.emit("fastlane-unavailable")
            log_constant(_LOGGER, LOG_FASTLANE_UNAVAILABLE, extra={"run_id": self._run_id})

    def _poll(self) -> None:
        reader = self._reader
        if reader is None:
            self._attach_reader()
            return
        try:
            if reader.capacity <= 0 or reader.slot_size <= 0:
                if not self._header_warning_emitted:
                    log_constant(
                        _LOGGER,
                        LOG_FASTLANE_HEADER_INVALID,
                        extra={
                            "run_id": self._run_id,
                            "capacity": reader.capacity,
                            "slot_size": reader.slot_size,
                        },
                    )
                    self._header_warning_emitted = True
                return
            frame = reader.latest_frame()
        except FileNotFoundError:
            log_constant(_LOGGER, LOG_FASTLANE_UNAVAILABLE, extra={"run_id": self._run_id})
            self._handle_reader_failure()
            return
        except Exception as exc:
            log_constant(
                _LOGGER,
                LOG_FASTLANE_FRAME_READ_ERROR,
                extra={"run_id": self._run_id},
                exc_info=exc,
            )
            self._handle_reader_failure()
            return
        if frame is None:
            return
        self._header_warning_emitted = False
        event = self._to_event(frame)
        self.frame_ready.emit(event)

    def _to_event(self, frame: FastLaneFrame) -> FastLaneFrameEvent:
        if frame.channels == 3:
            fmt = QtGui.QImage.Format.Format_RGB888
            stride = frame.width * 3
        else:
            fmt = QtGui.QImage.Format.Format_RGBA8888
            stride = frame.width * 4
        # Wrap the bytes payload directly; copy immediately so the QImage owns memory.
        qimage = QtGui.QImage(frame.data, frame.width, frame.height, stride, fmt)
        image = qimage.copy() if not qimage.isNull() else QtGui.QImage(frame.width, frame.height, fmt)
        metrics = frame.metrics
        hud = f"reward: {metrics.last_reward:.2f}\nreturn: {metrics.rolling_return:.2f}\nstep/sec: {metrics.step_rate_hz:.1f}"
        return FastLaneFrameEvent(image=image, hud_text=hud, metadata=frame.metadata)

    def stop(self) -> None:
        self._timer.stop()
        self._close_reader()

    def _handle_reader_failure(self) -> None:
        self._close_reader()
        self._header_warning_emitted = False
        self.status_changed.emit("fastlane-unavailable")

    def _close_reader(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
