from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6 import QtCore, QtGui

from gym_gui.fastlane import FastLaneReader, FastLaneFrame


@dataclass(slots=True)
class FastLaneFrameEvent:
    image: QtGui.QImage
    hud_text: str


class FastLaneConsumer(QtCore.QObject):
    """Background reader that polls FastLane shared memory and emits frames."""

    frame_ready = QtCore.pyqtSignal(object)
    status_changed = QtCore.pyqtSignal(str)

    def __init__(self, run_id: str, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._run_id = run_id
        self._reader: FastLaneReader | None = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._poll)
        self._attach_reader()
        self._timer.start()

    def _attach_reader(self) -> None:
        try:
            self._reader = FastLaneReader.attach(self._run_id)
            self.status_changed.emit("connected")
        except FileNotFoundError:
            self._reader = None
            self.status_changed.emit("fastlane-unavailable")

    def _poll(self) -> None:
        reader = self._reader
        if reader is None:
            self._attach_reader()
            return
        frame = reader.latest_frame()
        if frame is None:
            return
        event = self._to_event(frame)
        self.frame_ready.emit(event)

    def _to_event(self, frame: FastLaneFrame) -> FastLaneFrameEvent:
        if frame.channels == 3:
            fmt = QtGui.QImage.Format.Format_RGB888
            stride = frame.width * 3
        else:
            fmt = QtGui.QImage.Format.Format_RGBA8888
            stride = frame.width * 4
        image = QtGui.QImage(frame.width, frame.height, fmt)
        buffer = image.bits()
        buffer.setsize(image.byteCount())
        buffer[: stride * frame.height] = frame.data[: stride * frame.height]
        metrics = frame.metrics
        hud = f"reward: {metrics.last_reward:.2f}\nreturn: {metrics.rolling_return:.2f}\nstep/sec: {metrics.step_rate_hz:.1f}"
        return FastLaneFrameEvent(image=image, hud_text=hud)

    def stop(self) -> None:
        self._timer.stop()
        if self._reader is not None:
            self._reader.close()
            self._reader = None
