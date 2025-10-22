"""Renderer strategy for RGB frame payloads."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.enums import RenderMode
from gym_gui.rendering.interfaces import RendererContext, RendererStrategy


class RgbRendererStrategy(RendererStrategy):
    """Render RGB array payloads into a scrollable QLabel."""

    mode = RenderMode.RGB_ARRAY

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self._view = _RgbView(parent)

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._view

    def render(self, payload: Mapping[str, object], *, context: RendererContext | None = None) -> None:
        frame = payload.get("rgb")
        if frame is None:
            self.reset()
            return
        self._view.render_frame(frame, tooltip_payload=payload)

    def supports(self, payload: Mapping[str, object]) -> bool:
        return "rgb" in payload

    def reset(self) -> None:
        self._view.reset()

    def cleanup(self) -> None:
        """Clean up resources before widget destruction."""
        try:
            self._view.reset()
        except Exception:
            # Silently ignore errors during cleanup
            pass


class _RgbView(QtWidgets.QWidget):
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

    def render_frame(self, frame: object, *, tooltip_payload: Mapping[str, object] | None = None) -> None:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            self._label.setText("Unsupported RGB frame format")
            self._current_pixmap = None
            return

        array = np.ascontiguousarray(array)
        height, width, channels = array.shape
        if channels == 3:
            fmt = QtGui.QImage.Format.Format_RGB888
        else:
            fmt = QtGui.QImage.Format.Format_RGBA8888

        image = QtGui.QImage(array.data, width, height, width * channels, fmt).copy()
        self._current_pixmap = QtGui.QPixmap.fromImage(image)
        self._scale_pixmap()

        tooltip = ""
        if tooltip_payload is not None:
            ansi = tooltip_payload.get("ansi")
            if isinstance(ansi, str) and ansi:
                tooltip = _strip_ansi_codes(ansi)
        self._label.setToolTip(tooltip)

    def reset(self) -> None:
        self._label.clear()
        self._label.setText("No RGB frame available")
        self._current_pixmap = None

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


def _strip_ansi_codes(value: str) -> str:
    import re

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", value)


__all__ = ["RgbRendererStrategy"]
