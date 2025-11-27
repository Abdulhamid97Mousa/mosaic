"""Renderer strategy for RGB frame payloads."""

from __future__ import annotations

import logging
from typing import Callable, Mapping

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.core.enums import RenderMode
from gym_gui.rendering.interfaces import RendererContext, RendererStrategy

_LOGGER = logging.getLogger(__name__)


class RgbRendererStrategy(RendererStrategy):
    """Render RGB array payloads into a scrollable QLabel."""

    mode = RenderMode.RGB_ARRAY

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        self._view = _RgbView(parent)

    def set_mouse_action_callback(self, callback: Callable[[int], None] | None) -> None:
        """Set callback for mouse-triggered discrete actions (legacy mode)."""
        self._view.set_mouse_action_callback(callback)

    def set_mouse_delta_callback(self, callback: Callable[[float, float], None] | None) -> None:
        """Set callback for continuous mouse delta (true FPS mode).

        Args:
            callback: Function called with (delta_x, delta_y) in degrees.
                     Positive delta_x = turn right, positive delta_y = look down.
        """
        self._view.set_mouse_delta_callback(callback)

    def set_mouse_delta_scale(self, scale: float) -> None:
        """Set degrees per pixel for delta mode (default 0.5)."""
        self._view.set_mouse_delta_scale(scale)

    def set_mouse_capture_enabled(self, enabled: bool) -> None:
        """Enable/disable mouse capture support."""
        self._view.set_mouse_capture_enabled(enabled)

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._view

    def render(self, payload: Mapping[str, object], *, context: RendererContext | None = None) -> None:
        # Support both "rgb" (new) and "frame" (legacy) keys for backward compatibility
        frame = payload.get("rgb")
        if frame is None:
            frame = payload.get("frame")
        if frame is None:
            self.reset()
            return
        self._view.render_frame(frame, tooltip_payload=payload)

    def supports(self, payload: Mapping[str, object]) -> bool:
        return "rgb" in payload or "frame" in payload

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
    """Widget that displays RGB frames with optional FPS-style mouse capture."""

    # Signal emitted when mouse capture state changes (captured: bool)
    mouse_capture_changed = QtCore.Signal(bool)

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

        # Mouse capture state
        self._mouse_capture_enabled = False
        self._mouse_captured = False
        self._mouse_action_callback: Callable[[int], None] | None = None
        self._mouse_delta_callback: Callable[[float, float], None] | None = None
        self._last_mouse_pos: QtCore.QPoint | None = None
        self._mouse_sensitivity = 5.0  # Pixels per action trigger (for discrete mode)
        self._mouse_delta_scale = 0.5  # Degrees per pixel (for delta mode)
        self._accumulated_delta_x = 0.0
        self._use_delta_mode = False  # If True, use continuous delta; else discrete actions

        # Turn action indices (configurable per game, used in discrete mode)
        self._turn_left_action = 1   # Default: many scenarios use index 1
        self._turn_right_action = 2  # Default: many scenarios use index 2

        # Enable focus and mouse tracking
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def set_mouse_action_callback(self, callback: Callable[[int], None] | None) -> None:
        """Set callback for mouse-triggered discrete actions (legacy mode)."""
        self._mouse_action_callback = callback
        self._use_delta_mode = False

    def set_mouse_delta_callback(self, callback: Callable[[float, float], None] | None) -> None:
        """Set callback for continuous mouse delta (FPS mode).

        Args:
            callback: Function called with (delta_x, delta_y) in degrees.
                     Positive delta_x = turn right, positive delta_y = look down.
        """
        self._mouse_delta_callback = callback
        self._use_delta_mode = callback is not None

    def set_mouse_delta_scale(self, scale: float) -> None:
        """Set degrees per pixel for delta mode (default 0.5)."""
        self._mouse_delta_scale = max(0.01, scale)

    def set_mouse_capture_enabled(self, enabled: bool) -> None:
        """Enable/disable mouse capture support."""
        self._mouse_capture_enabled = enabled
        if not enabled and self._mouse_captured:
            self._release_mouse_capture()

    def set_turn_action_indices(self, turn_left: int, turn_right: int) -> None:
        """Set the action indices for turn left/right (varies by game)."""
        self._turn_left_action = turn_left
        self._turn_right_action = turn_right

    def set_mouse_sensitivity(self, sensitivity: float) -> None:
        """Set mouse sensitivity (lower = more sensitive)."""
        self._mouse_sensitivity = max(1.0, sensitivity)

    def is_mouse_captured(self) -> bool:
        """Return True if mouse is currently captured."""
        return self._mouse_captured

    def _capture_mouse(self) -> None:
        """Capture the mouse for FPS-style control."""
        if self._mouse_captured:
            return
        self._mouse_captured = True
        self._last_mouse_pos = QtGui.QCursor.pos()
        self._accumulated_delta_x = 0.0
        self.grabMouse()
        self.setCursor(QtCore.Qt.CursorShape.BlankCursor)
        self.mouse_capture_changed.emit(True)
        _LOGGER.debug("Mouse captured for FPS control")

    def _release_mouse_capture(self) -> None:
        """Release the captured mouse."""
        if not self._mouse_captured:
            return
        self._mouse_captured = False
        self._last_mouse_pos = None
        self.releaseMouse()
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.mouse_capture_changed.emit(False)
        _LOGGER.debug("Mouse released from FPS control")

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press - capture mouse on click."""
        if not self._mouse_capture_enabled:
            super().mousePressEvent(event)
            return

        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if not self._mouse_captured:
                self._capture_mouse()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move - convert to turn/look actions when captured."""
        if not self._mouse_captured or self._last_mouse_pos is None:
            super().mouseMoveEvent(event)
            return

        current_pos = QtGui.QCursor.pos()
        delta_x = current_pos.x() - self._last_mouse_pos.x()
        delta_y = current_pos.y() - self._last_mouse_pos.y()

        if self._use_delta_mode and self._mouse_delta_callback is not None:
            # Delta mode: send continuous rotation values (in degrees)
            # Positive delta_x = turn right, positive delta_y = look down
            degrees_x = delta_x * self._mouse_delta_scale
            degrees_y = delta_y * self._mouse_delta_scale
            if degrees_x != 0.0 or degrees_y != 0.0:
                self._mouse_delta_callback(degrees_x, degrees_y)
        elif self._mouse_action_callback is not None:
            # Discrete mode: accumulate and trigger discrete turn actions
            self._accumulated_delta_x += delta_x
            while abs(self._accumulated_delta_x) >= self._mouse_sensitivity:
                if self._accumulated_delta_x > 0:
                    # Moving right -> turn right
                    self._mouse_action_callback(self._turn_right_action)
                    self._accumulated_delta_x -= self._mouse_sensitivity
                else:
                    # Moving left -> turn left
                    self._mouse_action_callback(self._turn_left_action)
                    self._accumulated_delta_x += self._mouse_sensitivity

        # Re-center the cursor to allow continuous movement
        center = self.mapToGlobal(self.rect().center())
        QtGui.QCursor.setPos(center)
        self._last_mouse_pos = center

        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key press - ESC releases mouse capture."""
        if self._mouse_captured and event.key() == QtCore.Qt.Key.Key_Escape:
            self._release_mouse_capture()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        """Release mouse capture when focus is lost."""
        if self._mouse_captured:
            self._release_mouse_capture()
        super().focusOutEvent(event)

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
