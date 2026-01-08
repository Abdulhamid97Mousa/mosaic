"""Renderer strategy for RGB frame payloads."""

from __future__ import annotations

import logging
from typing import Callable, Mapping

import numpy as np
from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]
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
    """Widget that displays RGB frames with optional FPS-style mouse capture.

    Uses paintEvent for rendering instead of QLabel to ensure proper expansion
    and aspect ratio preservation (Qt6 best practice for custom image widgets).
    """

    # Signal emitted when mouse capture state changes (captured: bool)
    mouse_capture_changed = pyqtSignal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Qt6 best practice: Use Expanding policy for widgets that should fill space
        # Combined with paintEvent rendering, this ensures proper expansion
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        # Large minimum size to start big and ensure good visibility
        self.setMinimumSize(300, 250)

        # Dark background for the render area
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(17, 17, 17))  # #111
        self.setPalette(palette)

        self._current_pixmap: QtGui.QPixmap | None = None
        self._tooltip_text: str = ""

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

    def sizeHint(self) -> QtCore.QSize:
        """Return preferred size - large value to encourage expansion.

        Qt uses sizeHint along with sizePolicy to determine widget size.
        With Expanding policy and large sizeHint, layout will give max space.
        """
        return QtCore.QSize(600, 500)

    def minimumSizeHint(self) -> QtCore.QSize:
        """Return minimum acceptable size."""
        return QtCore.QSize(200, 200)

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

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
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

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
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

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        """Handle key press - ESC releases mouse capture."""
        if self._mouse_captured and event.key() == QtCore.Qt.Key.Key_Escape:
            self._release_mouse_capture()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:  # type: ignore[override]
        """Release mouse capture when focus is lost."""
        if self._mouse_captured:
            self._release_mouse_capture()
        super().focusOutEvent(event)

    def render_frame(self, frame: object, *, tooltip_payload: Mapping[str, object] | None = None) -> None:
        """Render an RGB frame array."""
        # IMPORTANT: Must specify dtype=np.uint8 because when frame comes from
        # JSON/list conversion (e.g., tolist() -> np.asarray()), numpy defaults
        # to int64 which corrupts the image data for QImage
        array = np.asarray(frame, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            self._current_pixmap = None
            self._tooltip_text = "Unsupported RGB frame format"
            self.update()
            return

        # Ensure contiguous memory layout for QImage
        array = np.ascontiguousarray(array, dtype=np.uint8)
        height, width, channels = array.shape
        if channels == 3:
            fmt = QtGui.QImage.Format.Format_RGB888
        else:
            fmt = QtGui.QImage.Format.Format_RGBA8888

        # Cast array.data (memoryview) to bytes for QImage compatibility
        image = QtGui.QImage(bytes(array.data), width, height, width * channels, fmt).copy()
        self._current_pixmap = QtGui.QPixmap.fromImage(image)

        # Update tooltip
        tooltip = ""
        if tooltip_payload is not None:
            ansi = tooltip_payload.get("ansi")
            if isinstance(ansi, str) and ansi:
                tooltip = _strip_ansi_codes(ansi)
        self._tooltip_text = tooltip
        self.setToolTip(tooltip)

        # Trigger repaint - paintEvent will handle scaling and drawing
        self.update()

    def reset(self) -> None:
        """Reset to show no frame message."""
        self._current_pixmap = None
        self._tooltip_text = ""
        self.setToolTip("")
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        """Paint the pixmap scaled to fit the widget, preserving aspect ratio.

        The image is scaled to fit within the container while maintaining
        its aspect ratio, then centered.
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)

        # Get widget dimensions
        widget_rect = self.rect()

        if self._current_pixmap is None or self._current_pixmap.isNull():
            # Draw placeholder text when no frame available
            painter.setPen(QtGui.QColor(153, 153, 153))  # #999
            painter.drawText(widget_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "No RGB frame available")
            painter.end()
            return

        # Get actual pixmap size
        pixmap_size = self._current_pixmap.size()
        pixmap_width = pixmap_size.width()
        pixmap_height = pixmap_size.height()

        # Calculate scale to fit within widget while preserving aspect ratio
        widget_width = widget_rect.width()
        widget_height = widget_rect.height()

        if pixmap_width > 0 and pixmap_height > 0:
            scale_x = widget_width / pixmap_width
            scale_y = widget_height / pixmap_height
            scale = min(scale_x, scale_y)  # Use smaller scale to fit entirely

            # Calculate scaled dimensions
            scaled_width = int(pixmap_width * scale)
            scaled_height = int(pixmap_height * scale)

            # Center the scaled pixmap in the widget
            x = (widget_width - scaled_width) // 2
            y = (widget_height - scaled_height) // 2

            # Draw the pixmap scaled to fit
            target_rect = QtCore.QRect(x, y, scaled_width, scaled_height)
            painter.drawPixmap(target_rect, self._current_pixmap)
        else:
            # Fallback: draw at original size if dimensions are invalid
            painter.drawPixmap(0, 0, self._current_pixmap)

        painter.end()


def _strip_ansi_codes(value: str) -> str:
    import re

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", value)


__all__ = ["RgbRendererStrategy"]
