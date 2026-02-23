from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtQuick, QtQml
from PyQt6.QtCore import pyqtSlot, pyqtProperty  # type: ignore[attr-defined]

try:
    from PyQt6.QtQml import QmlElement  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - PyQt6 versions < 6.5
    def QmlElement(*args, **kwargs):  # type: ignore
        def wrapper(cls):
            return cls
        return wrapper

@QmlElement("GymGui", 1, 0)
class FastLaneItem(QtQuick.QQuickPaintedItem):
    """QQuick item that paints frames from the fast lane."""

    def __init__(self, parent: QtQuick.QQuickItem | None = None) -> None:
        super().__init__(parent)
        self._image: QtGui.QImage | None = None
        self._hud_text = ""
        self.setRenderTarget(QtQuick.QQuickPaintedItem.RenderTarget.FramebufferObject)

    @pyqtSlot(QtGui.QImage)
    def setFrame(self, image: QtGui.QImage) -> None:
        self._image = image
        self.update()

    @pyqtProperty(str)
    def hudText(self) -> str:
        return self._hud_text

    @hudText.setter
    def hudText(self, value: str) -> None:
        if self._hud_text != value:
            self._hud_text = value
            self.update()

    def paint(self, painter: QtGui.QPainter) -> None:
        target_rect = self.contentsBoundingRect()
        if self._image is not None and not self._image.isNull():
            painter.drawImage(target_rect, self._image)
        else:
            painter.fillRect(target_rect, QtCore.Qt.GlobalColor.black)


# Ensure the type is registered when this module is imported
QtQml.qmlRegisterType(FastLaneItem, "GymGui", 1, 0, "FastLaneItem")
