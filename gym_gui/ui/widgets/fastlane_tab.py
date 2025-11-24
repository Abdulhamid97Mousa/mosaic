from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtWidgets, QtQuickWidgets, QtQml, QtGui

from gym_gui.ui.fastlane_consumer import FastLaneConsumer, FastLaneFrameEvent
from gym_gui.ui.renderers.fastlane_item import FastLaneItem  # ensures type registered


class FastLaneTab(QtWidgets.QWidget):
    """Qt Quick-based view that renders frames from the fast lane."""

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        mode_label: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._run_id = run_id
        self._agent_id = agent_id
        self._mode_label = mode_label or "Fast lane"
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

    def _on_status_changed(self, status: str) -> None:
        self._status_label.setText(f"{self._mode_label}: {status}")

    def _on_frame_ready(self, event: FastLaneFrameEvent) -> None:
        if self._root_obj is None:
            self._root_obj = self._quick.rootObject()
        if self._root_obj is None:
            return
        self._root_obj.setProperty("hudText", event.hud_text)
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
        self._consumer.stop()
        self._quick.setSource(QtCore.QUrl())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.cleanup()
        super().closeEvent(event)
