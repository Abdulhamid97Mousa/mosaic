from __future__ import annotations

"""Lightweight helpers for displaying modal busy indicators during blocking operations."""

from contextlib import contextmanager
from typing import Generator, Optional

from qtpy import QtCore, QtGui, QtWidgets


class _BusyDialog(QtWidgets.QDialog):
    _AUTO_PROGRESS_INTERVAL_MS = 45
    _AUTO_PROGRESS_LIMIT = 92

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        title: str,
        message: str,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("busy-dialog")
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinMaxButtonsHint, False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumWidth(360)

        palette = parent.palette() if parent else QtWidgets.QApplication.palette()
        base_color = palette.color(QtGui.QPalette.ColorRole.Base)
        text_color = palette.color(QtGui.QPalette.ColorRole.Text)
        border_color = palette.color(QtGui.QPalette.ColorRole.Mid)
        highlight_color = palette.color(QtGui.QPalette.ColorRole.Highlight)
        muted_highlight = QtGui.QColor(highlight_color)
        muted_highlight.setAlpha(200)

        # Translucent backdrop plus elevated surface container for better contrast.
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        container = QtWidgets.QFrame(self)
        container.setObjectName("busy-dialog-container")
        container.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        container.setAutoFillBackground(True)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QtWidgets.QVBoxLayout(container)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 24, 28, 24)

        label = QtWidgets.QLabel(message, container)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setObjectName("busy-dialog-label")
        layout.addWidget(label)

        progress = QtWidgets.QProgressBar(container)
        progress.setRange(0, 100)
        progress.setValue(6)
        progress.setTextVisible(False)
        progress.setFixedHeight(14)
        progress.setObjectName("busy-dialog-progress")
        layout.addWidget(progress)

        root_layout.addWidget(container)

        stylesheet = f"""
            #busy-dialog {{
                background-color: rgba(0, 0, 0, 96);
            }}
            #busy-dialog-container {{
                background-color: {base_color.name()};
                border-radius: 12px;
                border: 1px solid {border_color.name()};
            }}
            #busy-dialog-label {{
                color: {text_color.name()};
                font-weight: 500;
            }}
            #busy-dialog-progress {{
                border: 1px solid {border_color.name()};
                background-color: {base_color.lighter(110).name()};
                border-radius: 7px;
            }}
            #busy-dialog-progress::chunk {{
                background-color: {muted_highlight.name(QtGui.QColor.NameFormat.HexArgb)};
                border-radius: 7px;
            }}
        """
        self.setStyleSheet(stylesheet)

        self.setAccessibleName(title)
        self.setAccessibleDescription(message)
        progress.setAccessibleName("Background task progress")
        progress.setAccessibleDescription("Indicates that the application is busy")

        self._label = label
        self._progress = progress
        self._auto_progress_locked = False

        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.timeout.connect(self._advance_auto_progress)
        self._auto_timer.start(self._AUTO_PROGRESS_INTERVAL_MS)

    def _advance_auto_progress(self) -> None:
        if self._auto_progress_locked:
            return

        current = self._progress.value()
        if current < self._AUTO_PROGRESS_LIMIT:
            self._progress.setValue(min(self._AUTO_PROGRESS_LIMIT, current + 1))

    def set_message(self, message: str) -> None:
        self._label.setText(message)

    def set_progress(self, value: Optional[int]) -> None:
        if value is None:
            return

        self._auto_progress_locked = True
        self._progress.setValue(max(0, min(100, int(value))))

    def finish(self, *, delay_ms: int = 150) -> None:
        self._auto_progress_locked = True
        self._auto_timer.stop()
        self._progress.setValue(100)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        QtCore.QTimer.singleShot(delay_ms, self.accept)


class _BusyDialogHandle:
    def __init__(self, dialog: _BusyDialog) -> None:
        self._dialog = dialog

    def update(
        self,
        *,
        message: Optional[str] = None,
        progress: Optional[int] = None,
    ) -> None:
        if message is not None:
            self._dialog.set_message(message)
        if progress is not None:
            self._dialog.set_progress(progress)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)

    def close(self) -> None:
        self._dialog.finish(delay_ms=0)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)


@contextmanager
def modal_busy_indicator(
    parent: QtWidgets.QWidget | None,
    *,
    title: str = "Please wait",
    message: str,
) -> Generator[_BusyDialogHandle, None, None]:
    """Display a modal progress dialog while performing a blocking task.

    Returns a handle that can be used to update the message or drive determinate progress
    when the caller has insight into the work being done. If no progress updates are
    provided, the dialog animates itself toward completion and dismisses automatically once
    the context exits.
    """

    dialog = _BusyDialog(parent, title=title, message=message)
    dialog.show()
    QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
    handle: _BusyDialogHandle = _BusyDialogHandle(dialog)
    try:
        yield handle
    finally:
        dialog.finish()
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)


__all__ = ["modal_busy_indicator"]
