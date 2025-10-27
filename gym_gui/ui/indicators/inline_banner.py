from __future__ import annotations

"""Qt inline banner for displaying run status summaries."""

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from .state import IndicatorSeverity, IndicatorState


class InlineBanner(QtWidgets.QFrame):
    """A slim, dismissible strip to highlight run state within a panel."""

    dismissed = QtCore.pyqtSignal(str)

    _BASE_STYLE = """
    QFrame { border-radius: 6px; padding: 6px 10px; }
    QLabel#message { font-weight: 600; }
    QLabel#details { color: palette(mid); }
    """

    _SEVERITY_COLORS = {
        IndicatorSeverity.INFO: ("#0b6efd", "#d4e9ff"),
        IndicatorSeverity.WARNING: ("#fd7e14", "#fdebd3"),
        IndicatorSeverity.ERROR: ("#dc3545", "#f8d7da"),
        IndicatorSeverity.CRITICAL: ("#6f42c1", "#ead7f6"),
    }

    def __init__(self, *, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        self._current_state: Optional[IndicatorState] = None
        self.setObjectName("inlineBanner")
        root_layout = QtWidgets.QHBoxLayout()
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(12)
        self.setLayout(root_layout)

        self._icon = QtWidgets.QLabel(self)
        self._icon.setPixmap(self._build_icon_pixmap(IndicatorSeverity.INFO))
        root_layout.addWidget(self._icon)

        text_wrapper = QtWidgets.QVBoxLayout()
        text_wrapper.setContentsMargins(0, 0, 0, 0)
        text_wrapper.setSpacing(2)
        root_layout.addLayout(text_wrapper)

        self._message = QtWidgets.QLabel(self)
        self._message.setObjectName("message")
        text_wrapper.addWidget(self._message)

        self._details = QtWidgets.QLabel(self)
        self._details.setObjectName("details")
        self._details.setWordWrap(True)
        text_wrapper.addWidget(self._details)

        self._badges_container = QtWidgets.QHBoxLayout()
        self._badges_container.setContentsMargins(0, 0, 0, 0)
        self._badges_container.setSpacing(6)
        text_wrapper.addLayout(self._badges_container)

        self._close_button = QtWidgets.QToolButton(self)
        self._close_button.setText("×")
        self._close_button.clicked.connect(self._handle_dismiss)
        root_layout.addWidget(self._close_button)

        self._apply_style()
        self.hide()

    def _apply_style(self) -> None:
        self.setStyleSheet(self._BASE_STYLE)

    def _build_icon_pixmap(self, severity: IndicatorSeverity) -> QtGui.QPixmap:
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        color, _ = self._SEVERITY_COLORS[severity]
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(QtGui.QColor(color))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, 16, 16)
        painter.end()
        return pixmap

    def render_state(self, state: IndicatorState) -> None:
        if self._current_state and state.run_id == self._current_state.run_id and state == self._current_state:
            return
        self._current_state = state
        color, background = self._SEVERITY_COLORS[state.severity]
        palette = self.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(background))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self._icon.setPixmap(self._build_icon_pixmap(state.severity))
        self._message.setText(state.message)
        self._details.setText(state.details or "")
        self._details.setVisible(bool(state.details))
        self._render_badges(state.badges, accent=color)
        self.show()

    def _render_badges(self, badges: tuple[str, ...], *, accent: str) -> None:
        while self._badges_container.count():
            item = self._badges_container.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for badge in badges:
            label = QtWidgets.QLabel(badge, parent=self)
            label.setStyleSheet(f"border: 1px solid {accent}; border-radius: 8px; padding: 2px 6px; color: {accent};")
            self._badges_container.addWidget(label)

    def clear(self) -> None:
        if self._current_state:
            previous_run = self._current_state.run_id
            self._current_state = None
            self.hide()
            self.dismissed.emit(previous_run)

    def _handle_dismiss(self) -> None:
        if self._current_state:
            self.dismissed.emit(self._current_state.run_id)
        self.hide()
