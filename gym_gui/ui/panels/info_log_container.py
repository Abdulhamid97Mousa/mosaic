"""Game info and runtime log container widget."""

from __future__ import annotations

from typing import Dict, Optional

from qtpy import QtWidgets


class InfoLogContainer(QtWidgets.QWidget):
    """Container for game info and runtime log panels."""

    LOG_FILTER_OPTIONS: Dict[str, str | None] = {
        "All": None,
        "Controllers": "gym_gui.controllers",
        "Adapters": "gym_gui.core.adapters",
        "Agents": "gym_gui.agents",
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Create splitter for info and log
        splitter = QtWidgets.QSplitter(parent=self)

        # Game info panel
        self._info_group = QtWidgets.QGroupBox("Game Info", self)
        info_layout = QtWidgets.QVBoxLayout(self._info_group)
        self._game_info = QtWidgets.QTextBrowser(self._info_group)
        self._game_info.setReadOnly(True)
        self._game_info.setOpenExternalLinks(True)
        info_layout.addWidget(self._game_info, 1)
        splitter.addWidget(self._info_group)

        # Runtime log panel
        self._log_group = QtWidgets.QGroupBox("Runtime Log", self)
        log_layout = QtWidgets.QVBoxLayout(self._log_group)

        # Log filter row
        filter_row = QtWidgets.QHBoxLayout()
        filter_label = QtWidgets.QLabel("Filter:")
        self._log_filter = QtWidgets.QComboBox()
        self._log_filter.addItems(self.LOG_FILTER_OPTIONS.keys())
        filter_row.addWidget(filter_label)
        filter_row.addWidget(self._log_filter, 1)
        log_layout.addLayout(filter_row)

        # Log console
        self._log_console = QtWidgets.QPlainTextEdit()
        self._log_console.setReadOnly(True)
        self._log_console.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self._log_console, 1)
        splitter.addWidget(self._log_group)

        # Configure splitter stretch
        splitter.setStretchFactor(0, 1)  # Game Info
        splitter.setStretchFactor(1, 2)  # Runtime Log

        # Set up container layout
        container_layout = QtWidgets.QVBoxLayout(self)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(splitter)

    def get_info_group(self) -> QtWidgets.QGroupBox:
        """Get the game info group box."""
        return self._info_group

    def get_log_group(self) -> QtWidgets.QGroupBox:
        """Get the runtime log group box."""
        return self._log_group

    def get_game_info_widget(self) -> QtWidgets.QTextBrowser:
        """Get the game info text browser."""
        return self._game_info

    def get_log_console(self) -> QtWidgets.QPlainTextEdit:
        """Get the log console."""
        return self._log_console

    def get_log_filter(self) -> QtWidgets.QComboBox:
        """Get the log filter combo box."""
        return self._log_filter

    def set_game_info(self, html: str) -> None:
        """Set game info HTML content."""
        self._game_info.setHtml(html)

    def append_log(self, text: str) -> None:
        """Append text to the log console."""
        self._log_console.appendPlainText(text)

    def clear_log(self) -> None:
        """Clear the log console."""
        self._log_console.clear()

    def get_log_filter_value(self) -> Optional[str]:
        """Get the current log filter value."""
        current_text = self._log_filter.currentText()
        return self.LOG_FILTER_OPTIONS.get(current_text)

    def get_log_text(self) -> str:
        """Get all log text."""
        return self._log_console.toPlainText()


__all__ = ["InfoLogContainer"]

