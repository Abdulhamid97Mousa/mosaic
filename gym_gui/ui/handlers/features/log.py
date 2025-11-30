"""Log filtering and display handlers using composition pattern.

This module provides a handler class that manages log filtering, formatting,
and console display.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:
    from qtpy.QtWidgets import QComboBox, QPlainTextEdit
    from gym_gui.ui.logging_bridge import LogRecordPayload


class LogHandler:
    """Handles log filtering and display for the log console.

    This class encapsulates all log-related logic including filtering by
    component and severity, formatting log records, and managing the console.

    Args:
        log_filter: The component filter combo box.
        log_severity_filter: The severity filter combo box.
        log_console: The plain text edit widget for displaying logs.
        severity_options: Mapping of severity display names to filter values.
        initial_components: Initial set of known component names.
    """

    def __init__(
        self,
        log_filter: "QComboBox",
        log_severity_filter: "QComboBox",
        log_console: "QPlainTextEdit",
        severity_options: Dict[str, str | None],
        initial_components: List[str] | None = None,
    ) -> None:
        self._log_filter = log_filter
        self._log_severity_filter = log_severity_filter
        self._log_console = log_console
        self._severity_options = severity_options
        self._log_records: List["LogRecordPayload"] = []
        self._component_filter_set: Set[str] = set(initial_components or [])

    @property
    def log_records(self) -> List["LogRecordPayload"]:
        """Return the list of log records."""
        return self._log_records

    def append_log_record(self, payload: "LogRecordPayload") -> None:
        """Append a new log record and update the console if it passes filters."""
        if payload.component and payload.component not in self._component_filter_set:
            self._component_filter_set.add(payload.component)
            self._log_filter.addItem(payload.component)
        self._log_records.append(payload)
        if self._passes_filter(payload):
            self._log_console.appendPlainText(self._format_log(payload))
            scrollbar = self._log_console.verticalScrollBar()
            if scrollbar is not None:
                scrollbar.setValue(scrollbar.maximum())

    def on_filter_changed(self, _: str) -> None:
        """Handle log filter combo box changes."""
        self._refresh_console()

    def _passes_filter(self, payload: "LogRecordPayload") -> bool:
        """Check if a log record passes current component and severity filters."""
        # Check component filter
        selected_component = self._log_filter.currentText()
        if selected_component != "All" and payload.component != selected_component:
            return False

        # Check severity filter
        selected_severity = self._log_severity_filter.currentText()
        severity = self._severity_options.get(selected_severity)
        if severity and payload.level != severity:
            return False

        return True

    @staticmethod
    def _format_log(payload: "LogRecordPayload") -> str:
        """Format a log record for display in the console."""
        ts = datetime.fromtimestamp(payload.created).strftime("%H:%M:%S")
        component = payload.component or "Unknown"
        return f"{ts} | {payload.level:<7} | {component:<12} | {payload.name} | {payload.message}"

    def _refresh_console(self) -> None:
        """Refresh the log console by re-applying filters to all records."""
        self._log_console.clear()
        for record in self._log_records:
            if self._passes_filter(record):
                self._log_console.appendPlainText(self._format_log(record))
        scrollbar = self._log_console.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())


__all__ = ["LogHandler"]
