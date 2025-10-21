"""Live raw step data tab for agent training runs."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.ui.widgets.base_telemetry_tab import BaseTelemetryTab


class AgentOnlineRawTab(BaseTelemetryTab):
    """Displays scrolling raw JSON step data for debugging agent runs."""

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        *,
        max_lines: int = 100,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        self._max_lines = max_lines
        self._step_count = 0

        super().__init__(run_id, agent_id, parent=parent)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Use inherited header builder and extend it
        header = self._build_header()
        self._step_count_label = QtWidgets.QLabel("<b>Steps:</b> 0")
        header.addWidget(self._step_count_label)
        layout.addLayout(header)

        # Raw data view
        self._view = QtWidgets.QPlainTextEdit(self)
        self._view.setReadOnly(True)
        self._view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._view.setMaximumBlockCount(self._max_lines + 10)  # +10 buffer
        self._view.setFont(QtGui.QFont("Monospace", 9))
        layout.addWidget(self._view, 1)

        # Footer
        footer = QtWidgets.QLabel(
            f"Showing last {self._max_lines} steps • Older steps auto-discarded"
        )
        footer.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(footer)

    def on_step(self, step: Dict[str, Any]) -> None:
        """Append step as JSON line."""
        self._step_count += 1
        self._step_count_label.setText(f"<b>Steps:</b> {self._step_count}")

        try:
            # Compact JSON formatting
            line = json.dumps(step, separators=(",", ":"), sort_keys=False)
        except (TypeError, ValueError):
            line = f"<non-serializable: {type(step).__name__}>"

        self._view.appendPlainText(line)

        # Trim old lines if exceeding max
        doc = self._view.document()
        if doc is not None and doc.blockCount() > self._max_lines:
            cursor = self._view.textCursor()
            cursor.movePosition(QtWidgets.QTextCursor.MoveOperation.Start)
            cursor.select(QtWidgets.QTextCursor.SelectionType.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()  # Remove newline


__all__ = ["AgentOnlineRawTab"]
