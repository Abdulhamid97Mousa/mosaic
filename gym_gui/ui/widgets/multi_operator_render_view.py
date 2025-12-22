"""Multi-operator render view for side-by-side agent comparison.

This module provides a grid layout container that holds N operator render
containers, each displaying its own agent's output in real-time.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import pyqtSignal  # type: ignore[attr-defined]
from qtpy import QtCore, QtWidgets

from gym_gui.services.operator import OperatorConfig
from gym_gui.ui.widgets.operator_render_container import OperatorRenderContainer

_LOGGER = logging.getLogger(__name__)


class MultiOperatorRenderView(QtWidgets.QWidget):
    """Container for N operator render views in a dynamic grid layout.

    Automatically adjusts grid layout based on operator count:
    - 1 operator: full width
    - 2 operators: side by side (1 row, 2 cols)
    - 3-4 operators: 2x2 grid
    - 5-6 operators: 2x3 grid
    - 7-9 operators: 3x3 grid
    - etc.
    """

    operator_status_changed = pyqtSignal(str, str)  # operator_id, new_status

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._containers: Dict[str, OperatorRenderContainer] = {}
        self._operator_order: List[str] = []  # Maintain insertion order

        self._build_ui()

    def _build_ui(self) -> None:
        self._main_layout = QtWidgets.QVBoxLayout(self)
        self._main_layout.setContentsMargins(4, 4, 4, 4)
        self._main_layout.setSpacing(4)

        # Header with operator count
        self._header = QtWidgets.QWidget(self)
        header_layout = QtWidgets.QHBoxLayout(self._header)
        header_layout.setContentsMargins(4, 2, 4, 2)

        self._title_label = QtWidgets.QLabel("Multi-Operator View", self._header)
        self._title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self._title_label)

        self._count_label = QtWidgets.QLabel("0 operators", self._header)
        self._count_label.setStyleSheet("color: #666; font-size: 10px;")
        header_layout.addWidget(self._count_label)

        header_layout.addStretch()

        # Layout controls
        self._layout_label = QtWidgets.QLabel("Layout:", self._header)
        header_layout.addWidget(self._layout_label)

        self._layout_combo = QtWidgets.QComboBox(self._header)
        self._layout_combo.addItems(["Auto", "1 Column", "2 Columns", "3 Columns"])
        self._layout_combo.currentIndexChanged.connect(self._relayout)
        header_layout.addWidget(self._layout_combo)

        self._main_layout.addWidget(self._header)

        # Scroll area for grid
        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._grid_container = QtWidgets.QWidget(self._scroll)
        self._grid_layout = QtWidgets.QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(4, 4, 4, 4)
        self._grid_layout.setSpacing(8)

        self._scroll.setWidget(self._grid_container)
        self._main_layout.addWidget(self._scroll, 1)

        # Empty state placeholder
        self._empty_placeholder = QtWidgets.QLabel(
            "No operators configured.\n\n"
            "Add operators in the Control Panel to see their output here.",
            self._grid_container
        )
        self._empty_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._empty_placeholder.setStyleSheet(
            "color: #999; font-style: italic; font-size: 12px; padding: 40px;"
        )
        self._grid_layout.addWidget(self._empty_placeholder, 0, 0)

    def add_operator(self, config: OperatorConfig) -> None:
        """Add a new operator render container.

        Args:
            config: The operator configuration.
        """
        if config.operator_id in self._containers:
            _LOGGER.warning(f"Operator {config.operator_id} already exists")
            return

        container = OperatorRenderContainer(config, parent=self._grid_container)
        container.status_changed.connect(self.operator_status_changed.emit)

        self._containers[config.operator_id] = container
        self._operator_order.append(config.operator_id)

        self._relayout()
        _LOGGER.info(f"Added operator container: {config.operator_id}")

    def remove_operator(self, operator_id: str) -> None:
        """Remove an operator render container.

        Args:
            operator_id: ID of the operator to remove.
        """
        if operator_id not in self._containers:
            return

        container = self._containers.pop(operator_id)
        container.cleanup()

        # Remove from layout
        self._grid_layout.removeWidget(container)
        container.deleteLater()

        # Remove from order list
        if operator_id in self._operator_order:
            self._operator_order.remove(operator_id)

        self._relayout()
        _LOGGER.info(f"Removed operator container: {operator_id}")

    def update_operator(self, config: OperatorConfig) -> None:
        """Update an operator's configuration.

        Args:
            config: The updated configuration.
        """
        container = self._containers.get(config.operator_id)
        if container:
            container.set_config(config)

    def set_operator_status(self, operator_id: str, status: str) -> None:
        """Set the status of an operator.

        Args:
            operator_id: The operator ID.
            status: One of "pending", "running", "stopped", "error".
        """
        container = self._containers.get(operator_id)
        if container:
            container.set_status(status)

    def display_payload(self, operator_id: str, payload: Dict[str, Any]) -> None:
        """Display render payload for a specific operator.

        Args:
            operator_id: The operator to display the payload for.
            payload: The telemetry payload containing render data.
        """
        container = self._containers.get(operator_id)
        if container:
            container.display_payload(payload)

    def display_payload_by_run_id(self, run_id: str, payload: Dict[str, Any]) -> None:
        """Display render payload using run_id lookup.

        Args:
            run_id: The run ID associated with an operator.
            payload: The telemetry payload containing render data.
        """
        for container in self._containers.values():
            if container.config.run_id == run_id:
                container.display_payload(payload)
                return

    def get_operator_ids(self) -> List[str]:
        """Get list of all operator IDs in order."""
        return list(self._operator_order)

    def get_container(self, operator_id: str) -> Optional[OperatorRenderContainer]:
        """Get a specific operator container."""
        return self._containers.get(operator_id)

    def clear(self) -> None:
        """Remove all operator containers."""
        for operator_id in list(self._operator_order):
            self.remove_operator(operator_id)

    def _relayout(self) -> None:
        """Reorganize grid layout based on operator count and layout setting."""
        count = len(self._containers)

        # Update count label
        self._count_label.setText(f"{count} operator{'s' if count != 1 else ''}")

        # Show/hide empty placeholder
        if count == 0:
            if self._empty_placeholder.parent() is None:
                self._grid_layout.addWidget(self._empty_placeholder, 0, 0)
            self._empty_placeholder.show()
            return
        else:
            self._empty_placeholder.hide()

        # Remove all containers from layout (will re-add)
        for container in self._containers.values():
            self._grid_layout.removeWidget(container)

        # Determine number of columns
        layout_mode = self._layout_combo.currentText()
        if layout_mode == "1 Column":
            cols = 1
        elif layout_mode == "2 Columns":
            cols = 2
        elif layout_mode == "3 Columns":
            cols = 3
        else:  # Auto
            cols = self._calculate_optimal_columns(count)

        # Place containers in grid
        for i, operator_id in enumerate(self._operator_order):
            container = self._containers[operator_id]
            row = i // cols
            col = i % cols
            self._grid_layout.addWidget(container, row, col)

        # Set equal stretch factors
        for col in range(cols):
            self._grid_layout.setColumnStretch(col, 1)
        rows = math.ceil(count / cols) if cols > 0 else 1
        for row in range(rows):
            self._grid_layout.setRowStretch(row, 1)

    def _calculate_optimal_columns(self, count: int) -> int:
        """Calculate optimal number of columns based on operator count.

        Args:
            count: Number of operators.

        Returns:
            Optimal number of columns.
        """
        if count <= 1:
            return 1
        elif count == 2:
            return 2
        elif count <= 4:
            return 2
        elif count <= 6:
            return 3
        elif count <= 9:
            return 3
        else:
            return 4

    @property
    def operator_count(self) -> int:
        """Get the number of operator containers."""
        return len(self._containers)


__all__ = ["MultiOperatorRenderView"]
