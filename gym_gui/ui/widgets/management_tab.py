"""Management tab for viewing and managing training runs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.services.trainer.registry import RunStatus

if TYPE_CHECKING:
    from gym_gui.services.trainer.run_manager import TrainingRunManager


_LOGGER = logging.getLogger(__name__)


class ManagementTab(QtWidgets.QWidget):
    """Tab widget for managing training runs and their artifacts.

    This tab displays all training runs from the registry and provides
    controls for stopping running jobs, deleting runs, and bulk cleanup.

    Signals:
        runs_deleted: Emitted with list of run_ids when runs are deleted.
            Connect to RenderTabs to close associated dynamic tabs.
    """

    # Signal emitted when runs are deleted (list of run_ids)
    runs_deleted = QtCore.Signal(list)
    # Signal emitted when tabs should be closed without deleting data (list of run_ids)
    tabs_closed = QtCore.Signal(list)

    def __init__(
        self,
        run_manager: TrainingRunManager,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the management tab.

        Args:
            run_manager: The training run manager service.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._run_manager = run_manager
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header group box with title, description, and buttons
        header_group = QtWidgets.QGroupBox("Training Run Management")
        header_layout = QtWidgets.QVBoxLayout(header_group)
        header_layout.setContentsMargins(8, 4, 8, 8)
        header_layout.setSpacing(6)

        # Description inside the group
        desc = QtWidgets.QLabel(
            "Stop running jobs, delete runs and artifacts, clean up stale tabs."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888;")
        header_layout.addWidget(desc)

        # Button bar
        button_bar = QtWidgets.QHBoxLayout()
        button_bar.setSpacing(6)

        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.setToolTip("Reload the list of training runs")
        self._refresh_btn.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )

        self._stop_btn = QtWidgets.QPushButton("Stop Selected")
        self._stop_btn.setToolTip("Stop the selected running training jobs")
        self._stop_btn.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        self._stop_btn.setEnabled(False)

        self._close_tabs_btn = QtWidgets.QPushButton("Close Tabs")
        self._close_tabs_btn.setToolTip(
            "Close the associated tabs (CleanRL-Live, TensorBoard, etc.) "
            "without deleting run data"
        )
        self._close_tabs_btn.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        self._close_tabs_btn.setEnabled(False)

        self._delete_btn = QtWidgets.QPushButton("Delete Selected")
        self._delete_btn.setToolTip(
            "Delete selected runs and all associated artifacts"
        )
        self._delete_btn.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        self._delete_btn.setEnabled(False)

        self._clear_term_btn = QtWidgets.QPushButton("Clear Terminated")
        self._clear_term_btn.setToolTip("Delete all terminated/completed runs")
        self._clear_term_btn.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        self._clear_term_btn.setEnabled(False)

        self._clear_all_btn = QtWidgets.QPushButton("Clear All")
        self._clear_all_btn.setToolTip("Delete all training runs (requires confirmation)")
        self._clear_all_btn.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        self._clear_all_btn.setEnabled(False)

        for btn in [
            self._refresh_btn,
            self._stop_btn,
            self._close_tabs_btn,
            self._delete_btn,
            self._clear_term_btn,
            self._clear_all_btn,
        ]:
            button_bar.addWidget(btn)
        button_bar.addStretch()

        header_layout.addLayout(button_bar)
        layout.addWidget(header_group)

        # Stacked widget for table/placeholder
        self._content_stack = QtWidgets.QStackedWidget(self)

        # Table (index 0)
        self._table = QtWidgets.QTableWidget(0, 8, self)
        self._table.setHorizontalHeaderLabels(
            ["Run ID", "Agent", "Status", "Outcome", "Created", "Failure", "Reason", "Disk Size"]
        )
        header_view = self._table.horizontalHeader()
        if header_view is not None:
            header_view.setStretchLastSection(True)
            header_view.setSectionResizeMode(
                0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            header_view.setSectionResizeMode(
                1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
            header_view.setSectionResizeMode(
                2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )

        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setAlternatingRowColors(True)
        self._content_stack.addWidget(self._table)  # Index 0: Table

        # Placeholder for empty state (index 1)
        placeholder_widget = QtWidgets.QWidget()
        placeholder_layout = QtWidgets.QVBoxLayout(placeholder_widget)
        placeholder_layout.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QtWidgets.QLabel("No training runs found.")
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #888; font-style: italic; font-size: 13px; padding: 40px;"
        )

        placeholder_hint = QtWidgets.QLabel(
            "Training runs will appear here when you start training via\n"
            "CleanRL, Ray RLlib, or other training forms."
        )
        placeholder_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        placeholder_hint.setStyleSheet("color: #666; font-size: 11px;")
        placeholder_hint.setWordWrap(True)

        placeholder_layout.addStretch()
        placeholder_layout.addWidget(self._placeholder)
        placeholder_layout.addWidget(placeholder_hint)
        placeholder_layout.addStretch()

        self._content_stack.addWidget(placeholder_widget)  # Index 1: Placeholder

        # Add stacked widget to main layout
        layout.addWidget(self._content_stack, 1)

        # Initially show placeholder
        self._content_stack.setCurrentIndex(1)

    def _connect_signals(self) -> None:
        """Connect button signals to handlers."""
        self._refresh_btn.clicked.connect(self.refresh)
        self._stop_btn.clicked.connect(self._stop_selected)
        self._close_tabs_btn.clicked.connect(self._close_tabs_selected)
        self._delete_btn.clicked.connect(self._delete_selected)
        self._clear_term_btn.clicked.connect(self._clear_terminated)
        self._clear_all_btn.clicked.connect(self._clear_all)
        self._table.itemSelectionChanged.connect(self._update_button_states)

    def refresh(self) -> None:
        """Reload runs from registry with disk sizes."""
        runs = self._run_manager.list_runs()
        self._table.setRowCount(0)

        for run in runs:
            row = self._table.rowCount()
            self._table.insertRow(row)

            disk_size = self._run_manager.get_run_disk_size(run.run_id)
            agent_type = self._extract_agent_type(run.run_id)

            # Format timestamp
            created_str = run.created_at.strftime("%Y-%m-%d %H:%M")

            # Truncate reason for table display; full text in tooltip
            reason_text = run.reason or ""
            reason_display = reason_text.replace("\n", " ")
            if len(reason_display) > 120:
                reason_display = reason_display[:117] + "..."

            values = [
                run.run_id[:12] + "...",
                agent_type,
                run.status.value.upper(),
                run.outcome or "—",
                created_str,
                run.failure_reason or "—",
                reason_display or "—",
                self._format_size(disk_size),
            ]

            for col, val in enumerate(values):
                item = QtWidgets.QTableWidgetItem(val)
                # Store full run_id in user role for all columns
                item.setData(QtCore.Qt.ItemDataRole.UserRole, run.run_id)
                # Store status in a custom role for the status column
                if col == 2:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, run.status.value)
                    # Color code status
                    if run.status == RunStatus.EXECUTING:
                        item.setForeground(QtGui.QColor("#2196F3"))  # Blue
                    elif run.status == RunStatus.TERMINATED:
                        if run.outcome == "succeeded":
                            item.setForeground(QtGui.QColor("#4CAF50"))  # Green
                        else:
                            item.setForeground(QtGui.QColor("#f44336"))  # Red
                    elif run.status == RunStatus.READY:
                        item.setForeground(QtGui.QColor("#FF9800"))  # Orange
                # Reason column: show full stderr in tooltip
                if col == 6 and reason_text:
                    item.setToolTip(reason_text)
                # Right-align disk size
                if col == 7:
                    item.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                self._table.setItem(row, col, item)

        self._update_visibility()
        self._update_button_states()

    def _extract_agent_type(self, run_id: str) -> str:
        """Extract the agent/worker type from run config.

        The config structure is:
        {
            "entry_point": "...",
            "metadata": {
                "ui": {"worker_id": "ray_worker", ...},
                "worker": {"module": "ray_worker.cli", ...}
            }
        }
        """
        config_json = self._run_manager.get_run_config_json(run_id)
        if config_json is None:
            return "Unknown"

        try:
            config = json.loads(config_json)
            metadata = config.get("metadata", {})

            # Check metadata.ui.worker_id (most reliable for UI-submitted runs)
            ui_meta = metadata.get("ui", {})
            worker_id = ui_meta.get("worker_id", "")
            if worker_id:
                return self._worker_id_to_display_name(worker_id)

            # Check metadata.worker.module (worker dispatch info)
            worker_meta = metadata.get("worker", {})
            module = worker_meta.get("module", "")
            if module:
                return self._module_to_display_name(module)

            # Fallback: check entry_point at root level
            entry_point = config.get("entry_point", "")
            if entry_point:
                return self._entry_point_to_display_name(entry_point)

            return "Agent"
        except (json.JSONDecodeError, TypeError):
            return "Unknown"

    def _worker_id_to_display_name(self, worker_id: str) -> str:
        """Convert worker_id to human-readable display name."""
        mapping = {
            "ray_worker": "Ray RLlib",
            "cleanrl_worker": "CleanRL",
            "xuance_worker": "XuanCe",
        }
        lower_id = worker_id.lower()
        for key, name in mapping.items():
            if key in lower_id:
                return name
        return worker_id.replace("_", " ").title()

    def _module_to_display_name(self, module: str) -> str:
        """Convert module path to human-readable display name."""
        lower_module = module.lower()
        if "ray_worker" in lower_module:
            return "Ray RLlib"
        if "cleanrl_worker" in lower_module:
            return "CleanRL"
        if "xuance_worker" in lower_module:
            return "XuanCe"
        # Extract last part of module path
        parts = module.split(".")
        return parts[0].replace("_", " ").title() if parts else "Agent"

    def _entry_point_to_display_name(self, entry_point: str) -> str:
        """Convert entry_point to human-readable display name."""
        lower_ep = entry_point.lower()
        if "cleanrl" in lower_ep:
            return "CleanRL"
        if "ray" in lower_ep:
            return "Ray RLlib"
        if "xuance" in lower_ep:
            return "XuanCe"
        return "Agent"

    def _get_selected_run_ids(self) -> list[str]:
        """Get the run IDs of all selected rows."""
        selection = self._table.selectionModel()
        if selection is None:
            return []

        run_ids = set()
        for index in selection.selectedRows():
            item = self._table.item(index.row(), 0)
            if item is not None:
                run_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if run_id:
                    run_ids.add(run_id)

        return list(run_ids)

    def _get_selected_run_statuses(self) -> list[str]:
        """Get the statuses of all selected rows."""
        selection = self._table.selectionModel()
        if selection is None:
            return []

        statuses = []
        for index in selection.selectedRows():
            item = self._table.item(index.row(), 2)  # Status column
            if item is not None:
                status = item.data(QtCore.Qt.ItemDataRole.UserRole + 1)
                if status:
                    statuses.append(status)

        return statuses

    def _stop_selected(self) -> None:
        """Stop selected running jobs."""
        run_ids = self._get_selected_run_ids()
        if not run_ids:
            return

        # Filter to only EXECUTING runs
        executing_runs = []
        for run_id in run_ids:
            runs = self._run_manager.list_runs(statuses=[RunStatus.EXECUTING])
            if any(r.run_id == run_id for r in runs):
                executing_runs.append(run_id)

        if not executing_runs:
            QtWidgets.QMessageBox.information(
                self,
                "No Running Jobs",
                "None of the selected runs are currently executing.",
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Stop Training Jobs",
            f"Are you sure you want to stop {len(executing_runs)} running training job(s)?\n\n"
            "The jobs will be gracefully terminated.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        stopped = 0
        for run_id in executing_runs:
            if self._run_manager.cancel_run(run_id):
                stopped += 1

        _LOGGER.info(
            "Stopped training jobs",
            extra={"count": stopped, "requested": len(executing_runs)},
        )

        # Refresh after a short delay to allow status updates
        QtCore.QTimer.singleShot(500, self.refresh)

    def _close_tabs_selected(self) -> None:
        """Close associated tabs for selected runs without deleting data."""
        run_ids = self._get_selected_run_ids()
        if not run_ids:
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Close Tabs",
            f"Close all tabs associated with {len(run_ids)} selected run(s)?\n\n"
            "This will close:\n"
            "- CleanRL-Live tabs\n"
            "- TensorBoard tabs\n"
            "- WandB tabs\n"
            "- Any other dynamic tabs for these runs\n\n"
            "The run data will NOT be deleted.\n"
            "Tabs will reappear on next program restart.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )

        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        _LOGGER.info(
            "Closing tabs for runs",
            extra={"run_ids": run_ids, "count": len(run_ids)},
        )

        self.tabs_closed.emit(run_ids)

    def _delete_selected(self) -> None:
        """Delete selected runs with confirmation."""
        run_ids = self._get_selected_run_ids()
        if not run_ids:
            return

        if not self._confirm_delete(len(run_ids)):
            return

        for run_id in run_ids:
            self._run_manager.delete_run_completely(run_id)

        self.runs_deleted.emit(run_ids)
        self.refresh()

    def _clear_terminated(self) -> None:
        """Delete all TERMINATED runs."""
        terminated_runs = self._run_manager.get_terminated_runs()
        if not terminated_runs:
            QtWidgets.QMessageBox.information(
                self,
                "No Terminated Runs",
                "There are no terminated runs to clear.",
            )
            return

        if not self._confirm_delete(len(terminated_runs), category="terminated"):
            return

        run_ids = [r.run_id for r in terminated_runs]
        for run_id in run_ids:
            self._run_manager.delete_run_completely(run_id)

        self.runs_deleted.emit(run_ids)
        self.refresh()

    def _clear_all(self) -> None:
        """Delete all runs with extra confirmation."""
        all_runs = self._run_manager.list_runs()
        if not all_runs:
            QtWidgets.QMessageBox.information(
                self,
                "No Runs",
                "There are no training runs to clear.",
            )
            return

        # Extra warning for clearing all
        reply = QtWidgets.QMessageBox.warning(
            self,
            "Clear All Training Runs",
            f"Are you sure you want to delete ALL {len(all_runs)} training run(s)?\n\n"
            "This will:\n"
            "- Stop any running training jobs\n"
            "- Delete all telemetry data\n"
            "- Delete all logs and TensorBoard files\n"
            "- Close all associated tabs\n\n"
            "This action cannot be undone!",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        # Second confirmation for safety
        confirm_text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Confirm Clear All",
            'Type "DELETE ALL" to confirm:',
        )

        if not ok or confirm_text != "DELETE ALL":
            return

        run_ids = [r.run_id for r in all_runs]

        # Cancel any executing runs first
        for run in all_runs:
            if run.status == RunStatus.EXECUTING:
                self._run_manager.cancel_run(run.run_id)

        # Delete all runs
        for run_id in run_ids:
            self._run_manager.delete_run_completely(run_id)

        self.runs_deleted.emit(run_ids)
        self.refresh()

    def _confirm_delete(
        self, count: int, category: str = "selected"
    ) -> bool:
        """Show a confirmation dialog for deletion.

        Args:
            count: Number of runs to delete.
            category: Description of the runs (e.g., "selected", "terminated").

        Returns:
            True if the user confirmed, False otherwise.
        """
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {count} {category} run(s)?\n\n"
            "This will remove:\n"
            "- Database entries\n"
            "- Telemetry data (episodes/steps)\n"
            "- Log files and TensorBoard data\n"
            "- Configuration files\n"
            "- Associated tabs\n\n"
            "This action cannot be undone.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def _update_visibility(self) -> None:
        """Update visibility of table vs placeholder using stacked widget."""
        has_rows = self._table.rowCount() > 0
        # Index 0 = table, Index 1 = placeholder
        self._content_stack.setCurrentIndex(0 if has_rows else 1)

    def _update_button_states(self) -> None:
        """Update button enabled states based on selection and data."""
        run_ids = self._get_selected_run_ids()
        has_selection = len(run_ids) > 0

        # Check if any selected runs are executing
        statuses = self._get_selected_run_statuses()
        has_executing = RunStatus.EXECUTING.value in statuses

        # Check if there are any terminated runs
        terminated_runs = self._run_manager.get_terminated_runs()
        has_terminated = len(terminated_runs) > 0

        # Check if there are any runs at all
        all_runs = self._run_manager.list_runs()
        has_any = len(all_runs) > 0

        self._stop_btn.setEnabled(has_selection and has_executing)
        self._close_tabs_btn.setEnabled(has_selection)
        self._delete_btn.setEnabled(has_selection)
        self._clear_term_btn.setEnabled(has_terminated)
        self._clear_all_btn.setEnabled(has_any)

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes as human-readable size."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


__all__ = ["ManagementTab"]
