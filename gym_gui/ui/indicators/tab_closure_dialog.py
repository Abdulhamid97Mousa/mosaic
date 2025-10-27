"""Tab closure dialog with pause/keep/archive/delete workflow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_TAB_CLOSURE_DIALOG_OPENED,
    LOG_UI_TAB_CLOSURE_CHOICE_SELECTED,
    LOG_UI_TAB_CLOSURE_CHOICE_CANCELLED,
)

_LOGGER = logging.getLogger(__name__)


class TabClosureChoice(str, Enum):
    """User's choice when closing a live training tab."""

    CANCEL = "cancel"  # Don't close the tab
    KEEP_AND_CLOSE = "keep"  # Keep data, close tab (training may continue)
    ARCHIVE = "archive"  # Freeze run snapshot, close tab
    DELETE = "delete"  # Purge run data, close tab


@dataclass
class RunSummary:
    """Summary statistics about a training run for display in closure dialog."""

    run_id: str
    agent_id: str
    episodes_collected: int
    steps_collected: int
    dropped_episodes: int
    dropped_steps: int
    total_reward: float
    is_active: bool
    last_update_timestamp: str


class TabClosureDialog(QtWidgets.QDialog, LogConstantMixin):
    """
    Modal dialog for handling live training tab closure with pause/keep/archive/delete options.

    Follows TASK_6 design:
    - Shows run summary (episodes, dropped metrics, reward)
    - Pauses training if necessary
    - Offers three explicit paths: Keep / Archive / Delete
    - Emits decision for caller to execute
    """

    action_selected = QtCore.pyqtSignal(TabClosureChoice)
    paused = QtCore.pyqtSignal()  # Emitted when training has been paused

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._logger = _LOGGER  # Initialize LogConstantMixin
        self.setWindowTitle("Close Live Training Tab")
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setMinimumWidth(600)
        self.setMinimumHeight(700)

        self._current_summary: Optional[RunSummary] = None
        self._selected_choice: TabClosureChoice = TabClosureChoice.CANCEL

        self._init_ui()
        self._apply_styling()
        
        # Initialize selected choice to KEEP (default radio button)
        self._update_selected_choice()

    def _init_ui(self) -> None:
        """Build dialog layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title / description
        self._title_label = QtWidgets.QLabel("Run Summary")
        self._title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(self._title_label)

        # Summary info (episodes, reward, dropped)
        self._summary_group = QtWidgets.QGroupBox("Training Statistics")
        summary_layout = QtWidgets.QGridLayout()
        summary_layout.setSpacing(10)

        self._run_label = QtWidgets.QLabel("Run ID: —")
        self._run_label.setStyleSheet("font-weight: 500;")
        self._episodes_label = QtWidgets.QLabel("Episodes: 0")
        self._steps_label = QtWidgets.QLabel("Steps: 0")
        self._dropped_label = QtWidgets.QLabel("Dropped: 0 steps, 0 episodes")
        self._reward_label = QtWidgets.QLabel("Total Reward: 0.0")

        summary_layout.addWidget(self._run_label, 0, 0, 1, 2)
        summary_layout.addWidget(self._episodes_label, 1, 0)
        summary_layout.addWidget(self._steps_label, 1, 1)
        summary_layout.addWidget(self._dropped_label, 2, 0, 1, 2)
        summary_layout.addWidget(self._reward_label, 3, 0, 1, 2)

        self._summary_group.setLayout(summary_layout)
        layout.addWidget(self._summary_group)

        # Separator
        layout.addSpacing(10)

        # Choice prompt
        prompt_label = QtWidgets.QLabel("Choose what happens to this run's data:")
        prompt_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(prompt_label)

        # Radio buttons for choices - use button group to ensure mutual exclusivity
        self._choice_group = QtWidgets.QGroupBox("Data Retention")
        choice_layout = QtWidgets.QVBoxLayout()
        
        # Create a button group to manage radio buttons
        self._button_group = QtWidgets.QButtonGroup()

        # Keep option
        self._keep_radio = QtWidgets.QRadioButton("Keep data (default)")
        self._keep_radio.setChecked(True)
        self._button_group.addButton(self._keep_radio, 0)
        keep_desc = QtWidgets.QLabel(
            "Retain telemetry and keep run history accessible.\nTraining may continue."
        )
        keep_desc.setStyleSheet("color: #666; margin-left: 25px; font-size: 9pt;")
        choice_layout.addWidget(self._keep_radio)
        choice_layout.addWidget(keep_desc)

        choice_layout.addSpacing(5)

        # Archive option
        self._archive_radio = QtWidgets.QRadioButton("Archive snapshot")
        self._button_group.addButton(self._archive_radio, 1)
        archive_desc = QtWidgets.QLabel(
            "Seal the run for replay; move it to archived run list for review."
        )
        archive_desc.setStyleSheet("color: #666; margin-left: 25px; font-size: 9pt;")
        choice_layout.addWidget(self._archive_radio)
        choice_layout.addWidget(archive_desc)

        choice_layout.addSpacing(5)

        # Delete option (with warning icon)
        self._delete_radio = QtWidgets.QRadioButton("Delete data  (⚠ Warning)")
        self._button_group.addButton(self._delete_radio, 2)
        delete_desc = QtWidgets.QLabel(
            "Remove telemetry from storage and worker cache. CANNOT BE UNDONE."
        )
        delete_desc.setStyleSheet("color: #d9534f; margin-left: 25px; font-size: 9pt;")
        choice_layout.addWidget(self._delete_radio)
        choice_layout.addWidget(delete_desc)

        choice_layout.addStretch()

        self._choice_group.setLayout(choice_layout)
        layout.addWidget(self._choice_group)

        # Batch close checkbox
        self._batch_checkbox = QtWidgets.QCheckBox("Apply to other tabs from this run")
        layout.addWidget(self._batch_checkbox)

        # Separator
        layout.addSpacing(5)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        self._cancel_button = QtWidgets.QPushButton("Cancel")
        self._cancel_button.setMinimumWidth(100)
        self._cancel_button.clicked.connect(self._on_cancel)

        self._continue_button = QtWidgets.QPushButton("Continue")
        self._continue_button.setMinimumWidth(100)
        self._continue_button.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold;")
        self._continue_button.clicked.connect(self._on_continue)

        button_layout.addWidget(self._cancel_button)
        button_layout.addWidget(self._continue_button)

        layout.addLayout(button_layout)

    def _apply_styling(self) -> None:
        """Apply consistent styling to dialog."""
        self.setStyleSheet(
            """
            QDialog {
                background-color: #f5f5f5;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px 15px;
                background-color: #fff;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
            }
            """
        )

    def set_run_summary(self, summary: RunSummary) -> None:
        """
        Set the run summary data to display in the dialog.

        Args:
            summary: RunSummary dataclass with episode/step/reward counts
        """
        self._current_summary = summary

        # Log dialog opened with run info
        self.log_constant(
            LOG_UI_TAB_CLOSURE_DIALOG_OPENED,
            message=f"Dialog opened for run_id={summary.run_id}, agent_id={summary.agent_id}, "
                    f"episodes={summary.episodes_collected}, steps={summary.steps_collected}",
            extra={"run_id": summary.run_id, "agent_id": summary.agent_id}
        )

        # Update labels
        self._episodes_label.setText(f"Episodes: {summary.episodes_collected}")
        self._steps_label.setText(f"Steps: {summary.steps_collected}")
        self._dropped_label.setText(f"Dropped: {summary.dropped_steps} steps, {summary.dropped_episodes} episodes")
        self._reward_label.setText(f"Total Reward: {summary.total_reward:.2f}")

        # Update title with agent ID
        title_run_preview = summary.run_id if len(summary.run_id) <= 16 else f"{summary.run_id[:12]}…"
        self._title_label.setText(f"Close Live Tab for {summary.agent_id} — Run {title_run_preview}")
        self._run_label.setText(f"Run ID: {summary.run_id}")

    def get_selected_choice(self) -> TabClosureChoice:
        """Get the user's selected action."""
        if self._archive_radio.isChecked():
            return TabClosureChoice.ARCHIVE
        elif self._delete_radio.isChecked():
            return TabClosureChoice.DELETE
        elif self._keep_radio.isChecked():
            return TabClosureChoice.KEEP_AND_CLOSE
        else:
            return TabClosureChoice.CANCEL

    def _update_selected_choice(self) -> None:
        """Update the internal selected choice based on current radio button state."""
        self._selected_choice = self.get_selected_choice()

    def is_batch_apply(self) -> bool:
        """Check if user wants to apply decision to all tabs from this run."""
        return self._batch_checkbox.isChecked()

    def _on_cancel(self) -> None:
        """User clicked Cancel button."""
        self._selected_choice = TabClosureChoice.CANCEL
        run_id = self._current_summary.run_id if self._current_summary else "unknown"
        self.log_constant(
            LOG_UI_TAB_CLOSURE_CHOICE_CANCELLED,
            message=f"Tab closure cancelled for run_id={run_id}",
            extra={"run_id": run_id}
        )
        self.reject()

    def _on_continue(self) -> None:
        """User clicked Continue button."""
        self._selected_choice = self.get_selected_choice()
        run_id = self._current_summary.run_id if self._current_summary else "unknown"
        self.log_constant(
            LOG_UI_TAB_CLOSURE_CHOICE_SELECTED,
            message=f"Tab closure choice selected - choice={self._selected_choice.value} for run_id={run_id}",
            extra={"run_id": run_id, "choice": self._selected_choice.value}
        )
        self.action_selected.emit(self._selected_choice)
        self.accept()

    def exec(self) -> TabClosureChoice:
        """
        Show modal dialog and return user's choice.

        Returns:
            TabClosureChoice enum value
        """
        result = super().exec()
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            return self._selected_choice
        return TabClosureChoice.CANCEL
