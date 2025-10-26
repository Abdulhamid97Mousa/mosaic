from __future__ import annotations

"""Centralised confirmation dialog helpers used across the GUI."""

from typing import Optional

from PyQt6 import QtWidgets


class ConfirmationService:
    """Provides standard confirmation prompts so callers stay consistent."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._parent = parent

    def confirm_run_cancel(self, run_name: str) -> bool:
        message = (
            "Stopping \"{run}\" will discard any buffered telemetry and queued jobs. \n" "Do you want to continue?"
        ).format(run=run_name)
        return self._ask_yes_no("Cancel Training Run", message)

    def confirm_close_tab(self, run_name: str) -> bool:
        message = (
            "Closing the tab hides progress indicators for \"{run}\". \n" "You can restore it from the replay menu later."
        ).format(run=run_name)
        return self._ask_yes_no("Close Run Tab", message)

    def confirm_overwrite_checkpoint(self, checkpoint_name: str) -> bool:
        message = (
            "A checkpoint named \"{checkpoint}\" already exists. \n" "Overwriting it will replace the previous data."
        ).format(checkpoint=checkpoint_name)
        return self._ask_yes_no("Overwrite Checkpoint", message)

    def _ask_yes_no(self, title: str, message: str) -> bool:
        box = QtWidgets.QMessageBox(parent=self._parent)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(message)
        box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
        result = box.exec()
        return result == QtWidgets.QMessageBox.StandardButton.Yes
