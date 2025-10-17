from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

from qtpy import QtCore, QtWidgets, QtGui

from gym_gui.config.paths import VAR_TRAINER_DIR
from gym_gui.core.enums import GameId


@dataclass(slots=True)
class PolicyInfo:
    environment: str
    agent_id: str
    path: Path
    updated: datetime
    metadata: dict


def _policy_root() -> Path:
    custom_root = os.environ.get("GYM_GUI_VAR_DIR")
    if custom_root:
        return (
            Path(custom_root).expanduser().resolve() / "policies"
        )
    return (VAR_TRAINER_DIR / "policies").resolve()


class PolicySelectionDialog(QtWidgets.QDialog):
    """Dialog to select an existing trained policy/checkpoint."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        current_game: Optional[GameId] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Trained Policy")
        self.resize(640, 360)
        self._policy_root = _policy_root()
        self._policies: List[PolicyInfo] = self._discover_policies()
        self._selected_path: Optional[Path] = None
        self._current_game = current_game
        self._build_ui()

    # ------------------------------------------------------------------
    @property
    def selected_path(self) -> Optional[Path]:
        return self._selected_path

    # ------------------------------------------------------------------
    def _discover_policies(self) -> List[PolicyInfo]:
        if not self._policy_root.exists():
            return []

        policies: List[PolicyInfo] = []
        for path in sorted(self._policy_root.rglob("*.json")):
            rel = path.relative_to(self._policy_root)
            parts = rel.parts
            env_id = parts[0] if parts else "unknown"
            agent_id = path.stem
            try:
                payload = json.loads(path.read_text())
            except Exception:
                payload = {}
            timestamp = datetime.fromtimestamp(path.stat().st_mtime)
            info = PolicyInfo(
                environment=env_id,
                agent_id=payload.get("agent_id") or agent_id,
                path=path,
                updated=timestamp,
                metadata=payload if isinstance(payload, dict) else {},
            )
            policies.append(info)
        return policies

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        summary = QtWidgets.QLabel(self)
        if self._policies:
            summary.setText(
                f"Select a trained policy from <code>{self._policy_root}</code> to evaluate."
            )
        else:
            summary.setText(
                f"No policies found under <code>{self._policy_root}</code>."
            )
        summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
        summary.setWordWrap(True)
        layout.addWidget(summary)

        self._table = QtWidgets.QTableWidget(self)
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Environment", "Agent", "Updated", "Path"])
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        
        # Configure headers with null checks
        vertical_header = self._table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        
        horizontal_header = self._table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setStretchLastSection(True)
        
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table, 1)

        self._populate_table()

        controls = QtWidgets.QHBoxLayout()
        controls.addStretch(1)
        open_dir_button = QtWidgets.QPushButton("Open Folder", self)
        open_dir_button.setEnabled(self._policy_root.exists())
        open_dir_button.clicked.connect(self._open_policy_folder)
        controls.addWidget(open_dir_button)
        layout.addLayout(controls)

        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        self._ok_button = self._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if self._ok_button is not None:
            self._ok_button.setEnabled(False)
        layout.addWidget(self._button_box)

    # ------------------------------------------------------------------
    def _populate_table(self) -> None:
        self._table.setRowCount(0)
        for info in self._filtered_policies():
            row = self._table.rowCount()
            self._table.insertRow(row)
            items = [
                info.environment,
                info.agent_id,
                info.updated.strftime("%Y-%m-%d %H:%M:%S"),
                str(info.path),
            ]
            for col, text in enumerate(items):
                item = QtWidgets.QTableWidgetItem(text)
                if col == 0 and self._current_game is not None:
                    if text == self._current_game.value:
                        item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, True)
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                if col == 3:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, info.path)
                self._table.setItem(row, col, item)

    def _filtered_policies(self) -> Iterable[PolicyInfo]:
        if self._current_game is None:
            return self._policies
        preferred_env = self._current_game.value
        prioritized = [info for info in self._policies if info.environment == preferred_env]
        others = [info for info in self._policies if info.environment != preferred_env]
        return prioritized + others

    # ------------------------------------------------------------------
    def _on_selection_changed(self) -> None:
        selection = self._table.selectionModel()
        enable = bool(selection and selection.hasSelection())
        if self._ok_button is not None:
            self._ok_button.setEnabled(enable)

    def _on_accept(self) -> None:
        selection = self._table.selectionModel()
        if selection is None or not selection.hasSelection():
            return
        row = selection.selectedRows()[0].row()
        item = self._table.item(row, 3)
        path = item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
        if isinstance(path, Path):
            self._selected_path = path
        self.accept()

    def _open_policy_folder(self) -> None:
        if not self._policy_root.exists():
            QtWidgets.QMessageBox.information(
                self,
                "Folder Missing",
                f"{self._policy_root} does not exist yet.",
            )
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self._policy_root)))


__all__ = ["PolicySelectionDialog"]
