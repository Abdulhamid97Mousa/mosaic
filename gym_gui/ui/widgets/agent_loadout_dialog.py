from __future__ import annotations

"""Dialog allowing the user to select an available agent backend."""

from typing import Any, Optional, Tuple

from qtpy import QtCore, QtWidgets

from gym_gui.services.actor import ActorDescriptor
from gym_gui.core.enums import GameId


class AgentLoadoutDialog(QtWidgets.QDialog):
    """Simple dialog that allows selecting an available agent backend."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        descriptors: Tuple[ActorDescriptor, ...],
        *,
        current_actor: Optional[str] = None,
        default_game: Optional[GameId] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Agent Loadout")
        self.setModal(True)
        self._descriptor_by_id = {descriptor.actor_id: descriptor for descriptor in descriptors}
        self._default_game = default_game

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        intro = QtWidgets.QLabel(
            "Select the agent backend you want to activate. "
            "Use the dedicated “Train Agent” button in the control panel to launch headless training runs.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        content = QtWidgets.QWidget(self)
        content_layout = QtWidgets.QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        self._list = QtWidgets.QListWidget(self)
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setMinimumWidth(220)
        content_layout.addWidget(self._list, 1)

        details_box = QtWidgets.QGroupBox("Details", self)
        details_layout = QtWidgets.QFormLayout(details_box)
        details_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop)

        self._name_value = QtWidgets.QLabel("—", details_box)
        self._name_value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        details_layout.addRow("Name:", self._name_value)

        self._id_value = QtWidgets.QLabel("—", details_box)
        self._id_value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        details_layout.addRow("Actor ID:", self._id_value)

        self._policy_value = QtWidgets.QLabel("—", details_box)
        self._policy_value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        details_layout.addRow("Policy:", self._policy_value)

        self._backend_value = QtWidgets.QLabel("—", details_box)
        self._backend_value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        details_layout.addRow("Backend:", self._backend_value)

        self._description_value = QtWidgets.QLabel("Select an agent to view details.", details_box)
        self._description_value.setWordWrap(True)
        self._description_value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        details_layout.addRow("Description:", self._description_value)

        content_layout.addWidget(details_box, 1)
        layout.addWidget(content, 1)

        for descriptor in descriptors:
            item = QtWidgets.QListWidgetItem(descriptor.display_name, self._list)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, descriptor.actor_id)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, descriptor)
            tooltip_lines = [descriptor.display_name]
            if descriptor.policy_label:
                tooltip_lines.append(f"Policy: {descriptor.policy_label}")
            if descriptor.backend_label:
                tooltip_lines.append(f"Backend: {descriptor.backend_label}")
            if descriptor.description:
                if tooltip_lines:
                    tooltip_lines.append("")
                tooltip_lines.append(descriptor.description)
            item.setToolTip("\n".join(tooltip_lines))
            if descriptor.actor_id == current_actor:
                self._list.setCurrentItem(item)

        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        layout.addWidget(self._button_box)

        self._ok_button = self._button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self._list.currentItemChanged.connect(self._on_selection_changed)
        self._list.itemDoubleClicked.connect(lambda _: self.accept())

        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        if self._list.currentItem() is None and self._list.count() > 0:
            self._list.setCurrentRow(0)
        self._update_ok_state(self._list.currentItem())
        self._update_details(self._list.currentItem())

    @property
    def selected_actor_id(self) -> Optional[str]:
        item = self._list.currentItem()
        if item is None:
            return None
        data = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return data if isinstance(data, str) else None

    @property
    def training_config(self) -> Optional[dict[str, Any]]:
        """Training is now configured exclusively via the dedicated Train Agent button."""
        return None

    def _on_selection_changed(
        self,
        current: Optional[QtWidgets.QListWidgetItem],
        _: Optional[QtWidgets.QListWidgetItem],
    ) -> None:
        self._update_ok_state(current)
        self._update_details(current)

    def _update_ok_state(self, item: Optional[QtWidgets.QListWidgetItem]) -> None:
        if self._ok_button is not None:
            self._ok_button.setEnabled(item is not None)

    def _update_details(self, item: Optional[QtWidgets.QListWidgetItem]) -> None:
        if item is None:
            self._name_value.setText("—")
            self._id_value.setText("—")
            self._policy_value.setText("—")
            self._backend_value.setText("—")
            self._description_value.setText("Select an agent to view details.")
            return

        descriptor = item.data(QtCore.Qt.ItemDataRole.UserRole + 1)
        if not isinstance(descriptor, ActorDescriptor):
            actor_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
            descriptor = self._descriptor_by_id.get(actor_id if isinstance(actor_id, str) else "")

        if descriptor is None:
            self._name_value.setText("—")
            self._id_value.setText("—")
            self._policy_value.setText("—")
            self._backend_value.setText("—")
            self._description_value.setText("No metadata available.")
            return

        self._name_value.setText(descriptor.display_name)
        self._id_value.setText(descriptor.actor_id)
        self._policy_value.setText(descriptor.policy_label or "Not specified")
        self._backend_value.setText(descriptor.backend_label or "Not specified")
        self._description_value.setText(descriptor.description or "No description provided.")


__all__ = ["AgentLoadoutDialog"]
