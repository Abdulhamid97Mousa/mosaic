"""Factory for creating input widgets based on setting type.

This module provides a factory class that creates the appropriate Qt widget
for each setting type (string, boolean, integer, enum, etc.).
"""

from __future__ import annotations

from typing import Optional

from qtpy import QtWidgets

from gym_gui.services.settings_service import SettingMetadata, SettingType


class SettingFieldFactory:
    """Factory for creating input widgets based on setting type.

    This factory creates Qt widgets appropriate for each setting type:
    - STRING/URL/EMAIL: QLineEdit (with password mode if sensitive)
    - BOOLEAN: QCheckBox
    - INTEGER: QSpinBox
    - ENUM: QComboBox

    Example:
        factory = SettingFieldFactory()
        widget = factory.create_field(metadata, parent)
    """

    @staticmethod
    def create_field(
        metadata: SettingMetadata,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> QtWidgets.QWidget:
        """Create appropriate widget based on setting type.

        Args:
            metadata: Setting metadata with type information
            parent: Parent widget

        Returns:
            Qt widget appropriate for the setting type
        """
        if metadata.value_type == SettingType.BOOLEAN:
            return SettingFieldFactory.create_boolean_field(metadata, parent)
        elif metadata.value_type == SettingType.INTEGER:
            return SettingFieldFactory.create_integer_field(metadata, parent)
        elif metadata.value_type == SettingType.ENUM:
            return SettingFieldFactory.create_enum_field(metadata, parent)
        else:
            # STRING, URL, EMAIL all use text input
            return SettingFieldFactory.create_string_field(metadata, parent)

    @staticmethod
    def create_string_field(
        metadata: SettingMetadata,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> QtWidgets.QLineEdit:
        """Create text input field.

        For sensitive fields (API keys, passwords), this will use password mode
        initially. The parent SettingRowWidget should add a show/hide toggle button.

        Pattern reference: chat_panel.py lines 296-304

        Args:
            metadata: Setting metadata
            parent: Parent widget

        Returns:
            QLineEdit configured for the setting type
        """
        field = QtWidgets.QLineEdit(parent)
        field.setPlaceholderText(f"Enter {metadata.description.lower()}")

        # For sensitive fields, use password mode
        if metadata.is_sensitive:
            field.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)

        # Set max length for reasonable input
        if metadata.value_type == SettingType.EMAIL:
            field.setMaxLength(254)  # RFC 5321 max email length
        elif metadata.value_type == SettingType.URL:
            field.setMaxLength(2048)  # Reasonable URL length
        else:
            field.setMaxLength(500)  # General string limit

        return field

    @staticmethod
    def create_boolean_field(
        metadata: SettingMetadata,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> QtWidgets.QCheckBox:
        """Create checkbox for boolean values.

        The checkbox represents boolean values as:
        - Checked (True): "1", "true", "yes", "on"
        - Unchecked (False): "0", "false", "no", "off"

        Args:
            metadata: Setting metadata
            parent: Parent widget

        Returns:
            QCheckBox configured for boolean input
        """
        field = QtWidgets.QCheckBox(parent)
        field.setText("")  # Label is handled by SettingRowWidget

        # Set tooltip to explain boolean values
        field.setToolTip(
            f"{metadata.description}\n\n"
            "Checked = 1/true, Unchecked = 0/false"
        )

        return field

    @staticmethod
    def create_integer_field(
        metadata: SettingMetadata,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> QtWidgets.QSpinBox:
        """Create spin box for numeric values.

        Args:
            metadata: Setting metadata
            parent: Parent widget

        Returns:
            QSpinBox configured for integer input
        """
        field = QtWidgets.QSpinBox(parent)

        # Set reasonable range
        # Most settings use 0 as "use defaults", so allow 0
        field.setMinimum(0)
        field.setMaximum(999999)  # Reasonable maximum

        # For episode step/second limits, show that 0 means "use defaults"
        if "steps" in metadata.key.lower() or "seconds" in metadata.key.lower():
            field.setSpecialValueText("Use defaults")

        field.setSuffix("")  # Could add " steps" or " seconds" based on key

        return field

    @staticmethod
    def create_enum_field(
        metadata: SettingMetadata,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> QtWidgets.QComboBox:
        """Create dropdown for enum values.

        Pattern reference: control_panel.py lines 744-750

        Args:
            metadata: Setting metadata with enum_options
            parent: Parent widget

        Returns:
            QComboBox configured with enum options
        """
        field = QtWidgets.QComboBox(parent)

        # Add enum options
        if metadata.enum_options:
            field.addItems(metadata.enum_options)

        # Make it searchable for long lists
        field.setEditable(False)  # Not editable, but can type to search
        field.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)

        return field


class SettingFieldHelper:
    """Helper methods for working with setting field widgets."""

    @staticmethod
    def get_value_from_widget(widget: QtWidgets.QWidget) -> str:
        """Get the current value from a setting widget as a string.

        Args:
            widget: Qt widget (QLineEdit, QCheckBox, QSpinBox, or QComboBox)

        Returns:
            String representation of the current value
        """
        if isinstance(widget, QtWidgets.QLineEdit):
            return widget.text()
        elif isinstance(widget, QtWidgets.QCheckBox):
            # Return "1" or "0" for boolean
            return "1" if widget.isChecked() else "0"
        elif isinstance(widget, QtWidgets.QSpinBox):
            return str(widget.value())
        elif isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText()
        else:
            raise TypeError(f"Unsupported widget type: {type(widget)}")

    @staticmethod
    def set_value_in_widget(widget: QtWidgets.QWidget, value: str) -> None:
        """Set the value in a setting widget from a string.

        Args:
            widget: Qt widget (QLineEdit, QCheckBox, QSpinBox, or QComboBox)
            value: String value to set
        """
        if isinstance(widget, QtWidgets.QLineEdit):
            widget.setText(value)
        elif isinstance(widget, QtWidgets.QCheckBox):
            # Parse boolean string
            checked = value.lower() in ("1", "true", "yes", "on")
            widget.setChecked(checked)
        elif isinstance(widget, QtWidgets.QSpinBox):
            try:
                widget.setValue(int(value))
            except ValueError:
                widget.setValue(0)
        elif isinstance(widget, QtWidgets.QComboBox):
            # Find and select the matching item
            index = widget.findText(value)
            if index >= 0:
                widget.setCurrentIndex(index)
        else:
            raise TypeError(f"Unsupported widget type: {type(widget)}")


__all__ = ["SettingFieldFactory", "SettingFieldHelper"]
