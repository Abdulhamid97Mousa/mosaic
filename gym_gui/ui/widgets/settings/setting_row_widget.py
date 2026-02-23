"""Widget for a single setting row in the settings dialog.

This module provides a widget that displays a single setting with its label,
input field, and optional show/hide button for sensitive fields.
"""

from __future__ import annotations

from typing import Optional

from qtpy import QtCore, QtWidgets

from gym_gui.services.settings_service import SettingMetadata, SettingsService
from gym_gui.ui.widgets.settings.setting_field_factory import (
    SettingFieldFactory,
    SettingFieldHelper,
)


class SettingRowWidget(QtWidgets.QWidget):
    """Widget for a single setting row: [Label] [Input] [Show/Hide Button].

    Layout: HBoxLayout with:
    - Label (30%): Setting name
    - Input widget (60%): Text field, checkbox, spinner, or dropdown
    - Show/Hide button (10%): For sensitive fields only

    Pattern reference: Similar to individual setting rows in ChatPanel

    Signals:
        value_changed: Emitted when the setting value changes (key, new_value)
    """

    value_changed = QtCore.Signal(str, str)  # (key, new_value)

    def __init__(
        self,
        metadata: SettingMetadata,
        service: SettingsService,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize the setting row widget.

        Args:
            metadata: Setting metadata
            service: Settings service for validation and persistence
            parent: Parent widget
        """
        super().__init__(parent)
        self._metadata = metadata
        self._service = service
        self._input_widget: Optional[QtWidgets.QWidget] = None
        self._show_hide_btn: Optional[QtWidgets.QPushButton] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the setting row UI."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Label (30%)
        label = QtWidgets.QLabel(self._format_label_text(self._metadata.key), self)
        label.setToolTip(self._metadata.description)
        label.setMinimumWidth(200)
        label.setMaximumWidth(300)
        label.setWordWrap(True)
        layout.addWidget(label, 3)  # 30% stretch

        # Input widget (60%)
        self._input_widget = SettingFieldFactory.create_field(self._metadata, self)
        self._input_widget.setToolTip(self._metadata.description)
        layout.addWidget(self._input_widget, 6)  # 60% stretch

        # Show/Hide button for sensitive fields (10%)
        if self._metadata.is_sensitive:
            self._show_hide_btn = self._create_show_hide_button()
            layout.addWidget(self._show_hide_btn, 1)  # 10% stretch
        else:
            # Add spacer to maintain layout consistency
            spacer = QtWidgets.QWidget(self)
            spacer.setFixedWidth(60)
            layout.addWidget(spacer, 1)

        # Connect signals
        self._connect_signals()

    def _format_label_text(self, key: str) -> str:
        """Format environment variable key as readable label.

        Converts "SOME_VAR_NAME" to "Some Var Name"

        Args:
            key: Environment variable name

        Returns:
            Formatted label text
        """
        # Replace underscores with spaces and title case
        words = key.replace("_", " ").split()
        # Special handling for acronyms (keep uppercase if 2-3 chars)
        formatted_words = []
        for word in words:
            if len(word) <= 3 and word.isupper():
                # Keep acronyms uppercase (QT, API, URL, etc.)
                formatted_words.append(word)
            else:
                formatted_words.append(word.capitalize())

        return " ".join(formatted_words)

    def _create_show_hide_button(self) -> QtWidgets.QPushButton:
        """Create show/hide toggle button for sensitive fields.

        Pattern reference: chat_panel.py lines 305-309

        Returns:
            QPushButton configured to toggle password visibility
        """
        btn = QtWidgets.QPushButton("Show", self)
        btn.setMaximumWidth(60)
        btn.setCheckable(True)
        btn.setToolTip("Toggle visibility of sensitive value")
        btn.clicked.connect(self._on_show_hide_clicked)
        return btn

    def _connect_signals(self) -> None:
        """Connect widget signals to emit value_changed."""
        if isinstance(self._input_widget, QtWidgets.QLineEdit):
            self._input_widget.textChanged.connect(self._on_value_changed)
        elif isinstance(self._input_widget, QtWidgets.QCheckBox):
            self._input_widget.stateChanged.connect(self._on_value_changed)
        elif isinstance(self._input_widget, QtWidgets.QSpinBox):
            self._input_widget.valueChanged.connect(self._on_value_changed)
        elif isinstance(self._input_widget, QtWidgets.QComboBox):
            self._input_widget.currentTextChanged.connect(self._on_value_changed)

    def _on_value_changed(self) -> None:
        """Handle value change in input widget."""
        if self._input_widget is None:
            return

        # Get current value from widget
        value = SettingFieldHelper.get_value_from_widget(self._input_widget)

        # Emit signal (parent will handle validation and saving)
        self.value_changed.emit(self._metadata.key, value)

    def _on_show_hide_clicked(self, checked: bool) -> None:
        """Toggle password visibility for sensitive fields.

        Args:
            checked: True if "Hide" mode (showing text), False if "Show" mode (hiding text)
        """
        if not isinstance(self._input_widget, QtWidgets.QLineEdit):
            return

        if self._show_hide_btn is None:
            return

        if checked:
            # Show the text
            self._input_widget.setEchoMode(QtWidgets.QLineEdit.EchoMode.Normal)
            self._show_hide_btn.setText("Hide")
        else:
            # Hide the text
            self._input_widget.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
            self._show_hide_btn.setText("Show")

    def get_value(self) -> str:
        """Get current value from input widget.

        Returns:
            Current value as string
        """
        if self._input_widget is None:
            return ""
        return SettingFieldHelper.get_value_from_widget(self._input_widget)

    def set_value(self, value: str) -> None:
        """Set value in input widget.

        Args:
            value: Value to set
        """
        if self._input_widget is None:
            return
        SettingFieldHelper.set_value_in_widget(self._input_widget, value)

    def reset_to_default(self) -> None:
        """Reset the setting to its default value."""
        self.set_value(self._metadata.default_value)


__all__ = ["SettingRowWidget"]
