"""Main settings dialog for MOSAIC application.

This module provides the main settings dialog that allows users to configure
all environment variables from the .env file through a user-friendly GUI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from qtpy import QtCore, QtWidgets

from gym_gui.services.settings_service import SettingsService
from gym_gui.ui.widgets.settings.settings_category_tab import SettingsCategoryTab


_LOGGER = logging.getLogger(__name__)


class SettingsDialog(QtWidgets.QDialog):
    """Main settings dialog for MOSAIC application.

    Pattern reference: LoadPolicyDialog for dialog structure

    Features:
    - Non-modal (setModal(False)) - user can interact with main window
    - Tabbed interface (QTabWidget) - one tab per category
    - Global search/filter bar
    - Real-time .env persistence on each change
    - Validation feedback with inline errors
    - Reset to defaults functionality

    Signals:
        setting_changed: Emitted when any setting changes (key, new_value)
        settings_reset: Emitted when all settings are reset to defaults
    """

    setting_changed = QtCore.Signal(str, str)  # (key, new_value)
    settings_reset = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initialize the settings dialog.

        Args:
            parent: Parent widget (typically MainWindow)
        """
        super().__init__(parent)

        # Initialize service with .env path
        env_path = Path.cwd() / ".env"
        self._service = SettingsService(env_path)
        self._category_tabs: Dict[str, SettingsCategoryTab] = {}

        # UI components
        self._tab_widget: Optional[QtWidgets.QTabWidget] = None
        self._search_input: Optional[QtWidgets.QLineEdit] = None
        self._status_label: Optional[QtWidgets.QLabel] = None
        self._reset_btn: Optional[QtWidgets.QPushButton] = None

        # Configure dialog
        self.setWindowTitle("MOSAIC Settings")
        self.setModal(False)  # CRITICAL: Non-modal dialog
        self.resize(900, 700)

        # Build UI
        self._build_ui()
        self._connect_signals()
        self._load_current_values()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Header: [Title] [Search: ________]
        header = self._create_header()
        layout.addWidget(header)

        # Tab widget for 8 categories
        self._tab_widget = QtWidgets.QTabWidget(self)
        for category in self._service.get_categories():
            tab = SettingsCategoryTab(category, self._service, self)
            self._category_tabs[category] = tab
            self._tab_widget.addTab(tab, category)
        layout.addWidget(self._tab_widget, 1)  # Stretch factor 1

        # Status bar (green for success, red for errors)
        self._status_label = QtWidgets.QLabel("")
        self._status_label.setStyleSheet("color: gray; font-size: 11px;")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Button bar: [Reset All] ... [Close]
        button_layout = self._create_button_bar()
        layout.addLayout(button_layout)

    def _create_header(self) -> QtWidgets.QWidget:
        """Create header with title and search bar.

        Returns:
            Header widget
        """
        header = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)

        # Title
        title = QtWidgets.QLabel("Application Settings", header)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Search bar
        search_label = QtWidgets.QLabel("Search:", header)
        self._search_input = QtWidgets.QLineEdit(header)
        self._search_input.setPlaceholderText("Filter settings by name or description...")
        self._search_input.setClearButtonEnabled(True)
        layout.addWidget(search_label)
        layout.addWidget(self._search_input, 1)

        return header

    def _create_button_bar(self) -> QtWidgets.QHBoxLayout:
        """Create button bar with Reset and Close buttons.

        Returns:
            Button bar layout
        """
        layout = QtWidgets.QHBoxLayout()

        self._reset_btn = QtWidgets.QPushButton("Reset All to Defaults", self)
        self._reset_btn.setMaximumWidth(180)
        self._reset_btn.setToolTip(
            "Reset all settings to their default values from .env.example"
        )
        layout.addWidget(self._reset_btn)

        layout.addStretch(1)

        close_btn = QtWidgets.QPushButton("Close", self)
        close_btn.setMaximumWidth(80)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        return layout

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        if self._search_input is not None:
            self._search_input.textChanged.connect(self._on_search_changed)

        if self._reset_btn is not None:
            self._reset_btn.clicked.connect(self._on_reset_all)

        # Connect each category tab's setting_changed signal
        for tab in self._category_tabs.values():
            tab.setting_changed.connect(self._on_setting_changed)

    def _on_search_changed(self, query: str) -> None:
        """Filter settings based on search query.

        Args:
            query: Search query (case-insensitive)
        """
        normalized_query = query.lower().strip()

        # Apply filter to all tabs
        total_visible = 0
        for tab in self._category_tabs.values():
            tab.apply_search_filter(normalized_query)
            total_visible += tab.get_visible_count()

        # Update status with match count
        if normalized_query:
            if total_visible == 0:
                self._set_status_message("No settings match your search", "warning")
            else:
                self._set_status_message(
                    f"Found {total_visible} setting{'s' if total_visible != 1 else ''} matching '{query}'",
                    "info"
                )
        else:
            self._set_status_message("", "info")

    def _on_setting_changed(self, key: str, value: str) -> None:
        """Handle setting change from any tab.

        Args:
            key: Setting key
            value: New value
        """
        metadata = self._service._settings_metadata.get(key)
        if not metadata:
            _LOGGER.warning(f"Unknown setting key: {key}")
            return

        # Validate value
        is_valid, error_msg = self._service.validate_value(metadata, value)
        if not is_valid:
            self._show_validation_error(key, error_msg)
            return

        # Save to .env file using dotenv.set_key()
        success = self._service.set_value(key, value)
        if success:
            # Show success feedback
            display_value = "***" if metadata.is_sensitive else value
            if not display_value:
                display_value = "(empty)"

            self._set_status_message(
                f"✓ Saved: {key} = {display_value}",
                "success"
            )

            # Show restart warning if needed
            if metadata.requires_restart:
                self._show_restart_warning(key)

            # Emit signal for potential app-wide updates
            self.setting_changed.emit(key, value)
        else:
            self._show_save_error(key)

    def _on_reset_all(self) -> None:
        """Reset all settings to defaults with confirmation."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Reset All Settings",
            "Are you sure you want to reset all settings to their default values?\n\n"
            "This will overwrite your current .env file with defaults from .env.example.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            # Reset all settings to defaults
            for metadata in self._service._settings_metadata.values():
                self._service.set_value(metadata.key, metadata.default_value)

            # Reload UI to show new values
            self._load_current_values()
            self._set_status_message(
                "⚠ All settings reset to defaults (restart required)",
                "warning"
            )
            self.settings_reset.emit()

    def _load_current_values(self) -> None:
        """Load current values from .env into UI widgets."""
        for category_tab in self._category_tabs.values():
            for row_widget in category_tab._row_widgets.values():
                current_value = self._service.get_value(row_widget._metadata.key)
                if current_value is not None:
                    row_widget.set_value(current_value)
                else:
                    # Use default if not set in .env
                    row_widget.set_value(row_widget._metadata.default_value)

    def _show_validation_error(self, key: str, message: str) -> None:
        """Show validation error message.

        Args:
            key: Setting key
            message: Error message
        """
        self._set_status_message(
            f"✗ Validation error for {key}: {message}",
            "error"
        )

    def _show_save_error(self, key: str) -> None:
        """Show save error message.

        Args:
            key: Setting key
        """
        self._set_status_message(
            f"✗ Failed to save {key} to .env file",
            "error"
        )

    def _show_restart_warning(self, key: str) -> None:
        """Show warning that restart is required.

        Args:
            key: Setting key that requires restart
        """
        QtWidgets.QMessageBox.information(
            self,
            "Restart Required",
            f"The setting '{key}' requires an application restart to take effect.\n\n"
            "Please restart MOSAIC for this change to be applied.",
            QtWidgets.QMessageBox.StandardButton.Ok,
        )

    def _set_status_message(self, message: str, level: str = "info") -> None:
        """Set status message with appropriate styling.

        Args:
            message: Status message to display
            level: Message level ("info", "success", "warning", "error")
        """
        if self._status_label is None:
            return

        color_map = {
            "info": "gray",
            "success": "green",
            "warning": "orange",
            "error": "red",
        }

        color = color_map.get(level, "gray")
        self._status_label.setText(message)
        self._status_label.setStyleSheet(f"color: {color}; font-size: 11px;")


__all__ = ["SettingsDialog"]
