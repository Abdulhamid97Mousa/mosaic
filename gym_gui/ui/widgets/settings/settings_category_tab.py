"""Tab widget for a single settings category.

This module provides a tab widget that displays all settings for a given category
in a scrollable list. It supports search filtering to show/hide settings based on
search queries.
"""

from __future__ import annotations

from typing import Dict, Optional

from qtpy import QtCore, QtWidgets

from gym_gui.services.settings_service import SettingsService
from gym_gui.ui.widgets.settings.setting_row_widget import SettingRowWidget


class SettingsCategoryTab(QtWidgets.QWidget):
    """Tab widget for a single settings category.

    Contains a scrollable list of SettingRowWidget instances for all settings
    in the category. Supports search filtering to show/hide individual settings.

    Pattern reference: Similar structure to control_panel.py tabs

    Signals:
        setting_changed: Emitted when any setting in this category changes (key, new_value)
    """

    setting_changed = QtCore.Signal(str, str)  # (key, new_value)

    def __init__(
        self,
        category: str,
        service: SettingsService,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize the category tab.

        Args:
            category: Category name
            service: Settings service
            parent: Parent widget
        """
        super().__init__(parent)
        self._category = category
        self._service = service
        self._row_widgets: Dict[str, SettingRowWidget] = {}  # key -> widget

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the category tab UI."""
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scroll area for settings
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        main_layout.addWidget(scroll_area)

        # Container for settings
        container = QtWidgets.QWidget()
        scroll_area.setWidget(container)

        # Layout for settings rows
        settings_layout = QtWidgets.QVBoxLayout(container)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(8)

        # Get all settings for this category
        settings = self._service.get_settings_by_category(self._category)

        if not settings:
            # No settings in this category
            no_settings_label = QtWidgets.QLabel("No settings in this category", container)
            no_settings_label.setStyleSheet("color: gray; font-style: italic;")
            settings_layout.addWidget(no_settings_label)
        else:
            # Create row widget for each setting
            for metadata in settings:
                row_widget = SettingRowWidget(metadata, self._service, container)
                row_widget.value_changed.connect(self._on_row_value_changed)
                self._row_widgets[metadata.key] = row_widget
                settings_layout.addWidget(row_widget)

        # Add stretch to push settings to the top
        settings_layout.addStretch(1)

    def _on_row_value_changed(self, key: str, value: str) -> None:
        """Handle value change from a row widget.

        Args:
            key: Setting key
            value: New value
        """
        # Forward the signal to parent
        self.setting_changed.emit(key, value)

    def apply_search_filter(self, query: str) -> None:
        """Show/hide rows based on search query.

        Searches in setting key, description, and category. Case-insensitive.

        Args:
            query: Search query (empty string shows all)
        """
        if not query:
            # Empty query - show all rows
            for row in self._row_widgets.values():
                row.setVisible(True)
            return

        query_lower = query.lower()

        # Check each row
        for key, row in self._row_widgets.items():
            metadata = row._metadata

            # Search in key, description, and category
            matches = (
                query_lower in key.lower()
                or query_lower in metadata.description.lower()
                or query_lower in metadata.category.lower()
            )

            row.setVisible(matches)

    def get_visible_count(self) -> int:
        """Get number of currently visible settings.

        Returns:
            Count of visible setting rows
        """
        return sum(1 for row in self._row_widgets.values() if row.isVisible())


__all__ = ["SettingsCategoryTab"]
