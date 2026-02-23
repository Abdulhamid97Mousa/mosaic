from __future__ import annotations

"""Utility for overlaying status badges on QTabWidget tabs."""

from dataclasses import dataclass
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from .state import IndicatorSeverity


@dataclass(slots=True)
class TabBadgeState:
    tab_index: int
    severity: IndicatorSeverity
    text: str


class TabBadgeController(QtCore.QObject):
    """Attaches colored badges to a QTabWidget for quick run summaries."""

    _SEVERITY_COLORS = {
        IndicatorSeverity.INFO: "#0b6efd",
        IndicatorSeverity.WARNING: "#fd7e14",
        IndicatorSeverity.ERROR: "#dc3545",
        IndicatorSeverity.CRITICAL: "#6f42c1",
    }

    def __init__(self, tab_widget: QtWidgets.QTabWidget) -> None:
        super().__init__(parent=tab_widget)
        self._tab_widget = tab_widget
        self._badges: dict[int, QtWidgets.QLabel] = {}

    def apply_badge(self, badge_state: TabBadgeState) -> None:
        label = self._badges.get(badge_state.tab_index)
        if label is None:
            label = QtWidgets.QLabel(parent=self._tab_widget)
            self._badges[badge_state.tab_index] = label
        label.setText(badge_state.text)
        label.setStyleSheet(
            "background: {color}; color: white; border-radius: 8px; padding: 2px 6px; font-size: 11px;".format(
                color=self._SEVERITY_COLORS[badge_state.severity]
            )
        )
        self._set_tab_badge(badge_state.tab_index, label)

    def clear_badge(self, tab_index: int) -> None:
        label = self._badges.pop(tab_index, None)
        if label:
            label.deleteLater()
        self._set_tab_badge(tab_index, None)

    def _set_tab_badge(self, tab_index: int, badge: Optional[QtWidgets.QLabel]) -> None:
        tab_bar = self._tab_widget.tabBar()
        if tab_bar is None:
            return
        base_widget = tab_bar.tabButton(tab_index, QtWidgets.QTabBar.ButtonPosition.RightSide)
        if base_widget:
            base_widget.deleteLater()
        if badge:
            tab_bar.setTabButton(
                tab_index,
                QtWidgets.QTabBar.ButtonPosition.RightSide,
                badge,
            )
