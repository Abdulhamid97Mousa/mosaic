"""Telemetry/render view container widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtWidgets

from gym_gui.ui.widgets.render_tabs import RenderTabs

if TYPE_CHECKING:
    from gym_gui.services.telemetry import TelemetryService


class TelemetryContainer(QtWidgets.QWidget):
    """Container for the render/telemetry view with tabs."""

    def __init__(
        self,
        telemetry_service: TelemetryService,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._telemetry_service = telemetry_service

        # Create group box
        self._group = QtWidgets.QGroupBox("Render View", self)
        layout = QtWidgets.QVBoxLayout(self._group)

        # Create render tabs
        self._render_tabs = RenderTabs(
            self._group,
            telemetry_service=self._telemetry_service,
        )
        layout.addWidget(self._render_tabs)

        # Set up container layout
        container_layout = QtWidgets.QVBoxLayout(self)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self._group)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def get_render_tabs(self) -> RenderTabs:
        """Get the underlying render tabs widget."""
        return self._render_tabs

    def get_group_box(self) -> QtWidgets.QGroupBox:
        """Get the group box widget."""
        return self._group

    def display_payload(self, payload):
        """Display a render payload."""
        self._render_tabs.display_payload(payload)

    def refresh_replays(self):
        """Refresh replay tabs."""
        self._render_tabs.refresh_replays()


__all__ = ["TelemetryContainer"]
