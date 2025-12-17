"""Godot launch and management handlers using composition pattern.

This module provides a handler class that manages Godot game engine instances,
including launching, creating tabs, and stopping instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from qtpy import QtCore, QtWidgets

from gym_gui.logging_config.log_constants import (
    GODOT_BINARY_NOT_FOUND_MSG,
    GODOT_BINARY_NOT_FOUND_TITLE,
    GODOT_NOT_INSTALLED_MSG,
    GODOT_NOT_INSTALLED_TITLE,
)

if TYPE_CHECKING:
    from qtpy.QtWidgets import QStatusBar
    from gym_gui.ui.widgets.control_panel import ControlPanelWidget
    from gym_gui.ui.widgets.render_tabs import RenderTabs


class GodotHandler:
    """Handles Godot game engine instance management.

    This class encapsulates all Godot-related logic including launching instances,
    creating tabs for display, and stopping instances.

    Args:
        godot_launcher: The Godot launcher instance.
        render_tabs: The render tabs widget for adding/removing Godot tabs.
        control_panel: The control panel widget for updating instance counts.
        status_bar: The status bar for showing feedback messages.
    """

    def __init__(
        self,
        godot_launcher: Any,
        render_tabs: "RenderTabs",
        control_panel: "ControlPanelWidget",
        status_bar: "QStatusBar",
    ) -> None:
        self._godot_launcher = godot_launcher
        self._render_tabs = render_tabs
        self._control_panel = control_panel
        self._status_bar = status_bar
        self._godot_tabs: Dict[int, QtWidgets.QWidget] = {}

    @property
    def godot_tabs(self) -> Dict[int, QtWidgets.QWidget]:
        """Return the dictionary of Godot tabs."""
        return self._godot_tabs

    def on_launch_requested(self, display_mode: str) -> None:
        """Handle launch request for Godot - launches Godot and creates tab.

        Args:
            display_mode: Either "external" (separate window) or "embedded" (in Render View)
        """
        # Check if Godot launcher is available
        if self._godot_launcher is None:
            QtWidgets.QMessageBox.warning(
                None,
                GODOT_NOT_INSTALLED_TITLE,
                GODOT_NOT_INSTALLED_MSG,
            )
            return

        # Check if Godot is available
        if not self._godot_launcher.is_available():
            status = self._godot_launcher.get_status()
            QtWidgets.QMessageBox.warning(
                None,
                GODOT_BINARY_NOT_FOUND_TITLE,
                GODOT_BINARY_NOT_FOUND_MSG.format(godot_binary=status["godot_binary"]),
            )
            return

        # Launch Godot
        process, message = self._godot_launcher.launch()
        if process is None:
            QtWidgets.QMessageBox.critical(
                None,
                "Launch Failed",
                f"Failed to launch Godot:\n{message}",
            )
            return

        instance_id = process.instance_id
        tab_name = f"Godot-{instance_id}"

        if display_mode == "embedded":
            self._create_embedded_tab(process, instance_id, tab_name)
        else:
            self._create_external_tab(process, instance_id, tab_name)

        # Update sidebar instance count
        godot_tab = self._control_panel.get_godot_ue_tab()
        godot_tab.update_instance_count(len(self._godot_tabs))

        self._status_bar.showMessage(
            f"Launched {tab_name} (PID: {process.process.pid})", 3000
        )

    def on_editor_requested(self) -> None:
        """Handle launch editor request for Godot."""
        # Check if Godot is available
        if not self._godot_launcher.is_available():
            status = self._godot_launcher.get_status()
            QtWidgets.QMessageBox.warning(
                None,
                "Godot Not Available",
                "Godot binary not found.\n\n"
                f"Expected at: {status['godot_binary']}",
            )
            return

        # Launch Godot editor
        process, message = self._godot_launcher.launch_editor()
        if process is None:
            QtWidgets.QMessageBox.critical(
                None,
                "Launch Failed",
                f"Failed to launch Godot editor:\n{message}",
            )
            return

        instance_id = process.instance_id
        tab_name = f"Godot-Editor-{instance_id}"

        self._create_editor_tab(process, instance_id, tab_name)

        # Update sidebar instance count
        godot_tab = self._control_panel.get_godot_ue_tab()
        godot_tab.update_instance_count(len(self._godot_tabs))

        self._status_bar.showMessage(
            f"Launched {tab_name} (PID: {process.process.pid})", 3000
        )

    def _create_external_tab(
        self, process: Any, instance_id: int, tab_name: str
    ) -> None:
        """Create a tab showing status for external Godot window."""
        tab_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab_widget)

        # Status display
        status_label = QtWidgets.QLabel(
            f"<h2>{tab_name}</h2>"
            f"<p><b>Status:</b> Running</p>"
            f"<p><b>PID:</b> {process.process.pid}</p>"
            "<p><i>Godot is running in a separate window.</i></p>",
            tab_widget,
        )
        status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_label)

        # Close button for this instance
        close_btn = QtWidgets.QPushButton(f"Stop {tab_name}", tab_widget)
        close_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "font-weight: bold; padding: 10px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )
        close_btn.clicked.connect(lambda: self.on_stop_instance(instance_id))
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()

        # Add tab to render view
        self._render_tabs.add_dynamic_tab(
            run_id=f"godot-{instance_id}",
            name=tab_name,
            widget=tab_widget,
        )
        self._godot_tabs[instance_id] = tab_widget

    def _create_embedded_tab(
        self, process: Any, instance_id: int, tab_name: str
    ) -> None:
        """Create a tab for embedded Godot - currently shows coming soon message.

        Future implementation will embed Godot rendering via shared memory or TCP.
        """
        tab_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab_widget)

        # Coming soon message
        info_label = QtWidgets.QLabel(
            f"<h2>{tab_name}</h2>"
            f"<p><b>Status:</b> Running (PID: {process.process.pid})</p>"
            "<br/>"
            "<p style='color: #666;'><b>Embedded mode coming soon!</b></p>"
            "<p style='color: #888;'>Will use TCP/WebSocket for RL communication.</p>"
            "<p style='color: #888;'>For now, Godot is running in a separate window.</p>",
            tab_widget,
        )
        info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

        layout.addStretch()

        # Close button
        close_btn = QtWidgets.QPushButton(f"Stop {tab_name}", tab_widget)
        close_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "font-weight: bold; padding: 10px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )
        close_btn.clicked.connect(lambda: self.on_stop_instance(instance_id))
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()

        # Add tab to render view
        self._render_tabs.add_dynamic_tab(
            run_id=f"godot-{instance_id}",
            name=tab_name,
            widget=tab_widget,
        )
        self._godot_tabs[instance_id] = tab_widget

    def _create_editor_tab(
        self, process: Any, instance_id: int, tab_name: str
    ) -> None:
        """Create a tab showing status for Godot editor."""
        tab_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab_widget)

        # Status display
        status_label = QtWidgets.QLabel(
            f"<h2>{tab_name}</h2>"
            f"<p><b>Status:</b> Running</p>"
            f"<p><b>PID:</b> {process.process.pid}</p>"
            "<p><i>Godot Editor is running in a separate window.</i></p>"
            "<p style='color: #666;'>Use the editor to create/modify Godot projects.</p>",
            tab_widget,
        )
        status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_label)

        # Close button for this instance
        close_btn = QtWidgets.QPushButton(f"Stop {tab_name}", tab_widget)
        close_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "font-weight: bold; padding: 10px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )
        close_btn.clicked.connect(lambda: self.on_stop_instance(instance_id))
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()

        # Add tab to render view
        self._render_tabs.add_dynamic_tab(
            run_id=f"godot-{instance_id}",
            name=tab_name,
            widget=tab_widget,
        )
        self._godot_tabs[instance_id] = tab_widget

    def on_stop_instance(self, instance_id: int) -> None:
        """Stop a specific Godot instance."""
        tab_name = f"Godot-{instance_id}"

        # Terminate the process
        self._godot_launcher.terminate(instance_id)

        # Remove tab
        widget = self._godot_tabs.pop(instance_id, None)
        if widget:
            idx = self._render_tabs.indexOf(widget)
            if idx >= 0:
                self._render_tabs.removeTab(idx)
            run_id = f"godot-{instance_id}"
            if run_id in self._render_tabs._agent_tabs:
                del self._render_tabs._agent_tabs[run_id]

        # Update sidebar instance count
        godot_tab = self._control_panel.get_godot_ue_tab()
        godot_tab.update_instance_count(len(self._godot_tabs))

        self._status_bar.showMessage(f"Stopped {tab_name}", 3000)

    def on_stop_all_requested(self) -> None:
        """Handle stop all request for Godot - closes all Godot tabs."""
        # Terminate all processes
        count = self._godot_launcher.terminate_all()

        # Remove all tabs
        for instance_id, widget in list(self._godot_tabs.items()):
            idx = self._render_tabs.indexOf(widget)
            if idx >= 0:
                self._render_tabs.removeTab(idx)
            run_id = f"godot-{instance_id}"
            if run_id in self._render_tabs._agent_tabs:
                del self._render_tabs._agent_tabs[run_id]

        self._godot_tabs.clear()

        # Update sidebar instance count
        godot_tab = self._control_panel.get_godot_ue_tab()
        godot_tab.update_instance_count(0)

        self._status_bar.showMessage(f"Stopped {count} Godot instance(s)", 3000)


__all__ = ["GodotHandler"]
