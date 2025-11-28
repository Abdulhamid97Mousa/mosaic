"""MuJoCo MPC launch and management handlers using composition pattern.

This module provides a handler class that manages MuJoCo MPC instances,
including launching, creating tabs, and stopping instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from qtpy import QtCore, QtWidgets

if TYPE_CHECKING:
    from qtpy.QtWidgets import QStatusBar
    from gym_gui.ui.widgets.control_panel import ControlPanelWidget
    from gym_gui.ui.widgets.render_tabs import RenderTabs


class MPCHandler:
    """Handles MuJoCo MPC instance management.

    This class encapsulates all MPC-related logic including launching instances,
    creating tabs for display, and stopping instances.

    Args:
        mjpc_launcher: The MJPC launcher instance.
        render_tabs: The render tabs widget for adding/removing MPC tabs.
        control_panel: The control panel widget for updating instance counts.
        status_bar: The status bar for showing feedback messages.
    """

    def __init__(
        self,
        mjpc_launcher: Any,
        render_tabs: "RenderTabs",
        control_panel: "ControlPanelWidget",
        status_bar: "QStatusBar",
    ) -> None:
        self._mjpc_launcher = mjpc_launcher
        self._render_tabs = render_tabs
        self._control_panel = control_panel
        self._status_bar = status_bar
        self._mpc_tabs: Dict[int, QtWidgets.QWidget] = {}

    @property
    def mpc_tabs(self) -> Dict[int, QtWidgets.QWidget]:
        """Return the dictionary of MPC tabs."""
        return self._mpc_tabs

    def on_launch_requested(self, display_mode: str) -> None:
        """Handle launch request for MuJoCo MPC - launches MJPC and creates tab.

        Args:
            display_mode: Either "external" (separate window) or "embedded" (in Render View)
        """
        # Check if MJPC is built
        if not self._mjpc_launcher.is_built():
            build_status = self._mjpc_launcher.get_build_status()
            QtWidgets.QMessageBox.warning(
                None,
                "MJPC Not Built",
                "MuJoCo MPC needs to be built first.\n\n"
                "Run the following commands:\n\n"
                "  cd 3rd_party/mujoco_mpc_worker/mujoco_mpc/build\n"
                "  cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja\n"
                "  ninja -j$(nproc)\n\n"
                f"Source dir: {build_status['source_dir']}\n"
                f"Source exists: {build_status['source_exists']}",
            )
            return

        # Launch MJPC
        process, message = self._mjpc_launcher.launch()
        if process is None:
            QtWidgets.QMessageBox.critical(
                None,
                "Launch Failed",
                f"Failed to launch MJPC:\n{message}",
            )
            return

        instance_id = process.instance_id
        tab_name = f"MuJoCo-MPC-{instance_id}"

        if display_mode == "embedded":
            self._create_embedded_tab(process, instance_id, tab_name)
        else:
            self._create_external_tab(process, instance_id, tab_name)

        # Update sidebar instance count
        mpc_tab = self._control_panel.get_mujoco_mpc_tab()
        mpc_tab.update_instance_count(len(self._mpc_tabs))

        self._status_bar.showMessage(
            f"Launched {tab_name} (PID: {process.process.pid})", 3000
        )

    def _create_external_tab(
        self, process: Any, instance_id: int, tab_name: str
    ) -> None:
        """Create a tab showing status for external MJPC window."""
        tab_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab_widget)

        # Status display
        status_label = QtWidgets.QLabel(
            f"<h2>{tab_name}</h2>"
            f"<p><b>Status:</b> Running</p>"
            f"<p><b>PID:</b> {process.process.pid}</p>"
            "<p><i>MJPC GUI is running in a separate window.</i></p>",
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
            run_id=f"mpc-{instance_id}",
            name=tab_name,
            widget=tab_widget,
        )
        self._mpc_tabs[instance_id] = tab_widget

    def _create_embedded_tab(
        self, process: Any, instance_id: int, tab_name: str
    ) -> None:
        """Create a tab for embedded MJPC - currently shows coming soon message.

        Future implementation will use agent_server gRPC + MuJoCo Python rendering.
        """
        tab_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab_widget)

        # Coming soon message
        info_label = QtWidgets.QLabel(
            f"<h2>{tab_name}</h2>"
            f"<p><b>Status:</b> Running (PID: {process.process.pid})</p>"
            "<br/>"
            "<p style='color: #666;'><b>Embedded mode coming soon!</b></p>"
            "<p style='color: #888;'>Will use agent_server gRPC + MuJoCo Python rendering.</p>"
            "<p style='color: #888;'>For now, MJPC is running in a separate window.</p>",
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
            run_id=f"mpc-{instance_id}",
            name=tab_name,
            widget=tab_widget,
        )
        self._mpc_tabs[instance_id] = tab_widget

    def on_stop_instance(self, instance_id: int) -> None:
        """Stop a specific MJPC instance."""
        tab_name = f"MuJoCo-MPC-{instance_id}"

        # Terminate the process
        self._mjpc_launcher.terminate(instance_id)

        # Remove tab
        widget = self._mpc_tabs.pop(instance_id, None)
        if widget:
            idx = self._render_tabs.indexOf(widget)
            if idx >= 0:
                self._render_tabs.removeTab(idx)
            run_id = f"mpc-{instance_id}"
            if run_id in self._render_tabs._agent_tabs:
                del self._render_tabs._agent_tabs[run_id]

        # Update sidebar instance count
        mpc_tab = self._control_panel.get_mujoco_mpc_tab()
        mpc_tab.update_instance_count(len(self._mpc_tabs))

        self._status_bar.showMessage(f"Stopped {tab_name}", 3000)

    def on_stop_all_requested(self) -> None:
        """Handle stop all request for MuJoCo MPC - closes all MPC tabs."""
        # Terminate all processes
        count = self._mjpc_launcher.terminate_all()

        # Remove all tabs
        for instance_id, widget in list(self._mpc_tabs.items()):
            idx = self._render_tabs.indexOf(widget)
            if idx >= 0:
                self._render_tabs.removeTab(idx)
            run_id = f"mpc-{instance_id}"
            if run_id in self._render_tabs._agent_tabs:
                del self._render_tabs._agent_tabs[run_id]

        self._mpc_tabs.clear()

        # Update sidebar instance count
        mpc_tab = self._control_panel.get_mujoco_mpc_tab()
        mpc_tab.update_instance_count(0)

        self._status_bar.showMessage(f"Stopped {count} MuJoCo MPC instance(s)", 3000)


__all__ = ["MPCHandler"]
