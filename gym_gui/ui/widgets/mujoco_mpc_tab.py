"""MuJoCo MPC Tab Widget for the control panel sidebar.

This module provides a simple launcher tab for MuJoCo MPC sessions.
MuJoCo MPC (MJPC) has its own GUI with task/planner selection built-in,
so this tab only provides launch/stop controls.

When launched, a new tab appears in the Render View (MuJoCo-MPC-1, MuJoCo-MPC-2, etc.)

Display Modes:
- External Window: MJPC opens as a separate popup window (default)
- Embedded: MJPC window is embedded inside the Gym GUI Render View tab
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal


class MJPCDisplayMode(Enum):
    """Display mode for MJPC GUI."""
    EXTERNAL = "external"  # Separate popup window
    EMBEDDED = "embedded"  # Embedded in Render View tab


class MuJoCoMPCTab(QtWidgets.QWidget):
    """Simple launcher tab for MuJoCo MPC.

    MJPC has its own GUI with task/planner selection, so this tab only:
    - Shows info about MuJoCo MPC
    - Provides Launch/Stop buttons
    - Allows choosing display mode (external window vs embedded)
    - Displays running instance count
    """

    # Signals - now include display mode
    launch_mpc_requested = pyqtSignal(str)  # Request with display mode ("external" or "embedded")
    stop_all_requested = pyqtSignal()  # Request to stop all MJPC instances

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._instance_count: int = 0
        self._display_mode: MJPCDisplayMode = MJPCDisplayMode.EXTERNAL
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        """Build the tab UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Info banner
        layout.addWidget(self._create_info_banner())

        # Launch Button Group
        layout.addWidget(self._create_launch_group())

        # Status Group
        layout.addWidget(self._create_status_group())

        # Add stretch to push everything to top
        layout.addStretch(1)

    def _create_info_banner(self) -> QtWidgets.QWidget:
        """Create an info banner explaining MuJoCo MPC."""
        banner = QtWidgets.QFrame(self)
        banner.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout = QtWidgets.QVBoxLayout(banner)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QtWidgets.QLabel("<b>MuJoCo MPC (MJPC)</b>", banner)
        layout.addWidget(title)

        desc = QtWidgets.QLabel(
            "Real-time Model Predictive Control for MuJoCo.\n\n"
            "Click 'Launch' to open a new MJPC instance. "
            "Each instance appears as a tab in the Render View. "
            "Task and planner selection is done inside the MJPC GUI.",
            banner
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        return banner

    def _create_launch_group(self) -> QtWidgets.QGroupBox:
        """Create the launch button group with display mode selection."""
        group = QtWidgets.QGroupBox("Control", self)
        layout = QtWidgets.QVBoxLayout(group)

        # Display mode selection
        mode_layout = QtWidgets.QHBoxLayout()
        mode_label = QtWidgets.QLabel("Display Mode:", group)
        mode_layout.addWidget(mode_label)

        self._mode_combo = QtWidgets.QComboBox(group)
        self._mode_combo.addItem("External Window", MJPCDisplayMode.EXTERNAL.value)
        self._mode_combo.addItem("Embedded in Gym GUI", MJPCDisplayMode.EMBEDDED.value)
        self._mode_combo.setCurrentIndex(0)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo)
        layout.addLayout(mode_layout)

        # Mode description
        self._mode_desc = QtWidgets.QLabel(
            "Opens MJPC as a separate window.",
            group
        )
        self._mode_desc.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        self._mode_desc.setWordWrap(True)
        layout.addWidget(self._mode_desc)

        layout.addSpacing(8)

        # Launch button
        self._launch_btn = QtWidgets.QPushButton("Launch MJPC", group)
        self._launch_btn.setMinimumHeight(36)
        layout.addWidget(self._launch_btn)

        # Stop all button
        self._stop_all_btn = QtWidgets.QPushButton("Stop All Instances", group)
        self._stop_all_btn.setMinimumHeight(36)
        self._stop_all_btn.setEnabled(False)
        layout.addWidget(self._stop_all_btn)

        return group

    def _on_mode_changed(self, index: int) -> None:
        """Handle display mode combo box change."""
        mode_value = self._mode_combo.currentData()
        if mode_value == MJPCDisplayMode.EXTERNAL.value:
            self._display_mode = MJPCDisplayMode.EXTERNAL
            self._mode_desc.setText("Opens MJPC as a separate window.")
        else:
            self._display_mode = MJPCDisplayMode.EMBEDDED
            self._mode_desc.setText("Embeds MJPC window inside the Render View tab.")

    def _create_status_group(self) -> QtWidgets.QGroupBox:
        """Create the status display group."""
        group = QtWidgets.QGroupBox("Status", self)
        layout = QtWidgets.QFormLayout(group)

        self._instance_label = QtWidgets.QLabel("0", group)
        self._instance_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addRow("Running Instances:", self._instance_label)

        return group

    def _connect_signals(self) -> None:
        """Connect internal widget signals."""
        self._launch_btn.clicked.connect(self._on_launch_clicked)
        self._stop_all_btn.clicked.connect(self.stop_all_requested.emit)

    def _on_launch_clicked(self) -> None:
        """Handle launch button click - emit signal with current display mode."""
        self.launch_mpc_requested.emit(self._display_mode.value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_instance_count(self, count: int) -> None:
        """Update the running instance count display.

        Args:
            count: Number of running MJPC instances.
        """
        self._instance_count = count
        self._instance_label.setText(str(count))
        self._stop_all_btn.setEnabled(count > 0)

    def increment_instance_count(self) -> int:
        """Increment instance count and return new value."""
        self._instance_count += 1
        self.update_instance_count(self._instance_count)
        return self._instance_count

    def decrement_instance_count(self) -> int:
        """Decrement instance count and return new value."""
        self._instance_count = max(0, self._instance_count - 1)
        self.update_instance_count(self._instance_count)
        return self._instance_count

    def get_instance_count(self) -> int:
        """Return current instance count."""
        return self._instance_count

    def get_display_mode(self) -> MJPCDisplayMode:
        """Return current display mode."""
        return self._display_mode
