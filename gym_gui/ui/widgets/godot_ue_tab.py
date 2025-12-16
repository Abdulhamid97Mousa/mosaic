"""Godot UE Tab Widget for the control panel sidebar.

This module provides a launcher tab for Godot game engine sessions.
Godot serves as our first "Unreal Engine" style 3D environment integration.

When launched, Godot opens either as an external window or embedded.

Display Modes:
- External Window: Godot opens as a separate popup window (default)
- Embedded: Godot window is embedded inside the Gym GUI Render View tab
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal


class GodotDisplayMode(Enum):
    """Display mode for Godot."""
    EXTERNAL = "external"  # Separate popup window
    EMBEDDED = "embedded"  # Embedded in Render View tab


class GodotUETab(QtWidgets.QWidget):
    """Launcher tab for Godot game engine.

    This tab provides:
    - Info about Godot integration
    - Launch/Stop buttons for Godot instances
    - Display mode selection (external window vs embedded)
    - Running instance count
    - Editor mode launch option
    """

    # Signals
    launch_godot_requested = pyqtSignal(str)  # Request with display mode
    launch_editor_requested = pyqtSignal()  # Request to launch Godot editor
    stop_all_requested = pyqtSignal()  # Request to stop all Godot instances

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._instance_count: int = 0
        self._display_mode: GodotDisplayMode = GodotDisplayMode.EXTERNAL
        self._godot_available: bool = False
        self._godot_version: Optional[str] = None
        self._build_ui()
        self._connect_signals()
        self._check_godot_availability()

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
        """Create an info banner explaining Godot integration."""
        banner = QtWidgets.QFrame(self)
        banner.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        banner.setStyleSheet(
            "QFrame { background-color: #f3e8ff; border: 1px solid #c4b5d4; "
            "border-radius: 4px; padding: 8px; }"
        )
        layout = QtWidgets.QVBoxLayout(banner)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QtWidgets.QLabel("<b>Godot Game Engine (UE)</b>", banner)
        layout.addWidget(title)

        desc = QtWidgets.QLabel(
            "Godot 4.x game engine for 3D RL environments.\n\n"
            "Click 'Launch Godot' to open a new instance. "
            "Use 'Launch Editor' to open the Godot editor for project development. "
            "Projects can communicate with the RL framework via TCP/WebSocket.",
            banner
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #555; font-size: 11px;")
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
        self._mode_combo.addItem("External Window", GodotDisplayMode.EXTERNAL.value)
        self._mode_combo.addItem("Embedded in Gym GUI", GodotDisplayMode.EMBEDDED.value)
        self._mode_combo.setCurrentIndex(0)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo)
        layout.addLayout(mode_layout)

        # Mode description
        self._mode_desc = QtWidgets.QLabel(
            "Opens Godot as a separate window.",
            group
        )
        self._mode_desc.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        self._mode_desc.setWordWrap(True)
        layout.addWidget(self._mode_desc)

        layout.addSpacing(8)

        # Launch button
        self._launch_btn = QtWidgets.QPushButton("Launch Godot", group)
        self._launch_btn.setMinimumHeight(40)
        self._launch_btn.setStyleSheet(
            "QPushButton { background-color: #7c3aed; color: white; "
            "font-weight: bold; border-radius: 4px; font-size: 14px; }"
            "QPushButton:hover { background-color: #6d28d9; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        layout.addWidget(self._launch_btn)

        # Launch Editor button
        self._editor_btn = QtWidgets.QPushButton("Launch Editor", group)
        self._editor_btn.setMinimumHeight(40)
        self._editor_btn.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: white; "
            "font-weight: bold; border-radius: 4px; font-size: 14px; }"
            "QPushButton:hover { background-color: #1d4ed8; }"
            "QPushButton:disabled { background-color: #cccccc; color: #888; }"
        )
        layout.addWidget(self._editor_btn)

        # Stop all button
        self._stop_all_btn = QtWidgets.QPushButton("Stop All Instances", group)
        self._stop_all_btn.setMinimumHeight(40)
        self._stop_all_btn.setEnabled(False)
        self._stop_all_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "font-weight: bold; border-radius: 4px; font-size: 14px; }"
            "QPushButton:hover { background-color: #da190b; }"
            "QPushButton:disabled { background-color: #cccccc; color: #888; }"
        )
        layout.addWidget(self._stop_all_btn)

        return group

    def _on_mode_changed(self, index: int) -> None:
        """Handle display mode combo box change."""
        mode_value = self._mode_combo.currentData()
        if mode_value == GodotDisplayMode.EXTERNAL.value:
            self._display_mode = GodotDisplayMode.EXTERNAL
            self._mode_desc.setText("Opens Godot as a separate window.")
        else:
            self._display_mode = GodotDisplayMode.EMBEDDED
            self._mode_desc.setText("Embeds Godot window inside the Render View tab.")

    def _create_status_group(self) -> QtWidgets.QGroupBox:
        """Create the status display group."""
        group = QtWidgets.QGroupBox("Status", self)
        layout = QtWidgets.QFormLayout(group)

        self._availability_label = QtWidgets.QLabel("Checking...", group)
        self._availability_label.setStyleSheet("font-weight: bold;")
        layout.addRow("Availability:", self._availability_label)

        self._version_label = QtWidgets.QLabel("—", group)
        layout.addRow("Version:", self._version_label)

        self._instance_label = QtWidgets.QLabel("0", group)
        self._instance_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addRow("Running Instances:", self._instance_label)

        return group

    def _connect_signals(self) -> None:
        """Connect internal widget signals."""
        self._launch_btn.clicked.connect(self._on_launch_clicked)
        self._editor_btn.clicked.connect(self.launch_editor_requested.emit)
        self._stop_all_btn.clicked.connect(self.stop_all_requested.emit)

    def _on_launch_clicked(self) -> None:
        """Handle launch button click - emit signal with current display mode."""
        self.launch_godot_requested.emit(self._display_mode.value)

    def _check_godot_availability(self) -> None:
        """Check if Godot binary is available."""
        try:
            from godot_worker import get_launcher
            launcher = get_launcher()
            self._godot_available = launcher.is_available()
            self._godot_version = launcher.get_version()
        except ImportError:
            self._godot_available = False
            self._godot_version = None

        self._update_availability_display()

    def _update_availability_display(self) -> None:
        """Update the availability labels based on Godot status."""
        if self._godot_available:
            self._availability_label.setText("Available")
            self._availability_label.setStyleSheet("font-weight: bold; color: #22c55e;")
            self._launch_btn.setEnabled(True)
            self._editor_btn.setEnabled(True)
        else:
            self._availability_label.setText("Not Available")
            self._availability_label.setStyleSheet("font-weight: bold; color: #ef4444;")
            self._launch_btn.setEnabled(False)
            self._editor_btn.setEnabled(False)

        if self._godot_version:
            self._version_label.setText(self._godot_version)
        else:
            self._version_label.setText("—")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_instance_count(self, count: int) -> None:
        """Update the running instance count display.

        Args:
            count: Number of running Godot instances.
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

    def get_display_mode(self) -> GodotDisplayMode:
        """Return current display mode."""
        return self._display_mode

    def refresh_availability(self) -> None:
        """Re-check Godot availability."""
        self._check_godot_availability()
