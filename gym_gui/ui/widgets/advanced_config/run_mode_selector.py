"""Run mode selection widget.

Step 4 of the Unified Flow: Select how to run the session.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal

_LOGGER = logging.getLogger(__name__)


class RunMode(Enum):
    """Available run modes for the session."""

    INTERACTIVE = auto()
    """Interactive mode with rendering - for play/visualization."""

    HEADLESS = auto()
    """Headless training mode - no rendering, maximum speed."""

    EVALUATION = auto()
    """Evaluation mode - load trained policy, with rendering."""


RUN_MODE_METADATA = {
    RunMode.INTERACTIVE: {
        "label": "Interactive (with rendering)",
        "description": (
            "Run with full visualization. Use for human play, "
            "demonstrations, or debugging trained agents."
        ),
        "icon": "play",
    },
    RunMode.HEADLESS: {
        "label": "Headless Training (no rendering)",
        "description": (
            "Maximum training speed without visualization. "
            "Telemetry and metrics still collected."
        ),
        "icon": "train",
    },
    RunMode.EVALUATION: {
        "label": "Evaluation (load trained policy)",
        "description": (
            "Load a trained policy and evaluate with rendering. "
            "No training updates applied."
        ),
        "icon": "eval",
    },
}


class RunModeSelector(QtWidgets.QGroupBox):
    """Step 4: Run mode selection.

    Lets user choose between interactive, headless, and evaluation modes.

    Signals:
        mode_changed: Emitted when the selected mode changes

    Example:
        selector = RunModeSelector()
        selector.mode_changed.connect(on_mode_change)
    """

    # Signals
    mode_changed = pyqtSignal(object)  # RunMode

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Step 4: Run Mode", parent)
        self._selected_mode: RunMode = RunMode.INTERACTIVE
        self._radio_buttons: dict[RunMode, QtWidgets.QRadioButton] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        # Info label
        info = QtWidgets.QLabel(
            "Choose how to run the session. "
            "Interactive mode shows the environment, headless maximizes training speed."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info)

        # Radio buttons for each mode
        for mode in RunMode:
            meta = RUN_MODE_METADATA[mode]

            # Create radio button
            radio = QtWidgets.QRadioButton(meta["label"])
            radio.setToolTip(meta["description"])
            radio.toggled.connect(lambda checked, m=mode: self._on_radio_toggled(m, checked))

            self._radio_buttons[mode] = radio
            layout.addWidget(radio)

            # Description label (smaller, indented)
            desc = QtWidgets.QLabel(meta["description"])
            desc.setWordWrap(True)
            desc.setStyleSheet(
                "color: #888; font-size: 10px; margin-left: 20px; margin-bottom: 8px;"
            )
            layout.addWidget(desc)

        # Set default
        self._radio_buttons[RunMode.INTERACTIVE].setChecked(True)

    def _on_radio_toggled(self, mode: RunMode, checked: bool) -> None:
        """Handle radio button toggle."""
        if checked:
            self._selected_mode = mode
            self.mode_changed.emit(mode)
            _LOGGER.debug("Run mode changed to: %s", mode.name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def selected_mode(self) -> RunMode:
        """Get the currently selected run mode."""
        return self._selected_mode

    def set_mode(self, mode: RunMode) -> None:
        """Set the selected run mode.

        Args:
            mode: The run mode to select.
        """
        radio = self._radio_buttons.get(mode)
        if radio:
            radio.setChecked(True)

    def is_interactive(self) -> bool:
        """Check if interactive mode is selected."""
        return self._selected_mode == RunMode.INTERACTIVE

    def is_headless(self) -> bool:
        """Check if headless training mode is selected."""
        return self._selected_mode == RunMode.HEADLESS

    def is_evaluation(self) -> bool:
        """Check if evaluation mode is selected."""
        return self._selected_mode == RunMode.EVALUATION
