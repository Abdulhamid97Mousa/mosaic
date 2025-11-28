"""Human vs Agent environment configuration form.

This dialog provides configuration options for Human vs Agent gameplay,
including AI opponent selection, difficulty settings, and game-specific options.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from qtpy import QtCore, QtGui, QtWidgets

_LOG = logging.getLogger(__name__)


# =============================================================================
# Configuration Data Classes
# =============================================================================


@dataclass
class StockfishConfig:
    """Configuration for Stockfish chess engine.

    Attributes:
        skill_level: Engine skill level (0-20). Higher = stronger.
            - 0-5: Beginner level, makes intentional mistakes
            - 6-10: Intermediate level
            - 11-15: Advanced level
            - 16-20: Expert/Master level
        depth: Search depth (1-30). How many moves ahead to analyze.
            - Higher depth = stronger play but slower response
            - Depth 5-8 is good for beginners
            - Depth 15-20 for challenging play
        time_limit_ms: Time limit per move in milliseconds.
            - Limits how long Stockfish thinks per move
            - Lower = faster but weaker moves
            - 500-2000ms is typical range
        threads: Number of CPU threads (1-8).
            - More threads = faster analysis
            - Default 1 is fine for most cases
        hash_mb: Hash table size in MB (16-256).
            - Larger = better for long games
            - 16-64MB is sufficient for casual play
    """

    skill_level: int = 10
    depth: int = 12
    time_limit_ms: int = 1000
    threads: int = 1
    hash_mb: int = 16


@dataclass
class HumanVsAgentConfig:
    """Complete configuration for Human vs Agent gameplay.

    Attributes:
        opponent_type: Type of AI opponent
            - "random": Makes random legal moves (for testing)
            - "stockfish": Stockfish chess engine (strong AI)
            - "custom": Load a custom trained policy
        difficulty: Named difficulty preset (for quick selection)
        stockfish: Detailed Stockfish configuration
        custom_policy_path: Path to custom policy file (if opponent_type="custom")
    """

    opponent_type: str = "stockfish"
    difficulty: str = "medium"
    stockfish: StockfishConfig = field(default_factory=StockfishConfig)
    custom_policy_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "opponent_type": self.opponent_type,
            "difficulty": self.difficulty,
            "stockfish": {
                "skill_level": self.stockfish.skill_level,
                "depth": self.stockfish.depth,
                "time_limit_ms": self.stockfish.time_limit_ms,
                "threads": self.stockfish.threads,
                "hash_mb": self.stockfish.hash_mb,
            },
            "custom_policy_path": self.custom_policy_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanVsAgentConfig":
        """Create from dictionary."""
        stockfish_data = data.get("stockfish", {})
        return cls(
            opponent_type=data.get("opponent_type", "stockfish"),
            difficulty=data.get("difficulty", "medium"),
            stockfish=StockfishConfig(
                skill_level=stockfish_data.get("skill_level", 10),
                depth=stockfish_data.get("depth", 12),
                time_limit_ms=stockfish_data.get("time_limit_ms", 1000),
                threads=stockfish_data.get("threads", 1),
                hash_mb=stockfish_data.get("hash_mb", 16),
            ),
            custom_policy_path=data.get("custom_policy_path"),
        )


# =============================================================================
# Difficulty Presets
# =============================================================================

DIFFICULTY_PRESETS: Dict[str, StockfishConfig] = {
    "beginner": StockfishConfig(skill_level=1, depth=5, time_limit_ms=500),
    "easy": StockfishConfig(skill_level=5, depth=8, time_limit_ms=500),
    "medium": StockfishConfig(skill_level=10, depth=12, time_limit_ms=1000),
    "hard": StockfishConfig(skill_level=15, depth=18, time_limit_ms=1500),
    "expert": StockfishConfig(skill_level=20, depth=20, time_limit_ms=2000),
}

DIFFICULTY_DESCRIPTIONS: Dict[str, str] = {
    "beginner": "Perfect for learning. AI makes intentional mistakes and plays slowly.",
    "easy": "Casual play. AI plays reasonably but misses some tactics.",
    "medium": "Balanced challenge. Good for intermediate players.",
    "hard": "Strong play. AI rarely makes mistakes and plays aggressively.",
    "expert": "Maximum strength. Tournament-level play for experienced players.",
}


# =============================================================================
# Configuration Form Dialog
# =============================================================================


class HumanVsAgentConfigForm(QtWidgets.QDialog):
    """Configuration dialog for Human vs Agent gameplay.

    This dialog allows users to configure:
    - AI opponent type (Random, Stockfish, Custom Policy)
    - Difficulty presets with detailed explanations
    - Advanced Stockfish settings (for fine-tuning)
    - Custom policy loading

    Signals:
        config_accepted(HumanVsAgentConfig): Emitted when user clicks OK
    """

    config_accepted = QtCore.Signal(object)  # HumanVsAgentConfig

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        initial_config: Optional[HumanVsAgentConfig] = None,
    ) -> None:
        super().__init__(parent)
        self._config = initial_config or HumanVsAgentConfig()
        self._setup_ui()
        self._connect_signals()
        self._load_config(self._config)

    def _setup_ui(self) -> None:
        """Build the dialog UI."""
        self.setWindowTitle("Environment Configuration - Human vs Agent")
        self.setMinimumWidth(550)
        self.setMinimumHeight(600)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)

        # Header
        header = QtWidgets.QLabel(
            "<h3>Human vs Agent Configuration</h3>"
            "<p style='color: #666;'>Configure the AI opponent and gameplay settings.</p>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Scroll area for settings
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(16)

        # AI Opponent Selection
        scroll_layout.addWidget(self._create_opponent_group())

        # Difficulty Presets
        scroll_layout.addWidget(self._create_difficulty_group())

        # Advanced Settings
        scroll_layout.addWidget(self._create_advanced_group())

        # Requirements Info
        scroll_layout.addWidget(self._create_requirements_group())

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Dialog buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_opponent_group(self) -> QtWidgets.QGroupBox:
        """Create the AI opponent selection group."""
        group = QtWidgets.QGroupBox("AI Opponent")
        layout = QtWidgets.QVBoxLayout(group)

        # Opponent type selection
        self._opponent_combo = QtWidgets.QComboBox()
        self._opponent_combo.addItem("Random (for testing)", "random")
        self._opponent_combo.addItem("Stockfish Chess Engine", "stockfish")
        self._opponent_combo.addItem("Custom Trained Policy", "custom")

        type_layout = QtWidgets.QHBoxLayout()
        type_layout.addWidget(QtWidgets.QLabel("Opponent Type:"))
        type_layout.addWidget(self._opponent_combo, 1)
        layout.addLayout(type_layout)

        # Opponent description
        self._opponent_desc = QtWidgets.QLabel()
        self._opponent_desc.setWordWrap(True)
        self._opponent_desc.setStyleSheet(
            "color: #666; font-size: 11px; padding: 8px; "
            "background-color: #f5f5f5; border-radius: 4px;"
        )
        layout.addWidget(self._opponent_desc)

        return group

    def _create_difficulty_group(self) -> QtWidgets.QGroupBox:
        """Create the difficulty presets group."""
        self._difficulty_group = QtWidgets.QGroupBox("Difficulty Level")
        layout = QtWidgets.QVBoxLayout(self._difficulty_group)

        # Difficulty buttons (radio)
        self._difficulty_buttons: Dict[str, QtWidgets.QRadioButton] = {}

        for difficulty, desc in DIFFICULTY_DESCRIPTIONS.items():
            preset = DIFFICULTY_PRESETS[difficulty]

            # Create radio button with detailed label
            btn = QtWidgets.QRadioButton()
            self._difficulty_buttons[difficulty] = btn

            # Create detailed info layout
            info_widget = QtWidgets.QWidget()
            info_layout = QtWidgets.QVBoxLayout(info_widget)
            info_layout.setContentsMargins(0, 0, 0, 8)
            info_layout.setSpacing(2)

            # Title with stats
            title = QtWidgets.QLabel(
                f"<b>{difficulty.capitalize()}</b> "
                f"<span style='color: #888;'>(Skill: {preset.skill_level}, "
                f"Depth: {preset.depth}, Time: {preset.time_limit_ms}ms)</span>"
            )
            title.setStyleSheet("font-size: 12px;")

            # Description
            desc_label = QtWidgets.QLabel(desc)
            desc_label.setStyleSheet("color: #666; font-size: 11px; margin-left: 20px;")
            desc_label.setWordWrap(True)

            info_layout.addWidget(title)
            info_layout.addWidget(desc_label)

            # Combine radio + info
            row_layout = QtWidgets.QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(btn)
            row_layout.addWidget(info_widget, 1)

            layout.addLayout(row_layout)

        return self._difficulty_group

    def _create_advanced_group(self) -> QtWidgets.QGroupBox:
        """Create the advanced settings group."""
        self._advanced_group = QtWidgets.QGroupBox("Advanced Settings")
        self._advanced_group.setCheckable(True)
        self._advanced_group.setChecked(False)

        layout = QtWidgets.QFormLayout(self._advanced_group)
        layout.setSpacing(12)

        # Skill Level
        self._skill_spin = QtWidgets.QSpinBox()
        self._skill_spin.setRange(0, 20)
        self._skill_spin.setValue(10)
        self._skill_spin.setToolTip(
            "Stockfish skill level (0-20).\n\n"
            "0-5: Beginner - Makes intentional mistakes\n"
            "6-10: Intermediate - Reasonable play\n"
            "11-15: Advanced - Strong tactics\n"
            "16-20: Expert - Near-perfect play\n\n"
            "Higher values make the engine stronger."
        )
        layout.addRow("Skill Level (0-20):", self._skill_spin)

        # Search Depth
        self._depth_spin = QtWidgets.QSpinBox()
        self._depth_spin.setRange(1, 30)
        self._depth_spin.setValue(12)
        self._depth_spin.setToolTip(
            "Search depth - how many moves ahead Stockfish analyzes.\n\n"
            "1-5: Very fast, weak play\n"
            "6-10: Fast, moderate strength\n"
            "11-15: Balanced speed/strength\n"
            "16-20: Slow but strong\n"
            "21-30: Very slow, maximum strength\n\n"
            "Higher depth = stronger but slower moves."
        )
        layout.addRow("Search Depth (1-30):", self._depth_spin)

        # Time Limit
        self._time_spin = QtWidgets.QSpinBox()
        self._time_spin.setRange(100, 10000)
        self._time_spin.setSingleStep(100)
        self._time_spin.setSuffix(" ms")
        self._time_spin.setValue(1000)
        self._time_spin.setToolTip(
            "Maximum time per move in milliseconds.\n\n"
            "100-500ms: Very fast responses\n"
            "500-1000ms: Normal play speed\n"
            "1000-2000ms: Thoughtful play\n"
            "2000-5000ms: Deep analysis\n\n"
            "This limits thinking time even if depth isn't reached."
        )
        layout.addRow("Time Limit:", self._time_spin)

        # CPU Threads
        self._threads_spin = QtWidgets.QSpinBox()
        self._threads_spin.setRange(1, 8)
        self._threads_spin.setValue(1)
        self._threads_spin.setToolTip(
            "Number of CPU threads for analysis.\n\n"
            "More threads = faster analysis on multi-core CPUs.\n"
            "1 thread is usually sufficient for casual play.\n"
            "Use 2-4 for faster responses on modern CPUs."
        )
        layout.addRow("CPU Threads (1-8):", self._threads_spin)

        # Hash Size
        self._hash_spin = QtWidgets.QSpinBox()
        self._hash_spin.setRange(16, 256)
        self._hash_spin.setSingleStep(16)
        self._hash_spin.setSuffix(" MB")
        self._hash_spin.setValue(16)
        self._hash_spin.setToolTip(
            "Hash table size for position caching.\n\n"
            "16-32 MB: Sufficient for short games\n"
            "64-128 MB: Good for longer games\n"
            "256 MB: Maximum for very long analysis\n\n"
            "Larger tables help in long games but use more RAM."
        )
        layout.addRow("Hash Table:", self._hash_spin)

        # Info note
        note = QtWidgets.QLabel(
            "<i>Note: Advanced settings override the difficulty preset. "
            "Uncheck this box to use preset values.</i>"
        )
        note.setStyleSheet("color: #888; font-size: 10px;")
        note.setWordWrap(True)
        layout.addRow("", note)

        return self._advanced_group

    def _create_requirements_group(self) -> QtWidgets.QGroupBox:
        """Create the requirements info group."""
        group = QtWidgets.QGroupBox("Requirements")
        layout = QtWidgets.QVBoxLayout(group)

        # Check Stockfish availability
        stockfish_available = self._check_stockfish()

        if stockfish_available:
            status_text = (
                '<span style="color: green;">&#x2714;</span> '
                "<b>Stockfish is installed</b> and ready to use."
            )
        else:
            status_text = (
                '<span style="color: red;">&#x2718;</span> '
                "<b>Stockfish is not installed.</b><br>"
                "Install with: <code>sudo apt install stockfish</code>"
            )

        status = QtWidgets.QLabel(status_text)
        status.setWordWrap(True)
        layout.addWidget(status)

        # Info about settings
        info = QtWidgets.QLabel(
            "<hr>"
            "<b>What do these settings mean?</b><br><br>"
            "<b>Skill Level:</b> Controls how well Stockfish plays. "
            "Lower levels make intentional mistakes to give humans a chance.<br><br>"
            "<b>Search Depth:</b> How many moves ahead the engine calculates. "
            "Deeper search finds better moves but takes longer.<br><br>"
            "<b>Time Limit:</b> Maximum thinking time per move. "
            "Prevents the engine from taking too long on complex positions.<br><br>"
            "<b>Threads:</b> Parallel processing for faster analysis. "
            "More threads use more CPU but respond faster.<br><br>"
            "<b>Hash Table:</b> Memory cache for analyzed positions. "
            "Larger tables help in long games with repeated positions."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 11px; color: #555;")
        layout.addWidget(info)

        return group

    def _check_stockfish(self) -> bool:
        """Check if Stockfish is available."""
        import shutil

        paths = [
            "/usr/games/stockfish",
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish",
            "stockfish",
        ]
        for path in paths:
            if shutil.which(path):
                return True
        return False

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._opponent_combo.currentIndexChanged.connect(self._on_opponent_changed)

        for difficulty, btn in self._difficulty_buttons.items():
            btn.toggled.connect(lambda checked, d=difficulty: self._on_difficulty_changed(d, checked))

        self._advanced_group.toggled.connect(self._on_advanced_toggled)

    def _load_config(self, config: HumanVsAgentConfig) -> None:
        """Load configuration into the form."""
        # Opponent type
        index = self._opponent_combo.findData(config.opponent_type)
        if index >= 0:
            self._opponent_combo.setCurrentIndex(index)

        # Difficulty
        if config.difficulty in self._difficulty_buttons:
            self._difficulty_buttons[config.difficulty].setChecked(True)

        # Advanced settings
        self._skill_spin.setValue(config.stockfish.skill_level)
        self._depth_spin.setValue(config.stockfish.depth)
        self._time_spin.setValue(config.stockfish.time_limit_ms)
        self._threads_spin.setValue(config.stockfish.threads)
        self._hash_spin.setValue(config.stockfish.hash_mb)

        # Update descriptions
        self._update_opponent_description()

    def _on_opponent_changed(self, index: int) -> None:
        """Handle opponent type change."""
        opponent_type = self._opponent_combo.currentData()

        # Enable/disable difficulty and advanced based on opponent
        is_stockfish = opponent_type == "stockfish"
        self._difficulty_group.setEnabled(is_stockfish)
        self._advanced_group.setEnabled(is_stockfish)

        self._update_opponent_description()

    def _update_opponent_description(self) -> None:
        """Update the opponent description label."""
        opponent_type = self._opponent_combo.currentData()

        descriptions = {
            "random": (
                "<b>Random AI</b><br>"
                "Makes random legal moves. Useful for testing the interface "
                "or practicing basic moves without challenge."
            ),
            "stockfish": (
                "<b>Stockfish Chess Engine</b><br>"
                "One of the strongest chess engines in the world. "
                "Adjustable difficulty from beginner to grandmaster level. "
                "Requires Stockfish to be installed on your system."
            ),
            "custom": (
                "<b>Custom Trained Policy</b><br>"
                "Load a neural network policy trained with reinforcement learning. "
                "Use this to test your own trained agents."
            ),
        }

        self._opponent_desc.setText(descriptions.get(opponent_type, ""))

    def _on_difficulty_changed(self, difficulty: str, checked: bool) -> None:
        """Handle difficulty preset selection."""
        if not checked:
            return

        # Only apply preset if advanced settings are not manually enabled
        if not self._advanced_group.isChecked():
            preset = DIFFICULTY_PRESETS.get(difficulty)
            if preset:
                self._skill_spin.setValue(preset.skill_level)
                self._depth_spin.setValue(preset.depth)
                self._time_spin.setValue(preset.time_limit_ms)

    def _on_advanced_toggled(self, checked: bool) -> None:
        """Handle advanced settings toggle."""
        if not checked:
            # Restore preset values
            for difficulty, btn in self._difficulty_buttons.items():
                if btn.isChecked():
                    preset = DIFFICULTY_PRESETS.get(difficulty)
                    if preset:
                        self._skill_spin.setValue(preset.skill_level)
                        self._depth_spin.setValue(preset.depth)
                        self._time_spin.setValue(preset.time_limit_ms)
                    break

    def _on_accept(self) -> None:
        """Handle OK button click."""
        config = self.get_config()
        self.config_accepted.emit(config)
        self.accept()

    def get_config(self) -> HumanVsAgentConfig:
        """Get the current configuration from the form."""
        opponent_type = self._opponent_combo.currentData() or "stockfish"

        # Get selected difficulty
        difficulty = "medium"
        for diff, btn in self._difficulty_buttons.items():
            if btn.isChecked():
                difficulty = diff
                break

        return HumanVsAgentConfig(
            opponent_type=opponent_type,
            difficulty=difficulty,
            stockfish=StockfishConfig(
                skill_level=self._skill_spin.value(),
                depth=self._depth_spin.value(),
                time_limit_ms=self._time_spin.value(),
                threads=self._threads_spin.value(),
                hash_mb=self._hash_spin.value(),
            ),
            custom_policy_path=None,  # TODO: Add file picker for custom policies
        )

    def get_stockfish_config(self) -> StockfishConfig:
        """Get just the Stockfish configuration."""
        return self.get_config().stockfish


__all__ = [
    "HumanVsAgentConfigForm",
    "HumanVsAgentConfig",
    "StockfishConfig",
    "DIFFICULTY_PRESETS",
    "DIFFICULTY_DESCRIPTIONS",
]
