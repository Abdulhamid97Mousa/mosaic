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
class KataGoConfig:
    """Configuration for KataGo Go engine.

    Attributes:
        playouts: Number of playouts/visits per move (strength control).
            - More playouts = stronger but slower
            - 10-50: Fast, weak play
            - 200-800: Balanced play
            - 3200+: Very strong play
        max_visits: Maximum visits (alternative strength control).
        time_limit_sec: Time limit per move in seconds.
        threads: Number of CPU threads for search.
    """

    playouts: int = 200
    max_visits: int = 400
    time_limit_sec: float = 5.0
    threads: int = 1


@dataclass
class GnuGoConfig:
    """Configuration for GNU Go engine.

    Attributes:
        level: Strength level (0-10). Higher = stronger.
            - 0-3: Beginner level
            - 4-6: Intermediate level
            - 7-9: Advanced level
            - 10: Maximum strength (amateur dan level)
        chinese_rules: Use Chinese rules instead of Japanese.
    """

    level: int = 10
    chinese_rules: bool = False


@dataclass
class HumanVsAgentConfig:
    """Complete configuration for Human vs Agent gameplay.

    Attributes:
        opponent_type: Type of AI opponent
            Chess: "random", "stockfish", "custom"
            Go: "random", "katago", "gnugo", "custom"
        difficulty: Named difficulty preset (for quick selection)
        stockfish: Detailed Stockfish configuration (for Chess)
        katago: Detailed KataGo configuration (for Go)
        gnugo: Detailed GNU Go configuration (for Go)
        custom_policy_path: Path to custom policy file (if opponent_type="custom")
    """

    opponent_type: str = "stockfish"
    difficulty: str = "medium"
    stockfish: StockfishConfig = field(default_factory=StockfishConfig)
    katago: KataGoConfig = field(default_factory=KataGoConfig)
    gnugo: GnuGoConfig = field(default_factory=GnuGoConfig)
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
            "katago": {
                "playouts": self.katago.playouts,
                "max_visits": self.katago.max_visits,
                "time_limit_sec": self.katago.time_limit_sec,
                "threads": self.katago.threads,
            },
            "gnugo": {
                "level": self.gnugo.level,
                "chinese_rules": self.gnugo.chinese_rules,
            },
            "custom_policy_path": self.custom_policy_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanVsAgentConfig":
        """Create from dictionary."""
        stockfish_data = data.get("stockfish", {})
        katago_data = data.get("katago", {})
        gnugo_data = data.get("gnugo", {})
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
            katago=KataGoConfig(
                playouts=katago_data.get("playouts", 200),
                max_visits=katago_data.get("max_visits", 400),
                time_limit_sec=katago_data.get("time_limit_sec", 5.0),
                threads=katago_data.get("threads", 1),
            ),
            gnugo=GnuGoConfig(
                level=gnugo_data.get("level", 10),
                chinese_rules=gnugo_data.get("chinese_rules", False),
            ),
            custom_policy_path=data.get("custom_policy_path"),
        )


# =============================================================================
# Difficulty Presets
# =============================================================================

# Chess (Stockfish) presets
STOCKFISH_DIFFICULTY_PRESETS: Dict[str, StockfishConfig] = {
    "beginner": StockfishConfig(skill_level=1, depth=5, time_limit_ms=500),
    "easy": StockfishConfig(skill_level=5, depth=8, time_limit_ms=500),
    "medium": StockfishConfig(skill_level=10, depth=12, time_limit_ms=1000),
    "hard": StockfishConfig(skill_level=15, depth=18, time_limit_ms=1500),
    "expert": StockfishConfig(skill_level=20, depth=20, time_limit_ms=2000),
}

STOCKFISH_DIFFICULTY_DESCRIPTIONS: Dict[str, str] = {
    "beginner": "Perfect for learning. AI makes intentional mistakes and plays slowly.",
    "easy": "Casual play. AI plays reasonably but misses some tactics.",
    "medium": "Balanced challenge. Good for intermediate players.",
    "hard": "Strong play. AI rarely makes mistakes and plays aggressively.",
    "expert": "Maximum strength. Tournament-level play for experienced players.",
}

# Go (KataGo) presets
KATAGO_DIFFICULTY_PRESETS: Dict[str, KataGoConfig] = {
    "beginner": KataGoConfig(playouts=10, max_visits=20, time_limit_sec=1.0),
    "easy": KataGoConfig(playouts=50, max_visits=100, time_limit_sec=2.0),
    "medium": KataGoConfig(playouts=200, max_visits=400, time_limit_sec=5.0),
    "hard": KataGoConfig(playouts=800, max_visits=1600, time_limit_sec=10.0),
    "expert": KataGoConfig(playouts=3200, max_visits=6400, time_limit_sec=30.0),
}

KATAGO_DIFFICULTY_DESCRIPTIONS: Dict[str, str] = {
    "beginner": "Very weak play. Perfect for learning Go basics.",
    "easy": "Casual play. Makes mistakes but plays sensible Go.",
    "medium": "Balanced challenge. Good for intermediate players (10-15 kyu).",
    "hard": "Strong play. Good for dan-level players.",
    "expert": "Maximum strength. Professional-level play.",
}

# Go (GNU Go) presets
GNUGO_DIFFICULTY_PRESETS: Dict[str, GnuGoConfig] = {
    "beginner": GnuGoConfig(level=1),
    "easy": GnuGoConfig(level=4),
    "medium": GnuGoConfig(level=7),
    "hard": GnuGoConfig(level=9),
    "expert": GnuGoConfig(level=10),
}

GNUGO_DIFFICULTY_DESCRIPTIONS: Dict[str, str] = {
    "beginner": "Very weak play. Perfect for learning Go basics.",
    "easy": "Casual play. Makes obvious mistakes.",
    "medium": "Moderate challenge. Good for beginners improving.",
    "hard": "Strong classical play. Good for intermediate players.",
    "expert": "Maximum GNU Go strength. Amateur dan level.",
}

# Backwards compatibility aliases
DIFFICULTY_PRESETS = STOCKFISH_DIFFICULTY_PRESETS
DIFFICULTY_DESCRIPTIONS = STOCKFISH_DIFFICULTY_DESCRIPTIONS


# =============================================================================
# Configuration Form Dialog
# =============================================================================


class HumanVsAgentConfigForm(QtWidgets.QDialog):
    """Configuration dialog for Human vs Agent gameplay.

    This dialog allows users to configure:
    - AI opponent type (game-specific: Stockfish for Chess, KataGo/GNU Go for Go)
    - Difficulty presets with detailed explanations
    - Advanced settings (for fine-tuning)
    - Custom policy loading

    Signals:
        config_accepted(HumanVsAgentConfig): Emitted when user clicks OK
    """

    config_accepted = QtCore.Signal(object)  # HumanVsAgentConfig

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        initial_config: Optional[HumanVsAgentConfig] = None,
        game_type: str = "chess",
    ) -> None:
        """Initialize the configuration form.

        Args:
            parent: Parent widget.
            initial_config: Initial configuration to load.
            game_type: Game type - "chess" or "go". Determines which AI options to show.
        """
        super().__init__(parent)
        self._config = initial_config or HumanVsAgentConfig()
        self._game_type = game_type.lower()
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

        # Opponent type selection (game-specific)
        self._opponent_combo = QtWidgets.QComboBox()
        self._opponent_combo.addItem("Random (for testing)", "random")

        if self._game_type == "go":
            self._opponent_combo.addItem("KataGo Go Engine (Recommended)", "katago")
            self._opponent_combo.addItem("GNU Go Engine", "gnugo")
        else:
            # Chess (default)
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
        self._difficulty_labels: Dict[str, QtWidgets.QLabel] = {}

        # Get game-specific descriptions (presets will be looked up dynamically)
        descriptions = self._get_difficulty_descriptions()

        for difficulty, desc in descriptions.items():
            # Create radio button with detailed label
            btn = QtWidgets.QRadioButton()
            self._difficulty_buttons[difficulty] = btn

            # Create detailed info layout
            info_widget = QtWidgets.QWidget()
            info_layout = QtWidgets.QVBoxLayout(info_widget)
            info_layout.setContentsMargins(0, 0, 0, 8)
            info_layout.setSpacing(2)

            # Title with stats (will be updated based on opponent type)
            title = QtWidgets.QLabel()
            title.setStyleSheet("font-size: 12px;")
            self._difficulty_labels[difficulty] = title

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

    def _get_difficulty_descriptions(self) -> Dict[str, str]:
        """Get difficulty descriptions based on game type."""
        if self._game_type == "go":
            return KATAGO_DIFFICULTY_DESCRIPTIONS  # Same for GNU Go
        else:
            return STOCKFISH_DIFFICULTY_DESCRIPTIONS

    def _update_difficulty_labels(self) -> None:
        """Update difficulty labels based on current opponent type."""
        opponent_type = self._opponent_combo.currentData()

        for difficulty, label in self._difficulty_labels.items():
            stats_text = self._get_difficulty_stats(difficulty, opponent_type)
            label.setText(
                f"<b>{difficulty.capitalize()}</b> "
                f"<span style='color: #888;'>{stats_text}</span>"
            )

    def _get_difficulty_stats(self, difficulty: str, opponent_type: str) -> str:
        """Get stats text for a difficulty level based on opponent type."""
        if opponent_type == "stockfish":
            preset = STOCKFISH_DIFFICULTY_PRESETS.get(difficulty)
            if preset:
                return f"(Skill: {preset.skill_level}, Depth: {preset.depth}, Time: {preset.time_limit_ms}ms)"
        elif opponent_type == "katago":
            preset = KATAGO_DIFFICULTY_PRESETS.get(difficulty)
            if preset:
                return f"(Playouts: {preset.playouts}, Time: {preset.time_limit_sec}s)"
        elif opponent_type == "gnugo":
            preset = GNUGO_DIFFICULTY_PRESETS.get(difficulty)
            if preset:
                return f"(Level: {preset.level}/10)"
        return ""

    def _create_advanced_group(self) -> QtWidgets.QGroupBox:
        """Create the advanced settings group."""
        self._advanced_group = QtWidgets.QGroupBox("Advanced Settings")
        self._advanced_group.setCheckable(True)
        self._advanced_group.setChecked(False)

        layout = QtWidgets.QFormLayout(self._advanced_group)
        layout.setSpacing(12)

        if self._game_type == "go":
            self._create_go_advanced_settings(layout)
        else:
            self._create_chess_advanced_settings(layout)

        # Info note
        note = QtWidgets.QLabel(
            "<i>Note: Advanced settings override the difficulty preset. "
            "Uncheck this box to use preset values.</i>"
        )
        note.setStyleSheet("color: #888; font-size: 10px;")
        note.setWordWrap(True)
        layout.addRow("", note)

        return self._advanced_group

    def _create_chess_advanced_settings(self, layout: QtWidgets.QFormLayout) -> None:
        """Create Chess (Stockfish) advanced settings."""
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

    def _create_go_advanced_settings(self, layout: QtWidgets.QFormLayout) -> None:
        """Create Go (KataGo/GNU Go) advanced settings."""
        # Playouts (KataGo)
        self._playouts_spin = QtWidgets.QSpinBox()
        self._playouts_spin.setRange(1, 10000)
        self._playouts_spin.setValue(200)
        self._playouts_spin.setToolTip(
            "Number of playouts per move (KataGo).\n\n"
            "10-50: Very fast, weak play\n"
            "100-400: Balanced speed/strength\n"
            "800-3200: Strong play\n"
            "3200+: Maximum strength (slow)\n\n"
            "Higher playouts = stronger but slower moves."
        )
        layout.addRow("Playouts (KataGo):", self._playouts_spin)

        # Level (GNU Go)
        self._level_spin = QtWidgets.QSpinBox()
        self._level_spin.setRange(0, 10)
        self._level_spin.setValue(10)
        self._level_spin.setToolTip(
            "GNU Go strength level (0-10).\n\n"
            "0-3: Beginner - Very weak play\n"
            "4-6: Intermediate - Moderate play\n"
            "7-9: Advanced - Good play\n"
            "10: Expert - Maximum strength\n\n"
            "Higher level = stronger play."
        )
        layout.addRow("Level (GNU Go, 0-10):", self._level_spin)

        # Time Limit (seconds for KataGo)
        self._go_time_spin = QtWidgets.QDoubleSpinBox()
        self._go_time_spin.setRange(0.5, 60.0)
        self._go_time_spin.setSingleStep(0.5)
        self._go_time_spin.setSuffix(" sec")
        self._go_time_spin.setValue(5.0)
        self._go_time_spin.setToolTip(
            "Maximum time per move in seconds (KataGo).\n\n"
            "1-2s: Fast responses\n"
            "5-10s: Normal play\n"
            "30-60s: Deep analysis\n\n"
            "This limits thinking time."
        )
        layout.addRow("Time Limit:", self._go_time_spin)

        # CPU Threads
        self._threads_spin = QtWidgets.QSpinBox()
        self._threads_spin.setRange(1, 8)
        self._threads_spin.setValue(1)
        self._threads_spin.setToolTip(
            "Number of CPU threads for analysis.\n\n"
            "More threads = faster analysis on multi-core CPUs.\n"
            "1 thread is usually sufficient for casual play."
        )
        layout.addRow("CPU Threads:", self._threads_spin)

    def _create_requirements_group(self) -> QtWidgets.QGroupBox:
        """Create the requirements info group."""
        group = QtWidgets.QGroupBox("Requirements")
        layout = QtWidgets.QVBoxLayout(group)

        if self._game_type == "go":
            self._create_go_requirements(layout)
        else:
            self._create_chess_requirements(layout)

        return group

    def _create_chess_requirements(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Create Chess (Stockfish) requirements info."""
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

        info = QtWidgets.QLabel(
            "<hr>"
            "<b>What do these settings mean?</b><br><br>"
            "<b>Skill Level:</b> Controls how well Stockfish plays. "
            "Lower levels make intentional mistakes to give humans a chance.<br><br>"
            "<b>Search Depth:</b> How many moves ahead the engine calculates. "
            "Deeper search finds better moves but takes longer.<br><br>"
            "<b>Time Limit:</b> Maximum thinking time per move. "
            "Prevents the engine from taking too long on complex positions."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 11px; color: #555;")
        layout.addWidget(info)

    def _create_go_requirements(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Create Go (KataGo/GNU Go) requirements info."""
        katago_available = self._check_katago()
        gnugo_available = self._check_gnugo()

        status_parts = []

        if katago_available:
            status_parts.append(
                '<span style="color: green;">&#x2714;</span> '
                "<b>KataGo is installed</b> (superhuman strength)"
            )
        else:
            status_parts.append(
                '<span style="color: orange;">&#x26A0;</span> '
                "<b>KataGo not available</b> - requires binary + neural net model"
            )

        if gnugo_available:
            status_parts.append(
                '<span style="color: green;">&#x2714;</span> '
                "<b>GNU Go is installed</b> (amateur dan level)"
            )
        else:
            status_parts.append(
                '<span style="color: red;">&#x2718;</span> '
                "<b>GNU Go is not installed.</b><br>"
                "&nbsp;&nbsp;&nbsp;Install with: <code>sudo apt install gnugo</code>"
            )

        status = QtWidgets.QLabel("<br>".join(status_parts))
        status.setWordWrap(True)
        layout.addWidget(status)

        info = QtWidgets.QLabel(
            "<hr>"
            "<b>Go AI Engines:</b><br><br>"
            "<b>KataGo:</b> Superhuman-strength neural network AI. "
            "Requires GPU or lots of CPU. Best for strong play.<br><br>"
            "<b>GNU Go:</b> Classical Go AI without neural networks. "
            "Amateur dan level. Simpler setup, works everywhere.<br><br>"
            "<b>Playouts:</b> Number of game simulations per move. "
            "More playouts = stronger but slower play."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 11px; color: #555;")
        layout.addWidget(info)

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

    def _check_katago(self) -> bool:
        """Check if KataGo is available."""
        import shutil
        import os
        from gym_gui.config.paths import VAR_BIN_DIR, VAR_MODELS_KATAGO_DIR

        # Check binary
        paths = [
            str(VAR_BIN_DIR / "katago"),
            "/usr/games/katago",
            "/usr/bin/katago",
            "/usr/local/bin/katago",
            "katago",
        ]
        binary_found = False
        for path in paths:
            if shutil.which(path) or (os.path.isfile(path) and os.access(path, os.X_OK)):
                binary_found = True
                break

        if not binary_found:
            return False

        # Check model
        model_dir = str(VAR_MODELS_KATAGO_DIR)
        if os.path.isdir(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith(".bin.gz") or f.endswith(".txt.gz"):
                    return True
        return False

    def _check_gnugo(self) -> bool:
        """Check if GNU Go is available."""
        import shutil

        paths = [
            "/usr/games/gnugo",
            "/usr/bin/gnugo",
            "/usr/local/bin/gnugo",
            "gnugo",
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
        # Opponent type - find appropriate default if not in combo
        index = self._opponent_combo.findData(config.opponent_type)
        if index >= 0:
            self._opponent_combo.setCurrentIndex(index)
        else:
            # Set default based on game type
            if self._game_type == "go":
                self._opponent_combo.setCurrentIndex(1)  # KataGo or GNU Go
            else:
                self._opponent_combo.setCurrentIndex(1)  # Stockfish

        # Difficulty
        if config.difficulty in self._difficulty_buttons:
            self._difficulty_buttons[config.difficulty].setChecked(True)

        # Advanced settings (game-specific)
        if self._game_type == "go":
            # Go settings
            if hasattr(self, "_playouts_spin"):
                self._playouts_spin.setValue(config.katago.playouts)
            if hasattr(self, "_level_spin"):
                self._level_spin.setValue(config.gnugo.level)
            if hasattr(self, "_go_time_spin"):
                self._go_time_spin.setValue(config.katago.time_limit_sec)
            if hasattr(self, "_threads_spin"):
                self._threads_spin.setValue(config.katago.threads)
        else:
            # Chess settings
            if hasattr(self, "_skill_spin"):
                self._skill_spin.setValue(config.stockfish.skill_level)
            if hasattr(self, "_depth_spin"):
                self._depth_spin.setValue(config.stockfish.depth)
            if hasattr(self, "_time_spin"):
                self._time_spin.setValue(config.stockfish.time_limit_ms)
            if hasattr(self, "_threads_spin"):
                self._threads_spin.setValue(config.stockfish.threads)
            if hasattr(self, "_hash_spin"):
                self._hash_spin.setValue(config.stockfish.hash_mb)

        # Update descriptions and labels
        self._update_opponent_description()
        self._update_difficulty_labels()

    def _on_opponent_changed(self, index: int) -> None:
        """Handle opponent type change."""
        opponent_type = self._opponent_combo.currentData()

        # Enable/disable difficulty and advanced based on opponent
        is_engine = opponent_type in ("stockfish", "katago", "gnugo")
        self._difficulty_group.setEnabled(is_engine)
        self._advanced_group.setEnabled(is_engine)

        self._update_opponent_description()
        self._update_difficulty_labels()

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
            "katago": (
                "<b>KataGo Go Engine</b><br>"
                "Superhuman-strength Go AI using neural networks. "
                "Adjustable difficulty via playouts. "
                "Requires KataGo binary and neural network model."
            ),
            "gnugo": (
                "<b>GNU Go Engine</b><br>"
                "Classical Go AI (no neural network required). "
                "Amateur dan-level strength. "
                "Simpler setup - just install the package."
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
            # Restore preset values based on game type
            for difficulty, btn in self._difficulty_buttons.items():
                if btn.isChecked():
                    self._apply_difficulty_preset(difficulty)
                    break

    def _apply_difficulty_preset(self, difficulty: str) -> None:
        """Apply a difficulty preset to the advanced settings."""
        opponent_type = self._opponent_combo.currentData()

        if opponent_type == "stockfish":
            preset = STOCKFISH_DIFFICULTY_PRESETS.get(difficulty)
            if preset and hasattr(self, "_skill_spin"):
                self._skill_spin.setValue(preset.skill_level)
                self._depth_spin.setValue(preset.depth)
                self._time_spin.setValue(preset.time_limit_ms)
        elif opponent_type == "katago":
            preset = KATAGO_DIFFICULTY_PRESETS.get(difficulty)
            if preset and hasattr(self, "_playouts_spin"):
                self._playouts_spin.setValue(preset.playouts)
                self._go_time_spin.setValue(preset.time_limit_sec)
                self._threads_spin.setValue(preset.threads)
        elif opponent_type == "gnugo":
            preset = GNUGO_DIFFICULTY_PRESETS.get(difficulty)
            if preset and hasattr(self, "_level_spin"):
                self._level_spin.setValue(preset.level)

    def _on_accept(self) -> None:
        """Handle OK button click."""
        config = self.get_config()
        self.config_accepted.emit(config)
        self.accept()

    def get_config(self) -> HumanVsAgentConfig:
        """Get the current configuration from the form."""
        opponent_type = self._opponent_combo.currentData() or "random"

        # Get selected difficulty
        difficulty = "medium"
        for diff, btn in self._difficulty_buttons.items():
            if btn.isChecked():
                difficulty = diff
                break

        # Build game-specific config
        if self._game_type == "go":
            return HumanVsAgentConfig(
                opponent_type=opponent_type,
                difficulty=difficulty,
                stockfish=StockfishConfig(),  # Default, not used
                katago=KataGoConfig(
                    playouts=self._playouts_spin.value() if hasattr(self, "_playouts_spin") else 200,
                    max_visits=self._playouts_spin.value() * 2 if hasattr(self, "_playouts_spin") else 400,
                    time_limit_sec=self._go_time_spin.value() if hasattr(self, "_go_time_spin") else 5.0,
                    threads=self._threads_spin.value() if hasattr(self, "_threads_spin") else 1,
                ),
                gnugo=GnuGoConfig(
                    level=self._level_spin.value() if hasattr(self, "_level_spin") else 10,
                ),
                custom_policy_path=None,
            )
        else:
            return HumanVsAgentConfig(
                opponent_type=opponent_type,
                difficulty=difficulty,
                stockfish=StockfishConfig(
                    skill_level=self._skill_spin.value() if hasattr(self, "_skill_spin") else 10,
                    depth=self._depth_spin.value() if hasattr(self, "_depth_spin") else 12,
                    time_limit_ms=self._time_spin.value() if hasattr(self, "_time_spin") else 1000,
                    threads=self._threads_spin.value() if hasattr(self, "_threads_spin") else 1,
                    hash_mb=self._hash_spin.value() if hasattr(self, "_hash_spin") else 16,
                ),
                katago=KataGoConfig(),  # Default, not used
                gnugo=GnuGoConfig(),  # Default, not used
                custom_policy_path=None,
            )

    def get_stockfish_config(self) -> StockfishConfig:
        """Get just the Stockfish configuration."""
        return self.get_config().stockfish


__all__ = [
    "HumanVsAgentConfigForm",
    "HumanVsAgentConfig",
    "StockfishConfig",
    "KataGoConfig",
    "GnuGoConfig",
    "DIFFICULTY_PRESETS",
    "DIFFICULTY_DESCRIPTIONS",
    "STOCKFISH_DIFFICULTY_PRESETS",
    "STOCKFISH_DIFFICULTY_DESCRIPTIONS",
    "KATAGO_DIFFICULTY_PRESETS",
    "KATAGO_DIFFICULTY_DESCRIPTIONS",
    "GNUGO_DIFFICULTY_PRESETS",
    "GNUGO_DIFFICULTY_DESCRIPTIONS",
]
