"""Environment selection widget with paradigm auto-detection.

Step 1 of the Unified Flow: Select environment and detect its properties.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal

from gym_gui.core.enums import EnvironmentFamily, GameId, SteppingParadigm
from gym_gui.core.pettingzoo_enums import (
    PettingZooEnvId,
    PettingZooFamily,
    get_api_type,
    get_description,
    get_display_name,
)

_LOGGER = logging.getLogger(__name__)


# Environment family metadata for the dropdown
FAMILY_METADATA = {
    "gymnasium": {
        "label": "Gymnasium (Single-Agent)",
        "paradigm": SteppingParadigm.SINGLE_AGENT,
        "description": "Classic RL environments: CartPole, LunarLander, MuJoCo, etc.",
    },
    "pettingzoo_classic": {
        "label": "PettingZoo Classic (Turn-Based)",
        "paradigm": SteppingParadigm.SEQUENTIAL,
        "description": "Board games: Chess, Go, Connect Four, etc.",
    },
    "pettingzoo_mpe": {
        "label": "PettingZoo MPE (Multi-Agent)",
        "paradigm": SteppingParadigm.SIMULTANEOUS,
        "description": "Multi-Agent Particle Environments for cooperative/competitive tasks.",
    },
    "pettingzoo_sisl": {
        "label": "PettingZoo SISL (Cooperative)",
        "paradigm": SteppingParadigm.SIMULTANEOUS,
        "description": "Stanford SISL cooperative multi-agent benchmark environments.",
    },
    "pettingzoo_butterfly": {
        "label": "PettingZoo Butterfly (Cooperative)",
        "paradigm": SteppingParadigm.SIMULTANEOUS,
        "description": "Challenging Pygame-based cooperative environments by Farama.",
    },
    "pettingzoo_atari": {
        "label": "PettingZoo Atari (Multi-Player)",
        "paradigm": SteppingParadigm.SIMULTANEOUS,
        "description": "Multi-player Atari games.",
    },
    "vizdoom": {
        "label": "ViZDoom",
        "paradigm": SteppingParadigm.SINGLE_AGENT,
        "description": "Doom-based 3D environments for visual RL.",
    },
    "minigrid": {
        "label": "MiniGrid",
        "paradigm": SteppingParadigm.SINGLE_AGENT,
        "description": "Grid-world environments for goal-oriented tasks.",
    },
}

# Environments per family
FAMILY_ENVIRONMENTS = {
    "gymnasium": [
        ("CartPole-v1", "CartPole-v1", "Balance a pole on a cart"),
        ("LunarLander-v3", "LunarLander-v3", "Land a spacecraft on the moon"),
        ("MountainCar-v0", "MountainCar-v0", "Drive a car up a mountain"),
        ("Acrobot-v1", "Acrobot-v1", "Swing up a two-link robot"),
        ("BipedalWalker-v3", "BipedalWalker-v3", "Walk with a bipedal robot"),
        ("HalfCheetah-v5", "HalfCheetah-v5", "Run with a cheetah robot (MuJoCo)"),
    ],
    "pettingzoo_classic": [
        ("chess_v6", "Chess", "Two-player chess"),
        ("connect_four_v3", "Connect Four", "Classic connect four game"),
        ("go_v5", "Go", "Ancient board game"),
        ("tictactoe_v3", "Tic-Tac-Toe", "Simple 3x3 grid game"),
    ],
    # MPE - Multi-Agent Particle Environments
    "pettingzoo_mpe": [
        ("simple_v3", "Simple", "Single agent navigates to landmark"),
        ("simple_adversary_v3", "Simple Adversary", "Good agents vs adversary, split to deceive"),
        ("simple_crypto_v3", "Simple Crypto", "Communication and decryption task"),
        ("simple_push_v3", "Simple Push", "Agent pushes adversary away from landmark"),
        ("simple_reference_v3", "Simple Reference", "Cooperative communication task"),
        ("simple_speaker_listener_v4", "Simple Speaker Listener", "Speaker guides listener to landmark"),
        ("simple_spread_v3", "Simple Spread", "Cooperative navigation to landmarks"),
        ("simple_tag_v3", "Simple Tag", "Predators chase prey agent"),
        ("simple_world_comm_v3", "Simple World Comm", "Communication with food/forest/agents"),
    ],
    # SISL - Stanford Intelligent Systems Laboratory
    "pettingzoo_sisl": [
        ("multiwalker_v9", "Multiwalker", "Bipedal robots carry package together"),
        ("pursuit_v4", "Pursuit", "Pursuers cooperate to catch evaders"),
        ("waterworld_v4", "Waterworld", "Agents gather food, avoid poison"),
    ],
    # Butterfly - Pygame-based cooperative environments
    "pettingzoo_butterfly": [
        ("cooperative_pong_v5", "Cooperative Pong", "Keep ball in play with two paddles"),
        ("knights_archers_zombies_v10", "Knights Archers Zombies", "Defend against zombie waves"),
        ("pistonball_v6", "Pistonball", "Coordinated pistons move ball left"),
    ],
    "pettingzoo_atari": [
        ("pong_v3", "Pong", "Two-player Pong"),
        ("space_invaders_v2", "Space Invaders", "Cooperative Space Invaders"),
        ("tennis_v3", "Tennis", "Two-player Tennis"),
    ],
    "vizdoom": [
        ("ViZDoom-Basic-v0", "Basic", "Shoot a monster"),
        ("ViZDoom-DefendTheCenter-v0", "Defend The Center", "Survive enemy attacks"),
        ("ViZDoom-DeadlyCorridor-v0", "Deadly Corridor", "Navigate a dangerous corridor"),
    ],
    "minigrid": [
        ("MiniGrid-Empty-5x5-v0", "Empty 5x5", "Navigate empty grid"),
        ("MiniGrid-DoorKey-5x5-v0", "DoorKey 5x5", "Find key, open door"),
        ("MiniGrid-LavaGapS5-v0", "Lava Gap S5", "Cross lava gaps"),
    ],
}

# Agent counts per environment
AGENT_COUNTS = {
    # Gymnasium (single-agent)
    "CartPole-v1": ["agent_0"],
    "LunarLander-v3": ["agent_0"],
    "MountainCar-v0": ["agent_0"],
    "Acrobot-v1": ["agent_0"],
    "BipedalWalker-v3": ["agent_0"],
    "HalfCheetah-v5": ["agent_0"],
    # PettingZoo Classic (2 players)
    "chess_v6": ["player_0", "player_1"],
    "connect_four_v3": ["player_0", "player_1"],
    "go_v5": ["black_0", "white_0"],
    "tictactoe_v3": ["player_1", "player_2"],
    # PettingZoo MPE
    "simple_v3": ["agent_0"],  # Single agent
    "simple_adversary_v3": ["adversary_0", "agent_0", "agent_1"],  # 1 adversary, N=2 good agents
    "simple_crypto_v3": ["eve_0", "bob_0", "alice_0"],  # Eve decrypts, Bob receives, Alice sends
    "simple_push_v3": ["adversary_0", "agent_0"],  # 1 adversary, 1 good agent
    "simple_reference_v3": ["agent_0", "agent_1"],  # 2 cooperative agents
    "simple_speaker_listener_v4": ["speaker_0", "listener_0"],  # Speaker and listener
    "simple_spread_v3": ["agent_0", "agent_1", "agent_2"],  # N=3 cooperative agents
    "simple_tag_v3": ["adversary_0", "adversary_1", "adversary_2", "agent_0"],  # 3 predators, 1 prey
    "simple_world_comm_v3": [
        "leadadversary_0", "adversary_0", "adversary_1",
        "agent_0", "agent_1",
    ],  # Lead adversary + 2 adversaries + 2 good agents
    # PettingZoo SISL
    "multiwalker_v9": ["walker_0", "walker_1", "walker_2"],  # 3 bipedal walkers (default)
    "pursuit_v4": [f"pursuer_{i}" for i in range(8)],  # 8 pursuers (default)
    "waterworld_v4": [f"pursuer_{i}" for i in range(5)],  # 5 pursuers (default)
    # PettingZoo Butterfly
    "cooperative_pong_v5": ["paddle_0", "paddle_1"],  # 2 paddles
    "knights_archers_zombies_v10": ["archer_0", "archer_1", "knight_0", "knight_1"],  # 2 archers + 2 knights
    "pistonball_v6": [f"piston_{i}" for i in range(20)],  # 20 pistons (default)
    # PettingZoo Atari
    "pong_v3": ["first_0", "second_0"],
    "space_invaders_v2": ["first_0", "second_0"],
    "tennis_v3": ["first_0", "second_0"],
    # ViZDoom (single-agent)
    "ViZDoom-Basic-v0": ["agent_0"],
    "ViZDoom-DefendTheCenter-v0": ["agent_0"],
    "ViZDoom-DeadlyCorridor-v0": ["agent_0"],
    # MiniGrid (single-agent)
    "MiniGrid-Empty-5x5-v0": ["agent_0"],
    "MiniGrid-DoorKey-5x5-v0": ["agent_0"],
    "MiniGrid-LavaGapS5-v0": ["agent_0"],
}


class EnvironmentSelector(QtWidgets.QGroupBox):
    """Step 1: Environment selection with paradigm auto-detection.

    Signals:
        environment_changed: Emitted when environment selection changes
        paradigm_detected: Emitted when paradigm is determined
        agents_detected: Emitted with list of agent IDs

    Example:
        selector = EnvironmentSelector()
        selector.environment_changed.connect(on_env_change)
        selector.agents_detected.connect(on_agents_detected)
    """

    # Signals
    environment_changed = pyqtSignal(str)  # env_id
    paradigm_detected = pyqtSignal(object)  # SteppingParadigm
    agents_detected = pyqtSignal(list)  # List[str] agent IDs

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Step 1: Environment Selection", parent)
        self._selected_family: Optional[str] = None
        self._selected_env_id: Optional[str] = None
        self._current_paradigm: SteppingParadigm = SteppingParadigm.SINGLE_AGENT
        self._current_agents: List[str] = []
        self._build_ui()
        self._connect_signals()
        self._populate_families()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        # Family selection
        family_layout = QtWidgets.QHBoxLayout()
        family_layout.addWidget(QtWidgets.QLabel("Family:"))
        self._family_combo = QtWidgets.QComboBox()
        self._family_combo.setMinimumWidth(200)
        family_layout.addWidget(self._family_combo, 1)
        layout.addLayout(family_layout)

        # Environment selection
        env_layout = QtWidgets.QHBoxLayout()
        env_layout.addWidget(QtWidgets.QLabel("Environment:"))
        self._env_combo = QtWidgets.QComboBox()
        self._env_combo.setMinimumWidth(200)
        env_layout.addWidget(self._env_combo, 1)
        layout.addLayout(env_layout)

        # Seed selection
        seed_layout = QtWidgets.QHBoxLayout()
        seed_layout.addWidget(QtWidgets.QLabel("Seed:"))
        self._seed_spin = QtWidgets.QSpinBox()
        self._seed_spin.setRange(1, 10_000_000)
        self._seed_spin.setValue(42)
        seed_layout.addWidget(self._seed_spin)
        seed_layout.addStretch(1)
        layout.addLayout(seed_layout)

        # Info panel (paradigm, agents, action space)
        self._info_label = QtWidgets.QLabel()
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet(
            "background-color: #f5f5f5; padding: 8px; border-radius: 4px; "
            "font-size: 11px; color: #555;"
        )
        layout.addWidget(self._info_label)

        # Load button
        self._load_btn = QtWidgets.QPushButton("Load Environment")
        self._load_btn.setEnabled(False)
        layout.addWidget(self._load_btn)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)
        self._env_combo.currentIndexChanged.connect(self._on_env_changed)
        self._load_btn.clicked.connect(self._on_load_clicked)

    def _populate_families(self) -> None:
        """Populate family dropdown."""
        self._family_combo.clear()
        for family_id, meta in FAMILY_METADATA.items():
            self._family_combo.addItem(meta["label"], family_id)

        if self._family_combo.count() > 0:
            self._on_family_changed(0)

    def _on_family_changed(self, index: int) -> None:
        """Handle family selection change."""
        family_id = self._family_combo.currentData()
        if not family_id:
            return

        self._selected_family = family_id
        self._env_combo.clear()

        # Populate environments for this family
        envs = FAMILY_ENVIRONMENTS.get(family_id, [])
        for env_id, display_name, description in envs:
            self._env_combo.addItem(f"{display_name}", env_id)
            # Store description for tooltip
            idx = self._env_combo.count() - 1
            self._env_combo.setItemData(idx, description, role=256)  # Custom role

        if self._env_combo.count() > 0:
            self._on_env_changed(0)

    def _on_env_changed(self, index: int) -> None:
        """Handle environment selection change."""
        env_id = self._env_combo.currentData()
        if not env_id:
            self._update_info_panel(None)
            self._load_btn.setEnabled(False)
            return

        self._selected_env_id = env_id
        family_id = self._selected_family

        # Get paradigm from family
        if family_id is not None:
            meta = FAMILY_METADATA.get(family_id, {})
            paradigm = meta.get("paradigm")
            self._current_paradigm = paradigm if isinstance(paradigm, SteppingParadigm) else SteppingParadigm.SINGLE_AGENT
        else:
            self._current_paradigm = SteppingParadigm.SINGLE_AGENT

        # Get agents
        self._current_agents = AGENT_COUNTS.get(env_id, ["agent_0"])

        # Update UI
        self._update_info_panel(env_id)
        self._load_btn.setEnabled(True)

        # Emit signals
        self.paradigm_detected.emit(self._current_paradigm)
        self.agents_detected.emit(self._current_agents)

    def _update_info_panel(self, env_id: Optional[str]) -> None:
        """Update the info panel with environment details."""
        if not env_id:
            self._info_label.setText("<i>Select an environment to see details.</i>")
            return

        paradigm = self._current_paradigm
        agents = self._current_agents
        num_agents = len(agents)

        # Get description from tooltip
        idx = self._env_combo.currentIndex()
        description = self._env_combo.itemData(idx, role=256) or ""

        # Format agent list (truncate if too many)
        if num_agents <= 5:
            agents_display = ", ".join(agents)
        else:
            agents_display = f"{', '.join(agents[:3])}, ... , {agents[-1]}"

        # Determine agent type label
        if num_agents == 1:
            agent_label = "Single-Agent"
        else:
            agent_label = f"Multi-Agent ({num_agents} agents)"

        info_text = (
            f"<b>Environment:</b> {env_id}<br>"
            f"<b>Type:</b> <span style='color: #2196F3; font-weight: bold;'>{agent_label}</span><br>"
            f"<b>Paradigm:</b> {paradigm.value.replace('_', ' ').title()}<br>"
            f"<b>Agents:</b> {agents_display}<br>"
            f"<b>Description:</b> {description}"
        )
        self._info_label.setText(info_text)

    def _on_load_clicked(self) -> None:
        """Handle load button click."""
        if self._selected_env_id:
            self.environment_changed.emit(self._selected_env_id)
            _LOGGER.info(
                "Environment selected: %s (paradigm=%s, agents=%s)",
                self._selected_env_id,
                self._current_paradigm.value,
                self._current_agents,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def selected_env_id(self) -> Optional[str]:
        """Get the currently selected environment ID."""
        return self._selected_env_id

    @property
    def paradigm(self) -> SteppingParadigm:
        """Get the detected stepping paradigm."""
        return self._current_paradigm

    @property
    def agents(self) -> List[str]:
        """Get the list of agent IDs."""
        return list(self._current_agents)

    @property
    def seed(self) -> int:
        """Get the current seed value."""
        return self._seed_spin.value()

    def is_loaded(self) -> bool:
        """Check if an environment is selected."""
        return self._selected_env_id is not None
