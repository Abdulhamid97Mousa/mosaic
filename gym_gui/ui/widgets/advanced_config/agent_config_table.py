"""Per-agent policy/worker binding configuration table.

Step 2 of the Unified Flow: Configure which policy and worker controls each agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox

_LOGGER = logging.getLogger(__name__)


# Available actors/policies
AVAILABLE_ACTORS = [
    ("human_keyboard", "Human (Keyboard)", "Human player using keyboard input"),
    ("random", "Random", "Uniform random action selection"),
    # CleanRL policies
    ("cleanrl_ppo", "CleanRL PPO", "PPO policy from CleanRL"),
    ("cleanrl_dqn", "CleanRL DQN", "DQN policy from CleanRL"),
    ("cleanrl_sac", "CleanRL SAC", "SAC policy from CleanRL"),
    # RLlib policies
    ("rllib_ppo", "RLlib PPO", "PPO policy from Ray RLlib"),
    ("rllib_dqn", "RLlib DQN", "DQN policy from Ray RLlib"),
    # XuanCe MARL policies
    ("xuance_mappo", "XuanCe MAPPO", "Multi-Agent PPO from XuanCe"),
    ("xuance_maddpg", "XuanCe MADDPG", "Multi-Agent DDPG from XuanCe"),
    ("xuance_qmix", "XuanCe QMIX", "QMIX from XuanCe"),
    # Other
    ("stockfish", "Stockfish", "Stockfish chess engine (Chess only)"),
    ("llm", "LLM Agent", "Language model decision-maker"),
    ("bdi", "BDI Agent", "Belief-Desire-Intention agent"),
]

# Available workers
AVAILABLE_WORKERS = [
    ("local", "Local", "Run in main process (no worker)"),
    ("cleanrl", "CleanRL", "CleanRL single-agent RL training"),
    ("rllib", "Ray RLlib", "Distributed RL with Ray RLlib"),
    ("xuance", "XuanCe", "Multi-agent RL with XuanCe MARL"),
    ("llm", "LLM", "Language model decision-making"),
    ("jason", "Jason BDI", "AgentSpeak BDI agents"),
]

# Available modes
AVAILABLE_MODES = [
    ("play", "Play", "Interactive play (no training)"),
    ("train", "Train", "Training mode"),
    ("eval", "Evaluate", "Evaluation mode (frozen policy)"),
    ("frozen", "Frozen", "Frozen snapshot (for self-play)"),
]


@dataclass
class AgentRowConfig:
    """Configuration for a single agent row.

    Attributes:
        agent_id: Unique identifier for the agent
        actor_id: Selected actor/policy ID
        worker_id: Selected worker ID
        mode: Selected mode (play/train/eval/frozen)
        config: Additional worker-specific configuration
    """

    agent_id: str
    actor_id: str = "human_keyboard"
    worker_id: str = "local"
    mode: str = "play"
    config: Dict[str, Any] = field(default_factory=dict)


class AgentConfigTable(QtWidgets.QGroupBox):
    """Step 2: Per-agent policy/worker binding configuration.

    Displays a table where each row represents an agent in the environment.
    Users can select which actor/policy, worker, and mode for each agent.

    Signals:
        bindings_changed: Emitted when any binding changes

    Example:
        table = AgentConfigTable()
        table.set_agents(["player_0", "player_1"])
        table.bindings_changed.connect(on_bindings_change)
    """

    # Signals
    bindings_changed = pyqtSignal(dict)  # Dict[agent_id, AgentRowConfig]

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Step 2: Agent Configuration", parent)
        self._agent_ids: List[str] = []
        self._row_widgets: Dict[str, Dict[str, QComboBox | QtWidgets.QLabel]] = {}
        self._configs: Dict[str, AgentRowConfig] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        # Info label
        info = QtWidgets.QLabel(
            "Configure which policy and worker controls each agent. "
            "For multi-agent, you can assign different policies to different agents."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(info)

        # Bulk apply section (for many agents)
        bulk_frame = QtWidgets.QFrame()
        bulk_frame.setStyleSheet("background-color: #f8f8f8; border-radius: 4px; padding: 4px;")
        bulk_layout = QtWidgets.QHBoxLayout(bulk_frame)
        bulk_layout.setContentsMargins(8, 4, 8, 4)

        self._apply_all_check = QtWidgets.QCheckBox("Apply to all agents:")
        bulk_layout.addWidget(self._apply_all_check)

        self._bulk_actor_combo = QtWidgets.QComboBox()
        self._bulk_actor_combo.setMinimumWidth(120)
        for actor_id, label, _ in AVAILABLE_ACTORS:
            self._bulk_actor_combo.addItem(label, actor_id)
        bulk_layout.addWidget(self._bulk_actor_combo)

        self._bulk_worker_combo = QtWidgets.QComboBox()
        self._bulk_worker_combo.setMinimumWidth(100)
        for worker_id, label, _ in AVAILABLE_WORKERS:
            self._bulk_worker_combo.addItem(label, worker_id)
        bulk_layout.addWidget(self._bulk_worker_combo)

        self._bulk_mode_combo = QtWidgets.QComboBox()
        self._bulk_mode_combo.setMinimumWidth(80)
        for mode_id, label, _ in AVAILABLE_MODES:
            self._bulk_mode_combo.addItem(label, mode_id)
        bulk_layout.addWidget(self._bulk_mode_combo)

        self._apply_btn = QtWidgets.QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply_bulk)
        bulk_layout.addWidget(self._apply_btn)

        bulk_layout.addStretch(1)
        layout.addWidget(bulk_frame)

        # Table header
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setSpacing(4)

        headers = [
            ("Agent", 100),
            ("Actor/Policy", 140),
            ("Worker", 120),
            ("Mode", 90),
        ]
        for label, width in headers:
            lbl = QtWidgets.QLabel(f"<b>{label}</b>")
            lbl.setFixedWidth(width)
            header_layout.addWidget(lbl)

        header_layout.addStretch(1)
        layout.addLayout(header_layout)

        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # Scrollable area for agent rows
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setMaximumHeight(200)

        self._rows_container = QtWidgets.QWidget()
        self._rows_layout = QtWidgets.QVBoxLayout(self._rows_container)
        self._rows_layout.setSpacing(4)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.addStretch(1)

        scroll.setWidget(self._rows_container)
        layout.addWidget(scroll)

        # Placeholder when no agents
        self._placeholder = QtWidgets.QLabel(
            "<i>Load an environment to configure agents.</i>"
        )
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; padding: 20px;")
        layout.addWidget(self._placeholder)

    def set_agents(self, agent_ids: List[str]) -> None:
        """Set the list of agents to configure.

        Args:
            agent_ids: List of agent identifiers from the environment.
        """
        # Clear existing rows
        self._clear_rows()

        self._agent_ids = list(agent_ids)
        self._configs = {}

        if not agent_ids:
            self._placeholder.show()
            return

        self._placeholder.hide()

        # Create a row for each agent
        for agent_id in agent_ids:
            self._add_agent_row(agent_id)

        # Emit initial bindings
        self._emit_bindings()

    def _clear_rows(self) -> None:
        """Clear all agent rows."""
        for agent_id, widgets in self._row_widgets.items():
            for widget in widgets.values():
                widget.deleteLater()
        self._row_widgets.clear()

    def _add_agent_row(self, agent_id: str) -> None:
        """Add a configuration row for an agent."""
        row_layout = QtWidgets.QHBoxLayout()
        row_layout.setSpacing(4)

        # Agent label
        agent_label = QtWidgets.QLabel(agent_id)
        agent_label.setFixedWidth(100)
        agent_label.setStyleSheet("font-weight: bold;")
        row_layout.addWidget(agent_label)

        # Actor/Policy combo
        actor_combo = QtWidgets.QComboBox()
        actor_combo.setFixedWidth(140)
        for actor_id, label, tooltip in AVAILABLE_ACTORS:
            actor_combo.addItem(label, actor_id)
            idx = actor_combo.count() - 1
            actor_combo.setItemData(idx, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
        actor_combo.currentIndexChanged.connect(
            lambda _, aid=agent_id: self._on_row_changed(aid)
        )
        row_layout.addWidget(actor_combo)

        # Worker combo
        worker_combo = QtWidgets.QComboBox()
        worker_combo.setFixedWidth(120)
        for worker_id, label, tooltip in AVAILABLE_WORKERS:
            worker_combo.addItem(label, worker_id)
            idx = worker_combo.count() - 1
            worker_combo.setItemData(idx, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
        worker_combo.currentIndexChanged.connect(
            lambda _, aid=agent_id: self._on_row_changed(aid)
        )
        row_layout.addWidget(worker_combo)

        # Mode combo
        mode_combo = QtWidgets.QComboBox()
        mode_combo.setFixedWidth(90)
        for mode_id, label, tooltip in AVAILABLE_MODES:
            mode_combo.addItem(label, mode_id)
            idx = mode_combo.count() - 1
            mode_combo.setItemData(idx, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
        mode_combo.currentIndexChanged.connect(
            lambda _, aid=agent_id: self._on_row_changed(aid)
        )
        row_layout.addWidget(mode_combo)

        row_layout.addStretch(1)

        # Store widgets
        self._row_widgets[agent_id] = {
            "label": agent_label,
            "actor": actor_combo,
            "worker": worker_combo,
            "mode": mode_combo,
        }

        # Create initial config
        self._configs[agent_id] = AgentRowConfig(agent_id=agent_id)

        # Insert before stretch
        insert_idx = self._rows_layout.count() - 1
        self._rows_layout.insertLayout(insert_idx, row_layout)

    def _on_row_changed(self, agent_id: str) -> None:
        """Handle change in a row's selection."""
        widgets = self._row_widgets.get(agent_id)
        if not widgets:
            return

        actor_combo = widgets.get("actor")
        worker_combo = widgets.get("worker")
        mode_combo = widgets.get("mode")

        if not isinstance(actor_combo, QComboBox):
            return
        if not isinstance(worker_combo, QComboBox):
            return
        if not isinstance(mode_combo, QComboBox):
            return

        self._configs[agent_id] = AgentRowConfig(
            agent_id=agent_id,
            actor_id=actor_combo.currentData(),
            worker_id=worker_combo.currentData(),
            mode=mode_combo.currentData(),
        )

        self._emit_bindings()

    def _on_apply_bulk(self) -> None:
        """Apply bulk settings to all agents."""
        if not self._apply_all_check.isChecked():
            return

        actor_id = self._bulk_actor_combo.currentData()
        worker_id = self._bulk_worker_combo.currentData()
        mode = self._bulk_mode_combo.currentData()

        for agent_id, widgets in self._row_widgets.items():
            # Update combos
            actor_combo = widgets.get("actor")
            worker_combo = widgets.get("worker")
            mode_combo = widgets.get("mode")

            if not isinstance(actor_combo, QComboBox):
                continue
            if not isinstance(worker_combo, QComboBox):
                continue
            if not isinstance(mode_combo, QComboBox):
                continue

            # Find and set indices
            for i in range(actor_combo.count()):
                if actor_combo.itemData(i) == actor_id:
                    actor_combo.setCurrentIndex(i)
                    break

            for i in range(worker_combo.count()):
                if worker_combo.itemData(i) == worker_id:
                    worker_combo.setCurrentIndex(i)
                    break

            for i in range(mode_combo.count()):
                if mode_combo.itemData(i) == mode:
                    mode_combo.setCurrentIndex(i)
                    break

            # Update config
            self._configs[agent_id] = AgentRowConfig(
                agent_id=agent_id,
                actor_id=actor_id,
                worker_id=worker_id,
                mode=mode,
            )

        self._emit_bindings()
        _LOGGER.info("Applied bulk config to %d agents", len(self._agent_ids))

    def _emit_bindings(self) -> None:
        """Emit the current bindings."""
        self.bindings_changed.emit(dict(self._configs))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_bindings(self) -> Dict[str, AgentRowConfig]:
        """Get all configured bindings.

        Returns:
            Dictionary mapping agent_id to AgentRowConfig.
        """
        return dict(self._configs)

    def get_config(self, agent_id: str) -> Optional[AgentRowConfig]:
        """Get configuration for a specific agent.

        Args:
            agent_id: The agent identifier.

        Returns:
            AgentRowConfig if found, None otherwise.
        """
        return self._configs.get(agent_id)

    @property
    def agent_ids(self) -> List[str]:
        """Get list of configured agent IDs."""
        return list(self._agent_ids)
