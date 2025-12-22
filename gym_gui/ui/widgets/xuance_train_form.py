"""XuanCe worker training form.

Provides a configuration dialog for XuanCe RL training runs with:
- Backend selection (PyTorch, TensorFlow, MindSpore)
- Single-Agent / Multi-Agent paradigm tabs
- Dynamic algorithm filtering based on backend + paradigm
- Dynamic algorithm parameters based on selected algorithm
- Environment family and ID selection
- Core training parameters
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qtpy import QtCore, QtGui, QtWidgets

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_UI_TRAIN_FORM_INFO,
    LOG_UI_TRAIN_FORM_ERROR,
)

from xuance_worker import (
    Backend,
    Paradigm,
    get_algorithm_choices,
    get_algorithms_by_category,
    get_backend_summary,
)

from gym_gui.telemetry.semconv import VIDEO_MODE_DESCRIPTORS, VideoModes
from gym_gui.fastlane.worker_helpers import apply_fastlane_environment


_LOGGER = logging.getLogger("gym_gui.ui.xuance_train_form")
REPO_ROOT = Path(__file__).resolve().parents[3]


# --- Schema Loading ---
def _load_xuance_schemas() -> Tuple[Dict[str, Any], Optional[str]]:
    """Load algorithm schemas from metadata/xuance/*/schemas.json."""
    schema_root = REPO_ROOT / "metadata" / "xuance"
    if not schema_root.exists():
        return {}, None

    candidates: List[Tuple[str, Path]] = []
    for entry in schema_root.iterdir():
        if not entry.is_dir():
            continue
        schema_file = entry / "schemas.json"
        if schema_file.exists():
            candidates.append((entry.name, schema_file))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        fallback = schema_root / "schemas.json"
        if fallback.exists():
            candidates.append(("latest", fallback))

    for _, schema_file in candidates:
        try:
            data = json.loads(schema_file.read_text())
        except Exception:
            continue
        return data.get("algorithms", {}), data.get("xuance_version")

    return {}, None


_XUANCE_SCHEMAS, _XUANCE_SCHEMA_VERSION = _load_xuance_schemas()

# Fields to exclude from dynamic parameter UI (handled separately)
_SCHEMA_EXCLUDED_FIELDS: set[str] = {
    "seed",
    "running_steps",
    "parallels",
    "device",
    "gamma",  # Show gamma as it's important
}


# XuanCe environment families and their example environments
_XUANCE_ENVIRONMENT_FAMILIES: Dict[str, List[Tuple[str, str]]] = {
    "classic_control": [
        ("CartPole-v1", "CartPole-v1"),
        ("Pendulum-v1", "Pendulum-v1"),
        ("Acrobot-v1", "Acrobot-v1"),
        ("MountainCar-v0", "MountainCar-v0"),
        ("MountainCarContinuous-v0", "MountainCarContinuous-v0"),
    ],
    "box2d": [
        ("LunarLander-v2", "LunarLander-v2"),
        ("LunarLanderContinuous-v2", "LunarLanderContinuous-v2"),
        ("BipedalWalker-v3", "BipedalWalker-v3"),
        ("CarRacing-v2", "CarRacing-v2"),
    ],
    "mujoco": [
        ("HalfCheetah-v4", "HalfCheetah-v4"),
        ("Ant-v4", "Ant-v4"),
        ("Hopper-v4", "Hopper-v4"),
        ("Walker2d-v4", "Walker2d-v4"),
        ("Humanoid-v4", "Humanoid-v4"),
        ("Swimmer-v4", "Swimmer-v4"),
        ("Reacher-v4", "Reacher-v4"),
        ("InvertedPendulum-v4", "InvertedPendulum-v4"),
        ("InvertedDoublePendulum-v4", "InvertedDoublePendulum-v4"),
    ],
    "atari": [
        ("Pong-v5", "Pong-v5"),
        ("Breakout-v5", "Breakout-v5"),
        ("SpaceInvaders-v5", "SpaceInvaders-v5"),
        ("Qbert-v5", "Qbert-v5"),
        ("Seaquest-v5", "Seaquest-v5"),
        ("BeamRider-v5", "BeamRider-v5"),
    ],
    "mpe": [
        ("simple_spread_v3", "simple_spread_v3"),
        ("simple_adversary_v3", "simple_adversary_v3"),
        ("simple_tag_v3", "simple_tag_v3"),
        ("simple_push_v3", "simple_push_v3"),
        ("simple_reference_v3", "simple_reference_v3"),
        ("simple_speaker_listener_v4", "simple_speaker_listener_v4"),
    ],
    "smac": [
        ("3m", "3m"),
        ("8m", "8m"),
        ("2s3z", "2s3z"),
        ("3s5z", "3s5z"),
        ("1c3s5z", "1c3s5z"),
        ("corridor", "corridor"),
        ("27m_vs_30m", "27m_vs_30m"),
    ],
    "football": [
        ("academy_3_vs_1_with_keeper", "academy_3_vs_1_with_keeper"),
        ("academy_counterattack_easy", "academy_counterattack_easy"),
        ("11_vs_11_kaggle", "11_vs_11_kaggle"),
    ],
    # Advanced RL benchmarks (single-agent)
    "minigrid": [
        # Empty rooms
        ("MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-5x5-v0"),
        ("MiniGrid-Empty-Random-5x5-v0", "MiniGrid-Empty-Random-5x5-v0"),
        ("MiniGrid-Empty-6x6-v0", "MiniGrid-Empty-6x6-v0"),
        ("MiniGrid-Empty-Random-6x6-v0", "MiniGrid-Empty-Random-6x6-v0"),
        ("MiniGrid-Empty-8x8-v0", "MiniGrid-Empty-8x8-v0"),
        ("MiniGrid-Empty-16x16-v0", "MiniGrid-Empty-16x16-v0"),
        # DoorKey
        ("MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-5x5-v0"),
        ("MiniGrid-DoorKey-6x6-v0", "MiniGrid-DoorKey-6x6-v0"),
        ("MiniGrid-DoorKey-8x8-v0", "MiniGrid-DoorKey-8x8-v0"),
        ("MiniGrid-DoorKey-16x16-v0", "MiniGrid-DoorKey-16x16-v0"),
        # LavaGap
        ("MiniGrid-LavaGapS5-v0", "MiniGrid-LavaGapS5-v0"),
        ("MiniGrid-LavaGapS6-v0", "MiniGrid-LavaGapS6-v0"),
        ("MiniGrid-LavaGapS7-v0", "MiniGrid-LavaGapS7-v0"),
        # Dynamic Obstacles
        ("MiniGrid-Dynamic-Obstacles-5x5-v0", "MiniGrid-Dynamic-Obstacles-5x5-v0"),
        ("MiniGrid-Dynamic-Obstacles-Random-5x5-v0", "MiniGrid-Dynamic-Obstacles-Random-5x5-v0"),
        ("MiniGrid-Dynamic-Obstacles-6x6-v0", "MiniGrid-Dynamic-Obstacles-6x6-v0"),
        ("MiniGrid-Dynamic-Obstacles-Random-6x6-v0", "MiniGrid-Dynamic-Obstacles-Random-6x6-v0"),
        ("MiniGrid-Dynamic-Obstacles-8x8-v0", "MiniGrid-Dynamic-Obstacles-8x8-v0"),
        ("MiniGrid-Dynamic-Obstacles-16x16-v0", "MiniGrid-Dynamic-Obstacles-16x16-v0"),
        # MultiRoom
        ("MiniGrid-MultiRoom-N2-S4-v0", "MiniGrid-MultiRoom-N2-S4-v0"),
        ("MiniGrid-MultiRoom-N4-S5-v0", "MiniGrid-MultiRoom-N4-S5-v0"),
        ("MiniGrid-MultiRoom-N6-v0", "MiniGrid-MultiRoom-N6-v0"),
        # Obstructed Maze
        ("MiniGrid-ObstructedMaze-1Dlhb-v1", "MiniGrid-ObstructedMaze-1Dlhb-v1"),
        ("MiniGrid-ObstructedMaze-Full-v1", "MiniGrid-ObstructedMaze-Full-v1"),
        # Lava Crossing
        ("MiniGrid-LavaCrossingS9N1-v0", "MiniGrid-LavaCrossingS9N1-v0"),
        ("MiniGrid-LavaCrossingS9N2-v0", "MiniGrid-LavaCrossingS9N2-v0"),
        ("MiniGrid-LavaCrossingS9N3-v0", "MiniGrid-LavaCrossingS9N3-v0"),
        ("MiniGrid-LavaCrossingS11N5-v0", "MiniGrid-LavaCrossingS11N5-v0"),
        # Simple Crossing
        ("MiniGrid-SimpleCrossingS9N1-v0", "MiniGrid-SimpleCrossingS9N1-v0"),
        ("MiniGrid-SimpleCrossingS9N2-v0", "MiniGrid-SimpleCrossingS9N2-v0"),
        ("MiniGrid-SimpleCrossingS9N3-v0", "MiniGrid-SimpleCrossingS9N3-v0"),
        ("MiniGrid-SimpleCrossingS11N5-v0", "MiniGrid-SimpleCrossingS11N5-v0"),
        # Other
        ("MiniGrid-BlockedUnlockPickup-v0", "MiniGrid-BlockedUnlockPickup-v0"),
        ("MiniGrid-RedBlueDoors-6x6-v0", "MiniGrid-RedBlueDoors-6x6-v0"),
        ("MiniGrid-RedBlueDoors-8x8-v0", "MiniGrid-RedBlueDoors-8x8-v0"),
    ],
    "vizdoom": [
        ("ViZDoom-Basic-v0", "ViZDoom-Basic-v0"),
        ("ViZDoom-DeadlyCorridor-v0", "ViZDoom-DeadlyCorridor-v0"),
        ("ViZDoom-DefendTheCenter-v0", "ViZDoom-DefendTheCenter-v0"),
        ("ViZDoom-DefendTheLine-v0", "ViZDoom-DefendTheLine-v0"),
        ("ViZDoom-HealthGathering-v0", "ViZDoom-HealthGathering-v0"),
        ("ViZDoom-HealthGatheringSupreme-v0", "ViZDoom-HealthGatheringSupreme-v0"),
        ("ViZDoom-MyWayHome-v0", "ViZDoom-MyWayHome-v0"),
        ("ViZDoom-PredictPosition-v0", "ViZDoom-PredictPosition-v0"),
        ("ViZDoom-TakeCover-v0", "ViZDoom-TakeCover-v0"),
        ("ViZDoom-Deathmatch-v0", "ViZDoom-Deathmatch-v0"),
    ],
    "minihack": [
        # Navigation
        ("MiniHack-Room-5x5-v0", "MiniHack-Room-5x5-v0"),
        ("MiniHack-Room-15x15-v0", "MiniHack-Room-15x15-v0"),
        ("MiniHack-Corridor-R2-v0", "MiniHack-Corridor-R2-v0"),
        ("MiniHack-Corridor-R3-v0", "MiniHack-Corridor-R3-v0"),
        ("MiniHack-Corridor-R5-v0", "MiniHack-Corridor-R5-v0"),
        ("MiniHack-MazeWalk-9x9-v0", "MiniHack-MazeWalk-9x9-v0"),
        ("MiniHack-MazeWalk-15x15-v0", "MiniHack-MazeWalk-15x15-v0"),
        ("MiniHack-MazeWalk-45x19-v0", "MiniHack-MazeWalk-45x19-v0"),
        ("MiniHack-River-v0", "MiniHack-River-v0"),
        ("MiniHack-River-Narrow-v0", "MiniHack-River-Narrow-v0"),
        # Skills
        ("MiniHack-Eat-v0", "MiniHack-Eat-v0"),
        ("MiniHack-Wear-v0", "MiniHack-Wear-v0"),
        ("MiniHack-Wield-v0", "MiniHack-Wield-v0"),
        ("MiniHack-Zap-v0", "MiniHack-Zap-v0"),
        ("MiniHack-Read-v0", "MiniHack-Read-v0"),
        ("MiniHack-Quaff-v0", "MiniHack-Quaff-v0"),
        ("MiniHack-PutOn-v0", "MiniHack-PutOn-v0"),
        ("MiniHack-LavaCross-v0", "MiniHack-LavaCross-v0"),
        ("MiniHack-WoD-Easy-v0", "MiniHack-WoD-Easy-v0"),
        ("MiniHack-WoD-Medium-v0", "MiniHack-WoD-Medium-v0"),
        ("MiniHack-WoD-Hard-v0", "MiniHack-WoD-Hard-v0"),
        # Exploration
        ("MiniHack-ExploreMaze-Easy-v0", "MiniHack-ExploreMaze-Easy-v0"),
        ("MiniHack-ExploreMaze-Hard-v0", "MiniHack-ExploreMaze-Hard-v0"),
        ("MiniHack-HideNSeek-v0", "MiniHack-HideNSeek-v0"),
        ("MiniHack-Memento-F2-v0", "MiniHack-Memento-F2-v0"),
        ("MiniHack-Memento-F4-v0", "MiniHack-Memento-F4-v0"),
    ],
    "nethack": [
        ("NetHackChallenge-v0", "NetHackChallenge-v0"),
        ("NetHackScore-v0", "NetHackScore-v0"),
        ("NetHackStaircase-v0", "NetHackStaircase-v0"),
        ("NetHackStaircasePet-v0", "NetHackStaircasePet-v0"),
        ("NetHackOracle-v0", "NetHackOracle-v0"),
        ("NetHackGold-v0", "NetHackGold-v0"),
        ("NetHackEat-v0", "NetHackEat-v0"),
        ("NetHackScout-v0", "NetHackScout-v0"),
    ],
    "crafter": [
        ("CrafterReward-v1", "CrafterReward-v1"),
        ("CrafterNoReward-v1", "CrafterNoReward-v1"),
    ],
    "procgen": [
        ("BigFish", "procgen:procgen-bigfish-v0"),
        ("BossFight", "procgen:procgen-bossfight-v0"),
        ("CaveFlyer", "procgen:procgen-caveflyer-v0"),
        ("Chaser", "procgen:procgen-chaser-v0"),
        ("Climber", "procgen:procgen-climber-v0"),
        ("CoinRun", "procgen:procgen-coinrun-v0"),
        ("Dodgeball", "procgen:procgen-dodgeball-v0"),
        ("FruitBot", "procgen:procgen-fruitbot-v0"),
        ("Heist", "procgen:procgen-heist-v0"),
        ("Jumper", "procgen:procgen-jumper-v0"),
        ("Leaper", "procgen:procgen-leaper-v0"),
        ("Maze", "procgen:procgen-maze-v0"),
        ("Miner", "procgen:procgen-miner-v0"),
        ("Ninja", "procgen:procgen-ninja-v0"),
        ("Plunder", "procgen:procgen-plunder-v0"),
        ("StarPilot", "procgen:procgen-starpilot-v0"),
    ],
}

# Environment families for each paradigm
_SINGLE_AGENT_FAMILIES = [
    "classic_control", "box2d", "mujoco", "atari",
    "minigrid", "vizdoom", "minihack", "nethack", "crafter", "procgen",
]
_MULTI_AGENT_FAMILIES = ["mpe", "smac", "football"]


def _generate_run_id(prefix: str, method: str) -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = method.replace("_", "-").lower()
    short_uuid = str(uuid.uuid4())[:6]
    return f"{prefix}-{slug}-{timestamp}-{short_uuid}"


@dataclass(frozen=True)
class _FormState:
    """Captured state from the form."""
    backend: str
    paradigm: str
    method: str
    env: str
    env_id: str
    running_steps: int
    seed: Optional[int]
    device: str
    parallels: int
    test_mode: bool
    benchmark_mode: bool
    worker_id: Optional[str]
    notes: Optional[str]
    algo_params: Dict[str, Any] = field(default_factory=dict)
    # FastLane settings
    fastlane_enabled: bool = False
    fastlane_only: bool = True
    fastlane_slot: int = 0
    fastlane_video_mode: str = "single"
    fastlane_grid_limit: int = 4


class XuanCeTrainForm(QtWidgets.QDialog, LogConstantMixin):
    """Training configuration dialog for XuanCe worker."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        default_game: Optional[Any] = None,  # GameId from form factory
        default_env_id: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._logger = _LOGGER
        self.setWindowTitle("XuanCe Training Configuration")
        self.setModal(True)
        self.resize(900, 700)

        self._last_config: Optional[Dict[str, Any]] = None
        self._algo_param_inputs: Dict[str, QtWidgets.QWidget] = {}

        # Detect available GPUs
        self._gpu_count, self._gpu_name = self._detect_gpus()

        self._setup_ui(default_env_id)

        self.log_constant(
            LOG_UI_TRAIN_FORM_INFO,
            message="XuanCeTrainForm opened",
            extra={"default_env_id": default_env_id},
        )

    def _detect_gpus(self) -> Tuple[int, str]:
        """Detect available GPUs on the system.

        Returns:
            Tuple of (gpu_count, gpu_name). gpu_name is the name of the first GPU.
        """
        # Try torch first (most reliable for XuanCe since it's the primary backend)
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0) if count > 0 else ""
                return count, name
        except ImportError:
            pass

        # Fallback: check nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpus = [g.strip() for g in result.stdout.strip().split("\n") if g.strip()]
                if gpus:
                    return len(gpus), gpus[0]
        except Exception:
            pass

        return 0, ""

    def _setup_ui(self, default_env_id: Optional[str]) -> None:
        """Set up the form UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Add scroll area for content
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout.addWidget(scroll, 1)

        form_panel = QtWidgets.QWidget(scroll)
        form_layout = QtWidgets.QVBoxLayout(form_panel)
        form_layout.setSpacing(12)
        form_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(form_panel)

        # Header with backend summary
        self._setup_header(form_layout)

        # Main content with tabs
        self._setup_tabs(form_layout)

        # Algorithm parameters (dynamic)
        self._setup_algo_params(form_layout)

        # Training parameters
        self._setup_training_params(form_layout)

        # Notes section
        self._setup_notes(form_layout)

        # Analytics section
        self._setup_analytics(form_layout)

        # FastLane section
        self._setup_fastlane_section(form_layout)

        form_layout.addStretch(1)

        # Buttons (outside scroll area)
        self._setup_buttons(layout)

        # Initialize form state
        self._on_backend_changed(self._backend_combo.currentIndex())

    def _setup_header(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up header with backend selection and summary."""
        header = QtWidgets.QWidget(self)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Backend selection
        backend_label = QtWidgets.QLabel("Backend:", header)
        self._backend_combo = QtWidgets.QComboBox(header)
        self._backend_combo.addItem("PyTorch", "torch")
        self._backend_combo.addItem("TensorFlow", "tensorflow")
        self._backend_combo.addItem("MindSpore", "mindspore")
        self._backend_combo.setToolTip(
            "Select the deep learning backend.\n"
            "PyTorch has the most algorithms (50).\n"
            "TensorFlow and MindSpore have 40 algorithms each."
        )
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)

        header_layout.addWidget(backend_label)
        header_layout.addWidget(self._backend_combo)
        header_layout.addStretch()

        # Algorithm count label
        self._algo_count_label = QtWidgets.QLabel("", header)
        self._algo_count_label.setStyleSheet("color: #666666; font-size: 11px;")
        header_layout.addWidget(self._algo_count_label)

        layout.addWidget(header)

    def _setup_tabs(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up Single-Agent / Multi-Agent tabs."""
        self._paradigm_tabs = QtWidgets.QTabWidget(self)

        # Single-Agent tab
        sa_tab = QtWidgets.QWidget()
        sa_layout = QtWidgets.QVBoxLayout(sa_tab)
        self._sa_algo_combo, self._sa_env_family_combo, self._sa_env_combo = (
            self._create_paradigm_controls(sa_layout, "single_agent")
        )
        self._paradigm_tabs.addTab(sa_tab, "Single-Agent")

        # Multi-Agent tab
        ma_tab = QtWidgets.QWidget()
        ma_layout = QtWidgets.QVBoxLayout(ma_tab)
        self._ma_algo_combo, self._ma_env_family_combo, self._ma_env_combo = (
            self._create_paradigm_controls(ma_layout, "multi_agent")
        )
        self._paradigm_tabs.addTab(ma_tab, "Multi-Agent")

        self._paradigm_tabs.currentChanged.connect(self._on_paradigm_changed)

        layout.addWidget(self._paradigm_tabs)

    def _create_paradigm_controls(
        self,
        layout: QtWidgets.QVBoxLayout,
        paradigm: str,
    ) -> Tuple[QtWidgets.QComboBox, QtWidgets.QComboBox, QtWidgets.QComboBox]:
        """Create algorithm and environment controls for a paradigm tab."""
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        # Algorithm selection
        algo_label = QtWidgets.QLabel("Algorithm:", self)
        algo_combo = QtWidgets.QComboBox(self)
        algo_combo.setMaxVisibleItems(15)
        algo_combo.setToolTip("Select the RL algorithm")
        # Connect algorithm change to update parameters
        algo_combo.currentIndexChanged.connect(
            lambda idx, c=algo_combo: self._on_algorithm_changed(c)
        )
        grid.addWidget(algo_label, 0, 0)
        grid.addWidget(algo_combo, 0, 1)

        # Environment family
        env_family_label = QtWidgets.QLabel("Environment Family:", self)
        env_family_combo = QtWidgets.QComboBox(self)
        env_family_combo.setToolTip("Select the environment family")
        families = _SINGLE_AGENT_FAMILIES if paradigm == "single_agent" else _MULTI_AGENT_FAMILIES
        for family in families:
            env_family_combo.addItem(family.replace("_", " ").title(), family)
        env_family_combo.currentIndexChanged.connect(
            lambda idx, c=env_family_combo, e=None: self._on_env_family_changed(c, paradigm)
        )
        grid.addWidget(env_family_label, 1, 0)
        grid.addWidget(env_family_combo, 1, 1)

        # Environment ID
        env_id_label = QtWidgets.QLabel("Environment ID:", self)
        env_combo = QtWidgets.QComboBox(self)
        env_combo.setMaxVisibleItems(15)
        env_combo.setToolTip("Select the specific environment")
        grid.addWidget(env_id_label, 2, 0)
        grid.addWidget(env_combo, 2, 1)

        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addStretch()

        # Initialize environment combo
        if env_family_combo.count() > 0:
            self._populate_env_combo(env_combo, env_family_combo.currentData())

        return algo_combo, env_family_combo, env_combo

    def _setup_algo_params(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up algorithm parameters section (dynamic)."""
        self._algo_param_group = QtWidgets.QGroupBox("Algorithm Parameters", self)
        self._algo_param_layout = QtWidgets.QGridLayout(self._algo_param_group)
        self._algo_param_layout.setContentsMargins(12, 12, 12, 12)
        self._algo_param_layout.setHorizontalSpacing(16)
        self._algo_param_layout.setVerticalSpacing(10)

        # Placeholder label
        self._algo_param_placeholder = QtWidgets.QLabel(
            "Select an algorithm to see its parameters.", self
        )
        self._algo_param_placeholder.setStyleSheet("color: #888888; font-style: italic;")
        self._algo_param_layout.addWidget(self._algo_param_placeholder, 0, 0)

        layout.addWidget(self._algo_param_group)

    def _setup_training_params(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up training parameters section."""
        params_group = QtWidgets.QGroupBox("Training Parameters", self)
        params_layout = QtWidgets.QGridLayout(params_group)
        params_layout.setHorizontalSpacing(16)
        params_layout.setVerticalSpacing(8)

        # Running steps
        steps_label = QtWidgets.QLabel("Training Steps:", self)
        self._steps_spin = QtWidgets.QSpinBox(self)
        self._steps_spin.setRange(1000, 100_000_000)
        self._steps_spin.setSingleStep(10000)
        self._steps_spin.setValue(1_000_000)
        self._steps_spin.setToolTip("Total training steps")
        params_layout.addWidget(steps_label, 0, 0)
        params_layout.addWidget(self._steps_spin, 0, 1)

        # Seed
        seed_label = QtWidgets.QLabel("Seed:", self)
        self._seed_spin = QtWidgets.QSpinBox(self)
        self._seed_spin.setRange(0, 1_000_000_000)
        self._seed_spin.setValue(1)
        self._seed_spin.setSpecialValueText("Random")
        self._seed_spin.setToolTip("Random seed (0 = random)")
        params_layout.addWidget(seed_label, 0, 2)
        params_layout.addWidget(self._seed_spin, 0, 3)

        # Device - auto-detect and select GPU if available
        device_label = QtWidgets.QLabel("Device:", self)
        self._device_combo = QtWidgets.QComboBox(self)
        self._device_combo.addItem("CPU", "cpu")
        self._device_combo.addItem("CUDA (GPU)", "cuda")
        self._device_combo.addItem("CUDA:0", "cuda:0")
        self._device_combo.addItem("CUDA:1", "cuda:1")

        # Auto-select GPU if available, and update tooltip
        if self._gpu_count > 0:
            # Select cuda:0 by default when GPU is detected
            self._device_combo.setCurrentIndex(2)  # cuda:0
            self._device_combo.setToolTip(
                f"Computing device\n✓ Detected: {self._gpu_name} ({self._gpu_count} GPU{'s' if self._gpu_count > 1 else ''} available)"
            )
        else:
            self._device_combo.setToolTip("Computing device\nNo GPU detected - using CPU")

        params_layout.addWidget(device_label, 1, 0)
        params_layout.addWidget(self._device_combo, 1, 1)

        # GPU info indicator
        if self._gpu_count > 0:
            gpu_info = QtWidgets.QLabel(f"✓ {self._gpu_name}", self)
            gpu_info.setStyleSheet("color: green; font-size: 10px;")
        else:
            gpu_info = QtWidgets.QLabel("No GPU detected", self)
            gpu_info.setStyleSheet("color: #888; font-size: 10px;")
        params_layout.addWidget(gpu_info, 1, 2, 1, 2)

        # Parallels
        parallels_label = QtWidgets.QLabel("Parallel Envs:", self)
        self._parallels_spin = QtWidgets.QSpinBox(self)
        self._parallels_spin.setRange(1, 256)
        self._parallels_spin.setValue(8)
        self._parallels_spin.setToolTip("Number of parallel environments")
        params_layout.addWidget(parallels_label, 1, 2)
        params_layout.addWidget(self._parallels_spin, 1, 3)

        # Benchmark mode checkbox
        self._benchmark_checkbox = QtWidgets.QCheckBox("Benchmark Mode", self)
        self._benchmark_checkbox.setToolTip(
            "Enable benchmark mode (training with periodic evaluation)\n"
            "Saves best model during training"
        )
        params_layout.addWidget(self._benchmark_checkbox, 2, 0, 1, 2)

        # Test mode checkbox
        self._test_checkbox = QtWidgets.QCheckBox("Test Mode (Evaluate Only)", self)
        self._test_checkbox.setToolTip("Load and evaluate a trained model instead of training")
        params_layout.addWidget(self._test_checkbox, 2, 2, 1, 2)

        # Worker ID
        worker_label = QtWidgets.QLabel("Worker ID:", self)
        self._worker_id_input = QtWidgets.QLineEdit(self)
        self._worker_id_input.setPlaceholderText("Optional worker identifier")
        self._worker_id_input.setToolTip("Optional identifier for this worker")
        params_layout.addWidget(worker_label, 3, 0)
        params_layout.addWidget(self._worker_id_input, 3, 1, 1, 3)

        layout.addWidget(params_group)

    def _setup_notes(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up notes section."""
        notes_group = QtWidgets.QGroupBox("Notes", self)
        notes_layout = QtWidgets.QVBoxLayout(notes_group)

        self._notes_edit = QtWidgets.QPlainTextEdit(self)
        self._notes_edit.setPlaceholderText("Optional notes for this training run...")
        self._notes_edit.setMaximumHeight(80)
        notes_layout.addWidget(self._notes_edit)

        layout.addWidget(notes_group)

    def _setup_analytics(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up analytics & tracking section."""
        group = QtWidgets.QGroupBox("Analytics & Tracking", self)
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(8, 8, 8, 8)
        group_layout.setSpacing(6)

        # Hint label
        hint_label = QtWidgets.QLabel(
            "Select analytics to export during and after training.",
            group,
        )
        hint_label.setStyleSheet("color: #777777; font-size: 11px;")
        hint_label.setWordWrap(True)
        group_layout.addWidget(hint_label)

        # Checkbox row
        checkbox_layout = QtWidgets.QHBoxLayout()

        self._tensorboard_checkbox = QtWidgets.QCheckBox("Export TensorBoard", group)
        self._tensorboard_checkbox.setChecked(True)
        self._tensorboard_checkbox.setToolTip(
            "Write TensorBoard event files to var/trainer/runs/<run_id>/tensorboard"
        )
        checkbox_layout.addWidget(self._tensorboard_checkbox)

        self._wandb_checkbox = QtWidgets.QCheckBox("Export WandB", group)
        self._wandb_checkbox.setChecked(False)
        self._wandb_checkbox.setToolTip("Requires wandb login on the trainer host")
        self._wandb_checkbox.toggled.connect(self._on_wandb_toggled)
        checkbox_layout.addWidget(self._wandb_checkbox)

        checkbox_layout.addStretch(1)
        group_layout.addLayout(checkbox_layout)

        # WandB configuration section
        wandb_container = QtWidgets.QWidget(group)
        wandb_layout = QtWidgets.QFormLayout(wandb_container)
        wandb_layout.setContentsMargins(0, 4, 0, 0)
        wandb_layout.setSpacing(4)

        self._wandb_project_input = QtWidgets.QLineEdit(wandb_container)
        self._wandb_project_input.setPlaceholderText("e.g. xuance-training")
        self._wandb_project_input.setToolTip(
            "Project name inside wandb.ai where runs will be grouped."
        )
        wandb_layout.addRow("Project:", self._wandb_project_input)

        self._wandb_entity_input = QtWidgets.QLineEdit(wandb_container)
        self._wandb_entity_input.setPlaceholderText("e.g. your-username")
        self._wandb_entity_input.setToolTip(
            "WandB entity (team or user namespace) to publish to."
        )
        wandb_layout.addRow("Entity:", self._wandb_entity_input)

        self._wandb_run_name_input = QtWidgets.QLineEdit(wandb_container)
        self._wandb_run_name_input.setPlaceholderText("Optional custom run name")
        self._wandb_run_name_input.setToolTip(
            "Custom run name (defaults to run_id if not specified)."
        )
        wandb_layout.addRow("Run Name:", self._wandb_run_name_input)

        self._wandb_api_key_input = QtWidgets.QLineEdit(wandb_container)
        self._wandb_api_key_input.setPlaceholderText("Optional API key override")
        self._wandb_api_key_input.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self._wandb_api_key_input.setToolTip(
            "Override the default WandB API key (from wandb login)."
        )
        wandb_layout.addRow("API Key:", self._wandb_api_key_input)

        # VPN proxy settings
        self._wandb_use_vpn_checkbox = QtWidgets.QCheckBox(
            "Route WandB traffic through VPN proxy", wandb_container
        )
        self._wandb_use_vpn_checkbox.toggled.connect(self._on_wandb_vpn_toggled)
        wandb_layout.addRow("", self._wandb_use_vpn_checkbox)

        self._wandb_http_proxy_input = QtWidgets.QLineEdit(wandb_container)
        self._wandb_http_proxy_input.setPlaceholderText("e.g. http://127.0.0.1:7890")
        self._wandb_http_proxy_input.setToolTip("HTTP proxy for WandB traffic.")
        wandb_layout.addRow("HTTP Proxy:", self._wandb_http_proxy_input)

        self._wandb_https_proxy_input = QtWidgets.QLineEdit(wandb_container)
        self._wandb_https_proxy_input.setPlaceholderText("e.g. http://127.0.0.1:7890")
        self._wandb_https_proxy_input.setToolTip("HTTPS proxy for WandB traffic.")
        wandb_layout.addRow("HTTPS Proxy:", self._wandb_https_proxy_input)

        group_layout.addWidget(wandb_container)
        self._wandb_container = wandb_container

        # Initialize WandB control states
        self._update_wandb_controls()

        layout.addWidget(group)

    def _on_wandb_toggled(self, checked: bool) -> None:
        """Handle WandB checkbox toggle."""
        _ = checked
        self._update_wandb_controls()

    def _on_wandb_vpn_toggled(self, checked: bool) -> None:
        """Handle WandB VPN checkbox toggle."""
        _ = checked
        self._update_wandb_controls()

    def _update_wandb_controls(self) -> None:
        """Update WandB control enabled states based on checkbox state."""
        wandb_enabled = self._wandb_checkbox.isChecked()

        # Base WandB fields
        base_fields = (
            self._wandb_project_input,
            self._wandb_entity_input,
            self._wandb_run_name_input,
            self._wandb_api_key_input,
        )
        for field in base_fields:
            field.setEnabled(wandb_enabled)

        # VPN checkbox
        self._wandb_use_vpn_checkbox.setEnabled(wandb_enabled)
        if not wandb_enabled:
            self._wandb_use_vpn_checkbox.setChecked(False)

        # Proxy fields (only enabled if VPN is checked)
        vpn_enabled = wandb_enabled and self._wandb_use_vpn_checkbox.isChecked()
        self._wandb_http_proxy_input.setEnabled(vpn_enabled)
        self._wandb_https_proxy_input.setEnabled(vpn_enabled)

    def _setup_fastlane_section(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up FastLane live visualization section."""
        group = QtWidgets.QGroupBox("FastLane Live Visualization", self)
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(8, 8, 8, 8)
        group_layout.setSpacing(6)

        # Hint label
        hint_label = QtWidgets.QLabel(
            "Stream live training frames to the GUI for real-time visualization.",
            group,
        )
        hint_label.setStyleSheet("color: #777777; font-size: 11px;")
        hint_label.setWordWrap(True)
        group_layout.addWidget(hint_label)

        # Enable checkbox row
        checkbox_layout = QtWidgets.QHBoxLayout()

        self._fastlane_checkbox = QtWidgets.QCheckBox("Enable FastLane streaming", group)
        self._fastlane_checkbox.setChecked(False)
        self._fastlane_checkbox.setToolTip(
            "Stream live frames from training environment to the GUI"
        )
        self._fastlane_checkbox.toggled.connect(self._on_fastlane_toggled)
        checkbox_layout.addWidget(self._fastlane_checkbox)

        self._fastlane_only_checkbox = QtWidgets.QCheckBox("FastLane Only", group)
        self._fastlane_only_checkbox.setChecked(True)
        self._fastlane_only_checkbox.setToolTip(
            "Skip telemetry persistence (gRPC/SQLite) - only stream to FastLane"
        )
        checkbox_layout.addWidget(self._fastlane_only_checkbox)

        checkbox_layout.addStretch(1)
        group_layout.addLayout(checkbox_layout)

        # FastLane configuration section
        fastlane_container = QtWidgets.QWidget(group)
        fastlane_layout = QtWidgets.QGridLayout(fastlane_container)
        fastlane_layout.setContentsMargins(0, 4, 0, 0)
        fastlane_layout.setSpacing(6)

        # Video Mode dropdown
        video_mode_label = QtWidgets.QLabel("Video Mode:", fastlane_container)
        self._video_mode_combo = QtWidgets.QComboBox(fastlane_container)
        for mode_key, mode_descriptor in VIDEO_MODE_DESCRIPTORS.items():
            self._video_mode_combo.addItem(mode_descriptor.label, mode_key)
            # Set tooltip for each item
            idx = self._video_mode_combo.count() - 1
            self._video_mode_combo.setItemData(
                idx, mode_descriptor.description, QtCore.Qt.ItemDataRole.ToolTipRole
            )
        self._video_mode_combo.setToolTip(
            "Single: Show one environment\n"
            "Grid: Show multiple parallel environments in a grid\n"
            "Off: Disable video streaming"
        )
        fastlane_layout.addWidget(video_mode_label, 0, 0)
        fastlane_layout.addWidget(self._video_mode_combo, 0, 1)

        # Grid Limit spinner
        grid_limit_label = QtWidgets.QLabel("Grid Limit:", fastlane_container)
        self._grid_limit_spin = QtWidgets.QSpinBox(fastlane_container)
        self._grid_limit_spin.setRange(1, 16)
        self._grid_limit_spin.setValue(4)
        self._grid_limit_spin.setToolTip(
            "Maximum number of parallel environments to show in grid mode (1-16)"
        )
        fastlane_layout.addWidget(grid_limit_label, 0, 2)
        fastlane_layout.addWidget(self._grid_limit_spin, 0, 3)

        # Probe Env spinner
        probe_env_label = QtWidgets.QLabel("Probe Env:", fastlane_container)
        self._fastlane_slot_spin = QtWidgets.QSpinBox(fastlane_container)
        self._fastlane_slot_spin.setRange(0, 64)
        self._fastlane_slot_spin.setValue(0)
        self._fastlane_slot_spin.setToolTip(
            "Environment index to probe in single mode (0-64)"
        )
        fastlane_layout.addWidget(probe_env_label, 1, 0)
        fastlane_layout.addWidget(self._fastlane_slot_spin, 1, 1)

        # Stretch column
        fastlane_layout.setColumnStretch(4, 1)

        group_layout.addWidget(fastlane_container)
        self._fastlane_container = fastlane_container

        # Initialize control states
        self._update_fastlane_controls()

        layout.addWidget(group)

    def _on_fastlane_toggled(self, checked: bool) -> None:
        """Handle FastLane checkbox toggle."""
        _ = checked
        self._update_fastlane_controls()

    def _update_fastlane_controls(self) -> None:
        """Update FastLane control enabled states based on checkbox state."""
        fastlane_enabled = self._fastlane_checkbox.isChecked()

        # Enable/disable all FastLane controls
        self._fastlane_only_checkbox.setEnabled(fastlane_enabled)
        self._video_mode_combo.setEnabled(fastlane_enabled)
        self._grid_limit_spin.setEnabled(fastlane_enabled)
        self._fastlane_slot_spin.setEnabled(fastlane_enabled)

        if not fastlane_enabled:
            # Reset to defaults when disabled
            self._fastlane_only_checkbox.setChecked(True)

    def _setup_buttons(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up dialog buttons."""
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        buttons.accepted.connect(self._handle_accept)
        buttons.rejected.connect(self.reject)

        # Add dry-run button
        dry_run_btn = buttons.addButton(
            "Dry Run",
            QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
        )
        if dry_run_btn is not None:
            dry_run_btn.setToolTip("Validate configuration without training")
            dry_run_btn.clicked.connect(self._on_dry_run_clicked)

        layout.addWidget(buttons)

    def _on_backend_changed(self, index: int) -> None:
        """Handle backend selection change."""
        backend = self._backend_combo.currentData()
        if not backend:
            return

        # Update algorithm count label
        summary = get_backend_summary()
        counts = summary.get(backend, {})
        self._algo_count_label.setText(
            f"Algorithms: {counts.get('single_agent', 0)} single-agent, "
            f"{counts.get('multi_agent', 0)} multi-agent"
        )

        # Repopulate algorithm combos
        self._populate_algo_combo(self._sa_algo_combo, backend, "single_agent")
        self._populate_algo_combo(self._ma_algo_combo, backend, "multi_agent")

    def _on_paradigm_changed(self, index: int) -> None:
        """Handle paradigm tab change."""
        # Update algorithm parameters for the new tab
        is_single_agent = index == 0
        algo_combo = self._sa_algo_combo if is_single_agent else self._ma_algo_combo
        self._on_algorithm_changed(algo_combo)

    def _on_algorithm_changed(self, combo: QtWidgets.QComboBox) -> None:
        """Handle algorithm selection change - rebuild parameters."""
        algo = combo.currentData()
        if algo:
            self._rebuild_algo_params(algo)

    def _on_env_family_changed(
        self,
        family_combo: QtWidgets.QComboBox,
        paradigm: str,
    ) -> None:
        """Handle environment family change."""
        family = family_combo.currentData()
        env_combo = self._sa_env_combo if paradigm == "single_agent" else self._ma_env_combo
        self._populate_env_combo(env_combo, family)

    def _populate_algo_combo(
        self,
        combo: QtWidgets.QComboBox,
        backend: str,
        paradigm: str,
    ) -> None:
        """Populate algorithm combo based on backend and paradigm."""
        combo.blockSignals(True)
        combo.clear()

        try:
            # Get algorithms grouped by category, then flatten with inline category notes
            categories = get_algorithms_by_category(backend, paradigm)

            # Flatten all algorithms with category as inline note
            all_algos: List[Tuple[str, str, str]] = []  # (display_text, key, description)
            for category, algos in categories.items():
                for algo in algos:
                    # Format: "PPO (Clip) - Policy Optimization"
                    display_text = f"{algo.display_name} - {category}"
                    all_algos.append((display_text, algo.key, algo.description))

            # Sort by display name
            all_algos.sort(key=lambda x: x[0])

            # Add items to combo with tooltip showing description
            for display_text, key, description in all_algos:
                combo.addItem(display_text, key)
                # Set tooltip for this item
                idx = combo.count() - 1
                combo.setItemData(idx, description, QtCore.Qt.ItemDataRole.ToolTipRole)

        except Exception as e:
            _LOGGER.warning("Failed to populate algorithms: %s", e)
            combo.addItem("PPO_Clip", "PPO_Clip")

        combo.blockSignals(False)

        # Select first item and trigger parameter update
        if combo.count() > 0:
            combo.setCurrentIndex(0)

        # Trigger parameter update
        self._on_algorithm_changed(combo)

    def _populate_env_combo(
        self,
        combo: QtWidgets.QComboBox,
        family: Optional[str],
    ) -> None:
        """Populate environment combo based on family."""
        combo.blockSignals(True)
        combo.clear()

        if family and family in _XUANCE_ENVIRONMENT_FAMILIES:
            for label, env_id in _XUANCE_ENVIRONMENT_FAMILIES[family]:
                combo.addItem(label, env_id)

        combo.blockSignals(False)

    def _clear_layout(self, layout: QtWidgets.QLayout) -> None:
        """Clear all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            child_layout = item.layout()
            if child_layout is not None:
                self._clear_layout(child_layout)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _rebuild_algo_params(self, algo: str) -> None:
        """Rebuild algorithm parameters UI based on selected algorithm."""
        self._clear_layout(self._algo_param_layout)
        self._algo_param_inputs.clear()

        schema_entry = _XUANCE_SCHEMAS.get(algo)
        if not schema_entry:
            # No schema - show placeholder
            placeholder = QtWidgets.QLabel(
                f"No parameters available for {algo}.", self
            )
            placeholder.setStyleSheet("color: #888888; font-style: italic;")
            self._algo_param_layout.addWidget(placeholder, 0, 0)
            self._algo_param_group.setVisible(True)
            return

        fields = schema_entry.get("fields", [])
        if not fields:
            placeholder = QtWidgets.QLabel(
                f"No configurable parameters for {algo}.", self
            )
            placeholder.setStyleSheet("color: #888888; font-style: italic;")
            self._algo_param_layout.addWidget(placeholder, 0, 0)
            self._algo_param_group.setVisible(True)
            return

        # Build parameter widgets
        self._populate_params_from_schema(fields)
        self._algo_param_group.setVisible(True)

    def _populate_params_from_schema(self, fields: Sequence[Dict[str, Any]]) -> None:
        """Populate algorithm parameters from schema fields."""
        columns = 2
        row = col = 0

        for field_spec in fields:
            name = field_spec.get("name")
            if not isinstance(name, str):
                continue
            if name in _SCHEMA_EXCLUDED_FIELDS:
                continue

            widget = self._create_widget_from_schema_field(field_spec)
            if widget is None:
                continue

            self._algo_param_inputs[name] = widget

            # Create label
            label_text = self._format_label(name)
            container = self._wrap_with_label(label_text, widget)

            self._algo_param_layout.addWidget(container, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1

        # Set column stretch
        for col_idx in range(columns):
            self._algo_param_layout.setColumnStretch(col_idx, 1)

    def _create_widget_from_schema_field(
        self, spec: Dict[str, Any]
    ) -> Optional[QtWidgets.QWidget]:
        """Create a widget from a schema field specification."""
        field_type = spec.get("type")
        default = spec.get("default")
        tooltip = spec.get("help") or ""

        if field_type == "bool":
            checkbox = QtWidgets.QCheckBox(self)
            checkbox.setChecked(bool(default))
            if tooltip:
                checkbox.setToolTip(tooltip)
            return checkbox

        if field_type == "int":
            spin = QtWidgets.QSpinBox(self)
            spin.setRange(-1_000_000_000, 1_000_000_000)
            if isinstance(default, int):
                spin.setValue(default)
            if tooltip:
                spin.setToolTip(tooltip)
            return spin

        if field_type == "float":
            spin = QtWidgets.QDoubleSpinBox(self)
            spin.setDecimals(6)
            spin.setRange(-1e9, 1e9)
            if isinstance(default, (int, float)):
                spin.setValue(float(default))
                # Set reasonable step based on default value
                if abs(default) > 0:
                    spin.setSingleStep(abs(default) / 10)
                else:
                    spin.setSingleStep(0.001)
            if tooltip:
                spin.setToolTip(tooltip)
            return spin

        if field_type == "str":
            line = QtWidgets.QLineEdit(self)
            if isinstance(default, str):
                line.setText(default)
            if tooltip:
                line.setToolTip(tooltip)
            return line

        return None

    @staticmethod
    def _format_label(name: str) -> str:
        """Format a parameter name as a label."""
        return name.replace("_", " ").title()

    def _wrap_with_label(
        self, label: str, widget: QtWidgets.QWidget
    ) -> QtWidgets.QWidget:
        """Wrap a widget with a label above it."""
        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        label_widget = QtWidgets.QLabel(label, container)
        label_widget.setStyleSheet("font-weight: 600; font-size: 11px;")
        layout.addWidget(label_widget)
        layout.addWidget(widget)

        return container

    def _collect_state(self) -> _FormState:
        """Collect current form state."""
        is_single_agent = self._paradigm_tabs.currentIndex() == 0
        paradigm = "single_agent" if is_single_agent else "multi_agent"

        if is_single_agent:
            algo_combo = self._sa_algo_combo
            env_family_combo = self._sa_env_family_combo
            env_combo = self._sa_env_combo
        else:
            algo_combo = self._ma_algo_combo
            env_family_combo = self._ma_env_family_combo
            env_combo = self._ma_env_combo

        method = algo_combo.currentData() or "PPO_Clip"
        env = env_family_combo.currentData() or "classic_control"
        env_id = env_combo.currentData() or "CartPole-v1"

        seed_value = self._seed_spin.value()
        seed = seed_value if seed_value > 0 else None

        # Collect algorithm parameters
        algo_params: Dict[str, Any] = {}
        for key, widget in self._algo_param_inputs.items():
            if isinstance(widget, QtWidgets.QSpinBox):
                algo_params[key] = int(widget.value())
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                algo_params[key] = float(widget.value())
            elif isinstance(widget, QtWidgets.QCheckBox):
                algo_params[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QLineEdit):
                algo_params[key] = widget.text().strip()

        # Collect FastLane settings
        video_mode_data = self._video_mode_combo.currentData()
        video_mode = video_mode_data if isinstance(video_mode_data, str) else VideoModes.SINGLE

        return _FormState(
            backend=self._backend_combo.currentData() or "torch",
            paradigm=paradigm,
            method=method,
            env=env,
            env_id=env_id,
            running_steps=self._steps_spin.value(),
            seed=seed,
            device=self._device_combo.currentData() or "cpu",
            parallels=self._parallels_spin.value(),
            test_mode=self._test_checkbox.isChecked(),
            benchmark_mode=self._benchmark_checkbox.isChecked(),
            worker_id=self._worker_id_input.text().strip() or None,
            notes=self._notes_edit.toPlainText().strip() or None,
            algo_params=algo_params,
            # FastLane settings
            fastlane_enabled=self._fastlane_checkbox.isChecked(),
            fastlane_only=self._fastlane_only_checkbox.isChecked(),
            fastlane_slot=self._fastlane_slot_spin.value(),
            fastlane_video_mode=video_mode,
            fastlane_grid_limit=self._grid_limit_spin.value(),
        )

    def _build_config(
        self,
        state: _FormState,
        *,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the training configuration."""
        run_id = run_id or _generate_run_id("xuance", state.method)

        # Get analytics settings
        track_tensorboard = self._tensorboard_checkbox.isChecked()
        track_wandb = self._wandb_checkbox.isChecked()
        wandb_project = self._wandb_project_input.text().strip() or None
        wandb_entity = self._wandb_entity_input.text().strip() or None
        wandb_run_name = self._wandb_run_name_input.text().strip() or None
        wandb_api_key = self._wandb_api_key_input.text().strip() or None
        use_wandb_vpn = self._wandb_use_vpn_checkbox.isChecked()
        wandb_http_proxy = self._wandb_http_proxy_input.text().strip() or None
        wandb_https_proxy = self._wandb_https_proxy_input.text().strip() or None

        worker_config: Dict[str, Any] = {
            "run_id": run_id,
            "method": state.method,
            "env": state.env,
            "env_id": state.env_id,
            "dl_toolbox": state.backend,
            "running_steps": state.running_steps,
            "device": state.device,
            "parallels": state.parallels,
            "test_mode": state.test_mode,
        }
        if state.seed is not None:
            worker_config["seed"] = state.seed
        if state.worker_id:
            worker_config["worker_id"] = state.worker_id

        extras: Dict[str, Any] = {
            "benchmark_mode": state.benchmark_mode,
            "paradigm": state.paradigm,
            "track_tensorboard": track_tensorboard,
            "track_wandb": track_wandb,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
            "wandb_run_name": wandb_run_name,
            "wandb_use_vpn_proxy": use_wandb_vpn,
            # FastLane settings
            "fastlane_enabled": state.fastlane_enabled,
            "fastlane_only": state.fastlane_only,
            "fastlane_slot": state.fastlane_slot,
            "fastlane_video_mode": state.fastlane_video_mode,
            "fastlane_grid_limit": state.fastlane_grid_limit,
        }
        if state.notes:
            extras["notes"] = state.notes
        # Include algorithm-specific parameters
        if state.algo_params:
            extras["algo_params"] = state.algo_params
        worker_config["extras"] = extras

        metadata = {
            "ui": {
                "worker_id": state.worker_id or "xuance_worker",
                "method": state.method,
                "env": state.env,
                "env_id": state.env_id,
                "backend": state.backend,
                "paradigm": state.paradigm,
                # FastLane UI settings
                "fastlane_enabled": state.fastlane_enabled,
                "fastlane_only": state.fastlane_only,
                "fastlane_slot": state.fastlane_slot,
                "fastlane_video_mode": state.fastlane_video_mode,
                "fastlane_grid_limit": state.fastlane_grid_limit,
            },
            "worker": {
                "worker_id": state.worker_id or "xuance_worker",
                "module": "xuance_worker.cli",
                "config": worker_config,
            },
            "artifacts": {
                "tensorboard": {
                    "enabled": track_tensorboard,
                    "relative_path": "tensorboard",
                },
                "wandb": {
                    "enabled": track_wandb,
                    "project": wandb_project or "xuance-training",
                    "entity": wandb_entity,
                    "run_name": wandb_run_name,
                    "use_vpn_proxy": use_wandb_vpn,
                    "http_proxy": wandb_http_proxy if use_wandb_vpn else None,
                    "https_proxy": wandb_https_proxy if use_wandb_vpn else None,
                },
            },
        }

        # Build environment variables
        environment: Dict[str, str] = {
            "XUANCE_RUN_ID": run_id,
            "XUANCE_DL_TOOLBOX": state.backend,
        }

        # TensorBoard environment
        if track_tensorboard:
            environment["XUANCE_TENSORBOARD_DIR"] = f"var/trainer/runs/{run_id}/tensorboard"

        # WandB environment variables
        if track_wandb:
            environment["WANDB_MODE"] = "online"
            if wandb_project:
                environment["WANDB_PROJECT"] = wandb_project
            if wandb_entity:
                environment["WANDB_ENTITY"] = wandb_entity
            if wandb_run_name:
                environment["WANDB_NAME"] = wandb_run_name
            if wandb_api_key:
                environment["WANDB_API_KEY"] = wandb_api_key
            if use_wandb_vpn and wandb_http_proxy:
                environment["WANDB_HTTP_PROXY"] = wandb_http_proxy
                environment["HTTP_PROXY"] = wandb_http_proxy
                environment["http_proxy"] = wandb_http_proxy
            if use_wandb_vpn and wandb_https_proxy:
                environment["WANDB_HTTPS_PROXY"] = wandb_https_proxy
                environment["HTTPS_PROXY"] = wandb_https_proxy
                environment["https_proxy"] = wandb_https_proxy
        else:
            environment["WANDB_MODE"] = "offline"

        # FastLane environment variables
        if state.fastlane_enabled:
            apply_fastlane_environment(
                environment,
                fastlane_only=state.fastlane_only,
                fastlane_slot=state.fastlane_slot,
                video_mode=state.fastlane_video_mode,
                grid_limit=state.fastlane_grid_limit,
            )
            # Also set the master switch for XuanCe sitecustomize
            environment["MOSAIC_FASTLANE_ENABLED"] = "1"

        config: Dict[str, Any] = {
            "run_name": run_id,
            "entry_point": sys.executable,
            "arguments": ["-m", "xuance_worker.cli"],
            "environment": environment,
            "resources": {
                "cpus": 4,
                "memory_mb": 4096,
                "gpus": {"requested": 1 if "cuda" in state.device else 0, "mandatory": False},
            },
            "metadata": metadata,
            "artifacts": {
                "output_prefix": f"runs/{run_id}",
                "persist_logs": True,
                "keep_checkpoints": True,
            },
        }

        return config

    def _handle_accept(self) -> None:
        """Handle OK button click."""
        state = self._collect_state()

        if not state.method:
            QtWidgets.QMessageBox.warning(
                self,
                "Algorithm Required",
                "Please select an algorithm before proceeding.",
            )
            return

        if not state.env_id:
            QtWidgets.QMessageBox.warning(
                self,
                "Environment Required",
                "Please select an environment before proceeding.",
            )
            return

        run_id = _generate_run_id("xuance", state.method)
        self._last_config = self._build_config(state, run_id=run_id)

        self.log_constant(
            LOG_UI_TRAIN_FORM_INFO,
            message="XuanCe training config accepted",
            extra={
                "run_id": run_id,
                "method": state.method,
                "backend": state.backend,
                "paradigm": state.paradigm,
                "algo_params_count": len(state.algo_params),
            },
        )

        self.accept()

    def _on_dry_run_clicked(self) -> None:
        """Handle dry-run button click."""
        state = self._collect_state()

        if not state.method or not state.env_id:
            QtWidgets.QMessageBox.warning(
                self,
                "Incomplete Configuration",
                "Please select an algorithm and environment before dry-run.",
            )
            return

        run_id = _generate_run_id("xuance", state.method)
        config = self._build_config(state, run_id=run_id)

        # Show config in a dialog
        config_json = json.dumps(config, indent=2)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Dry Run Configuration")
        dialog.resize(700, 500)

        layout = QtWidgets.QVBoxLayout(dialog)
        text_edit = QtWidgets.QPlainTextEdit(dialog)
        text_edit.setPlainText(config_json)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        close_btn = QtWidgets.QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

    def get_config(self) -> Dict[str, Any]:
        """Return the generated training configuration."""
        if self._last_config is not None:
            return copy.deepcopy(self._last_config)

        state = self._collect_state()
        return self._build_config(state)


__all__ = ["XuanCeTrainForm"]


# Register with form factory
try:
    from gym_gui.ui.forms import get_worker_form_factory

    _factory = get_worker_form_factory()
    if not _factory.has_train_form("xuance_worker"):
        _factory.register_train_form(
            "xuance_worker",
            lambda parent=None, **kwargs: XuanCeTrainForm(parent=parent, **kwargs),
        )
except ImportError:
    pass  # Form factory not available
