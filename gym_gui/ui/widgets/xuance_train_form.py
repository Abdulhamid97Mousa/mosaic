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
    LOG_UI_TRAIN_FORM_TRACE,
    LOG_UI_TRAIN_FORM_INFO,
    LOG_UI_TRAIN_FORM_WARNING,
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
from gym_gui.validations.validation_xuance_worker_form import run_xuance_dry_run
from gym_gui.config.paths import XUANCE_SCRIPTS_DIR, VAR_CUSTOM_SCRIPTS_DIR
from gym_gui.core.enums import GameId, EnvironmentFamily, ENVIRONMENT_FAMILY_BY_GAME
from collections import defaultdict
import re


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


# Static environment families (fallback when dynamic discovery unavailable)
# For families with gymnasium registration (minigrid, babyai), these are only
# used if the package isn't installed. Dynamic discovery is preferred.
_STATIC_ENVIRONMENT_FAMILIES: Dict[str, List[Tuple[str, str]]] = {
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
        # Other MiniGrid environments
        ("MiniGrid-BlockedUnlockPickup-v0", "MiniGrid-BlockedUnlockPickup-v0"),
        ("MiniGrid-RedBlueDoors-6x6-v0", "MiniGrid-RedBlueDoors-6x6-v0"),
        ("MiniGrid-RedBlueDoors-8x8-v0", "MiniGrid-RedBlueDoors-8x8-v0"),
        ("MiniGrid-FourRooms-v0", "MiniGrid-FourRooms-v0"),
        ("MiniGrid-LockedRoom-v0", "MiniGrid-LockedRoom-v0"),
        ("MiniGrid-KeyCorridorS3R1-v0", "MiniGrid-KeyCorridorS3R1-v0"),
        ("MiniGrid-KeyCorridorS3R2-v0", "MiniGrid-KeyCorridorS3R2-v0"),
        ("MiniGrid-KeyCorridorS3R3-v0", "MiniGrid-KeyCorridorS3R3-v0"),
        ("MiniGrid-KeyCorridorS4R3-v0", "MiniGrid-KeyCorridorS4R3-v0"),
        ("MiniGrid-KeyCorridorS5R3-v0", "MiniGrid-KeyCorridorS5R3-v0"),
        ("MiniGrid-KeyCorridorS6R3-v0", "MiniGrid-KeyCorridorS6R3-v0"),
        ("MiniGrid-UnlockPickup-v0", "MiniGrid-UnlockPickup-v0"),
        ("MiniGrid-Unlock-v0", "MiniGrid-Unlock-v0"),
        ("MiniGrid-DistShift1-v0", "MiniGrid-DistShift1-v0"),
        ("MiniGrid-DistShift2-v0", "MiniGrid-DistShift2-v0"),
        ("MiniGrid-MemoryS17Random-v0", "MiniGrid-MemoryS17Random-v0"),
        ("MiniGrid-MemoryS13Random-v0", "MiniGrid-MemoryS13Random-v0"),
        ("MiniGrid-MemoryS13-v0", "MiniGrid-MemoryS13-v0"),
        ("MiniGrid-MemoryS11-v0", "MiniGrid-MemoryS11-v0"),
        ("MiniGrid-GoToDoor-5x5-v0", "MiniGrid-GoToDoor-5x5-v0"),
        ("MiniGrid-GoToDoor-6x6-v0", "MiniGrid-GoToDoor-6x6-v0"),
        ("MiniGrid-GoToDoor-8x8-v0", "MiniGrid-GoToDoor-8x8-v0"),
        ("MiniGrid-PutNear-6x6-N2-v0", "MiniGrid-PutNear-6x6-N2-v0"),
        ("MiniGrid-PutNear-8x8-N3-v0", "MiniGrid-PutNear-8x8-N3-v0"),
        ("MiniGrid-Fetch-5x5-N2-v0", "MiniGrid-Fetch-5x5-N2-v0"),
        ("MiniGrid-Fetch-6x6-N2-v0", "MiniGrid-Fetch-6x6-N2-v0"),
        ("MiniGrid-Fetch-8x8-N3-v0", "MiniGrid-Fetch-8x8-N3-v0"),
    ],
    # BabyAI environments (language-conditioned tasks built on MiniGrid)
    "babyai": [
        ("BabyAI-GoToRedBall-v0", "BabyAI-GoToRedBall-v0"),
        ("BabyAI-GoToRedBallGrey-v0", "BabyAI-GoToRedBallGrey-v0"),
        ("BabyAI-GoToRedBallNoDists-v0", "BabyAI-GoToRedBallNoDists-v0"),
        ("BabyAI-GoToObj-v0", "BabyAI-GoToObj-v0"),
        ("BabyAI-GoToObjS4-v0", "BabyAI-GoToObjS4-v0"),
        ("BabyAI-GoToObjS6-v0", "BabyAI-GoToObjS6-v0"),
        ("BabyAI-GoToLocal-v0", "BabyAI-GoToLocal-v0"),
        ("BabyAI-GoToLocalS5N2-v0", "BabyAI-GoToLocalS5N2-v0"),
        ("BabyAI-GoToLocalS6N2-v0", "BabyAI-GoToLocalS6N2-v0"),
        ("BabyAI-GoToLocalS6N3-v0", "BabyAI-GoToLocalS6N3-v0"),
        ("BabyAI-GoToLocalS6N4-v0", "BabyAI-GoToLocalS6N4-v0"),
        ("BabyAI-GoToLocalS7N4-v0", "BabyAI-GoToLocalS7N4-v0"),
        ("BabyAI-GoToLocalS7N5-v0", "BabyAI-GoToLocalS7N5-v0"),
        ("BabyAI-GoToLocalS8N2-v0", "BabyAI-GoToLocalS8N2-v0"),
        ("BabyAI-GoToLocalS8N3-v0", "BabyAI-GoToLocalS8N3-v0"),
        ("BabyAI-GoToLocalS8N4-v0", "BabyAI-GoToLocalS8N4-v0"),
        ("BabyAI-GoToLocalS8N5-v0", "BabyAI-GoToLocalS8N5-v0"),
        ("BabyAI-GoToLocalS8N6-v0", "BabyAI-GoToLocalS8N6-v0"),
        ("BabyAI-GoToLocalS8N7-v0", "BabyAI-GoToLocalS8N7-v0"),
        ("BabyAI-PickupLoc-v0", "BabyAI-PickupLoc-v0"),
        ("BabyAI-PickupDist-v0", "BabyAI-PickupDist-v0"),
        ("BabyAI-PickupDistDebug-v0", "BabyAI-PickupDistDebug-v0"),
        ("BabyAI-PutNextLocal-v0", "BabyAI-PutNextLocal-v0"),
        ("BabyAI-PutNextLocalS5N3-v0", "BabyAI-PutNextLocalS5N3-v0"),
        ("BabyAI-PutNextLocalS6N4-v0", "BabyAI-PutNextLocalS6N4-v0"),
        ("BabyAI-OpenDoor-v0", "BabyAI-OpenDoor-v0"),
        ("BabyAI-OpenDoorDebug-v0", "BabyAI-OpenDoorDebug-v0"),
        ("BabyAI-OpenDoorColor-v0", "BabyAI-OpenDoorColor-v0"),
        ("BabyAI-OpenDoorLoc-v0", "BabyAI-OpenDoorLoc-v0"),
        ("BabyAI-UnlockPickup-v0", "BabyAI-UnlockPickup-v0"),
        ("BabyAI-Unlock-v0", "BabyAI-Unlock-v0"),
        ("BabyAI-UnlockLocal-v0", "BabyAI-UnlockLocal-v0"),
        ("BabyAI-UnlockLocalDist-v0", "BabyAI-UnlockLocalDist-v0"),
        ("BabyAI-KeyInBox-v0", "BabyAI-KeyInBox-v0"),
        ("BabyAI-UnlockToUnlock-v0", "BabyAI-UnlockToUnlock-v0"),
        ("BabyAI-BlockedUnlockPickup-v0", "BabyAI-BlockedUnlockPickup-v0"),
        ("BabyAI-ActionObjDoor-v0", "BabyAI-ActionObjDoor-v0"),
        ("BabyAI-FindObjS5-v0", "BabyAI-FindObjS5-v0"),
        ("BabyAI-FindObjS6-v0", "BabyAI-FindObjS6-v0"),
        ("BabyAI-FindObjS7-v0", "BabyAI-FindObjS7-v0"),
        ("BabyAI-KeyCorridor-v0", "BabyAI-KeyCorridor-v0"),
        ("BabyAI-KeyCorridorS3R1-v0", "BabyAI-KeyCorridorS3R1-v0"),
        ("BabyAI-KeyCorridorS3R2-v0", "BabyAI-KeyCorridorS3R2-v0"),
        ("BabyAI-KeyCorridorS3R3-v0", "BabyAI-KeyCorridorS3R3-v0"),
        ("BabyAI-KeyCorridorS4R3-v0", "BabyAI-KeyCorridorS4R3-v0"),
        ("BabyAI-KeyCorridorS5R3-v0", "BabyAI-KeyCorridorS5R3-v0"),
        ("BabyAI-KeyCorridorS6R3-v0", "BabyAI-KeyCorridorS6R3-v0"),
        ("BabyAI-OneRoomS8-v0", "BabyAI-OneRoomS8-v0"),
        ("BabyAI-OneRoomS12-v0", "BabyAI-OneRoomS12-v0"),
        ("BabyAI-OneRoomS16-v0", "BabyAI-OneRoomS16-v0"),
        ("BabyAI-OneRoomS20-v0", "BabyAI-OneRoomS20-v0"),
        ("BabyAI-MoveTwoAcrossS5N2-v0", "BabyAI-MoveTwoAcrossS5N2-v0"),
        ("BabyAI-MoveTwoAcrossS8N9-v0", "BabyAI-MoveTwoAcrossS8N9-v0"),
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
    "minigrid", "babyai", "vizdoom", "minihack", "nethack", "crafter", "procgen",
]
_MULTI_AGENT_FAMILIES = ["mpe", "smac", "football"]


# --- Environment Discovery (Hybrid: Central Registry + Dynamic Discovery) ---
# Map XuanCe family names to EnvironmentFamily enum values
_XUANCE_FAMILY_TO_ENUM: Dict[str, EnvironmentFamily] = {
    "classic_control": EnvironmentFamily.CLASSIC_CONTROL,
    "box2d": EnvironmentFamily.BOX2D,
    "mujoco": EnvironmentFamily.MUJOCO,
    "atari": EnvironmentFamily.ATARI,
    "minigrid": EnvironmentFamily.MINIGRID,
    "babyai": EnvironmentFamily.BABYAI,
    "vizdoom": EnvironmentFamily.VIZDOOM,
    "minihack": EnvironmentFamily.MINIHACK,
    "nethack": EnvironmentFamily.NETHACK,
    "crafter": EnvironmentFamily.CRAFTER,
    "procgen": EnvironmentFamily.PROCGEN,
}

# Families that support dynamic discovery from gymnasium registry
# Maps family name to (package_to_import, prefix_pattern)
_DYNAMIC_DISCOVERY_FAMILIES: Dict[str, Tuple[str, str]] = {
    "minigrid": ("minigrid", "MiniGrid-"),
    "babyai": ("minigrid", "BabyAI-"),  # BabyAI envs are registered by minigrid
    "minihack": ("nle", "MiniHack-"),
    "nethack": ("nle", "NetHack"),
    "crafter": ("crafter", "Crafter"),
}


def _discover_gymnasium_envs(package: str, prefix: str) -> List[Tuple[str, str]]:
    """
    Dynamically discover environments from gymnasium registry.

    Args:
        package: Package to import to register environments
        prefix: Environment name prefix to filter

    Returns:
        List of (label, env_id) tuples, sorted alphabetically.
    """
    try:
        # Import the package to register its environments
        __import__(package)
    except ImportError:
        return []

    try:
        import gymnasium
        envs = []
        for env_id in gymnasium.registry.keys():
            if env_id.startswith(prefix):
                envs.append((env_id, env_id))
        return sorted(envs, key=lambda x: x[0])
    except Exception as e:
        _LOGGER.debug(f"Error discovering {prefix} environments: {e}")
        return []


def _build_environment_index() -> Dict[str, List[Tuple[str, str]]]:
    """
    Build environment index using hybrid approach:
    1. Central GameId enum (gym_gui.core.enums) as base
    2. Dynamic discovery from gymnasium for installed packages
    3. Static fallback for multi-agent families (mpe, smac, football)

    This ensures:
    - Single source of truth where possible (central enums)
    - Complete coverage when packages are installed (dynamic discovery)
    - No duplication
    """
    index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    # First: Try dynamic discovery for supported families
    for family, (package, prefix) in _DYNAMIC_DISCOVERY_FAMILIES.items():
        discovered = _discover_gymnasium_envs(package, prefix)
        if discovered:
            index[family] = discovered

    # Second: Fill in from central GameId enum for families not populated by dynamic discovery
    # Track which families were dynamically discovered (have data)
    dynamically_populated = {f for f in index if index[f]}

    for game in GameId:
        env_family = ENVIRONMENT_FAMILY_BY_GAME.get(game)
        if env_family is None:
            continue

        # Find which XuanCe family name maps to this EnvironmentFamily
        for xuance_family, mapped_family in _XUANCE_FAMILY_TO_ENUM.items():
            if env_family == mapped_family:
                # Skip if family was populated by dynamic discovery
                if xuance_family in dynamically_populated:
                    break
                # Add from central enum
                env_id = game.value
                index[xuance_family].append((env_id, env_id))
                break

    # Sort families populated from central enum
    for family in index:
        if family not in _DYNAMIC_DISCOVERY_FAMILIES:
            index[family].sort(key=lambda x: x[0])

    # Third: Add multi-agent families from static lists
    # (these are XuanCe-specific and not in gym_gui.core.enums)
    for family in ["mpe", "smac", "football"]:
        if family not in index and family in _STATIC_ENVIRONMENT_FAMILIES:
            index[family] = _STATIC_ENVIRONMENT_FAMILIES[family]

    return dict(index)


# Build the index once at module load
_XUANCE_ENVIRONMENT_INDEX: Dict[str, List[Tuple[str, str]]] = _build_environment_index()


def get_environment_ids(family: str) -> List[Tuple[str, str]]:
    """
    Get environment IDs for a family.

    Uses hybrid approach:
    - Dynamic discovery from gymnasium for installed packages (complete list)
    - Central registry fallback (curated list)
    - Static lists for multi-agent families

    Returns list of (label, env_id) tuples for the given family.
    """
    return _XUANCE_ENVIRONMENT_INDEX.get(family, [])


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
    # Custom script for curriculum learning
    custom_script_path: Optional[str] = None


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
        # Configure dropdown to show max 20 items with scrollbar
        env_view = QtWidgets.QListView(env_combo)
        env_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        env_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        env_view.setUniformItemSizes(True)
        env_combo.setView(env_view)
        env_combo.setStyleSheet("QComboBox { combobox-popup: 0; }")  # Required for setMaxVisibleItems to work
        env_combo.setMaxVisibleItems(20)
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

        # Device with GPU indicator inline
        device_container = QtWidgets.QWidget(self)
        device_layout = QtWidgets.QHBoxLayout(device_container)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(4)

        device_label = QtWidgets.QLabel("Device:", self)
        self._device_combo = QtWidgets.QComboBox(self)
        self._device_combo.addItem("CPU", "cpu")
        self._device_combo.addItem("CUDA (GPU)", "cuda")
        self._device_combo.addItem("CUDA:0", "cuda:0")
        self._device_combo.addItem("CUDA:1", "cuda:1")

        # Auto-select GPU if available, and update tooltip
        if self._gpu_count > 0:
            self._device_combo.setCurrentIndex(2)  # cuda:0
            self._device_combo.setToolTip(
                f"Computing device\nDetected: {self._gpu_name} ({self._gpu_count} GPU{'s' if self._gpu_count > 1 else ''} available)"
            )
            gpu_info = QtWidgets.QLabel(f"[GPU] {self._gpu_name}", self)
            gpu_info.setStyleSheet("color: green; font-size: 10px;")
        else:
            self._device_combo.setToolTip("Computing device\nNo GPU detected - using CPU")
            gpu_info = QtWidgets.QLabel("(No GPU)", self)
            gpu_info.setStyleSheet("color: #888; font-size: 10px;")

        device_layout.addWidget(self._device_combo)
        device_layout.addWidget(gpu_info)
        device_layout.addStretch()

        params_layout.addWidget(device_label, 1, 0)
        params_layout.addWidget(device_container, 1, 1)

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

        # Procedural generation checkbox
        self._procedural_generation_checkbox = QtWidgets.QCheckBox("Procedural Generation (Randomize Levels)", self)
        self._procedural_generation_checkbox.setChecked(True)
        self._procedural_generation_checkbox.setToolTip(
            "Enable procedural generation: each episode uses a different random level layout (standard RL training).\n"
            "Disable for fixed generation: all episodes use the same level layout (for debugging/memorization testing)."
        )
        params_layout.addWidget(self._procedural_generation_checkbox, 3, 0, 1, 2)

        # Worker ID
        worker_label = QtWidgets.QLabel("Worker ID:", self)
        self._worker_id_input = QtWidgets.QLineEdit(self)
        self._worker_id_input.setPlaceholderText("Optional worker identifier")
        self._worker_id_input.setToolTip("Optional identifier for this worker")
        params_layout.addWidget(worker_label, 4, 0)
        params_layout.addWidget(self._worker_id_input, 4, 1)

        # Custom Script for curriculum learning
        script_label = QtWidgets.QLabel("Custom Script:", self)
        self._custom_script_combo = QtWidgets.QComboBox(self)
        self._custom_script_combo.setToolTip(
            "Select a custom training script for curriculum learning or multi-phase training.\n"
            "'None' uses standard single-environment training.\n"
            "'Browse...' lets you import a script from your filesystem."
        )
        self._populate_custom_scripts()
        self._custom_script_combo.currentIndexChanged.connect(self._on_custom_script_changed)
        params_layout.addWidget(script_label, 4, 2)
        params_layout.addWidget(self._custom_script_combo, 4, 3)

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

    def _populate_custom_scripts(self) -> None:
        """Populate the custom scripts dropdown with available scripts."""
        self._custom_script_combo.blockSignals(True)
        self._custom_script_combo.clear()

        # First option: standard training (no script)
        self._custom_script_combo.addItem("None (Standard Training)", None)

        # Add scripts from XUANCE_SCRIPTS_DIR
        if XUANCE_SCRIPTS_DIR.is_dir():
            scripts = sorted(XUANCE_SCRIPTS_DIR.glob("*.sh"))
            for script_path in scripts:
                description = self._parse_script_metadata(script_path)
                label = f"{script_path.stem}"
                if description:
                    label = f"{script_path.stem} - {description}"
                self._custom_script_combo.addItem(label, str(script_path))

        # Last option: browse for custom script
        self._custom_script_combo.addItem("Browse...", "BROWSE")

        self._custom_script_combo.blockSignals(False)

    def _parse_script_metadata(self, script_path: Path) -> str:
        """Parse @description metadata from a script file."""
        try:
            content = script_path.read_text(encoding="utf-8")
            for line in content.split("\n")[:30]:
                if "@description:" in line:
                    desc = line.split("@description:")[-1].strip()
                    return desc
            return ""
        except Exception:
            return ""

    def _parse_script_full_metadata(self, script_path: Path) -> Dict[str, Any]:
        """Parse all metadata from a script file including environments."""
        metadata: Dict[str, Any] = {
            "description": "",
            "env_family": None,
            "phases": None,
            "total_timesteps": None,
            "environments": [],
        }
        try:
            content = script_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            for line in lines[:30]:
                if "@description:" in line:
                    metadata["description"] = line.split("@description:")[-1].strip()
                elif "@env_family:" in line:
                    metadata["env_family"] = line.split("@env_family:")[-1].strip()
                elif "@phases:" in line:
                    try:
                        metadata["phases"] = int(line.split("@phases:")[-1].strip())
                    except ValueError:
                        pass
                elif "@total_timesteps:" in line:
                    try:
                        metadata["total_timesteps"] = int(line.split("@total_timesteps:")[-1].strip())
                    except ValueError:
                        pass

            env_pattern = re.compile(r'PHASE\d+_ENV="([^"]+)"')
            for match in env_pattern.finditer(content):
                env_id = match.group(1)
                if env_id not in metadata["environments"]:
                    metadata["environments"].append(env_id)

        except Exception:
            pass
        return metadata

    def _on_custom_script_changed(self, index: int) -> None:
        """Handle custom script combo selection."""
        data = self._custom_script_combo.itemData(index)
        if data == "BROWSE":
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Custom Training Script",
                str(XUANCE_SCRIPTS_DIR) if XUANCE_SCRIPTS_DIR.is_dir() else str(Path.home()),
                "Shell Scripts (*.sh);;All Files (*)",
            )
            if file_path:
                script_name = Path(file_path).stem
                description = self._parse_script_metadata(Path(file_path))
                label = f"{script_name} (imported)"
                if description:
                    label = f"{script_name} - {description} (imported)"

                insert_index = self._custom_script_combo.count() - 1
                self._custom_script_combo.blockSignals(True)
                self._custom_script_combo.insertItem(insert_index, label, file_path)
                self._custom_script_combo.setCurrentIndex(insert_index)
                self._custom_script_combo.blockSignals(False)
            else:
                self._custom_script_combo.blockSignals(True)
                self._custom_script_combo.setCurrentIndex(0)
                self._custom_script_combo.blockSignals(False)

        # Update form controls based on script selection
        self._update_script_mode_controls()

    def _update_script_mode_controls(self) -> None:
        """Enable/disable form controls based on custom script selection.

        When a custom script is selected, the script controls the algorithm,
        environments, and training parameters - so those form fields should be
        disabled to indicate they'll be overridden.

        Note: XuanCe uses tabbed paradigms (single-agent/multi-agent), so we
        need to disable controls in both tabs.
        """
        custom_script_data = self._custom_script_combo.currentData()
        is_script_mode = custom_script_data is not None and custom_script_data != "BROWSE"

        # Disable algorithm/environment controls in both paradigm tabs
        self._sa_algo_combo.setEnabled(not is_script_mode)
        self._sa_env_family_combo.setEnabled(not is_script_mode)
        self._sa_env_combo.setEnabled(not is_script_mode)

        self._ma_algo_combo.setEnabled(not is_script_mode)
        self._ma_env_family_combo.setEnabled(not is_script_mode)
        self._ma_env_combo.setEnabled(not is_script_mode)

        # Disable training steps and algorithm parameters group
        self._steps_spin.setEnabled(not is_script_mode)
        self._algo_param_group.setEnabled(not is_script_mode)

        # Update tooltips to explain why controls are disabled
        if is_script_mode:
            script_name = Path(str(custom_script_data)).stem if custom_script_data else "script"
            disabled_tooltip = f"Controlled by custom script: {script_name}"

            # Single-agent controls
            self._sa_algo_combo.setToolTip(disabled_tooltip)
            self._sa_env_family_combo.setToolTip(disabled_tooltip)
            self._sa_env_combo.setToolTip(disabled_tooltip)

            # Multi-agent controls
            self._ma_algo_combo.setToolTip(disabled_tooltip)
            self._ma_env_family_combo.setToolTip(disabled_tooltip)
            self._ma_env_combo.setToolTip(disabled_tooltip)

            # Training parameters
            self._steps_spin.setToolTip(disabled_tooltip)
            self._algo_param_group.setToolTip(disabled_tooltip)

            # Auto-set GRID mode for custom scripts (curriculum learning typically uses multiple envs)
            # User can still change this if needed since FastLane controls remain enabled
            if hasattr(self, "_video_mode_combo"):
                grid_index = self._video_mode_combo.findData(VideoModes.GRID)
                if grid_index >= 0:
                    self._video_mode_combo.setCurrentIndex(grid_index)
            if hasattr(self, "_grid_limit_spin"):
                self._grid_limit_spin.setValue(4)
            # Enable FastLane if not already enabled for script mode
            if hasattr(self, "_fastlane_checkbox") and not self._fastlane_checkbox.isChecked():
                self._fastlane_checkbox.setChecked(True)
        else:
            # Restore original tooltips
            self._sa_algo_combo.setToolTip("Select the RL algorithm")
            self._sa_env_family_combo.setToolTip("Select the environment family")
            self._sa_env_combo.setToolTip("Select the specific environment")

            self._ma_algo_combo.setToolTip("Select the RL algorithm")
            self._ma_env_family_combo.setToolTip("Select the environment family")
            self._ma_env_combo.setToolTip("Select the specific environment")

            self._steps_spin.setToolTip("Total training steps")
            self._algo_param_group.setToolTip("")

    def _setup_buttons(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up dialog buttons."""
        # Validation status label (shows dry-run results)
        self._validation_status_label = QtWidgets.QLabel(
            "Dry-run validation has not been executed yet."
        )
        self._validation_status_label.setStyleSheet("color: #666666;")
        layout.addWidget(self._validation_status_label)

        # Initialize validation output storage
        self._last_validation_output: str = ""

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal,
            self,
        )
        buttons.accepted.connect(self._handle_accept)
        buttons.rejected.connect(self.reject)

        # Add validate button (renamed from "Dry Run" to match CleanRL)
        validate_btn = buttons.addButton(
            "Validate",
            QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
        )
        if validate_btn is not None:
            validate_btn.setToolTip("Validate configuration without training")
            validate_btn.clicked.connect(self._on_validate_clicked)

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
        """Populate environment combo based on family using dynamic discovery."""
        combo.blockSignals(True)
        combo.clear()

        if family:
            # Use dynamic discovery (falls back to static list if needed)
            env_list = get_environment_ids(family)
            for label, env_id in env_list:
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

        # Add procedural generation to algo_params
        algo_params["procedural_generation"] = self._procedural_generation_checkbox.isChecked()

        # Collect FastLane settings
        video_mode_data = self._video_mode_combo.currentData()
        video_mode = video_mode_data if isinstance(video_mode_data, str) else VideoModes.SINGLE

        # Custom script selection
        custom_script_data = self._custom_script_combo.currentData()
        custom_script_path: Optional[str] = None
        if custom_script_data and custom_script_data != "BROWSE":
            custom_script_path = str(custom_script_data)

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
            # Custom script
            custom_script_path=custom_script_path,
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
                "worker_id": "xuance_worker",  # Always xuance_worker for tab naming
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
                "worker_id": "xuance_worker",  # Always xuance_worker for detection
                "module": "xuance_worker.cli",
                "config": worker_config,
            },
            "artifacts": {
                "tensorboard": {
                    "enabled": track_tensorboard,
                    "relative_path": f"var/trainer/runs/{run_id}/tensorboard" if track_tensorboard else None,
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
        # Get procedural generation setting from algo_params
        procedural_gen = state.algo_params.get("procedural_generation", True)

        environment: Dict[str, str] = {
            "XUANCE_RUN_ID": run_id,
            "XUANCE_DL_TOOLBOX": state.backend,
            "XUANCE_PARALLELS": str(state.parallels),  # For FastLane grid mode
            "CLEANRL_PROCEDURAL_GENERATION": "1" if procedural_gen else "0",
        }
        # Add seed to environment if specified
        if state.seed is not None:
            environment["CLEANRL_SEED"] = str(state.seed)

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

        # Handle custom script mode vs standard training mode
        if state.custom_script_path:
            # Custom script mode: run bash script with config passed via environment
            config_file_path = VAR_CUSTOM_SCRIPTS_DIR / run_id / "base_config.json"
            VAR_CUSTOM_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
            (VAR_CUSTOM_SCRIPTS_DIR / run_id).mkdir(parents=True, exist_ok=True)

            # Parse script metadata to get the actual environments
            script_metadata = self._parse_script_full_metadata(Path(state.custom_script_path))
            script_envs = script_metadata.get("environments", [])
            script_env_family = script_metadata.get("env_family", "")
            script_name = Path(state.custom_script_path).stem

            # Override env_id with script's first environment or a descriptive name
            if script_envs:
                script_env_id = script_envs[0]  # First phase environment
            elif script_env_family:
                script_env_id = f"{script_env_family}-curriculum"
            else:
                script_env_id = f"{script_name}"

            # Update worker_config with script's environment
            worker_config["env_id"] = script_env_id

            # In custom script mode, the script controls ALL parameters.
            # Clear form-sourced algo_params completely - the script is self-contained
            # and defines its own algorithm, environment, and training settings.
            if "extras" in worker_config:
                worker_config["extras"]["algo_params"] = {}

            # Update metadata env_id for tab naming
            metadata["ui"]["env_id"] = script_env_id

            # Write worker_config to the config file for the script to read
            config_file_path.write_text(json.dumps(worker_config, indent=2))

            # Add MOSAIC environment variables for script
            environment["MOSAIC_CONFIG_FILE"] = str(config_file_path)
            environment["MOSAIC_RUN_ID"] = run_id
            environment["MOSAIC_CUSTOM_SCRIPTS_DIR"] = str(VAR_CUSTOM_SCRIPTS_DIR)
            environment["MOSAIC_CHECKPOINT_DIR"] = str(VAR_CUSTOM_SCRIPTS_DIR / run_id / "checkpoints")

            # Update metadata to reflect script mode
            metadata["ui"]["custom_script"] = state.custom_script_path
            metadata["ui"]["custom_script_name"] = script_name

            # CRITICAL: Update metadata.worker to use script instead of module
            # The dispatcher reads metadata.worker.script, NOT the top-level entry_point
            # If module is present, dispatcher runs 'python -m module' instead of bash script
            # This prevents the script's jq overrides from ever executing!
            del metadata["worker"]["module"]  # Remove the default module
            metadata["worker"]["script"] = "/bin/bash"  # Use bash to execute
            metadata["worker"]["arguments"] = [state.custom_script_path]  # Script path as argument

            entry_point = "/bin/bash"
            arguments = [state.custom_script_path]
        else:
            # Standard training mode: run xuance_worker.cli directly
            entry_point = sys.executable
            arguments = ["-m", "xuance_worker.cli"]

        config: Dict[str, Any] = {
            "run_name": run_id,
            "entry_point": entry_point,
            "arguments": arguments,
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

    def _run_validation(
        self,
        state: _FormState,
        *,
        run_id: str,
        persist_config: bool,
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Run dry-run validation via subprocess.

        Args:
            state: Current form state.
            run_id: Unique identifier for this run.
            persist_config: If True, store config in _last_config on success.

        Returns:
            Tuple of (success, config_dict or None).
        """
        config = self._build_config(state, run_id=run_id)

        # Handle custom script mode differently
        if state.custom_script_path:
            return self._run_script_validation(state, config, run_id, persist_config)

        # Extract worker config for validation (the actual config passed to xuance_worker)
        worker_config = config.get("metadata", {}).get("worker", {}).get("config", {})

        self._validation_status_label.setText("Running XuanCe dry-run validation...")
        self._validation_status_label.setStyleSheet("color: #1565c0;")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

        try:
            success, output = run_xuance_dry_run(worker_config)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._set_validation_result(success, output)
        self._append_validation_notes(success, output)

        if persist_config and success:
            self._last_config = copy.deepcopy(config)
        elif not persist_config:
            self._last_config = None

        return success, (copy.deepcopy(config) if success else None)

    def _run_script_validation(
        self,
        state: _FormState,
        config: Dict[str, Any],
        run_id: str,
        persist_config: bool,
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Validate a custom script configuration."""
        script_path = Path(state.custom_script_path) if state.custom_script_path else None

        # Log validation start
        self.log_constant(
            LOG_UI_TRAIN_FORM_TRACE,
            message="Starting custom script validation",
            extra={
                "run_id": run_id,
                "script_path": state.custom_script_path,
                "method": state.method,
                "backend": state.backend,
            },
        )

        self._validation_status_label.setText("Validating custom script configuration...")
        self._validation_status_label.setStyleSheet("color: #1565c0;")

        errors: List[str] = []
        warnings: List[str] = []
        script_metadata: Dict[str, Any] = {}

        # Check script file exists
        if script_path is None or not script_path.exists():
            errors.append(f"Script file not found: {state.custom_script_path}")
        elif not script_path.is_file():
            errors.append(f"Path is not a file: {state.custom_script_path}")
        else:
            # Parse full script metadata
            script_metadata = self._parse_script_full_metadata(script_path)

            # Log script metadata extraction
            self.log_constant(
                LOG_UI_TRAIN_FORM_TRACE,
                message="Parsed custom script metadata",
                extra={
                    "run_id": run_id,
                    "script_name": script_path.name,
                    "env_family": script_metadata.get("env_family"),
                    "phases": script_metadata.get("phases"),
                    "environments": script_metadata.get("environments", []),
                },
            )

            # Check script is readable and has shebang
            try:
                content = script_path.read_text(encoding="utf-8")
                lines = content.split("\n")
                if not lines or not lines[0].startswith("#!"):
                    warnings.append("Script missing shebang (e.g., #!/bin/bash)")
                elif "bash" not in lines[0] and "sh" not in lines[0]:
                    warnings.append(f"Unexpected shebang: {lines[0]}")

                # Check script references MOSAIC_CONFIG_FILE
                if "MOSAIC_CONFIG_FILE" not in content:
                    warnings.append("Script doesn't reference $MOSAIC_CONFIG_FILE - may not read config")

            except Exception as e:
                errors.append(f"Cannot read script: {e}")

        # NOTE: We do NOT validate worker_config.method in script mode
        # The script is responsible for specifying its own algorithm

        # Build output message
        success = len(errors) == 0
        output_lines: List[str] = []

        if success:
            output_lines.append("[PASSED] Custom Script Validation")
            output_lines.append("")
            output_lines.append(f"Script: {script_path.name if script_path else 'N/A'}")
            if script_metadata.get("description"):
                output_lines.append(f"Description: {script_metadata['description']}")
            output_lines.append("")

            # Show what the script will ACTUALLY do
            output_lines.append("--- Script Configuration ---")
            if script_metadata.get("env_family"):
                output_lines.append(f"Target Environment Family: {script_metadata['env_family']}")
            if script_metadata.get("phases"):
                output_lines.append(f"Training Phases: {script_metadata['phases']}")
            if script_metadata.get("total_timesteps"):
                output_lines.append(f"Total Timesteps: {script_metadata['total_timesteps']:,}")

            if script_metadata.get("environments"):
                output_lines.append("")
                output_lines.append("Environments (in order):")
                for i, env in enumerate(script_metadata["environments"], 1):
                    output_lines.append(f"  Phase {i}: {env}")

            # Only show settings that are actually passed to and used by the script
            output_lines.append("")
            output_lines.append("--- Settings Inherited by Script ---")
            output_lines.append(f"Seed: {state.seed if state.seed else 'Not set (script/algorithm decides)'}")
            output_lines.append(f"Device: {state.device}")
            if state.fastlane_enabled:
                output_lines.append(f"FastLane: {state.fastlane_video_mode} mode")

            if warnings:
                output_lines.append("")
                output_lines.append("Warnings:")
                for w in warnings:
                    output_lines.append(f"  - {w}")

            output_lines.append("")
            output_lines.append("Note: Script controls algorithm, environments, and timesteps.")
        else:
            output_lines.append("[FAILED] Custom Script Validation")
            output_lines.append("")
            output_lines.append("Errors:")
            for e in errors:
                output_lines.append(f"  - {e}")

            if warnings:
                output_lines.append("")
                output_lines.append("Warnings:")
                for w in warnings:
                    output_lines.append(f"  - {w}")

        output = "\n".join(output_lines)

        # Log validation outcome
        if success:
            self.log_constant(
                LOG_UI_TRAIN_FORM_INFO,
                message="Custom script validation passed",
                extra={
                    "run_id": run_id,
                    "script_name": script_path.name if script_path else None,
                    "phases": script_metadata.get("phases"),
                    "environments": script_metadata.get("environments", []),
                    "warnings_count": len(warnings),
                },
            )
        else:
            self.log_constant(
                LOG_UI_TRAIN_FORM_ERROR,
                message="Custom script validation failed",
                extra={
                    "run_id": run_id,
                    "script_path": state.custom_script_path,
                    "errors": errors,
                    "warnings": warnings,
                },
            )

        self._set_validation_result(success, output)
        self._append_validation_notes(success, output)

        if persist_config and success:
            self._last_config = copy.deepcopy(config)
        elif not persist_config:
            self._last_config = None

        return success, (copy.deepcopy(config) if success else None)

    def _set_validation_result(self, success: bool, output: str) -> None:
        """Update validation status label based on result.

        Args:
            success: Whether validation succeeded.
            output: Output from the dry-run subprocess.
        """
        self._last_validation_output = output or ""
        if success:
            self._validation_status_label.setText("Dry-run validation succeeded.")
            self._validation_status_label.setStyleSheet("color: #2e7d32;")
            self.log_constant(
                LOG_UI_TRAIN_FORM_INFO,
                message="XuanCe dry-run validation succeeded",
            )
        else:
            self._validation_status_label.setText(
                "Dry-run validation failed. Check the details in Notes."
            )
            self._validation_status_label.setStyleSheet("color: #c62828;")
            snippet = (output or "").strip()
            self.log_constant(
                LOG_UI_TRAIN_FORM_ERROR,
                message="XuanCe dry-run validation failed",
                extra={"output": snippet[:1000]},
            )

    def _append_validation_notes(self, success: bool, output: str) -> None:
        """Append validation result to the Notes field.

        Args:
            success: Whether validation succeeded.
            output: Output from the dry-run subprocess.
        """
        status = "SUCCESS" if success else "FAILED"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        details = output.strip() if output else (
            "Dry-run completed." if success else "Dry-run failed without output."
        )
        entry = (
            f"[Dry-Run {status} — {timestamp}]\n"
            f"{details}\n"
            f"{'-' * 40}\n"
        )
        self._notes_edit.appendPlainText(entry)

    def _on_validate_clicked(self) -> None:
        """Handle validate button click - run dry-run without persisting config."""
        state = self._collect_state()

        # Branch validation based on mode: custom script vs standard training
        if state.custom_script_path:
            # Custom script mode: script controls algo/env/timesteps, only validate script exists
            script_path = Path(state.custom_script_path)
            if not script_path.exists():
                self.log_constant(
                    LOG_UI_TRAIN_FORM_ERROR,
                    message="Validation rejected: custom script not found",
                    extra={"script_path": state.custom_script_path},
                )
                QtWidgets.QMessageBox.warning(
                    self,
                    "Script Not Found",
                    f"Custom script not found:\n{state.custom_script_path}",
                )
                return
            script_name = script_path.stem.replace("_", "-")
            run_id = _generate_run_id("xuance-script", script_name)
        else:
            # Standard training mode: validate form fields
            if not state.method or not state.env_id:
                self.log_constant(
                    LOG_UI_TRAIN_FORM_WARNING,
                    message="Validation rejected: incomplete configuration",
                    extra={"has_method": bool(state.method), "has_env_id": bool(state.env_id)},
                )
                QtWidgets.QMessageBox.warning(
                    self,
                    "Incomplete Configuration",
                    "Select an algorithm and environment before running validation.",
                )
                return
            run_id = _generate_run_id("xuance", state.method)

        self._run_validation(state, run_id=run_id, persist_config=False)

    def _handle_accept(self) -> None:
        """Handle OK button click - validates before accepting."""
        state = self._collect_state()

        # Branch validation based on mode: custom script vs standard training
        if state.custom_script_path:
            # Custom script mode: script controls algo/env/timesteps, only validate script exists
            script_path = Path(state.custom_script_path)
            if not script_path.exists():
                self.log_constant(
                    LOG_UI_TRAIN_FORM_ERROR,
                    message="Custom script not found",
                    extra={"script_path": state.custom_script_path},
                )
                QtWidgets.QMessageBox.warning(
                    self,
                    "Script Not Found",
                    f"Custom script not found:\n{state.custom_script_path}",
                )
                return
            # Use script name for run_id generation
            script_name = script_path.stem.replace("_", "-")
            run_id = _generate_run_id("xuance-script", script_name)
            self.log_constant(
                LOG_UI_TRAIN_FORM_INFO,
                message="Accepting custom script training config",
                extra={"script_name": script_path.stem, "run_id": run_id},
            )
        else:
            # Standard training mode: validate form fields
            if not state.method:
                self.log_constant(
                    LOG_UI_TRAIN_FORM_WARNING,
                    message="Accept rejected: algorithm not selected",
                )
                QtWidgets.QMessageBox.warning(
                    self,
                    "Algorithm Required",
                    "Please select an algorithm before proceeding.",
                )
                return

            if not state.env_id:
                self.log_constant(
                    LOG_UI_TRAIN_FORM_WARNING,
                    message="Accept rejected: environment not selected",
                )
                QtWidgets.QMessageBox.warning(
                    self,
                    "Environment Required",
                    "Please select an environment before proceeding.",
                )
                return

            run_id = _generate_run_id("xuance", state.method)
            self.log_constant(
                LOG_UI_TRAIN_FORM_INFO,
                message="Accepting standard training config",
                extra={"method": state.method, "env_id": state.env_id, "run_id": run_id},
            )

        # Run validation before accepting
        success, config = self._run_validation(state, run_id=run_id, persist_config=True)
        if not success:
            self._last_config = None
            return

        self._last_config = config

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
