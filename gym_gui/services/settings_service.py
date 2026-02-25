"""Settings service for managing environment variables from .env file.

This module provides a centralized service for reading and writing application
settings stored in the .env file. It includes metadata for all environment variables,
type-safe validation, and safe file operations using python-dotenv.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:
    from dotenv import get_key, set_key
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    get_key = None
    set_key = None


_LOGGER = logging.getLogger(__name__)


class SettingType(Enum):
    """Types of settings for appropriate widget rendering and validation."""

    STRING = "string"       # General text input
    BOOLEAN = "boolean"     # Checkbox (0/1, true/false)
    INTEGER = "integer"     # Numeric spinner
    URL = "url"             # URL input with validation
    EMAIL = "email"         # Email input with validation
    ENUM = "enum"           # Dropdown with fixed options


@dataclass
class SettingMetadata:
    """Metadata for a single environment variable.

    Attributes:
        key: Environment variable name (e.g., "QT_API")
        category: Category for UI grouping (e.g., "Qt Configuration")
        description: User-friendly description for tooltips
        default_value: Default value if not set in .env
        value_type: Type of setting (determines widget and validation)
        is_sensitive: Whether to mask input (passwords, API keys)
        enum_options: List of valid options for ENUM type
        validation_regex: Optional regex for additional validation
        requires_restart: Whether changing this requires app restart
    """

    key: str
    category: str
    description: str
    default_value: str
    value_type: SettingType
    is_sensitive: bool = False
    enum_options: Optional[List[str]] = None
    validation_regex: Optional[str] = None
    requires_restart: bool = False


class SettingsService:
    """Service for managing application settings in .env file.

    This service provides:
    - Metadata for all environment variables
    - Type-safe reading and writing using dotenv
    - Validation for different value types
    - Search functionality
    - Category-based organization

    Example:
        service = SettingsService(Path(".env"))
        categories = service.get_categories()
        settings = service.get_settings_by_category("Qt Configuration")
        value = service.get_value("QT_API")
        service.set_value("QT_API", "PyQt6")
    """

    def __init__(self, env_path: Path) -> None:
        """Initialize the settings service.

        Args:
            env_path: Path to the .env file
        """
        if not DOTENV_AVAILABLE:
            raise ImportError(
                "python-dotenv is required for settings management. "
                "Install with: pip install python-dotenv"
            )

        self._env_path = env_path
        self._settings_metadata = self._build_metadata()
        _LOGGER.info(f"Settings service initialized with {len(self._settings_metadata)} settings")

    def _build_metadata(self) -> Dict[str, SettingMetadata]:
        """Build complete metadata for all settings from .env.example.

        Returns:
            Dictionary mapping environment variable name to its metadata
        """
        metadata: Dict[str, SettingMetadata] = {}

        # Category 1: Qt Configuration (3 variables)
        metadata["QT_API"] = SettingMetadata(
            key="QT_API",
            category="Qt Configuration",
            description="Qt API to use (must be PyQt6, case-sensitive)",
            default_value="PyQt6",
            value_type=SettingType.ENUM,
            enum_options=["PyQt6", "PySide6"],
            requires_restart=True,
        )
        metadata["QT_DEBUG_PLUGINS"] = SettingMetadata(
            key="QT_DEBUG_PLUGINS",
            category="Qt Configuration",
            description="Enable Qt plugin debugging (0=disabled, 1=enabled)",
            default_value="0",
            value_type=SettingType.BOOLEAN,
        )
        metadata["QML_DISABLE_DISK_CACHE"] = SettingMetadata(
            key="QML_DISABLE_DISK_CACHE",
            category="Qt Configuration",
            description="Disable QML disk cache (0=use cache, 1=disable cache)",
            default_value="1",
            value_type=SettingType.BOOLEAN,
        )

        # Category 2: Gymnasium Defaults (2 variables)
        metadata["GYM_CONTROL_MODE"] = SettingMetadata(
            key="GYM_CONTROL_MODE",
            category="Gymnasium Defaults",
            description="Default control mode for environments",
            default_value="human_only",
            value_type=SettingType.ENUM,
            enum_options=["human_only", "agent_only", "hybrid_turn_based",
                         "hybrid_human_agent", "multi_agent_coop", "multi_agent_competitive"],
        )
        metadata["GYM_LOG_LEVEL"] = SettingMetadata(
            key="GYM_LOG_LEVEL",
            category="Gymnasium Defaults",
            description="Python logging level for gymnasium",
            default_value="DEBUG",
            value_type=SettingType.ENUM,
            enum_options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        )

        # Category 3: Environment Overrides - Generic Defaults (3 variables)
        metadata["DEFAULT_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="DEFAULT_MAX_EPISODE_STEPS",
            category="Environment Overrides",
            description="Default max episode steps for all environments (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["DEFAULT_MAX_EPISODE_SECONDS"] = SettingMetadata(
            key="DEFAULT_MAX_EPISODE_SECONDS",
            category="Environment Overrides",
            description="Default max episode seconds for all environments (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["DEFAULT_RENDER_FPS"] = SettingMetadata(
            key="DEFAULT_RENDER_FPS",
            category="Environment Overrides",
            description="Default rendering FPS for all environments",
            default_value="30",
            value_type=SettingType.INTEGER,
        )

        # Category 4: Classic Control Environments (4 variables)
        metadata["CART_POLE_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="CART_POLE_MAX_EPISODE_STEPS",
            category="Classic Control",
            description="CartPole max episode steps (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["ACROBOT_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="ACROBOT_MAX_EPISODE_STEPS",
            category="Classic Control",
            description="Acrobot max episode steps (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["MOUNTAIN_CAR_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="MOUNTAIN_CAR_MAX_EPISODE_STEPS",
            category="Classic Control",
            description="MountainCar max episode steps (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["PENDULUM_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="PENDULUM_MAX_EPISODE_STEPS",
            category="Classic Control",
            description="Pendulum max episode steps (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )

        # Category 5: Box2D Environments (9 variables)
        metadata["CAR_RACING_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="CAR_RACING_MAX_EPISODE_STEPS",
            category="Box2D Environments",
            description="CarRacing max episode steps (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["CAR_RACING_MAX_EPISODE_SECONDS"] = SettingMetadata(
            key="CAR_RACING_MAX_EPISODE_SECONDS",
            category="Box2D Environments",
            description="CarRacing max episode seconds (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["CAR_RACING_LAP_COMPLETE_PERCENT"] = SettingMetadata(
            key="CAR_RACING_LAP_COMPLETE_PERCENT",
            category="Box2D Environments",
            description="CarRacing lap completion threshold (0.0-1.0)",
            default_value="0.95",
            value_type=SettingType.STRING,  # Float as string
        )
        metadata["BIPEDAL_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="BIPEDAL_MAX_EPISODE_STEPS",
            category="Box2D Environments",
            description="Bipedal Walker max episode steps (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["BIPEDAL_MAX_EPISODE_SECONDS"] = SettingMetadata(
            key="BIPEDAL_MAX_EPISODE_SECONDS",
            category="Box2D Environments",
            description="Bipedal Walker max episode seconds (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["BIPEDAL_HARDCORE"] = SettingMetadata(
            key="BIPEDAL_HARDCORE",
            category="Box2D Environments",
            description="Enable hardcore mode for Bipedal Walker",
            default_value="false",
            value_type=SettingType.BOOLEAN,
        )
        metadata["LUNAR_LANDER_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="LUNAR_LANDER_MAX_EPISODE_STEPS",
            category="Box2D Environments",
            description="Lunar Lander max episode steps (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["LUNAR_LANDER_CONTINUOUS"] = SettingMetadata(
            key="LUNAR_LANDER_CONTINUOUS",
            category="Box2D Environments",
            description="Use continuous action space for Lunar Lander",
            default_value="false",
            value_type=SettingType.BOOLEAN,
        )

        # Category 6: Toy Text Environments (7 variables)
        metadata["FROZEN_LAKE_IS_SLIPPERY"] = SettingMetadata(
            key="FROZEN_LAKE_IS_SLIPPERY",
            category="Toy Text Environments",
            description="Enable slippery ice in FrozenLake",
            default_value="true",
            value_type=SettingType.BOOLEAN,
        )
        metadata["FROZEN_LAKE_MAP_NAME"] = SettingMetadata(
            key="FROZEN_LAKE_MAP_NAME",
            category="Toy Text Environments",
            description="FrozenLake map size (4x4 or 8x8)",
            default_value="4x4",
            value_type=SettingType.ENUM,
            enum_options=["4x4", "8x8"],
        )
        metadata["CLIFF_WALKING_IS_SLIPPERY"] = SettingMetadata(
            key="CLIFF_WALKING_IS_SLIPPERY",
            category="Toy Text Environments",
            description="Enable slippery movement in CliffWalking",
            default_value="false",
            value_type=SettingType.BOOLEAN,
        )
        metadata["TAXI_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="TAXI_MAX_EPISODE_STEPS",
            category="Toy Text Environments",
            description="Taxi max episode steps (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["BLACKJACK_NATURAL"] = SettingMetadata(
            key="BLACKJACK_NATURAL",
            category="Toy Text Environments",
            description="Blackjack natural (21 from first two cards) rule",
            default_value="false",
            value_type=SettingType.BOOLEAN,
        )
        metadata["BLACKJACK_SAB"] = SettingMetadata(
            key="BLACKJACK_SAB",
            category="Toy Text Environments",
            description="Blackjack SAB (stick and bust) rule",
            default_value="false",
            value_type=SettingType.BOOLEAN,
        )

        # Category 7: MuJoCo Environments (10 variables)
        metadata["MUJOCO_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="MUJOCO_MAX_EPISODE_STEPS",
            category="MuJoCo Environments",
            description="MuJoCo environments max episode steps (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["MUJOCO_FRAME_SKIP"] = SettingMetadata(
            key="MUJOCO_FRAME_SKIP",
            category="MuJoCo Environments",
            description="MuJoCo frame skip (physics steps per action)",
            default_value="5",
            value_type=SettingType.INTEGER,
        )
        metadata["MUJOCO_RENDER_FPS"] = SettingMetadata(
            key="MUJOCO_RENDER_FPS",
            category="MuJoCo Environments",
            description="MuJoCo rendering FPS",
            default_value="67",
            value_type=SettingType.INTEGER,
        )
        metadata["ANT_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="ANT_MAX_EPISODE_STEPS",
            category="MuJoCo Environments",
            description="Ant max episode steps (0=use defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["HALF_CHEETAH_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="HALF_CHEETAH_MAX_EPISODE_STEPS",
            category="MuJoCo Environments",
            description="HalfCheetah max episode steps (0=use defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["HOPPER_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="HOPPER_MAX_EPISODE_STEPS",
            category="MuJoCo Environments",
            description="Hopper max episode steps (0=use defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["HUMANOID_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="HUMANOID_MAX_EPISODE_STEPS",
            category="MuJoCo Environments",
            description="Humanoid max episode steps (0=use defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["WALKER2D_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="WALKER2D_MAX_EPISODE_STEPS",
            category="MuJoCo Environments",
            description="Walker2d max episode steps (0=use defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["SWIMMER_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="SWIMMER_MAX_EPISODE_STEPS",
            category="MuJoCo Environments",
            description="Swimmer max episode steps (0=use defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["REACHER_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="REACHER_MAX_EPISODE_STEPS",
            category="MuJoCo Environments",
            description="Reacher max episode steps (0=use defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )

        # Category 8: Atari Environments (5 variables)
        metadata["ATARI_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="ATARI_MAX_EPISODE_STEPS",
            category="Atari Environments",
            description="Atari environments max episode steps (0=use gym defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["ATARI_FRAMESKIP"] = SettingMetadata(
            key="ATARI_FRAMESKIP",
            category="Atari Environments",
            description="Atari frame skip (frames to repeat each action)",
            default_value="4",
            value_type=SettingType.INTEGER,
        )
        metadata["ATARI_REPEAT_ACTION_PROBABILITY"] = SettingMetadata(
            key="ATARI_REPEAT_ACTION_PROBABILITY",
            category="Atari Environments",
            description="Atari sticky actions probability (0.0-1.0)",
            default_value="0.0",
            value_type=SettingType.STRING,  # Float as string
        )
        metadata["ATARI_FULL_ACTION_SPACE"] = SettingMetadata(
            key="ATARI_FULL_ACTION_SPACE",
            category="Atari Environments",
            description="Use full action space (18 actions) instead of minimal",
            default_value="false",
            value_type=SettingType.BOOLEAN,
        )
        metadata["ATARI_RENDER_MODE"] = SettingMetadata(
            key="ATARI_RENDER_MODE",
            category="Atari Environments",
            description="Atari render mode",
            default_value="rgb_array",
            value_type=SettingType.ENUM,
            enum_options=["rgb_array", "human"],
        )

        # Category 9: MiniGrid Environments (5 variables)
        metadata["MINIGRID_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="MINIGRID_MAX_EPISODE_STEPS",
            category="MiniGrid Environments",
            description="MiniGrid max episode steps (0=use defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["MINIGRID_TILE_SIZE"] = SettingMetadata(
            key="MINIGRID_TILE_SIZE",
            category="MiniGrid Environments",
            description="MiniGrid tile size in pixels",
            default_value="32",
            value_type=SettingType.INTEGER,
        )
        metadata["MINIGRID_AGENT_VIEW_SIZE"] = SettingMetadata(
            key="MINIGRID_AGENT_VIEW_SIZE",
            category="MiniGrid Environments",
            description="MiniGrid agent's view size (grid cells)",
            default_value="7",
            value_type=SettingType.INTEGER,
        )
        metadata["MINIGRID_HIGHLIGHT"] = SettingMetadata(
            key="MINIGRID_HIGHLIGHT",
            category="MiniGrid Environments",
            description="Highlight agent's view area in MiniGrid",
            default_value="true",
            value_type=SettingType.BOOLEAN,
        )
        metadata["MINIGRID_AGENT_PNOV"] = SettingMetadata(
            key="MINIGRID_AGENT_PNOV",
            category="MiniGrid Environments",
            description="MiniGrid partially observable view",
            default_value="false",
            value_type=SettingType.BOOLEAN,
        )

        # Category 10: ViZDoom Environments (5 variables)
        metadata["VIZDOOM_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="VIZDOOM_MAX_EPISODE_STEPS",
            category="ViZDoom Environments",
            description="ViZDoom max episode steps (0=use defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["VIZDOOM_FRAME_SKIP"] = SettingMetadata(
            key="VIZDOOM_FRAME_SKIP",
            category="ViZDoom Environments",
            description="ViZDoom frame skip",
            default_value="4",
            value_type=SettingType.INTEGER,
        )
        metadata["VIZDOOM_RENDER_HUD"] = SettingMetadata(
            key="VIZDOOM_RENDER_HUD",
            category="ViZDoom Environments",
            description="Render HUD (heads-up display) in ViZDoom",
            default_value="false",
            value_type=SettingType.BOOLEAN,
        )
        metadata["VIZDOOM_RENDER_CROSSHAIR"] = SettingMetadata(
            key="VIZDOOM_RENDER_CROSSHAIR",
            category="ViZDoom Environments",
            description="Render crosshair in ViZDoom",
            default_value="false",
            value_type=SettingType.BOOLEAN,
        )
        metadata["VIZDOOM_RENDER_WEAPON"] = SettingMetadata(
            key="VIZDOOM_RENDER_WEAPON",
            category="ViZDoom Environments",
            description="Render weapon in ViZDoom",
            default_value="true",
            value_type=SettingType.BOOLEAN,
        )

        # Category 11: MiniHack/NetHack Environments (3 variables)
        metadata["MINIHACK_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="MINIHACK_MAX_EPISODE_STEPS",
            category="MiniHack/NetHack",
            description="MiniHack max episode steps (0=use defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["NETHACK_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="NETHACK_MAX_EPISODE_STEPS",
            category="MiniHack/NetHack",
            description="NetHack max episode steps (0=use defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["NETHACK_OBSERVATION_KEYS"] = SettingMetadata(
            key="NETHACK_OBSERVATION_KEYS",
            category="MiniHack/NetHack",
            description="NetHack observation keys (comma-separated)",
            default_value="glyphs,blstats,message",
            value_type=SettingType.STRING,
        )

        # Category 12: Crafter Environment (3 variables)
        metadata["CRAFTER_MAX_EPISODE_STEPS"] = SettingMetadata(
            key="CRAFTER_MAX_EPISODE_STEPS",
            category="Crafter",
            description="Crafter max episode steps (0=use defaults)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["CRAFTER_AREA_SIZE"] = SettingMetadata(
            key="CRAFTER_AREA_SIZE",
            category="Crafter",
            description="Crafter world area size (e.g., 64x64)",
            default_value="64x64",
            value_type=SettingType.STRING,
        )
        metadata["CRAFTER_VIEW_SIZE"] = SettingMetadata(
            key="CRAFTER_VIEW_SIZE",
            category="Crafter",
            description="Crafter agent view size (e.g., 9x9)",
            default_value="9x9",
            value_type=SettingType.STRING,
        )

        # Category 13: Procgen Environments (7 variables)
        metadata["PROCGEN_NUM_LEVELS"] = SettingMetadata(
            key="PROCGEN_NUM_LEVELS",
            category="Procgen",
            description="Procgen number of unique levels (0=unlimited)",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["PROCGEN_START_LEVEL"] = SettingMetadata(
            key="PROCGEN_START_LEVEL",
            category="Procgen",
            description="Procgen starting level seed",
            default_value="0",
            value_type=SettingType.INTEGER,
        )
        metadata["PROCGEN_PAINT_VEL_INFO"] = SettingMetadata(
            key="PROCGEN_PAINT_VEL_INFO",
            category="Procgen",
            description="Paint velocity info on Procgen observations",
            default_value="false",
            value_type=SettingType.BOOLEAN,
        )
        metadata["PROCGEN_USE_GENERATED_ASSETS"] = SettingMetadata(
            key="PROCGEN_USE_GENERATED_ASSETS",
            category="Procgen",
            description="Use procedurally generated assets in Procgen",
            default_value="false",
            value_type=SettingType.BOOLEAN,
        )
        metadata["PROCGEN_CENTER_AGENT"] = SettingMetadata(
            key="PROCGEN_CENTER_AGENT",
            category="Procgen",
            description="Center agent in Procgen view",
            default_value="true",
            value_type=SettingType.BOOLEAN,
        )
        metadata["PROCGEN_USE_SEQUENTIAL_LEVELS"] = SettingMetadata(
            key="PROCGEN_USE_SEQUENTIAL_LEVELS",
            category="Procgen",
            description="Use sequential Procgen levels instead of random",
            default_value="false",
            value_type=SettingType.BOOLEAN,
        )
        metadata["PROCGEN_DISTRIBUTION_MODE"] = SettingMetadata(
            key="PROCGEN_DISTRIBUTION_MODE",
            category="Procgen",
            description="Procgen difficulty distribution mode",
            default_value="easy",
            value_type=SettingType.ENUM,
            enum_options=["easy", "hard", "extreme", "memory", "exploration"],
        )

        # Category 14: Platform & Graphics (2 variables)
        metadata["PLATFORM"] = SettingMetadata(
            key="PLATFORM",
            category="Platform & Graphics",
            description="Platform identifier (ubuntu, windows, macos)",
            default_value="ubuntu",
            value_type=SettingType.ENUM,
            enum_options=["ubuntu", "windows", "macos"],
        )
        metadata["MUJOCO_GL"] = SettingMetadata(
            key="MUJOCO_GL",
            category="Platform & Graphics",
            description="MuJoCo graphics backend (egl, glfw, osmesa)",
            default_value="egl",
            value_type=SettingType.ENUM,
            enum_options=["egl", "glfw", "osmesa"],
        )

        # Category 5: gRPC (2 variables)
        metadata["GRPC_VERBOSITY"] = SettingMetadata(
            key="GRPC_VERBOSITY",
            category="gRPC",
            description="gRPC logging verbosity level",
            default_value="debug",
            value_type=SettingType.ENUM,
            enum_options=["debug", "info", "error"],
        )
        # Category 6: LLM & Chat (6 variables, 3 sensitive)
        metadata["OPENROUTER_API_KEY"] = SettingMetadata(
            key="OPENROUTER_API_KEY",
            category="LLM & Chat",
            description="OpenRouter API key for cloud LLM access",
            default_value="",
            value_type=SettingType.STRING,
            is_sensitive=True,
        )
        metadata["VLLM_BASE_URL"] = SettingMetadata(
            key="VLLM_BASE_URL",
            category="LLM & Chat",
            description="vLLM server base URL for local LLM",
            default_value="http://localhost:8000/v1",
            value_type=SettingType.URL,
        )
        metadata["VLLM_API_KEY"] = SettingMetadata(
            key="VLLM_API_KEY",
            category="LLM & Chat",
            description="vLLM server API key (if authentication enabled)",
            default_value="",
            value_type=SettingType.STRING,
            is_sensitive=True,
        )
        metadata["HF_TOKEN"] = SettingMetadata(
            key="HF_TOKEN",
            category="LLM & Chat",
            description="HuggingFace token for downloading gated models",
            default_value="",
            value_type=SettingType.STRING,
            is_sensitive=True,
        )
        metadata["HTTP_PROXY"] = SettingMetadata(
            key="HTTP_PROXY",
            category="LLM & Chat",
            description="HTTP proxy URL for HuggingFace downloads",
            default_value="",
            value_type=SettingType.URL,
        )
        metadata["HTTPS_PROXY"] = SettingMetadata(
            key="HTTPS_PROXY",
            category="LLM & Chat",
            description="HTTPS proxy URL for HuggingFace downloads",
            default_value="",
            value_type=SettingType.URL,
        )

        # Category 7: Weights & Biases (7 variables, 2 sensitive)
        metadata["WANDB_API_KEY"] = SettingMetadata(
            key="WANDB_API_KEY",
            category="Weights & Biases",
            description="Weights & Biases API key for experiment tracking",
            default_value="",
            value_type=SettingType.STRING,
            is_sensitive=True,
        )
        metadata["WANDB_PROJECT_NAME"] = SettingMetadata(
            key="WANDB_PROJECT_NAME",
            category="Weights & Biases",
            description="wandb project name",
            default_value="MOSAIC",
            value_type=SettingType.STRING,
        )
        metadata["WANDB_ENTITY_NAME"] = SettingMetadata(
            key="WANDB_ENTITY_NAME",
            category="Weights & Biases",
            description="wandb entity (username or team name)",
            default_value="",
            value_type=SettingType.STRING,
        )
        metadata["WANDB_EMAIL"] = SettingMetadata(
            key="WANDB_EMAIL",
            category="Weights & Biases",
            description="Email address for wandb account",
            default_value="",
            value_type=SettingType.EMAIL,
            is_sensitive=True,
        )
        metadata["WANDB_VPN_HTTPS_PROXY"] = SettingMetadata(
            key="WANDB_VPN_HTTPS_PROXY",
            category="Weights & Biases",
            description="HTTPS proxy URL for wandb (if using VPN)",
            default_value="",
            value_type=SettingType.URL,
        )
        metadata["WANDB_VPN_HTTP_PROXY"] = SettingMetadata(
            key="WANDB_VPN_HTTP_PROXY",
            category="Weights & Biases",
            description="HTTP proxy URL for wandb (if using VPN)",
            default_value="",
            value_type=SettingType.URL,
        )
        metadata["WANDB_MONITOR_GYM"] = SettingMetadata(
            key="WANDB_MONITOR_GYM",
            category="Weights & Biases",
            description="Enable wandb monitoring for gymnasium (0=disabled, 1=enabled)",
            default_value="0",
            value_type=SettingType.BOOLEAN,
        )

        # Category 8: Python Settings (1 variable)
        metadata["PYTHONWARNINGS"] = SettingMetadata(
            key="PYTHONWARNINGS",
            category="Python Settings",
            description="Python warnings filter (e.g., ignore::UserWarning:pkg_resources)",
            default_value="ignore::UserWarning:pkg_resources",
            value_type=SettingType.STRING,
        )

        return metadata

    def get_categories(self) -> List[str]:
        """Get list of unique categories in order.

        Returns:
            List of category names in the order they should be displayed
        """
        # Return categories in the order defined above
        return [
            "Qt Configuration",
            "Gymnasium Defaults",
            "Environment Overrides",  # Generic defaults
            "Classic Control",
            "Box2D Environments",
            "Toy Text Environments",
            "MuJoCo Environments",
            "Atari Environments",
            "MiniGrid Environments",
            "ViZDoom Environments",
            "MiniHack/NetHack",
            "Crafter",
            "Procgen",
            "Platform & Graphics",
            "gRPC",
            "LLM & Chat",
            "Weights & Biases",
            "Python Settings",
        ]

    def get_settings_by_category(self, category: str) -> List[SettingMetadata]:
        """Get all settings for a given category.

        Args:
            category: Category name

        Returns:
            List of SettingMetadata for the category, sorted by key
        """
        settings = [
            metadata for metadata in self._settings_metadata.values()
            if metadata.category == category
        ]
        return sorted(settings, key=lambda s: s.key)

    def get_value(self, key: str) -> Optional[str]:
        """Get current value from .env file.

        Args:
            key: Environment variable name

        Returns:
            Current value from .env file, or None if not set
        """
        try:
            # First check if file exists
            if not self._env_path.exists():
                _LOGGER.debug(f".env file does not exist at {self._env_path}")
                return None

            # Use dotenv to get value
            value = get_key(str(self._env_path), key)
            return value
        except Exception as e:
            _LOGGER.error(f"Error reading {key} from .env: {e}")
            return None

    def set_value(self, key: str, value: str) -> bool:
        """Write value to .env file immediately using dotenv.set_key().

        Args:
            key: Environment variable name
            value: New value to set

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure .env file exists
            if not self._env_path.exists():
                self._env_path.touch()
                _LOGGER.info(f"Created .env file at {self._env_path}")

            # Use dotenv to set value (atomic and safe)
            result = set_key(str(self._env_path), key, value)

            if result[0]:  # set_key returns (success, key, value)
                _LOGGER.debug(f"Set {key} in .env")
                # Also update os.environ for current session
                os.environ[key] = value
                return True
            else:
                _LOGGER.error(f"Failed to set {key} in .env")
                return False
        except Exception as e:
            _LOGGER.error(f"Error writing {key} to .env: {e}")
            return False

    def validate_value(self, metadata: SettingMetadata, value: str) -> Tuple[bool, str]:
        """Validate a value according to its type.

        Args:
            metadata: Setting metadata with type information
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
                is_valid: True if valid, False otherwise
                error_message: Empty string if valid, error description if invalid
        """
        if not value and metadata.value_type != SettingType.STRING:
            # Empty values are allowed for STRING type, but may be invalid for others
            # However, empty is often used for optional settings, so we'll allow it
            return True, ""

        if metadata.value_type == SettingType.BOOLEAN:
            return self._validate_boolean(value)
        elif metadata.value_type == SettingType.INTEGER:
            return self._validate_integer(value)
        elif metadata.value_type == SettingType.URL:
            return self._validate_url(value)
        elif metadata.value_type == SettingType.EMAIL:
            return self._validate_email(value)
        elif metadata.value_type == SettingType.ENUM:
            return self._validate_enum(value, metadata.enum_options or [])
        else:  # STRING or unknown
            return True, ""

    def _validate_boolean(self, value: str) -> Tuple[bool, str]:
        """Validate boolean value."""
        valid_values = {"0", "1", "true", "false", "yes", "no", "on", "off"}
        if value.lower() in valid_values:
            return True, ""
        return False, f"Must be one of: {', '.join(sorted(valid_values))}"

    def _validate_integer(self, value: str) -> Tuple[bool, str]:
        """Validate integer value."""
        try:
            int(value)
            return True, ""
        except ValueError:
            return False, "Must be a valid integer"

    def _validate_url(self, value: str) -> Tuple[bool, str]:
        """Validate URL format."""
        if not value:
            return True, ""  # Empty is okay (optional proxy)

        try:
            result = urlparse(value)
            if not all([result.scheme, result.netloc]):
                return False, "Must be a valid URL (e.g., http://example.com)"
            return True, ""
        except Exception:
            return False, "Must be a valid URL"

    def _validate_email(self, value: str) -> Tuple[bool, str]:
        """Validate email format."""
        if not value:
            return True, ""  # Empty is okay (optional)

        # Simple email regex
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_regex, value):
            return True, ""
        return False, "Must be a valid email address"

    def _validate_enum(self, value: str, options: List[str]) -> Tuple[bool, str]:
        """Validate enum value."""
        if value in options:
            return True, ""
        return False, f"Must be one of: {', '.join(options)}"

    def search_settings(self, query: str) -> List[SettingMetadata]:
        """Search settings by key or description.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching SettingMetadata objects
        """
        query_lower = query.lower()
        matches = []

        for metadata in self._settings_metadata.values():
            if (query_lower in metadata.key.lower() or
                query_lower in metadata.description.lower() or
                query_lower in metadata.category.lower()):
                matches.append(metadata)

        return sorted(matches, key=lambda s: s.key)


__all__ = ["SettingType", "SettingMetadata", "SettingsService"]
