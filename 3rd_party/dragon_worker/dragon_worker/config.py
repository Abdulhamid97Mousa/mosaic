"""Configuration for Dragon (Komodo Dragon) Worker.

Dragon is a UCI-compatible chess engine that can be downloaded from:
https://komodochess.com/installation.htm

Dragon skill levels map to approximate Elo ratings:
    Elo = 125 * (level + 1)

    Level 1  = ~250 Elo  (Beginner)
    Level 3  = ~500 Elo  (Novice)
    Level 5  = ~750 Elo  (Casual)
    Level 7  = ~1000 Elo (Club player)
    Level 10 = ~1375 Elo (Intermediate)
    Level 15 = ~2000 Elo (Expert)
    Level 20 = ~2625 Elo (Master)
    Level 25 = ~3250 Elo (Maximum)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# Dragon difficulty presets based on Elo = 125 * (level + 1)
DIFFICULTY_PRESETS: dict[str, dict[str, Any]] = {
    "beginner": {
        "skill_level": 1,
        "depth": 5,
        "time_limit_ms": 500,
        "description": "~250 Elo - Complete beginner",
    },
    "novice": {
        "skill_level": 3,
        "depth": 8,
        "time_limit_ms": 500,
        "description": "~500 Elo - Basic understanding",
    },
    "easy": {
        "skill_level": 5,
        "depth": 10,
        "time_limit_ms": 750,
        "description": "~750 Elo - Casual player",
    },
    "medium": {
        "skill_level": 7,
        "depth": 12,
        "time_limit_ms": 1000,
        "description": "~1000 Elo - Club player",
    },
    "hard": {
        "skill_level": 10,
        "depth": 15,
        "time_limit_ms": 1500,
        "description": "~1375 Elo - Intermediate",
    },
    "expert": {
        "skill_level": 15,
        "depth": 18,
        "time_limit_ms": 2000,
        "description": "~2000 Elo - Expert level",
    },
    "master": {
        "skill_level": 20,
        "depth": 20,
        "time_limit_ms": 2500,
        "description": "~2625 Elo - Master strength",
    },
    "maximum": {
        "skill_level": 25,
        "depth": 25,
        "time_limit_ms": 3000,
        "description": "~3250 Elo - Full strength",
    },
}


def _find_dragon_binary() -> Optional[str]:
    """Try to locate Dragon binary on the system.

    Dragon is typically placed in a 'dragon' directory with platform-specific names:
    - dragon-osx (macOS)
    - dragon-linux (Linux)
    - dragon.exe (Windows)
    """
    import platform

    system = platform.system().lower()

    # Platform-specific binary names
    if system == "darwin":
        binary_names = ["dragon-osx", "dragon", "komodo-dragon"]
    elif system == "linux":
        binary_names = ["dragon-linux", "dragon", "komodo-dragon"]
    elif system == "windows":
        binary_names = ["dragon.exe", "komodo-dragon.exe"]
    else:
        binary_names = ["dragon", "komodo-dragon"]

    # Common installation locations
    search_paths = [
        Path.cwd() / "dragon",  # ./dragon/
        Path.home() / "dragon",  # ~/dragon/
        Path.home() / ".local" / "bin",  # ~/.local/bin/
        Path("/usr/local/bin"),
        Path("/opt/dragon"),
    ]

    # Check PATH first
    for name in binary_names:
        path = shutil.which(name)
        if path:
            return path

    # Check common locations
    for search_dir in search_paths:
        if search_dir.exists():
            for name in binary_names:
                candidate = search_dir / name
                if candidate.exists() and candidate.is_file():
                    return str(candidate)

    return None


@dataclass
class DragonWorkerConfig:
    """Configuration for Dragon chess engine worker.

    Implements the MOSAIC WorkerConfig protocol.

    Attributes:
        run_id: Unique identifier for this run
        seed: Random seed for reproducibility
        difficulty: Preset difficulty level
        skill_level: Dragon skill level (1-25)
        depth: Maximum search depth (1-30)
        time_limit_ms: Time limit per move in milliseconds
        dragon_path: Path to Dragon binary (auto-detected if not provided)
        remove_history: Reset board history before each move
    """

    run_id: str = ""
    seed: Optional[int] = None
    difficulty: str = "medium"
    skill_level: Optional[int] = None
    depth: Optional[int] = None
    time_limit_ms: int = 1000
    dragon_path: Optional[str] = None
    remove_history: bool = False

    # Internal fields populated from presets
    _preset_applied: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and apply difficulty presets."""
        # Validate difficulty
        valid_difficulties = list(DIFFICULTY_PRESETS.keys())
        if self.difficulty not in valid_difficulties:
            raise ValueError(
                f"difficulty must be one of {valid_difficulties}, got '{self.difficulty}'"
            )

        # Apply preset values (if not explicitly overridden)
        preset = DIFFICULTY_PRESETS[self.difficulty]
        if self.skill_level is None:
            self.skill_level = preset["skill_level"]
        if self.depth is None:
            self.depth = preset["depth"]
        if self.time_limit_ms == 1000:  # Default value, might want preset
            self.time_limit_ms = preset["time_limit_ms"]

        self._preset_applied = True

        # Validate ranges (skill_level and depth are guaranteed non-None after preset application)
        skill = self.skill_level
        depth = self.depth
        if skill is None or depth is None:
            raise ValueError("skill_level and depth must be set (either explicitly or via preset)")

        if not 1 <= skill <= 25:
            raise ValueError(
                f"skill_level must be between 1 and 25, got {skill}"
            )

        if not 1 <= depth <= 30:
            raise ValueError(
                f"depth must be between 1 and 30, got {depth}"
            )

        if self.time_limit_ms < 100:
            raise ValueError(
                f"time_limit_ms must be at least 100, got {self.time_limit_ms}"
            )

    @property
    def estimated_elo(self) -> int:
        """Calculate estimated Elo rating based on skill level.

        Uses the formula: Elo = 125 * (level + 1)
        """
        level = self.skill_level if self.skill_level is not None else 7
        return 125 * (level + 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        d = asdict(self)
        d.pop("_preset_applied", None)
        d["estimated_elo"] = self.estimated_elo
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DragonWorkerConfig":
        """Create config from dictionary."""
        # Remove computed fields
        data = {k: v for k, v in data.items()
                if k not in ("_preset_applied", "estimated_elo")}
        return cls(**data)

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        run_id: str = "",
        **overrides: Any,
    ) -> "DragonWorkerConfig":
        """Create config from a named preset.

        Args:
            preset_name: One of the DIFFICULTY_PRESETS keys
            run_id: Unique run identifier
            **overrides: Additional config overrides

        Returns:
            DragonWorkerConfig with preset values applied
        """
        return cls(run_id=run_id, difficulty=preset_name, **overrides)


def load_worker_config(config_path: str) -> DragonWorkerConfig:
    """Load Dragon worker config from a JSON file.

    Supports multiple config formats:
    1. Direct format: {"run_id": "...", "difficulty": "...", ...}
    2. Nested format: {"metadata": {"worker": {"config": {...}}}}
    3. Operator format: {"run_id": "...", "settings": {...}}

    Args:
        config_path: Path to JSON configuration file

    Returns:
        DragonWorkerConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        data = json.load(f)

    # Try nested format (GUI metadata format)
    if "metadata" in data and "worker" in data["metadata"]:
        worker_data = data["metadata"]["worker"]
        if "config" in worker_data:
            return DragonWorkerConfig.from_dict(worker_data["config"])

    # Try operator settings format
    if "settings" in data:
        config_data = {"run_id": data.get("run_id", "")}
        config_data.update(data["settings"])
        return DragonWorkerConfig.from_dict(config_data)

    # Direct format
    return DragonWorkerConfig.from_dict(data)
