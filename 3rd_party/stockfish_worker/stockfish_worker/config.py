"""Configuration for Stockfish Worker.

This module provides the configuration dataclass for the Stockfish chess engine worker.
It implements the MOSAIC WorkerConfig protocol for integration with the operator system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


# Difficulty presets matching StockfishService
DIFFICULTY_PRESETS = {
    "beginner": {"skill_level": 1, "depth": 5, "time_limit_ms": 500},
    "easy": {"skill_level": 5, "depth": 8, "time_limit_ms": 500},
    "medium": {"skill_level": 10, "depth": 12, "time_limit_ms": 1000},
    "hard": {"skill_level": 15, "depth": 18, "time_limit_ms": 1500},
    "expert": {"skill_level": 20, "depth": 20, "time_limit_ms": 2000},
}


@dataclass
class StockfishWorkerConfig:
    """Configuration for Stockfish chess engine worker.

    This dataclass implements the WorkerConfig protocol for MOSAIC integration.

    Attributes:
        run_id: Unique run identifier (REQUIRED by protocol).
        seed: Random seed (REQUIRED by protocol, can be None).
        env_name: Environment family (always "pettingzoo" for chess).
        task: Environment task (always "chess_v6").
        difficulty: Difficulty preset ("beginner", "easy", "medium", "hard", "expert").
        skill_level: Stockfish skill level (0-20). Overrides difficulty preset.
        depth: Search depth (1-30). Overrides difficulty preset.
        time_limit_ms: Time limit per move in milliseconds.
        threads: Number of CPU threads to use.
        hash_mb: Hash table size in MB.
        stockfish_path: Custom path to Stockfish binary (auto-detected if None).
        telemetry_dir: Directory for telemetry output.
    """

    # Protocol-required fields
    run_id: str = ""
    seed: Optional[int] = None

    # Environment settings
    env_name: str = "pettingzoo"
    task: str = "chess_v6"

    # Difficulty settings
    difficulty: str = "medium"  # Preset name
    skill_level: Optional[int] = None  # Override preset
    depth: Optional[int] = None  # Override preset
    time_limit_ms: int = 1000
    threads: int = 1
    hash_mb: int = 16

    # Stockfish binary
    stockfish_path: Optional[str] = None  # Auto-detect if None

    # Telemetry
    telemetry_dir: str = "var/operators/telemetry"

    def __post_init__(self) -> None:
        """Validate configuration and apply difficulty presets."""
        # Validate difficulty preset
        if self.difficulty not in DIFFICULTY_PRESETS:
            valid = ", ".join(DIFFICULTY_PRESETS.keys())
            raise ValueError(f"difficulty must be one of: {valid}")

        # Apply difficulty preset if skill_level/depth not explicitly set
        preset = DIFFICULTY_PRESETS[self.difficulty]
        if self.skill_level is None:
            self.skill_level = preset["skill_level"]
        if self.depth is None:
            self.depth = preset["depth"]
        if self.time_limit_ms == 1000:  # Default value, use preset
            self.time_limit_ms = preset["time_limit_ms"]

        # Validate ranges
        if not (0 <= self.skill_level <= 20):
            raise ValueError("skill_level must be between 0 and 20")
        if not (1 <= self.depth <= 30):
            raise ValueError("depth must be between 1 and 30")
        if self.time_limit_ms < 100:
            raise ValueError("time_limit_ms must be at least 100")

        # Protocol compliance check
        try:
            from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
            assert isinstance(self, WorkerConfigProtocol), (
                "StockfishWorkerConfig must implement WorkerConfig protocol"
            )
        except ImportError:
            pass  # gym_gui not available, skip check

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (REQUIRED by protocol)."""
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "env_name": self.env_name,
            "task": self.task,
            "difficulty": self.difficulty,
            "skill_level": self.skill_level,
            "depth": self.depth,
            "time_limit_ms": self.time_limit_ms,
            "threads": self.threads,
            "hash_mb": self.hash_mb,
            "stockfish_path": self.stockfish_path,
            "telemetry_dir": self.telemetry_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StockfishWorkerConfig":
        """Create from dictionary (REQUIRED by protocol)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_preset(cls, difficulty: str, run_id: str = "") -> "StockfishWorkerConfig":
        """Create config from difficulty preset.

        Args:
            difficulty: One of "beginner", "easy", "medium", "hard", "expert"
            run_id: Unique run identifier

        Returns:
            StockfishWorkerConfig with preset values
        """
        return cls(run_id=run_id, difficulty=difficulty)


def load_worker_config(config_path: str) -> StockfishWorkerConfig:
    """Load worker configuration from JSON file.

    Handles both direct config format and nested metadata.worker.config structure.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Parsed StockfishWorkerConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw_data = json.load(f)

    # Handle nested metadata.worker.config structure from GUI
    if "metadata" in raw_data and "worker" in raw_data["metadata"]:
        worker_data = raw_data["metadata"]["worker"]
        config_data = worker_data.get("config", {})
    elif "settings" in raw_data:
        # Handle operator config format
        config_data = raw_data.get("settings", {})
        config_data["run_id"] = raw_data.get("run_id", "")
    else:
        # Direct config format
        config_data = raw_data

    return StockfishWorkerConfig.from_dict(config_data)


__all__ = [
    "StockfishWorkerConfig",
    "load_worker_config",
    "DIFFICULTY_PRESETS",
]
