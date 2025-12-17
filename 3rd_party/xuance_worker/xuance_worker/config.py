"""Configuration dataclass for XuanCe worker.

This module provides the XuanCeWorkerConfig dataclass which holds all
configuration required to execute a XuanCe training run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class XuanCeWorkerConfig:
    """Configuration for a single XuanCe training run.

    This dataclass captures all parameters needed to initialize and run
    a XuanCe algorithm through the MOSAIC worker interface.

    Attributes:
        run_id: Unique identifier for this training run.
        method: Algorithm name (e.g., "dqn", "ppo", "mappo", "qmix").
        env: Environment family (e.g., "classic_control", "mpe", "smac").
        env_id: Specific environment ID (e.g., "CartPole-v1", "simple_spread_v3").
        dl_toolbox: Deep learning backend ("torch", "tensorflow", "mindspore").
        running_steps: Total training steps.
        seed: Random seed for reproducibility.
        device: Computing device ("cpu", "cuda:0", "cuda:1").
        parallels: Number of parallel environments.
        test_mode: If True, loads model and evaluates instead of training.
        config_path: Custom YAML config path (None = use XuanCe defaults).
        worker_id: Optional worker identifier for logging.
        extras: Additional parameters to pass to XuanCe.
    """

    run_id: str
    method: str
    env: str
    env_id: str

    # Deep learning backend
    dl_toolbox: str = "torch"  # "torch", "tensorflow", "mindspore"

    # Training parameters
    running_steps: int = 1_000_000
    seed: int | None = None
    device: str = "cpu"
    parallels: int = 8
    test_mode: bool = False

    # Custom configuration
    config_path: str | None = None

    # Worker metadata
    worker_id: str | None = None

    # Extra parameters to pass to XuanCe
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json_file(cls, path: Path) -> "XuanCeWorkerConfig":
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            XuanCeWorkerConfig instance populated from the file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "XuanCeWorkerConfig":
        """Create config from a dictionary.

        Supports both XuanCe-style keys (method, env) and CleanRL-style
        keys (algo, env_id) for compatibility.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            XuanCeWorkerConfig instance.
        """
        return cls(
            run_id=str(data.get("run_id", "")),
            method=str(data.get("method", data.get("algo", "dqn"))),
            env=str(data.get("env", "classic_control")),
            env_id=str(data.get("env_id", "CartPole-v1")),
            dl_toolbox=str(data.get("dl_toolbox", "torch")),
            running_steps=int(
                data.get("running_steps", data.get("total_timesteps", 1_000_000))
            ),
            seed=data.get("seed"),
            device=str(data.get("device", "cpu")),
            parallels=int(data.get("parallels", 8)),
            test_mode=bool(data.get("test_mode", False)),
            config_path=data.get("config_path"),
            worker_id=data.get("worker_id"),
            extras=dict(data.get("extras", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "run_id": self.run_id,
            "method": self.method,
            "env": self.env,
            "env_id": self.env_id,
            "dl_toolbox": self.dl_toolbox,
            "running_steps": self.running_steps,
            "seed": self.seed,
            "device": self.device,
            "parallels": self.parallels,
            "test_mode": self.test_mode,
            "config_path": self.config_path,
            "worker_id": self.worker_id,
            "extras": dict(self.extras),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize configuration to JSON string.

        Args:
            indent: JSON indentation level (None for compact).

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)


__all__ = ["XuanCeWorkerConfig"]
