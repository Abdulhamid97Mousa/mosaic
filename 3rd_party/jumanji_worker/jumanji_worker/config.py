"""Configuration dataclass for Jumanji worker.

This module provides the JumanjiWorkerConfig dataclass that implements
the MOSAIC WorkerConfig protocol for standardized worker configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping, Optional


# Phase 1: Logic environments
LOGIC_ENVIRONMENTS = frozenset({
    "Game2048-v1",
    "GraphColoring-v1",
    "Minesweeper-v0",
    "RubiksCube-v0",
    "RubiksCube-partly-scrambled-v0",
    "SlidingTilePuzzle-v0",
    "Sudoku-v0",
    "Sudoku-very-easy-v0",
})

# Phase 2: Packing environments
PACKING_ENVIRONMENTS = frozenset({
    "BinPack-v2",
    "FlatPack-v0",
    "JobShop-v0",
    "Knapsack-v1",
    "Tetris-v0",
})

# Phase 3: Routing environments
ROUTING_ENVIRONMENTS = frozenset({
    "Cleaner-v0",
    "Connector-v2",
    "CVRP-v1",
    "Maze-v0",
    "MMST-v0",
    "MultiCVRP-v0",
    "PacMan-v1",
    "RobotWarehouse-v0",
    "Snake-v1",
    "Sokoban-v0",
    "TSP-v1",
})

# All supported environments (Phase 1 + Phase 2 + Phase 3)
SUPPORTED_ENVIRONMENTS = LOGIC_ENVIRONMENTS | PACKING_ENVIRONMENTS | ROUTING_ENVIRONMENTS

# All Jumanji environments (complete list)
ALL_ENVIRONMENTS = frozenset({
    # Logic
    "Game2048-v1",
    "GraphColoring-v1",
    "Minesweeper-v0",
    "RubiksCube-v0",
    "RubiksCube-partly-scrambled-v0",
    "SlidingTilePuzzle-v0",
    "Sudoku-v0",
    "Sudoku-very-easy-v0",
    # Packing
    "BinPack-v2",
    "FlatPack-v0",
    "JobShop-v0",
    "Knapsack-v1",
    "Tetris-v0",
    # Routing
    "Cleaner-v0",
    "Connector-v2",
    "CVRP-v1",
    "Maze-v0",
    "MMST-v0",
    "MultiCVRP-v0",
    "PacMan-v1",
    "RobotWarehouse-v0",
    "Snake-v1",
    "Sokoban-v0",
    "TSP-v1",
})


@dataclass
class JumanjiWorkerConfig:
    """Configuration for a Jumanji training run.

    This dataclass implements the MOSAIC WorkerConfig protocol.

    Attributes:
        run_id: Unique identifier for this training run (REQUIRED by protocol)
        env_id: Jumanji environment ID (e.g., "Game2048-v1")
        agent: Agent type ("a2c" or "random")
        seed: Random seed for reproducibility (REQUIRED by protocol, can be None)

        # Training hyperparameters
        num_epochs: Number of training epochs
        n_steps: Rollout length per update
        total_batch_size: Total batch size across devices
        num_learner_steps_per_epoch: Learning steps per epoch

        # A2C hyperparameters
        learning_rate: Optimizer learning rate
        discount_factor: Reward discounting (gamma)
        bootstrapping_factor: GAE lambda
        normalize_advantage: Whether to normalize advantages
        l_pg: Policy gradient loss coefficient
        l_td: TD loss coefficient
        l_en: Entropy loss coefficient

        # Device configuration
        device: JAX device ("cpu", "gpu", "tpu")

        # Logging
        logger_type: Logger type ("tensorboard", "terminal")
        save_checkpoint: Whether to save model checkpoints

        # Worker metadata
        worker_id: Optional worker identifier
        extras: Additional parameters
    """

    # Protocol-required fields
    run_id: str
    seed: int | None = None

    # Environment
    env_id: str = "Game2048-v1"

    # Agent
    agent: str = "a2c"

    # Training hyperparameters
    num_epochs: int = 100
    n_steps: int = 20
    total_batch_size: int = 128
    num_learner_steps_per_epoch: int = 64

    # A2C hyperparameters
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    bootstrapping_factor: float = 0.95
    normalize_advantage: bool = True
    l_pg: float = 1.0
    l_td: float = 0.5
    l_en: float = 0.01

    # Device configuration
    device: str = "cpu"

    # Logging
    logger_type: str = "tensorboard"
    save_checkpoint: bool = True

    # Worker metadata
    worker_id: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration and assert protocol compliance."""
        # Validate required fields
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.env_id:
            raise ValueError("env_id is required")

        # Validate environment (Logic + Packing + Routing environments)
        if self.env_id not in SUPPORTED_ENVIRONMENTS:
            raise ValueError(
                f"env_id must be one of {sorted(SUPPORTED_ENVIRONMENTS)}, "
                f"got '{self.env_id}'."
            )

        # Validate agent type
        if self.agent not in ("a2c", "random"):
            raise ValueError(f"agent must be 'a2c' or 'random', got '{self.agent}'")

        # Validate device
        if self.device not in ("cpu", "gpu", "tpu"):
            raise ValueError(f"device must be 'cpu', 'gpu', or 'tpu', got '{self.device}'")

        # Validate hyperparameters
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not 0 <= self.discount_factor <= 1:
            raise ValueError("discount_factor must be in [0, 1]")

        # Protocol compliance check
        try:
            from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
            assert isinstance(self, WorkerConfigProtocol), (
                "JumanjiWorkerConfig must implement WorkerConfig protocol"
            )
        except ImportError:
            pass  # gym_gui not available, skip protocol check

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JumanjiWorkerConfig":
        """Deserialize configuration from dictionary (REQUIRED by protocol).

        Handles both flat and nested formats for backwards compatibility.

        Args:
            data: Dictionary containing configuration values

        Returns:
            JumanjiWorkerConfig instance
        """
        return cls(
            run_id=str(data["run_id"]),
            env_id=str(data.get("env_id", "Game2048-v1")),
            agent=str(data.get("agent", "a2c")),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            num_epochs=int(data.get("num_epochs", 100)),
            n_steps=int(data.get("n_steps", 20)),
            total_batch_size=int(data.get("total_batch_size", 128)),
            num_learner_steps_per_epoch=int(data.get("num_learner_steps_per_epoch", 64)),
            learning_rate=float(data.get("learning_rate", 3e-4)),
            discount_factor=float(data.get("discount_factor", 0.99)),
            bootstrapping_factor=float(data.get("bootstrapping_factor", 0.95)),
            normalize_advantage=bool(data.get("normalize_advantage", True)),
            l_pg=float(data.get("l_pg", 1.0)),
            l_td=float(data.get("l_td", 0.5)),
            l_en=float(data.get("l_en", 0.01)),
            device=str(data.get("device", "cpu")),
            logger_type=str(data.get("logger_type", "tensorboard")),
            save_checkpoint=bool(data.get("save_checkpoint", True)),
            worker_id=data.get("worker_id"),
            extras=dict(data.get("extras", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary (REQUIRED by protocol).

        Returns:
            Dictionary representation of configuration
        """
        return {
            "run_id": self.run_id,
            "env_id": self.env_id,
            "agent": self.agent,
            "seed": self.seed,
            "num_epochs": self.num_epochs,
            "n_steps": self.n_steps,
            "total_batch_size": self.total_batch_size,
            "num_learner_steps_per_epoch": self.num_learner_steps_per_epoch,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "bootstrapping_factor": self.bootstrapping_factor,
            "normalize_advantage": self.normalize_advantage,
            "l_pg": self.l_pg,
            "l_td": self.l_td,
            "l_en": self.l_en,
            "device": self.device,
            "logger_type": self.logger_type,
            "save_checkpoint": self.save_checkpoint,
            "worker_id": self.worker_id,
            "extras": dict(self.extras),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def with_overrides(
        self,
        *,
        env_id: Optional[str] = None,
        agent: Optional[str] = None,
        seed: Optional[int] = None,
        num_epochs: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> "JumanjiWorkerConfig":
        """Return a new config with CLI overrides applied.

        Args:
            env_id: Override environment ID
            agent: Override agent type
            seed: Override random seed
            num_epochs: Override number of epochs
            device: Override device
            **kwargs: Additional overrides

        Returns:
            New JumanjiWorkerConfig with overrides applied
        """
        config_dict = self.to_dict()

        if env_id is not None:
            config_dict["env_id"] = env_id
        if agent is not None:
            config_dict["agent"] = agent
        if seed is not None:
            config_dict["seed"] = seed
        if num_epochs is not None:
            config_dict["num_epochs"] = num_epochs
        if device is not None:
            config_dict["device"] = device

        # Apply any additional overrides
        for key, value in kwargs.items():
            if value is not None and key in config_dict:
                config_dict[key] = value

        return JumanjiWorkerConfig.from_dict(config_dict)


def load_worker_config(config_path: str) -> JumanjiWorkerConfig:
    """Load worker configuration from JSON file.

    Handles both direct config format and nested metadata.worker.config structure.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Parsed JumanjiWorkerConfig object

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
    else:
        # Direct config format
        config_data = raw_data

    return JumanjiWorkerConfig.from_dict(config_data)


__all__ = [
    "JumanjiWorkerConfig",
    "load_worker_config",
    "LOGIC_ENVIRONMENTS",
    "PACKING_ENVIRONMENTS",
    "ROUTING_ENVIRONMENTS",
    "SUPPORTED_ENVIRONMENTS",
    "ALL_ENVIRONMENTS",
]
