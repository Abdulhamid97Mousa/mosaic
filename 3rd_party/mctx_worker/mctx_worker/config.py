"""Configuration module for MCTX Worker.

This module defines configuration dataclasses for MCTS-based training
and evaluation using mctx + PGX.

Configuration hierarchy:
- MCTXWorkerConfig: Top-level worker configuration
  - MCTSConfig: MCTS algorithm settings
  - NetworkConfig: Neural network architecture
  - TrainingConfig: Training hyperparameters
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar


class MCTXAlgorithm(str, Enum):
    """Supported MCTS algorithms from mctx library."""

    ALPHAZERO = "alphazero"
    MUZERO = "muzero"
    GUMBEL_MUZERO = "gumbel_muzero"
    STOCHASTIC_MUZERO = "stochastic_muzero"


class PGXEnvironment(str, Enum):
    """Supported PGX game environments.

    Two-player games marked with (2P), single-player with (1P).
    """

    # Classic board games (2P)
    CHESS = "chess"
    GO_9X9 = "go_9x9"
    GO_19X19 = "go_19x19"
    SHOGI = "shogi"
    OTHELLO = "othello"
    BACKGAMMON = "backgammon"

    # Simple games (2P)
    CONNECT_FOUR = "connect_four"
    TIC_TAC_TOE = "tic_tac_toe"
    HEX = "hex"

    # Card/poker games (2P)
    KUHN_POKER = "kuhn_poker"
    LEDUC_HOLDEM = "leduc_holdem"

    # Single-player (1P)
    GAME_2048 = "2048"
    MINATAR_ASTERIX = "minatar-asterix"
    MINATAR_BREAKOUT = "minatar-breakout"
    MINATAR_FREEWAY = "minatar-freeway"
    MINATAR_SEAQUEST = "minatar-seaquest"
    MINATAR_SPACE_INVADERS = "minatar-space_invaders"


@dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search.

    Attributes:
        num_simulations: Number of MCTS simulations per move.
        max_num_considered_actions: For Gumbel MuZero, max actions to consider.
        dirichlet_alpha: Dirichlet noise parameter for exploration.
        dirichlet_fraction: Fraction of policy replaced by Dirichlet noise.
        pb_c_base: UCB exploration constant base.
        pb_c_init: UCB exploration constant initial.
        temperature: Temperature for action selection (1=stochastic, 0=greedy).
        temperature_drop_step: Step to drop temperature to 0 for evaluation.
        root_discount: Discount for root value estimation.
        qtransform_epsilon: Small epsilon for Q-value transformation.
    """

    num_simulations: int = 800
    max_num_considered_actions: int = 16  # For Gumbel MuZero
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25
    temperature: float = 1.0
    temperature_drop_step: int = 30
    root_discount: float = 1.0
    qtransform_epsilon: float = 1e-6


@dataclass
class NetworkConfig:
    """Configuration for policy/value neural network.

    Attributes:
        hidden_dims: Hidden layer dimensions for MLP.
        num_res_blocks: Number of residual blocks (for CNN).
        channels: Number of channels for CNN.
        embedding_dim: Dimension for board embedding.
        use_resnet: Whether to use ResNet architecture.
        activation: Activation function name.
        normalize_inputs: Whether to normalize inputs.
    """

    hidden_dims: tuple[int, ...] = (256, 256)
    num_res_blocks: int = 8
    channels: int = 128
    embedding_dim: int = 256
    use_resnet: bool = True
    activation: str = "relu"
    normalize_inputs: bool = True


@dataclass
class TrainingConfig:
    """Configuration for self-play training.

    Attributes:
        learning_rate: Initial learning rate.
        lr_schedule: Learning rate schedule type.
        weight_decay: L2 regularization weight.
        batch_size: Batch size for training.
        num_epochs: Training epochs per iteration.
        replay_buffer_size: Maximum replay buffer size.
        min_buffer_size: Minimum buffer size before training.
        checkpoint_interval: Steps between checkpoints.
        eval_interval: Steps between evaluations.
        num_eval_games: Games to play for evaluation.
        num_actors: Number of self-play actors.
        games_per_iteration: Games to collect per training iteration.
        value_loss_weight: Weight for value loss.
        policy_loss_weight: Weight for policy loss.
    """

    learning_rate: float = 2e-4
    lr_schedule: str = "cosine"
    weight_decay: float = 1e-4
    batch_size: int = 256
    num_epochs: int = 4
    replay_buffer_size: int = 100_000
    min_buffer_size: int = 1000
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    num_eval_games: int = 100
    num_actors: int = 8
    games_per_iteration: int = 128
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0


@dataclass
class MCTXWorkerConfig:
    """Top-level configuration for MCTX Worker.

    This configuration follows the MOSAIC worker protocol with
    required fields (run_id, seed) and worker-specific settings.

    Attributes:
        run_id: Unique identifier for this run.
        seed: Random seed for reproducibility.
        env_id: PGX environment identifier.
        algorithm: MCTS algorithm to use.
        max_steps: Maximum training steps (0=unlimited).
        max_episodes: Maximum games to play (0=unlimited).
        mode: Run mode (train, eval, self_play).
        checkpoint_path: Path to load/save checkpoints.
        device: JAX device (cpu, gpu, tpu).
        mcts: MCTS configuration.
        network: Neural network configuration.
        training: Training configuration.
        logging_interval: Steps between log outputs.
        verbose: Verbosity level (0=quiet, 1=normal, 2=debug).
    """

    # Required by MOSAIC worker protocol
    run_id: str = "mctx_run"
    seed: int = 42

    # Environment
    env_id: str = "chess"

    # Algorithm selection
    algorithm: MCTXAlgorithm = MCTXAlgorithm.GUMBEL_MUZERO

    # Run limits
    max_steps: int = 0  # 0 = unlimited
    max_episodes: int = 0  # 0 = unlimited

    # Mode
    mode: str = "train"  # train, eval, self_play

    # Paths
    checkpoint_path: Optional[str] = None

    # Device
    device: str = "gpu"  # cpu, gpu, tpu

    # Sub-configurations
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Logging
    logging_interval: int = 100
    verbose: int = 1

    # Class variable for config version
    CONFIG_VERSION: ClassVar[str] = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        result = asdict(self)
        # Convert enum to string
        result["algorithm"] = self.algorithm.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCTXWorkerConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            MCTXWorkerConfig instance.
        """
        # Handle nested configs
        if "mcts" in data and isinstance(data["mcts"], dict):
            data["mcts"] = MCTSConfig(**data["mcts"])
        if "network" in data and isinstance(data["network"], dict):
            # Convert list to tuple for hidden_dims
            if "hidden_dims" in data["network"]:
                data["network"]["hidden_dims"] = tuple(data["network"]["hidden_dims"])
            data["network"] = NetworkConfig(**data["network"])
        if "training" in data and isinstance(data["training"], dict):
            data["training"] = TrainingConfig(**data["training"])

        # Handle algorithm enum
        if "algorithm" in data and isinstance(data["algorithm"], str):
            data["algorithm"] = MCTXAlgorithm(data["algorithm"])

        return cls(**data)

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        if not self.run_id:
            errors.append("run_id is required")

        if self.seed < 0:
            errors.append("seed must be non-negative")

        try:
            PGXEnvironment(self.env_id)
        except ValueError:
            errors.append(f"Unknown env_id: {self.env_id}")

        if self.mode not in ("train", "eval", "self_play"):
            errors.append(f"Invalid mode: {self.mode}")

        if self.mcts.num_simulations < 1:
            errors.append("num_simulations must be at least 1")

        if self.training.batch_size < 1:
            errors.append("batch_size must be at least 1")

        return errors


def load_worker_config(path: str | Path) -> MCTXWorkerConfig:
    """Load worker configuration from JSON file.

    Args:
        path: Path to JSON configuration file.

    Returns:
        MCTXWorkerConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If configuration is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    config = MCTXWorkerConfig.from_dict(data)

    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

    return config


def save_worker_config(config: MCTXWorkerConfig, path: str | Path) -> None:
    """Save worker configuration to JSON file.

    Args:
        config: MCTXWorkerConfig instance.
        path: Path to save configuration.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
