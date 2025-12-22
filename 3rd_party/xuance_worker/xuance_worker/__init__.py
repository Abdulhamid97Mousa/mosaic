"""XuanCe Worker - MOSAIC Integration for XuanCe RL Library.

This module provides CLI and programmatic interfaces for running XuanCe
algorithms within the MOSAIC framework.

XuanCe is a comprehensive deep reinforcement learning library with 46+
algorithms supporting single-agent, multi-agent, and offline RL.

Key Features:
- Single-agent RL: DQN, PPO, SAC, TD3, DDPG, A2C, DreamerV3
- Multi-agent RL: MAPPO, MADDPG, QMIX, VDN, COMA
- Multiple backends: PyTorch (primary), TensorFlow, MindSpore
- Integration with MOSAIC's training and telemetry infrastructure

Supported Runners:
- RunnerDRL: Single-agent Gymnasium environments
- RunnerMARL: Multi-agent cooperative environments (SMAC, etc.)
- RunnerPettingzoo: PettingZoo environments
- RunnerStarCraft2: StarCraft Multi-Agent Challenge
- RunnerFootball: Google Research Football

Example:
    >>> from xuance_worker import XuanCeWorkerConfig, XuanCeWorkerRuntime
    >>>
    >>> config = XuanCeWorkerConfig(
    ...     run_id="test_run",
    ...     method="ppo",
    ...     env="classic_control",
    ...     env_id="CartPole-v1",
    ...     running_steps=100000,
    ... )
    >>> runtime = XuanCeWorkerRuntime(config)
    >>> summary = runtime.run()
    >>> print(summary.status)
    'completed'

CLI Usage:
    # Direct parameter mode
    xuance-worker --method ppo --env classic_control --env-id CartPole-v1

    # Config file mode
    xuance-worker --config /path/to/config.json

    # Dry-run mode
    xuance-worker --method dqn --env atari --env-id Pong-v5 --dry-run
"""

from __future__ import annotations

from .config import XuanCeWorkerConfig
from .runtime import XuanCeRuntimeSummary, XuanCeWorkerRuntime
from .algorithm_registry import (
    Backend,
    Paradigm,
    AlgorithmInfo,
    get_algorithms,
    get_algorithms_for_backend,
    get_algorithms_for_paradigm,
    get_algorithm_info,
    get_algorithm_choices,
    get_algorithms_by_category,
    is_algorithm_available,
    get_backend_summary,
)

__version__ = "0.1.0"


def main(args: list[str] | None = None) -> int:
    """Main entry point for xuance-worker CLI (lazy import to avoid circular imports)."""
    from .cli import main as _main
    return _main(args)


__all__ = [
    "__version__",
    # Configuration
    "XuanCeWorkerConfig",
    # Runtime
    "XuanCeWorkerRuntime",
    "XuanCeRuntimeSummary",
    # CLI
    "main",
    # Algorithm Registry
    "Backend",
    "Paradigm",
    "AlgorithmInfo",
    "get_algorithms",
    "get_algorithms_for_backend",
    "get_algorithms_for_paradigm",
    "get_algorithm_info",
    "get_algorithm_choices",
    "get_algorithms_by_category",
    "is_algorithm_available",
    "get_backend_summary",
]
