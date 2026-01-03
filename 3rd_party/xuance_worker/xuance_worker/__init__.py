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


def get_worker_metadata() -> tuple:
    """Return worker metadata and capabilities for MOSAIC discovery.

    This function is called by the MOSAIC worker discovery system via
    entry points to register this worker with the framework.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="XuanCe Worker",
        version=__version__,
        description="Comprehensive deep RL library with 46+ algorithms for single-agent, multi-agent, and offline RL",
        author="MOSAIC Team",
        homepage="https://github.com/agi-brain/xuance",
        upstream_library="xuance",
        upstream_version="2.0.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="xuance",
        supported_paradigms=(
            "sequential",           # Single-agent DRL
            "parameter_sharing",    # Multi-agent with shared parameters
            "independent",          # Independent multi-agent learners
        ),
        env_families=(
            "gymnasium",            # Gymnasium single-agent
            "atari",               # Atari 2600 games
            "mujoco",              # MuJoCo physics
            "classic_control",     # Classic RL tasks
            "pettingzoo",          # PettingZoo multi-agent
            "mpe",                 # Multi-agent particle environments
            "smac",                # StarCraft Multi-Agent Challenge
            "football",            # Google Research Football
            "minigrid",            # MiniGrid environments
        ),
        action_spaces=("discrete", "continuous", "multi_discrete"),
        observation_spaces=("vector", "image", "dict"),
        max_agents=100,  # Supports large-scale multi-agent scenarios
        supports_self_play=False,  # Not explicitly designed for self-play
        supports_population=False,
        supports_checkpointing=True,  # XuanCe saves checkpoints
        supports_pause_resume=False,
        requires_gpu=False,  # Can run on CPU, but GPU recommended
        gpu_memory_mb=2048,  # Approximate for typical scenarios
        cpu_cores=1,
        estimated_memory_mb=2048,
    )

    return metadata, capabilities


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
    # Worker discovery
    "get_worker_metadata",
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
