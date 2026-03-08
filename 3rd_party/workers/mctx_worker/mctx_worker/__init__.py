"""MCTX Worker - Monte Carlo Tree Search with JAX for turn-based games.

This module provides MCTS-based agents using DeepMind's mctx library and
PGX (Pure GPU game eXecutor) for turn-based environments like Chess, Go,
Shogi, Connect Four, and other board games.

Key Features:
- GPU-accelerated MCTS using JAX
- AlphaZero and MuZero policy implementations
- Gumbel MuZero for improved policy optimization
- PGX environments for vectorized game simulation
- Support for self-play training and evaluation
- Batched inference for high throughput

Supported Algorithms:
- AlphaZero: Model-free MCTS with learned value/policy networks
- MuZero: Model-based MCTS with learned dynamics model
- Gumbel MuZero: Improved policy optimization with Gumbel sampling

Supported Environments (via PGX):
- Chess, Shogi, Go (9x9, 19x19)
- Connect Four, Tic-Tac-Toe, Othello
- Backgammon, Hex, Kuhn Poker
- 2048 (single-player)

Example:
    # Via CLI (recommended for MOSAIC integration):
    python -m mctx_worker.cli --config config.json

    # Via Python API:
    from mctx_worker import MCTXWorkerConfig, MCTXWorkerRuntime

    config = MCTXWorkerConfig(
        run_id="chess_alphazero_run1",
        env_id="chess",
        algorithm="gumbel_muzero",
        num_simulations=800,
    )
    runtime = MCTXWorkerRuntime(config)
    runtime.run()

References:
- mctx: https://github.com/google-deepmind/mctx
- PGX: https://github.com/sotetsuk/pgx
- AlphaZero: https://www.nature.com/articles/nature24270
- MuZero: https://www.nature.com/articles/s41586-020-03051-4
- Gumbel MuZero: https://openreview.net/forum?id=bERaNdoegnO
"""

from __future__ import annotations

__version__ = "0.1.0"

# Lazy imports to avoid loading JAX unless needed
def __getattr__(name):
    """Lazy import pattern for heavy dependencies."""
    if name in (
        "MCTXWorkerConfig",
        "load_worker_config",
        "MCTXAlgorithm",
        "PGXEnvironment",
        "NetworkConfig",
        "TrainingConfig",
        "MCTSConfig",
    ):
        from .config import (
            MCTXWorkerConfig,
            load_worker_config,
            MCTXAlgorithm,
            PGXEnvironment,
            NetworkConfig,
            TrainingConfig,
            MCTSConfig,
        )
        return locals()[name]

    if name in ("MCTXWorkerRuntime",):
        from .runtime import MCTXWorkerRuntime
        return MCTXWorkerRuntime

    if name in ("MCTXPolicyActor", "create_mctx_actor"):
        from .policy_actor import MCTXPolicyActor, create_mctx_actor
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_worker_metadata() -> tuple:
    """Return worker metadata and capabilities for MOSAIC discovery.

    This function is called by the MOSAIC worker discovery system via
    entry points to register this worker with the framework.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="MCTX Worker",
        version=__version__,
        description="Monte Carlo Tree Search using mctx + PGX for turn-based games (Chess, Go, etc.)",
        author="MOSAIC Team",
        homepage="https://github.com/google-deepmind/mctx",
        upstream_library="mctx",
        upstream_version="0.0.5",
        license="Apache-2.0",
    )

    capabilities = WorkerCapabilities(
        worker_type="mctx",
        supported_paradigms=("self_play", "evaluation", "training"),
        env_families=("pgx", "turn_based", "board_games"),
        action_spaces=("discrete",),
        observation_spaces=("vector", "image", "board"),
        max_agents=2,  # Two-player games
        supports_self_play=True,
        supports_population=False,
        supports_checkpointing=True,
        supports_pause_resume=True,
        requires_gpu=True,  # JAX GPU acceleration
        gpu_memory_mb=4096,  # Typical for MCTS with neural nets
        cpu_cores=1,  # GPU does the heavy lifting
        estimated_memory_mb=4096,
    )

    return metadata, capabilities


def main(*args, **kwargs):
    """CLI entry point (lazy loaded)."""
    from .cli import main as _main
    return _main(*args, **kwargs)


__all__ = [
    "__version__",
    # Config classes
    "MCTXWorkerConfig",
    "load_worker_config",
    "MCTXAlgorithm",
    "PGXEnvironment",
    "NetworkConfig",
    "TrainingConfig",
    "MCTSConfig",
    # Runtime
    "MCTXWorkerRuntime",
    # Policy Actor
    "MCTXPolicyActor",
    "create_mctx_actor",
    # CLI
    "main",
    # Discovery
    "get_worker_metadata",
]
