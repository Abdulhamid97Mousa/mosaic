"""Jumanji Worker - JAX-based RL environments for MOSAIC.

This worker provides integration between InstaDeep's Jumanji library and
the MOSAIC BDI-RL framework. It offers two modes of operation:

1. **Standalone Training**: Native JAX-based A2C training using Jumanji's
   built-in training infrastructure with GPU/TPU acceleration.

2. **Cross-Worker Compatibility**: Exposes Jumanji environments via the
   standard Gymnasium API, allowing CleanRL, Ray, and XuanCe workers to
   use Jumanji environments.

Supported Environments (Phase 1 - Logic):
- Game2048: Reach high-valued tile by merging
- GraphColoring: Color vertices without adjacent same-color
- Minesweeper: Clear board without detonating mines
- RubiksCube: Match all stickers on each face
- SlidingTilePuzzle: Arrange tiles in order
- Sudoku: Fill grid with 1-N in rows/cols/subgrids

Example Usage:
    # Standalone training
    from jumanji_worker import JumanjiWorkerConfig, JumanjiWorkerRuntime

    config = JumanjiWorkerConfig(
        run_id="game2048_run1",
        env_id="Game2048-v1",
        agent="a2c",
        num_epochs=100,
        device="gpu",
    )
    runtime = JumanjiWorkerRuntime(config)
    runtime.run()

    # Cross-worker usage (by CleanRL, Ray, etc.)
    import gymnasium
    env = gymnasium.make("jumanji/Game2048-v1")
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "MOSAIC Team"

from jumanji_worker.config import JumanjiWorkerConfig, load_worker_config


def get_worker_metadata() -> tuple:
    """Return worker metadata and capabilities for MOSAIC discovery.

    This function is called by the MOSAIC worker discovery system via
    entry points to register this worker with the framework.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="Jumanji Worker",
        version=__version__,
        description="JAX-based logic puzzle environments with native A2C training",
        author="MOSAIC Team",
        homepage="https://github.com/instadeepai/jumanji",
        upstream_library="jumanji",
        upstream_version="1.0.0",
        license="Apache-2.0",
    )

    capabilities = WorkerCapabilities(
        worker_type="jumanji",
        supported_paradigms=("sequential",),
        env_families=("jumanji", "gymnasium"),  # Native + Gymnasium bridge
        action_spaces=("discrete",),
        observation_spaces=("vector", "structured"),
        max_agents=1,
        supports_self_play=False,
        supports_population=False,
        supports_checkpointing=True,
        supports_pause_resume=False,
        requires_gpu=False,  # JAX can use CPU, GPU, or TPU
        gpu_memory_mb=None,
        cpu_cores=1,
        estimated_memory_mb=1024,
    )

    return metadata, capabilities


# Lazy import for runtime to avoid heavy JAX imports on module load
def _get_runtime():
    from jumanji_worker.runtime import JumanjiWorkerRuntime
    return JumanjiWorkerRuntime


# Lazy import for CLI to avoid circular imports
def main(*args, **kwargs):
    """CLI entry point (lazy loaded)."""
    from jumanji_worker.cli import main as _main
    return _main(*args, **kwargs)


# For type hints without importing
JumanjiWorkerRuntime = property(lambda self: _get_runtime())


__all__ = [
    "__version__",
    "JumanjiWorkerConfig",
    "load_worker_config",
    "get_worker_metadata",
    "main",
]
