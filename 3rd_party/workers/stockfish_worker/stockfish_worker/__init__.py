"""Stockfish Worker - Chess engine integration for MOSAIC operators.

This worker provides Stockfish chess engine integration for the MOSAIC
multi-operator system. It can be used for:
- Human vs Stockfish games
- LLM vs Stockfish games
- Stockfish vs Stockfish (self-play at different levels)

The worker operates in interactive mode, receiving chess positions via stdin
and emitting moves via stdout using the JSONL protocol.
"""

__version__ = "1.0.0"

from .config import StockfishWorkerConfig, load_worker_config, DIFFICULTY_PRESETS
from .runtime import StockfishWorkerRuntime, StockfishState


def get_worker_metadata() -> tuple:
    """Return worker metadata and capabilities for MOSAIC discovery.

    This function is called by the MOSAIC worker discovery system via
    entry points to register this worker with the framework.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="Stockfish Worker",
        version=__version__,
        description="Stockfish chess engine for multi-agent chess games",
        author="MOSAIC Team",
        homepage="https://stockfishchess.org/",
        upstream_library="stockfish",
        upstream_version="16",
        license="GPL-3.0",
    )

    capabilities = WorkerCapabilities(
        worker_type="stockfish",
        supported_paradigms=("self_play", "human_vs_ai", "ai_vs_ai"),
        env_families=("pettingzoo",),
        action_spaces=("discrete",),
        observation_spaces=("structured",),
        max_agents=2,
        supports_self_play=True,
        supports_population=False,
        supports_checkpointing=False,
        supports_pause_resume=True,
        requires_gpu=False,
        gpu_memory_mb=None,
        cpu_cores=1,
        estimated_memory_mb=64,
    )

    return metadata, capabilities


__all__ = [
    "__version__",
    "StockfishWorkerConfig",
    "load_worker_config",
    "DIFFICULTY_PRESETS",
    "StockfishWorkerRuntime",
    "StockfishState",
    "get_worker_metadata",
]
