"""LLM Game Worker - LLM-based player for PettingZoo classic board games.

Supports multiple two-player games:
- Tic-Tac-Toe (tictactoe_v3)
- Connect Four (connect_four_v3)
- Go (go_v5)

Uses multi-turn conversation prompting similar to llm_chess.
"""

__version__ = "0.1.0"

from .runtime import LLMGameWorkerRuntime
from .config import LLMGameWorkerConfig


def get_worker_metadata() -> tuple:
    """Return worker metadata for MOSAIC registry.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="LLM Game Worker",
        version=__version__,
        description="LLM-based player for PettingZoo classic board games (Tic-Tac-Toe, Connect Four, Go)",
        author="MOSAIC Team",
        homepage="https://github.com/Abdulhamid97Mousa/MOSAIC",
        upstream_library="llm_game_benchmark",
        upstream_version="0.1.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="llm_game",
        supported_paradigms=("self_play", "human_vs_ai"),
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
        estimated_memory_mb=512,
    )

    return metadata, capabilities


__all__ = ["LLMGameWorkerRuntime", "LLMGameWorkerConfig", "__version__", "get_worker_metadata"]
