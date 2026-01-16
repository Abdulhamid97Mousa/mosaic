"""Dragon Worker - Komodo Dragon chess engine integration for MOSAIC operators.

This worker provides Komodo Dragon chess engine integration for the MOSAIC
multi-operator system. It can be used for:
- Human vs Dragon games
- LLM vs Dragon games (Elo calibration)
- Dragon vs Dragon (self-play at different levels)
- Dragon vs Stockfish comparisons

Dragon provides calibrated Elo ratings using the formula:
    Elo = 125 * (skill_level + 1)

This makes it ideal for benchmarking and research applications.

The worker operates in interactive mode, receiving chess positions via stdin
and emitting moves via stdout using the JSONL protocol.

Download Dragon from: https://komodochess.com/installation.htm
"""

__version__ = "1.0.0"

from .config import DragonWorkerConfig, load_worker_config, DIFFICULTY_PRESETS
from .runtime import DragonWorkerRuntime, DragonState


def get_worker_metadata() -> tuple:
    """Return worker metadata and capabilities for MOSAIC discovery.

    This function is called by the MOSAIC worker discovery system via
    entry points to register this worker with the framework.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="Dragon Worker",
        version=__version__,
        description="Komodo Dragon chess engine for multi-agent chess games with Elo calibration",
        author="MOSAIC Team",
        homepage="https://komodochess.com/",
        upstream_library="komodo-dragon",
        upstream_version="3.3",
        license="Proprietary (free for personal/research use)",
    )

    capabilities = WorkerCapabilities(
        worker_type="dragon",
        supported_paradigms=("self_play", "human_vs_ai", "ai_vs_ai", "benchmark"),
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
        estimated_memory_mb=128,
    )

    return metadata, capabilities


__all__ = [
    "__version__",
    "DragonWorkerConfig",
    "load_worker_config",
    "DIFFICULTY_PRESETS",
    "DragonWorkerRuntime",
    "DragonState",
    "get_worker_metadata",
]
