"""Chess Worker - LLM-based chess player using llm_chess prompting style."""

__version__ = "0.1.0"

from .runtime import ChessWorkerRuntime
from .config import ChessWorkerConfig


def get_worker_metadata() -> tuple:
    """Return worker metadata for MOSAIC registry.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="Chess Worker",
        version=__version__,
        description="LLM-based chess player using llm_chess prompting style",
        author="MOSAIC Team",
        homepage="https://github.com/Abdulhamid97Mousa/MOSAIC",
        upstream_library="llm_chess",
        upstream_version="0.1.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="chess",
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


__all__ = ["ChessWorkerRuntime", "ChessWorkerConfig", "__version__", "get_worker_metadata"]
