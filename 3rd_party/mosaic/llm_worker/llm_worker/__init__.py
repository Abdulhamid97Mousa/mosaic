"""MOSAIC LLM Worker - Based on BALROG.

This package provides LLM-based agents for RL environments.
"""

__version__ = "0.1.0"

from llm_worker.config import LLMWorkerConfig
from llm_worker.runtime import LLMWorkerRuntime, InteractiveLLMRuntime


def get_worker_metadata() -> tuple:
    """Return worker metadata for MOSAIC registry.

    This function is called by the MOSAIC worker discovery system via
    entry points to register this worker with the framework.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="LLM Worker",
        version=__version__,
        description="LLM-based agents for RL environments (based on BALROG)",
        author="MOSAIC Team",
        homepage="https://github.com/MOSAIC-RL/GUI_BDI_RL",
        upstream_library="balrog",
        upstream_version=None,
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="llm",
        supported_paradigms=("single_agent", "multi_agent"),
        env_families=("gymnasium", "pettingzoo", "minigrid", "babyai"),
        action_spaces=("discrete",),
        observation_spaces=("structured", "text"),
        max_agents=8,
        supports_self_play=False,
        supports_population=False,
        supports_checkpointing=False,
        supports_pause_resume=True,
        requires_gpu=False,  # API-based by default
        gpu_memory_mb=None,
        cpu_cores=1,
        estimated_memory_mb=512,
    )

    return metadata, capabilities


__all__ = [
    "__version__",
    "LLMWorkerConfig",
    "LLMWorkerRuntime",
    "InteractiveLLMRuntime",
    "get_worker_metadata",
]
