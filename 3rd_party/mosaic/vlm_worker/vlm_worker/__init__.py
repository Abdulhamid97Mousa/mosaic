"""MOSAIC VLM Worker - Vision-Language Model agents.

This package provides VLM-based agents with image observation support for RL environments.
"""

__version__ = "0.1.0"

from vlm_worker.config import VLMWorkerConfig
from vlm_worker.runtime import VLMWorkerRuntime, InteractiveVLMRuntime


def get_worker_metadata() -> tuple:
    """Return worker metadata for MOSAIC registry.

    This function is called by the MOSAIC worker discovery system via
    entry points to register this worker with the framework.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="VLM Worker",
        version=__version__,
        description="Vision-Language Model agents with image observations for RL environments",
        author="MOSAIC Team",
        homepage="https://github.com/MOSAIC-RL/GUI_BDI_RL",
        upstream_library=None,
        upstream_version=None,
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="vlm",
        supported_paradigms=("single_agent", "multi_agent"),
        env_families=("gymnasium", "pettingzoo", "minigrid", "babyai"),
        action_spaces=("discrete",),
        observation_spaces=("structured", "text", "image"),
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
    "VLMWorkerConfig",
    "VLMWorkerRuntime",
    "InteractiveVLMRuntime",
    "get_worker_metadata",
]
