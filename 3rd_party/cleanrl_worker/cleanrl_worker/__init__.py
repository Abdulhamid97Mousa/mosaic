"""CleanRL worker integration scaffolding."""

from __future__ import annotations

from .config import WorkerConfig, load_worker_config
from .runtime import CleanRLWorkerRuntime, RuntimeSummary
from . import wrappers

from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities


def get_worker_metadata() -> tuple[WorkerMetadata, WorkerCapabilities]:
    """Get CleanRL worker metadata and capabilities for MOSAIC discovery.

    This function is called by the worker discovery system to populate
    the worker registry with CleanRL's metadata and capabilities.

    Returns:
        tuple: (WorkerMetadata, WorkerCapabilities)
    """
    metadata = WorkerMetadata(
        name="CleanRL Worker",
        version=__version__,
        description="Single-file RL implementations from CleanRL library",
        author="MOSAIC Team",
        homepage="https://github.com/vwxyzjn/cleanrl",
        upstream_library="cleanrl",
        upstream_version="2.0.0",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="cleanrl",
        supported_paradigms=("sequential",),
        env_families=("gymnasium", "atari", "procgen", "mujoco", "dm_control", "minigrid", "babyai"),
        action_spaces=("discrete", "continuous"),
        observation_spaces=("vector", "image"),
        max_agents=1,
        supports_self_play=False,
        supports_population=False,
        supports_checkpointing=True,
        supports_pause_resume=False,
        requires_gpu=False,
        gpu_memory_mb=None,
        cpu_cores=1,
        estimated_memory_mb=512,
    )

    return metadata, capabilities


__all__ = [
    "CleanRLWorkerRuntime",
    "RuntimeSummary",
    "WorkerConfig",
    "load_worker_config",
    "get_worker_metadata",
    "wrappers",
]

__version__ = "0.1.0"
