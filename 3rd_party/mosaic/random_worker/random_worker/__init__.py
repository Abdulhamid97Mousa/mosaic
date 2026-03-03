"""Random action-selector worker for MOSAIC multi-agent environments.

Provides a lightweight subprocess that selects uniformly random actions,
serving as a stochastic baseline for comparison against RL, LLM, and human
decision-makers.
"""

__version__ = "0.1.0"

from random_worker.config import RandomWorkerConfig
from random_worker.runtime import RandomWorkerRuntime


def get_worker_metadata() -> tuple:
    """Return worker metadata for MOSAIC registry.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="Random Worker",
        version=__version__,
        description="Uniform random action selection baseline",
        author="MOSAIC Team",
        homepage="https://github.com/Abdulhamid97Mousa/MOSAIC",
        upstream_library=None,
        upstream_version=None,
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="random",
        supported_paradigms=("baseline",),
        env_families=("gymnasium", "pettingzoo"),
        action_spaces=("discrete",),
        observation_spaces=("structured",),
        max_agents=8,
        supports_self_play=False,
        supports_population=False,
        supports_checkpointing=False,
        supports_pause_resume=False,
        requires_gpu=False,
        gpu_memory_mb=None,
        cpu_cores=1,
        estimated_memory_mb=128,
    )

    return metadata, capabilities


__all__ = [
    "RandomWorkerConfig",
    "RandomWorkerRuntime",
    "__version__",
    "get_worker_metadata",
]
