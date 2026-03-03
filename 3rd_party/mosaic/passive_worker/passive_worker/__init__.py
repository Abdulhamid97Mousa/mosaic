"""Passive (noop) baseline worker for MOSAIC multi-agent environments.

Provides a lightweight subprocess that always selects action 0 (do nothing),
serving as a passive baseline for comparison against RL, LLM, and human
decision-makers.
"""

__version__ = "0.1.0"

from passive_worker.config import PassiveWorkerConfig
from passive_worker.runtime import PassiveWorkerRuntime


def get_worker_metadata() -> tuple:
    """Return worker metadata for MOSAIC registry.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="Passive Worker",
        version=__version__,
        description="Deterministic do-nothing baseline (NOOP/STILL action)",
        author="MOSAIC Team",
        homepage="https://github.com/Abdulhamid97Mousa/MOSAIC",
        upstream_library=None,
        upstream_version=None,
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="passive",
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
    "PassiveWorkerConfig",
    "PassiveWorkerRuntime",
    "__version__",
    "get_worker_metadata",
]
