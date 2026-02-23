"""MARLlib worker integration for MOSAIC.

Wraps the MARLlib library (https://github.com/Replicable-MARL/MARLlib)
providing 18 multi-agent RL algorithms across three paradigms:
Independent Learning (IL), Centralized Critic (CC), and
Value Decomposition (VD).

NOTE: MARLlib depends on ray==1.8.0 which is older than the Ray version
used by other MOSAIC workers.  Install MARLlib in the same environment
only if there are no Ray version conflicts.
"""

from __future__ import annotations

from .config import MARLlibWorkerConfig, load_worker_config
from .runtime import MARLlibWorkerRuntime

__version__ = "0.1.0"


def get_worker_metadata() -> tuple:
    """Return worker metadata and capabilities for MOSAIC discovery.

    Called by the worker discovery system via the
    ``mosaic.workers`` entry-point group.
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="MARLlib Worker",
        version=__version__,
        description=(
            "Multi-Agent RL library with 17 algorithms across "
            "IL / CC / VD paradigms (Ray/RLlib 1.8 backend)"
        ),
        author="MOSAIC Team",
        homepage="https://github.com/Replicable-MARL/MARLlib",
        upstream_library="marllib",
        upstream_version="1.0.3",
        license="MIT",
    )

    capabilities = WorkerCapabilities(
        worker_type="marllib",
        supported_paradigms=(
            "independent_learning",
            "centralized_critic",
            "value_decomposition",
        ),
        env_families=(
            "mpe",
            "smac",
            "mamujoco",
            "football",
            "magent",
            "rware",
            "lbf",
            "pommerman",
            "hanabi",
            "metadrive",
            "mate",
            "gobigger",
            "overcooked",
            "voltage",
            "aircombat",
            "hns",
            "sisl",
            "gymnasium_mpe",
            "gymnasium_mamujoco",
        ),
        action_spaces=("discrete", "continuous", "multi_discrete"),
        observation_spaces=("vector", "dict"),
        max_agents=100,
        supports_self_play=False,
        supports_population=False,
        supports_checkpointing=True,
        supports_pause_resume=False,
        requires_gpu=False,
        gpu_memory_mb=None,
        cpu_cores=2,
        estimated_memory_mb=2048,
    )

    return metadata, capabilities


__all__ = [
    "__version__",
    "MARLlibWorkerConfig",
    "MARLlibWorkerRuntime",
    "load_worker_config",
    "get_worker_metadata",
]
