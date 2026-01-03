"""Ray/RLlib worker integration for MOSAIC BDI-RL framework.

This module provides the integration layer between MOSAIC's GUI and
Ray/RLlib's distributed reinforcement learning capabilities.

Key Features:
- Multi-agent training with RLlib algorithms (PPO, DQN, A2C, IMPALA, APPO)
- Distributed training across multiple workers
- Support for both discrete and continuous action spaces
- PettingZoo environment integration (SISL, Classic, Butterfly, MPE)
- Multiple policy configurations (parameter sharing, independent, self-play)

Policy Configurations:
- Parameter Sharing: All agents share one policy (cooperative)
- Independent: Each agent has its own policy
- Self-Play: Agent plays against copies of itself (competitive)
- Shared Value Function: CTDE (Centralized Training, Decentralized Execution)

Supported Environments:
- SISL: Waterworld, Multiwalker, Pursuit (cooperative continuous control)
- Classic: Chess, Go, Connect Four, Tic-Tac-Toe (board games)
- Butterfly: Knights Archers Zombies, Cooperative Pong, Pistonball
- MPE: Simple Spread, Simple Adversary, Simple Tag

Example:
    # Via CLI (recommended for MOSAIC integration):
    python -m ray_worker.cli --config config.json

    # Via Python API:
    from ray_worker import RayWorkerConfig, RayWorkerRuntime

    config = RayWorkerConfig(
        run_id="waterworld_ppo_run1",
        environment=EnvironmentConfig(
            family="sisl",
            env_id="waterworld_v4",
        ),
        policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
    )
    runtime = RayWorkerRuntime(config)
    runtime.run()
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import (
    PolicyConfiguration,
    TrainingParadigm,  # Backwards compatibility alias (deprecated)
    PettingZooAPIType,
    ResourceConfig,
    TrainingConfig,
    CheckpointConfig,
    EnvironmentConfig,
    RayWorkerConfig,
    load_worker_config,
)
from .runtime import (
    EnvironmentFactory,
    RayWorkerRuntime,
)
from .algo_params import (
    SCHEMA_VERSION,
    get_available_algorithms,
    get_algorithm_info,
    get_algorithm_fields,
    get_common_fields,
    get_all_fields,
    get_field_names,
    get_default_params,
    validate_params,
    filter_params_for_algorithm,
    merge_with_defaults,
)
from .policy_actor import (
    RayPolicyConfig,
    RayPolicyActor,
    RayPolicyController,
    create_ray_actor,
    list_checkpoint_policies,
)
from .policy_evaluator import (
    EvaluationConfig,
    EpisodeMetrics,
    PolicyEvaluator,
    run_evaluation,
)

# Lazy import for CLI to avoid circular import warning
def main(*args, **kwargs):
    """CLI entry point (lazy loaded)."""
    from .cli import main as _main
    return _main(*args, **kwargs)


def get_worker_metadata() -> tuple:
    """Return worker metadata and capabilities for MOSAIC discovery.

    This function is called by the MOSAIC worker discovery system via
    entry points to register this worker with the framework.

    Returns:
        Tuple of (WorkerMetadata, WorkerCapabilities)
    """
    from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

    metadata = WorkerMetadata(
        name="Ray RLlib Worker",
        version=__version__,
        description="Distributed multi-agent RL using Ray RLlib with PettingZoo environments",
        author="MOSAIC Team",
        homepage="https://github.com/ray-project/ray",
        upstream_library="ray[rllib]",
        upstream_version="2.40.0",
        license="Apache-2.0",
    )

    capabilities = WorkerCapabilities(
        worker_type="ray",
        supported_paradigms=("parameter_sharing", "independent", "self_play", "shared_value_function"),
        env_families=("sisl", "classic", "butterfly", "mpe"),  # PettingZoo environment families
        action_spaces=("discrete", "continuous", "multi_discrete"),
        observation_spaces=("vector", "image", "dict"),
        max_agents=100,
        supports_self_play=True,
        supports_population=True,
        supports_checkpointing=True,
        supports_pause_resume=True,
        requires_gpu=False,
        gpu_memory_mb=None,
        cpu_cores=2,  # Default: 2 workers
        estimated_memory_mb=2048,  # Ray can be memory-intensive
    )

    return metadata, capabilities

__all__ = [
    "__version__",
    # Config classes
    "PolicyConfiguration",
    "TrainingParadigm",  # Backwards compatibility alias (deprecated)
    "PettingZooAPIType",
    "ResourceConfig",
    "TrainingConfig",
    "CheckpointConfig",
    "EnvironmentConfig",
    "RayWorkerConfig",
    "load_worker_config",
    # Runtime
    "EnvironmentFactory",
    "RayWorkerRuntime",
    # Algorithm parameters (schema-based)
    "SCHEMA_VERSION",
    "get_available_algorithms",
    "get_algorithm_info",
    "get_algorithm_fields",
    "get_common_fields",
    "get_all_fields",
    "get_field_names",
    "get_default_params",
    "validate_params",
    "filter_params_for_algorithm",
    "merge_with_defaults",
    # Policy Actor (for inference)
    "RayPolicyConfig",
    "RayPolicyActor",
    "RayPolicyController",
    "create_ray_actor",
    "list_checkpoint_policies",
    # Policy Evaluator (for visualization)
    "EvaluationConfig",
    "EpisodeMetrics",
    "PolicyEvaluator",
    "run_evaluation",
    # CLI
    "main",
    # Worker Discovery
    "get_worker_metadata",
]
