"""Ray/RLlib worker integration for MOSAIC BDI-RL framework.

This module provides the integration layer between MOSAIC's GUI and
Ray/RLlib's distributed reinforcement learning capabilities.

Key Features:
- Multi-agent training with RLlib algorithms (PPO, DQN, A2C, IMPALA, APPO)
- Distributed training across multiple workers
- Support for both discrete and continuous action spaces
- PettingZoo environment integration (SISL, Classic, Butterfly, MPE)
- Multiple training paradigms (parameter sharing, independent, self-play)

Training Paradigms:
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
        paradigm=TrainingParadigm.PARAMETER_SHARING,
    )
    runtime = RayWorkerRuntime(config)
    runtime.run()
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import (
    TrainingParadigm,
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

# Lazy import for CLI to avoid circular import warning
def main(*args, **kwargs):
    """CLI entry point (lazy loaded)."""
    from .cli import main as _main
    return _main(*args, **kwargs)

__all__ = [
    "__version__",
    # Config classes
    "TrainingParadigm",
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
    # CLI
    "main",
]
