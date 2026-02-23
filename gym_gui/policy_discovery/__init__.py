"""Policy discovery utilities for trained checkpoints.

This package contains:
- Policy metadata loaders for discovering trained checkpoints
- Support for CleanRL, Ray RLlib, and PettingZoo checkpoints
"""

from gym_gui.policy_discovery.cleanrl_policy_metadata import (
    CleanRlCheckpoint,
    discover_policies as discover_cleanrl_policies,
    load_metadata_for_policy as load_cleanrl_metadata,
)
from gym_gui.policy_discovery.ray_policy_metadata import (
    RayRLlibCheckpoint,
    RayRLlibPolicy,
    discover_ray_checkpoints,
    discover_ray_policies,
    get_checkpoints_for_env,
    get_latest_checkpoint,
    load_checkpoint_metadata as load_ray_checkpoint_metadata,
)

__all__ = [
    # CleanRL
    "CleanRlCheckpoint",
    "discover_cleanrl_policies",
    "load_cleanrl_metadata",
    # Ray RLlib
    "RayRLlibCheckpoint",
    "RayRLlibPolicy",
    "discover_ray_checkpoints",
    "discover_ray_policies",
    "get_checkpoints_for_env",
    "get_latest_checkpoint",
    "load_ray_checkpoint_metadata",
]
