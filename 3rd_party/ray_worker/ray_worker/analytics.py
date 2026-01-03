"""Analytics manifest helpers for Ray RLlib worker runs.

This module provides helpers to create and write analytics manifests that
integrate with the MOSAIC GUI's analytics tab system using the standardized
WorkerAnalyticsManifest format.

See Also:
    - gym_gui/core/worker/analytics.py for manifest structure
    - gym_gui/config/paths.py for path constants
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from gym_gui.core.worker import (
    WorkerAnalyticsManifest,
    ArtifactsMetadata,
    TensorBoardMetadata,
    WandBMetadata,
    CheckpointMetadata,
)

if TYPE_CHECKING:
    from ray_worker.config import RayWorkerConfig


def write_analytics_manifest(
    config: "RayWorkerConfig",
    *,
    wandb_run_path: Optional[str] = None,
    notes: Optional[str] = None,
    num_agents: Optional[int] = None,
) -> Path:
    """Build and write analytics manifest for a Ray run.

    Args:
        config: Ray worker configuration.
        wandb_run_path: Optional W&B run path after wandb.init().
        notes: Optional notes about the run.
        num_agents: Number of agents in the environment.

    Returns:
        Path to the written manifest file.
    """
    # Build TensorBoard metadata if enabled
    tensorboard = None
    if config.tensorboard and config.tensorboard_relative_path:
        tensorboard = TensorBoardMetadata(
            log_dir=config.tensorboard_relative_path,
            enabled=True,
        )

    # Build WandB metadata if enabled
    wandb = None
    if config.wandb and wandb_run_path:
        # Parse W&B run path to extract run_id
        # Format: "entity/project/run_id"
        parts = wandb_run_path.split("/")
        wandb_run_id = parts[-1] if len(parts) >= 3 else wandb_run_path

        wandb = WandBMetadata(
            project=config.wandb_project or "ray-marl",
            entity=config.wandb_entity,
            run_id=wandb_run_id,
            run_name=config.wandb_run_name or config.run_id,
            url=f"https://wandb.ai/{wandb_run_path}" if wandb_run_path else None,
            enabled=True,
        )

    # Build checkpoint metadata
    checkpoints = CheckpointMetadata(
        directory="checkpoints",
        format="ray_rllib",
    )

    # Build artifacts metadata
    artifacts = ArtifactsMetadata(
        tensorboard=tensorboard,
        wandb=wandb,
        checkpoints=checkpoints,
        logs_dir="logs",
        videos_dir="videos" if config.fastlane_enabled else None,
    )

    # Build Ray-specific metadata
    metadata = {
        "worker_type": "ray",
        "policy_configuration": config.policy_configuration.value if hasattr(config.policy_configuration, "value") else str(config.policy_configuration),
        "algorithm": config.training.algorithm,
        "env_id": config.environment.env_id,
        "env_family": config.environment.family,
        "full_env_id": config.environment.full_env_id,
        "api_type": config.environment.api_type.value if hasattr(config.environment.api_type, "value") else str(config.environment.api_type),
        "total_timesteps": config.training.total_timesteps,
        "num_agents": num_agents,
        "num_workers": config.resources.num_workers,
        "num_gpus": config.resources.num_gpus,
        "seed": config.seed,
        "checkpoint_freq": config.checkpoint.checkpoint_freq,
        "keep_checkpoints_num": config.checkpoint.keep_checkpoints_num,
    }

    if notes:
        metadata["notes"] = notes

    # Create the manifest
    manifest = WorkerAnalyticsManifest(
        run_id=config.run_id,
        worker_type="ray",
        artifacts=artifacts,
        metadata=metadata,
    )

    # Save to analytics.json
    manifest_path = config.analytics_manifest_path
    manifest.save(manifest_path)
    return manifest_path


__all__ = [
    "write_analytics_manifest",
]
