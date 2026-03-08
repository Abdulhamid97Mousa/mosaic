"""Analytics manifest helpers for MCTX worker runs.

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
    from mctx_worker.config import MCTXWorkerConfig


def write_analytics_manifest(
    config: "MCTXWorkerConfig",
    *,
    wandb_run_path: Optional[str] = None,
    notes: Optional[str] = None,
) -> Path:
    """Build and write analytics manifest for an MCTX run.

    Args:
        config: MCTX worker configuration.
        wandb_run_path: Optional W&B run path after wandb.init().
        notes: Optional notes about the run.

    Returns:
        Path to the written manifest file.
    """
    # Determine run directory
    run_dir = Path(config.checkpoint_path or f"var/trainer/runs/{config.run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build TensorBoard metadata
    tensorboard = TensorBoardMetadata(
        log_dir="tensorboard",
        enabled=True,
    )

    # Build WandB metadata if provided
    wandb = None
    if wandb_run_path:
        parts = wandb_run_path.split("/")
        wandb_run_id = parts[-1] if len(parts) >= 3 else wandb_run_path

        wandb = WandBMetadata(
            project="mctx-training",
            entity=None,
            run_id=wandb_run_id,
            run_name=config.run_id,
            url=f"https://wandb.ai/{wandb_run_path}" if wandb_run_path else None,
            enabled=True,
        )

    # Build checkpoint metadata
    checkpoints = CheckpointMetadata(
        directory="checkpoints",
        format="flax",  # Using Flax for JAX models
    )

    # Build artifacts metadata
    artifacts = ArtifactsMetadata(
        tensorboard=tensorboard,
        wandb=wandb,
        checkpoints=checkpoints,
        logs_dir="logs",
        videos_dir=None,  # MCTX doesn't generate videos
    )

    # Build MCTX-specific metadata
    metadata = {
        "worker_type": "mctx",
        "env_id": config.env_id,
        "algorithm": config.algorithm.value if hasattr(config.algorithm, "value") else str(config.algorithm),
        "device": config.device,
        "seed": config.seed,
        "max_steps": config.max_steps,
        "max_episodes": config.max_episodes,
        "mode": config.mode,
        # Network config
        "network": {
            "num_res_blocks": config.network.num_res_blocks,
            "channels": config.network.channels,
            "hidden_dims": list(config.network.hidden_dims),
        },
        # MCTS config
        "mcts": {
            "num_simulations": config.mcts.num_simulations,
            "dirichlet_alpha": config.mcts.dirichlet_alpha,
            "temperature": config.mcts.temperature,
        },
        # Training config
        "training": {
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "replay_buffer_size": config.training.replay_buffer_size,
        },
    }

    if notes:
        metadata["notes"] = notes

    # Create the manifest
    manifest = WorkerAnalyticsManifest(
        run_id=config.run_id,
        worker_type="mctx",
        artifacts=artifacts,
        metadata=metadata,
    )

    # Save to analytics.json
    manifest_path = run_dir / "analytics.json"
    manifest.save(manifest_path)
    return manifest_path


__all__ = [
    "write_analytics_manifest",
]
