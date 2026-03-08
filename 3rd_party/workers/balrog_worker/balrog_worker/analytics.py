"""Analytics manifest helpers for BARLOG worker runs.

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

try:
    from gym_gui.core.worker import (
        WorkerAnalyticsManifest,
        ArtifactsMetadata,
        CheckpointMetadata,
    )
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    WorkerAnalyticsManifest = None
    ArtifactsMetadata = None
    CheckpointMetadata = None

if TYPE_CHECKING:
    from balrog_worker.config import BarlogWorkerConfig


def write_analytics_manifest(
    config: "BarlogWorkerConfig",
    *,
    notes: Optional[str] = None,
) -> Path:
    """Build and write analytics manifest for a BARLOG run.

    Args:
        config: BARLOG worker configuration.
        notes: Optional notes about the run.

    Returns:
        Path to the written manifest file.
    """
    if not _HAS_GYM_GUI:
        raise ImportError(
            "gym_gui.core.worker not available. "
            "Cannot generate standardized analytics manifest."
        )

    # Build checkpoint metadata (BARLOG doesn't use checkpoints like RL agents)
    checkpoints = CheckpointMetadata(
        directory=None,
        format="none",
    )

    # Build artifacts metadata
    artifacts = ArtifactsMetadata(
        tensorboard=None,  # BARLOG doesn't use TensorBoard
        wandb=None,  # BARLOG doesn't use WandB yet
        checkpoints=checkpoints,
        logs_dir="logs",
        videos_dir=None,
    )

    # Build BARLOG-specific metadata
    metadata = {
        "worker_type": "balrog",
        "env_name": config.env_name,
        "task": config.task,
        "client_name": config.client_name,
        "model_id": config.model_id,
        "agent_type": config.agent_type,
        "num_episodes": config.num_episodes,
        "max_steps": config.max_steps,
        "temperature": config.temperature,
        "seed": config.seed,
        "max_image_history": config.max_image_history,
    }

    if notes:
        metadata["notes"] = notes

    # Determine manifest path
    telemetry_dir = Path(config.telemetry_dir)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = telemetry_dir / f"{config.run_id}_analytics.json"

    # Create the manifest
    manifest = WorkerAnalyticsManifest(
        run_id=config.run_id,
        worker_type="balrog",
        artifacts=artifacts,
        metadata=metadata,
    )

    # Save to analytics.json
    manifest.save(manifest_path)
    return manifest_path


__all__ = [
    "write_analytics_manifest",
]
