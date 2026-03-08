"""Analytics manifest helpers for VLM worker runs.

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
    from vlm_worker.config import VLMWorkerConfig


def write_analytics_manifest(
    config: "VLMWorkerConfig",
    *,
    notes: Optional[str] = None,
) -> Path:
    """Build and write analytics manifest for a VLM run.

    Args:
        config: VLM worker configuration.
        notes: Optional notes about the run.

    Returns:
        Path to the written manifest file.
    """
    if not _HAS_GYM_GUI:
        raise ImportError(
            "gym_gui.core.worker not available. "
            "Cannot generate standardized analytics manifest."
        )

    # Build checkpoint metadata (VLM doesn't use checkpoints like RL agents)
    checkpoints = CheckpointMetadata(
        directory=None,
        format="none",
    )

    # Build artifacts metadata
    artifacts = ArtifactsMetadata(
        tensorboard=None,
        wandb=None,
        checkpoints=checkpoints,
        logs_dir="logs",
        videos_dir=None,
    )

    # Build VLM-specific metadata
    metadata = {
        "worker_type": "vlm",
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
        worker_type="vlm",
        artifacts=artifacts,
        metadata=metadata,
    )

    # Save to analytics.json
    manifest.save(manifest_path)
    return manifest_path


__all__ = [
    "write_analytics_manifest",
]
