"""Analytics manifest helpers for XuanCe worker runs.

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
    from xuance_worker.config import XuanCeWorkerConfig


def write_analytics_manifest(
    config: "XuanCeWorkerConfig",
    *,
    notes: Optional[str] = None,
    tensorboard_dir: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
    videos_dir: Optional[str] = None,
) -> Path:
    """Build and write analytics manifest for a XuanCe run.

    Args:
        config: XuanCe worker configuration.
        notes: Optional notes about the run.
        tensorboard_dir: Optional TensorBoard directory (relative to run directory).
        checkpoints_dir: Optional checkpoints directory (relative to run directory).
        logs_dir: Optional logs directory (relative to run directory).
        videos_dir: Optional videos directory (relative to run directory).

    Returns:
        Path to the written manifest file.
    """
    if not _HAS_GYM_GUI:
        raise ImportError(
            "gym_gui.core.worker not available. "
            "Cannot generate standardized analytics manifest."
        )

    # Build checkpoint metadata (XuanCe uses .pth for PyTorch, .h5 for TensorFlow)
    checkpoint_format = {
        "torch": "pytorch",
        "tensorflow": "tensorflow",
        "mindspore": "mindspore",
    }.get(config.dl_toolbox, "unknown")

    checkpoints = CheckpointMetadata(
        directory=checkpoints_dir or "checkpoints",
        format=checkpoint_format,
    )

    # Build artifacts metadata
    artifacts = ArtifactsMetadata(
        tensorboard=tensorboard_dir,
        wandb=None,  # XuanCe supports WandB but not configured by default
        checkpoints=checkpoints,
        logs_dir=logs_dir or "logs",
        videos_dir=videos_dir,
    )

    # Build XuanCe-specific metadata
    metadata = {
        "worker_type": "xuance",
        "method": config.method,
        "env": config.env,
        "env_id": config.env_id,
        "dl_toolbox": config.dl_toolbox,
        "running_steps": config.running_steps,
        "seed": config.seed,
        "device": config.device,
        "parallels": config.parallels,
        "test_mode": config.test_mode,
    }

    if notes:
        metadata["notes"] = notes

    # Determine manifest path (use current working directory + run_id)
    manifest_path = Path.cwd() / f"{config.run_id}_analytics.json"

    # Create the manifest
    manifest = WorkerAnalyticsManifest(
        run_id=config.run_id,
        worker_type="xuance",
        artifacts=artifacts,
        metadata=metadata,
    )

    # Save to analytics.json
    manifest.save(manifest_path)
    return manifest_path


__all__ = [
    "write_analytics_manifest",
]
