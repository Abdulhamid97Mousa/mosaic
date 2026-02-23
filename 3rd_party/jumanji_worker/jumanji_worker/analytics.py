"""Analytics manifest helpers for Jumanji worker runs.

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
        TensorBoardMetadata,
        CheckpointMetadata,
    )
    _HAS_GYM_GUI = True
except ImportError:
    _HAS_GYM_GUI = False
    WorkerAnalyticsManifest = None
    ArtifactsMetadata = None
    TensorBoardMetadata = None
    CheckpointMetadata = None

if TYPE_CHECKING:
    from jumanji_worker.config import JumanjiWorkerConfig


def write_analytics_manifest(
    config: "JumanjiWorkerConfig",
    *,
    notes: Optional[str] = None,
    tensorboard_dir: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
) -> Path:
    """Build and write analytics manifest for a Jumanji run.

    Args:
        config: Jumanji worker configuration.
        notes: Optional notes about the run.
        tensorboard_dir: Optional TensorBoard directory (relative to run directory).
        checkpoints_dir: Optional checkpoints directory (relative to run directory).
        logs_dir: Optional logs directory (relative to run directory).

    Returns:
        Path to the written manifest file.

    Raises:
        ImportError: If gym_gui.core.worker is not available.
    """
    if not _HAS_GYM_GUI:
        raise ImportError(
            "gym_gui.core.worker not available. "
            "Cannot generate standardized analytics manifest."
        )

    # Build TensorBoard metadata if logger type is tensorboard
    tensorboard = None
    if config.logger_type == "tensorboard":
        tensorboard = TensorBoardMetadata(
            log_dir=tensorboard_dir or "tensorboard",
            enabled=True,
        )

    # Build checkpoint metadata (Jumanji uses JAX-style checkpoints)
    checkpoints = None
    if config.save_checkpoint:
        checkpoints = CheckpointMetadata(
            directory=checkpoints_dir or "checkpoints",
            format="jax",  # JAX-style serialization
        )

    # Build artifacts metadata
    artifacts = ArtifactsMetadata(
        tensorboard=tensorboard,
        wandb=None,  # Jumanji native training doesn't use W&B by default
        checkpoints=checkpoints,
        logs_dir=logs_dir or "logs",
        videos_dir=None,  # Jumanji logic envs don't produce videos
    )

    # Build Jumanji-specific metadata
    metadata = {
        "worker_type": "jumanji",
        "env_id": config.env_id,
        "agent": config.agent,
        "device": config.device,
        "num_epochs": config.num_epochs,
        "n_steps": config.n_steps,
        "total_batch_size": config.total_batch_size,
        "num_learner_steps_per_epoch": config.num_learner_steps_per_epoch,
        "learning_rate": config.learning_rate,
        "discount_factor": config.discount_factor,
        "bootstrapping_factor": config.bootstrapping_factor,
        "normalize_advantage": config.normalize_advantage,
        "l_pg": config.l_pg,
        "l_td": config.l_td,
        "l_en": config.l_en,
        "seed": config.seed,
    }

    if notes:
        metadata["notes"] = notes

    # Determine manifest path (use current working directory + run_id)
    manifest_path = Path.cwd() / f"{config.run_id}_analytics.json"

    # Create the manifest
    manifest = WorkerAnalyticsManifest(
        run_id=config.run_id,
        worker_type="jumanji",
        artifacts=artifacts,
        metadata=metadata,
    )

    # Save to analytics.json
    manifest.save(manifest_path)
    return manifest_path


def detect_artifacts(run_dir: Path) -> dict:
    """Auto-detect artifacts in a Jumanji run directory.

    This function scans a run directory and identifies common artifact
    locations for TensorBoard logs, checkpoints, and log files.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Dictionary with detected artifact paths.
    """
    artifacts = {
        "tensorboard_dir": None,
        "checkpoints_dir": None,
        "logs_dir": None,
    }

    # Check for TensorBoard directory
    tb_candidates = ["tensorboard", "tb_logs", "logs/tensorboard"]
    for candidate in tb_candidates:
        tb_path = run_dir / candidate
        if tb_path.exists() and tb_path.is_dir():
            artifacts["tensorboard_dir"] = candidate
            break

    # Check for checkpoints directory
    ckpt_candidates = ["checkpoints", "ckpts", "models"]
    for candidate in ckpt_candidates:
        ckpt_path = run_dir / candidate
        if ckpt_path.exists() and ckpt_path.is_dir():
            artifacts["checkpoints_dir"] = candidate
            break

    # Check for logs directory
    log_candidates = ["logs", "log"]
    for candidate in log_candidates:
        log_path = run_dir / candidate
        if log_path.exists() and log_path.is_dir():
            artifacts["logs_dir"] = candidate
            break

    return artifacts


__all__ = [
    "write_analytics_manifest",
    "detect_artifacts",
]
