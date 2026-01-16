"""Analytics manifest helpers for LLM worker runs.

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
    from llm_worker.config import LLMWorkerConfig


def write_analytics_manifest(
    config: "LLMWorkerConfig",
    *,
    notes: Optional[str] = None,
) -> Path:
    """Build and write analytics manifest for an LLM worker run.

    Args:
        config: LLM worker configuration.
        notes: Optional notes about the run.

    Returns:
        Path to the written manifest file.
    """
    if not _HAS_GYM_GUI:
        raise ImportError(
            "gym_gui.core.worker not available. "
            "Cannot generate standardized analytics manifest."
        )

    # Build checkpoint metadata (LLM workers don't use checkpoints like RL agents)
    checkpoints = CheckpointMetadata(
        directory=None,
        format="none",
    )

    # Build artifacts metadata
    artifacts = ArtifactsMetadata(
        tensorboard=None,  # LLM workers don't use TensorBoard
        wandb=None,
        checkpoints=checkpoints,
        logs_dir="logs",
        videos_dir=None,
    )

    # Build LLM worker-specific metadata
    metadata = {
        "worker_type": "llm",
        "env_name": config.env_name,
        "task": config.task,
        "num_agents": config.num_agents,
        "client_name": config.client_name,
        "model_id": config.model_id,
        "coordination_level": config.coordination_level,
        "observation_mode": config.observation_mode,
        "num_episodes": config.num_episodes,
        "max_steps_per_episode": config.max_steps_per_episode,
        "temperature": config.temperature,
        "seed": config.seed,
    }

    if config.agent_roles:
        metadata["agent_roles"] = config.agent_roles

    if notes:
        metadata["notes"] = notes

    # Determine manifest path
    telemetry_dir = Path(config.telemetry_dir)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = telemetry_dir / f"{config.run_id}_analytics.json"

    # Create the manifest
    manifest = WorkerAnalyticsManifest(
        run_id=config.run_id,
        worker_type="llm",
        artifacts=artifacts,
        metadata=metadata,
    )

    # Save to analytics.json
    manifest.save(manifest_path)
    return manifest_path


__all__ = [
    "write_analytics_manifest",
]
