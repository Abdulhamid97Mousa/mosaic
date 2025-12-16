"""Analytics manifest helpers for Ray RLlib worker runs.

This module provides helpers to create and write analytics manifests that
integrate with the MOSAIC GUI's analytics tab system.

The manifest structure matches the expected format from CleanRL worker:
- artifacts.tensorboard for TensorBoard integration
- artifacts.wandb for WANDB integration
- Relative paths for portability

See Also:
    - 3rd_party/cleanrl_worker/cleanrl_worker/analytics.py
    - gym_gui/config/paths.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ray_worker.config import RayWorkerConfig


@dataclass(frozen=True)
class RayAnalyticsManifest:
    """Describe analytics artifacts produced by a Ray RLlib run.

    Uses nested structure matching GUI expectations:
    - artifacts.tensorboard for TensorBoard integration
    - artifacts.wandb for WANDB integration
    - artifacts.ray_checkpoints for Ray RLlib checkpoint paths

    Paths are stored as relative paths (relative to the run directory) for portability.
    """

    run_id: str
    tensorboard_relative: Optional[str] = None
    wandb_run_path: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    checkpoints_relative: Optional[str] = None
    logs_relative: Optional[str] = None
    videos_relative: Optional[str] = None
    notes: Optional[str] = None
    # Ray-specific metadata
    paradigm: Optional[str] = None
    algorithm: Optional[str] = None
    env_id: Optional[str] = None
    env_family: Optional[str] = None
    num_agents: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested artifacts structure matching GUI expectations."""
        return {
            "run_id": self.run_id,
            "worker_type": "ray_worker",
            "artifacts": {
                "tensorboard": {
                    "enabled": self.tensorboard_relative is not None,
                    "relative_path": self.tensorboard_relative,
                },
                "wandb": {
                    "enabled": self.wandb_run_path is not None,
                    "run_path": self.wandb_run_path,
                    "entity": self.wandb_entity,
                    "project": self.wandb_project,
                },
                "checkpoints": {
                    "enabled": self.checkpoints_relative is not None,
                    "relative_path": self.checkpoints_relative,
                },
                "logs": {
                    "enabled": self.logs_relative is not None,
                    "relative_path": self.logs_relative,
                },
                "videos": {
                    "enabled": self.videos_relative is not None,
                    "relative_path": self.videos_relative,
                },
            },
            # Ray-specific metadata for UI display
            "ray_metadata": {
                "paradigm": self.paradigm,
                "algorithm": self.algorithm,
                "env_id": self.env_id,
                "env_family": self.env_family,
                "num_agents": self.num_agents,
            },
            # Legacy flat fields for backward compatibility
            "tensorboard_dir": self.tensorboard_relative,
            "checkpoints_dir": self.checkpoints_relative,
            "notes": self.notes,
        }

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "RayAnalyticsManifest":
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)

        artifacts = data.get("artifacts", {})
        tb = artifacts.get("tensorboard", {})
        wandb = artifacts.get("wandb", {})
        checkpoints = artifacts.get("checkpoints", {})
        logs = artifacts.get("logs", {})
        videos = artifacts.get("videos", {})
        ray_meta = data.get("ray_metadata", {})

        return cls(
            run_id=data.get("run_id", ""),
            tensorboard_relative=tb.get("relative_path"),
            wandb_run_path=wandb.get("run_path"),
            wandb_entity=wandb.get("entity"),
            wandb_project=wandb.get("project"),
            checkpoints_relative=checkpoints.get("relative_path"),
            logs_relative=logs.get("relative_path"),
            videos_relative=videos.get("relative_path"),
            notes=data.get("notes"),
            paradigm=ray_meta.get("paradigm"),
            algorithm=ray_meta.get("algorithm"),
            env_id=ray_meta.get("env_id"),
            env_family=ray_meta.get("env_family"),
            num_agents=ray_meta.get("num_agents"),
        )


def build_manifest_from_config(config: "RayWorkerConfig") -> RayAnalyticsManifest:
    """Build an analytics manifest from RayWorkerConfig.

    Args:
        config: Ray worker configuration.

    Returns:
        RayAnalyticsManifest ready to be saved.
    """
    return RayAnalyticsManifest(
        run_id=config.run_id,
        tensorboard_relative=config.tensorboard_relative_path,
        wandb_run_path=None,  # Set after wandb.init()
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        checkpoints_relative="checkpoints",
        logs_relative="logs",
        videos_relative="videos",
        paradigm=config.paradigm.value if hasattr(config.paradigm, "value") else str(config.paradigm),
        algorithm=config.training.algorithm,
        env_id=config.environment.env_id,
        env_family=config.environment.family,
    )


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
    manifest = RayAnalyticsManifest(
        run_id=config.run_id,
        tensorboard_relative=config.tensorboard_relative_path,
        wandb_run_path=wandb_run_path,
        wandb_entity=config.wandb_entity,
        wandb_project=config.wandb_project,
        checkpoints_relative="checkpoints",
        logs_relative="logs",
        videos_relative="videos",
        notes=notes,
        paradigm=config.paradigm.value if hasattr(config.paradigm, "value") else str(config.paradigm),
        algorithm=config.training.algorithm,
        env_id=config.environment.env_id,
        env_family=config.environment.family,
        num_agents=num_agents,
    )

    manifest_path = config.analytics_manifest_path
    manifest.save(manifest_path)
    return manifest_path


__all__ = [
    "RayAnalyticsManifest",
    "build_manifest_from_config",
    "write_analytics_manifest",
]
