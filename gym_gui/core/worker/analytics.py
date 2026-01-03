"""Standardized analytics manifest for MOSAIC workers.

All workers should generate an analytics manifest file that describes the
artifacts produced during training (TensorBoard logs, W&B runs, checkpoints, etc.).

The GUI reads this manifest to display analytics tabs and artifact links.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class TensorBoardMetadata:
    """TensorBoard artifact metadata.

    Attributes:
        log_dir: Relative path to TensorBoard log directory
        enabled: Whether TensorBoard logging was enabled
        port: TensorBoard server port (if launched)
        url: TensorBoard server URL (if launched)
    """

    log_dir: str
    enabled: bool
    port: Optional[int] = None
    url: Optional[str] = None


@dataclass(frozen=True)
class WandBMetadata:
    """Weights & Biases artifact metadata.

    Attributes:
        project: W&B project name
        entity: W&B entity/team name
        run_id: W&B run identifier
        run_name: W&B run display name
        url: W&B run URL
        enabled: Whether W&B tracking was enabled
    """

    project: str
    entity: Optional[str]
    run_id: str
    run_name: str
    url: str
    enabled: bool


@dataclass(frozen=True)
class CheckpointMetadata:
    """Model checkpoint metadata.

    Attributes:
        directory: Relative path to checkpoints directory
        format: Checkpoint format (e.g., "pytorch", "tensorflow", "jax")
        files: List of checkpoint file paths (relative)
        best_checkpoint: Path to best checkpoint (if any)
        final_checkpoint: Path to final checkpoint
    """

    directory: str
    format: str
    files: list[str] = field(default_factory=list)
    best_checkpoint: Optional[str] = None
    final_checkpoint: Optional[str] = None


@dataclass(frozen=True)
class ArtifactsMetadata:
    """Container for all artifact metadata.

    Attributes:
        tensorboard: TensorBoard metadata (if available)
        wandb: W&B metadata (if available)
        checkpoints: Checkpoint metadata (if available)
        logs_dir: Relative path to log files directory
        videos_dir: Relative path to recorded videos (if any)
    """

    tensorboard: Optional[TensorBoardMetadata] = None
    wandb: Optional[WandBMetadata] = None
    checkpoints: Optional[CheckpointMetadata] = None
    logs_dir: Optional[str] = None
    videos_dir: Optional[str] = None


@dataclass(frozen=True)
class WorkerAnalyticsManifest:
    """Universal analytics manifest for all MOSAIC workers.

    This manifest should be written to:
        var/trainer/runs/<run_id>/analytics.json

    The GUI reads this file to populate analytics tabs and artifact links.

    Attributes:
        run_id: Unique run identifier
        worker_type: Worker type identifier (e.g., "cleanrl", "ray")
        artifacts: Artifact metadata
        metadata: Worker-specific metadata

    Example:
        manifest = WorkerAnalyticsManifest(
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            worker_type="cleanrl",
            artifacts=ArtifactsMetadata(
                tensorboard=TensorBoardMetadata(
                    log_dir="tensorboard",
                    enabled=True
                ),
                checkpoints=CheckpointMetadata(
                    directory="checkpoints",
                    format="pytorch",
                    files=["model_1000.pt", "model_2000.pt"],
                    final_checkpoint="model_final.pt"
                ),
                logs_dir="logs"
            ),
            metadata={
                "algo": "ppo",
                "env_id": "CartPole-v1",
                "total_timesteps": 10000,
                "episodes_completed": 100,
                "final_reward": 195.0
            }
        )

        manifest.save(Path("var/trainer/runs/01ARZ.../analytics.json"))
    """

    run_id: str
    worker_type: str
    artifacts: ArtifactsMetadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest to dictionary.

        Returns:
            Dictionary representation with nested structure:
            {
                "run_id": str,
                "worker_type": str,
                "artifacts": {
                    "tensorboard": {...},
                    "wandb": {...},
                    "checkpoints": {...},
                    "logs_dir": str,
                    "videos_dir": str
                },
                "metadata": {...}
            }
        """
        return {
            "run_id": self.run_id,
            "worker_type": self.worker_type,
            "artifacts": self._serialize_artifacts(),
            "metadata": self.metadata,
        }

    def _serialize_artifacts(self) -> Dict[str, Any]:
        """Serialize artifacts to dictionary."""
        result: Dict[str, Any] = {}

        if self.artifacts.tensorboard:
            result["tensorboard"] = asdict(self.artifacts.tensorboard)

        if self.artifacts.wandb:
            result["wandb"] = asdict(self.artifacts.wandb)

        if self.artifacts.checkpoints:
            result["checkpoints"] = asdict(self.artifacts.checkpoints)

        if self.artifacts.logs_dir:
            result["logs_dir"] = self.artifacts.logs_dir

        if self.artifacts.videos_dir:
            result["videos_dir"] = self.artifacts.videos_dir

        return result

    def save(self, path: Path) -> None:
        """Save manifest to JSON file.

        Args:
            path: Path to analytics.json file

        Example:
            manifest.save(Path("var/trainer/runs/01ARZ.../analytics.json"))
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> WorkerAnalyticsManifest:
        """Load manifest from JSON file.

        Args:
            path: Path to analytics.json file

        Returns:
            Deserialized manifest

        Example:
            manifest = WorkerAnalyticsManifest.load(
                Path("var/trainer/runs/01ARZ.../analytics.json")
            )
        """
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkerAnalyticsManifest:
        """Deserialize manifest from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Manifest instance
        """
        artifacts_data = data.get("artifacts", {})

        # Deserialize nested artifacts
        tensorboard = None
        if "tensorboard" in artifacts_data:
            tensorboard = TensorBoardMetadata(**artifacts_data["tensorboard"])

        wandb = None
        if "wandb" in artifacts_data:
            wandb = WandBMetadata(**artifacts_data["wandb"])

        checkpoints = None
        if "checkpoints" in artifacts_data:
            checkpoints = CheckpointMetadata(**artifacts_data["checkpoints"])

        artifacts = ArtifactsMetadata(
            tensorboard=tensorboard,
            wandb=wandb,
            checkpoints=checkpoints,
            logs_dir=artifacts_data.get("logs_dir"),
            videos_dir=artifacts_data.get("videos_dir"),
        )

        return cls(
            run_id=data["run_id"],
            worker_type=data["worker_type"],
            artifacts=artifacts,
            metadata=data.get("metadata", {}),
        )
