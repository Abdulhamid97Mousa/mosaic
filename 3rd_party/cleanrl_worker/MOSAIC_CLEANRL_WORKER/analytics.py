"""Analytics manifest helpers for CleanRL worker runs."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AnalyticsManifest:
    """Describe analytics artifacts produced by a CleanRL run.
    
    Uses nested structure matching GUI expectations:
    - artifacts.tensorboard for TensorBoard integration
    - artifacts.wandb for WANDB integration
    """

    tensorboard_dir: Optional[str] = None
    wandb_run_path: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    optuna_db_path: Optional[str] = None
    checkpoints_dir: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested artifacts structure matching GUI expectations."""
        return {
            "artifacts": {
                "tensorboard": {
                    "enabled": self.tensorboard_dir is not None,
                    "log_dir": self.tensorboard_dir,
                    "relative_path": self.tensorboard_dir,
                },
                "wandb": {
                    "enabled": self.wandb_run_path is not None,
                    "run_path": self.wandb_run_path,
                    "entity": self.wandb_entity,
                    "project": self.wandb_project,
                },
            },
            # Keep legacy flat fields for backward compatibility
            "tensorboard_dir": self.tensorboard_dir,
            "wandb_run_path": self.wandb_run_path,
            "optuna_db_path": self.optuna_db_path,
            "checkpoints_dir": self.checkpoints_dir,
            "notes": self.notes,
        }


def build_manifest(run_dir: Path, *, extras: Dict[str, Any]) -> AnalyticsManifest:
    """Inspect the run directory and extras payload to build a manifest.
    
    Extracts analytics metadata from the extras dict and constructs a manifest
    with nested artifacts structure matching GUI expectations.
    """

    tensorboard_dir = extras.get("tensorboard_dir")
    if isinstance(tensorboard_dir, str):
        candidate = (run_dir / tensorboard_dir).expanduser()
        tensorboard_path = str(candidate)
    else:
        tensorboard_path = None

    wandb_run = extras.get("wandb_run_path")
    wandb_entity = extras.get("wandb_entity")
    wandb_project = extras.get("wandb_project_name") or extras.get("wandb_project")
    optuna_db = extras.get("optuna_db")
    checkpoints = extras.get("checkpoints_dir")
    notes = extras.get("notes") if isinstance(extras.get("notes"), str) else None

    return AnalyticsManifest(
        tensorboard_dir=tensorboard_path,
        wandb_run_path=str(wandb_run) if isinstance(wandb_run, str) else None,
        wandb_entity=str(wandb_entity) if isinstance(wandb_entity, str) else None,
        wandb_project=str(wandb_project) if isinstance(wandb_project, str) else None,
        optuna_db_path=str(run_dir / optuna_db) if isinstance(optuna_db, str) else None,
        checkpoints_dir=str(run_dir / checkpoints) if isinstance(checkpoints, str) else None,
        notes=notes,
    )
