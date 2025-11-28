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

    Paths are stored as relative paths (relative to the run directory) for portability.
    The GUI resolves them to absolute paths at load time based on VAR_ROOT.
    """

    tensorboard_relative: Optional[str] = None
    wandb_run_path: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    optuna_relative: Optional[str] = None
    checkpoints_relative: Optional[str] = None
    notes: Optional[str] = None
    # run_id is stored to allow resolution of relative paths
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested artifacts structure matching GUI expectations.

        Uses relative paths for portability. The GUI resolves these paths
        relative to VAR_TRAINER_DIR/runs/<run_id>/.
        """
        return {
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
            },
            # Store run_id so GUI can resolve relative paths
            "run_id": self.run_id,
            # Keep legacy flat fields for backward compatibility (now relative)
            "tensorboard_dir": self.tensorboard_relative,
            "wandb_run_path": self.wandb_run_path,
            "optuna_db_path": self.optuna_relative,
            "checkpoints_dir": self.checkpoints_relative,
            "notes": self.notes,
        }


def build_manifest(
    run_dir: Path, *, extras: Dict[str, Any], run_id: Optional[str] = None
) -> AnalyticsManifest:
    """Inspect the run directory and extras payload to build a manifest.

    Extracts analytics metadata from the extras dict and constructs a manifest
    with nested artifacts structure matching GUI expectations.

    All paths are stored as relative paths (relative to run_dir) for portability.
    The GUI resolves them at load time using the run_id.
    """
    # Extract the relative tensorboard directory name (e.g., "tensorboard" or "tensorboard_eval")
    tensorboard_dir = extras.get("tensorboard_dir")
    tensorboard_relative: Optional[str] = None
    if isinstance(tensorboard_dir, str) and tensorboard_dir.strip():
        # Store just the relative directory name, not an absolute path
        tensorboard_relative = tensorboard_dir.strip()

    wandb_run = extras.get("wandb_run_path")
    wandb_entity = extras.get("wandb_entity")
    wandb_project = extras.get("wandb_project_name") or extras.get("wandb_project")

    # Store relative paths for optuna and checkpoints as well
    optuna_db = extras.get("optuna_db")
    optuna_relative: Optional[str] = None
    if isinstance(optuna_db, str) and optuna_db.strip():
        optuna_relative = optuna_db.strip()

    checkpoints = extras.get("checkpoints_dir")
    checkpoints_relative: Optional[str] = None
    if isinstance(checkpoints, str) and checkpoints.strip():
        checkpoints_relative = checkpoints.strip()

    notes = extras.get("notes") if isinstance(extras.get("notes"), str) else None

    # Try to extract run_id from run_dir if not provided
    if run_id is None:
        run_id = run_dir.name

    return AnalyticsManifest(
        tensorboard_relative=tensorboard_relative,
        wandb_run_path=str(wandb_run) if isinstance(wandb_run, str) else None,
        wandb_entity=str(wandb_entity) if isinstance(wandb_entity, str) else None,
        wandb_project=str(wandb_project) if isinstance(wandb_project, str) else None,
        optuna_relative=optuna_relative,
        checkpoints_relative=checkpoints_relative,
        notes=notes,
        run_id=run_id,
    )
