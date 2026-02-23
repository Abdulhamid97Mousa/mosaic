"""Analytics manifest generation for MARLlib worker runs.

Discovers Ray Tune output artifacts (TensorBoard events, checkpoints)
and produces a ``WorkerAnalyticsManifest`` compatible with the MOSAIC GUI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

try:
    from gym_gui.core.worker import (
        WorkerAnalyticsManifest,
        ArtifactsMetadata,
        TensorBoardMetadata,
        CheckpointMetadata,
    )

    _HAS_ANALYTICS = True
except ImportError:
    _HAS_ANALYTICS = False
    WorkerAnalyticsManifest = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from .config import MARLlibWorkerConfig


def build_manifest(
    run_dir: Path,
    ray_tune_dir: Optional[Path],
    config: "MARLlibWorkerConfig",
) -> "WorkerAnalyticsManifest":
    """Build an analytics manifest from Ray Tune output.

    Ray Tune trial directories contain:
    - ``events.out.tfevents.*`` — TensorBoard logs
    - ``progress.csv`` — training metrics
    - ``params.json`` — experiment parameters
    - ``checkpoint_<N>/`` — model checkpoints

    Args:
        run_dir: MOSAIC run directory (parent of Ray Tune output).
        ray_tune_dir: Discovered Ray Tune trial directory, or ``None``.
        config: The worker configuration used for the run.

    Returns:
        A populated ``WorkerAnalyticsManifest``.

    Raises:
        ImportError: If ``gym_gui.core.worker`` is not available.
    """
    if not _HAS_ANALYTICS:
        raise ImportError(
            "gym_gui.core.worker is not available — "
            "cannot generate standardised analytics manifest."
        )

    tensorboard_meta = None
    checkpoint_meta = None

    if ray_tune_dir is not None and ray_tune_dir.exists():
        # --- TensorBoard ---
        tb_files = list(ray_tune_dir.glob("events.out.tfevents.*"))
        if tb_files:
            try:
                rel_tb = str(ray_tune_dir.relative_to(run_dir))
            except ValueError:
                rel_tb = str(ray_tune_dir)
            tensorboard_meta = TensorBoardMetadata(
                log_dir=rel_tb,
                enabled=True,
            )

        # --- Checkpoints ---
        ckpt_dirs = sorted(ray_tune_dir.glob("checkpoint_*"))
        if ckpt_dirs:
            try:
                rel_parent = str(ray_tune_dir.relative_to(run_dir))
            except ValueError:
                rel_parent = str(ray_tune_dir)
            ckpt_files = [
                str(Path(rel_parent) / d.name) for d in ckpt_dirs
            ]
            checkpoint_meta = CheckpointMetadata(
                directory=rel_parent,
                format="ray_rllib",
                files=ckpt_files,
                best_checkpoint=None,
                final_checkpoint=ckpt_files[-1] if ckpt_files else None,
            )

    artifacts = ArtifactsMetadata(
        tensorboard=tensorboard_meta,
        checkpoints=checkpoint_meta,
        logs_dir="logs",
    )

    metadata: dict[str, Any] = {
        "algo": config.algo,
        "algo_type": _get_algo_type_safe(config.algo),
        "environment": config.environment_name,
        "map_name": config.map_name,
        "share_policy": config.share_policy,
        "core_arch": config.core_arch,
        "seed": config.seed,
        "stop_timesteps": config.stop_timesteps,
    }

    return WorkerAnalyticsManifest(
        run_id=config.run_id,
        worker_type="marllib",
        artifacts=artifacts,
        metadata=metadata,
    )


def _get_algo_type_safe(algo_name: str) -> str:
    """Return algo type without raising on unknown names."""
    try:
        from .registries import get_algo_type

        return get_algo_type(algo_name)
    except (ImportError, ValueError):
        return "unknown"
