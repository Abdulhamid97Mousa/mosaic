from __future__ import annotations

"""TensorBoard defaults and helpers centralised for the GUI stack."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gym_gui.config.paths import VAR_TENSORBOARD_DIR, ensure_var_directories


@dataclass(frozen=True)
class TensorboardDefaults:
    """Runtime defaults for embedding TensorBoard within the GUI."""

    server_host: str = "127.0.0.1"
    default_port: int = 6006
    status_refresh_ms: int = 4000
    server_probe_interval_ms: int = 750
    port_probe_attempts: int = 12
    cli_executable: str = "tensorboard"


DEFAULT_TENSORBOARD = TensorboardDefaults()


def build_tensorboard_relative_path(run_id: str, worker_id: Optional[str] = None) -> str:
    """Compute the canonical relative path for a run's TensorBoard logs."""

    base = f"var/trainer/runs/{run_id}"
    # Future multi-worker support could append worker-specific subdirectories here.
    return f"{base}/tensorboard"


def build_tensorboard_log_dir(run_id: str, worker_id: Optional[str] = None) -> Path:
    """Return the absolute path for a run's TensorBoard directory."""

    ensure_var_directories()
    root = VAR_TENSORBOARD_DIR / run_id
    # Mirror the relative path helper so callers stay consistent.
    return (root / "tensorboard").resolve()


__all__ = [
    "TensorboardDefaults",
    "DEFAULT_TENSORBOARD",
    "build_tensorboard_log_dir",
    "build_tensorboard_relative_path",
]
