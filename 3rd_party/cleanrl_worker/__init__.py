"""Compatibility facade for the CleanRL worker package.

All implementation now lives under :mod:`cleanrl_worker.MOSAIC_CLEANRL_WORKER`.
"""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
repo_str = str(repo_root)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

from .MOSAIC_CLEANRL_WORKER import (  # noqa: F401  # isort: skip
    CleanRLWorkerRuntime,
    RuntimeSummary,
    WorkerConfig,
    load_worker_config,
)


__all__ = [
    "CleanRLWorkerRuntime",
    "RuntimeSummary",
    "WorkerConfig",
    "load_worker_config",
]
