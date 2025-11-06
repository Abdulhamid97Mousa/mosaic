"""Compatibility facade for the CleanRL worker package.

All implementation now lives under :mod:`cleanrl_worker.MOSAIC_CLEANRL_WORKER`.
"""

from __future__ import annotations

from .MOSAIC_CLEANRL_WORKER import (  # noqa: F401
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
