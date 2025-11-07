"""Compatibility wrapper for CleanRL worker telemetry."""

from .MOSAIC_CLEANRL_WORKER.telemetry import (  # noqa: F401
    LifecycleEmitter,
    LifecycleEvent,
)

__all__ = ["LifecycleEmitter", "LifecycleEvent"]
