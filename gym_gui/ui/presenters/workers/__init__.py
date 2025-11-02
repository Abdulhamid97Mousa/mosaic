"""Worker presenters package for UI orchestration.

This package provides presenter implementations for each supported worker type.
Presenters handle:
1. Building training configurations from form data
2. Creating worker-specific UI tabs
3. Extracting metadata for API contracts

Included presenters:
- SpadeBdiWorkerPresenter: Orchestration for SPADE-BDI RL agents
- (Future) HuggingFaceWorkerPresenter: Orchestration for HuggingFace models
- (Future) TensorFlowWorkerPresenter: Orchestration for TensorFlow models

The registry is auto-populated at module load to support service discovery.
"""

from .registry import WorkerPresenter, WorkerPresenterRegistry
from .spade_bdi_worker_presenter import SpadeBdiWorkerPresenter


# Create and auto-register default presenters
_registry = WorkerPresenterRegistry()
_registry.register("spade_bdi_worker", SpadeBdiWorkerPresenter())


def get_worker_presenter_registry() -> WorkerPresenterRegistry:
    """Get the global worker presenter registry.

    Returns:
        WorkerPresenterRegistry: Singleton registry of available workers
    """
    return _registry


__all__ = [
    "WorkerPresenter",
    "WorkerPresenterRegistry",
    "SpadeBdiWorkerPresenter",
    "get_worker_presenter_registry",
]
