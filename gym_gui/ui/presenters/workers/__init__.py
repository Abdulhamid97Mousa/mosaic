"""Worker presenters package for UI orchestration.

This package provides presenter implementations for each supported worker type.
Presenters handle:
1. Building training configurations from form data
2. Creating worker-specific UI tabs
3. Extracting metadata for API contracts

Included presenters:
- SpadeBdiWorkerPresenter: Orchestration for SPADE-BDI RL agents
- CleanRlWorkerPresenter: Placeholder for analytics-first CleanRL worker
- PettingZooWorkerPresenter: Multi-agent environments (PettingZoo)

The registry is auto-populated at module load to support service discovery.
"""

from .registry import WorkerPresenter, WorkerPresenterRegistry
from .spade_bdi_worker_presenter import SpadeBdiWorkerPresenter
from .cleanrl_worker_presenter import CleanRlWorkerPresenter
from .pettingzoo_worker_presenter import PettingZooWorkerPresenter
from .ray_worker_presenter import RayWorkerPresenter


# Create and auto-register default presenters
_registry = WorkerPresenterRegistry()
_registry.register("spade_bdi_worker", SpadeBdiWorkerPresenter())
_registry.register("cleanrl_worker", CleanRlWorkerPresenter())
_registry.register("pettingzoo_worker", PettingZooWorkerPresenter())
_registry.register("ray_worker", RayWorkerPresenter())


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
    "CleanRlWorkerPresenter",
    "PettingZooWorkerPresenter",
    "RayWorkerPresenter",
    "get_worker_presenter_registry",
]
