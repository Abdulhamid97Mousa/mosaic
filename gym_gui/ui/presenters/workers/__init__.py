"""Worker presenters package for UI orchestration.

This package provides presenter implementations for each supported worker type.
Presenters handle:
1. Building training configurations from form data
2. Creating worker-specific UI tabs
3. Extracting metadata for API contracts

Included presenters:
- CleanRlWorkerPresenter: Placeholder for analytics-first CleanRL worker
- PettingZooWorkerPresenter: Multi-agent environments (PettingZoo)
- RayWorkerPresenter: Ray RLlib distributed training
- XuanCeWorkerPresenter: XuanCe 46+ algorithm RL library

The registry is auto-populated at module load to support service discovery.
"""

from .registry import WorkerPresenter, WorkerPresenterRegistry
from .cleanrl_worker_presenter import CleanRlWorkerPresenter
from .pettingzoo_worker_presenter import PettingZooWorkerPresenter
from .ray_worker_presenter import RayWorkerPresenter
from .xuance_worker_presenter import XuanCeWorkerPresenter


# Create and auto-register default presenters
_registry = WorkerPresenterRegistry()
_registry.register("cleanrl_worker", CleanRlWorkerPresenter())
_registry.register("pettingzoo_worker", PettingZooWorkerPresenter())
_registry.register("ray_worker", RayWorkerPresenter())
_registry.register("xuance_worker", XuanCeWorkerPresenter())


def get_worker_presenter_registry() -> WorkerPresenterRegistry:
    """Get the global worker presenter registry.

    Returns:
        WorkerPresenterRegistry: Singleton registry of available workers
    """
    return _registry


__all__ = [
    "WorkerPresenter",
    "WorkerPresenterRegistry",
    "CleanRlWorkerPresenter",
    "PettingZooWorkerPresenter",
    "RayWorkerPresenter",
    "XuanCeWorkerPresenter",
    "get_worker_presenter_registry",
]
