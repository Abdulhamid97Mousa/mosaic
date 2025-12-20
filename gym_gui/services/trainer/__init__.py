from __future__ import annotations

"""Trainer service primitives shared between the daemon and clients."""

from .config import TrainRunConfig, TrainerRunMetadata, validate_train_run_config
from .client import TrainerClient, TrainerClientConfig, TrainerClientConnectionError
from .client_runner import TrainerClientRunner, TrainerWatchStopped, TrainerWatchSubscription
from .registry import RunRecord, RunRegistry, RunStatus
from .gpu import GPUAllocator, GPUReservation, GPUReservationError
from .dispatcher import TrainerDispatcher, WorkerHandle
from .signals import TrainerSignals, get_trainer_signals
from .run_manager import TrainingRunManager

__all__ = [
    "TrainRunConfig",
    "TrainerRunMetadata",
    "validate_train_run_config",
    "RunRegistry",
    "RunRecord",
    "RunStatus",
    "TrainerClient",
    "TrainerClientConfig",
    "TrainerClientConnectionError",
    "TrainerClientRunner",
    "TrainerWatchStopped",
    "TrainerWatchSubscription",
    "GPUAllocator",
    "GPUReservation",
    "GPUReservationError",
    "TrainerDispatcher",
    "WorkerHandle",
    "TrainerSignals",
    "get_trainer_signals",
    "TrainingRunManager",
]
