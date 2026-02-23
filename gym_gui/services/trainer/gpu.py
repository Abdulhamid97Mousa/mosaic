from __future__ import annotations

"""GPU allocation helpers for trainer workloads."""

from dataclasses import dataclass
from typing import Iterable

from .registry import GPUReservationError, RunRegistry


@dataclass(slots=True)
class GPUReservation:
    run_id: str
    slots: list[int]


class GPUAllocator:
    """Coordinates GPU slot reservations using the shared run registry."""

    def __init__(self, registry: RunRegistry) -> None:
        self._registry = registry

    def reserve(self, run_id: str, requested: int, mandatory: bool) -> GPUReservation:
        slots = self._registry.claim_gpu_slot(run_id, requested, mandatory)
        return GPUReservation(run_id=run_id, slots=slots)

    def release(self, reservation: GPUReservation) -> None:
        if not reservation.slots:
            return
        self._registry.release_gpu_slots(reservation.run_id)

    def release_many(self, run_ids: Iterable[str]) -> None:
        for run_id in run_ids:
            self._registry.release_gpu_slots(run_id)


__all__ = ["GPUAllocator", "GPUReservation", "GPUReservationError"]
