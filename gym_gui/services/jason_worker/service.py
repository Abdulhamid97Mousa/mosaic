from __future__ import annotations

"""Lightweight service tracking Jason worker connectivity and percepts."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Iterable
import logging

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_SERVICE_JASON_WORKER_EVENT,
    LOG_SERVICE_JASON_WORKER_WARNING,
)


@dataclass(frozen=True, slots=True)
class WorkerPercept:
    """Structured representation of a percept emitted by the Jason worker."""

    name: str
    payload: dict[str, Any]
    timestamp: datetime


class JasonWorkerService(LogConstantMixin):
    """Track Jason worker activity and provide recent percept history."""

    def __init__(
        self,
        *,
        buffer_limit: int = 256,
        heartbeat_timeout_s: float = 5.0,
    ) -> None:
        if buffer_limit <= 0:
            raise ValueError("buffer_limit must be positive")
        if heartbeat_timeout_s <= 0:
            raise ValueError("heartbeat_timeout_s must be positive")

        self._logger = logging.getLogger("gym_gui.services.jason_worker")
        self._percepts: Deque[WorkerPercept] = deque(maxlen=int(buffer_limit))
        self._last_seen: datetime | None = None
        self._heartbeat_timeout_s = float(heartbeat_timeout_s)

    # ------------------------------------------------------------------
    # Recording and telemetry helpers
    # ------------------------------------------------------------------
    def record_percept(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a percept received from the Jason worker."""

        ts = timestamp or datetime.now(timezone.utc)
        entry = WorkerPercept(name=name, payload=dict(payload or {}), timestamp=ts)
        self._percepts.append(entry)
        self._last_seen = ts
        self.log_constant(
            LOG_SERVICE_JASON_WORKER_EVENT,
            message="percept_recorded",
            extra={"name": name, "buffer_size": len(self._percepts)},
        )

    def mark_disconnected(self) -> None:
        """Mark the worker as disconnected until a new percept arrives."""

        if self._last_seen is not None:
            self.log_constant(
                LOG_SERVICE_JASON_WORKER_WARNING,
                message="worker_disconnected",
            )
        self._last_seen = None

    def recent_percepts(self) -> list[WorkerPercept]:
        """Return a copy of the recent percept buffer."""

        return list(self._percepts)

    def iter_percepts(self) -> Iterable[WorkerPercept]:
        """Iterate over buffered percepts in chronological order."""

        return tuple(self._percepts)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def is_connected(self, *, now: datetime | None = None) -> bool:
        """Return whether the worker is considered connected."""

        if self._last_seen is None:
            return False
        current = now or datetime.now(timezone.utc)
        delta = current - self._last_seen
        return delta.total_seconds() <= self._heartbeat_timeout_s

    def snapshot(self) -> dict[str, Any]:
        """Return a lightweight snapshot for UI consumers."""

        last = self._percepts[-1] if self._percepts else None
        return {
            "connected": self.is_connected(),
            "percept_count": len(self._percepts),
            "last_percept": last.name if last else None,
            "last_timestamp": last.timestamp.isoformat() if last else None,
            "last_payload": last.payload if last else None,
        }

    def clear(self) -> None:
        """Clear buffered percepts without affecting connection state."""

        self._percepts.clear()


__all__ = ["JasonWorkerService", "WorkerPercept"]
