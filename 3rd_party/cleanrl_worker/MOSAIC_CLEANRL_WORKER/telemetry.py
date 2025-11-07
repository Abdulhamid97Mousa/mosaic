"""Lifecycle telemetry helpers for the CleanRL worker."""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class LifecycleEvent:
    """Structured lifecycle telemetry event."""

    event: str
    run_id: str
    timestamp: float
    payload: Dict[str, Any]


class LifecycleEmitter:
    """Emit lifecycle events to stdout or an optional sink (for tests)."""

    def __init__(self, *, sink: Optional[Iterable[LifecycleEvent]] = None) -> None:
        self._sink = sink
        self._lock = threading.Lock()

    def _record(self, event: LifecycleEvent) -> None:
        serialized = json.dumps(asdict(event), separators=(",", ":"), sort_keys=True)
        with self._lock:
            print(serialized, file=sys.stdout, flush=True)
            sink = self._sink
            if sink is not None and hasattr(sink, "append"):
                getattr(sink, "append")(event)  # type: ignore[attr-defined]

    def run_started(self, run_id: str, payload: Dict[str, Any]) -> None:
        self._record(
            LifecycleEvent(event="run_started", run_id=run_id, timestamp=time.time(), payload=payload)
        )

    def heartbeat(self, run_id: str, payload: Dict[str, Any]) -> None:
        self._record(
            LifecycleEvent(event="heartbeat", run_id=run_id, timestamp=time.time(), payload=payload)
        )

    def run_completed(self, run_id: str, payload: Dict[str, Any]) -> None:
        self._record(
            LifecycleEvent(event="run_completed", run_id=run_id, timestamp=time.time(), payload=payload)
        )

    def run_failed(self, run_id: str, payload: Dict[str, Any]) -> None:
        self._record(
            LifecycleEvent(event="run_failed", run_id=run_id, timestamp=time.time(), payload=payload)
        )
