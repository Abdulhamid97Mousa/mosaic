"""JSONL telemetry helpers for the refactored SPADE-BDI worker."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, IO


def _utc_timestamp() -> str:
    """Return an ISO-8601 timestamp (UTC, millisecond precision)."""

    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="milliseconds")


class TelemetryEmitter:
    """Handle emission of newline-delimited JSON telemetry events."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream: IO[str] = stream or sys.stdout

    def emit(self, event_type: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {"type": event_type, "ts": _utc_timestamp(), **fields}
        json.dump(payload, self._stream, separators=(",", ":"))
        self._stream.write("\n")
        self._stream.flush()

    # Convenience helpers -------------------------------------------------
    def run_started(self, run_id: str, config: Dict[str, Any]) -> None:
        self.emit("run_started", run_id=run_id, config=config)

    def run_completed(self, run_id: str, status: str, **fields: Any) -> None:
        self.emit("run_completed", run_id=run_id, status=status, **fields)

    def step(self, run_id: str, episode: int, step_index: int, **fields: Any) -> None:
        # Add both 'ts' and 'timestamp' for compatibility
        ts = _utc_timestamp()
        self.emit(
            "step",
            run_id=run_id,
            episode=int(episode),
            step=int(step_index),
            timestamp=ts,  # Add timestamp field explicitly
            **fields,
        )

    def episode(self, run_id: str, episode: int, **fields: Any) -> None:
        self.emit("episode", run_id=run_id, episode=int(episode), **fields)

    def artifact(self, run_id: str, kind: str, path: str, **fields: Any) -> None:
        self.emit("artifact", run_id=run_id, kind=kind, path=path, **fields)
