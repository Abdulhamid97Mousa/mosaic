"""JSONL telemetry helpers for the refactored SPADE-BDI worker."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, IO


def _utc_timestamp_ns() -> int:
    """Return current time as Unix nanoseconds (for protobuf Timestamp conversion)."""
    return time.time_ns()


class TelemetryEmitter:
    """Handle emission of newline-delimited JSON telemetry events."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream: IO[str] = stream or sys.stdout

    def emit(self, event_type: str, **fields: Any) -> None:
        ts_ns = _utc_timestamp_ns()
        # Emit both ts (ISO format) and ts_unix_ns for compatibility
        payload: Dict[str, Any] = {
            "type": event_type,
            "ts": datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat(),
            "ts_unix_ns": ts_ns,
            **fields,
        }
        json.dump(payload, self._stream, separators=(",", ":"))
        self._stream.write("\n")
        self._stream.flush()

    # Convenience helpers -------------------------------------------------
    def run_started(self, run_id: str, config: Dict[str, Any]) -> None:
        self.emit("run_started", run_id=run_id, config=config)

    def run_completed(self, run_id: str, status: str, **fields: Any) -> None:
        self.emit("run_completed", run_id=run_id, status=status, **fields)

    def step(self, run_id: str, episode: int, step_index: int, **fields: Any) -> None:
        # CRITICAL: episode parameter is the display value (episode_index + seed)
        # Extract episode_index from fields if present, otherwise derive from episode_seed
        episode_index = fields.get("episode_index")
        if episode_index is None and "episode_seed" in fields:
            episode_index = fields["episode_seed"]  # episode_seed IS episode_index
        
        # ts_unix_ns is automatically added by emit()
        # Preserve backward compatibility by exposing both step_index and step
        if "step" not in fields:
            fields["step"] = int(step_index)

        self.emit(
            "step",
            run_id=run_id,
            episode=int(episode),  # Display value (episode_index + seed) - for backward compatibility
            episode_index=int(episode_index) if episode_index is not None else 0,  # 0-based index
            step_index=int(step_index),  # Use step_index (not "step")
            **fields,
        )

    def episode(self, run_id: str, episode: int, **fields: Any) -> None:
        # CRITICAL: episode parameter is the display value (episode_index + seed)
        # Extract episode_index from metadata if present
        episode_index = None
        if "metadata" in fields:
            metadata = fields.get("metadata")
            if isinstance(metadata, dict):
                episode_index = metadata.get("episode_index")
        
        # Convert metadata dict to metadata_json string if present
        if "metadata" in fields:
            metadata = fields.pop("metadata")
            fields["metadata_json"] = json.dumps(metadata) if isinstance(metadata, dict) else str(metadata)
        
        # Emit episode with both episode (display) and episode_index (0-based)
        self.emit(
            "episode",
            run_id=run_id,
            episode=int(episode),  # Display value for backward compatibility
            episode_index=int(episode_index) if episode_index is not None else 0,  # 0-based index
            **fields,
        )

    def artifact(self, run_id: str, kind: str, path: str, **fields: Any) -> None:
        self.emit("artifact", run_id=run_id, kind=kind, path=path, **fields)
