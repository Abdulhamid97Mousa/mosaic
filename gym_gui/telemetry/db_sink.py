"""Background writer thread that persists telemetry events to SQLite.

This module implements the "durable path" of the dual-path architecture:
- Subscribes to STEP_APPENDED and EPISODE_FINALIZED events from RunBus
- Batches events for efficient SQLite writes
- Uses WAL mode for concurrent reads/writes
- Periodic checkpoint operations to manage WAL file size
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from typing import Optional

from gym_gui.core.data_model.telemetry_core import StepRecord, EpisodeRollup
from gym_gui.telemetry.sqlite_store import TelemetrySQLiteStore
from gym_gui.telemetry.events import Topic, TelemetryEvent
from gym_gui.telemetry.run_bus import RunBus
from gym_gui.constants import (
    DB_SINK_BATCH_SIZE,
    DB_SINK_CHECKPOINT_INTERVAL,
    DB_SINK_WRITER_QUEUE_SIZE,
)
from gym_gui.constants.constants_telemetry import (
    TELEMETRY_KEY_SPACE_SIGNATURE,
    TELEMETRY_KEY_TIME_STEP,
    TELEMETRY_KEY_VECTOR_METADATA,
)
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_SERVICE_DB_SINK_INITIALIZED,
    LOG_SERVICE_DB_SINK_STARTED,
    LOG_SERVICE_DB_SINK_ALREADY_RUNNING,
    LOG_SERVICE_DB_SINK_STOPPED,
    LOG_SERVICE_DB_SINK_STOP_TIMEOUT,
    LOG_SERVICE_DB_SINK_FATAL,
    LOG_SERVICE_DB_SINK_LOOP_EXITED,
    LOG_SERVICE_DB_SINK_QUEUE_DEPTH,
    LOG_SERVICE_DB_SINK_QUEUE_PRESSURE,
    LOG_SERVICE_DB_SINK_FLUSH_STATS,
    LOG_SERVICE_DB_SINK_FLUSH_LATENCY,
)


_LOGGER = logging.getLogger(__name__)


class TelemetryDBSink(LogConstantMixin):
    """Background writer that persists telemetry events to SQLite.
    
    This class subscribes to the RunBus and batches events for efficient
    persistence to SQLite. It runs in a background thread to avoid blocking
    the UI or other components.
    """

    def __init__(
        self,
        store: TelemetrySQLiteStore,
        bus: RunBus,
        *,
        batch_size: int = DB_SINK_BATCH_SIZE,
        checkpoint_interval: int = DB_SINK_CHECKPOINT_INTERVAL,
        writer_queue_size: int = DB_SINK_WRITER_QUEUE_SIZE,
    ) -> None:
        """Initialize the DB sink.

        Args:
            store: TelemetrySQLiteStore instance for persistence
            bus: RunBus instance to subscribe to
            batch_size: Number of events to batch before flushing
            checkpoint_interval: Number of writes before WAL checkpoint
            writer_queue_size: Queue size for writer subscriber (larger than UI)

        Note: ALL events are written to the database (no sampling/dropping).
        The database uses efficient batching (batch_size=64) and WAL mode.
        UI rendering throttling is handled separately in LiveTelemetryTab.
        """
        self._logger = _LOGGER
        self._store = store
        self._bus = bus
        self._batch_size = batch_size
        self._checkpoint_interval = checkpoint_interval
        self._writer_queue_size = writer_queue_size

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._step_batch: list[StepRecord] = []
        self._episode_batch: list[EpisodeRollup] = []
        self._write_count = 0
        self._queue_log_interval = 2.0  # seconds between queue depth snapshots
        self._last_queue_log = 0.0
        self._queue_pressure_ratio = 0.8  # warn when queue >= 80% full
        self._flush_latency_warn_s = 0.25  # warn when flush slower than 250ms

        self.log_constant(
            LOG_SERVICE_DB_SINK_INITIALIZED,
            extra={
                "batch_size": batch_size,
                "checkpoint_interval": checkpoint_interval,
                "writer_queue_size": writer_queue_size,
            },
        )

    def start(self) -> None:
        """Start the background writer thread."""
        if self._thread is not None:
            self.log_constant(LOG_SERVICE_DB_SINK_ALREADY_RUNNING)
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="telemetry-db-sink",
            daemon=True,
        )
        self._thread.start()
        self.log_constant(LOG_SERVICE_DB_SINK_STARTED)

    def stop(self) -> None:
        """Stop the background writer thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self.log_constant(LOG_SERVICE_DB_SINK_STOP_TIMEOUT)
        else:
            self.log_constant(LOG_SERVICE_DB_SINK_STOPPED)
        self._thread = None

    def _run(self) -> None:
        """Main loop for the background writer thread."""
        try:
            # Subscribe to events with larger queue size (writer can handle backlog)
            step_queue = self._bus.subscribe_with_size(
                Topic.STEP_APPENDED, "db-sink-steps", self._writer_queue_size
            )
            episode_queue = self._bus.subscribe_with_size(
                Topic.EPISODE_FINALIZED, "db-sink-episodes", self._writer_queue_size
            )

            _LOGGER.debug(
                "DB sink subscribed to RunBus topics",
                extra={"writer_queue_size": self._writer_queue_size},
            )
            
            while not self._stop_event.is_set():
                try:
                    # Process step events (non-blocking)
                    self._process_step_queue(step_queue)

                    # Process episode events (non-blocking)
                    self._process_episode_queue(episode_queue)

                    # Flush batches if needed
                    self._flush_batches()

                    # Periodically emit queue depth metrics for observability
                    self._maybe_log_queue_depth(step_queue, episode_queue)

                    # Small sleep to avoid busy-waiting
                    self._stop_event.wait(timeout=0.1)
                except Exception as e:
                    self.log_constant(
                        LOG_SERVICE_DB_SINK_FATAL,
                        exc_info=e,
                        extra={"error": str(e), "context": "loop"},
                    )
        except Exception as e:
            self.log_constant(
                LOG_SERVICE_DB_SINK_FATAL,
                exc_info=e,
                extra={"error": str(e)},
            )
        finally:
            # Flush any remaining events
            self._flush_batches(force=True)
            self.log_constant(LOG_SERVICE_DB_SINK_LOOP_EXITED)

    def _process_step_queue(self, q: queue.Queue) -> None:
        """Process all available step events from the queue."""
        while True:
            try:
                evt = q.get_nowait()
                if not isinstance(evt, TelemetryEvent):
                    continue

                # Convert event payload to StepRecord
                payload = evt.payload

                # Parse timestamp if available
                timestamp = None
                if evt.ts_iso:
                    try:
                        timestamp = datetime.fromisoformat(evt.ts_iso)
                    except (ValueError, TypeError):
                        pass

                step = StepRecord(
                    episode_id=payload.get("episode_id", ""),
                    step_index=payload.get("step_index", 0),
                    action=payload.get("action"),
                    observation=payload.get("observation"),
                    reward=payload.get("reward", 0.0),
                    terminated=payload.get("terminated", False),
                    truncated=payload.get("truncated", False),
                    info=payload.get("info", {}),
                    render_payload=payload.get("render_payload"),
                    agent_id=evt.agent_id,
                    render_hint=payload.get("render_hint"),
                    frame_ref=payload.get("frame_ref"),
                    payload_version=payload.get("payload_version", 0),
                    run_id=evt.run_id,
                    worker_id=payload.get("worker_id"),
                    time_step=payload.get(TELEMETRY_KEY_TIME_STEP),
                    space_signature=payload.get(TELEMETRY_KEY_SPACE_SIGNATURE),
                    vector_metadata=payload.get(TELEMETRY_KEY_VECTOR_METADATA),
                )
                if timestamp is not None:
                    step.timestamp = timestamp

                # Log seq_id for gap detection
                _LOGGER.debug(
                    "Step event received from RunBus",
                    extra={
                        "run_id": evt.run_id,
                        "agent_id": evt.agent_id,
                        "seq_id": evt.seq_id,
                        "episode_id": payload.get("episode_id", ""),
                        "step_index": payload.get("step_index", 0),
                    },
                )

                self._step_batch.append(step)
            except queue.Empty:
                break

    def _process_episode_queue(self, q: queue.Queue) -> None:
        """Process all available episode events from the queue."""
        while True:
            try:
                evt = q.get_nowait()
                if not isinstance(evt, TelemetryEvent):
                    continue

                # Convert event payload to EpisodeRollup
                payload = evt.payload

                # Parse timestamp if available
                timestamp = None
                if evt.ts_iso:
                    try:
                        timestamp = datetime.fromisoformat(evt.ts_iso)
                    except (ValueError, TypeError):
                        pass

                # CRITICAL FIX: Parse metadata_json if metadata is not present
                # The gRPC proto has metadata_json (string), not metadata (dict)
                metadata = payload.get("metadata", {})
                if not metadata and "metadata_json" in payload:
                    metadata_json = payload.get("metadata_json", "")
                    if isinstance(metadata_json, str) and metadata_json:
                        try:
                            import json
                            metadata = json.loads(metadata_json)
                            _LOGGER.debug(
                                "Parsed metadata_json from payload",
                                extra={
                                    "run_id": evt.run_id,
                                    "agent_id": evt.agent_id,
                                    "episode_id": payload.get("episode_id", ""),
                                    "metadata_keys": list(metadata.keys()) if isinstance(metadata, dict) else "not_dict",
                                }
                            )
                        except (json.JSONDecodeError, TypeError) as e:
                            _LOGGER.warning(
                                "Failed to parse metadata_json",
                                extra={
                                    "run_id": evt.run_id,
                                    "agent_id": evt.agent_id,
                                    "error": str(e),
                                }
                            )
                            metadata = {}

                episode = EpisodeRollup(
                    episode_id=payload.get("episode_id", ""),
                    total_reward=payload.get("total_reward", 0.0),
                    steps=payload.get("steps", 0),
                    terminated=payload.get("terminated", False),
                    truncated=payload.get("truncated", False),
                    metadata=metadata,
                    agent_id=evt.agent_id,
                    run_id=evt.run_id,
                )
                if timestamp is not None:
                    episode.timestamp = timestamp

                # Log seq_id for gap detection
                _LOGGER.debug(
                    "Episode event received from RunBus",
                    extra={
                        "run_id": evt.run_id,
                        "agent_id": evt.agent_id,
                        "seq_id": evt.seq_id,
                        "episode_id": payload.get("episode_id", ""),
                        "total_reward": payload.get("total_reward", 0.0),
                        "steps": payload.get("steps", 0),
                    },
                )

                self._episode_batch.append(episode)
            except queue.Empty:
                break

    def _flush_batches(self, force: bool = False) -> None:
        """Flush batches to SQLite if they exceed batch_size or force=True."""
        # Flush steps
        if force or len(self._step_batch) >= self._batch_size:
            if self._step_batch:
                try:
                    batch_count = len(self._step_batch)
                    start = time.perf_counter()
                    for step in self._step_batch:
                        self._store.record_step(step)
                    duration = time.perf_counter() - start
                    self._write_count += batch_count
                    self._step_batch.clear()
                    self._log_flush_stats("steps", batch_count, duration, force)
                except Exception as e:
                    _LOGGER.exception("Failed to flush step batch", extra={"error": str(e)})
        
        # Flush episodes
        if force or len(self._episode_batch) >= self._batch_size:
            if self._episode_batch:
                try:
                    batch_count = len(self._episode_batch)
                    start = time.perf_counter()
                    for episode in self._episode_batch:
                        self._store.record_episode(episode)
                    duration = time.perf_counter() - start
                    self._write_count += batch_count
                    self._episode_batch.clear()
                    self._log_flush_stats("episodes", batch_count, duration, force)
                except Exception as e:
                    _LOGGER.exception("Failed to flush episode batch", extra={"error": str(e)})
        
        # Periodic WAL checkpoint
        if self._write_count >= self._checkpoint_interval:
            try:
                self._store.checkpoint_wal()
                _LOGGER.debug(
                    "WAL checkpoint completed",
                    extra={"writes_since_checkpoint": self._write_count},
                )
                self._write_count = 0
            except Exception as e:
                _LOGGER.warning("WAL checkpoint failed", extra={"error": str(e)})

    def _maybe_log_queue_depth(self, step_queue: queue.Queue, episode_queue: queue.Queue) -> None:
        """Emit periodic queue depth metrics to aid diagnosing backpressure."""

        now = time.monotonic()
        if now - self._last_queue_log < self._queue_log_interval:
            return

        step_depth = step_queue.qsize() if hasattr(step_queue, "qsize") else 0
        episode_depth = episode_queue.qsize() if hasattr(episode_queue, "qsize") else 0
        if step_depth <= 0 and episode_depth <= 0:
            return

        self._last_queue_log = now
        extra = {
            "step_depth": step_depth,
            "episode_depth": episode_depth,
            "queue_capacity": self._writer_queue_size,
        }
        self.log_constant(LOG_SERVICE_DB_SINK_QUEUE_DEPTH, extra=extra)

        capacity = max(1, self._writer_queue_size)
        high_water = max(step_depth, episode_depth)
        if high_water >= capacity * self._queue_pressure_ratio:
            extra = {
                **extra,
                "high_water_pct": float(high_water) / float(capacity),
            }
            self.log_constant(LOG_SERVICE_DB_SINK_QUEUE_PRESSURE, extra=extra)

    def _log_flush_stats(self, kind: str, count: int, duration_s: float, force: bool) -> None:
        """Log flush duration/size for observability and detect slow writes."""

        extra = {
            "batch_kind": kind,
            "count": count,
            "duration_ms": round(duration_s * 1000.0, 3),
            "force": force,
        }
        self.log_constant(LOG_SERVICE_DB_SINK_FLUSH_STATS, extra=extra)
        if duration_s >= self._flush_latency_warn_s:
            self.log_constant(LOG_SERVICE_DB_SINK_FLUSH_LATENCY, extra=extra)


__all__ = ["TelemetryDBSink"]
