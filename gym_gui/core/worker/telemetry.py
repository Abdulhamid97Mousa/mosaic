"""Standardized telemetry emission for MOSAIC workers.

All workers should use this module for emitting lifecycle events to ensure
consistent telemetry format across all worker types.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional, TextIO

from gym_gui.logging_config.helpers import log_constant
from gym_gui.logging_config.log_constants import LogConstant


class LifecycleEventType(str, Enum):
    """Standard lifecycle event types."""

    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    HEARTBEAT = "heartbeat"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_LOADED = "checkpoint_loaded"


@dataclass(frozen=True)
class LifecycleEvent:
    """Structured lifecycle telemetry event.

    All lifecycle events follow this schema for consistent parsing by the
    telemetry proxy.

    Attributes:
        event: Event type (see LifecycleEventType)
        run_id: Unique run identifier
        timestamp: Unix timestamp (seconds since epoch)
        payload: Event-specific data

    Example:
        event = LifecycleEvent(
            event="run_started",
            run_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            timestamp=1704067200.123,
            payload={
                "worker_type": "cleanrl",
                "algo": "ppo",
                "env_id": "CartPole-v1"
            }
        )
    """

    event: str
    run_id: str
    timestamp: float
    payload: Dict[str, Any]


class TelemetryEmitter:
    """Emit lifecycle events to stdout in JSONL format.

    This class provides a thread-safe interface for workers to emit lifecycle
    events that are captured by the telemetry proxy and streamed to the trainer
    daemon via gRPC.

    All events are emitted as newline-delimited JSON to stdout, which is
    captured by the telemetry proxy subprocess.

    Usage:
        emitter = TelemetryEmitter(run_id="01ARZ3ND...")

        emitter.run_started({"algo": "ppo", "env": "CartPole"})

        try:
            # Training loop
            emitter.heartbeat()
        except Exception as e:
            emitter.run_failed({"error": str(e)})
        else:
            emitter.run_completed({"episodes": 100, "reward": 5000})

    Args:
        run_id: Unique run identifier
        sink: Optional file-like object for testing (default: stdout)
        logger: Optional logger for structured log constants (default: None)
    """

    def __init__(
        self,
        run_id: str,
        *,
        sink: Optional[TextIO] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._run_id = run_id
        self._sink = sink or sys.stdout
        self._lock = threading.Lock()
        self._logger = logger

    def _emit(self, event: LifecycleEvent) -> None:
        """Emit a lifecycle event to stdout.

        Thread-safe emission with JSON serialization.

        Args:
            event: Lifecycle event to emit
        """
        serialized = json.dumps(
            asdict(event),
            separators=(",", ":"),
            sort_keys=True,
        )

        with self._lock:
            print(serialized, file=self._sink, flush=True)

    def run_started(
        self,
        payload: Dict[str, Any],
        *,
        constant: Optional[LogConstant] = None,
    ) -> None:
        """Emit run_started event.

        Should be called once at the beginning of worker execution.

        Args:
            payload: Worker-specific startup information
                Suggested fields:
                - worker_type: str (e.g., "cleanrl")
                - algo: str (algorithm name)
                - env_id: str (environment identifier)
                - seed: int (random seed)
                - config: dict (full configuration)
            constant: Optional log constant for structured logging

        Example:
            emitter.run_started({
                "worker_type": "cleanrl",
                "algo": "ppo",
                "env_id": "CartPole-v1",
                "seed": 42,
                "total_timesteps": 10000
            })
        """
        self._emit(
            LifecycleEvent(
                event=LifecycleEventType.RUN_STARTED.value,
                run_id=self._run_id,
                timestamp=time.time(),
                payload=payload,
            )
        )

        if constant and self._logger:
            log_constant(
                self._logger,
                constant,
                extra={"run_id": self._run_id, **payload},
            )

    def run_completed(
        self,
        payload: Dict[str, Any],
        *,
        constant: Optional[LogConstant] = None,
    ) -> None:
        """Emit run_completed event.

        Should be called when worker finishes successfully.

        Args:
            payload: Execution results
                Suggested fields:
                - episodes: int (number of episodes completed)
                - total_reward: float (cumulative reward)
                - duration_seconds: float (execution time)
                - final_metrics: dict (algorithm-specific metrics)
            constant: Optional log constant for structured logging

        Example:
            emitter.run_completed({
                "episodes": 100,
                "total_reward": 19500.0,
                "duration_seconds": 123.45,
                "final_metrics": {
                    "final_lr": 0.0003,
                    "total_timesteps": 10000
                }
            })
        """
        self._emit(
            LifecycleEvent(
                event=LifecycleEventType.RUN_COMPLETED.value,
                run_id=self._run_id,
                timestamp=time.time(),
                payload=payload,
            )
        )

        if constant and self._logger:
            log_constant(
                self._logger,
                constant,
                extra={"run_id": self._run_id, **payload},
            )

    def run_failed(
        self,
        payload: Dict[str, Any],
        *,
        constant: Optional[LogConstant] = None,
        exc_info: Optional[BaseException] = None,
    ) -> None:
        """Emit run_failed event.

        Should be called when worker encounters an unrecoverable error.

        Args:
            payload: Error information
                Suggested fields:
                - error: str (error message)
                - error_type: str (exception class name)
                - traceback: str (full traceback)
                - episode: int (episode where failure occurred)
                - step: int (step where failure occurred)
            constant: Optional log constant for structured logging
            exc_info: Optional exception info for traceback

        Example:
            emitter.run_failed({
                "error": "CUDA out of memory",
                "error_type": "RuntimeError",
                "traceback": traceback.format_exc(),
                "episode": 42,
                "step": 1337
            })
        """
        self._emit(
            LifecycleEvent(
                event=LifecycleEventType.RUN_FAILED.value,
                run_id=self._run_id,
                timestamp=time.time(),
                payload=payload,
            )
        )

        if constant and self._logger:
            log_constant(
                self._logger,
                constant,
                extra={"run_id": self._run_id, **payload},
                exc_info=exc_info,
            )

    def run_cancelled(
        self,
        payload: Dict[str, Any],
        *,
        constant: Optional[LogConstant] = None,
    ) -> None:
        """Emit run_cancelled event.

        Should be called when worker is cancelled by user or daemon.

        Args:
            payload: Cancellation information
                Suggested fields:
                - reason: str (cancellation reason)
                - completed_episodes: int
                - partial_results: dict (metrics at cancellation time)
            constant: Optional log constant for structured logging

        Example:
            emitter.run_cancelled({
                "reason": "user_request",
                "completed_episodes": 50,
                "partial_results": {"avg_reward": 150.0}
            })
        """
        self._emit(
            LifecycleEvent(
                event=LifecycleEventType.RUN_CANCELLED.value,
                run_id=self._run_id,
                timestamp=time.time(),
                payload=payload,
            )
        )

        if constant and self._logger:
            log_constant(
                self._logger,
                constant,
                extra={"run_id": self._run_id, **payload},
            )

    def heartbeat(
        self,
        payload: Optional[Dict[str, Any]] = None,
        *,
        constant: Optional[LogConstant] = None,
    ) -> None:
        """Emit heartbeat event.

        Should be called periodically (e.g., every 60 seconds) to indicate
        worker is still alive. The trainer daemon uses heartbeats to detect
        hung workers.

        Args:
            payload: Optional heartbeat data
                Suggested fields:
                - episode: int (current episode)
                - step: int (current step)
                - memory_mb: float (current memory usage)
                - cpu_percent: float (CPU usage percentage)
            constant: Optional log constant for structured logging

        Example:
            emitter.heartbeat({
                "episode": 42,
                "step": 1337,
                "memory_mb": 512.5,
                "cpu_percent": 85.2
            })
        """
        self._emit(
            LifecycleEvent(
                event=LifecycleEventType.HEARTBEAT.value,
                run_id=self._run_id,
                timestamp=time.time(),
                payload=payload or {},
            )
        )

        if constant and self._logger:
            log_constant(
                self._logger,
                constant,
                extra={"run_id": self._run_id, **(payload or {})},
            )

    def checkpoint_saved(
        self,
        payload: Dict[str, Any],
        *,
        constant: Optional[LogConstant] = None,
    ) -> None:
        """Emit checkpoint_saved event.

        Should be called when worker saves a checkpoint to disk.

        Args:
            payload: Checkpoint information
                Suggested fields:
                - checkpoint_path: str (relative path to checkpoint)
                - episode: int (episode at checkpoint)
                - step: int (global step at checkpoint)
                - metrics: dict (performance metrics at checkpoint)
            constant: Optional log constant for structured logging

        Example:
            emitter.checkpoint_saved({
                "checkpoint_path": "checkpoints/checkpoint_1000.pt",
                "episode": 100,
                "step": 10000,
                "metrics": {"avg_reward": 195.0}
            })
        """
        self._emit(
            LifecycleEvent(
                event=LifecycleEventType.CHECKPOINT_SAVED.value,
                run_id=self._run_id,
                timestamp=time.time(),
                payload=payload,
            )
        )

        if constant and self._logger:
            log_constant(
                self._logger,
                constant,
                extra={"run_id": self._run_id, **payload},
            )

    def checkpoint_loaded(
        self,
        payload: Dict[str, Any],
        *,
        constant: Optional[LogConstant] = None,
    ) -> None:
        """Emit checkpoint_loaded event.

        Should be called when worker loads a checkpoint from disk.

        Args:
            payload: Checkpoint information
                Suggested fields:
                - checkpoint_path: str (relative path to checkpoint)
                - episode: int (episode from checkpoint)
                - step: int (global step from checkpoint)
            constant: Optional log constant for structured logging

        Example:
            emitter.checkpoint_loaded({
                "checkpoint_path": "checkpoints/checkpoint_1000.pt",
                "episode": 100,
                "step": 10000
            })
        """
        self._emit(
            LifecycleEvent(
                event=LifecycleEventType.CHECKPOINT_LOADED.value,
                run_id=self._run_id,
                timestamp=time.time(),
                payload=payload,
            )
        )

        if constant and self._logger:
            log_constant(
                self._logger,
                constant,
                extra={"run_id": self._run_id, **payload},
            )
