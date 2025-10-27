# /home/hamid/Desktop/Projects/GUI_BDI_RL/gym_gui/services/telemetry.py

from __future__ import annotations

"""Telemetry collection and routing service."""

import asyncio
import logging
from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional, TYPE_CHECKING

from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.storage.session import EpisodeRecord
from gym_gui.logging_config.log_constants import (
    LOG_SERVICE_TELEMETRY_ASYNC_ERROR,
    LOG_SERVICE_TELEMETRY_STEP_REJECTED,
)
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.telemetry.constants import TELEMETRY_SERVICE_HISTORY_LIMIT

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from gym_gui.services.storage import StorageRecorderService
    from gym_gui.services.validation import ValidationService
    from gym_gui.telemetry import TelemetrySQLiteStore

_LOGGER = logging.getLogger(__name__)


class TelemetryService(LogConstantMixin):
    """Aggregates telemetry events and forwards them to storage."""

    def __init__(
        self,
        *,
        history_limit: int = TELEMETRY_SERVICE_HISTORY_LIMIT,
        validation_service: Optional["ValidationService"] = None,
    ) -> None:
        self._logger = _LOGGER
        self._history_limit = max(1, history_limit)
        self._step_history: Deque[StepRecord] = deque(maxlen=self._history_limit)
        self._episode_history: Deque[EpisodeRollup] = deque(
            maxlen=max(1, self._history_limit // 2)
        )
        self._storage: "StorageRecorderService | None" = None
        self._store: "TelemetrySQLiteStore | None" = None
        self._validation_service: "ValidationService | None" = validation_service

    def attach_storage(self, storage: "StorageRecorderService") -> None:
        self._storage = storage

    def attach_store(self, store: "TelemetrySQLiteStore") -> None:
        self._store = store

    def attach_validation_service(self, validation_service: "ValidationService") -> None:
        """Attach a validation service for data validation."""
        self._validation_service = validation_service

    # ------------------------------------------------------------------
    def record_step(self, record: StepRecord) -> None:
        # Log incoming step with all 10 fields (DEBUG level - per-step is noisy)
        _LOGGER.debug(f"[TELEMETRY] Incoming StepRecord: "
                    f"episode_id={record.episode_id} "
                    f"step_index={record.step_index} "
                    f"agent_id={record.agent_id} "
                    f"reward={record.reward} "
                    f"action={record.action} "
                    f"terminated={record.terminated} "
                    f"truncated={record.truncated}")
        
        # Log field types to spot mismatches (DEBUG level)
        _LOGGER.debug(f"[TELEMETRY TYPES] episode_id={type(record.episode_id).__name__} "
                    f"step_index={type(record.step_index).__name__} "
                    f"reward={type(record.reward).__name__} "
                    f"action={type(record.action).__name__} "
                    f"terminated={type(record.terminated).__name__} "
                    f"truncated={type(record.truncated).__name__}")
        
        # Validate step data if validation service is attached
        if self._validation_service:
            try:
                episode_num = int(record.episode_id.split("_")[-1]) if "_" in record.episode_id else 0
            except (ValueError, IndexError):
                episode_num = 0
            is_valid = self._validation_service.validate_step_data(
                episode=episode_num,
                step=record.step_index,
                action=record.action or 0,
                reward=record.reward,
                state=0,  # Not available in StepRecord
                next_state=0,  # Not available in StepRecord
            )
            if not is_valid:
                self.log_constant(
                    LOG_SERVICE_TELEMETRY_STEP_REJECTED,
                    extra={
                        "episode_id": record.episode_id,
                        "step_index": record.step_index,
                    },
                )

        self._step_history.append(record)
        if self._storage:
            storage_record = EpisodeRecord(
                episode_id=record.episode_id,
                step_index=record.step_index,
                observation=record.observation,
                reward=record.reward,
                terminated=record.terminated,
                truncated=record.truncated,
                info=dict(record.info),
            )
            self._storage.record_step(storage_record)
        if self._store:
            _LOGGER.debug(f"[TELEMETRY PERSIST] Persisting to SQLite store: "
                        f"episode_id={record.episode_id} step_index={record.step_index}")
            # Schedule async write without blocking drain loop
            try:
                # Check if there's a RUNNING event loop
                # asyncio.get_running_loop() raises RuntimeError if no loop is running
                asyncio.get_running_loop()
                # If we get here, there's a running loop - use async write
                asyncio.create_task(self._async_record_step(record))
                _LOGGER.debug(f"[TELEMETRY] Scheduled async write for episode_id={record.episode_id}")
            except RuntimeError:
                # No running event loop - use sync write immediately
                _LOGGER.debug(f"[TELEMETRY] No running event loop, using sync write for episode_id={record.episode_id}")
                self._store.record_step(record)

    async def _async_record_step(self, record: StepRecord) -> None:
        """Record step asynchronously to avoid blocking drain loop."""
        if not self._store:
            return
        try:
            # Run blocking DB write in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._store.record_step, record)
        except Exception as e:
            self.log_constant(
                LOG_SERVICE_TELEMETRY_ASYNC_ERROR,
                exc_info=e,
                extra={
                    "episode_id": record.episode_id,
                    "step_index": record.step_index,
                },
            )

    def complete_episode(self, rollup: EpisodeRollup) -> None:
        self._episode_history.append(rollup)
        if self._storage:
            self._storage.flush_episode()
        if self._store:
            self._store.record_episode(rollup, wait=False)

    # ------------------------------------------------------------------
    def recent_steps(self, *, limit: Optional[int] = None) -> Iterable[StepRecord]:
        if self._store:
            self._store.flush()
            return self._store.recent_steps(limit or self._history_limit)
        steps = list(self._step_history)
        if limit is None:
            return tuple(steps)
        return tuple(steps[-limit:])

    def recent_episodes(self) -> Iterable[EpisodeRollup]:
        if self._store:
            self._store.flush()
            return self._store.recent_episodes(self._episode_history.maxlen or 20)
        return tuple(self._episode_history)

    def episode_steps(self, episode_id: str) -> Iterable[StepRecord]:
        if self._store:
            self._store.flush()
            return self._store.episode_steps(episode_id)
        return tuple(step for step in self._step_history if step.episode_id == episode_id)

    def reset(self) -> None:
        self._step_history.clear()
        self._episode_history.clear()
        if self._storage:
            self._storage.reset_session()

    # ------------------------------------------------------------------
    def delete_episode(self, episode_id: str) -> None:
        """Remove a single episode from recent telemetry buffers and store."""

        self._step_history = deque(
            (step for step in self._step_history if step.episode_id != episode_id),
            maxlen=self._history_limit,
        )
        episode_maxlen = self._episode_history.maxlen or max(1, self._history_limit // 2)
        self._episode_history = deque(
            (episode for episode in self._episode_history if episode.episode_id != episode_id),
            maxlen=episode_maxlen,
        )
        if self._store:
            self._store.delete_episode(episode_id, wait=True)

    def clear_all_episodes(self) -> None:
        """Clear all stored episodes and telemetry history."""

        self._step_history.clear()
        self._episode_history.clear()
        if self._store:
            self._store.delete_all_episodes(wait=True)

    # ------------------------------------------------------------------
    def delete_run(self, run_id: str, *, wait: bool = True) -> None:
        """Delete a run's telemetry data and mark it as deleted."""

        if self._store:
            self._store.delete_run(run_id, wait=wait)

    def archive_run(self, run_id: str, *, wait: bool = True) -> None:
        """Archive a run's telemetry data for replay."""

        if self._store:
            self._store.archive_run(run_id, wait=wait)

    def is_run_deleted(self, run_id: str) -> bool:
        """Return True if a run has been marked as deleted."""

        if not self._store:
            return False
        return self._store.is_run_deleted(run_id)

    def is_run_archived(self, run_id: str) -> bool:
        """Return True if a run has been marked as archived."""

        if not self._store:
            return False
        return self._store.is_run_archived(run_id)

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Fetch aggregate metrics for a run from persistent storage."""

        if not self._store:
            return {
                "run_id": run_id,
                "episodes": 0,
                "steps": 0,
                "total_reward": 0.0,
                "last_update": "",
                "status": "unknown",
                "agent_id": "",
            }

        # Ensure any queued writes are persisted before reading summary data
        self._store.flush()
        return self._store.get_run_summary(run_id)


__all__ = ["TelemetryService"]
