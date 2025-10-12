from __future__ import annotations

"""Telemetry collection and routing service."""

from collections import deque
from typing import Deque, Iterable, Optional, TYPE_CHECKING

from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.storage.session import EpisodeRecord

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from gym_gui.services.storage import StorageRecorderService
    from gym_gui.telemetry import TelemetrySQLiteStore


class TelemetryService:
    """Aggregates telemetry events and forwards them to storage."""

    def __init__(self, *, history_limit: int = 512) -> None:
        self._history_limit = max(1, history_limit)
        self._step_history: Deque[StepRecord] = deque(maxlen=self._history_limit)
        self._episode_history: Deque[EpisodeRollup] = deque(
            maxlen=max(1, self._history_limit // 2)
        )
        self._storage: "StorageRecorderService | None" = None
        self._store: "TelemetrySQLiteStore | None" = None

    def attach_storage(self, storage: "StorageRecorderService") -> None:
        self._storage = storage

    def attach_store(self, store: "TelemetrySQLiteStore") -> None:
        self._store = store

    # ------------------------------------------------------------------
    def record_step(self, record: StepRecord) -> None:
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
            self._store.record_step(record)

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


__all__ = ["TelemetryService"]
