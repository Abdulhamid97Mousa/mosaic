import unittest
from dataclasses import dataclass
from typing import Any, Protocol, cast

from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.storage.session import EpisodeRecord
from gym_gui.services.telemetry import TelemetryService


@dataclass
class _DummyStepRecord:
    episode_id: str
    step_index: int = 0
    observation: object = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.info is None:
            self.info = {}


class StorageRecorderProtocol(Protocol):
    """Protocol for storage recorder services."""
    def record_step(self, record: Any) -> None: ...
    def flush_episode(self) -> None: ...
    def reset_session(self) -> None: ...


class TelemetryStoreProtocol(Protocol):
    """Protocol for telemetry stores."""
    def record_step(self, record: StepRecord) -> None: ...
    def record_episode(self, rollup: EpisodeRollup, *, wait: bool = False) -> None: ...
    def flush(self) -> None: ...
    def recent_steps(self, limit: int) -> list[StepRecord]: ...
    def recent_episodes(self, limit: int) -> list[EpisodeRollup]: ...
    def episode_steps(self, episode_id: str) -> list[StepRecord]: ...
    def delete_episode(self, episode_id: str, *, wait: bool = True) -> None: ...
    def delete_all_episodes(self, *, wait: bool = True) -> None: ...


class _DummyStorage:
    def __init__(self) -> None:
        self.steps: list[Any] = []
        self.flushed = False
        self.reset_called = False

    def record_step(self, record: Any) -> None:
        self.steps.append(record)

    def flush_episode(self) -> None:
        self.flushed = True

    def reset_session(self) -> None:
        self.reset_called = True


class _DummyStore:
    def __init__(self) -> None:
        self.steps: list[StepRecord] = []
        self.episodes: list[tuple[EpisodeRollup, bool]] = []
        self.flushed = False
        self.deleted: list[str] = []

    def record_step(self, record: StepRecord) -> None:
        self.steps.append(record)

    def record_episode(self, rollup: EpisodeRollup, *, wait: bool = False) -> None:
        self.episodes.append((rollup, wait))

    def flush(self) -> None:
        self.flushed = True

    def recent_steps(self, limit: int) -> list[StepRecord]:
        return self.steps[-limit:]

    def recent_episodes(self, limit: int) -> list[EpisodeRollup]:
        return [ep[0] for ep in self.episodes[-limit:]]

    def episode_steps(self, episode_id: str) -> list[StepRecord]:
        return [step for step in self.steps if step.episode_id == episode_id]

    def delete_episode(self, episode_id: str, *, wait: bool = True) -> None:
        self.deleted.append(episode_id)

    def delete_all_episodes(self, *, wait: bool = True) -> None:
        self.deleted.append("__all__")


class TelemetryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TelemetryService(history_limit=10)
        self.storage = _DummyStorage()
        self.store = _DummyStore()
        # Type ignore needed because these are test mocks
        self.service.attach_storage(self.storage)  # type: ignore[arg-type]
        self.service.attach_store(self.store)  # type: ignore[arg-type]

    def test_record_step_forwards_to_storage_and_store(self) -> None:
        record = StepRecord(
            episode_id="ep-1",
            step_index=0,
            action=None,
            observation={"obs": 1},
            reward=1.0,
            terminated=False,
            truncated=False,
            info={},
        )
        self.service.record_step(record)

        self.assertEqual(len(self.storage.steps), 1)
        self.assertEqual(len(self.store.steps), 1)

    def test_record_step_propagates_metadata(self) -> None:
        space_signature = {"observation": {"shape": [3], "dtype": "float32"}, "action": {"n": 2}}
        vector_metadata = {"autoreset_mode": "NextStep", "batch_size": 4}
        record = StepRecord(
            episode_id="ep-meta",
            step_index=3,
            action=None,
            observation={"obs": 2},
            reward=2.5,
            terminated=False,
            truncated=False,
            info={},
            time_step=7,
            space_signature=space_signature,
            vector_metadata=vector_metadata,
        )
        self.service.record_step(record)

        storage_record = cast(EpisodeRecord, self.storage.steps[-1])
        self.assertEqual(storage_record.time_step, 7)
        self.assertEqual(storage_record.space_signature, space_signature)
        self.assertEqual(storage_record.vector_metadata, vector_metadata)

        stored_step = self.store.steps[-1]
        self.assertEqual(stored_step.vector_metadata, vector_metadata)
        self.assertEqual(stored_step.space_signature, space_signature)
        self.assertEqual(stored_step.time_step, 7)

    def test_complete_episode_flushes_storage_and_store(self) -> None:
        rollup = EpisodeRollup(
            episode_id="ep-1",
            total_reward=5.0,
            steps=10,
            terminated=True,
            truncated=False,
            metadata={},
        )
        self.service.complete_episode(rollup)
        self.assertTrue(self.storage.flushed)
        self.assertIn((rollup, False), self.store.episodes)

    def test_recent_steps_uses_store(self) -> None:
        for idx in range(3):
            record = StepRecord(
                episode_id="ep",
                step_index=idx,
                action=None,
                observation=idx,
                reward=0.0,
                terminated=False,
                truncated=False,
                info={},
            )
            self.service.record_step(record)
        recent = list(self.service.recent_steps(limit=2))
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[-1].step_index, 2)

    def test_delete_episode_forwards_to_store(self) -> None:
        self.service.delete_episode("ep-42")
        self.assertIn("ep-42", self.store.deleted)

    def test_clear_all_episodes(self) -> None:
        self.service.clear_all_episodes()
        self.assertIn("__all__", self.store.deleted)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
