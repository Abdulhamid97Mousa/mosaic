from __future__ import annotations

"""Replay loaders built on top of the telemetry storage layer."""

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

from gym_gui.core.data_model import EpisodeRollup, StepRecord
from gym_gui.services.telemetry import TelemetryService


@dataclass(slots=True)
class EpisodeReplay:
    """In-memory representation of a telemetry-backed episode."""

    rollup: EpisodeRollup
    steps: List[StepRecord]

    def __iter__(self) -> Iterator[StepRecord]:
        return iter(self.steps)

    @property
    def total_reward(self) -> float:
        return self.rollup.total_reward

    @property
    def episode_id(self) -> str:
        return self.rollup.episode_id


__all__ = ["EpisodeReplay", "EpisodeReplayLoader"]


class EpisodeReplayLoader:
    """Load full episodes from telemetry services for playback."""

    def __init__(self, telemetry: TelemetryService) -> None:
        self._telemetry = telemetry

    def recent_episode_ids(self, *, limit: int = 20) -> Iterable[str]:
        return [episode.episode_id for episode in self._telemetry.recent_episodes()][:limit]

    def load_episode(self, episode_id: str) -> Optional[EpisodeReplay]:
        steps = list(self._telemetry.episode_steps(episode_id))
        if not steps:
            return None
        rollup = next(
            (episode for episode in self._telemetry.recent_episodes() if episode.episode_id == episode_id),
            None,
        )
        if rollup is None:
            rollup = EpisodeRollup(
                episode_id=episode_id,
                total_reward=sum(step.reward for step in steps),
                steps=len(steps),
                terminated=steps[-1].terminated,
                truncated=steps[-1].truncated,
                metadata={},
                agent_id=steps[-1].agent_id,
            )
        return EpisodeReplay(rollup=rollup, steps=steps)
