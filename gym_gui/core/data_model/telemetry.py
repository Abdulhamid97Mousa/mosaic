from __future__ import annotations

"""Telemetry records captured during environment interaction."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(slots=True)
class StepRecord:
    """Canonical telemetry record for a single environment step."""

    episode_id: str
    step_index: int
    action: int | None
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    render_payload: Any | None = None


@dataclass(slots=True)
class EpisodeRollup:
    """Aggregated metrics emitted when an episode completes."""

    episode_id: str
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


__all__ = ["StepRecord", "EpisodeRollup"]
