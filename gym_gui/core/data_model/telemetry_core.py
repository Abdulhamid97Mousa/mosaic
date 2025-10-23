from __future__ import annotations

"""Telemetry records captured during environment interaction."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    timestamp: datetime = field(default_factory=_utc_now)
    render_payload: Any | None = None
    agent_id: str | None = None
    render_hint: Mapping[str, Any] | None = None
    frame_ref: str | None = None
    payload_version: int = 0
    run_id: str | None = None  # NEW: Training run identifier for correlation


@dataclass(slots=True)
class EpisodeRollup:
    """Aggregated metrics emitted when an episode completes."""

    episode_id: str
    total_reward: float
    steps: int
    terminated: bool
    truncated: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utc_now)
    agent_id: str | None = None
    run_id: str | None = None  # NEW: Training run identifier for correlation
    game_id: str | None = None  # NEW: Game environment identifier


__all__ = ["StepRecord", "EpisodeRollup"]
