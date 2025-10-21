"""Telemetry lifecycle and control events."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class TrainingOutcome(str, Enum):
    """Training completion outcome."""
    SUCCESS = "success"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class EpisodeFinalized:
    """Emitted when an episode is complete and persisted to DB."""
    run_id: str
    agent_id: str
    episode_index: int
    seq_id: int
    timestamp_iso: str
    
    @classmethod
    def now(cls, run_id: str, agent_id: str, episode_index: int, seq_id: int) -> "EpisodeFinalized":
        """Create with current timestamp."""
        return cls(
            run_id=run_id,
            agent_id=agent_id,
            episode_index=episode_index,
            seq_id=seq_id,
            timestamp_iso=datetime.now(timezone.utc).isoformat()
        )


@dataclass(slots=True, frozen=True)
class TrainingStarted:
    """Emitted when training run is accepted and starts."""
    run_id: str
    agent_id: Optional[str]
    game_id: Optional[str]
    timestamp_iso: str
    seed: Optional[int] = None
    meta: Optional[dict] = None
    
    @classmethod
    def now(cls, run_id: str, agent_id: Optional[str], game_id: Optional[str], 
            seed: Optional[int] = None, meta: Optional[dict] = None) -> "TrainingStarted":
        """Create with current timestamp."""
        return cls(
            run_id=run_id,
            agent_id=agent_id,
            game_id=game_id,
            seed=seed,
            meta=meta or {},
            timestamp_iso=datetime.now(timezone.utc).isoformat()
        )


@dataclass(slots=True, frozen=True)
class TrainingFinished:
    """Emitted when training run completes (success/cancel/fail)."""
    run_id: str
    agent_id: Optional[str]
    outcome: TrainingOutcome
    timestamp_iso: str
    episodes_total: Optional[int] = None
    steps_total: Optional[int] = None
    total_reward: Optional[float] = None
    error_message: Optional[str] = None
    
    @classmethod
    def now(cls, run_id: str, agent_id: Optional[str], outcome: TrainingOutcome,
            episodes_total: Optional[int] = None, steps_total: Optional[int] = None,
            total_reward: Optional[float] = None, error_message: Optional[str] = None) -> "TrainingFinished":
        """Create with current timestamp."""
        return cls(
            run_id=run_id,
            agent_id=agent_id,
            outcome=outcome,
            episodes_total=episodes_total,
            steps_total=steps_total,
            total_reward=total_reward,
            error_message=error_message,
            timestamp_iso=datetime.now(timezone.utc).isoformat()
        )

