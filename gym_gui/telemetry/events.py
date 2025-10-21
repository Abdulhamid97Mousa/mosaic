"""Telemetry event protocol for Pub/Sub bus (Ray-inspired dual-path architecture).

This module defines the event contract that all telemetry producers and consumers
understand. Events flow through an in-process Pub/Sub bus (RunBus) which decouples
live UI updates from durable database persistence.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional


class Topic(Enum):
    """Event topics for the telemetry Pub/Sub bus."""

    RUN_STARTED = auto()          # Training run initialized
    RUN_HEARTBEAT = auto()        # Periodic heartbeat during training
    RUN_COMPLETED = auto()        # Training finished (success/failure)
    STEP_APPENDED = auto()        # New step recorded
    EPISODE_FINALIZED = auto()    # Episode completed
    OVERFLOW = auto()             # Queue overflow warning
    CONTROL = auto()              # Control plane messages (credit grants, backpressure signals)


@dataclass(slots=True)
class TelemetryEvent:
    """Immutable telemetry event for Pub/Sub distribution.
    
    Attributes:
        topic: Event topic (RUN_STARTED, STEP_APPENDED, etc.)
        run_id: Training run identifier
        agent_id: Agent identifier (optional)
        seq_id: Strictly increasing sequence number per stream
        ts_iso: ISO-8601 timestamp from producer
        payload: Event-specific data (step, episode, run metadata)
    """
    
    topic: Topic
    run_id: str
    agent_id: Optional[str]
    seq_id: int                    # Strictly increasing per stream
    ts_iso: str                    # Producer timestamp (ISO-8601)
    payload: Dict[str, Any]        # step|episode|run metadata
    
    def __repr__(self) -> str:
        """Compact representation for logging."""
        return (
            f"TelemetryEvent(topic={self.topic.name}, run_id={self.run_id}, "
            f"agent_id={self.agent_id}, seq_id={self.seq_id})"
        )

