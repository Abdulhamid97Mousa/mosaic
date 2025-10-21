"""Pydantic models for data validation across the application."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class TelemetryEventBase(BaseModel):
    """Base model for all telemetry events."""

    type: str = Field(..., description="Event type identifier")
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Event timestamp")

    class Config:
        """Pydantic config."""
        extra = "allow"  # Allow additional fields for extensibility


class RunStartedEvent(TelemetryEventBase):
    """Telemetry event emitted when a training run starts."""

    type: str = Field(default="run_started", description="Event type")
    run_id: str = Field(..., description="Unique run identifier")
    config: Dict[str, Any] = Field(..., description="Training configuration")

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: str) -> str:
        """Validate run_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("run_id must be non-empty")
        return v.strip()


class RunCompletedEvent(TelemetryEventBase):
    """Telemetry event emitted when a training run completes."""

    type: str = Field(default="run_completed", description="Event type")
    run_id: str = Field(..., description="Unique run identifier")
    status: str = Field(..., description="Completion status (completed, failed, cancelled)")
    error: Optional[str] = Field(None, description="Error message if status is failed")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status is one of allowed values."""
        allowed = {"completed", "failed", "cancelled"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}, got {v}")
        return v


class StepEvent(TelemetryEventBase):
    """Telemetry event emitted for each environment step."""

    type: str = Field(default="step", description="Event type")
    run_id: str = Field(..., description="Unique run identifier")
    episode: int = Field(..., ge=0, description="Episode number")
    step: int = Field(..., ge=0, description="Step index within episode")
    action: int = Field(..., ge=0, description="Action taken")
    reward: float = Field(..., description="Reward received")
    state: int = Field(..., ge=0, description="Current state")
    next_state: int = Field(..., ge=0, description="Next state")
    terminated: bool = Field(..., description="Whether episode terminated")
    truncated: bool = Field(..., description="Whether episode was truncated")
    q_before: float = Field(..., description="Q-value before update")
    q_after: float = Field(..., description="Q-value after update")
    epsilon: float = Field(..., ge=0.0, le=1.0, description="Exploration rate")
    observation: Optional[Any] = Field(None, description="Observation data")
    next_observation: Optional[Any] = Field(None, description="Next observation data")

    @field_validator("episode", "step")
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        """Validate episode and step are non-negative."""
        if v < 0:
            raise ValueError("episode and step must be non-negative")
        return v


class EpisodeEvent(TelemetryEventBase):
    """Telemetry event emitted when an episode completes."""

    type: str = Field(default="episode", description="Event type")
    run_id: str = Field(..., description="Unique run identifier")
    episode: int = Field(..., ge=0, description="Episode number")
    reward: float = Field(..., description="Total episode reward")
    steps: int = Field(..., ge=0, description="Number of steps in episode")
    success: bool = Field(..., description="Whether episode was successful")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Episode metadata")

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, v: int) -> int:
        """Validate steps is non-negative."""
        if v < 0:
            raise ValueError("steps must be non-negative")
        return v


class ArtifactEvent(TelemetryEventBase):
    """Telemetry event emitted when an artifact is saved."""

    type: str = Field(default="artifact", description="Event type")
    run_id: str = Field(..., description="Unique run identifier")
    kind: str = Field(..., description="Artifact kind (policy, video, etc.)")
    path: str = Field(..., description="Path to artifact file")

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: str) -> str:
        """Validate artifact kind."""
        allowed = {"policy", "video", "checkpoint", "log"}
        if v not in allowed:
            raise ValueError(f"kind must be one of {allowed}, got {v}")
        return v


class TrainingConfig(BaseModel):
    """Validated training configuration."""

    run_id: str = Field(..., description="Unique run identifier")
    game_id: str = Field(..., description="Game identifier")
    seed: int = Field(default=0, ge=0, description="Random seed")
    max_episodes: int = Field(default=1, ge=1, description="Maximum episodes to train")
    max_steps_per_episode: int = Field(default=200, ge=1, description="Max steps per episode")
    policy_strategy: str = Field(default="eval", description="Policy strategy")
    policy_path: Optional[str] = Field(None, description="Path to policy file")
    agent_id: str = Field(default="bdi_rl", description="Agent identifier")
    capture_video: bool = Field(default=False, description="Whether to capture video")
    headless: bool = Field(default=True, description="Whether to run headless")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Extra configuration")

    @field_validator("policy_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate policy strategy."""
        allowed = {"train", "load", "eval", "train_and_save"}
        if v not in allowed:
            raise ValueError(f"policy_strategy must be one of {allowed}, got {v}")
        return v


def validate_telemetry_event(event_data: Dict[str, Any]) -> TelemetryEventBase:
    """Validate and parse a telemetry event from raw data.
    
    Args:
        event_data: Raw event data dictionary
        
    Returns:
        Validated telemetry event
        
    Raises:
        ValueError: If event data is invalid
    """
    event_type = event_data.get("type")
    
    if event_type == "run_started":
        return RunStartedEvent(**event_data)
    elif event_type == "run_completed":
        return RunCompletedEvent(**event_data)
    elif event_type == "step":
        return StepEvent(**event_data)
    elif event_type == "episode":
        return EpisodeEvent(**event_data)
    elif event_type == "artifact":
        return ArtifactEvent(**event_data)
    else:
        raise ValueError(f"Unknown event type: {event_type}")


__all__ = [
    "TelemetryEventBase",
    "RunStartedEvent",
    "RunCompletedEvent",
    "StepEvent",
    "EpisodeEvent",
    "ArtifactEvent",
    "TrainingConfig",
    "validate_telemetry_event",
]

