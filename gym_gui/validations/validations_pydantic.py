"""Pydantic models and helpers for telemetry and trainer validation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class TelemetryEventBase(BaseModel):
    """Base model for all telemetry events."""

    type: str = Field(..., description="Event type identifier")
    ts: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp",
    )

    class Config:
        """Pydantic config allowing forward-compatible fields."""

        extra = "allow"  # Allow additional fields for extensibility


class RunStartedEvent(TelemetryEventBase):
    """Telemetry event emitted when a training run starts."""

    type: str = Field(default="run_started", description="Event type")
    run_id: str = Field(..., description="Unique run identifier")
    config: Dict[str, Any] = Field(..., description="Training configuration")

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("run_id must be non-empty")
        return value.strip()


class RunCompletedEvent(TelemetryEventBase):
    """Telemetry event emitted when a training run completes."""

    type: str = Field(default="run_completed", description="Event type")
    run_id: str = Field(..., description="Unique run identifier")
    status: str = Field(..., description="Completion status (completed, failed, cancelled)")
    error: Optional[str] = Field(None, description="Error message if status is failed")

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        allowed = {"completed", "failed", "cancelled"}
        if value not in allowed:
            raise ValueError(f"status must be one of {allowed}, got {value}")
        return value


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
    def validate_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("episode and step must be non-negative")
        return value


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
    def validate_steps(cls, value: int) -> int:
        if value < 0:
            raise ValueError("steps must be non-negative")
        return value


class ArtifactEvent(TelemetryEventBase):
    """Telemetry event emitted when an artifact is saved."""

    type: str = Field(default="artifact", description="Event type")
    run_id: str = Field(..., description="Unique run identifier")
    kind: str = Field(..., description="Artifact kind (policy, video, etc.)")
    path: str = Field(..., description="Path to artifact file")

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        allowed = {"policy", "video", "checkpoint", "log", "tensorboard"}
        if value not in allowed:
            raise ValueError(f"kind must be one of {allowed}, got {value}")
        return value


class SubprocessCommand(BaseModel):
    """Validated subprocess command with argument type safety."""

    args: list[str] = Field(..., description="Command arguments")

    @field_validator("args")
    @classmethod
    def validate_args(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"args must be a list, got {type(value).__name__}")
        if not value:
            raise ValueError("args list cannot be empty")

        for idx, arg in enumerate(value):
            if not isinstance(arg, str):
                raise ValueError(
                    f"Command argument at index {idx} must be a string, "
                    f"got {type(arg).__name__}: {arg!r}"
                )
            if not arg:
                raise ValueError(f"Command argument at index {idx} is empty string")
        return value


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
    def validate_strategy(cls, value: str) -> str:
        allowed = {"train", "load", "eval", "train_and_save"}
        if value not in allowed:
            raise ValueError(f"policy_strategy must be one of {allowed}, got {value}")
        return value


class TrainerControlUpdate(BaseModel):
    """Structured supervisor control update for trainer parameters.

    This model intentionally stays generic to allow diverse knobs with
    bounded ranges. Unknown keys are allowed to enable gradual adoption.
    """

    run_id: str = Field(..., description="Target run identifier")
    reason: str = Field(..., description="Human-readable reason for the change")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameter overrides")

    class Config:
        extra = "allow"

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("run_id must be non-empty")
        return value.strip()

    @field_validator("params")
    @classmethod
    def validate_params(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        # Bounds checking for common knobs (epsilon, per alpha/beta, lr_mult, tau)
        if not isinstance(value, dict):
            raise ValueError("params must be a mapping")
        # Optional best-effort validation without hard failures
        try:
            eps = value.get("epsilon")
            if eps is not None and not (0.0 <= float(eps) <= 1.0):
                raise ValueError("epsilon must be within [0, 1]")
            alpha = value.get("per_alpha")
            if alpha is not None and not (0.0 <= float(alpha) <= 1.0):
                raise ValueError("per_alpha must be within [0, 1]")
            beta = value.get("per_beta")
            if beta is not None and not (0.0 <= float(beta) <= 1.0):
                raise ValueError("per_beta must be within [0, 1]")
            tau = value.get("tau")
            if tau is not None and not (0.0 < float(tau) <= 1.0):
                raise ValueError("tau must be within (0, 1]")
            lr_mult = value.get("lr_multiplier")
            if lr_mult is not None and not (0.0 < float(lr_mult) <= 10.0):
                raise ValueError("lr_multiplier must be within (0, 10]")
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc))
        return value


def validate_telemetry_event(event_data: Dict[str, Any]) -> TelemetryEventBase:
    """Validate and parse a telemetry event from raw data."""

    event_type = event_data.get("type")

    if event_type == "run_started":
        return RunStartedEvent(**event_data)
    if event_type == "run_completed":
        return RunCompletedEvent(**event_data)
    if event_type == "step":
        return StepEvent(**event_data)
    if event_type == "episode":
        return EpisodeEvent(**event_data)
    if event_type == "artifact":
        return ArtifactEvent(**event_data)
    raise ValueError(f"Unknown event type: {event_type}")


__all__ = [
    "TelemetryEventBase",
    "RunStartedEvent",
    "RunCompletedEvent",
    "StepEvent",
    "EpisodeEvent",
    "ArtifactEvent",
    "SubprocessCommand",
    "TrainingConfig",
    "TrainerControlUpdate",
    "validate_telemetry_event",
]
