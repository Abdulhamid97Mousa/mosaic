from __future__ import annotations

"""Actor abstractions and registry for human and autonomous agents."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Protocol


class Actor(Protocol):
    """Protocol every actor implementation must follow."""

    id: str

    def select_action(self, step: "StepSnapshot") -> Optional[int]:
        """Return the next action to apply or ``None`` if the actor abstains."""

    def on_step(self, step: "StepSnapshot") -> None:
        """Receive feedback after an action has been applied."""

    def on_episode_end(self, summary: "EpisodeSummary") -> None:
        """Episode lifecycle hook for cleanup or learning updates."""


@dataclass(slots=True)
class StepSnapshot:
    """Minimal snapshot describing the current environment state."""

    step_index: int
    observation: object
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeSummary:
    """Aggregated information produced when an episode finishes."""

    episode_index: int
    total_reward: float
    steps: int
    metadata: dict[str, object] = field(default_factory=dict)


class ActorService:
    """Registry that coordinates active actors for the current session."""

    def __init__(self) -> None:
        self._actors: Dict[str, Actor] = {}
        self._active_actor_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_actor(self, actor: Actor, *, activate: bool = False) -> None:
        self._actors[actor.id] = actor
        if activate or self._active_actor_id is None:
            self._active_actor_id = actor.id

    def available_actor_ids(self) -> Iterable[str]:
        return self._actors.keys()

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------
    def set_active_actor(self, actor_id: str) -> None:
        if actor_id not in self._actors:
            raise KeyError(f"Unknown actor '{actor_id}'")
        self._active_actor_id = actor_id

    def get_active_actor(self) -> Optional[Actor]:
        if self._active_actor_id is None:
            return None
        return self._actors.get(self._active_actor_id)

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------
    def select_action(self, snapshot: StepSnapshot) -> Optional[int]:
        actor = self.get_active_actor()
        if actor is None:
            return None
        return actor.select_action(snapshot)

    def notify_step(self, snapshot: StepSnapshot) -> None:
        actor = self.get_active_actor()
        if actor is None:
            return
        actor.on_step(snapshot)

    def notify_episode_end(self, summary: EpisodeSummary) -> None:
        actor = self.get_active_actor()
        if actor is None:
            return
        actor.on_episode_end(summary)


# Placeholder implementations -------------------------------------------------

@dataclass(slots=True)
class HumanKeyboardActor:
    """Handles human keyboard input forwarded by ``HumanInputController``."""

    id: str = "human_keyboard"

    def select_action(self, step: StepSnapshot) -> Optional[int]:  # pragma: no cover - UI only
        return None

    def on_step(self, step: StepSnapshot) -> None:  # pragma: no cover - UI only
        return

    def on_episode_end(self, summary: EpisodeSummary) -> None:  # pragma: no cover - UI only
        return


@dataclass(slots=True)
class BDIQAgent:
    """Skeleton for a BDI + Q-learning driven agent implementation."""

    id: str = "bdi_q_agent"

    def select_action(self, step: StepSnapshot) -> Optional[int]:
        # Placeholder: real implementation will consult Q-table / policy.
        return None

    def on_step(self, step: StepSnapshot) -> None:
        # Placeholder hook for Q-value updates.
        return

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        # Placeholder for logging or checkpointing.
        return


@dataclass(slots=True)
class LLMMultiStepAgent:
    """Agent that leverages an LLM with tool calls for decision making."""

    id: str = "llm_multi_step"

    def select_action(self, step: StepSnapshot) -> Optional[int]:
        # Placeholder: integrate with tool/snapshot pipeline.
        return None

    def on_step(self, step: StepSnapshot) -> None:
        return

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        return


__all__ = [
    "Actor",
    "ActorService",
    "StepSnapshot",
    "EpisodeSummary",
    "HumanKeyboardActor",
    "BDIQAgent",
    "LLMMultiStepAgent",
]
