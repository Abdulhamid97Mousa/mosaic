# gym_gui/services/actor.py

from __future__ import annotations

"""Actor abstractions and registry for human and autonomous agents."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Protocol

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import LOG_SERVICE_ACTOR_SEED_ERROR


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
    seed: int | None = None
    info: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeSummary:
    """Aggregated information produced when an episode finishes."""

    episode_index: int
    total_reward: float
    steps: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ActorDescriptor:
    """Metadata describing a registered actor for UI presentation."""

    actor_id: str
    display_name: str
    description: str | None = None
    policy_label: str | None = None
    backend_label: str | None = None


class ActorService(LogConstantMixin):
    """Registry that coordinates active actors for the current session."""

    def __init__(self) -> None:
        self._actors: Dict[str, Actor] = {}
        self._descriptors: Dict[str, ActorDescriptor] = {}
        self._active_actor_id: Optional[str] = None
        self._last_seed: int | None = None
        self._logger = logging.getLogger("gym_gui.services.actor")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_actor(
        self,
        actor: Actor,
        *,
        display_name: str | None = None,
        description: str | None = None,
        policy_label: str | None = None,
        backend_label: str | None = None,
        activate: bool = False,
    ) -> None:
        actor_id = actor.id
        label = display_name or actor_id.replace("_", " ").title()
        self._actors[actor_id] = actor
        self._descriptors[actor_id] = ActorDescriptor(
            actor_id=actor_id,
            display_name=label,
            description=description,
            policy_label=policy_label,
            backend_label=backend_label,
        )
        if activate or self._active_actor_id is None:
            self._active_actor_id = actor_id

    def available_actor_ids(self) -> Iterable[str]:
        return self._actors.keys()

    def describe_actors(self) -> tuple[ActorDescriptor, ...]:
        """Return metadata for all registered actors in registration order."""

        return tuple(self._descriptors[actor_id] for actor_id in self._actors.keys())

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

    def get_active_actor_id(self) -> Optional[str]:
        return self._active_actor_id

    def get_actor_descriptor(self, actor_id: str) -> Optional[ActorDescriptor]:
        return self._descriptors.get(actor_id)

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

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    def seed(self, seed: int) -> None:
        """Propagate a deterministic seed to all registered actors."""

        self._last_seed = seed
        for actor_id, actor in self._actors.items():
            callback = getattr(actor, "seed", None)
            if not callable(callback):
                continue
            try:
                callback(seed)
            except Exception as exc:  # pragma: no cover - defensive guard
                self.log_constant(
                    LOG_SERVICE_ACTOR_SEED_ERROR,
                    message="actor_seed_failed",
                    extra={"actor_id": actor_id},
                    exc_info=exc,
                )

    @property
    def last_seed(self) -> Optional[int]:
        return self._last_seed


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
class LLMMultiStepAgent:
    """Agent that leverages an LLM with tool calls for decision making."""

    id: str = "llm_multi_step"

    def select_action(self, step: StepSnapshot) -> Optional[int]:
        # Placeholder: integrate with tool/snapshot pipeline.
        return None


@dataclass(slots=True)
class CleanRLWorkerActor:
    """Placeholder actor representing the CleanRL worker backend (no direct actions)."""

    id: str = "cleanrl_worker"

    def select_action(self, step: StepSnapshot) -> Optional[int]:  # pragma: no cover - managed out-of-band
        return None

    def on_step(self, step: StepSnapshot) -> None:  # pragma: no cover - managed out-of-band
        return

    def on_episode_end(self, summary: EpisodeSummary) -> None:  # pragma: no cover - managed out-of-band
        return


__all__ = [
    "Actor",
    "ActorService",
    "ActorDescriptor",
    "StepSnapshot",
    "EpisodeSummary",
    "HumanKeyboardActor",
    "LLMMultiStepAgent",
    "CleanRLWorkerActor",
]
