# gym_gui/services/actor.py

from __future__ import annotations

"""Actor abstractions and registry for human and autonomous agents.

This module provides:
- Actor: Simple protocol for single-agent action selection
- PolicyController: Paradigm-aware protocol for multi-agent/multi-paradigm support
- ActorService: Registry for managing active actors
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Protocol

from gym_gui.core.enums import SteppingParadigm
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import LOG_SERVICE_ACTOR_SEED_ERROR


class Actor(Protocol):
    """Protocol every actor implementation must follow.

    This is the simple, legacy protocol for single-agent environments.
    For multi-agent or paradigm-aware control, use PolicyController instead.
    """

    id: str

    def select_action(self, step: "StepSnapshot") -> Optional[int]:
        """Return the next action to apply or ``None`` if the actor abstains."""

    def on_step(self, step: "StepSnapshot") -> None:
        """Receive feedback after an action has been applied."""

    def on_episode_end(self, summary: "EpisodeSummary") -> None:
        """Episode lifecycle hook for cleanup or learning updates."""


class PolicyController(Protocol):
    """Paradigm-aware protocol for multi-agent and multi-paradigm policy control.

    This protocol extends the Actor concept with:
    1. Agent-specific action selection (for multi-agent environments)
    2. Batch action selection (for SIMULTANEOUS/POSG paradigms)
    3. Explicit paradigm declaration

    PolicyController is designed to work with the WorkerOrchestrator and
    PolicyMappingService for paradigm-agnostic training coordination.

    Example (Sequential/AEC):
        >>> controller.select_action("player_0", observation, info)

    Example (Simultaneous/POSG):
        >>> controller.select_actions({"player_0": obs0, "player_1": obs1})

    See Also:
        - docs/1.0_DAY_41/TASK_1/00_multi_paradigm_orchestrator_plan.md
        - docs/1.0_DAY_41/TASK_1/01_paradigm_comparison.md
    """

    @property
    def id(self) -> str:
        """Unique identifier for this policy controller."""
        ...

    @property
    def paradigm(self) -> SteppingParadigm:
        """The stepping paradigm this controller is designed for."""
        ...

    def select_action(
        self,
        agent_id: str,
        observation: Any,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Select action for a specific agent (Sequential/AEC mode).

        Args:
            agent_id: The identifier of the agent needing an action.
            observation: The agent's current observation.
            info: Optional environment info dict.

        Returns:
            The action to take, or None to abstain.
        """
        ...

    def select_actions(
        self,
        observations: Dict[str, Any],
        infos: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Select actions for all agents at once (Simultaneous/POSG mode).

        Args:
            observations: Dict mapping agent_id to observation.
            infos: Optional dict mapping agent_id to info dict.

        Returns:
            Dict mapping agent_id to action.
        """
        ...

    def on_step_result(
        self,
        agent_id: str,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        """Receive feedback after a step (for learning updates).

        Args:
            agent_id: The agent that took the action.
            observation: New observation after the step.
            reward: Reward received.
            terminated: Whether episode ended naturally.
            truncated: Whether episode was truncated.
            info: Environment info dict.
        """
        ...

    def on_episode_end(
        self,
        agent_id: str,
        summary: "EpisodeSummary",
    ) -> None:
        """Called when an episode ends for a specific agent.

        Args:
            agent_id: The agent whose episode ended.
            summary: Aggregated episode information.
        """
        ...

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state for a new episode.

        Args:
            seed: Optional deterministic seed.
        """
        ...


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
    "PolicyController",
    "ActorService",
    "ActorDescriptor",
    "StepSnapshot",
    "EpisodeSummary",
    "HumanKeyboardActor",
    "LLMMultiStepAgent",
    "CleanRLWorkerActor",
]
