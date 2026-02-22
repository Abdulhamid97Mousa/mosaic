# gym_gui/services/policy_mapping.py

"""PolicyMappingService for per-agent policy mapping in multi-agent environments.

This module provides:
- AgentPolicyBinding: Binding between an agent and its policy controller
- PolicyMappingService: Per-agent policy mapping with paradigm awareness

The PolicyMappingService extends ActorService to support:
1. Multiple active policies (one per agent)
2. Paradigm-aware action selection (Sequential vs Simultaneous)
3. Worker-specific routing

For single-agent environments, it delegates to ActorService.
For multi-agent, it maintains agent_id → policy_id mapping.

See Also:
    - :doc:`/documents/architecture/policy_mapping` for policy mapping details
    - :doc:`/documents/architecture/paradigms` for stepping paradigm architecture
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from gym_gui.core.enums import SteppingParadigm
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.services.actor import (
    Actor,
    ActorService,
    EpisodeSummary,
    StepSnapshot,
)


@dataclass
class AgentPolicyBinding:
    """Binding between an agent and its policy controller.

    Attributes:
        agent_id: Unique identifier for the agent in the environment.
        policy_id: References an Actor registered in ActorService.
        worker_id: Optional worker identifier (e.g., "cleanrl_worker", "llm_worker").
        config: Worker-specific configuration options.
    """

    agent_id: str
    policy_id: str
    worker_id: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


class PolicyMappingService(LogConstantMixin):
    """Per-agent policy mapping for multi-agent environments.

    Extends ActorService to support:
    1. Multiple active policies (one per agent)
    2. Paradigm-aware action selection
    3. Worker-specific routing

    For single-agent environments, delegates to ActorService.
    For multi-agent, maintains agent_id → policy_id mapping.

    Example:
        >>> actor_service = ActorService()
        >>> actor_service.register_actor(HumanKeyboardActor(), activate=True)
        >>> actor_service.register_actor(CleanRLWorkerActor())
        >>>
        >>> mapping = PolicyMappingService(actor_service)
        >>> mapping.set_paradigm(SteppingParadigm.SEQUENTIAL)
        >>> mapping.set_agents(["player_0", "player_1"])
        >>> mapping.bind_agent_policy("player_0", "human_keyboard")
        >>> mapping.bind_agent_policy("player_1", "cleanrl_worker")

    See Also:
        - :doc:`/documents/architecture/policy_mapping` for policy mapping details
    """

    def __init__(self, actor_service: ActorService) -> None:
        """Initialize PolicyMappingService.

        Args:
            actor_service: The underlying ActorService for policy management.
        """
        self._actor_service = actor_service
        self._bindings: Dict[str, AgentPolicyBinding] = {}
        self._paradigm: SteppingParadigm = SteppingParadigm.SINGLE_AGENT
        self._agent_ids: List[str] = []
        self._logger = logging.getLogger("gym_gui.services.policy_mapping")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_paradigm(self, paradigm: SteppingParadigm) -> None:
        """Set the stepping paradigm for this session.

        Args:
            paradigm: The stepping paradigm (SINGLE_AGENT, SEQUENTIAL, etc.)
        """
        self._paradigm = paradigm
        self._logger.debug(f"Paradigm set to {paradigm.name}")

    def set_agents(self, agent_ids: List[str]) -> None:
        """Configure the list of agents in the environment.

        Auto-binds agents to the default policy if not already bound.

        Args:
            agent_ids: List of agent identifiers from the environment.
        """
        self._agent_ids = list(agent_ids)
        default_policy = self._actor_service.get_active_actor_id()

        # Auto-bind to default policy if not already bound
        for agent_id in agent_ids:
            if agent_id not in self._bindings and default_policy is not None:
                self._bindings[agent_id] = AgentPolicyBinding(
                    agent_id=agent_id,
                    policy_id=default_policy,
                )
                self._logger.debug(
                    f"Auto-bound agent '{agent_id}' to policy '{default_policy}'"
                )

    def bind_agent_policy(
        self,
        agent_id: str,
        policy_id: str,
        *,
        worker_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Bind an agent to a specific policy.

        Args:
            agent_id: The agent to bind.
            policy_id: The policy (Actor) to use for this agent.
            worker_id: Optional worker identifier for remote execution.
            config: Optional worker-specific configuration.

        Raises:
            KeyError: If policy_id is not registered in ActorService.
        """
        available = list(self._actor_service.available_actor_ids())
        if policy_id not in available:
            raise KeyError(
                f"Unknown policy '{policy_id}'. Available: {available}"
            )

        self._bindings[agent_id] = AgentPolicyBinding(
            agent_id=agent_id,
            policy_id=policy_id,
            worker_id=worker_id,
            config=config or {},
        )
        self._logger.debug(f"Bound agent '{agent_id}' to policy '{policy_id}'")

    def unbind_agent(self, agent_id: str) -> None:
        """Remove binding for an agent.

        Args:
            agent_id: The agent to unbind.
        """
        if agent_id in self._bindings:
            del self._bindings[agent_id]
            self._logger.debug(f"Unbound agent '{agent_id}'")

    def get_binding(self, agent_id: str) -> Optional[AgentPolicyBinding]:
        """Get the policy binding for an agent.

        Args:
            agent_id: The agent identifier.

        Returns:
            The binding if found, otherwise None.
        """
        return self._bindings.get(agent_id)

    def get_all_bindings(self) -> Dict[str, AgentPolicyBinding]:
        """Get all agent-policy bindings.

        Returns:
            Copy of bindings dictionary.
        """
        return dict(self._bindings)

    def available_policy_ids(self) -> Iterable[str]:
        """Get available policy IDs from ActorService.

        Returns:
            Iterable of policy IDs.
        """
        return self._actor_service.available_actor_ids()

    # ------------------------------------------------------------------
    # Action Selection (Paradigm-Aware)
    # ------------------------------------------------------------------

    def select_action(
        self,
        agent_id: str,
        snapshot: StepSnapshot,
    ) -> Optional[int]:
        """Select action for a specific agent (Sequential/AEC mode).

        Args:
            agent_id: The agent needing an action.
            snapshot: Current step state.

        Returns:
            The action to take, or None to abstain.
        """
        binding = self._bindings.get(agent_id)

        if binding is None:
            # Fallback to legacy ActorService for unbound agents
            self._logger.debug(
                f"No binding for agent '{agent_id}', using legacy ActorService"
            )
            return self._actor_service.select_action(snapshot)

        # Get the actor for this agent's policy
        actor = self._get_actor(binding.policy_id)
        if actor is None:
            self._logger.warning(
                f"Policy '{binding.policy_id}' not found for agent '{agent_id}'"
            )
            return None

        return actor.select_action(snapshot)

    def select_actions(
        self,
        observations: Dict[str, Any],
        snapshots: Dict[str, StepSnapshot],
    ) -> Dict[str, Optional[int]]:
        """Select actions for all agents (Simultaneous/POSG mode).

        Args:
            observations: Dict mapping agent_id to observation.
            snapshots: Dict mapping agent_id to StepSnapshot.

        Returns:
            Dict mapping agent_id to action (or None).
        """
        actions: Dict[str, Optional[int]] = {}

        for agent_id, snapshot in snapshots.items():
            actions[agent_id] = self.select_action(agent_id, snapshot)

        return actions

    # ------------------------------------------------------------------
    # Step Notification
    # ------------------------------------------------------------------

    def notify_step(
        self,
        agent_id: str,
        snapshot: StepSnapshot,
    ) -> None:
        """Notify the appropriate policy of a step result.

        Args:
            agent_id: The agent that took the step.
            snapshot: The step result.
        """
        binding = self._bindings.get(agent_id)

        if binding is None:
            self._actor_service.notify_step(snapshot)
            return

        actor = self._get_actor(binding.policy_id)
        if actor is not None:
            actor.on_step(snapshot)

    def notify_steps(
        self,
        snapshots: Dict[str, StepSnapshot],
    ) -> None:
        """Notify all agents of their step results (Simultaneous mode).

        Args:
            snapshots: Dict mapping agent_id to StepSnapshot.
        """
        for agent_id, snapshot in snapshots.items():
            self.notify_step(agent_id, snapshot)

    def notify_episode_end(
        self,
        agent_id: str,
        summary: EpisodeSummary,
    ) -> None:
        """Notify the appropriate policy of episode end.

        Args:
            agent_id: The agent whose episode ended.
            summary: Episode summary information.
        """
        binding = self._bindings.get(agent_id)

        if binding is None:
            self._actor_service.notify_episode_end(summary)
            return

        actor = self._get_actor(binding.policy_id)
        if actor is not None:
            actor.on_episode_end(summary)

    def notify_all_episode_end(
        self,
        summaries: Dict[str, EpisodeSummary],
    ) -> None:
        """Notify all agents of episode end.

        Args:
            summaries: Dict mapping agent_id to EpisodeSummary.
        """
        for agent_id, summary in summaries.items():
            self.notify_episode_end(agent_id, summary)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all bindings for a new session."""
        self._bindings.clear()
        self._agent_ids.clear()
        self._paradigm = SteppingParadigm.SINGLE_AGENT
        self._logger.debug("PolicyMappingService reset")

    # ------------------------------------------------------------------
    # Convenience Properties
    # ------------------------------------------------------------------

    def is_multi_agent(self) -> bool:
        """Check if we're in multi-agent mode.

        Returns:
            True if more than one agent is configured.
        """
        return len(self._agent_ids) > 1

    @property
    def paradigm(self) -> SteppingParadigm:
        """Get the current stepping paradigm."""
        return self._paradigm

    @property
    def agent_ids(self) -> List[str]:
        """Get list of configured agent IDs."""
        return list(self._agent_ids)

    @property
    def actor_service(self) -> ActorService:
        """Get the underlying ActorService."""
        return self._actor_service

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_actor(self, policy_id: str) -> Optional[Actor]:
        """Get an Actor by ID from the underlying ActorService.

        Args:
            policy_id: The policy/actor ID.

        Returns:
            The Actor if found, otherwise None.
        """
        # Access internal _actors dict (composition pattern)
        return self._actor_service._actors.get(policy_id)


__all__ = [
    "AgentPolicyBinding",
    "PolicyMappingService",
]
