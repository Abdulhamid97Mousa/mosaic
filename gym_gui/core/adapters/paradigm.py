"""Paradigm-specific adapter abstractions for multi-paradigm RL orchestration.

This module provides:
- ParadigmAdapter: ABC for paradigm-aware stepping behavior
- Concrete adapters: SingleAgentAdapter, SequentialAdapter, SimultaneousAdapter

The ParadigmAdapter bridges between the GUI/orchestrator and paradigm-specific
environments (Gymnasium, PettingZoo AEC, PettingZoo Parallel).

See Also:
    - :doc:`/documents/architecture/paradigms` for stepping paradigm details
    - :doc:`/documents/architecture/operators/concept` for operator architecture
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

from gym_gui.core.enums import SteppingParadigm

if TYPE_CHECKING:
    from gym_gui.core.adapters.base import AdapterStep


@dataclass(slots=True)
class ParadigmStepResult:
    """Unified step result for all paradigms.

    This normalizes results from different paradigms:
    - Single-agent: One observation, one reward
    - Sequential (AEC): Per-agent observation/reward for current agent
    - Simultaneous (POSG): Dict of observations/rewards for all agents

    Attributes:
        observations: Mapping from agent_id to observation.
            For single-agent, uses key "agent_0".
        rewards: Mapping from agent_id to reward.
        terminations: Mapping from agent_id to terminated flag.
        truncations: Mapping from agent_id to truncated flag.
        infos: Mapping from agent_id to info dict.
        current_agent: The agent that just acted (Sequential mode).
        all_done: Whether the episode is complete for all agents.
        adapter_steps: Raw AdapterStep results (if available).
    """

    observations: Dict[str, Any] = field(default_factory=dict)
    rewards: Dict[str, float] = field(default_factory=dict)
    terminations: Dict[str, bool] = field(default_factory=dict)
    truncations: Dict[str, bool] = field(default_factory=dict)
    infos: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    current_agent: Optional[str] = None
    all_done: bool = False
    adapter_steps: Dict[str, "AdapterStep[Any]"] = field(default_factory=dict)

    def is_agent_done(self, agent_id: str) -> bool:
        """Check if a specific agent's episode is done."""
        return self.terminations.get(agent_id, False) or self.truncations.get(agent_id, False)


class ParadigmAdapter(ABC):
    """Abstract base class for paradigm-specific stepping behavior.

    ParadigmAdapter bridges between Mosaic's GUI/orchestrator and paradigm-specific
    workers. It abstracts:
    1. Which agents need actions at any given time
    2. How to execute a step (single action vs. joint action dict)
    3. How to normalize results across paradigms

    Subclasses implement paradigm-specific logic:
    - SingleAgentAdapter: Gymnasium-style single agent
    - SequentialAdapter: PettingZoo AEC-style turn-based
    - SimultaneousAdapter: PettingZoo Parallel / RLlib POSG-style
    Example:
        >>> adapter = get_paradigm_adapter(env)
        >>> while not adapter.is_done():
        ...     agents = adapter.get_agents_to_act()
        ...     actions = {a: policy(a, adapter.get_observation(a)) for a in agents}
        ...     result = adapter.step(actions)

    See Also:
        - :doc:`/documents/architecture/paradigms` for paradigm details
    """

    @property
    @abstractmethod
    def paradigm(self) -> SteppingParadigm:
        """The stepping paradigm this adapter implements."""
        ...

    @property
    @abstractmethod
    def agent_ids(self) -> Sequence[str]:
        """All agent identifiers in this environment.

        For single-agent environments, returns ["agent_0"].
        For multi-agent environments, returns all agent IDs.
        """
        ...

    @abstractmethod
    def get_agents_to_act(self) -> List[str]:
        """Return agents that need actions NOW.

        Returns:
            List of agent IDs that require actions in the current step.

            - SINGLE_AGENT: ["agent_0"]
            - SEQUENTIAL: [current_agent_id] (one agent per step)
            - SIMULTANEOUS: [all active agent IDs]
            - HIERARCHICAL: [agents with pending goals]
        """
        ...

    @abstractmethod
    def get_observation(self, agent_id: str) -> Any:
        """Get the current observation for a specific agent.

        Args:
            agent_id: The agent to get observation for.

        Returns:
            The observation for the specified agent.

        Raises:
            KeyError: If agent_id is not valid.
        """
        ...

    @abstractmethod
    def get_observations(self, agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get observations for multiple agents.

        Args:
            agent_ids: List of agent IDs. If None, returns all active agents.

        Returns:
            Dict mapping agent_id to observation.
        """
        ...

    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> ParadigmStepResult:
        """Execute paradigm-appropriate step.

        Args:
            actions: Dict mapping agent_id to action.
                - SINGLE_AGENT: {"agent_0": action}
                - SEQUENTIAL: {current_agent: action}
                - SIMULTANEOUS: {agent_id: action for all active agents}

        Returns:
            ParadigmStepResult with normalized observations, rewards, etc.
        """
        ...

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ParadigmStepResult:
        """Reset the environment and return initial observations.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional reset options dict.

        Returns:
            ParadigmStepResult with initial observations (rewards=0, done=False).
        """
        ...

    @abstractmethod
    def is_done(self) -> bool:
        """Check if the episode is complete for all agents."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        ...

    # ------------------------------------------------------------------
    # Optional lifecycle hooks (subclasses may override)
    # ------------------------------------------------------------------

    def get_info(self, agent_id: str) -> Dict[str, Any]:
        """Get the info dict for a specific agent.

        Default implementation returns empty dict. Subclasses should override.
        """
        return {}

    def get_infos(self, agent_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Get info dicts for multiple agents.

        Default implementation calls get_info for each agent.
        """
        ids = agent_ids if agent_ids is not None else list(self.agent_ids)
        return {agent_id: self.get_info(agent_id) for agent_id in ids}

    def render(self) -> Any:
        """Render the environment.

        Default implementation returns None. Subclasses should override.
        """
        return None

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def is_single_agent(self) -> bool:
        """Check if this is a single-agent environment."""
        return self.paradigm == SteppingParadigm.SINGLE_AGENT

    def is_sequential(self) -> bool:
        """Check if this is a sequential (AEC) environment."""
        return self.paradigm == SteppingParadigm.SEQUENTIAL

    def is_simultaneous(self) -> bool:
        """Check if this is a simultaneous (POSG) environment."""
        return self.paradigm == SteppingParadigm.SIMULTANEOUS

    def is_hierarchical(self) -> bool:
        """Check if this is a hierarchical environment."""
        return self.paradigm == SteppingParadigm.HIERARCHICAL

    def num_agents(self) -> int:
        """Return the number of agents in the environment."""
        return len(self.agent_ids)


# =============================================================================
# Concrete Paradigm Adapters
# =============================================================================


class SingleAgentParadigmAdapter(ParadigmAdapter):
    """Paradigm adapter for single-agent Gymnasium environments.

    Wraps a standard Gymnasium environment with the ParadigmAdapter interface.
    Uses "agent_0" as the canonical agent ID.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("CartPole-v1")
        >>> adapter = SingleAgentParadigmAdapter(env)
        >>> result = adapter.reset()
        >>> while not adapter.is_done():
        ...     action = policy(result.observations["agent_0"])
        ...     result = adapter.step({"agent_0": action})
    """

    AGENT_ID = "agent_0"

    def __init__(self, env: Any) -> None:
        """Initialize with a Gymnasium environment.

        Args:
            env: A Gymnasium-compatible environment.
        """
        self._env = env
        self._current_obs: Any = None
        self._current_info: Dict[str, Any] = {}
        self._done = False

    @property
    def paradigm(self) -> SteppingParadigm:
        return SteppingParadigm.SINGLE_AGENT

    @property
    def agent_ids(self) -> Sequence[str]:
        return (self.AGENT_ID,)

    def get_agents_to_act(self) -> List[str]:
        if self._done:
            return []
        return [self.AGENT_ID]

    def get_observation(self, agent_id: str) -> Any:
        if agent_id != self.AGENT_ID:
            raise KeyError(f"Unknown agent '{agent_id}'. Single-agent env uses '{self.AGENT_ID}'.")
        return self._current_obs

    def get_observations(self, agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        if agent_ids is not None and self.AGENT_ID not in agent_ids:
            return {}
        return {self.AGENT_ID: self._current_obs}

    def get_info(self, agent_id: str) -> Dict[str, Any]:
        if agent_id != self.AGENT_ID:
            raise KeyError(f"Unknown agent '{agent_id}'.")
        return self._current_info

    def step(self, actions: Dict[str, Any]) -> ParadigmStepResult:
        action = actions.get(self.AGENT_ID)
        if action is None:
            raise ValueError(f"Missing action for '{self.AGENT_ID}'")

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._current_obs = obs
        self._current_info = info if isinstance(info, dict) else {}
        self._done = terminated or truncated

        return ParadigmStepResult(
            observations={self.AGENT_ID: obs},
            rewards={self.AGENT_ID: float(reward)},
            terminations={self.AGENT_ID: terminated},
            truncations={self.AGENT_ID: truncated},
            infos={self.AGENT_ID: self._current_info},
            current_agent=self.AGENT_ID,
            all_done=self._done,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ParadigmStepResult:
        reset_kwargs: Dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        if options is not None:
            reset_kwargs["options"] = options

        obs, info = self._env.reset(**reset_kwargs)
        self._current_obs = obs
        self._current_info = info if isinstance(info, dict) else {}
        self._done = False

        return ParadigmStepResult(
            observations={self.AGENT_ID: obs},
            rewards={self.AGENT_ID: 0.0},
            terminations={self.AGENT_ID: False},
            truncations={self.AGENT_ID: False},
            infos={self.AGENT_ID: self._current_info},
            current_agent=self.AGENT_ID,
            all_done=False,
        )

    def is_done(self) -> bool:
        return self._done

    def close(self) -> None:
        self._env.close()

    def render(self) -> Any:
        return self._env.render()


class SequentialParadigmAdapter(ParadigmAdapter):
    """Paradigm adapter for sequential (AEC) multi-agent environments.

    Wraps a PettingZoo AEC environment with the ParadigmAdapter interface.
    Agents take turns acting one at a time.

    Example:
        >>> from pettingzoo.classic import chess_v6
        >>> env = chess_v6.env()
        >>> adapter = SequentialParadigmAdapter(env)
        >>> result = adapter.reset()
        >>> while not adapter.is_done():
        ...     agents = adapter.get_agents_to_act()  # Returns [current_agent]
        ...     action = policy(agents[0], result.observations[agents[0]])
        ...     result = adapter.step({agents[0]: action})
    """

    def __init__(self, env: Any) -> None:
        """Initialize with a PettingZoo AEC environment.

        Args:
            env: A PettingZoo AEC-compatible environment.
        """
        self._env = env
        self._all_done = False

    @property
    def paradigm(self) -> SteppingParadigm:
        return SteppingParadigm.SEQUENTIAL

    @property
    def agent_ids(self) -> Sequence[str]:
        return tuple(self._env.possible_agents)

    def get_agents_to_act(self) -> List[str]:
        if self._all_done or not self._env.agents:
            return []
        # AEC: only current agent needs to act
        agent = self._env.agent_selection
        return [agent] if agent else []

    def get_observation(self, agent_id: str) -> Any:
        # In AEC, use observe() or last() depending on implementation
        if hasattr(self._env, "observe"):
            return self._env.observe(agent_id)
        # Fallback: use last() for current agent
        if agent_id == self._env.agent_selection:
            obs, _, _, _, _ = self._env.last()
            return obs
        raise KeyError(f"Cannot get observation for non-current agent '{agent_id}' in AEC mode.")

    def get_observations(self, agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        if agent_ids is None:
            agent_ids = self.get_agents_to_act()
        return {agent_id: self.get_observation(agent_id) for agent_id in agent_ids}

    def get_info(self, agent_id: str) -> Dict[str, Any]:
        if agent_id == self._env.agent_selection:
            _, _, _, _, info = self._env.last()
            return info if isinstance(info, dict) else {}
        return {}

    def step(self, actions: Dict[str, Any]) -> ParadigmStepResult:
        current_agent = self._env.agent_selection
        action = actions.get(current_agent)

        # Check if agent is terminated/truncated (pass None)
        obs, reward, terminated, truncated, info = self._env.last()
        if terminated or truncated:
            action = None

        self._env.step(action)

        # Get new state after step
        if self._env.agents:
            new_agent = self._env.agent_selection
            new_obs, new_reward, new_terminated, new_truncated, new_info = self._env.last()
        else:
            self._all_done = True
            new_agent = current_agent
            new_obs, new_reward, new_terminated, new_truncated, new_info = obs, reward, True, False, info

        return ParadigmStepResult(
            observations={new_agent: new_obs} if new_agent else {},
            rewards={current_agent: float(reward)},
            terminations={current_agent: terminated},
            truncations={current_agent: truncated},
            infos={current_agent: info if isinstance(info, dict) else {}},
            current_agent=new_agent,
            all_done=self._all_done or not self._env.agents,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ParadigmStepResult:
        reset_kwargs: Dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        if options is not None:
            reset_kwargs["options"] = options

        self._env.reset(**reset_kwargs)
        self._all_done = False

        current_agent = self._env.agent_selection
        obs, _, _, _, info = self._env.last()

        return ParadigmStepResult(
            observations={current_agent: obs},
            rewards={agent: 0.0 for agent in self._env.agents},
            terminations={agent: False for agent in self._env.agents},
            truncations={agent: False for agent in self._env.agents},
            infos={current_agent: info if isinstance(info, dict) else {}},
            current_agent=current_agent,
            all_done=False,
        )

    def is_done(self) -> bool:
        return self._all_done or not self._env.agents

    def close(self) -> None:
        self._env.close()

    def render(self) -> Any:
        return self._env.render()


class SimultaneousParadigmAdapter(ParadigmAdapter):
    """Paradigm adapter for simultaneous (POSG) multi-agent environments.

    Wraps a PettingZoo Parallel or RLlib MultiAgentEnv with the ParadigmAdapter interface.
    All agents act simultaneously each step.

    Example:
        >>> from pettingzoo.butterfly import pistonball_v6
        >>> env = pistonball_v6.parallel_env()
        >>> adapter = SimultaneousParadigmAdapter(env)
        >>> result = adapter.reset()
        >>> while not adapter.is_done():
        ...     agents = adapter.get_agents_to_act()  # Returns all agents
        ...     actions = {a: policy(a, result.observations[a]) for a in agents}
        ...     result = adapter.step(actions)
    """

    def __init__(self, env: Any) -> None:
        """Initialize with a PettingZoo Parallel or RLlib MultiAgentEnv.

        Args:
            env: A parallel multi-agent environment.
        """
        self._env = env
        self._current_obs: Dict[str, Any] = {}
        self._current_infos: Dict[str, Dict[str, Any]] = {}
        self._terminations: Dict[str, bool] = {}
        self._truncations: Dict[str, bool] = {}
        self._all_done = False

    @property
    def paradigm(self) -> SteppingParadigm:
        return SteppingParadigm.SIMULTANEOUS

    @property
    def agent_ids(self) -> Sequence[str]:
        return tuple(self._env.possible_agents)

    def get_agents_to_act(self) -> List[str]:
        if self._all_done:
            return []
        # In parallel mode, all active (not done) agents act
        return [
            agent
            for agent in self._env.agents
            if not self._terminations.get(agent, False) and not self._truncations.get(agent, False)
        ]

    def get_observation(self, agent_id: str) -> Any:
        if agent_id not in self._current_obs:
            raise KeyError(f"No observation for agent '{agent_id}'")
        return self._current_obs[agent_id]

    def get_observations(self, agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        if agent_ids is None:
            return dict(self._current_obs)
        return {agent_id: self._current_obs[agent_id] for agent_id in agent_ids if agent_id in self._current_obs}

    def get_info(self, agent_id: str) -> Dict[str, Any]:
        return self._current_infos.get(agent_id, {})

    def get_infos(self, agent_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        if agent_ids is None:
            return dict(self._current_infos)
        return {agent_id: self._current_infos.get(agent_id, {}) for agent_id in agent_ids}

    def step(self, actions: Dict[str, Any]) -> ParadigmStepResult:
        obs, rewards, terminations, truncations, infos = self._env.step(actions)

        self._current_obs = obs
        self._current_infos = {k: v if isinstance(v, dict) else {} for k, v in infos.items()}
        self._terminations = terminations
        self._truncations = truncations

        # Check if all agents are done
        # PettingZoo parallel uses "__all__" key or empty agents list
        if "__all__" in terminations:
            self._all_done = terminations["__all__"]
        else:
            self._all_done = not self._env.agents

        return ParadigmStepResult(
            observations=obs,
            rewards={k: float(v) for k, v in rewards.items()},
            terminations=terminations,
            truncations=truncations,
            infos=self._current_infos,
            current_agent=None,  # No single current agent in simultaneous mode
            all_done=self._all_done,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ParadigmStepResult:
        reset_kwargs: Dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        if options is not None:
            reset_kwargs["options"] = options

        obs, infos = self._env.reset(**reset_kwargs)
        self._current_obs = obs
        self._current_infos = {k: v if isinstance(v, dict) else {} for k, v in infos.items()}
        self._terminations = {agent: False for agent in self._env.agents}
        self._truncations = {agent: False for agent in self._env.agents}
        self._all_done = False

        return ParadigmStepResult(
            observations=obs,
            rewards={agent: 0.0 for agent in self._env.agents},
            terminations=self._terminations,
            truncations=self._truncations,
            infos=self._current_infos,
            current_agent=None,
            all_done=False,
        )

    def is_done(self) -> bool:
        return self._all_done

    def close(self) -> None:
        self._env.close()

    def render(self) -> Any:
        return self._env.render()


# =============================================================================
# Factory Function
# =============================================================================


def create_paradigm_adapter(
    env: Any,
    paradigm: Optional[SteppingParadigm] = None,
) -> ParadigmAdapter:
    """Create a ParadigmAdapter for the given environment.

    Args:
        env: The environment to wrap.
        paradigm: Optional explicit paradigm. If None, auto-detected.

    Returns:
        An appropriate ParadigmAdapter subclass instance.

    Raises:
        ValueError: If paradigm cannot be determined.
    """
    # Auto-detect paradigm if not specified
    if paradigm is None:
        paradigm = _detect_paradigm(env)

    if paradigm == SteppingParadigm.SINGLE_AGENT:
        return SingleAgentParadigmAdapter(env)
    elif paradigm == SteppingParadigm.SEQUENTIAL:
        return SequentialParadigmAdapter(env)
    elif paradigm == SteppingParadigm.SIMULTANEOUS:
        return SimultaneousParadigmAdapter(env)
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")


def _detect_paradigm(env: Any) -> SteppingParadigm:
    """Auto-detect the stepping paradigm from environment type.

    Args:
        env: The environment to inspect.

    Returns:
        The detected SteppingParadigm.
    """
    # Check for PettingZoo AEC (has agent_iter and last methods)
    if hasattr(env, "agent_iter") and hasattr(env, "last"):
        return SteppingParadigm.SEQUENTIAL

    # Check for PettingZoo Parallel (has possible_agents but step takes dict)
    if hasattr(env, "possible_agents") and hasattr(env, "agents"):
        # Parallel envs don't have agent_iter
        if not hasattr(env, "agent_iter"):
            return SteppingParadigm.SIMULTANEOUS

    # Default to single-agent (standard Gymnasium)
    return SteppingParadigm.SINGLE_AGENT


__all__ = [
    "ParadigmAdapter",
    "ParadigmStepResult",
    "SingleAgentParadigmAdapter",
    "SequentialParadigmAdapter",
    "SimultaneousParadigmAdapter",
    "create_paradigm_adapter",
]
