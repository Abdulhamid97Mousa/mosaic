"""PettingZoo environment wrapper with unified interface.

This module provides a wrapper that unifies AEC and Parallel API environments
into a common interface suitable for the Mosaic GUI.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import PettingZooConfig


@dataclass
class StepResult:
    """Result of an environment step.

    For AEC environments, this contains single-agent data.
    For Parallel environments, observations/rewards/etc are dicts.
    """

    observations: Union[Any, Dict[str, Any]]
    rewards: Union[float, Dict[str, float]]
    terminations: Union[bool, Dict[str, bool]]
    truncations: Union[bool, Dict[str, bool]]
    infos: Union[Dict[str, Any], Dict[str, Dict[str, Any]]]
    render_frame: Optional[np.ndarray] = None
    current_agent: Optional[str] = None
    action_mask: Optional[np.ndarray] = None


def create_aec_env(config: PettingZooConfig):
    """Create an AEC (turn-based) environment.

    Args:
        config: PettingZoo configuration

    Returns:
        AECEnv instance
    """
    module_path = f"pettingzoo.{config.family}"
    module = importlib.import_module(module_path)

    # Get the env function (e.g., chess_v6.env)
    env_module = getattr(module, config.env_id.replace(f"_{config.env_id.split('_')[-1]}", ""))
    if hasattr(env_module, "env"):
        env_fn = env_module.env
    else:
        # For some envs like tictactoe_v3, the module itself has .env
        env_fn = getattr(module, config.env_id.split("_v")[0]).env

    kwargs = {"render_mode": config.render_mode, **config.env_kwargs}
    return env_fn(**kwargs)


def create_parallel_env(config: PettingZooConfig):
    """Create a Parallel (simultaneous) environment.

    Args:
        config: PettingZoo configuration

    Returns:
        ParallelEnv instance
    """
    module_path = f"pettingzoo.{config.family}"
    module = importlib.import_module(module_path)

    # Get the parallel_env function
    env_name = config.env_id.split("_v")[0]
    env_module = getattr(module, env_name, None)

    if env_module and hasattr(env_module, "parallel_env"):
        env_fn = env_module.parallel_env
    else:
        # Fallback: try to import directly
        full_module = importlib.import_module(f"{module_path}.{env_name}")
        env_fn = full_module.parallel_env

    kwargs = {"render_mode": config.render_mode, **config.env_kwargs}
    if config.max_cycles:
        kwargs["max_cycles"] = config.max_cycles

    return env_fn(**kwargs)


class PettingZooWrapper:
    """Unified wrapper for PettingZoo AEC and Parallel environments.

    This wrapper provides a consistent interface regardless of the underlying
    API type, making it easier to integrate into the Mosaic GUI.
    """

    def __init__(self, config: PettingZooConfig, is_parallel: bool = False):
        """Initialize the wrapper.

        Args:
            config: Environment configuration
            is_parallel: Whether to use Parallel API (vs AEC)
        """
        self._config = config
        self._is_parallel = is_parallel
        self._env = None
        self._agents: List[str] = []
        self._current_agent: Optional[str] = None
        self._step_count: int = 0
        self._episode_rewards: Dict[str, float] = {}
        self._terminated_agents: set[str] = set()

    @property
    def is_parallel(self) -> bool:
        """Check if using Parallel API."""
        return self._is_parallel

    @property
    def agents(self) -> List[str]:
        """Get list of active agent names."""
        if self._env is None:
            return []
        return list(self._env.agents) if hasattr(self._env, "agents") else []

    @property
    def possible_agents(self) -> List[str]:
        """Get list of all possible agent names."""
        if self._env is None:
            return []
        return list(self._env.possible_agents)

    @property
    def current_agent(self) -> Optional[str]:
        """Get current agent (for AEC mode)."""
        return self._current_agent

    @property
    def num_agents(self) -> int:
        """Get number of agents."""
        return len(self.possible_agents)

    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_count

    @property
    def episode_rewards(self) -> Dict[str, float]:
        """Get cumulative episode rewards per agent."""
        return self._episode_rewards.copy()

    def make(self) -> None:
        """Create the underlying environment."""
        if self._is_parallel:
            self._env = create_parallel_env(self._config)
        else:
            self._env = create_aec_env(self._config)

    def reset(self, seed: Optional[int] = None) -> StepResult:
        """Reset the environment.

        Args:
            seed: Optional random seed

        Returns:
            Initial step result
        """
        if self._env is None:
            self.make()

        self._step_count = 0
        self._terminated_agents.clear()

        actual_seed = seed if seed is not None else self._config.seed

        if self._is_parallel:
            observations, infos = self._env.reset(seed=actual_seed)
            self._agents = list(self._env.agents)
            self._current_agent = None
            self._episode_rewards = {agent: 0.0 for agent in self._agents}

            return StepResult(
                observations=observations,
                rewards={agent: 0.0 for agent in self._agents},
                terminations={agent: False for agent in self._agents},
                truncations={agent: False for agent in self._agents},
                infos=infos,
                render_frame=self._try_render(),
                current_agent=None,
            )
        else:
            # AEC API
            self._env.reset(seed=actual_seed)
            self._agents = list(self._env.agents)
            self._current_agent = self._env.agent_selection
            self._episode_rewards = {agent: 0.0 for agent in self.possible_agents}

            observation, reward, termination, truncation, info = self._env.last()
            action_mask = info.get("action_mask") if isinstance(info, dict) else None

            return StepResult(
                observations=observation,
                rewards=reward,
                terminations=termination,
                truncations=truncation,
                infos=info,
                render_frame=self._try_render(),
                current_agent=self._current_agent,
                action_mask=action_mask,
            )

    def step(self, action: Union[Any, Dict[str, Any]]) -> StepResult:
        """Execute action(s) in the environment.

        Args:
            action: Single action (AEC) or dict of actions (Parallel)

        Returns:
            Step result
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._step_count += 1

        if self._is_parallel:
            return self._step_parallel(action)
        else:
            return self._step_aec(action)

    def _step_parallel(self, actions: Dict[str, Any]) -> StepResult:
        """Step in Parallel mode."""
        observations, rewards, terminations, truncations, infos = self._env.step(actions)

        # Update episode rewards
        for agent, reward in rewards.items():
            if agent in self._episode_rewards:
                self._episode_rewards[agent] += reward

        # Track terminated agents
        for agent, terminated in terminations.items():
            if terminated:
                self._terminated_agents.add(agent)

        return StepResult(
            observations=observations,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            infos=infos,
            render_frame=self._try_render(),
            current_agent=None,
        )

    def _step_aec(self, action: Any) -> StepResult:
        """Step in AEC mode."""
        # Execute action for current agent
        self._env.step(action)

        # Get next agent
        if self._env.agents:  # If there are still active agents
            self._current_agent = self._env.agent_selection
            observation, reward, termination, truncation, info = self._env.last()

            # Update episode rewards
            if self._current_agent in self._episode_rewards:
                self._episode_rewards[self._current_agent] += reward

            # Track terminated agents
            if termination:
                self._terminated_agents.add(self._current_agent)

            action_mask = info.get("action_mask") if isinstance(info, dict) else None

            return StepResult(
                observations=observation,
                rewards=reward,
                terminations=termination,
                truncations=truncation,
                infos=info,
                render_frame=self._try_render(),
                current_agent=self._current_agent,
                action_mask=action_mask,
            )
        else:
            # Episode is done
            return StepResult(
                observations=None,
                rewards=0.0,
                terminations=True,
                truncations=False,
                infos={},
                render_frame=self._try_render(),
                current_agent=None,
            )

    def get_action_space(self, agent: Optional[str] = None):
        """Get action space for an agent.

        Args:
            agent: Agent name (uses current agent if None for AEC)

        Returns:
            Gymnasium Space
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized")

        if agent is None:
            if self._is_parallel:
                raise ValueError("Agent name required for Parallel environments")
            agent = self._current_agent

        return self._env.action_space(agent)

    def get_observation_space(self, agent: Optional[str] = None):
        """Get observation space for an agent.

        Args:
            agent: Agent name (uses current agent if None for AEC)

        Returns:
            Gymnasium Space
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized")

        if agent is None:
            if self._is_parallel:
                raise ValueError("Agent name required for Parallel environments")
            agent = self._current_agent

        return self._env.observation_space(agent)

    def sample_action(self, agent: Optional[str] = None) -> Any:
        """Sample a random action for an agent.

        Args:
            agent: Agent name (uses current agent if None for AEC)

        Returns:
            Sampled action
        """
        return self.get_action_space(agent).sample()

    def sample_actions(self) -> Dict[str, Any]:
        """Sample random actions for all active agents.

        Returns:
            Dict mapping agent names to sampled actions
        """
        return {agent: self.sample_action(agent) for agent in self.agents}

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", else None
        """
        return self._try_render()

    def _try_render(self) -> Optional[np.ndarray]:
        """Attempt to render, returning None on failure."""
        if self._env is None:
            return None
        try:
            result = self._env.render()
            if isinstance(result, np.ndarray):
                return result
            return None
        except Exception:
            return None

    def close(self) -> None:
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def is_done(self) -> bool:
        """Check if episode is complete."""
        if self._env is None:
            return True
        return len(self.agents) == 0

    def agent_iter(self):
        """Generator for AEC agent iteration.

        Yields agent names in turn order. Use with AEC environments.
        """
        if self._is_parallel:
            raise RuntimeError("agent_iter() is only for AEC environments")

        if self._env is None:
            raise RuntimeError("Environment not initialized")

        for agent in self._env.agent_iter():
            self._current_agent = agent
            yield agent

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def get_env_info(config: PettingZooConfig, is_parallel: bool = False) -> Dict[str, Any]:
    """Get information about an environment without fully initializing it.

    Args:
        config: Environment configuration
        is_parallel: Whether to use Parallel API

    Returns:
        Dict with environment metadata
    """
    wrapper = PettingZooWrapper(config, is_parallel)
    wrapper.make()

    try:
        wrapper.reset()
        info = {
            "env_id": config.env_id,
            "family": config.family,
            "is_parallel": is_parallel,
            "num_agents": wrapper.num_agents,
            "possible_agents": wrapper.possible_agents,
            "action_spaces": {
                agent: str(wrapper.get_action_space(agent))
                for agent in wrapper.possible_agents
            },
            "observation_spaces": {
                agent: str(wrapper.get_observation_space(agent))
                for agent in wrapper.possible_agents
            },
        }
        return info
    finally:
        wrapper.close()
