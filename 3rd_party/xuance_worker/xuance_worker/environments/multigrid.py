"""MultiGrid environment wrapper for XuanCe multi-agent training.

This module provides XuanCe-compatible wrappers for gym-multigrid environments
with flexible training mode support: competitive, cooperative, or independent.

gym-multigrid Repository: https://github.com/ArnaudFickinger/gym-multigrid
Location: 3rd_party/gym-multigrid/

Training Modes:
---------------
- COMPETITIVE: Per-team policies (uses environment's team structure)
  → XuanCe uses runner_competition.py
  → Example: Soccer [1,1,2,2] → 2 policies (Red team, Blue team)

- COOPERATIVE: All agents share one policy
  → XuanCe uses runner_basic.py with parameter sharing
  → Example: Soccer [1,1,2,2] → 1 policy (all 4 agents share)

- INDEPENDENT: Each agent has its own policy
  → XuanCe uses runner_competition.py with n groups of 1
  → Example: Soccer [1,1,2,2] → 4 policies (one per agent)

Configuration:
--------------
    config.training_mode = "competitive"  # or "cooperative" or "independent"

Example:
--------
    config = SimpleNamespace(
        env_name="multigrid",
        env_id="soccer",
        env_seed=42,
        training_mode="competitive",  # Per-team policies
    )
    env = MultiGrid_Env(config)

    # For cooperative (shared policy):
    config.training_mode = "cooperative"
    env = MultiGrid_Env(config)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from gymnasium import spaces

_logger = logging.getLogger(__name__)

try:
    from xuance.environment import RawMultiAgentEnv
except ImportError:
    RawMultiAgentEnv = object  # Fallback for type hints

try:
    from gym_gui.core.wrappers import ReproducibleMultiGridWrapper
except ImportError:
    ReproducibleMultiGridWrapper = None

# FastLane integration for visualization during training
try:
    from xuance_worker.fastlane import maybe_wrap_env, is_fastlane_enabled
except ImportError:
    maybe_wrap_env = None
    is_fastlane_enabled = lambda: False


# =============================================================================
# Gymnasium Compatibility Wrapper
# =============================================================================

# Import gymnasium for proper Env inheritance
try:
    import gymnasium as _gymnasium
    _HAS_GYMNASIUM = True
except ImportError:
    _gymnasium = None
    _HAS_GYMNASIUM = False


class GymToGymnasiumWrapper(_gymnasium.Env if _HAS_GYMNASIUM else object):
    """Wraps old Gym API (4-tuple step) to new Gymnasium API (5-tuple step).

    gym-multigrid uses the old Gym API:
        step() -> (obs, rewards, done, info)  # 4-tuple

    FastLane expects the new Gymnasium API:
        step() -> (obs, reward, terminated, truncated, info)  # 5-tuple

    This wrapper converts between the two APIs.

    IMPORTANT: Inherits from gymnasium.Env so it passes isinstance() checks
    required by gymnasium.Wrapper (used by FastLaneTelemetryWrapper).
    """

    # Gymnasium metadata
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, env: Any) -> None:
        self._env = env
        self._step_count = 0
        # Get max_steps from env if available
        self._max_steps = getattr(env, 'max_steps', 10000)

        # Copy spaces from wrapped env (required by gymnasium.Env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Set render_mode if available
        self.render_mode = getattr(env, 'render_mode', None)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped environment."""
        # Avoid infinite recursion for _env
        if name == '_env':
            raise AttributeError(name)
        return getattr(self._env, name)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset returning Gymnasium-style 2-tuple."""
        self._step_count = 0
        # Handle seed if the underlying env supports it
        if seed is not None and hasattr(self._env, 'seed'):
            self._env.seed(seed)
        obs = self._env.reset()
        return obs, {}

    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        """Step returning Gymnasium-style 5-tuple.

        Converts from old Gym 4-tuple:
            (obs, reward, done, info)
        To new Gymnasium 5-tuple:
            (obs, reward, terminated, truncated, info)
        """
        result = self._env.step(action)

        if len(result) == 4:
            obs, reward, done, info = result
            # Determine if terminated (episode end) or truncated (max steps)
            self._step_count += 1
            truncated = self._step_count >= self._max_steps and not done
            terminated = done
        else:
            # Already 5-tuple (shouldn't happen but handle gracefully)
            obs, reward, terminated, truncated, info = result

        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        """Render the environment."""
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        if hasattr(self._env, 'close'):
            self._env.close()

    @property
    def unwrapped(self) -> Any:
        """Return the base unwrapped environment."""
        return getattr(self._env, 'unwrapped', self._env)


# =============================================================================
# Training Mode Enumeration
# =============================================================================

class TrainingMode(str, Enum):
    """Training mode determines how agent policies are structured.

    COMPETITIVE: Per-team policies based on environment's team structure.
        - Soccer: 2 policies (Red team shares π_A, Blue team shares π_B)
        - Uses runner_competition.py in XuanCe
        - Best for: Adversarial games, team vs team scenarios

    COOPERATIVE: All agents share a single policy.
        - Soccer: 1 policy (all 4 agents share π)
        - Uses runner_basic.py with parameter sharing
        - Best for: Cooperative tasks, homogeneous agents

    INDEPENDENT: Each agent has its own separate policy.
        - Soccer: 4 policies (π_0, π_1, π_2, π_3)
        - Uses runner_competition.py with n groups of 1
        - Best for: Heterogeneous agents, individual learning
    """
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"
    INDEPENDENT = "independent"


# =============================================================================
# Environment Registry
# =============================================================================

# Lazy-loaded environment classes to avoid import errors
_ENV_CLASSES_CACHE: Dict[str, Optional[Type]] = {}


def _get_env_class(env_id: str) -> Optional[Type]:
    """Lazily load and cache environment classes.

    This prevents import errors if gym-multigrid is not installed.
    """
    if env_id in _ENV_CLASSES_CACHE:
        return _ENV_CLASSES_CACHE[env_id]

    env_cls = None

    # Try to import from gym-multigrid
    try:
        if env_id.lower() in ("soccer", "soccergame4henv10x15n2"):
            from gym_multigrid.envs import SoccerGame4HEnv10x15N2
            env_cls = SoccerGame4HEnv10x15N2

        elif env_id.lower() in ("collect", "collectgame4henv10x10n2"):
            from gym_multigrid.envs import CollectGame4HEnv10x10N2
            env_cls = CollectGame4HEnv10x10N2

    except ImportError as e:
        _logger.warning(f"Could not import gym-multigrid environment '{env_id}': {e}")

    _ENV_CLASSES_CACHE[env_id] = env_cls
    return env_cls


def get_available_environments() -> List[str]:
    """Return list of available MultiGrid environment IDs."""
    available = []
    for env_id in ["soccer", "collect"]:
        if _get_env_class(env_id) is not None:
            available.append(env_id)
    return available


# Environment metadata: describes team structure and recommended training modes
MULTIGRID_ENV_INFO = {
    "soccer": {
        "full_name": "SoccerGame4HEnv10x15N2",
        "description": "2v2 Soccer - score in opponent's goal",
        "num_agents": 4,
        "default_teams": [[0, 1], [2, 3]],  # Red vs Blue
        "team_names": ["red", "blue"],
        "recommended_mode": TrainingMode.COMPETITIVE,
        "zero_sum": True,
    },
    "collect": {
        "full_name": "CollectGame4HEnv10x10N2",
        "description": "3-player ball collection - free-for-all",
        "num_agents": 3,
        "default_teams": [[0], [1], [2]],  # Each agent is their own "team"
        "team_names": ["agent_0", "agent_1", "agent_2"],
        "recommended_mode": TrainingMode.INDEPENDENT,
        "zero_sum": True,
    },
}


# =============================================================================
# Main Environment Wrapper
# =============================================================================

class MultiGrid_Env(RawMultiAgentEnv):
    """XuanCe-compatible wrapper for gym-multigrid environments.

    This wrapper converts gym-multigrid's list-based multi-agent API to
    XuanCe's dict-based API, with flexible training mode support.

    Attributes:
        env: The wrapped gym-multigrid environment
        agents: List of agent IDs (e.g., ['agent_0', 'agent_1', ...])
        training_mode: How policies are structured (competitive/cooperative/independent)
        agent_groups: List of agent groups (structure depends on training_mode)
        observation_space: Dict mapping agent ID to observation space
        action_space: Dict mapping agent ID to action space

    Example:
        # Competitive (per-team policies)
        config = SimpleNamespace(env_id="soccer", training_mode="competitive")
        env = MultiGrid_Env(config)
        print(env.groups_info['num_groups'])  # 2 (Red team, Blue team)

        # Cooperative (shared policy)
        config = SimpleNamespace(env_id="soccer", training_mode="cooperative")
        env = MultiGrid_Env(config)
        print(env.groups_info['num_groups'])  # 1 (all agents share)

        # Independent (per-agent policies)
        config = SimpleNamespace(env_id="soccer", training_mode="independent")
        env = MultiGrid_Env(config)
        print(env.groups_info['num_groups'])  # 4 (one per agent)
    """

    def __init__(self, config: Any) -> None:
        """Initialize the MultiGrid environment wrapper.

        Args:
            config: Configuration object with attributes:
                - env_name: Environment family name (default: "multigrid")
                - env_id: Specific environment (e.g., "soccer", "collect")
                - env_seed: Random seed for reproducibility (optional)
                - render_mode: Render mode ("human", "rgb_array", or None)
                - training_mode: Policy structure - "competitive", "cooperative",
                                 or "independent" (default: "competitive")
                - custom_teams: Optional custom team assignments (list of lists)
        """
        super().__init__()

        # Extract configuration
        env_name = getattr(config, 'env_name', 'multigrid')
        env_id = getattr(config, 'env_id', 'soccer')
        env_seed = getattr(config, 'env_seed', None)
        self.render_mode = getattr(config, 'render_mode', None)

        # Parse training mode
        training_mode_str = getattr(config, 'training_mode', 'competitive')
        try:
            self.training_mode = TrainingMode(training_mode_str.lower())
        except ValueError:
            _logger.warning(
                f"Unknown training_mode '{training_mode_str}', "
                f"defaulting to 'competitive'"
            )
            self.training_mode = TrainingMode.COMPETITIVE

        # Custom team assignments (optional override)
        self._custom_teams: Optional[List[List[int]]] = getattr(
            config, 'custom_teams', None
        )

        # Store env_id for reference
        self.env_id = env_id.lower()

        # Create environment
        env_cls = _get_env_class(env_id)
        if env_cls is None:
            available = get_available_environments()
            raise ValueError(
                f"Unknown or unavailable MultiGrid environment: '{env_id}'. "
                f"Available: {available}. "
                f"Make sure gym-multigrid is installed."
            )

        self.env = env_cls()

        # Wrap with ReproducibleMultiGridWrapper for deterministic training
        if ReproducibleMultiGridWrapper is not None:
            self.env = ReproducibleMultiGridWrapper(self.env)
            _logger.debug("Applied ReproducibleMultiGridWrapper for deterministic training")

        # Seed the environment BEFORE wrapping with FastLane
        # (FastLane wrapper doesn't forward the seed method)
        if env_seed is not None:
            if hasattr(self.env, 'seed'):
                self.env.seed(env_seed)
            _logger.debug("Seeded environment with seed=%d", env_seed)

        # =====================================================================
        # IMPORTANT: Store env metadata BEFORE wrapping with FastLane
        # FastLane wrapper doesn't forward all attributes (agents, max_steps, etc.)
        # =====================================================================
        self._base_env_agents = self.env.agents  # gym-multigrid agent objects
        self._base_max_steps = getattr(self.env, 'max_steps', 10000)

        # Store scenario name for reference
        self.scenario_name = f"{env_name}.{env_id}"

        # Build agent list: ['agent_0', 'agent_1', ...]
        self.num_agents = len(self._base_env_agents)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Wrap with FastLane for visualization during training
        # NOTE: gym-multigrid uses old Gym API (4-tuple step return).
        # FastLane expects new Gymnasium API (5-tuple step return).
        # We wrap with GymToGymnasiumWrapper first to ensure compatibility.
        if maybe_wrap_env is not None and is_fastlane_enabled():
            # Apply Gymnasium compatibility wrapper before FastLane
            self.env = GymToGymnasiumWrapper(self.env)
            self.env = maybe_wrap_env(self.env)
            _logger.info("Applied FastLane wrapper for training visualization")

        # Build agent_groups based on training mode
        self.agent_groups = self._build_agent_groups()

        # Log configuration
        _logger.info(
            f"MultiGrid_Env initialized: env={env_id}, "
            f"training_mode={self.training_mode.value}, "
            f"num_agents={self.num_agents}, "
            f"num_groups={len(self.agent_groups)}"
        )

        # Build observation and action spaces (dict format for XuanCe)
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        self.observation_space = {
            agent: spaces.Box(
                low=0,
                high=255,
                shape=obs_space.shape,
                dtype=obs_space.dtype
            )
            for agent in self.agents
        }

        self.action_space = {
            agent: spaces.Discrete(act_space.n)
            for agent in self.agents
        }

        # State space (global state) - concatenated agent observations
        single_obs_dim = int(np.prod(obs_space.shape))
        self.state_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.num_agents * single_obs_dim,),
            dtype=np.float32
        )

        # Episode tracking (use stored value, not wrapped env)
        self.max_episode_steps = self._base_max_steps
        self.individual_episode_reward = {k: 0.0 for k in self.agents}
        self._episode_step = 0
        self._last_obs: Optional[List[np.ndarray]] = None

    def _build_agent_groups(self) -> List[List[str]]:
        """Build agent_groups based on training mode.

        Returns:
            List of agent groups. Structure depends on training_mode:
            - COMPETITIVE: Groups based on environment's team structure
            - COOPERATIVE: Single group with all agents
            - INDEPENDENT: Each agent in their own group
        """
        if self.training_mode == TrainingMode.COOPERATIVE:
            # All agents share one policy
            return [self.agents.copy()]

        elif self.training_mode == TrainingMode.INDEPENDENT:
            # Each agent has their own policy
            return [[agent] for agent in self.agents]

        else:  # COMPETITIVE (default)
            # Use custom teams if provided
            if self._custom_teams is not None:
                return [
                    [f"agent_{i}" for i in team]
                    for team in self._custom_teams
                ]

            # Otherwise, use environment's natural team structure
            return self._build_teams_from_env()

    def _build_teams_from_env(self) -> List[List[str]]:
        """Build team groups from gym-multigrid's agent.index assignments.

        Returns:
            List of agent groups based on environment's team structure.
            Example for Soccer: [['agent_0', 'agent_1'], ['agent_2', 'agent_3']]
        """
        team_to_agents: Dict[int, List[str]] = defaultdict(list)

        # Use stored base env agents (not wrapped env which may not forward .agents)
        for i, agent in enumerate(self._base_env_agents):
            team_idx = agent.index  # Team assignment (1, 2, etc.)
            team_to_agents[team_idx].append(f"agent_{i}")

        # Sort by team index and return as list of lists
        sorted_teams = sorted(team_to_agents.keys())
        return [team_to_agents[team] for team in sorted_teams]

    def close(self) -> None:
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def render(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        """Render the environment."""
        return self.env.render(*args, **kwargs)

    def reset(self, **kwargs: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state.

        Returns:
            Tuple of (observations_dict, info_dict)
        """
        # Handle both old Gym (returns obs) and new Gymnasium (returns obs, info) APIs
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs_list, _info = result
        else:
            obs_list = result

        observations = {
            f"agent_{i}": obs.astype(np.float32)
            for i, obs in enumerate(obs_list)
        }

        self._episode_step = 0
        self.individual_episode_reward = {k: 0.0 for k in self.agents}
        self._last_obs = obs_list

        reset_info = {
            "infos": {},
            "individual_episode_rewards": self.individual_episode_reward.copy(),
            "state": self.state(),
            "avail_actions": self.avail_actions(),
        }

        return observations, reset_info

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], bool, Dict[str, Any]]:
        """Execute actions for all agents.

        Args:
            actions: Dict mapping agent ID to action

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        actions_list = [actions[f"agent_{i}"] for i in range(self.num_agents)]

        # Handle both old Gym (4-tuple) and new Gymnasium (5-tuple) APIs
        # FastLane wrapping converts to 5-tuple, raw gym-multigrid returns 4-tuple
        result = self.env.step(actions_list)
        if len(result) == 5:
            obs_list, rewards_list, done, _truncated, info = result
        else:
            obs_list, rewards_list, done, info = result

        observations = {
            f"agent_{i}": obs.astype(np.float32)
            for i, obs in enumerate(obs_list)
        }

        rewards = {
            f"agent_{i}": float(rewards_list[i])
            for i in range(self.num_agents)
        }

        for agent, reward in rewards.items():
            self.individual_episode_reward[agent] += reward
        self._episode_step += 1
        self._last_obs = obs_list

        terminated = {agent: bool(done) for agent in self.agents}
        truncated = self._episode_step >= self.max_episode_steps

        step_info = {
            "infos": info,
            "individual_episode_rewards": self.individual_episode_reward.copy(),
            "state": self.state(),
            "avail_actions": self.avail_actions(),
        }

        return observations, rewards, terminated, truncated, step_info

    def state(self) -> np.ndarray:
        """Returns the global state of the environment."""
        if self._last_obs is None:
            return np.zeros(self.state_space.shape, dtype=np.float32)

        state = np.concatenate([obs.flatten() for obs in self._last_obs])
        return state.astype(np.float32)

    def agent_mask(self) -> Dict[str, bool]:
        """Returns boolean mask indicating which agents are alive."""
        return {agent: True for agent in self.agents}

    def avail_actions(self) -> Dict[str, np.ndarray]:
        """Returns available actions mask for each agent."""
        num_actions = self.action_space[self.agents[0]].n
        return {
            agent: np.ones(num_actions, dtype=np.bool_)
            for agent in self.agents
        }

    @property
    def env_info(self) -> Dict[str, Any]:
        """Return environment information dict."""
        return {
            'state_space': self.state_space,
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'agents': self.agents,
            'num_agents': self.num_agents,
            'max_episode_steps': self.max_episode_steps,
            'training_mode': self.training_mode.value,
            'num_groups': len(self.agent_groups),
        }

    @property
    def groups_info(self) -> Dict[str, Any]:
        """Return groups information for XuanCe runners.

        This is the key interface that determines policy architecture:
        - runner_competition.py uses this to create per-group policies
        - Structure depends on training_mode setting

        Returns:
            Dict with group structure information.
        """
        return {
            'num_groups': len(self.agent_groups),
            'agent_groups': self.agent_groups,
            'observation_space_groups': [
                {agent: self.observation_space[agent] for agent in group}
                for group in self.agent_groups
            ],
            'action_space_groups': [
                {agent: self.action_space[agent] for agent in group}
                for group in self.agent_groups
            ],
            'num_agents_groups': [len(group) for group in self.agent_groups],
        }

    def get_training_mode_info(self) -> Dict[str, Any]:
        """Return detailed information about current training mode.

        Useful for debugging and logging.
        """
        mode_descriptions = {
            TrainingMode.COMPETITIVE: "Per-team policies (adversarial training)",
            TrainingMode.COOPERATIVE: "Shared policy (parameter sharing)",
            TrainingMode.INDEPENDENT: "Per-agent policies (individual learning)",
        }

        return {
            "training_mode": self.training_mode.value,
            "description": mode_descriptions[self.training_mode],
            "num_policies": len(self.agent_groups),
            "policy_structure": [
                {
                    "policy_id": i,
                    "agents": group,
                    "num_agents": len(group),
                }
                for i, group in enumerate(self.agent_groups)
            ],
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MultiGrid_Env",
    "TrainingMode",
    "MULTIGRID_ENV_INFO",
    "get_available_environments",
]
