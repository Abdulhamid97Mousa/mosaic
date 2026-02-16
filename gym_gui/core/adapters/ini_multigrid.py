"""INI MultiGrid adapter - Cooperative exploration multi-agent environments.

Local Package: 3rd_party/multigrid-ini/
GitHub: https://github.com/ini/multigrid

Environments:
- MultiGrid-Empty-* (6 variants): Training grounds with configurable sizes
- MultiGrid-RedBlueDoors-* (2 variants): Sequential door-opening puzzle
- MultiGrid-LockedHallway-* (3 variants): Multi-room navigation with keys
- MultiGrid-BlockedUnlockPickup-v0: Complex pickup with obstacles
- MultiGrid-Playground-v0: Diverse object interactions

Features:
- Gymnasium 1.0+ API (5-tuple dict-keyed observations)
- 7 actions (LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE, DONE - no STILL)
- view_size=7 (larger field of view for exploration)
- Cooperative rewards, puzzle-solving mechanics

Installation:
    Local package from 3rd_party/multigrid-ini/

Usage:
    from gym_gui.core.adapters.ini_multigrid import MultiGridAdapter
    from gym_gui.config.game_configs import MultiGridConfig

    config = MultiGridConfig(env_id="MultiGrid-Empty-5x5-v0")
    adapter = MultiGridAdapter(config=config)
    adapter.load()
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from gym_gui.config.game_configs import MultiGridConfig
from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
    WorkerCapabilities,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode, SteppingParadigm
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_CLOSED,
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
)

try:  # pragma: no cover - import guard
    import gym
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]

# Try importing INI multigrid environments (local package)
try:  # pragma: no cover - import guard
    import sys
    import os
    # Add INI multigrid to path if available
    ini_multigrid_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "3rd_party", "multigrid-ini"
    )
    if os.path.exists(ini_multigrid_path) and ini_multigrid_path not in sys.path:
        sys.path.insert(0, ini_multigrid_path)

    from multigrid.envs import CONFIGURATIONS as INI_CONFIGURATIONS
except ImportError:  # pragma: no cover
    INI_CONFIGURATIONS = {}  # type: ignore[assignment]


# INI multigrid action names (7 actions - NO STILL)
# Used by: BlockedUnlockPickup, Empty, LockedHallway, RedBlueDoors, Playground
# Repository: https://github.com/ini/multigrid
INI_MULTIGRID_ACTIONS: List[str] = [
    "LEFT",     # 0 - Turn left
    "RIGHT",    # 1 - Turn right
    "FORWARD",  # 2 - Move forward
    "PICKUP",   # 3 - Pick up object
    "DROP",     # 4 - Drop object
    "TOGGLE",   # 5 - Toggle/activate object
    "DONE",     # 6 - Done
]

# Log frequency for step events
_MULTIGRID_STEP_LOG_FREQUENCY = 50


class MultiGridAdapter(EnvironmentAdapter[List[np.ndarray], List[int]]):
    """Adapter for INI MultiGrid multi-agent environments.

    This adapter handles cooperative exploration environments from the local
    multigrid-ini package. Key characteristics:
    - Multiple agents acting simultaneously (2-8 agents configurable)
    - Cooperative puzzle-solving and exploration
    - Gymnasium 1.0+ API (5-tuple dict-keyed observations)
    - view_size=7 for larger field of view (better for exploration)
    - 7 actions (no STILL action)

    The environment provides:
    - Observations: Dict of encoded grid views per agent
    - Actions: Dict of discrete actions (0-6) per agent
    - Rewards: Dict of float rewards per agent
    - Terminated/Truncated: Dict per agent + __all__ flag
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,  # Multi-human multi-keyboard gameplay
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    )

    # Multi-agent capability declaration
    capabilities = WorkerCapabilities(
        stepping_paradigm=SteppingParadigm.SIMULTANEOUS,
        supported_paradigms=(SteppingParadigm.SIMULTANEOUS,),
        env_types=("gym", "ini_multigrid"),
        action_spaces=("discrete",),
        observation_spaces=("box",),
        max_agents=8,  # INI supports more agents for cooperative scenarios
        supports_self_play=True,
        supports_record=True,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        """Initialize the INI MultiGrid adapter.

        Args:
            context: Adapter context with settings and control mode
            config: MultiGrid configuration
        """
        super().__init__(context)
        if config is None:
            config = MultiGridConfig()
        self._config = config
        self._env_id = config.env_id
        self._step_counter = 0
        self._num_agents = 0
        self._agent_observations: List[np.ndarray] = []
        self._agent_rewards: List[float] = []

    @property
    def id(self) -> str:  # type: ignore[override]
        """Return the environment identifier."""
        return self._env_id

    @property
    def num_agents(self) -> int:
        """Return the number of agents in the environment."""
        return self._num_agents

    def load(self) -> None:
        """Instantiate the INI MultiGrid environment."""
        if gym is None:
            raise RuntimeError(
                "gym package not installed. "
                "Install with: pip install gym"
            )

        try:
            if self._env_id in INI_CONFIGURATIONS:
                # INI multigrid environment - instantiate directly from configurations
                env_cls, config_kwargs = INI_CONFIGURATIONS[self._env_id]
                # Add num_agents parameter if specified in config
                if self._config.num_agents is not None:
                    config_kwargs = {**config_kwargs, "agents": self._config.num_agents}
                # INI multigrid uses Gymnasium API - must pass render_mode at creation
                config_kwargs = {**config_kwargs, "render_mode": "rgb_array"}
                env = env_cls(**config_kwargs)
            else:
                # Try to make via gym.make if registered
                try:
                    env = gym.make(self._env_id)
                except Exception as e:
                    available_envs = list(INI_CONFIGURATIONS.keys())
                    raise RuntimeError(
                        f"Unknown INI MultiGrid environment: {self._env_id}. "
                        f"Available: {', '.join(available_envs)}. Error: {e}"
                    )

            self._env = env
            self._num_agents = len(env.agents)

            self.log_constant(
                LOG_ADAPTER_ENV_CREATED,
                extra={
                    "env_id": self._env_id,
                    "num_agents": self._num_agents,
                    "grid_size": f"{env.width}x{env.height}",
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create INI MultiGrid environment '{self._env_id}': {exc}"
            ) from exc

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[List[np.ndarray]]:
        """Reset the environment.

        Args:
            seed: Optional random seed
            options: Additional reset options

        Returns:
            Initial step result with list of observations (one per agent)
        """
        env = self._require_env()

        # INI multigrid uses Gymnasium API: reset(seed=seed) -> (obs, info)
        # Seed is passed directly to reset() for proper PRNG initialization
        if seed is not None:
            reset_result = env.reset(seed=seed)
        else:
            reset_result = env.reset()

        # Gymnasium API returns (obs, info) tuple
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            raw_obs = reset_result[0]
        else:
            raw_obs = reset_result

        # Convert observations dict to list (uses {0: obs0, 1: obs1})
        if isinstance(raw_obs, dict):
            self._agent_observations = [raw_obs[i] for i in range(self._num_agents)]
        else:
            self._agent_observations = list(raw_obs)

        self._agent_rewards = [0.0] * self._num_agents
        self._step_counter = 0
        self._episode_step = 0
        self._episode_return = 0.0

        info: Dict[str, Any] = {
            "num_agents": self._num_agents,
            "agent_observations": self._agent_observations,
            "env_id": self._env_id,
        }

        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "env_id": self._env_id,
                "num_agents": self._num_agents,
                "seed": seed if seed is not None else "None",
            },
        )

        return self._package_step(self._agent_observations, 0.0, False, False, info)

    def step(self, action: List[int] | int) -> AdapterStep[List[np.ndarray]]:
        """Execute actions for all agents simultaneously.

        Args:
            action: List of actions (one per agent) or single action.
                    If single action, it's broadcast to all agents.

        Returns:
            Step result with list of observations, sum of rewards, and info
        """
        env = self._require_env()

        # Handle single action (broadcast to all agents)
        if isinstance(action, int):
            actions = [action] * self._num_agents
        else:
            actions = list(action)
            if len(actions) != self._num_agents:
                raise ValueError(
                    f"Expected {self._num_agents} actions, got {len(actions)}"
                )

        # INI multigrid uses Gymnasium API with dict actions
        # CRITICAL: INI multigrid expects a DICT mapping agent index to action,
        # NOT a list! Convert list to dict: [2, 6] -> {0: 2, 1: 6}
        actions_dict = {i: actions[i] for i in range(len(actions))}
        step_result = env.step(actions_dict)

        # Gymnasium API returns 5 values
        if len(step_result) == 5:
            raw_obs, rewards, terminated, truncated, info = step_result
        else:
            # Fallback for unexpected format
            raw_obs, rewards, done, info = step_result[:4]
            terminated = bool(done)
            truncated = False

        # Convert dict terminated/truncated to bool for multi-agent environments
        # Gymnasium multi-agent envs return dicts like {"agent_0": False, "__all__": True}
        if isinstance(terminated, dict):
            # Use __all__ key if present, otherwise any agent terminated
            terminated = terminated.get("__all__", any(terminated.values()))
        if isinstance(truncated, dict):
            # Use __all__ key if present, otherwise any agent truncated
            truncated = truncated.get("__all__", any(truncated.values()))

        # Convert observations dict to list
        if isinstance(raw_obs, dict):
            self._agent_observations = [raw_obs[i] for i in range(self._num_agents)]
        else:
            self._agent_observations = list(raw_obs)

        # Handle rewards - INI multigrid uses integer keys: {0: 0.5, 1: 0.5}
        if isinstance(rewards, dict):
            self._agent_rewards = []
            for i in range(self._num_agents):
                if i in rewards:
                    self._agent_rewards.append(float(rewards[i]))
                else:
                    self._agent_rewards.append(0.0)
        else:
            self._agent_rewards = [float(r) for r in rewards] if hasattr(rewards, '__iter__') else [float(rewards)] * self._num_agents

        # Prepare info dict
        step_info: Dict[str, Any] = dict(info) if isinstance(info, dict) else {}
        step_info["num_agents"] = self._num_agents
        step_info["agent_observations"] = self._agent_observations
        step_info["agent_rewards"] = self._agent_rewards
        step_info["actions"] = actions
        step_info["action_names"] = [
            INI_MULTIGRID_ACTIONS[a] if 0 <= a < len(INI_MULTIGRID_ACTIONS) else str(a)
            for a in actions
        ]

        # Sum rewards for total episode reward tracking
        total_reward = float(sum(self._agent_rewards))

        # Update episode tracking
        self._step_counter += 1
        self._episode_step += 1
        self._episode_return += total_reward
        step_info["episode_step"] = self._episode_step
        step_info["episode_score"] = self._episode_return

        if self._step_counter % _MULTIGRID_STEP_LOG_FREQUENCY == 1:
            self.log_constant(
                LOG_ADAPTER_STEP_SUMMARY,
                extra={
                    "env_id": self._env_id,
                    "step": self._step_counter,
                    "actions": step_info["action_names"],
                    "rewards": self._agent_rewards,
                    "total_reward": total_reward,
                    "terminated": terminated,
                },
            )

        return self._package_step(
            self._agent_observations, total_reward, terminated, truncated, step_info
        )

    def render(self) -> Dict[str, Any]:
        """Render the environment.

        Returns:
            Dictionary with RGB array and metadata
        """
        env = self._require_env()
        try:
            # Gymnasium API: render_mode is set during env creation
            # Just call render() without arguments
            frame = env.render()

            if frame is None:
                # Some environments return None before first step
                return {
                    "mode": RenderMode.RGB_ARRAY.value,
                    "rgb": np.zeros((320, 480, 3), dtype=np.uint8),
                    "game_id": self._env_id,
                    "num_agents": self._num_agents,
                }

            array = np.asarray(frame)
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": array,
                "game_id": self._env_id,
                "num_agents": self._num_agents,
                "step": self._step_counter,
            }
        except Exception as e:
            # Return empty frame if render fails
            import logging
            logging.getLogger(__name__).warning(f"Render failed for {self._env_id}: {e}")
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": np.zeros((320, 480, 3), dtype=np.uint8),
                "game_id": self._env_id,
            }

    def close(self) -> None:
        """Close the environment."""
        if self._env is not None:
            self.log_constant(
                LOG_ADAPTER_ENV_CLOSED,
                extra={"env_id": self.id},
            )
            if hasattr(self._env, "close"):
                self._env.close()
            self._env = None

    def get_agent_observation(self, agent_idx: int) -> np.ndarray:
        """Get observation for a specific agent.

        Args:
            agent_idx: Index of the agent (0 to num_agents-1)

        Returns:
            Observation array for the specified agent
        """
        if agent_idx < 0 or agent_idx >= self._num_agents:
            raise IndexError(f"Agent index {agent_idx} out of range [0, {self._num_agents})")
        return self._agent_observations[agent_idx]

    def get_agent_reward(self, agent_idx: int) -> float:
        """Get last reward for a specific agent.

        Args:
            agent_idx: Index of the agent (0 to num_agents-1)

        Returns:
            Reward for the specified agent from last step
        """
        if agent_idx < 0 or agent_idx >= self._num_agents:
            raise IndexError(f"Agent index {agent_idx} out of range [0, {self._num_agents})")
        return self._agent_rewards[agent_idx]

    def build_step_state(
        self,
        observation: List[np.ndarray],
        info: Mapping[str, Any],
    ) -> StepState:
        """Construct the canonical StepState for the current step."""
        env = self._require_env()

        # Build agent snapshots
        agent_snapshots = []
        for i, agent in enumerate(env.agents):
            agent_snapshots.append(
                AgentSnapshot(
                    name=f"agent_{i}",
                    role=None,  # INI environments don't have teams
                    position=tuple(agent.pos) if hasattr(agent, "pos") else None,
                    orientation=str(agent.dir) if hasattr(agent, "dir") else None,
                    info={
                        "carrying": str(agent.carrying) if hasattr(agent, "carrying") and agent.carrying else None,
                        "color": agent.color if hasattr(agent, "color") else None,
                    },
                )
            )

        return StepState(
            active_agent=None,  # All agents act simultaneously
            agents=tuple(agent_snapshots),
            metrics={
                "step_count": self._step_counter,
                "num_agents": self._num_agents,
                "agent_rewards": self._agent_rewards,
            },
            environment={
                "env_id": self._env_id,
                "grid_size": f"{env.width}x{env.height}",
                "stepping_paradigm": "simultaneous",
            },
            raw=dict(info) if isinstance(info, Mapping) else {},
        )

    def get_action_meanings(self) -> List[str]:
        """Get human-readable action names.

        Returns:
            List of 7 action names
        """
        return INI_MULTIGRID_ACTIONS.copy()

    def sample_action(self) -> List[int]:
        """Sample random actions for all agents.

        Returns:
            List of random action indices
        """
        import random
        return [random.randint(0, len(INI_MULTIGRID_ACTIONS) - 1) for _ in range(self._num_agents)]

    def sample_single_action(self) -> int:
        """Sample a random action for one agent.

        Returns:
            Random action index
        """
        import random
        return random.randint(0, len(INI_MULTIGRID_ACTIONS) - 1)


# INI MultiGrid adapter registry - cooperative exploration environments from local package
INI_MULTIGRID_ADAPTERS: Dict[GameId, type[MultiGridAdapter]] = {
    GameId.INI_MULTIGRID_BLOCKED_UNLOCK_PICKUP: MultiGridAdapter,
    GameId.INI_MULTIGRID_EMPTY_5X5: MultiGridAdapter,
    GameId.INI_MULTIGRID_EMPTY_RANDOM_5X5: MultiGridAdapter,
    GameId.INI_MULTIGRID_EMPTY_6X6: MultiGridAdapter,
    GameId.INI_MULTIGRID_EMPTY_RANDOM_6X6: MultiGridAdapter,
    GameId.INI_MULTIGRID_EMPTY_8X8: MultiGridAdapter,
    GameId.INI_MULTIGRID_EMPTY_16X16: MultiGridAdapter,
    GameId.INI_MULTIGRID_LOCKED_HALLWAY_2ROOMS: MultiGridAdapter,
    GameId.INI_MULTIGRID_LOCKED_HALLWAY_4ROOMS: MultiGridAdapter,
    GameId.INI_MULTIGRID_LOCKED_HALLWAY_6ROOMS: MultiGridAdapter,
    GameId.INI_MULTIGRID_PLAYGROUND: MultiGridAdapter,
    GameId.INI_MULTIGRID_RED_BLUE_DOORS_6X6: MultiGridAdapter,
    GameId.INI_MULTIGRID_RED_BLUE_DOORS_8X8: MultiGridAdapter,
}

__all__ = [
    "MultiGridAdapter",
    "INI_MULTIGRID_ADAPTERS",
    "INI_MULTIGRID_ACTIONS",
]
