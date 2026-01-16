"""MultiGrid environment adapter for the MOSAIC GUI.

gym-multigrid is a multi-agent extension of MiniGrid for training cooperative
and competitive multi-agent RL policies. Agents act simultaneously each step.

Repository: https://github.com/ArnaudFickinger/gym-multigrid
Location: 3rd_party/gym-multigrid/

Key characteristics:
- Multi-agent: Multiple agents controlled by Operators
- Simultaneous stepping: All agents act at once (PARALLEL paradigm)
- Old gym API: Uses `import gym` (not gymnasium)
- Returns lists: obs/rewards/actions are all lists indexed by agent
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

try:  # pragma: no cover - import guard
    from gym_multigrid.envs import SoccerGame4HEnv10x15N2, CollectGame4HEnv10x10N2
except ImportError:  # pragma: no cover
    SoccerGame4HEnv10x15N2 = None  # type: ignore[assignment, misc]
    CollectGame4HEnv10x10N2 = None  # type: ignore[assignment, misc]


# MultiGrid action names (from gym_multigrid.multigrid.Actions)
MULTIGRID_ACTIONS: List[str] = [
    "STILL",    # 0 - Do nothing
    "LEFT",     # 1 - Turn left
    "RIGHT",    # 2 - Turn right
    "FORWARD",  # 3 - Move forward
    "PICKUP",   # 4 - Pick up object
    "DROP",     # 5 - Drop object
    "TOGGLE",   # 6 - Toggle/activate object
    "DONE",     # 7 - Done (not used by default)
]

# Log frequency for step events
_MULTIGRID_STEP_LOG_FREQUENCY = 50


class MultiGridAdapter(EnvironmentAdapter[List[np.ndarray], List[int]]):
    """Adapter for gym-multigrid multi-agent environments.

    This adapter handles the unique characteristics of gym-multigrid:
    - Multiple agents acting simultaneously
    - List-based observations, actions, and rewards
    - Old gym API compatibility

    The environment provides:
    - Observations: List of encoded grid views per agent
    - Actions: List of discrete actions (0-7) per agent
    - Rewards: List of float rewards per agent
    - Done: Single boolean for episode termination
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    )

    # Multi-agent capability declaration
    capabilities = WorkerCapabilities(
        stepping_paradigm=SteppingParadigm.SIMULTANEOUS,
        supported_paradigms=(SteppingParadigm.SIMULTANEOUS,),
        env_types=("gym", "multigrid"),
        action_spaces=("discrete",),
        observation_spaces=("box",),
        max_agents=4,  # Soccer uses 4 agents (2v2)
        supports_self_play=True,
        supports_record=True,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        """Initialize the MultiGrid adapter.

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
        # Map env_id to proper GameId format
        if self._env_id == "soccer":
            return "MultiGrid-Soccer-v0"
        elif self._env_id == "collect":
            return "MultiGrid-Collect-v0"
        else:
            # For already-formatted IDs, return as-is
            return self._env_id

    @property
    def num_agents(self) -> int:
        """Return the number of agents in the environment."""
        return self._num_agents

    def load(self) -> None:
        """Instantiate the MultiGrid environment."""
        if gym is None:
            raise RuntimeError(
                "gym package not installed. "
                "Install with: pip install gym"
            )

        try:
            # Create environment based on env_id
            if self._env_id == "soccer" or self._env_id == "MultiGrid-Soccer-v0":
                if SoccerGame4HEnv10x15N2 is None:
                    raise RuntimeError(
                        "gym-multigrid not installed. "
                        "Install from: 3rd_party/gym-multigrid/"
                    )
                env = SoccerGame4HEnv10x15N2()
            elif self._env_id == "collect" or self._env_id == "MultiGrid-Collect-v0":
                if CollectGame4HEnv10x10N2 is None:
                    raise RuntimeError(
                        "gym-multigrid not installed. "
                        "Install from: 3rd_party/gym-multigrid/"
                    )
                env = CollectGame4HEnv10x10N2()
            else:
                # Try to make via gym.make if registered
                try:
                    env = gym.make(self._env_id)
                except Exception as e:
                    raise RuntimeError(
                        f"Unknown MultiGrid environment: {self._env_id}. "
                        f"Available: soccer, collect. Error: {e}"
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
                f"Failed to create MultiGrid environment '{self._env_id}': {exc}"
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

        # Seed if provided (gym-multigrid uses seed() method)
        if seed is not None:
            env.seed(seed)

        # Reset returns list of observations (old gym API)
        raw_obs = env.reset()
        self._agent_observations = list(raw_obs)
        self._agent_rewards = [0.0] * self._num_agents
        self._step_counter = 0
        # Reset episode tracking (from base class)
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

        # Step returns (obs_list, rewards_list, done, info) - old gym API
        raw_obs, rewards, done, info = env.step(actions)

        self._agent_observations = list(raw_obs)
        self._agent_rewards = [float(r) for r in rewards]

        # Prepare info dict
        step_info: Dict[str, Any] = dict(info) if info else {}
        step_info["num_agents"] = self._num_agents
        step_info["agent_observations"] = self._agent_observations
        step_info["agent_rewards"] = self._agent_rewards
        step_info["actions"] = actions
        step_info["action_names"] = [
            MULTIGRID_ACTIONS[a] if 0 <= a < len(MULTIGRID_ACTIONS) else str(a)
            for a in actions
        ]

        # Sum rewards for total episode reward tracking
        total_reward = float(sum(self._agent_rewards))

        # Old gym API uses single 'done' flag
        terminated = bool(done)
        truncated = False

        # Update episode tracking (from base class)
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
            # Render with agent view highlighting
            frame = env.render(
                mode="rgb_array",
                highlight=self._config.highlight,
            )
            array = np.asarray(frame)
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": array,
                "game_id": self._env_id,
                "num_agents": self._num_agents,
                "step": self._step_counter,
            }
        except Exception:
            # Return empty frame if render fails
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
                    role=f"team_{agent.index}" if hasattr(agent, "index") else None,
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
            List of action names
        """
        return MULTIGRID_ACTIONS.copy()

    def sample_action(self) -> List[int]:
        """Sample random actions for all agents.

        Returns:
            List of random action indices
        """
        import random
        return [random.randint(0, len(MULTIGRID_ACTIONS) - 1) for _ in range(self._num_agents)]

    def sample_single_action(self) -> int:
        """Sample a random action for one agent.

        Returns:
            Random action index
        """
        import random
        return random.randint(0, len(MULTIGRID_ACTIONS) - 1)


# Specific adapter classes for each environment variant
class MultiGridSoccerAdapter(MultiGridAdapter):
    """Adapter for MultiGrid Soccer environment (4 players, 2v2)."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="soccer")
        super().__init__(context, config=config)


class MultiGridCollectAdapter(MultiGridAdapter):
    """Adapter for MultiGrid Collect environment (3 players)."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MultiGridConfig | None = None,
    ) -> None:
        if config is None:
            config = MultiGridConfig(env_id="collect")
        super().__init__(context, config=config)


# Adapter registry for factory pattern
MULTIGRID_ADAPTERS: Dict[GameId, type[MultiGridAdapter]] = {
    GameId.MULTIGRID_SOCCER: MultiGridSoccerAdapter,
    GameId.MULTIGRID_COLLECT: MultiGridCollectAdapter,
}


def create_multigrid_adapter(
    env_id: str = "soccer",
    context: AdapterContext | None = None,
    config: MultiGridConfig | None = None,
) -> MultiGridAdapter:
    """Factory function to create a MultiGrid adapter.

    Args:
        env_id: Environment variant ("soccer" or "collect")
        context: Adapter context
        config: Optional configuration

    Returns:
        MultiGrid adapter instance
    """
    if config is None:
        config = MultiGridConfig(env_id=env_id)
    return MultiGridAdapter(context, config=config)


__all__ = [
    "MultiGridAdapter",
    "MultiGridSoccerAdapter",
    "MultiGridCollectAdapter",
    "MULTIGRID_ADAPTERS",
    "MULTIGRID_ACTIONS",
    "create_multigrid_adapter",
]
