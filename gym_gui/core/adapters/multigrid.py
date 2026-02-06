"""MultiGrid environment adapter for the MOSAIC GUI.

gym-multigrid is a multi-agent extension of MiniGrid for training cooperative
and competitive multi-agent RL policies. Agents act simultaneously each step.

Supported packages:
- Old gym-multigrid (ArnaudFickinger): Soccer, Collect
  Repository: https://github.com/ArnaudFickinger/gym-multigrid
  Location: 3rd_party/gym-multigrid/

- New multigrid (INI): 13 environments including Empty, RedBlueDoors, LockedHallway, etc.
  Repository: https://github.com/ini/multigrid
  Location: 3rd_party/multigrid-ini/

Key characteristics:
- Multi-agent: Multiple agents controlled by Operators
- Simultaneous stepping: All agents act at once (PARALLEL paradigm)
- Old gym API: Uses `import gym` (not gymnasium)
- Returns lists: obs/rewards/actions are all lists indexed by agent

Reproducibility Fix:
- Legacy gym-multigrid has a bug where step() uses np.random.permutation()
  instead of self.np_random.permutation(), ignoring seeds set via env.seed().
- This adapter wraps legacy environments with ReproducibleMultiGridWrapper
  which seeds np.random from env.np_random before each step().
- This fix is transparent to Operators (human play) and only affects
  training/evaluation reproducibility.
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
from gym_gui.core.wrappers.multigrid_reproducibility import ReproducibleMultiGridWrapper

try:  # pragma: no cover - import guard
    import gym
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from gym_multigrid.envs import SoccerGame4HEnv10x15N2, CollectGame4HEnv10x10N2
except ImportError:  # pragma: no cover
    SoccerGame4HEnv10x15N2 = None  # type: ignore[assignment, misc]
    CollectGame4HEnv10x10N2 = None  # type: ignore[assignment, misc]

# Try importing INI multigrid environments (new package)
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


# Legacy gym-multigrid action names (8 actions - includes STILL)
# Used by: Soccer, Collect (ArnaudFickinger's gym-multigrid)
LEGACY_MULTIGRID_ACTIONS: List[str] = [
    "STILL",    # 0 - Do nothing
    "LEFT",     # 1 - Turn left
    "RIGHT",    # 2 - Turn right
    "FORWARD",  # 3 - Move forward
    "PICKUP",   # 4 - Pick up object
    "DROP",     # 5 - Drop object
    "TOGGLE",   # 6 - Toggle/activate object
    "DONE",     # 7 - Done (not used by default)
]

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

# Backwards compatibility alias
MULTIGRID_ACTIONS = LEGACY_MULTIGRID_ACTIONS

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
        ControlMode.HUMAN_ONLY,  # Multi-human multi-keyboard gameplay
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
        self._is_ini_env = self._check_is_ini_environment()

    def _check_is_ini_environment(self) -> bool:
        """Check if this is an INI multigrid environment (vs legacy gym-multigrid).

        INI multigrid environments use 7 actions (no STILL).
        Legacy gym-multigrid (Soccer, Collect) uses 8 actions (with STILL at index 0).
        """
        # Legacy gym-multigrid environments
        if self._env_id in ("soccer", "collect", "MultiGrid-Soccer-v0", "MultiGrid-Collect-v0"):
            return False
        # INI multigrid environments (or check if in INI_CONFIGURATIONS)
        if self._env_id in INI_CONFIGURATIONS:
            return True
        # Check by name pattern - INI environments include these
        ini_patterns = ("Empty", "BlockedUnlockPickup", "LockedHallway", "RedBlueDoors", "Playground")
        return any(pattern in self._env_id for pattern in ini_patterns)

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
                # Wrap with ReproducibleMultiGridWrapper to fix np.random.permutation bug
                # in step() that ignores env.np_random seeding. This ensures reproducible
                # trajectories for training/evaluation without affecting Operators.
                env = ReproducibleMultiGridWrapper(env)
            elif self._env_id == "collect" or self._env_id == "MultiGrid-Collect-v0":
                if CollectGame4HEnv10x10N2 is None:
                    raise RuntimeError(
                        "gym-multigrid not installed. "
                        "Install from: 3rd_party/gym-multigrid/"
                    )
                env = CollectGame4HEnv10x10N2()
                # Wrap with ReproducibleMultiGridWrapper to fix np.random.permutation bug
                env = ReproducibleMultiGridWrapper(env)
            elif self._env_id in INI_CONFIGURATIONS:
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
                    available_envs = list(INI_CONFIGURATIONS.keys()) + ["soccer", "collect"]
                    raise RuntimeError(
                        f"Unknown MultiGrid environment: {self._env_id}. "
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

        # Handle seeding based on environment type:
        # - Legacy gym-multigrid (Soccer, Collect): Old Gym API with env.seed() method
        # - INI multigrid (BlockedUnlockPickup, etc.): Gymnasium API with reset(seed=seed)
        #
        # This distinction is important for REPRODUCIBILITY in research!
        # See: https://gymnasium.farama.org/introduction/migration_guide/

        if self._is_ini_env:
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
        else:
            # Legacy gym-multigrid uses old Gym API: env.seed() then env.reset() -> obs
            if seed is not None and hasattr(env, 'seed'):
                env.seed(seed)
            raw_obs = env.reset()
        # Convert observations dict to list (INI uses {0: obs0, 1: obs1})
        if isinstance(raw_obs, dict):
            self._agent_observations = [raw_obs[i] for i in range(self._num_agents)]
        else:
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

        # Handle step() API differences:
        # - Legacy gym-multigrid (Old Gym): returns (obs, rewards, done, info) - 4 values
        #   Action format: List[int] e.g., [0, 1, 2, 3]
        # - INI multigrid (Gymnasium): returns (obs, rewards, terminated, truncated, info) - 5 values
        #   Action format: Dict[AgentID, Action] e.g., {0: Action.left, 1: Action.right}
        #
        # CRITICAL: INI multigrid expects a DICT mapping agent index to action,
        # NOT a list! If you pass a list, the environment checks `if i not in actions`
        # which checks if the agent INDEX is a VALUE in the list (wrong!) instead of
        # checking if it's a KEY in the dict (correct), causing actions to be skipped.
        if self._is_ini_env:
            # Convert list to dict for INI multigrid: [2, 6] -> {0: 2, 1: 6}
            actions_dict = {i: actions[i] for i in range(len(actions))}
            step_result = env.step(actions_dict)
        else:
            # Legacy gym-multigrid uses list format
            step_result = env.step(actions)

        if self._is_ini_env:
            # INI multigrid uses Gymnasium API (5 values)
            if len(step_result) == 5:
                raw_obs, rewards, terminated, truncated, info = step_result
            else:
                # Fallback for unexpected format
                raw_obs, rewards, done, info = step_result[:4]
                terminated = bool(done)
                truncated = False
        else:
            # Legacy gym-multigrid uses old Gym API (4 values)
            raw_obs, rewards, done, info = step_result
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

        # Convert observations dict to list (INI uses {0: obs0, 1: obs1})
        if isinstance(raw_obs, dict):
            self._agent_observations = [raw_obs[i] for i in range(self._num_agents)]
        else:
            self._agent_observations = list(raw_obs)
        # Handle rewards - could be dict (per-agent) or list
        # INI multigrid uses integer keys: {0: 0.5, 1: 0.5}
        # Legacy gym-multigrid may use string keys: {"agent_0": 0.5}
        if isinstance(rewards, dict):
            # Try integer keys first (INI multigrid), then string keys (legacy)
            self._agent_rewards = []
            for i in range(self._num_agents):
                if i in rewards:
                    self._agent_rewards.append(float(rewards[i]))
                elif f"agent_{i}" in rewards:
                    self._agent_rewards.append(float(rewards[f"agent_{i}"]))
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
        action_list = INI_MULTIGRID_ACTIONS if self._is_ini_env else LEGACY_MULTIGRID_ACTIONS
        step_info["action_names"] = [
            action_list[a] if 0 <= a < len(action_list) else str(a)
            for a in actions
        ]

        # Sum rewards for total episode reward tracking
        total_reward = float(sum(self._agent_rewards))

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
            # Handle render() API differences:
            # - Legacy gym-multigrid: env.render(mode="rgb_array", highlight=True)
            # - INI multigrid (Gymnasium): env.render() - mode set at creation time
            if self._is_ini_env:
                # Gymnasium API: render_mode is set during env creation
                # Just call render() without arguments
                frame = env.render()
            else:
                # Legacy gym API: pass mode and highlight arguments
                frame = env.render(
                    mode="rgb_array",
                    highlight=self._config.highlight,
                )

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
            List of action names (7 for INI multigrid, 8 for legacy gym-multigrid)
        """
        if self._is_ini_env:
            return INI_MULTIGRID_ACTIONS.copy()
        return LEGACY_MULTIGRID_ACTIONS.copy()

    def sample_action(self) -> List[int]:
        """Sample random actions for all agents.

        Returns:
            List of random action indices
        """
        import random
        action_list = INI_MULTIGRID_ACTIONS if self._is_ini_env else LEGACY_MULTIGRID_ACTIONS
        return [random.randint(0, len(action_list) - 1) for _ in range(self._num_agents)]

    def sample_single_action(self) -> int:
        """Sample a random action for one agent.

        Returns:
            Random action index
        """
        import random
        action_list = INI_MULTIGRID_ACTIONS if self._is_ini_env else LEGACY_MULTIGRID_ACTIONS
        return random.randint(0, len(action_list) - 1)


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
    # Legacy gym-multigrid environments (with dedicated adapter classes)
    GameId.MULTIGRID_SOCCER: MultiGridSoccerAdapter,
    GameId.MULTIGRID_COLLECT: MultiGridCollectAdapter,
    # INI multigrid environments (all use generic MultiGridAdapter)
    GameId.MULTIGRID_BLOCKED_UNLOCK_PICKUP: MultiGridAdapter,
    GameId.MULTIGRID_EMPTY_5X5: MultiGridAdapter,
    GameId.MULTIGRID_EMPTY_RANDOM_5X5: MultiGridAdapter,
    GameId.MULTIGRID_EMPTY_6X6: MultiGridAdapter,
    GameId.MULTIGRID_EMPTY_RANDOM_6X6: MultiGridAdapter,
    GameId.MULTIGRID_EMPTY_8X8: MultiGridAdapter,
    GameId.MULTIGRID_EMPTY_16X16: MultiGridAdapter,
    GameId.MULTIGRID_LOCKED_HALLWAY_2ROOMS: MultiGridAdapter,
    GameId.MULTIGRID_LOCKED_HALLWAY_4ROOMS: MultiGridAdapter,
    GameId.MULTIGRID_LOCKED_HALLWAY_6ROOMS: MultiGridAdapter,
    GameId.MULTIGRID_PLAYGROUND: MultiGridAdapter,
    GameId.MULTIGRID_RED_BLUE_DOORS_6X6: MultiGridAdapter,
    GameId.MULTIGRID_RED_BLUE_DOORS_8X8: MultiGridAdapter,
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
    "MULTIGRID_ACTIONS",  # Backwards compatibility (alias for LEGACY_MULTIGRID_ACTIONS)
    "LEGACY_MULTIGRID_ACTIONS",
    "INI_MULTIGRID_ACTIONS",
    "create_multigrid_adapter",
]
