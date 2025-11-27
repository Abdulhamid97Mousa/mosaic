"""PettingZoo multi-agent environment adapters for the Gym GUI.

This module provides adapters for PettingZoo environments, supporting both
AEC (Agent Environment Cycle - turn-based) and Parallel (simultaneous) APIs.

PettingZoo environments can operate in:
- Single-Agent Mode: Run one agent while others use random/scripted policies
- Multi-Agent Mode: Multiple agents with different controllers (human, AI, policy)
- Human Control: Human plays turn-based games (Chess, Tic-Tac-Toe, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, Union

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, RenderMode
from gym_gui.core.pettingzoo_enums import (
    HUMAN_CONTROLLABLE_ENVS,
    PETTINGZOO_CONTROL_MODES,
    PETTINGZOO_ENV_METADATA,
    PETTINGZOO_RENDER_MODES,
    PettingZooAPIType,
    PettingZooEnvId,
    PettingZooFamily,
    get_api_type,
    get_display_name,
    is_aec_env,
)
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_CLOSED,
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_RENDER_ERROR,
    LOG_ADAPTER_STEP_SUMMARY,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class PettingZooConfig:
    """Configuration for PettingZoo multi-agent environments.

    Attributes:
        env_id: The PettingZoo environment identifier (e.g., "chess_v6")
        family: The environment family (classic, mpe, sisl, butterfly, atari)
        render_mode: Rendering mode ("rgb_array", "human", "ansi")
        max_cycles: Maximum number of cycles before truncation
        seed: Random seed for reproducibility
        env_kwargs: Additional keyword arguments passed to environment constructor
        human_player: Name of the agent controlled by human (for hybrid modes)
        agent_controllers: Per-agent controller type ("human", "random", "policy")
    """

    env_id: PettingZooEnvId
    family: PettingZooFamily = PettingZooFamily.CLASSIC
    render_mode: str = "rgb_array"
    max_cycles: int = 500
    seed: Optional[int] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    human_player: Optional[str] = None
    agent_controllers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and set family from env_id if not provided."""
        if self.env_id in PETTINGZOO_ENV_METADATA:
            metadata = PETTINGZOO_ENV_METADATA[self.env_id]
            self.family = metadata[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "env_id": self.env_id.value if isinstance(self.env_id, PettingZooEnvId) else self.env_id,
            "family": self.family.value if isinstance(self.family, PettingZooFamily) else self.family,
            "render_mode": self.render_mode,
            "max_cycles": self.max_cycles,
            "seed": self.seed,
            "env_kwargs": self.env_kwargs,
            "human_player": self.human_player,
            "agent_controllers": self.agent_controllers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PettingZooConfig":
        """Create config from dictionary."""
        env_id = data.get("env_id", "")
        if isinstance(env_id, str):
            try:
                env_id = PettingZooEnvId(env_id)
            except ValueError:
                pass

        family = data.get("family", "classic")
        if isinstance(family, str):
            try:
                family = PettingZooFamily(family)
            except ValueError:
                family = PettingZooFamily.CLASSIC

        return cls(
            env_id=env_id,
            family=family,
            render_mode=data.get("render_mode", "rgb_array"),
            max_cycles=data.get("max_cycles", 500),
            seed=data.get("seed"),
            env_kwargs=data.get("env_kwargs", {}),
            human_player=data.get("human_player"),
            agent_controllers=data.get("agent_controllers", {}),
        )


class PettingZooAdapter(EnvironmentAdapter[Any, Any]):
    """Unified adapter for PettingZoo AEC and Parallel environments.

    This adapter provides a consistent interface for multi-agent environments,
    supporting human control for turn-based games and AI control for all others.

    Attributes:
        id: Environment identifier (e.g., "chess_v6")
        supported_control_modes: Tuple of supported control modes
        default_render_mode: Default rendering mode (RGB_ARRAY)
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: PettingZooConfig | None = None,
        env_id: PettingZooEnvId | str | None = None,
    ) -> None:
        """Initialize the PettingZoo adapter.

        Args:
            context: Adapter context with settings and control mode
            config: Full PettingZoo configuration
            env_id: Environment ID (alternative to config)
        """
        super().__init__(context)

        # Resolve env_id
        if config is not None:
            self._config = config
            self._env_id = config.env_id
        elif env_id is not None:
            if isinstance(env_id, str):
                try:
                    self._env_id = PettingZooEnvId(env_id)
                except ValueError:
                    self._env_id = env_id  # type: ignore
            else:
                self._env_id = env_id
            self._config = PettingZooConfig(env_id=self._env_id)
        else:
            raise ValueError("Either config or env_id must be provided")

        # Set id for adapter interface
        self.id = self._env_id.value if isinstance(self._env_id, PettingZooEnvId) else str(self._env_id)

        # Determine control modes from enum or use defaults
        if isinstance(self._env_id, PettingZooEnvId) and self._env_id in PETTINGZOO_CONTROL_MODES:
            self.supported_control_modes = PETTINGZOO_CONTROL_MODES[self._env_id]
        else:
            self.supported_control_modes = (ControlMode.AGENT_ONLY,)

        # Multi-agent state
        self._pz_env: Any = None  # PettingZoo environment instance
        self._is_parallel: bool = False
        self._agents: List[str] = []
        self._current_agent: Optional[str] = None
        self._step_count: int = 0
        self._episode_rewards: Dict[str, float] = {}
        self._terminated_agents: set[str] = set()
        self._action_masks: Dict[str, Optional[np.ndarray]] = {}
        self._last_observations: Dict[str, Any] = {}

    @property
    def is_parallel(self) -> bool:
        """Check if using Parallel API (vs AEC)."""
        return self._is_parallel

    @property
    def agents(self) -> List[str]:
        """Get list of active agent names."""
        if self._pz_env is None:
            return []
        return list(self._pz_env.agents) if hasattr(self._pz_env, "agents") else []

    @property
    def possible_agents(self) -> List[str]:
        """Get list of all possible agent names."""
        if self._pz_env is None:
            return []
        return list(self._pz_env.possible_agents) if hasattr(self._pz_env, "possible_agents") else []

    @property
    def current_agent(self) -> Optional[str]:
        """Get current agent (for AEC mode)."""
        return self._current_agent

    @property
    def num_agents(self) -> int:
        """Get number of agents."""
        return len(self.possible_agents)

    def load(self) -> None:
        """Instantiate the PettingZoo environment."""
        try:
            # Import pettingzoo dynamically
            import importlib

            # Determine API type
            if isinstance(self._env_id, PettingZooEnvId):
                self._is_parallel = get_api_type(self._env_id) == PettingZooAPIType.PARALLEL
                family = PETTINGZOO_ENV_METADATA[self._env_id][0].value
                # Use full env_id with version (e.g., "tictactoe_v3")
                env_module_name = self._env_id.value
            else:
                # Fallback for string env_id
                self._is_parallel = False
                family = self._config.family.value if isinstance(self._config.family, PettingZooFamily) else self._config.family
                env_module_name = str(self._env_id)

            # Import the environment module (e.g., pettingzoo.classic.tictactoe_v3)
            module_path = f"pettingzoo.{family}.{env_module_name}"
            env_module = importlib.import_module(module_path)

            # Build kwargs
            kwargs: Dict[str, Any] = {"render_mode": self._config.render_mode}
            kwargs.update(self._config.env_kwargs)

            if self._config.max_cycles and self._is_parallel:
                kwargs["max_cycles"] = self._config.max_cycles

            # Create environment using appropriate API
            if self._is_parallel:
                if hasattr(env_module, "parallel_env"):
                    self._pz_env = env_module.parallel_env(**kwargs)
                else:
                    # Fallback to AEC
                    self._is_parallel = False
                    self._pz_env = env_module.env(**kwargs)
            else:
                self._pz_env = env_module.env(**kwargs)

            self.log_constant(
                LOG_ADAPTER_ENV_CREATED,
                extra={
                    "env_id": self.id,
                    "api_type": "parallel" if self._is_parallel else "aec",
                    "render_mode": self._config.render_mode,
                    "family": family,
                },
            )

        except Exception as exc:
            _LOGGER.error("Failed to load PettingZoo environment %s: %s", self.id, exc)
            raise

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[Any]:
        """Reset the environment.

        Args:
            seed: Optional random seed
            options: Additional reset options

        Returns:
            Initial step result
        """
        if self._pz_env is None:
            self.load()

        self._step_count = 0
        self._terminated_agents.clear()
        self._action_masks.clear()

        actual_seed = seed if seed is not None else self._config.seed

        if self._is_parallel:
            observations, infos = self._pz_env.reset(seed=actual_seed)
            self._agents = list(self._pz_env.agents)
            self._current_agent = None
            self._episode_rewards = {agent: 0.0 for agent in self._agents}
            self._last_observations = dict(observations)

            # Package initial observations
            obs = observations.get(self._agents[0]) if self._agents else None

            self.log_constant(
                LOG_ADAPTER_ENV_RESET,
                extra={
                    "env_id": self.id,
                    "seed": actual_seed,
                    "num_agents": len(self._agents),
                    "agents": ",".join(self._agents),
                },
            )

            return self._package_step(
                observation=obs,
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "agents": self._agents,
                    "all_observations": observations,
                    "all_infos": infos,
                },
            )
        else:
            # AEC API
            self._pz_env.reset(seed=actual_seed)
            self._agents = list(self._pz_env.agents)
            self._current_agent = self._pz_env.agent_selection
            self._episode_rewards = {agent: 0.0 for agent in self.possible_agents}

            observation, reward, termination, truncation, info = self._pz_env.last()

            # Extract action mask if available
            action_mask = None
            if isinstance(info, dict) and "action_mask" in info:
                action_mask = info["action_mask"]
                self._action_masks[self._current_agent] = action_mask

            self._last_observations[self._current_agent] = observation

            self.log_constant(
                LOG_ADAPTER_ENV_RESET,
                extra={
                    "env_id": self.id,
                    "seed": actual_seed,
                    "num_agents": len(self._agents),
                    "current_agent": self._current_agent,
                    "has_action_mask": action_mask is not None,
                },
            )

            info_dict: Dict[str, Any] = {
                "current_agent": self._current_agent,
                "action_mask": action_mask,
                "agents": self._agents,
            }
            if isinstance(info, dict):
                info_dict.update(info)

            return self._package_step(
                observation=observation,
                reward=float(reward) if reward else 0.0,
                terminated=bool(termination),
                truncated=bool(truncation),
                info=info_dict,
            )

    def step(self, action: Any) -> AdapterStep[Any]:
        """Execute action(s) in the environment.

        For AEC environments, pass a single action for the current agent.
        For Parallel environments, pass a dict mapping agent names to actions.

        Args:
            action: Single action (AEC) or dict of actions (Parallel)

        Returns:
            Step result
        """
        if self._pz_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._step_count += 1

        if self._is_parallel:
            return self._step_parallel(action)
        else:
            return self._step_aec(action)

    def _step_parallel(self, actions: Dict[str, Any]) -> AdapterStep[Any]:
        """Step in Parallel mode."""
        observations, rewards, terminations, truncations, infos = self._pz_env.step(actions)

        # Update episode rewards
        for agent, reward in rewards.items():
            if agent in self._episode_rewards:
                self._episode_rewards[agent] += float(reward)

        # Track terminated agents
        for agent, terminated in terminations.items():
            if terminated:
                self._terminated_agents.add(agent)

        self._last_observations = dict(observations)

        # Check if all agents are done
        all_terminated = all(terminations.values()) if terminations else False
        all_truncated = all(truncations.values()) if truncations else False

        # Sum rewards for primary output
        total_reward = sum(rewards.values()) if rewards else 0.0

        self.log_constant(
            LOG_ADAPTER_STEP_SUMMARY,
            extra={
                "env_id": self.id,
                "step": self._step_count,
                "total_reward": total_reward,
                "terminated_agents": len(self._terminated_agents),
                "active_agents": len(self.agents),
            },
        )

        return self._package_step(
            observation=observations,
            reward=total_reward,
            terminated=all_terminated,
            truncated=all_truncated,
            info={
                "all_rewards": rewards,
                "all_terminations": terminations,
                "all_truncations": truncations,
                "all_infos": infos,
                "agents": self.agents,
                "episode_rewards": self._episode_rewards.copy(),
            },
        )

    def _step_aec(self, action: Any) -> AdapterStep[Any]:
        """Step in AEC mode."""
        previous_agent = self._current_agent

        # Execute action for current agent
        self._pz_env.step(action)

        # Get next agent's state
        if self._pz_env.agents:
            self._current_agent = self._pz_env.agent_selection
            observation, reward, termination, truncation, info = self._pz_env.last()

            # Update episode rewards
            if previous_agent and previous_agent in self._episode_rewards:
                self._episode_rewards[previous_agent] += float(reward)

            # Track terminated agents
            if termination and self._current_agent:
                self._terminated_agents.add(self._current_agent)

            # Extract action mask
            action_mask = None
            if isinstance(info, dict) and "action_mask" in info:
                action_mask = info["action_mask"]
                self._action_masks[self._current_agent] = action_mask

            self._last_observations[self._current_agent] = observation

            self.log_constant(
                LOG_ADAPTER_STEP_SUMMARY,
                extra={
                    "env_id": self.id,
                    "step": self._step_count,
                    "previous_agent": previous_agent,
                    "current_agent": self._current_agent,
                    "reward": float(reward),
                    "terminated": termination,
                    "has_action_mask": action_mask is not None,
                },
            )

            aec_info: Dict[str, Any] = {
                "current_agent": self._current_agent,
                "previous_agent": previous_agent,
                "action_mask": action_mask,
                "agents": self.agents,
                "episode_rewards": self._episode_rewards.copy(),
            }
            if isinstance(info, dict):
                aec_info.update(info)

            return self._package_step(
                observation=observation,
                reward=float(reward),
                terminated=bool(termination),
                truncated=bool(truncation),
                info=aec_info,
            )
        else:
            # Episode is done - no more agents
            self.log_constant(
                LOG_ADAPTER_STEP_SUMMARY,
                extra={
                    "env_id": self.id,
                    "step": self._step_count,
                    "status": "episode_complete",
                    "episode_rewards": self._episode_rewards,
                },
            )

            return self._package_step(
                observation=None,
                reward=0.0,
                terminated=True,
                truncated=False,
                info={
                    "current_agent": None,
                    "agents": [],
                    "episode_rewards": self._episode_rewards.copy(),
                    "status": "episode_complete",
                },
            )

    def render(self) -> Any:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", else None
        """
        if self._pz_env is None:
            return None

        try:
            result = self._pz_env.render()
            if isinstance(result, np.ndarray):
                return {
                    "mode": RenderMode.RGB_ARRAY.value,
                    "rgb": result,
                    "game_id": self.id,
                    "current_agent": self._current_agent,
                    "agents": self.agents,
                    "step": self._step_count,
                }
            return result
        except Exception as exc:
            self.log_constant(
                LOG_ADAPTER_RENDER_ERROR,
                exc_info=exc,
                extra={"env_id": self.id},
            )
            return None

    def close(self) -> None:
        """Close the environment."""
        if self._pz_env is not None:
            self.log_constant(
                LOG_ADAPTER_ENV_CLOSED,
                extra={"env_id": self.id},
            )
            self._pz_env.close()
            self._pz_env = None

    def build_step_state(
        self,
        observation: Any,
        info: Mapping[str, Any],
    ) -> StepState:
        """Construct the canonical StepState for the current step."""
        agent_snapshots: List[AgentSnapshot] = []

        for agent_name in self.possible_agents:
            is_active = agent_name == self._current_agent
            is_terminated = agent_name in self._terminated_agents

            snapshot = AgentSnapshot(
                name=agent_name,
                role="active" if is_active else ("terminated" if is_terminated else "waiting"),
                info={
                    "reward": self._episode_rewards.get(agent_name, 0.0),
                    "has_action_mask": agent_name in self._action_masks,
                },
            )
            agent_snapshots.append(snapshot)

        return StepState(
            active_agent=self._current_agent,
            agents=tuple(agent_snapshots),
            metrics={
                "step_count": self._step_count,
                "active_agents": len(self.agents),
                "terminated_agents": len(self._terminated_agents),
            },
            environment={
                "is_parallel": self._is_parallel,
                "family": self._config.family.value if isinstance(self._config.family, PettingZooFamily) else self._config.family,
            },
            raw=dict(info) if isinstance(info, Mapping) else {},
        )

    # ─────────────────────────────────────────────────────────────────
    # Multi-agent specific methods
    # ─────────────────────────────────────────────────────────────────

    def get_action_space(self, agent: Optional[str] = None):
        """Get action space for an agent.

        Args:
            agent: Agent name (uses current agent if None for AEC)

        Returns:
            Gymnasium Space
        """
        if self._pz_env is None:
            raise RuntimeError("Environment not initialized")

        if agent is None:
            if self._is_parallel:
                raise ValueError("Agent name required for Parallel environments")
            agent = self._current_agent

        if agent is None:
            raise RuntimeError("No current agent available")

        return self._pz_env.action_space(agent)

    def get_observation_space(self, agent: Optional[str] = None):
        """Get observation space for an agent.

        Args:
            agent: Agent name (uses current agent if None for AEC)

        Returns:
            Gymnasium Space
        """
        if self._pz_env is None:
            raise RuntimeError("Environment not initialized")

        if agent is None:
            if self._is_parallel:
                raise ValueError("Agent name required for Parallel environments")
            agent = self._current_agent

        if agent is None:
            raise RuntimeError("No current agent available")

        return self._pz_env.observation_space(agent)

    def sample_action(self, agent: Optional[str] = None) -> Any:
        """Sample a random action for an agent.

        Args:
            agent: Agent name (uses current agent if None for AEC)

        Returns:
            Sampled action
        """
        action_space = self.get_action_space(agent)

        # Apply action mask if available
        if agent is None:
            agent = self._current_agent

        if agent and agent in self._action_masks and self._action_masks[agent] is not None:
            mask = self._action_masks[agent]
            valid_actions = np.where(mask)[0]
            if len(valid_actions) > 0:
                return np.random.choice(valid_actions)

        return action_space.sample()

    def sample_actions(self) -> Dict[str, Any]:
        """Sample random actions for all active agents.

        Returns:
            Dict mapping agent names to sampled actions
        """
        return {agent: self.sample_action(agent) for agent in self.agents}

    def get_action_mask(self, agent: Optional[str] = None) -> Optional[np.ndarray]:
        """Get action mask for an agent.

        Args:
            agent: Agent name (uses current agent if None)

        Returns:
            Action mask array or None if not available
        """
        if agent is None:
            agent = self._current_agent
        if agent is None:
            return None
        return self._action_masks.get(agent)

    def is_done(self) -> bool:
        """Check if episode is complete."""
        if self._pz_env is None:
            return True
        return len(self.agents) == 0

    def is_human_controllable(self) -> bool:
        """Check if this environment supports human control."""
        if isinstance(self._env_id, PettingZooEnvId):
            return self._env_id in HUMAN_CONTROLLABLE_ENVS
        return False

    def get_human_agent(self) -> Optional[str]:
        """Get the agent designated for human control."""
        if self._config.human_player:
            return self._config.human_player
        # Default to first agent for human-controllable envs
        if self.is_human_controllable() and self.possible_agents:
            return self.possible_agents[0]
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Concrete adapter classes for specific environments
# ═══════════════════════════════════════════════════════════════════════════


class ChessAdapter(PettingZooAdapter):
    """Adapter for Chess environment."""

    id = PettingZooEnvId.CHESS.value
    supported_control_modes = PETTINGZOO_CONTROL_MODES.get(
        PettingZooEnvId.CHESS,
        (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    )

    def __init__(self, context: AdapterContext | None = None, *, config: PettingZooConfig | None = None) -> None:
        if config is None:
            config = PettingZooConfig(env_id=PettingZooEnvId.CHESS)
        super().__init__(context, config=config)


class ConnectFourAdapter(PettingZooAdapter):
    """Adapter for Connect Four environment."""

    id = PettingZooEnvId.CONNECT_FOUR.value
    supported_control_modes = PETTINGZOO_CONTROL_MODES.get(
        PettingZooEnvId.CONNECT_FOUR,
        (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    )

    def __init__(self, context: AdapterContext | None = None, *, config: PettingZooConfig | None = None) -> None:
        if config is None:
            config = PettingZooConfig(env_id=PettingZooEnvId.CONNECT_FOUR)
        super().__init__(context, config=config)


class TicTacToeAdapter(PettingZooAdapter):
    """Adapter for Tic-Tac-Toe environment."""

    id = PettingZooEnvId.TIC_TAC_TOE.value
    supported_control_modes = PETTINGZOO_CONTROL_MODES.get(
        PettingZooEnvId.TIC_TAC_TOE,
        (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    )

    def __init__(self, context: AdapterContext | None = None, *, config: PettingZooConfig | None = None) -> None:
        if config is None:
            config = PettingZooConfig(env_id=PettingZooEnvId.TIC_TAC_TOE)
        super().__init__(context, config=config)


class GoAdapter(PettingZooAdapter):
    """Adapter for Go environment."""

    id = PettingZooEnvId.GO.value
    supported_control_modes = PETTINGZOO_CONTROL_MODES.get(
        PettingZooEnvId.GO,
        (ControlMode.HUMAN_ONLY, ControlMode.AGENT_ONLY, ControlMode.HYBRID_TURN_BASED),
    )

    def __init__(self, context: AdapterContext | None = None, *, config: PettingZooConfig | None = None) -> None:
        if config is None:
            config = PettingZooConfig(env_id=PettingZooEnvId.GO)
        super().__init__(context, config=config)


class SimpleSpreadAdapter(PettingZooAdapter):
    """Adapter for Simple Spread (MPE) environment."""

    id = PettingZooEnvId.SIMPLE_SPREAD.value
    supported_control_modes = PETTINGZOO_CONTROL_MODES.get(
        PettingZooEnvId.SIMPLE_SPREAD,
        (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    )

    def __init__(self, context: AdapterContext | None = None, *, config: PettingZooConfig | None = None) -> None:
        if config is None:
            config = PettingZooConfig(env_id=PettingZooEnvId.SIMPLE_SPREAD)
        super().__init__(context, config=config)


class SimpleTagAdapter(PettingZooAdapter):
    """Adapter for Simple Tag (MPE) environment."""

    id = PettingZooEnvId.SIMPLE_TAG.value
    supported_control_modes = PETTINGZOO_CONTROL_MODES.get(
        PettingZooEnvId.SIMPLE_TAG,
        (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COMPETITIVE),
    )

    def __init__(self, context: AdapterContext | None = None, *, config: PettingZooConfig | None = None) -> None:
        if config is None:
            config = PettingZooConfig(env_id=PettingZooEnvId.SIMPLE_TAG)
        super().__init__(context, config=config)


class PistonballAdapter(PettingZooAdapter):
    """Adapter for Pistonball (Butterfly) environment."""

    id = PettingZooEnvId.PISTONBALL.value
    supported_control_modes = PETTINGZOO_CONTROL_MODES.get(
        PettingZooEnvId.PISTONBALL,
        (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    )

    def __init__(self, context: AdapterContext | None = None, *, config: PettingZooConfig | None = None) -> None:
        if config is None:
            config = PettingZooConfig(env_id=PettingZooEnvId.PISTONBALL)
        super().__init__(context, config=config)


class MultiwalkerAdapter(PettingZooAdapter):
    """Adapter for Multiwalker (SISL) environment."""

    id = PettingZooEnvId.MULTIWALKER.value
    supported_control_modes = PETTINGZOO_CONTROL_MODES.get(
        PettingZooEnvId.MULTIWALKER,
        (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP),
    )

    def __init__(self, context: AdapterContext | None = None, *, config: PettingZooConfig | None = None) -> None:
        if config is None:
            config = PettingZooConfig(env_id=PettingZooEnvId.MULTIWALKER)
        super().__init__(context, config=config)


# ═══════════════════════════════════════════════════════════════════════════
# Adapter registry for factory pattern
# ═══════════════════════════════════════════════════════════════════════════

# Note: This maps to GameId-compatible keys for integration with existing factory
# PettingZoo environments use their own enum but need string keys for compatibility
PETTINGZOO_ADAPTERS: Dict[str, Type[PettingZooAdapter]] = {
    PettingZooEnvId.CHESS.value: ChessAdapter,
    PettingZooEnvId.CONNECT_FOUR.value: ConnectFourAdapter,
    PettingZooEnvId.TIC_TAC_TOE.value: TicTacToeAdapter,
    PettingZooEnvId.GO.value: GoAdapter,
    PettingZooEnvId.SIMPLE_SPREAD.value: SimpleSpreadAdapter,
    PettingZooEnvId.SIMPLE_TAG.value: SimpleTagAdapter,
    PettingZooEnvId.PISTONBALL.value: PistonballAdapter,
    PettingZooEnvId.MULTIWALKER.value: MultiwalkerAdapter,
}


def create_pettingzoo_adapter(
    env_id: PettingZooEnvId | str,
    context: AdapterContext | None = None,
    config: PettingZooConfig | None = None,
) -> PettingZooAdapter:
    """Factory function to create a PettingZoo adapter.

    Args:
        env_id: Environment identifier
        context: Adapter context
        config: Optional configuration

    Returns:
        PettingZoo adapter instance
    """
    env_id_str = env_id.value if isinstance(env_id, PettingZooEnvId) else env_id

    if env_id_str in PETTINGZOO_ADAPTERS:
        adapter_cls = PETTINGZOO_ADAPTERS[env_id_str]
        return adapter_cls(context, config=config)
    else:
        # Generic adapter for unlisted environments
        return PettingZooAdapter(context, env_id=env_id, config=config)


__all__ = [
    "PettingZooConfig",
    "PettingZooAdapter",
    "ChessAdapter",
    "ConnectFourAdapter",
    "TicTacToeAdapter",
    "GoAdapter",
    "SimpleSpreadAdapter",
    "SimpleTagAdapter",
    "PistonballAdapter",
    "MultiwalkerAdapter",
    "PETTINGZOO_ADAPTERS",
    "create_pettingzoo_adapter",
]
