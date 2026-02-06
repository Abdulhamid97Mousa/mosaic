"""Melting Pot environment adapter for the MOSAIC GUI.

Melting Pot is a suite of test scenarios for multi-agent reinforcement learning
developed by Google DeepMind. It assesses generalization to novel social situations
involving both familiar and unfamiliar individuals.

Repository: https://github.com/google-deepmind/meltingpot
Shimmy: https://shimmy.farama.org/environments/meltingpot/

Key characteristics:
- Multi-agent: Up to 16 agents in social scenarios
- Parallel API: All agents act simultaneously (SIMULTANEOUS paradigm)
- Dict observations: RGB image + COLLECTIVE_REWARD per agent
- Shimmy wrapper: Converts dm_env to PettingZoo API

NOTE: Linux/macOS only (Windows NOT supported)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from gym_gui.config.game_configs import MeltingPotConfig
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
    from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0
except ImportError:  # pragma: no cover
    MeltingPotCompatibilityV0 = None  # type: ignore[assignment, misc]


# Melting Pot action meanings (from dm_env spec)
# Actions are typically: 0=NOOP, 1-4=MOVE(up/down/left/right), 5-6=TURN(left/right), 7=INTERACT
MELTINGPOT_ACTION_NAMES: List[str] = [
    "NOOP",      # 0 - Do nothing
    "FORWARD",   # 1 - Move forward
    "BACKWARD",  # 2 - Move backward
    "LEFT",      # 3 - Strafe left
    "RIGHT",     # 4 - Strafe right
    "TURN_LEFT", # 5 - Turn left
    "TURN_RIGHT",# 6 - Turn right
    "INTERACT",  # 7 - Interact/use
]

# Log frequency for step events
_MELTINGPOT_STEP_LOG_FREQUENCY = 50


class MeltingPotAdapter(EnvironmentAdapter[Dict[str, np.ndarray], Dict[str, int]]):
    """Adapter for Melting Pot multi-agent social scenarios.

    This adapter handles the Shimmy-wrapped Melting Pot environments using
    PettingZoo Parallel API. All agents act simultaneously.

    The environment provides:
    - Observations: Dict[agent_name, Dict[str, ndarray]] with RGB and COLLECTIVE_REWARD
    - Actions: Dict[agent_name, int] - discrete action per agent
    - Rewards: Dict[agent_name, float/ndarray] - reward per agent
    - Terminated/Truncated: Dict[agent_name, bool] - per-agent flags
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
        env_types=("pettingzoo", "parallel", "meltingpot"),
        action_spaces=("discrete",),
        observation_spaces=("dict",),
        max_agents=16,  # Some substrates support up to 16 agents
        supports_self_play=True,
        supports_record=True,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MeltingPotConfig | None = None,
    ) -> None:
        """Initialize the Melting Pot adapter.

        Args:
            context: Adapter context with settings and control mode
            config: Melting Pot configuration
        """
        super().__init__(context)
        if config is None:
            config = MeltingPotConfig()
        self._config = config
        self._substrate_name = config.substrate_name
        self._step_counter = 0
        self._num_agents = 0
        self._agent_names: List[str] = []
        self._agent_observations: Dict[str, Dict[str, np.ndarray]] = {}
        self._agent_rewards: Dict[str, float] = {}

    @property
    def id(self) -> str:  # type: ignore[override]
        """Return the environment identifier."""
        # Map substrate_name to proper GameId format
        substrate_map = {
            "collaborative_cooking__circuit": "meltingpot/collaborative_cooking__circuit",
            "clean_up": "meltingpot/clean_up",
            "clean_up__repeated": "meltingpot/clean_up",  # Legacy name (incorrect)
            "commons_harvest__open": "meltingpot/commons_harvest__open",
            "territory__rooms": "meltingpot/territory__rooms",
            "king_of_the_hill__repeated": "meltingpot/king_of_the_hill__repeated",
            "prisoners_dilemma_in_the_matrix__repeated": "meltingpot/prisoners_dilemma_in_the_matrix__repeated",
            "stag_hunt_in_the_matrix__repeated": "meltingpot/stag_hunt_in_the_matrix__repeated",
            "allelopathic_harvest__open": "meltingpot/allelopathic_harvest__open",
        }
        return substrate_map.get(self._substrate_name, f"meltingpot/{self._substrate_name}")

    @property
    def num_agents(self) -> int:
        """Return the number of agents in the environment."""
        return self._num_agents

    def load(self) -> None:
        """Instantiate the Melting Pot environment via Shimmy wrapper."""
        if MeltingPotCompatibilityV0 is None:
            raise RuntimeError(
                "shimmy[meltingpot] not installed. "
                "Install with: pip install 'shimmy[meltingpot]>=1.3.0' dm-meltingpot>=2.4.0"
            )

        try:
            # Create environment via Shimmy wrapper with rgb_array render mode
            # This enables env.render() to return a higher-resolution world view
            self._env = MeltingPotCompatibilityV0(
                substrate_name=self._substrate_name,
                render_mode="rgb_array",
            )

            # Get agent list
            self._agent_names = list(self._env.agents)
            self._num_agents = len(self._agent_names)

            self.log_constant(
                LOG_ADAPTER_ENV_CREATED,
                extra={
                    "substrate_name": self._substrate_name,
                    "num_agents": self._num_agents,
                    "agents": self._agent_names,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Melting Pot environment '{self._substrate_name}': {exc}"
            ) from exc

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[Dict[str, np.ndarray]]:
        """Reset the environment.

        Args:
            seed: Optional random seed
            options: Additional reset options

        Returns:
            Initial step result with dict of observations (one per agent)
        """
        env = self._require_env()

        # Seed if provided
        if seed is not None:
            env.reset(seed=seed)
            obs, info = env.reset()
        else:
            obs, info = env.reset()

        # Store observations (each is a dict with 'RGB' and 'COLLECTIVE_REWARD')
        self._agent_observations = dict(obs)
        self._agent_rewards = {agent: 0.0 for agent in self._agent_names}
        self._step_counter = 0

        # Reset episode tracking (from base class)
        self._episode_step = 0
        self._episode_return = 0.0

        step_info: Dict[str, Any] = {
            "num_agents": self._num_agents,
            "agents": self._agent_names,
            "substrate_name": self._substrate_name,
        }
        if info:
            step_info.update(info)

        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "substrate_name": self._substrate_name,
                "num_agents": self._num_agents,
                "seed": seed if seed is not None else "None",
            },
        )

        # Package as single observation dict for adapter interface
        # The actual per-agent observations are stored in _agent_observations
        return self._package_step(self._agent_observations, 0.0, False, False, step_info)

    def step(self, action: Dict[str, int] | int) -> AdapterStep[Dict[str, np.ndarray]]:
        """Execute actions for all agents simultaneously.

        Args:
            action: Dict of actions per agent or single action (broadcast to all)

        Returns:
            Step result with dict of observations, sum of rewards, and info
        """
        env = self._require_env()

        # Handle single action (broadcast to all agents)
        if isinstance(action, int):
            actions = {agent: action for agent in self._agent_names}
        else:
            actions = dict(action)
            if len(actions) != self._num_agents:
                raise ValueError(
                    f"Expected {self._num_agents} actions, got {len(actions)}"
                )

        # Step returns (obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict)
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Store observations and rewards
        self._agent_observations = dict(obs)

        # Convert rewards (may be 0-d arrays) to floats
        self._agent_rewards = {}
        for agent in self._agent_names:
            reward = rewards[agent]
            if isinstance(reward, np.ndarray):
                self._agent_rewards[agent] = float(reward)
            else:
                self._agent_rewards[agent] = float(reward)

        # Prepare info dict
        step_info: Dict[str, Any] = dict(info) if info else {}
        step_info["num_agents"] = self._num_agents
        step_info["agents"] = self._agent_names
        step_info["agent_rewards"] = self._agent_rewards
        step_info["actions"] = actions
        step_info["action_names"] = {
            agent: MELTINGPOT_ACTION_NAMES[act] if 0 <= act < len(MELTINGPOT_ACTION_NAMES) else str(act)
            for agent, act in actions.items()
        }

        # Sum rewards for total episode reward tracking
        total_reward = float(sum(self._agent_rewards.values()))

        # Check if any agent is done
        any_terminated = any(terminated.values())
        any_truncated = any(truncated.values())

        # Update episode tracking (from base class)
        self._step_counter += 1
        self._episode_step += 1
        self._episode_return += total_reward
        step_info["episode_step"] = self._episode_step
        step_info["episode_score"] = self._episode_return

        if self._step_counter % _MELTINGPOT_STEP_LOG_FREQUENCY == 1:
            self.log_constant(
                LOG_ADAPTER_STEP_SUMMARY,
                extra={
                    "substrate_name": self._substrate_name,
                    "step": self._step_counter,
                    "total_reward": total_reward,
                    "agent_rewards": self._agent_rewards,
                    "terminated": any_terminated,
                },
            )

        return self._package_step(
            self._agent_observations, total_reward, any_terminated, any_truncated, step_info
        )

    def render(self) -> Dict[str, Any]:
        """Render the environment.

        Uses env.render() to get a higher-resolution global world view.
        Falls back to per-agent RGB observation if render() is unavailable.
        Applies render_scale for enhanced visibility (using PIL NEAREST for speed).

        Returns:
            Dictionary with RGB array and metadata
        """
        from PIL import Image

        env = self._require_env()
        frame: np.ndarray | None = None

        try:
            # Primary: Use env.render() for higher resolution world view
            # With render_mode='rgb_array', this returns ~240×232 pixels (varies by substrate)
            rgb_array = env.render()
            if rgb_array is not None:
                frame = np.asarray(rgb_array, dtype=np.uint8)

            # Fallback: Get RGB from first agent's observation (88×88 per-agent view)
            if frame is None and self._agent_names and self._agent_observations:
                first_agent = self._agent_names[0]
                obs = self._agent_observations.get(first_agent, {})
                rgb_array = obs.get('RGB')
                if rgb_array is not None:
                    frame = np.asarray(rgb_array, dtype=np.uint8)

            # Last resort: return empty frame
            if frame is None:
                frame = np.zeros((240, 232, 3), dtype=np.uint8)

            # Apply render_scale if > 1 (using NEAREST for fast real-time rendering)
            scale = self._config.render_scale
            if scale > 1:
                img = Image.fromarray(frame)
                new_size = (frame.shape[1] * scale, frame.shape[0] * scale)
                img = img.resize(new_size, Image.Resampling.NEAREST)
                frame = np.array(img)

            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": frame,
                "substrate_name": self._substrate_name,
                "num_agents": self._num_agents,
                "step": self._step_counter,
            }
        except Exception:
            # Return empty frame if render fails
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": np.zeros((240, 232, 3), dtype=np.uint8),
                "substrate_name": self._substrate_name,
            }

    def close(self) -> None:
        """Close the environment."""
        if self._env is not None:
            self.log_constant(
                LOG_ADAPTER_ENV_CLOSED,
                extra={"substrate_name": self.id},
            )
            if hasattr(self._env, "close"):
                self._env.close()
            self._env = None

    def get_agent_observation(self, agent_name: str) -> Dict[str, np.ndarray]:
        """Get observation for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., 'player_0')

        Returns:
            Observation dict with 'RGB' and 'COLLECTIVE_REWARD' keys
        """
        if agent_name not in self._agent_names:
            raise KeyError(f"Agent '{agent_name}' not found. Available: {self._agent_names}")
        return self._agent_observations[agent_name]

    def get_agent_reward(self, agent_name: str) -> float:
        """Get last reward for a specific agent.

        Args:
            agent_name: Name of the agent (e.g., 'player_0')

        Returns:
            Reward for the specified agent from last step
        """
        if agent_name not in self._agent_names:
            raise KeyError(f"Agent '{agent_name}' not found. Available: {self._agent_names}")
        return self._agent_rewards[agent_name]

    def build_step_state(
        self,
        observation: Dict[str, np.ndarray],
        info: Mapping[str, Any],
    ) -> StepState:
        """Construct the canonical StepState for the current step."""
        # Build agent snapshots
        agent_snapshots = []
        for agent_name in self._agent_names:
            obs = self._agent_observations.get(agent_name, {})
            reward = self._agent_rewards.get(agent_name, 0.0)

            agent_snapshots.append(
                AgentSnapshot(
                    name=agent_name,
                    role=None,  # Melting Pot doesn't have fixed roles
                    position=None,  # Position not directly available
                    orientation=None,
                    info={
                        "reward": reward,
                        "collective_reward": float(obs.get('COLLECTIVE_REWARD', 0.0))
                            if isinstance(obs.get('COLLECTIVE_REWARD'), np.ndarray)
                            else obs.get('COLLECTIVE_REWARD', 0.0),
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
                "total_reward": sum(self._agent_rewards.values()),
            },
            environment={
                "substrate_name": self._substrate_name,
                "stepping_paradigm": "simultaneous",
            },
            raw=dict(info) if isinstance(info, Mapping) else {},
        )

    def get_action_meanings(self) -> List[str]:
        """Get human-readable action names.

        Returns:
            List of action names
        """
        return MELTINGPOT_ACTION_NAMES.copy()

    def sample_action(self) -> Dict[str, int]:
        """Sample random actions for all agents.

        Returns:
            Dict of random action indices per agent
        """
        env = self._require_env()
        return {agent: env.action_space(agent).sample() for agent in self._agent_names}

    def sample_single_action(self) -> int:
        """Sample a random action for one agent.

        Returns:
            Random action index
        """
        import random
        return random.randint(0, len(MELTINGPOT_ACTION_NAMES) - 1)


# Specific adapter classes for each substrate
class CollaborativeCookingAdapter(MeltingPotAdapter):
    """Adapter for Collaborative Cooking substrate."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MeltingPotConfig | None = None,
    ) -> None:
        if config is None:
            config = MeltingPotConfig(substrate_name="collaborative_cooking__circuit")
        super().__init__(context, config=config)


class CleanUpAdapter(MeltingPotAdapter):
    """Adapter for Clean Up substrate."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MeltingPotConfig | None = None,
    ) -> None:
        if config is None:
            config = MeltingPotConfig(substrate_name="clean_up")
        super().__init__(context, config=config)


class CommonsHarvestAdapter(MeltingPotAdapter):
    """Adapter for Commons Harvest substrate."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MeltingPotConfig | None = None,
    ) -> None:
        if config is None:
            config = MeltingPotConfig(substrate_name="commons_harvest__open")
        super().__init__(context, config=config)


class TerritoryAdapter(MeltingPotAdapter):
    """Adapter for Territory substrate."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MeltingPotConfig | None = None,
    ) -> None:
        if config is None:
            config = MeltingPotConfig(substrate_name="territory__rooms")
        super().__init__(context, config=config)


class KingOfTheHillAdapter(MeltingPotAdapter):
    """Adapter for King of the Hill substrate."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MeltingPotConfig | None = None,
    ) -> None:
        if config is None:
            config = MeltingPotConfig(substrate_name="king_of_the_hill__repeated")
        super().__init__(context, config=config)


class PrisonersDilemmaAdapter(MeltingPotAdapter):
    """Adapter for Prisoners Dilemma substrate."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MeltingPotConfig | None = None,
    ) -> None:
        if config is None:
            config = MeltingPotConfig(substrate_name="prisoners_dilemma_in_the_matrix__repeated")
        super().__init__(context, config=config)


class StagHuntAdapter(MeltingPotAdapter):
    """Adapter for Stag Hunt substrate."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MeltingPotConfig | None = None,
    ) -> None:
        if config is None:
            config = MeltingPotConfig(substrate_name="stag_hunt_in_the_matrix__repeated")
        super().__init__(context, config=config)


class AllelopathicHarvestAdapter(MeltingPotAdapter):
    """Adapter for Allelopathic Harvest substrate."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MeltingPotConfig | None = None,
    ) -> None:
        if config is None:
            config = MeltingPotConfig(substrate_name="allelopathic_harvest__open")
        super().__init__(context, config=config)


# Adapter registry for factory pattern - all 49 substrates use the base MeltingPotAdapter
# The substrate name is determined from the GameId value (e.g., "meltingpot/clean_up" -> "clean_up")
MELTINGPOT_ADAPTERS: Dict[GameId, type[MeltingPotAdapter]] = {
    GameId.MELTINGPOT_ALLELOPATHIC_HARVEST__OPEN: MeltingPotAdapter,
    GameId.MELTINGPOT_BACH_OR_STRAVINSKY_IN_THE_MATRIX__ARENA: MeltingPotAdapter,
    GameId.MELTINGPOT_BACH_OR_STRAVINSKY_IN_THE_MATRIX__REPEATED: MeltingPotAdapter,
    GameId.MELTINGPOT_BOAT_RACE__EIGHT_RACES: MeltingPotAdapter,
    GameId.MELTINGPOT_CHEMISTRY__THREE_METABOLIC_CYCLES: MeltingPotAdapter,
    GameId.MELTINGPOT_CHEMISTRY__THREE_METABOLIC_CYCLES_WITH_PLENTIFUL_DISTRACTORS: MeltingPotAdapter,
    GameId.MELTINGPOT_CHEMISTRY__TWO_METABOLIC_CYCLES: MeltingPotAdapter,
    GameId.MELTINGPOT_CHEMISTRY__TWO_METABOLIC_CYCLES_WITH_DISTRACTORS: MeltingPotAdapter,
    GameId.MELTINGPOT_CHICKEN_IN_THE_MATRIX__ARENA: MeltingPotAdapter,
    GameId.MELTINGPOT_CHICKEN_IN_THE_MATRIX__REPEATED: MeltingPotAdapter,
    GameId.MELTINGPOT_CLEAN_UP: MeltingPotAdapter,
    GameId.MELTINGPOT_COINS: MeltingPotAdapter,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__ASYMMETRIC: MeltingPotAdapter,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__CIRCUIT: MeltingPotAdapter,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__CRAMPED: MeltingPotAdapter,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__CROWDED: MeltingPotAdapter,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__FIGURE_EIGHT: MeltingPotAdapter,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__FORCED: MeltingPotAdapter,
    GameId.MELTINGPOT_COLLABORATIVE_COOKING__RING: MeltingPotAdapter,
    GameId.MELTINGPOT_COMMONS_HARVEST__CLOSED: MeltingPotAdapter,
    GameId.MELTINGPOT_COMMONS_HARVEST__OPEN: MeltingPotAdapter,
    GameId.MELTINGPOT_COMMONS_HARVEST__PARTNERSHIP: MeltingPotAdapter,
    GameId.MELTINGPOT_COOP_MINING: MeltingPotAdapter,
    GameId.MELTINGPOT_DAYCARE: MeltingPotAdapter,
    GameId.MELTINGPOT_EXTERNALITY_MUSHROOMS__DENSE: MeltingPotAdapter,
    GameId.MELTINGPOT_FACTORY_COMMONS__EITHER_OR: MeltingPotAdapter,
    GameId.MELTINGPOT_FRUIT_MARKET__CONCENTRIC_RIVERS: MeltingPotAdapter,
    GameId.MELTINGPOT_GIFT_REFINEMENTS: MeltingPotAdapter,
    GameId.MELTINGPOT_HIDDEN_AGENDA: MeltingPotAdapter,
    GameId.MELTINGPOT_PAINTBALL__CAPTURE_THE_FLAG: MeltingPotAdapter,
    GameId.MELTINGPOT_PAINTBALL__KING_OF_THE_HILL: MeltingPotAdapter,
    GameId.MELTINGPOT_PREDATOR_PREY__ALLEY_HUNT: MeltingPotAdapter,
    GameId.MELTINGPOT_PREDATOR_PREY__OPEN: MeltingPotAdapter,
    GameId.MELTINGPOT_PREDATOR_PREY__ORCHARD: MeltingPotAdapter,
    GameId.MELTINGPOT_PREDATOR_PREY__RANDOM_FOREST: MeltingPotAdapter,
    GameId.MELTINGPOT_PRISONERS_DILEMMA_IN_THE_MATRIX__ARENA: MeltingPotAdapter,
    GameId.MELTINGPOT_PRISONERS_DILEMMA_IN_THE_MATRIX__REPEATED: MeltingPotAdapter,
    GameId.MELTINGPOT_PURE_COORDINATION_IN_THE_MATRIX__ARENA: MeltingPotAdapter,
    GameId.MELTINGPOT_PURE_COORDINATION_IN_THE_MATRIX__REPEATED: MeltingPotAdapter,
    GameId.MELTINGPOT_RATIONALIZABLE_COORDINATION_IN_THE_MATRIX__ARENA: MeltingPotAdapter,
    GameId.MELTINGPOT_RATIONALIZABLE_COORDINATION_IN_THE_MATRIX__REPEATED: MeltingPotAdapter,
    GameId.MELTINGPOT_RUNNING_WITH_SCISSORS_IN_THE_MATRIX__ARENA: MeltingPotAdapter,
    GameId.MELTINGPOT_RUNNING_WITH_SCISSORS_IN_THE_MATRIX__ONE_SHOT: MeltingPotAdapter,
    GameId.MELTINGPOT_RUNNING_WITH_SCISSORS_IN_THE_MATRIX__REPEATED: MeltingPotAdapter,
    GameId.MELTINGPOT_STAG_HUNT_IN_THE_MATRIX__ARENA: MeltingPotAdapter,
    GameId.MELTINGPOT_STAG_HUNT_IN_THE_MATRIX__REPEATED: MeltingPotAdapter,
    GameId.MELTINGPOT_TERRITORY__INSIDE_OUT: MeltingPotAdapter,
    GameId.MELTINGPOT_TERRITORY__OPEN: MeltingPotAdapter,
    GameId.MELTINGPOT_TERRITORY__ROOMS: MeltingPotAdapter,
}


def create_meltingpot_adapter(
    substrate_name: str = "collaborative_cooking__circuit",
    context: AdapterContext | None = None,
    config: MeltingPotConfig | None = None,
) -> MeltingPotAdapter:
    """Factory function to create a Melting Pot adapter.

    Args:
        substrate_name: Substrate identifier
        context: Adapter context
        config: Optional configuration

    Returns:
        Melting Pot adapter instance
    """
    if config is None:
        config = MeltingPotConfig(substrate_name=substrate_name)
    return MeltingPotAdapter(context, config=config)


__all__ = [
    "MeltingPotAdapter",
    "CollaborativeCookingAdapter",
    "CleanUpAdapter",
    "CommonsHarvestAdapter",
    "TerritoryAdapter",
    "KingOfTheHillAdapter",
    "PrisonersDilemmaAdapter",
    "StagHuntAdapter",
    "AllelopathicHarvestAdapter",
    "MELTINGPOT_ADAPTERS",
    "MELTINGPOT_ACTION_NAMES",
    "create_meltingpot_adapter",
]
