"""Overcooked-AI environment adapter for the MOSAIC GUI.

Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance.
Two agents must coordinate to prepare and deliver soups by collecting ingredients, placing
them in pots, waiting for them to cook, and delivering the finished soups.

Repository: https://github.com/HumanCompatibleAI/overcooked_ai
Paper: https://arxiv.org/abs/1910.05789 (NeurIPS 2019)
Location: 3rd_party/overcooked_ai/

Key characteristics:
- Multi-agent: 2 agents cooperating
- Simultaneous stepping: Both agents act at once (PARALLEL paradigm)
- Custom API: Uses custom OvercookedEnv and OvercookedState
- Research focus: Human-AI coordination, zero-shot coordination (ZSC)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from gym_gui.config.game_configs import OvercookedConfig
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
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.actions import Action, Direction
except ImportError:  # pragma: no cover
    OvercookedGridworld = None  # type: ignore[assignment, misc]
    OvercookedEnv = None  # type: ignore[assignment, misc]
    Recipe = None  # type: ignore[assignment, misc]
    Action = None  # type: ignore[assignment, misc]
    Direction = None  # type: ignore[assignment, misc]


# Overcooked action names (from overcooked_ai_py.mdp.actions.Action)
OVERCOOKED_ACTIONS: List[str] = [
    "NORTH",     # 0 - Move up
    "SOUTH",     # 1 - Move down
    "EAST",      # 2 - Move right
    "WEST",      # 3 - Move left
    "STAY",      # 4 - Do nothing
    "INTERACT",  # 5 - Interact with object (pickup, drop, use pot, etc.)
]

# Log frequency for step events
_OVERCOOKED_STEP_LOG_FREQUENCY = 50


class OvercookedAdapter(EnvironmentAdapter[List[np.ndarray], List[int]]):
    """Adapter for Overcooked-AI cooperative cooking environments.

    This adapter handles the unique characteristics of Overcooked:
    - Two agents acting simultaneously
    - Custom state representation that needs featurization
    - Cooperative soup cooking mechanics
    - Custom MDP and environment API

    The environment provides:
    - Observations: List of featurized state vectors (one per agent)
    - Actions: List of discrete actions (0-5) per agent
    - Rewards: Sparse rewards for soup deliveries
    - Done: Single boolean for episode termination (based on horizon)
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,  # Multi-human multi-keyboard gameplay (2 players)
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    )

    # Multi-agent capability declaration
    capabilities = WorkerCapabilities(
        stepping_paradigm=SteppingParadigm.SIMULTANEOUS,
        supported_paradigms=(SteppingParadigm.SIMULTANEOUS,),
        env_types=("overcooked", "cooperative"),
        action_spaces=("discrete",),
        observation_spaces=("box",),
        max_agents=2,  # Always 2 agents
        supports_self_play=True,
        supports_record=True,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: OvercookedConfig | None = None,
    ) -> None:
        """Initialize the Overcooked adapter.

        Args:
            context: Adapter context with settings and control mode
            config: Overcooked configuration
        """
        super().__init__(context)
        if config is None:
            config = OvercookedConfig()
        self._config = config
        self._layout_name = config.layout_name
        self._step_counter = 0
        self._num_agents = 2  # Always 2 agents in Overcooked
        self._agent_observations: List[np.ndarray] = []
        self._agent_rewards: List[float] = []
        self._current_state = None

    @property
    def id(self) -> str:  # type: ignore[override]
        """Return the environment identifier."""
        # Map layout_name to proper GameId format
        layout_to_game_id = {
            "cramped_room": "overcooked/cramped_room",
            "asymmetric_advantages": "overcooked/asymmetric_advantages",
            "coordination_ring": "overcooked/coordination_ring",
            "forced_coordination": "overcooked/forced_coordination",
            "counter_circuit": "overcooked/counter_circuit",
        }
        return layout_to_game_id.get(self._layout_name, f"overcooked/{self._layout_name}")

    @property
    def num_agents(self) -> int:
        """Return the number of agents in the environment."""
        return self._num_agents

    def load(self) -> None:
        """Instantiate the Overcooked environment."""
        if OvercookedGridworld is None or OvercookedEnv is None:
            raise RuntimeError(
                "overcooked_ai package not installed. "
                "Install from: 3rd_party/overcooked_ai/"
            )

        try:
            # Configure Recipe class (required before creating MDP)
            if Recipe is not None:
                Recipe.configure({})

            # Create MDP from layout
            mdp_params = self._config.mdp_params or {}
            mdp = OvercookedGridworld.from_layout_name(
                layout_name=self._layout_name,
                **mdp_params
            )

            # Wrap in environment
            env_params = self._config.env_params or {}
            self._env = OvercookedEnv.from_mdp(
                mdp,
                horizon=self._config.horizon,
                **env_params
            )

            # Store for featurization
            self._mdp = mdp

            self.log_constant(
                LOG_ADAPTER_ENV_CREATED,
                extra={
                    "layout_name": self._layout_name,
                    "num_agents": self._num_agents,
                    "horizon": self._config.horizon,
                    "grid_size": f"{mdp.width}x{mdp.height}",
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Overcooked environment '{self._layout_name}': {exc}"
            ) from exc

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[List[np.ndarray]]:
        """Reset the environment.

        Args:
            seed: Optional random seed (not used in Overcooked)
            options: Additional reset options

        Returns:
            Initial step result with list of observations (one per agent)
        """
        env = self._require_env()

        # Reset environment (old API - doesn't return observation)
        env.reset()

        # Get current state after reset
        self._current_state = env.state

        # Featurize state for both agents
        self._agent_observations = self._featurize_state_for_agents(self._current_state)
        self._agent_rewards = [0.0, 0.0]
        self._step_counter = 0

        # Reset episode tracking (from base class)
        self._episode_step = 0
        self._episode_return = 0.0

        info: Dict[str, Any] = {
            "num_agents": self._num_agents,
            "layout_name": self._layout_name,
            "horizon": self._config.horizon,
        }

        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "layout_name": self._layout_name,
                "num_agents": self._num_agents,
                "seed": seed if seed is not None else "None",
            },
        )

        return self._package_step(self._agent_observations, 0.0, False, False, info)

    def step(self, action: List[int] | int) -> AdapterStep[List[np.ndarray]]:
        """Execute actions for both agents simultaneously.

        Args:
            action: List of actions (one per agent) or single action.
                    If single action, it's broadcast to both agents.

        Returns:
            Step result with list of observations, sum of rewards, and info
        """
        env = self._require_env()

        # Handle single action (broadcast to both agents)
        if isinstance(action, int):
            actions = [action, action]
        else:
            actions = list(action)
            if len(actions) != self._num_agents:
                raise ValueError(
                    f"Expected {self._num_agents} actions, got {len(actions)}"
                )

        # Convert action indices to Overcooked Action objects
        if Action is not None:
            overcooked_actions = [Action.INDEX_TO_ACTION[a] for a in actions]
        else:
            overcooked_actions = actions

        # Step returns (state, reward, done, info)
        state, reward, done, info = env.step(overcooked_actions)
        self._current_state = state

        # Featurize state for both agents
        self._agent_observations = self._featurize_state_for_agents(state)

        # Extract per-agent rewards from info
        if "sparse_reward_by_agent" in info:
            self._agent_rewards = [float(r) for r in info["sparse_reward_by_agent"]]
        else:
            # If not provided, split reward equally
            self._agent_rewards = [float(reward) / 2.0, float(reward) / 2.0]

        # Prepare info dict
        step_info: Dict[str, Any] = dict(info) if info else {}
        step_info["num_agents"] = self._num_agents
        step_info["agent_observations"] = self._agent_observations
        step_info["agent_rewards"] = self._agent_rewards
        step_info["actions"] = actions
        step_info["action_names"] = [
            OVERCOOKED_ACTIONS[a] if 0 <= a < len(OVERCOOKED_ACTIONS) else str(a)
            for a in actions
        ]

        # Sum rewards for total episode reward tracking
        total_reward = float(reward)

        # Overcooked uses horizon-based termination
        terminated = bool(done)
        truncated = False

        # Update episode tracking (from base class)
        self._step_counter += 1
        self._episode_step += 1
        self._episode_return += total_reward
        step_info["episode_step"] = self._episode_step
        step_info["episode_score"] = self._episode_return

        if self._step_counter % _OVERCOOKED_STEP_LOG_FREQUENCY == 1:
            self.log_constant(
                LOG_ADAPTER_STEP_SUMMARY,
                extra={
                    "layout_name": self._layout_name,
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

    def _featurize_state_for_agents(self, state) -> List[np.ndarray]:
        """Convert Overcooked state to observations for both agents.

        Args:
            state: Overcooked OvercookedState object

        Returns:
            List of featurized observations (one per agent)
        """
        env = self._require_env()

        # Choose featurization method based on config
        if self._config.featurization == "lossless_encoding":
            # Use lossless encoding (returns same for both agents)
            encoding = env.lossless_state_encoding_mdp(state)
            # Replicate for both agents
            return [np.array(encoding, dtype=np.float32)] * 2
        else:
            # Use featurize_state (can be agent-specific)
            # For Overcooked, featurize_state returns same features for both agents
            # but we can call it separately if needed in future
            features = env.featurize_state_mdp(state)
            return [np.array(features, dtype=np.float32)] * 2

    def render(self) -> Dict[str, Any]:
        """Render the environment.

        Returns:
            Dictionary with RGB array and metadata
        """
        env = self._require_env()
        try:
            # Use StateVisualizer for high-quality rendering
            import pygame
            from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

            # Use higher tile_size for better native resolution
            # tile_size=100 → 500×400, tile_size=150 → 750×600
            tile_size = 100
            visualizer = StateVisualizer(tile_size=tile_size)
            surface = visualizer.render_state(env.state, grid=self._mdp.terrain_mtx)

            # Convert pygame Surface to numpy array
            rgb_array = pygame.surfarray.array3d(surface)
            # surfarray returns (width, height, 3), transpose to (height, width, 3)
            rgb_frame = np.transpose(rgb_array, (1, 0, 2))

            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": rgb_frame,
                "game_id": self.id,
                "num_agents": self._num_agents,
                "step": self._step_counter,
            }

        except Exception:
            # Return empty frame if render fails
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": np.zeros((400, 600, 3), dtype=np.uint8),
                "game_id": self.id,
            }

    def close(self) -> None:
        """Close the environment."""
        if self._env is not None:
            self.log_constant(
                LOG_ADAPTER_ENV_CLOSED,
                extra={"layout_name": self._layout_name},
            )
            # Overcooked doesn't have explicit close method
            self._env = None
            self._mdp = None

    def get_agent_observation(self, agent_idx: int) -> np.ndarray:
        """Get observation for a specific agent.

        Args:
            agent_idx: Index of the agent (0 or 1)

        Returns:
            Observation array for the specified agent
        """
        if agent_idx < 0 or agent_idx >= self._num_agents:
            raise IndexError(f"Agent index {agent_idx} out of range [0, {self._num_agents})")
        return self._agent_observations[agent_idx]

    def get_agent_reward(self, agent_idx: int) -> float:
        """Get last reward for a specific agent.

        Args:
            agent_idx: Index of the agent (0 or 1)

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
        if self._current_state is None:
            # No state available yet (before first step)
            return StepState(
                active_agent=None,
                agent_snapshots=[],
                custom_state={},
            )

        # Build agent snapshots from Overcooked state
        agent_snapshots = []
        for i in range(2):
            player = self._current_state.players[i]
            position = tuple(player.position) if hasattr(player, "position") else None
            orientation = player.orientation if hasattr(player, "orientation") else None

            # Get what agent is holding
            held_object = None
            if hasattr(player, "held_object") and player.held_object is not None:
                obj = player.held_object
                if hasattr(obj, "name"):
                    held_object = obj.name
                else:
                    held_object = str(obj)

            agent_snapshots.append(
                AgentSnapshot(
                    name=f"agent_{i}",
                    role="chef",  # Both are chefs
                    position=position,
                    orientation=str(orientation) if orientation is not None else None,
                    info={
                        "holding": held_object,
                    },
                )
            )

        # Extract custom state info
        custom_state = {}
        if hasattr(self._current_state, "all_orders"):
            custom_state["orders"] = str(self._current_state.all_orders)
        if hasattr(self._current_state, "bonus_orders"):
            custom_state["bonus_orders"] = str(self._current_state.bonus_orders)

        return StepState(
            active_agent=None,  # Both agents act simultaneously
            agents=tuple(agent_snapshots),
            metrics={
                "step_count": self._step_counter,
                "num_agents": self._num_agents,
                "agent_rewards": self._agent_rewards,
                "sparse_reward": self._agent_rewards[0] + self._agent_rewards[1],
            },
            environment={
                "env_id": self.id,
                "layout": self._config.layout_name,
                "horizon": self._config.horizon,
                "stepping_paradigm": "simultaneous",
                **custom_state,
            },
            raw=dict(info) if isinstance(info, Mapping) else {},
        )


# ============================================================================
# Layout-specific adapter subclasses
# ============================================================================


class CrampedRoomAdapter(OvercookedAdapter):
    """Cramped Room layout: tight kitchen requiring close coordination."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: OvercookedConfig | None = None,
    ) -> None:
        if config is None:
            config = OvercookedConfig(layout_name="cramped_room")
        super().__init__(context, config=config)


class AsymmetricAdvantagesAdapter(OvercookedAdapter):
    """Asymmetric Advantages layout: agents have different access to resources."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: OvercookedConfig | None = None,
    ) -> None:
        if config is None:
            config = OvercookedConfig(layout_name="asymmetric_advantages")
        super().__init__(context, config=config)


class CoordinationRingAdapter(OvercookedAdapter):
    """Coordination Ring layout: circular kitchen requiring rotation."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: OvercookedConfig | None = None,
    ) -> None:
        if config is None:
            config = OvercookedConfig(layout_name="coordination_ring")
        super().__init__(context, config=config)


class ForcedCoordinationAdapter(OvercookedAdapter):
    """Forced Coordination layout: tasks require explicit coordination."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: OvercookedConfig | None = None,
    ) -> None:
        if config is None:
            config = OvercookedConfig(layout_name="forced_coordination")
        super().__init__(context, config=config)


class CounterCircuitAdapter(OvercookedAdapter):
    """Counter Circuit layout: multiple counters in circuit configuration."""

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: OvercookedConfig | None = None,
    ) -> None:
        if config is None:
            config = OvercookedConfig(layout_name="counter_circuit")
        super().__init__(context, config=config)


# ============================================================================
# Adapter registry for factory pattern
# ============================================================================

OVERCOOKED_ADAPTERS: Dict[GameId, type[OvercookedAdapter]] = {
    GameId.OVERCOOKED_CRAMPED_ROOM: CrampedRoomAdapter,
    GameId.OVERCOOKED_ASYMMETRIC_ADVANTAGES: AsymmetricAdvantagesAdapter,
    GameId.OVERCOOKED_COORDINATION_RING: CoordinationRingAdapter,
    GameId.OVERCOOKED_FORCED_COORDINATION: ForcedCoordinationAdapter,
    GameId.OVERCOOKED_COUNTER_CIRCUIT: CounterCircuitAdapter,
}


__all__ = [
    "OvercookedAdapter",
    "CrampedRoomAdapter",
    "AsymmetricAdvantagesAdapter",
    "CoordinationRingAdapter",
    "ForcedCoordinationAdapter",
    "CounterCircuitAdapter",
    "OVERCOOKED_ADAPTERS",
    "OVERCOOKED_ACTIONS",
]
