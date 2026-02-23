"""Crafter environment adapters for the Crafter open-world survival benchmark.

Crafter is an open-world survival game benchmark that evaluates a wide range
of agent capabilities within a single environment, including generalization,
exploration, credit assignment, memory, and representation learning.

Paper: Hafner, D. (2022). Benchmarking the Spectrum of Agent Capabilities. ICLR 2022.
Repository: https://github.com/danijar/crafter
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from gym_gui.config.game_configs import CrafterConfig
from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
    LOG_ENV_CRAFTER_ACHIEVEMENT,
    LOG_ENV_CRAFTER_BOOT,
    LOG_ENV_CRAFTER_ERROR,
    LOG_ENV_CRAFTER_RENDER_WARNING,
    LOG_ENV_CRAFTER_STEP,
)

try:  # pragma: no cover - import guard exercised in integration tests
    import crafter
except ImportError:  # pragma: no cover - handled gracefully at runtime
    crafter = None  # type: ignore[assignment]


_CRAFTER_STEP_LOG_FREQUENCY = 100

# Crafter achievement names (22 total)
CRAFTER_ACHIEVEMENTS = [
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
]

# Crafter action names (17 total)
CRAFTER_ACTIONS = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]


@dataclass(slots=True)
class _CrafterMetrics:
    """Container describing Crafter-specific telemetry traits."""

    health: int | None = None
    food: int | None = None
    drink: int | None = None
    energy: int | None = None
    achievements_unlocked: int = 0
    player_pos: tuple[int, int] | None = None
    inventory: dict[str, int] | None = None
    episode_achievements: dict[str, int] | None = None


class CrafterAdapter(EnvironmentAdapter[np.ndarray, int]):
    """Adapter for Crafter environments with RGB observations.

    Crafter provides 64x64 RGB images as observations and has a discrete
    action space of 17 actions including movement, interaction, and crafting.
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )

    DEFAULT_ENV_ID = GameId.CRAFTER_REWARD.value

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: CrafterConfig | None = None,
    ) -> None:
        super().__init__(context)
        if config is None:
            config = CrafterConfig(env_id=self.DEFAULT_ENV_ID)
        self._config = config
        self._env_id = config.env_id or self.DEFAULT_ENV_ID
        self._step_counter = 0
        self._render_warning_emitted = False
        self._previous_achievements: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def render(self) -> dict[str, Any]:
        env = self._require_env()
        frame = env.render()
        array = np.asarray(frame)
        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": array,
            "game_id": self._env_id,
        }

    @property
    def id(self) -> str:  # type: ignore[override]
        return self._env_id

    def gym_kwargs(self) -> dict[str, Any]:
        kwargs = super().gym_kwargs()
        kwargs.update(self._config.to_gym_kwargs())
        return kwargs

    def load(self) -> None:
        if crafter is None:
            raise RuntimeError(
                "Crafter package not installed. Install with: pip install crafter"
            )
        try:
            # Create Crafter environment directly since it registers with old gym, not gymnasium
            # We use crafter.Env directly instead of gymnasium.make()
            env = crafter.Env(
                area=self._config.area,
                view=self._config.view,
                size=self._config.size,
                reward=self._config.reward,
                length=self._config.length,
                seed=self._config.seed,
            )
            self._env = env
        except Exception as exc:  # pragma: no cover - defensive logging
            self.log_constant(
                LOG_ENV_CRAFTER_ERROR,
                exc_info=exc,
                extra={
                    "env_id": self._env_id,
                    "stage": "load",
                },
            )
            raise
        self.log_constant(
            LOG_ENV_CRAFTER_BOOT,
            extra={
                "env_id": self._env_id,
                "reward_enabled": self._config.reward,
                "reward_multiplier": float(self._config.reward_multiplier),
                "max_steps": self._config.length,
                "world_size": self._config.area,
                "view_size": self._config.view,
                "image_size": self._config.size,
            },
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[np.ndarray]:
        env = self._require_env()
        # Crafter uses old gym API: reset() returns observation only
        # If seed is provided, we need to recreate the environment
        if seed is not None:
            env = crafter.Env(
                area=self._config.area,
                view=self._config.view,
                size=self._config.size,
                reward=self._config.reward,
                length=self._config.length,
                seed=seed,
            )
            self._env = env

        observation = env.reset()
        processed_obs = self._process_observation(observation)

        # Reset achievement tracking - initial info is empty on reset
        self._previous_achievements = {}

        # Build initial info dict (crafter reset doesn't return info)
        info: dict[str, Any] = {
            "_crafter_raw_observation": observation,
            "achievements": {},
            "inventory": {},
        }
        self._step_counter = 0

        applied_seed = seed if seed is not None else self._config.seed
        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "env_id": self._env_id,
                "seed": applied_seed if applied_seed is not None else "None",
            },
        )
        return self._package_step(processed_obs, 0.0, False, False, info)

    def step(self, action: int) -> AdapterStep[np.ndarray]:
        env = self._require_env()
        # Crafter uses old gym API: step returns (obs, reward, done, info)
        # Not the new gymnasium API with (obs, reward, terminated, truncated, info)
        observation, reward, done, info = env.step(action)
        processed_obs = self._process_observation(observation)
        info = dict(info)
        info["_crafter_raw_observation"] = observation
        scaled_reward = float(reward) * float(self._config.reward_multiplier)

        # Old gym API uses single 'done' flag
        # Determine if it was termination (death) or truncation (time limit)
        terminated = done and info.get("discount", 1.0) == 0.0  # Died
        truncated = done and not terminated  # Time limit reached

        # Track newly unlocked achievements
        self._check_achievements(info)

        self._step_counter += 1
        if self._step_counter % _CRAFTER_STEP_LOG_FREQUENCY == 1:
            self.log_constant(
                LOG_ENV_CRAFTER_STEP,
                extra={
                    "env_id": self._env_id,
                    "step": self._step_counter,
                    "action": CRAFTER_ACTIONS[action] if 0 <= action < len(CRAFTER_ACTIONS) else action,
                    "reward": scaled_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                },
            )

        return self._package_step(processed_obs, scaled_reward, terminated, truncated, info)

    def close(self) -> None:
        """Close the Crafter environment.

        Crafter's Env class doesn't have a close() method, so we override
        the base class close() to handle this gracefully.
        """
        from gym_gui.logging_config.log_constants import LOG_ADAPTER_ENV_CLOSED

        if self._env is not None:
            self.log_constant(
                LOG_ADAPTER_ENV_CLOSED,
                extra={"env_id": self.id},
            )
            # Crafter Env doesn't have close() method, so just clear reference
            self._env = None

    # ------------------------------------------------------------------
    # Adapter customisations
    # ------------------------------------------------------------------

    def build_step_state(self, observation: np.ndarray, info: Mapping[str, Any]) -> StepState:
        metrics = _CrafterMetrics()

        # Extract inventory information
        inventory = info.get("inventory", {})
        if inventory:
            metrics.inventory = dict(inventory)
            metrics.health = inventory.get("health")
            metrics.food = inventory.get("food")
            metrics.drink = inventory.get("drink")
            metrics.energy = inventory.get("energy")

        # Extract achievements
        achievements = info.get("achievements", {})
        if achievements:
            metrics.episode_achievements = dict(achievements)
            metrics.achievements_unlocked = sum(1 for v in achievements.values() if v > 0)

        # Extract player position
        player_pos = info.get("player_pos")
        if player_pos is not None:
            metrics.player_pos = tuple(player_pos)

        environment_meta: dict[str, Any] = {"env_id": self._env_id}
        metrics_map: dict[str, Any] = {}

        if metrics.health is not None:
            metrics_map["health"] = metrics.health
        if metrics.food is not None:
            metrics_map["food"] = metrics.food
        if metrics.drink is not None:
            metrics_map["drink"] = metrics.drink
        if metrics.energy is not None:
            metrics_map["energy"] = metrics.energy
        if metrics.achievements_unlocked:
            metrics_map["achievements_unlocked"] = metrics.achievements_unlocked
        if metrics.player_pos:
            metrics_map["player_pos"] = metrics.player_pos

        raw_payload: dict[str, Any] = {}
        if metrics.inventory:
            raw_payload["inventory"] = metrics.inventory
        if metrics.episode_achievements:
            raw_payload["achievements"] = metrics.episode_achievements

        return StepState(
            metrics=metrics_map,
            environment=environment_meta,
            raw=raw_payload,
        )

    def build_render_hint(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
        state: StepState,
    ) -> Mapping[str, Any] | None:
        base_hint = super().build_render_hint(observation, info, state) or {}
        hint: dict[str, Any] = dict(base_hint)
        hint["image_shape"] = observation.shape if hasattr(observation, "shape") else None
        return hint or None

    def build_frame_reference(self, render_payload: Any | None, state: StepState) -> str | None:
        rgb_payload: np.ndarray | None = None
        if isinstance(render_payload, np.ndarray):
            rgb_payload = render_payload
        elif isinstance(render_payload, Mapping):
            candidate = render_payload.get("rgb")
            if isinstance(candidate, np.ndarray):
                rgb_payload = candidate

        if rgb_payload is None:
            if not self._render_warning_emitted:
                self._render_warning_emitted = True
                self.log_constant(
                    LOG_ENV_CRAFTER_RENDER_WARNING,
                    extra={
                        "env_id": self._env_id,
                        "payload_type": type(render_payload).__name__,
                    },
                )
            return None

        achievements = state.metrics.get("achievements_unlocked", 0)
        return f"frames/crafter/{achievements}_{self._step_counter}.png"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _process_observation(self, observation: Any) -> np.ndarray:
        """Process observation to ensure correct format."""
        return np.asarray(observation, dtype=np.uint8)

    def _check_achievements(self, info: Mapping[str, Any]) -> None:
        """Check for newly unlocked achievements and log them."""
        achievements = info.get("achievements", {})
        for name, count in achievements.items():
            prev_count = self._previous_achievements.get(name, 0)
            if count > prev_count:
                self.log_constant(
                    LOG_ENV_CRAFTER_ACHIEVEMENT,
                    extra={
                        "env_id": self._env_id,
                        "achievement": name,
                        "count": count,
                        "step": self._step_counter,
                    },
                )
        self._previous_achievements = dict(achievements)

    def get_action_name(self, action: int) -> str:
        """Get the human-readable name for an action."""
        if 0 <= action < len(CRAFTER_ACTIONS):
            return CRAFTER_ACTIONS[action]
        return f"unknown_{action}"

    @staticmethod
    def get_achievement_names() -> list[str]:
        """Get list of all achievement names."""
        return list(CRAFTER_ACHIEVEMENTS)


class CrafterRewardAdapter(CrafterAdapter):
    """Adapter specialising defaults for CrafterReward-v1 (with rewards)."""

    DEFAULT_ENV_ID = GameId.CRAFTER_REWARD.value


class CrafterNoRewardAdapter(CrafterAdapter):
    """Adapter specialising defaults for CrafterNoReward-v1 (reward-free)."""

    DEFAULT_ENV_ID = GameId.CRAFTER_NO_REWARD.value


CRAFTER_ADAPTERS: dict[GameId, type[CrafterAdapter]] = {
    GameId.CRAFTER_REWARD: CrafterRewardAdapter,
    GameId.CRAFTER_NO_REWARD: CrafterNoRewardAdapter,
}


__all__ = [
    "CrafterAdapter",
    "CrafterRewardAdapter",
    "CrafterNoRewardAdapter",
    "CrafterConfig",
    "CRAFTER_ADAPTERS",
    "CRAFTER_ACHIEVEMENTS",
    "CRAFTER_ACTIONS",
]
