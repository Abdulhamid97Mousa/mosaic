"""Box2D environment adapters that stream RGB frames into the Qt shell."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import gymnasium.spaces as spaces
from gym_gui.core.wrappers.time_limits import EpisodeTimeLimitSeconds, configure_step_limit

from gym_gui.config.game_configs import BipedalWalkerConfig, CarRacingConfig, LunarLanderConfig
from gym_gui.core.adapters.base import AdapterContext, AdapterStep, EnvironmentAdapter, StepState
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.services.action_mapping import ContinuousActionMapper
from gym_gui.services.service_locator import get_service_locator
from gym_gui.logging_config.log_constants import LOG_ADAPTER_STEP_SUMMARY


class Box2DAdapter(EnvironmentAdapter[np.ndarray, Any]):
    """Shared behaviour for Box2D environments that render RGB arrays."""

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (ControlMode.AGENT_ONLY,)

    def render(self) -> dict[str, Any]:
        frame = super().render()
        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": frame,
            "game_id": self.id,
        }

    def build_frame_reference(self, render_payload: Any | None, state: StepState) -> str | None:
        """Generate timestamped frame reference for media storage.

        Args:
            render_payload: The render payload (unused for Box2D)
            state: The step state (unused for Box2D)

        Returns:
            Timestamped frame reference string or None if payload is invalid
        """
        if render_payload is None:
            return None

        # Generate timestamp: YYYY-MM-DD_HH-MM-SS_NNN
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        microseconds = now.microsecond // 1000  # Convert to milliseconds

        return f"frames/{timestamp}_{microseconds:03d}.png"

    def build_step_state(self, observation: np.ndarray, info: Mapping[str, Any]) -> StepState:
        return StepState(
            metrics=self._extract_metrics(observation, info),
            raw=dict(info),
        )

    def _extract_metrics(self, observation: np.ndarray, info: Mapping[str, Any]) -> dict[str, Any]:
        return {}

    @staticmethod
    def _resolve_action_mapper() -> ContinuousActionMapper | None:
        locator = get_service_locator()
        return locator.resolve(ContinuousActionMapper)

    def _map_discrete_action(
        self,
        game_id: GameId,
        discrete_action: int,
        *,
        fallback: np.ndarray | None = None,
    ) -> np.ndarray | None:
        mapper = self._resolve_action_mapper()
        if mapper is None:
            return fallback.copy() if fallback is not None else None
        vector = mapper.map(game_id, discrete_action)
        if vector is None:
            return fallback.copy() if fallback is not None else None
        return vector


class LunarLanderAdapter(Box2DAdapter):
    """Adapter for the LunarLander-v3 environment."""

    id = GameId.LUNAR_LANDER.value
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: LunarLanderConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config = config or LunarLanderConfig()

    def gym_kwargs(self) -> dict[str, Any]:
        kwargs = super().gym_kwargs()
        kwargs.update(self._config.to_gym_kwargs())
        return kwargs

    def apply_wrappers(self, env: gym.Env[Any, Any]) -> gym.Env[Any, Any]:
        env = super().apply_wrappers(env)
        max_steps = self._config.sanitized_step_limit()
        env = configure_step_limit(env, max_steps)
        return env

    def step(self, action: Any) -> AdapterStep:
        """Override step to handle discrete keyboard actions in continuous mode."""
        # Convert discrete keyboard actions to continuous if needed
        if self._config.continuous and isinstance(action, (int, np.integer)):
            mapped = self._map_discrete_action(
                GameId.LUNAR_LANDER,
                int(action),
                fallback=np.zeros(2, dtype=np.float32),
            )
            if mapped is not None:
                action = mapped
            else:
                self.log_constant(
                    LOG_ADAPTER_STEP_SUMMARY,
                    message="continuous_mapping_missing",
                    extra={"env_id": self.id, "fallback": "idle_thrust"},
                )
                action = np.zeros(2, dtype=np.float32)
        return super().step(action)

    def _extract_metrics(self, observation: np.ndarray, info: Mapping[str, Any]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if observation is not None:
            arr = np.asarray(observation, dtype=float)
            if arr.shape and arr.shape[0] >= 8:
                metrics.update(
                    x=arr[0],
                    y=arr[1],
                    velocity_x=arr[2],
                    velocity_y=arr[3],
                    angle=arr[4],
                    angular_velocity=arr[5],
                    leg_left_contact=bool(arr[6] > 0.0),
                    leg_right_contact=bool(arr[7] > 0.0),
                )
        if info:
            shaping = info.get("shaping")
            if shaping is not None:
                metrics["shaping"] = float(shaping)
            landing = info.get("landed")
            if landing is not None:
                metrics["landed"] = bool(landing)
        return metrics


class CarRacingAdapter(Box2DAdapter):
    """Adapter for the CarRacing-v3 environment."""

    id = GameId.CAR_RACING.value
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: CarRacingConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config = config or CarRacingConfig()

    def gym_kwargs(self) -> dict[str, Any]:
        kwargs = super().gym_kwargs()
        kwargs.update(self._config.to_gym_kwargs())
        return kwargs

    def apply_wrappers(self, env: gym.Env[Any, Any]) -> gym.Env[Any, Any]:
        env = super().apply_wrappers(env)
        max_steps, max_seconds = self._config.sanitized_time_limits()
        env = configure_step_limit(env, max_steps)
        if max_seconds is not None:
            env = EpisodeTimeLimitSeconds(env, max_seconds)
        return env

    def step(self, action: np.ndarray | int) -> AdapterStep[np.ndarray]:
        env_space = None
        try:
            env_space = self.action_space
        except Exception:
            env_space = None
        is_continuous = isinstance(env_space, spaces.Box)

        if isinstance(action, (int, np.integer)):
            if is_continuous:
                mapped = self._map_discrete_action(
                    GameId.CAR_RACING,
                    int(action),
                    fallback=np.zeros(3, dtype=np.float32),
                )
                if mapped is None:
                    raise ValueError(f"Unknown human action preset '{action}' for CarRacing")
                action = mapped
            else:
                action = int(action)
        else:
            action_array = np.asarray(action, dtype=np.float32)
            if is_continuous:
                action = action_array
            else:
                if action_array.size != 0:
                    action = int(action_array.item())
                else:
                    action = 0
        return super().step(action)

    def build_step_state(self, observation: np.ndarray, info: Mapping[str, Any]) -> StepState:
        metrics: dict[str, Any] = {}
        for key in ("speed", "lap", "tile_visited_count", "tile_visited_count_2", "reward"):
            value = info.get(key)
            if value is not None:
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    metrics[key] = value
        return StepState(metrics=metrics, raw=dict(info))


class BipedalWalkerAdapter(Box2DAdapter):
    """Adapter for the BipedalWalker-v3 environment."""

    id = GameId.BIPEDAL_WALKER.value
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: BipedalWalkerConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config = config or BipedalWalkerConfig()

    def gym_kwargs(self) -> dict[str, Any]:
        kwargs = super().gym_kwargs()
        kwargs.update(self._config.to_gym_kwargs())
        return kwargs

    def step(self, action: np.ndarray | int) -> AdapterStep[np.ndarray]:
        env_space = None
        try:
            env_space = self.action_space
        except Exception:
            env_space = None
        is_continuous = isinstance(env_space, spaces.Box)

        if isinstance(action, (int, np.integer)):
            mapped = self._map_discrete_action(
                GameId.BIPEDAL_WALKER,
                int(action),
                fallback=np.zeros(4, dtype=np.float32),
            )
            if mapped is None:
                raise ValueError(f"Unknown human action preset '{action}' for BipedalWalker")
            action = mapped
        else:
            action_array = np.asarray(action, dtype=np.float32)
            if is_continuous:
                if env_space is not None and isinstance(env_space, spaces.Box):
                    low = np.asarray(env_space.low, dtype=np.float32)
                    high = np.asarray(env_space.high, dtype=np.float32)
                    action = np.clip(action_array, low, high)
                else:
                    action = action_array
            else:
                action = int(action_array.item()) if action_array.size else 0
        return super().step(action)

    def _extract_metrics(self, observation: np.ndarray, info: Mapping[str, Any]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if observation is not None:
            arr = np.asarray(observation, dtype=float)
            if arr.shape and arr.size >= 24:
                metrics.update(
                    hull_angle=arr[0],
                    hull_angular_velocity=arr[1],
                    horizontal_velocity=arr[2],
                    vertical_velocity=arr[3],
                    hip_joint_1=arr[4],
                    knee_joint_1=arr[5],
                    hip_joint_2=arr[6],
                    knee_joint_2=arr[7],
                    leg_contact_1=bool(arr[8] > 0.0),
                    leg_contact_2=bool(arr[9] > 0.0),
                )
        if info:
            for key in ("reward_run", "reward_ctrl", "state_velocity"):
                if key in info and info[key] is not None:
                    try:
                        metrics[key] = float(info[key])
                    except (TypeError, ValueError):
                        metrics[key] = info[key]
        return metrics


BOX2D_ADAPTERS: dict[GameId, type[Box2DAdapter]] = {
    GameId.LUNAR_LANDER: LunarLanderAdapter,
    GameId.CAR_RACING: CarRacingAdapter,
    GameId.BIPEDAL_WALKER: BipedalWalkerAdapter,
}

__all__ = [
    "Box2DAdapter",
    "LunarLanderAdapter",
    "CarRacingAdapter",
    "BipedalWalkerAdapter",
    "BOX2D_ADAPTERS",
]
