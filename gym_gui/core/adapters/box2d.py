"""Box2D environment adapters that stream RGB frames into the Qt shell."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import gymnasium.spaces as spaces
from gym_gui.core.wrappers.time_limits import EpisodeTimeLimitSeconds, configure_step_limit

from gym_gui.config.game_configs import BipedalWalkerConfig, CarRacingConfig, LunarLanderConfig
from gym_gui.core.adapters.base import AdapterContext, AdapterStep, EnvironmentAdapter, StepState
from gym_gui.core.enums import ControlMode, GameId, RenderMode


class Box2DAdapter(EnvironmentAdapter[np.ndarray, Any]):
    """Shared behaviour for Box2D environments that render RGB arrays."""

    default_render_mode = RenderMode.RGB_ARRAY
    supported_control_modes = (ControlMode.AGENT_ONLY,)

    def render(self) -> dict[str, Any]:
        frame = super().render()
        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": frame,
            "game_id": self.id,
        }

    def build_step_state(self, observation: np.ndarray, info: Mapping[str, Any]) -> StepState:
        return StepState(
            metrics=self._extract_metrics(observation, info),
            raw=dict(info),
        )

    def _extract_metrics(self, observation: np.ndarray, info: Mapping[str, Any]) -> dict[str, Any]:
        return {}


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

    def step(self, action: Any) -> AdapterStep:
        """Override step to handle discrete keyboard actions in continuous mode."""
        # Convert discrete keyboard actions to continuous if needed
        if self._config.continuous and isinstance(action, (int, np.integer)):
            # Map discrete actions (0-3) to continuous control
            # 0: idle/no thrust, 1: left engine, 2: main engine, 3: right engine
            action = self._discrete_to_continuous(int(action))
        return super().step(action)

    @staticmethod
    def _discrete_to_continuous(discrete_action: int) -> np.ndarray:
        """Convert discrete keyboard action to continuous control array.
        
        Returns array [main_engine, side_engines] where:
        - main_engine: 0.0 to 1.0 (throttle)
        - side_engines: -1.0 to 1.0 (left/right)
        """
        mapping = {
            0: np.array([0.0, 0.0], dtype=np.float32),   # Idle
            1: np.array([0.0, -1.0], dtype=np.float32),  # Fire left engine
            2: np.array([1.0, 0.0], dtype=np.float32),   # Fire main engine
            3: np.array([0.0, 1.0], dtype=np.float32),   # Fire right engine
        }
        return mapping.get(discrete_action, np.array([0.0, 0.0], dtype=np.float32))

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

    _HUMAN_ACTION_PRESETS: dict[int, np.ndarray] = {
        0: np.array([0.0, 0.0, 0.0], dtype=np.float32),   # coast
        1: np.array([1.0, 0.3, 0.0], dtype=np.float32),   # steer right
        2: np.array([-1.0, 0.3, 0.0], dtype=np.float32),  # steer left
        3: np.array([0.0, 1.0, 0.0], dtype=np.float32),   # accelerate
        4: np.array([0.0, 0.0, 0.8], dtype=np.float32),   # brake
    }

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
                preset = self._HUMAN_ACTION_PRESETS.get(int(action))
                if preset is None:
                    raise ValueError(f"Unknown human action preset '{action}' for CarRacing")
                action = preset
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

    _HUMAN_ACTION_PRESETS: dict[int, np.ndarray] = {
        0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # neutral stance
        1: np.array([0.8, 0.6, -0.8, -0.6], dtype=np.float32),  # stride forward
        2: np.array([-0.8, -0.6, 0.8, 0.6], dtype=np.float32),  # stride backward
        3: np.array([0.4, -1.0, 0.4, -1.0], dtype=np.float32),  # crouch / dip
        4: np.array([-0.4, 1.0, -0.4, 1.0], dtype=np.float32),  # hop / push
    }

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
            preset = self._HUMAN_ACTION_PRESETS.get(int(action))
            if preset is None:
                raise ValueError(f"Unknown human action preset '{action}' for BipedalWalker")
            action = preset
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
