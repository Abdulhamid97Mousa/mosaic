"""MiniGrid environment adapters providing flattened observations.

This adapter mirrors the xuance baseline wrapper while integrating with the
Gym GUI lifecycle contracts (structured logging, step packaging, telemetry
metadata). It supports optional partial-observation and image wrappers while
exposing flattened RGB observations with appended agent direction, which keeps
the payload compact for trainer workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from gym_gui.config.game_configs import MiniGridConfig
from gym_gui.core.adapters.base import AdapterContext, AdapterStep, EnvironmentAdapter, StepState
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
    LOG_ENV_MINIGRID_BOOT,
    LOG_ENV_MINIGRID_ERROR,
    LOG_ENV_MINIGRID_RENDER_WARNING,
    LOG_ENV_MINIGRID_STEP,
)

try:  # pragma: no cover - import guard exercised in integration tests
    from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
except ImportError:  # pragma: no cover - handled gracefully at runtime
    ImgObsWrapper = None  # type: ignore[assignment]
    RGBImgPartialObsWrapper = None  # type: ignore[assignment]


_MINIGRID_STEP_LOG_FREQUENCY = 100


@dataclass(slots=True)
class _MiniGridMetrics:
    """Container describing MiniGrid-specific telemetry traits."""

    direction: int | None = None
    mission: str | None = None
    grid_width: int | None = None
    grid_height: int | None = None


class MiniGridAdapter(EnvironmentAdapter[np.ndarray, int]):
    """Adapter for MiniGrid environments with flattened RGB observations."""

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
        ControlMode.HYBRID_HUMAN_AGENT,
        ControlMode.MULTI_AGENT_COOP,
        ControlMode.MULTI_AGENT_COMPETITIVE,
    )

    DEFAULT_ENV_ID = GameId.MINIGRID_EMPTY_5x5.value

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MiniGridConfig | None = None,
    ) -> None:
        super().__init__(context)
        if config is None:
            config = MiniGridConfig(env_id=self.DEFAULT_ENV_ID)
        self._config = config
        self._env_id = config.env_id or self.DEFAULT_ENV_ID
        self._step_counter = 0
        self._render_warning_emitted = False

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def render(self) -> dict[str, Any]:
        env = self._require_env()
        frame = env.render()
        array = np.asarray(frame)
        base_env = self._resolve_base_env()
        mission = getattr(base_env, "mission", None)
        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": array,
            "game_id": self._env_id,
            "mission": mission,
        }

    @property
    def id(self) -> str:  # type: ignore[override]
        return self._env_id

    def gym_kwargs(self) -> dict[str, Any]:
        kwargs = super().gym_kwargs()
        kwargs.update(self._config.to_gym_kwargs())
        return kwargs

    def apply_wrappers(self, env: gym.Env[Any, Any]) -> gym.Env[Any, Any]:
        env = super().apply_wrappers(env)
        if self._config.partial_observation:
            if RGBImgPartialObsWrapper is None:
                raise RuntimeError(
                    "MiniGrid partial observation wrapper requested but minigrid package is missing."
                )
            env = RGBImgPartialObsWrapper(env)
        if self._config.image_observation:
            if ImgObsWrapper is None:
                raise RuntimeError(
                    "MiniGrid ImgObsWrapper requested but minigrid package is missing."
                )
            env = ImgObsWrapper(env)
        return env

    def load(self) -> None:
        try:
            super().load()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.log_constant(
                LOG_ENV_MINIGRID_ERROR,
                exc_info=exc,
                extra={
                    "env_id": self._env_id,
                    "stage": "load",
                },
            )
            raise
        base = self._resolve_base_env()
        metrics = self._snapshot_metrics(base)
        self.log_constant(
            LOG_ENV_MINIGRID_BOOT,
            extra={
                "env_id": self._env_id,
                "partial_obs": self._config.partial_observation,
                "img_obs": self._config.image_observation,
                "reward_multiplier": float(self._config.reward_multiplier),
                "agent_view_size": self._config.agent_view_size or "default",
                "grid_width": metrics.grid_width or "unknown",
                "grid_height": metrics.grid_height or "unknown",
            },
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[np.ndarray]:
        env = self._require_env()
        applied_seed = seed if seed is not None else self._config.seed
        observation, info = env.reset(seed=applied_seed, options=options)
        processed_obs, raw = self._process_observation(observation)
        info = dict(info)
        info["_minigrid_raw_observation"] = raw
        self._step_counter = 0
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
        observation, reward, terminated, truncated, info = env.step(action)
        processed_obs, raw = self._process_observation(observation)
        info = dict(info)
        info["_minigrid_raw_observation"] = raw
        scaled_reward = float(reward) * float(self._config.reward_multiplier)

        self._step_counter += 1
        if self._step_counter % _MINIGRID_STEP_LOG_FREQUENCY == 1:
            self.log_constant(
                LOG_ENV_MINIGRID_STEP,
                extra={
                    "env_id": self._env_id,
                    "step": self._step_counter,
                    "reward": scaled_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                },
            )

        return self._package_step(processed_obs, scaled_reward, terminated, truncated, info)

    # ------------------------------------------------------------------
    # Adapter customisations
    # ------------------------------------------------------------------

    def build_step_state(self, observation: np.ndarray, info: Mapping[str, Any]) -> StepState:
        raw = info.get("_minigrid_raw_observation")
        metrics = _MiniGridMetrics()
        if isinstance(raw, Mapping):
            direction = raw.get("direction")
            mission = raw.get("mission")
            metrics.direction = int(direction) if direction is not None else None
            metrics.mission = str(mission) if mission is not None else None
            metrics.grid_height = raw.get("grid_height")
            metrics.grid_width = raw.get("grid_width")

        environment_meta: dict[str, Any] = {"env_id": self._env_id}
        metrics_map: dict[str, Any] = {}
        if metrics.direction is not None:
            metrics_map["direction"] = metrics.direction
        if metrics.mission:
            environment_meta["mission"] = metrics.mission
        if metrics.grid_width is not None and metrics.grid_height is not None:
            environment_meta["grid_size"] = (
                int(metrics.grid_height),
                int(metrics.grid_width),
            )

        raw_payload: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}

        return StepState(
            metrics=metrics_map,
            environment=environment_meta,
            raw=dict(raw_payload) if raw_payload else {},
        )

    def build_render_hint(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
        state: StepState,
    ) -> Mapping[str, Any] | None:
        base_hint = super().build_render_hint(observation, info, state) or {}
        hint: dict[str, Any] = dict(base_hint)
        raw = info.get("_minigrid_raw_observation")
        if isinstance(raw, Mapping) and "image" in raw:
            env_section = dict(hint.get("environment", {}))
            env_section.setdefault("image_shape", np.asarray(raw["image"]).shape)
            hint["environment"] = env_section
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
                    LOG_ENV_MINIGRID_RENDER_WARNING,
                    extra={
                        "env_id": self._env_id,
                        "payload_type": type(render_payload).__name__,
                    },
                )
            return None

        return f"frames/minigrid/{state.metrics.get('direction', 0)}_{self._step_counter}.png"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _process_observation(self, observation: Any) -> tuple[np.ndarray, Mapping[str, Any]]:
        base = self._resolve_base_env()
        mission = getattr(base, "mission", None)
        grid = getattr(base, "grid", None)
        grid_height = getattr(grid, "height", None)
        grid_width = getattr(grid, "width", None)
        direction = getattr(base, "agent_dir", None)

        if isinstance(observation, Mapping):
            image = np.asarray(observation.get("image"), dtype=np.uint8)
            direction = observation.get("direction", direction)
        else:
            image = np.asarray(observation, dtype=np.uint8)

        flat = np.concatenate(
            (
                image.reshape(-1).astype(np.uint8, copy=False),
                np.array([int(direction or 0)], dtype=np.uint8),
            )
        ).astype(np.uint8, copy=False)
        raw = {
            "image": image,
            "direction": int(direction or 0),
            "mission": mission,
            "grid_height": grid_height,
            "grid_width": grid_width,
        }
        return flat, raw

    def _resolve_base_env(self) -> Any:
        env = self._require_env()
        visited: set[int] = set()
        current = env
        while True:
            next_env = getattr(current, "env", None)
            if next_env is None or id(next_env) in visited:
                break
            visited.add(id(current))
            current = next_env
        return getattr(current, "unwrapped", current)

    def _snapshot_metrics(self, base_env: Any) -> _MiniGridMetrics:
        metrics = _MiniGridMetrics()
        metrics.mission = getattr(base_env, "mission", None)
        metrics.direction = getattr(base_env, "agent_dir", None)
        grid = getattr(base_env, "grid", None)
        if grid is not None:
            metrics.grid_height = getattr(grid, "height", None)
            metrics.grid_width = getattr(grid, "width", None)
        return metrics


class MiniGridDoorKeyAdapter(MiniGridAdapter):
    """Adapter specialising defaults for MiniGrid-DoorKey-8x8-v0."""

    DEFAULT_ENV_ID = GameId.MINIGRID_DOORKEY_8x8.value


class MiniGridDoorKey5x5Adapter(MiniGridAdapter):
    """Adapter specialising defaults for MiniGrid-DoorKey-5x5-v0."""

    DEFAULT_ENV_ID = GameId.MINIGRID_DOORKEY_5x5.value


class MiniGridDoorKey6x6Adapter(MiniGridAdapter):
    """Adapter specialising defaults for MiniGrid-DoorKey-6x6-v0."""

    DEFAULT_ENV_ID = GameId.MINIGRID_DOORKEY_6x6.value


class MiniGridDoorKey16x16Adapter(MiniGridAdapter):
    """Adapter specialising defaults for MiniGrid-DoorKey-16x16-v0."""

    DEFAULT_ENV_ID = GameId.MINIGRID_DOORKEY_16x16.value


class MiniGridLavaGapAdapter(MiniGridAdapter):
    """Adapter specialising defaults for MiniGrid-LavaGapS7-v0."""

    DEFAULT_ENV_ID = GameId.MINIGRID_LAVAGAP_S7.value


class MiniGridDynamicObstaclesAdapter(MiniGridAdapter):
    """Adapter specialising defaults for MiniGrid-Dynamic-Obstacles-16x16-v0."""

    DEFAULT_ENV_ID = GameId.MINIGRID_DYNAMIC_OBSTACLES_16X16.value


class MiniGridBlockedUnlockPickupAdapter(MiniGridAdapter):
    """Adapter specialising defaults for MiniGrid-BlockedUnlockPickup-v0."""

    DEFAULT_ENV_ID = GameId.MINIGRID_BLOCKED_UNLOCK_PICKUP.value


class MiniGridMultiRoomAdapter(MiniGridAdapter):
    """Adapter specialising defaults for MiniGrid-MultiRoom-N6-v0."""

    DEFAULT_ENV_ID = GameId.MINIGRID_MULTIROOM_N6.value


class MiniGridObstructedMazeAdapter(MiniGridAdapter):
    """Adapter specialising defaults for MiniGrid Obstructed Maze environments."""

    DEFAULT_ENV_ID = GameId.MINIGRID_OBSTRUCTED_MAZE_FULL.value


MINIGRID_ADAPTERS: dict[GameId, type[MiniGridAdapter]] = {
    GameId.MINIGRID_EMPTY_5x5: MiniGridAdapter,
    GameId.MINIGRID_EMPTY_RANDOM_5x5: MiniGridAdapter,
    GameId.MINIGRID_EMPTY_6x6: MiniGridAdapter,
    GameId.MINIGRID_EMPTY_RANDOM_6x6: MiniGridAdapter,
    GameId.MINIGRID_EMPTY_8x8: MiniGridAdapter,
    GameId.MINIGRID_EMPTY_16x16: MiniGridAdapter,
    GameId.MINIGRID_DOORKEY_5x5: MiniGridDoorKey5x5Adapter,
    GameId.MINIGRID_DOORKEY_6x6: MiniGridDoorKey6x6Adapter,
    GameId.MINIGRID_DOORKEY_8x8: MiniGridDoorKeyAdapter,
    GameId.MINIGRID_DOORKEY_16x16: MiniGridDoorKey16x16Adapter,
    GameId.MINIGRID_LAVAGAP_S5: MiniGridLavaGapAdapter,
    GameId.MINIGRID_LAVAGAP_S6: MiniGridLavaGapAdapter,
    GameId.MINIGRID_LAVAGAP_S7: MiniGridLavaGapAdapter,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_5X5: MiniGridDynamicObstaclesAdapter,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_5X5: MiniGridDynamicObstaclesAdapter,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_6X6: MiniGridDynamicObstaclesAdapter,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_RANDOM_6X6: MiniGridDynamicObstaclesAdapter,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_8X8: MiniGridDynamicObstaclesAdapter,
    GameId.MINIGRID_DYNAMIC_OBSTACLES_16X16: MiniGridDynamicObstaclesAdapter,
    GameId.MINIGRID_BLOCKED_UNLOCK_PICKUP: MiniGridBlockedUnlockPickupAdapter,
    GameId.MINIGRID_MULTIROOM_N2_S4: MiniGridMultiRoomAdapter,
    GameId.MINIGRID_MULTIROOM_N4_S5: MiniGridMultiRoomAdapter,
    GameId.MINIGRID_MULTIROOM_N6: MiniGridMultiRoomAdapter,
    GameId.MINIGRID_OBSTRUCTED_MAZE_1DLHB: MiniGridObstructedMazeAdapter,
    GameId.MINIGRID_OBSTRUCTED_MAZE_FULL: MiniGridObstructedMazeAdapter,
}


__all__ = [
    "MiniGridAdapter",
    "MiniGridDoorKey5x5Adapter",
    "MiniGridDoorKey6x6Adapter",
    "MiniGridDoorKeyAdapter",
    "MiniGridDoorKey16x16Adapter",
    "MiniGridLavaGapAdapter",
    "MiniGridDynamicObstaclesAdapter",
    "MiniGridBlockedUnlockPickupAdapter",
    "MiniGridMultiRoomAdapter",
    "MiniGridObstructedMazeAdapter",
    "MINIGRID_ADAPTERS",
]
