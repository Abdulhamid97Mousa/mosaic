"""Abstract adapter contract for Gymnasium environments."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

import gymnasium as gym

from gym_gui.core.enums import ControlMode, RenderMode
from gym_gui.core.spaces.serializer import describe_space
from gym_gui.core.spaces.vector_metadata import describe_vector_environment
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_CLOSED,
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_INIT_ERROR,
    LOG_ADAPTER_PAYLOAD_ERROR,
    LOG_ADAPTER_RENDER_ERROR,
    LOG_ADAPTER_STEP_ERROR,
    LOG_ADAPTER_STEP_SUMMARY,
    LOG_ADAPTER_STATE_INVALID,
    LogConstant,
)
from gym_gui.logging_config.helpers import LogConstantMixin

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AdapterContext:
    """Context payload adapters can use for configuration and callbacks."""

    settings: Any
    control_mode: ControlMode
    logger_factory: Callable[[str], logging.Logger] | None = None

    def get_logger(self, name: str) -> logging.Logger:
        if self.logger_factory is not None:
            return self.logger_factory(name)
        return logging.getLogger(name)


@dataclass(slots=True)
class AgentSnapshot:
    """State for a single agent participant in the environment."""

    name: str
    role: str | None = None
    position: tuple[int, int] | None = None
    orientation: str | None = None
    info: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "position": self.position,
            "orientation": self.orientation,
            "info": dict(self.info),
        }


@dataclass(slots=True)
class StepState:
    """Machine-readable snapshot of an environment step."""

    active_agent: str | None = None
    agents: Sequence[AgentSnapshot] = field(default_factory=tuple)
    objectives: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    hazards: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    inventory: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation for policies and UI."""

        return {
            "active_agent": self.active_agent,
            "agents": [agent.as_dict() for agent in self.agents],
            "objectives": [dict(obj) for obj in self.objectives],
            "hazards": [dict(hazard) for hazard in self.hazards],
            "inventory": dict(self.inventory),
            "metrics": dict(self.metrics),
            "environment": dict(self.environment),
            "raw": dict(self.raw),
        }


@dataclass(slots=True)
class AdapterStep(Generic[ObservationT]):
    """Standardised step result consumed by orchestrators."""

    observation: ObservationT
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    render_payload: Any | None = None
    render_hint: Mapping[str, Any] | None = None
    agent_id: str | None = None
    frame_ref: str | None = None
    payload_version: int = 1
    state: StepState = field(default_factory=StepState)


class AdapterNotReadyError(RuntimeError):
    """Raised when an adapter is used before `load` has been called."""


class UnsupportedModeError(RuntimeError):
    """Raised when a requested control mode is incompatible with the adapter."""


class EnvironmentAdapter(ABC, Generic[ObservationT, ActionT], LogConstantMixin):
    """Lifecycle contract for all Gymnasium environment adapters."""

    id: str
    supported_control_modes: tuple[ControlMode, ...]
    supported_render_modes: tuple[RenderMode, ...] = ()
    default_render_mode: RenderMode

    def __init__(self, context: AdapterContext | None = None) -> None:
        self._context = context
        self._logger = _LOGGER
        self._env: gym.Env[Any, Any] | None = None
        self._space_signature: Mapping[str, Any] | None = None
        self._vector_metadata: Mapping[str, Any] | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def bind(self, context: AdapterContext) -> None:
        """Bind the adapter to a runtime context after instantiation."""

        self._context = context

    def load(self) -> None:
        """Instantiate underlying Gymnasium environment resources."""

        kwargs: dict[str, Any] = {"render_mode": self.default_render_mode.value}
        extra_kwargs = self.gym_kwargs()
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        env = gym.make(self.id, **kwargs)
        env = self.apply_wrappers(env)
        self.log_constant(
            LOG_ADAPTER_ENV_CREATED,
            extra={
                "env_id": self.id,
                "render_mode": self.default_render_mode.value,
                "gym_kwargs": ",".join(sorted(extra_kwargs.keys())) if extra_kwargs else "-",
                "wrapped_class": env.__class__.__name__,
            },
        )
        self._set_env(env)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> AdapterStep[ObservationT]:
        env = self._require_env()
        observation, info = env.reset(seed=seed, options=options)
        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "env_id": self.id,
                "seed": seed if seed is not None else "None",
                "has_options": bool(options),
            },
        )
        return self._package_step(observation, 0.0, False, False, info)

    def step(self, action: ActionT) -> AdapterStep[ObservationT]:
        env = self._require_env()
        observation, reward, terminated, truncated, info = env.step(action)
        self.log_constant(
            LOG_ADAPTER_STEP_SUMMARY,
            extra={
                "env_id": self.id,
                "action": repr(action),
                "reward": float(reward) if isinstance(reward, (int, float)) else repr(reward),
                "terminated": terminated,
                "truncated": truncated,
            },
        )
        return self._package_step(observation, float(reward), terminated, truncated, info)

    def close(self) -> None:
        if self._env is not None:
            self.log_constant(
                LOG_ADAPTER_ENV_CLOSED,
                extra={"env_id": self.id},
            )
            self._env.close()
            self._env = None
            self._space_signature = None
            self._vector_metadata = None

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    def _require_env(self) -> gym.Env[Any, Any]:
        if self._env is None:
            raise AdapterNotReadyError(f"Adapter '{self.id}' has not been loaded.")
        return self._env

    def _set_env(self, env: gym.Env[Any, Any]) -> None:
        self._env = env
        self._space_signature = self._build_space_signature(env)
        self._vector_metadata = describe_vector_environment(env)

    def render(self) -> Any:
        env = self._require_env()
        return env.render()

    def gym_kwargs(self) -> dict[str, Any]:
        """Keyword arguments forwarded to :func:`gymnasium.make`."""

        return {}

    def apply_wrappers(self, env: gym.Env[Any, Any]) -> gym.Env[Any, Any]:
        """Hook for subclasses to apply Gymnasium wrappers before use."""

        return env

    def _package_step(
        self,
        observation: ObservationT,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> AdapterStep[ObservationT]:
        state = self.build_step_state(observation, info)
        render_payload: Any | None = None
        try:
            render_payload = self.render()
        except Exception as exc:
            self.log_constant(
                LOG_ADAPTER_RENDER_ERROR,
                exc_info=exc,
                extra={
                    "env_id": self.id,
                    "state_snapshot": bool(state.raw),
                },
            )
        render_hint = None
        try:
            render_hint = self.build_render_hint(observation, info, state)
        except Exception as exc:
            self.log_constant(
                LOG_ADAPTER_PAYLOAD_ERROR,
                exc_info=exc,
                extra={
                    "env_id": self.id,
                    "context": "render_hint",
                },
            )
        frame_ref = None
        try:
            frame_ref = self.build_frame_reference(render_payload, state)
        except Exception as exc:
            self.log_constant(
                LOG_ADAPTER_PAYLOAD_ERROR,
                exc_info=exc,
                extra={
                    "env_id": self.id,
                    "context": "frame_reference",
                },
            )
        return AdapterStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            render_payload=render_payload,
            render_hint=render_hint,
            agent_id=state.active_agent,
            frame_ref=frame_ref,
            payload_version=self.telemetry_payload_version(),
            state=state,
        )

    def build_step_state(self, observation: ObservationT, info: Mapping[str, Any]) -> StepState:
        """Construct the canonical :class:`StepState` for the current step."""

        return StepState()

    def build_render_hint(
        self,
        observation: ObservationT,
        info: Mapping[str, Any],
        state: StepState,
    ) -> Mapping[str, Any] | None:
        """Return lightweight render metadata for downstream consumers."""

        hint: dict[str, Any] = {}
        if state.active_agent:
            hint["active_agent"] = state.active_agent
        if state.metrics:
            hint["metrics"] = dict(state.metrics)
        if state.environment:
            hint["environment"] = dict(state.environment)
        if state.inventory:
            hint["inventory"] = dict(state.inventory)
        return hint or None

    def build_frame_reference(self, render_payload: Any | None, state: StepState) -> str | None:
        """Optional hook to derive an external frame reference for media pipelines."""

        del render_payload, state
        return None

    def telemetry_payload_version(self) -> int:
        """Version marker for downstream telemetry consumers."""

        return 1

    # ------------------------------------------------------------------
    # Optional utilities
    # ------------------------------------------------------------------

    def supports_control_mode(self, mode: ControlMode) -> bool:
        return mode in self.supported_control_modes

    def ensure_control_mode(self, mode: ControlMode) -> None:
        if not self.supports_control_mode(mode):
            raise UnsupportedModeError(
                f"Adapter '{self.id}' does not support control mode '{mode.value}'."
            )

    def supports_render_mode(self, mode: RenderMode) -> bool:
        if self.supported_render_modes:
            return mode in self.supported_render_modes
        return mode == self.default_render_mode

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def context(self) -> AdapterContext | None:
        return self._context

    @property
    def logger(self) -> logging.Logger:
        """Return the module-level logger for backward compatibility."""
        return _LOGGER

    @property
    def settings(self) -> Any | None:
        return self._context.settings if self._context else None

    @property
    def action_space(self) -> gym.Space[Any]:
        return self._require_env().action_space

    @property
    def observation_space(self) -> gym.Space[Any]:
        return self._require_env().observation_space

    @property
    def space_signature(self) -> Mapping[str, Any] | None:
        if self._space_signature is None and self._env is not None:
            self._space_signature = self._build_space_signature(self._env)
        return self._space_signature

    @property
    def vector_metadata(self) -> Mapping[str, Any] | None:
        if self._vector_metadata is None and self._env is not None:
            self._vector_metadata = describe_vector_environment(self._env)
        return self._vector_metadata

    def elapsed_steps(self) -> int | None:
        env = self._env
        visited: set[int] = set()
        while env is not None:
            elapsed = getattr(env, "_elapsed_steps", None)
            if elapsed is not None:
                try:
                    return int(elapsed)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    return None
            next_env = getattr(env, "unwrapped", None)
            if next_env is None or id(next_env) in visited:
                break
            visited.add(id(env))
            env = next_env
        return None

    def _build_space_signature(self, env: gym.Env[Any, Any]) -> Mapping[str, Any] | None:
        try:
            observation = describe_space(env.observation_space)
            action = describe_space(env.action_space)
        except Exception:  # pragma: no cover - best-effort metadata capture
            return None
        return {"observation": observation, "action": action}


__all__ = [
    "AdapterContext",
    "AdapterStep",
    "AdapterNotReadyError",
    "UnsupportedModeError",
    "EnvironmentAdapter",
    "StepState",
    "AgentSnapshot",
]
