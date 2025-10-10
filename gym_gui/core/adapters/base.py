"""Abstract adapter contract for Gymnasium environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

import gymnasium as gym

from gym_gui.core.enums import ControlMode, RenderMode

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


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
    state: StepState = field(default_factory=StepState)


class AdapterNotReadyError(RuntimeError):
    """Raised when an adapter is used before `load` has been called."""


class UnsupportedModeError(RuntimeError):
    """Raised when a requested control mode is incompatible with the adapter."""


class EnvironmentAdapter(ABC, Generic[ObservationT, ActionT]):
    """Lifecycle contract for all Gymnasium environment adapters."""

    id: str
    supported_control_modes: tuple[ControlMode, ...]
    default_render_mode: RenderMode

    def __init__(self, context: AdapterContext | None = None) -> None:
        self._context = context
        self._env: gym.Env[Any, Any] | None = None
        self._logger = (
            context.get_logger(self.__class__.__name__) if context else logging.getLogger(self.__class__.__name__)
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def bind(self, context: AdapterContext) -> None:
        """Bind the adapter to a runtime context after instantiation."""

        self._context = context
        self._logger = context.get_logger(self.__class__.__name__)

    def load(self) -> None:
        """Instantiate underlying Gymnasium environment resources."""

        kwargs: dict[str, Any] = {"render_mode": self.default_render_mode.value}
        extra_kwargs = self.gym_kwargs()
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        env = gym.make(self.id, **kwargs)
        env = self.apply_wrappers(env)
        self._logger.debug("Created Gymnasium environment '%s'", self.id)
        self._set_env(env)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> AdapterStep[ObservationT]:
        env = self._require_env()
        observation, info = env.reset(seed=seed, options=options)
        self._logger.debug("Reset environment '%s' with seed=%s", self.id, seed)
        return self._package_step(observation, 0.0, False, False, info)

    def step(self, action: ActionT) -> AdapterStep[ObservationT]:
        env = self._require_env()
        observation, reward, terminated, truncated, info = env.step(action)
        self._logger.debug(
            "Step env='%s' action=%s -> reward=%s terminated=%s truncated=%s",
            self.id,
            action,
            reward,
            terminated,
            truncated,
        )
        return self._package_step(observation, float(reward), terminated, truncated, info)

    def close(self) -> None:
        if self._env is not None:
            self._logger.debug("Closing environment '%s'", self.id)
            self._env.close()
            self._env = None

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    def _require_env(self) -> gym.Env[Any, Any]:
        if self._env is None:
            raise AdapterNotReadyError(f"Adapter '{self.id}' has not been loaded.")
        return self._env

    def _set_env(self, env: gym.Env[Any, Any]) -> None:
        self._env = env

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
        return AdapterStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            render_payload=self.render(),
            state=self.build_step_state(observation, info),
        )

    def build_step_state(self, observation: ObservationT, info: Mapping[str, Any]) -> StepState:
        """Construct the canonical :class:`StepState` for the current step."""

        return StepState()

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

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def context(self) -> AdapterContext | None:
        return self._context

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def settings(self) -> Any | None:
        return self._context.settings if self._context else None

    @property
    def action_space(self) -> gym.Space[Any]:
        return self._require_env().action_space

    @property
    def observation_space(self) -> gym.Space[Any]:
        return self._require_env().observation_space


__all__ = [
    "AdapterContext",
    "AdapterStep",
    "AdapterNotReadyError",
    "UnsupportedModeError",
    "EnvironmentAdapter",
    "StepState",
    "AgentSnapshot",
]
