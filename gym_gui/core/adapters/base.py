"""Abstract adapter contract for Gymnasium environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Any, Callable, Generic, Mapping, TypeVar

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
class AdapterStep(Generic[ObservationT]):
    """Standardised step result consumed by orchestrators."""

    observation: ObservationT
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    render_payload: Any | None = None


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

        env = gym.make(self.id, render_mode=self.default_render_mode.value)
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
        )

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
]
