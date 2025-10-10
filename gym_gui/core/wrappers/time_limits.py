"""Custom time-limit utilities and wrappers for Gymnasium environments."""

from __future__ import annotations

import math
import time
from typing import Any, Tuple

import gymnasium as gym
from gymnasium.wrappers import TimeLimit


class EpisodeTimeLimitSeconds(gym.Wrapper):
    """Truncates episodes that exceed a wall-clock time budget."""

    def __init__(self, env: gym.Env[Any, Any], max_episode_seconds: float) -> None:
        if max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive")
        super().__init__(env)
        self.max_episode_seconds = float(max_episode_seconds)
        self._start_time: float | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._start_time = time.perf_counter()
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        if not terminated and not truncated and self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            if elapsed >= self.max_episode_seconds:
                truncated = True
                info = dict(info)
                info["time_limit_reached"] = True
                info["time_limit_seconds"] = self.max_episode_seconds
        return observation, reward, terminated, truncated, info


def configure_step_limit(env: gym.Env[Any, Any], max_episode_steps: int | None) -> gym.Env[Any, Any]:
    """Adjust or install a :class:`TimeLimit` wrapper for step budgets."""

    wrapper = _find_time_limit(env)
    if wrapper is not None:
        if max_episode_steps is None:
            wrapper._max_episode_steps = math.inf  # type: ignore[attr-defined]
        else:
            wrapper._max_episode_steps = int(max_episode_steps)  # type: ignore[attr-defined]
        return env

    if max_episode_steps is not None:
        return TimeLimit(env, max_episode_steps=int(max_episode_steps))
    return env


def _find_time_limit(env: gym.Env[Any, Any]) -> TimeLimit | None:
    current = env
    visited: set[int] = set()
    while True:
        if id(current) in visited:
            break
        visited.add(id(current))
        if isinstance(current, TimeLimit):
            return current
        if not hasattr(current, "env"):
            break
        nested = getattr(current, "env")
        if nested is current:
            break
        current = nested  # type: ignore[assignment]
    return None


__all__ = ["EpisodeTimeLimitSeconds", "configure_step_limit"]
