"""Adapter wrapping Gymnasium's FrozenLake for unified runtime use."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


@dataclass(slots=True)
class FrozenLakeAdapter:
    """Provide consistent semantics over Gymnasium's FrozenLake environment."""

    map_size: str = "8x8"
    is_slippery: bool = True
    _env: gym.Env = field(init=False, repr=False)
    _action_meanings: list[str] = field(init=False, repr=False)
    _nrow: int = field(init=False, repr=False)
    _ncol: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        map_name = self.map_size.replace("x", "x")  # e.g., "8x8" or "4x4"
        self._env = gym.make(
            "FrozenLake-v1",
            map_name=map_name,
            is_slippery=self.is_slippery,
            render_mode=None,
        )
        self._action_meanings = ["LEFT", "DOWN", "RIGHT", "UP"]
        
        # Extract grid dimensions from the environment
        desc = self._env.unwrapped.desc  # type: ignore[attr-defined]
        self._nrow = desc.shape[0]
        self._ncol = desc.shape[1]

    # Public API ---------------------------------------------------------
    @property
    def action_space_n(self) -> int:
        return int(self._env.action_space.n)  # type: ignore[attr-defined]

    @property
    def observation_space_n(self) -> int:
        return int(self._env.observation_space.n)  # type: ignore[attr-defined]

    def reset(self, *, seed: int | None = None) -> Tuple[int, Dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
        
        obs, info = self._env.reset(seed=seed)
        state = int(obs)
        return state, self._obs_dict(state, info)

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        state = int(obs)
        return state, float(reward), bool(terminated), bool(truncated), self._obs_dict(state, info)

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert flat state index to (row, col) position."""
        row = state // self._ncol
        col = state % self._ncol
        return (int(row), int(col))

    def goal_pos(self) -> Tuple[int, int]:
        """Find the goal position by scanning the map."""
        desc = self._env.unwrapped.desc  # type: ignore[attr-defined]
        goal_positions = np.where(desc == b'G')
        if len(goal_positions[0]) > 0:
            return (int(goal_positions[0][0]), int(goal_positions[1][0]))
        return (self._nrow - 1, self._ncol - 1)  # Default to bottom-right

    def action_meanings(self) -> list[str]:
        return list(self._action_meanings)

    def close(self) -> None:
        """Clean up the environment."""
        if self._env is not None:
            self._env.close()

    # Internal helpers ---------------------------------------------------
    def _obs_dict(self, state: int, info: Dict[str, Any] | None = None) -> Dict[str, Any]:
        row, col = self.state_to_pos(state)
        goal_row, goal_col = self.goal_pos()
        
        result = {
            "state": int(state),
            "position": {"row": int(row), "col": int(col)},
            "goal": {"row": int(goal_row), "col": int(goal_col)},
        }
        
        if info:
            result.update(info)
        
        return result

