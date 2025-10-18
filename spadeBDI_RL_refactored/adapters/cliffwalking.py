"""Adapter wrapping Gymnasium's CliffWalking-v1 for unified runtime use."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


@dataclass(slots=True)
class CliffWalkingAdapter:
    """Provide consistent semantics over Gymnasium's CliffWalking-v1 environment."""

    _env: gym.Env = field(init=False, repr=False)
    _action_meanings: list[str] = field(init=False, repr=False)
    _nrow: int = field(init=False, repr=False, default=4)
    _ncol: int = field(init=False, repr=False, default=12)

    def __post_init__(self) -> None:
        self._env = gym.make(
            "CliffWalking-v1",
            render_mode=None,
        )
        self._action_meanings = ["UP", "RIGHT", "DOWN", "LEFT"]
        
        # CliffWalking is always 4 rows × 12 columns
        self._nrow = 4
        self._ncol = 12

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
        """Return the goal position (always bottom-right corner)."""
        return (self._nrow - 1, self._ncol - 1)

    def start_pos(self) -> Tuple[int, int]:
        """Return the start position (always bottom-left corner)."""
        return (self._nrow - 1, 0)

    def is_cliff(self, row: int, col: int) -> bool:
        """Check if a position is a cliff (bottom row, excluding start and goal)."""
        return row == self._nrow - 1 and 0 < col < self._ncol - 1

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
        start_row, start_col = self.start_pos()
        
        result = {
            "state": int(state),
            "position": {"row": int(row), "col": int(col)},
            "goal": {"row": int(goal_row), "col": int(goal_col)},
            "start": {"row": int(start_row), "col": int(start_col)},
            "is_cliff": self.is_cliff(row, col),
        }
        
        if info:
            result.update(info)
        
        return result
