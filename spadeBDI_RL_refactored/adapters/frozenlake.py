"""Adapter wrapping Gymnasium's FrozenLake-v1 and FrozenLake-v2 for unified runtime use."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


@dataclass(slots=True)
class FrozenLakeAdapter:
    """Provide consistent semantics over Gymnasium's FrozenLake-v1 environment."""

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


@dataclass(slots=True)
class FrozenLakeV2Adapter:
    """Provide consistent semantics over Gymnasium's FrozenLake-v2 environment."""

    grid_height: int = 8
    grid_width: int = 8
    is_slippery: bool = True
    start_position: Tuple[int, int] | None = None
    goal_position: Tuple[int, int] | None = None
    hole_count: int | None = None
    random_holes: bool = False
    _env: gym.Env = field(init=False, repr=False)
    _action_meanings: list[str] = field(init=False, repr=False)
    _nrow: int = field(init=False, repr=False)
    _ncol: int = field(init=False, repr=False)
    _custom_desc: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._nrow = self.grid_height
        self._ncol = self.grid_width
        self._custom_desc = self._generate_map_descriptor()
        
        self._env = gym.make(
            "FrozenLake-v1",
            desc=self._custom_desc,
            is_slippery=self.is_slippery,
            render_mode=None,
        )
        self._action_meanings = ["LEFT", "DOWN", "RIGHT", "UP"]

    def _generate_map_descriptor(self) -> list[str]:
        """Generate custom map descriptor based on configuration."""
        start_pos = self.start_position or (0, 0)
        goal_pos = self.goal_position or (self._nrow - 1, self._ncol - 1)
        hole_count = self.hole_count
        
        # Default hole count if not specified
        if hole_count is None:
            total_tiles = self.grid_height * self.grid_width
            if self.grid_height == 4 and self.grid_width == 4:
                hole_count = 4  # Gymnasium default 4×4 map
            elif self.grid_height == 8 and self.grid_width == 8:
                hole_count = 10  # Gymnasium default 8×8 map
            else:
                hole_count = max(1, int(total_tiles * 0.15))  # ~15% holes
        
        # Initialize grid with frozen tiles
        grid = [['F' for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        # Place start and goal
        grid[start_pos[0]][start_pos[1]] = 'S'
        grid[goal_pos[0]][goal_pos[1]] = 'G'
        
        # Collect available positions for holes (exclude start and goal)
        available_positions = [
            (r, c) for r in range(self.grid_height) for c in range(self.grid_width)
            if (r, c) != start_pos and (r, c) != goal_pos
        ]
        
        # Randomly place holes
        hole_count = min(hole_count, len(available_positions))
        hole_positions = random.sample(available_positions, hole_count)
        
        for r, c in hole_positions:
            grid[r][c] = 'H'
        
        # Convert to list of strings
        return [''.join(row) for row in grid]

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


