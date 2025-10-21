"""Adapter wrapping Gymnasium's Taxi-v3 for unified runtime use."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


@dataclass(slots=True)
class TaxiAdapter:
    """Provide consistent semantics over Gymnasium's Taxi-v3 environment."""

    game_config: Any = None  # Optional game config from gym_gui (for compatibility)
    _env: gym.Env = field(init=False, repr=False)
    _action_meanings: list[str] = field(init=False, repr=False)
    _nrow: int = field(init=False, repr=False, default=5)
    _ncol: int = field(init=False, repr=False, default=5)

    def __post_init__(self) -> None:
        # Taxi-v3 doesn't have configurable options, but accept game_config for consistency
        # (game_config is ignored for Taxi)

        self._env = gym.make(
            "Taxi-v3",
            render_mode=None,
        )
        self._action_meanings = ["SOUTH", "NORTH", "EAST", "WEST", "PICKUP", "DROPOFF"]

        # Taxi is always 5 rows × 5 columns
        self._nrow = 5
        self._ncol = 5

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

    def decode_state(self, state: int) -> Tuple[int, int, int, int]:
        """Decode the state into (taxi_row, taxi_col, passenger_idx, destination_idx).
        
        Returns:
            taxi_row: Row position of taxi (0-4)
            taxi_col: Column position of taxi (0-4)
            passenger_idx: Passenger location (0-3 = R/G/Y/B depot, 4 = in taxi)
            destination_idx: Destination location (0-3 = R/G/Y/B depot)
        """
        unwrapped = self._env.unwrapped
        if hasattr(unwrapped, 'decode'):
            decoded = unwrapped.decode(state)  # type: ignore[attr-defined]
            return tuple(decoded)  # type: ignore[return-value]
        
        # Fallback manual decode (state encoding from Taxi-v3)
        # State = (taxi_row * 5 + taxi_col) * 5 * 4 + passenger_idx * 4 + destination_idx
        destination_idx = state % 4
        state = state // 4
        passenger_idx = state % 5
        state = state // 5
        taxi_col = state % 5
        taxi_row = state // 5
        
        return (taxi_row, taxi_col, passenger_idx, destination_idx)

    def action_meanings(self) -> list[str]:
        return list(self._action_meanings)

    def close(self) -> None:
        """Clean up the environment."""
        if self._env is not None:
            self._env.close()

    # Internal helpers ---------------------------------------------------
    def _obs_dict(self, state: int, info: Dict[str, Any] | None = None) -> Dict[str, Any]:
        taxi_row, taxi_col, passenger_idx, destination_idx = self.decode_state(state)
        
        # Depot locations: R(0,0), G(0,4), Y(4,0), B(4,3)
        depot_locations = [
            (0, 0),  # Red
            (0, 4),  # Green
            (4, 0),  # Yellow
            (4, 3),  # Blue
        ]
        
        passenger_location = depot_locations[passenger_idx] if passenger_idx < 4 else None
        destination_location = depot_locations[destination_idx]
        
        result = {
            "state": int(state),
            "taxi_position": {"row": int(taxi_row), "col": int(taxi_col)},
            "passenger_index": int(passenger_idx),  # 0-3 = depot, 4 = in taxi
            "destination_index": int(destination_idx),  # 0-3 = depot
            "passenger_location": passenger_location,  # None if in taxi
            "destination_location": destination_location,
            "passenger_in_taxi": passenger_idx == 4,
            "grid_size": 5,  # Taxi is always 5x5
        }

        if info:
            result.update(info)

        return result
