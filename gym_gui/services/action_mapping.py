# gym_gui/services/action_mapping.py

"""Mappings that transform discrete human inputs into continuous action vectors."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence

import numpy as np

from gym_gui.core.enums import GameId

MappingFunc = Callable[[int], np.ndarray]


class ContinuousActionMapper:
    """Registry that converts discrete UI actions into continuous vectors."""

    def __init__(self) -> None:
        self._registry: Dict[GameId, MappingFunc] = {}

    def register_mapping(self, game_id: GameId, mapping: MappingFunc) -> None:
        self._registry[game_id] = mapping

    def register_table(
        self,
        game_id: GameId,
        table: Mapping[int, Sequence[float]],
        *,
        default: Sequence[float] | None = None,
        dtype: np.dtype | None = None,
    ) -> None:
        """Register a discrete->continuous lookup table for ``game_id``."""

        if not table and default is None:
            raise ValueError("table or default must provide at least one vector")

        resolved_dtype = dtype or np.float32
        arrays: Dict[int, np.ndarray] = {
            key: np.asarray(value, dtype=resolved_dtype) for key, value in table.items()
        }
        if not arrays and default is None:
            raise ValueError("Cannot infer vector shape without table entries or default")

        if default is None:
            sample = next(iter(arrays.values()))
            default_array = np.zeros_like(sample)
        else:
            default_array = np.asarray(default, dtype=resolved_dtype)

        def mapping(index: int) -> np.ndarray:
            vector = arrays.get(index)
            if vector is None:
                return default_array.copy()
            return vector.copy()

        self.register_mapping(game_id, mapping)

    def map(self, game_id: GameId, discrete_action: int) -> np.ndarray | None:
        mapping = self._registry.get(game_id)
        if mapping is None:
            return None
        return mapping(int(discrete_action))

    def has_mapping(self, game_id: GameId) -> bool:
        return game_id in self._registry


LUNAR_LANDER_CONTINUOUS_PRESETS: Mapping[int, Sequence[float]] = {
    0: (0.0, 0.0),
    1: (0.0, -1.0),
    2: (1.0, 0.0),
    3: (0.0, 1.0),
}

CAR_RACING_CONTINUOUS_PRESETS: Mapping[int, Sequence[float]] = {
    0: (0.0, 0.0, 0.0),
    1: (1.0, 0.3, 0.0),
    2: (-1.0, 0.3, 0.0),
    3: (0.0, 1.0, 0.0),
    4: (0.0, 0.0, 0.8),
}

BIPEDAL_WALKER_CONTINUOUS_PRESETS: Mapping[int, Sequence[float]] = {
    0: (0.0, 0.0, 0.0, 0.0),
    1: (0.8, 0.6, -0.8, -0.6),
    2: (-0.8, -0.6, 0.8, 0.6),
    3: (0.4, -1.0, 0.4, -1.0),
    4: (-0.4, 1.0, -0.4, 1.0),
}


def create_default_action_mapper() -> ContinuousActionMapper:
    mapper = ContinuousActionMapper()
    mapper.register_table(GameId.LUNAR_LANDER, LUNAR_LANDER_CONTINUOUS_PRESETS, default=(0.0, 0.0))
    mapper.register_table(GameId.CAR_RACING, CAR_RACING_CONTINUOUS_PRESETS, default=(0.0, 0.0, 0.0))
    mapper.register_table(
        GameId.BIPEDAL_WALKER,
        BIPEDAL_WALKER_CONTINUOUS_PRESETS,
        default=(0.0, 0.0, 0.0, 0.0),
    )
    return mapper


__all__ = [
    "ContinuousActionMapper",
    "create_default_action_mapper",
    "LUNAR_LANDER_CONTINUOUS_PRESETS",
    "CAR_RACING_CONTINUOUS_PRESETS",
    "BIPEDAL_WALKER_CONTINUOUS_PRESETS",
]
