"""Shared defaults for toy-text environments (FrozenLake, CliffWalking, Taxi).

These mirror Gymnasium maps and give the UI/adapters a single source of truth.
Values can be overridden by UI metadata or env vars if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from gym_gui.core.enums import GameId


@dataclass(frozen=True, slots=True)
class ToyTextDefaults:
    grid_height: int
    grid_width: int
    start: tuple[int, int]
    goal: tuple[int, int]
    slippery: bool = True
    hole_count: int | None = None
    random_holes: bool = False
    random_map_prob: float | None = None
    official_map: tuple[str, ...] | None = None


FROZEN_LAKE_DEFAULTS = ToyTextDefaults(
    grid_height=4,
    grid_width=4,
    start=(0, 0),
    goal=(3, 3),
    slippery=True,
    hole_count=4,
    random_holes=False,
    official_map=(
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ),
)

FROZEN_LAKE_V2_DEFAULTS = ToyTextDefaults(
    grid_height=8,
    grid_width=8,
    start=(0, 0),
    goal=(7, 7),
    slippery=False,
    hole_count=10,
    random_holes=False,
    official_map=(
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ),
)

CLIFF_WALKING_DEFAULTS = ToyTextDefaults(
    grid_height=4,
    grid_width=12,
    start=(3, 0),
    goal=(3, 11),
    slippery=False,
)

TAXI_DEFAULTS = ToyTextDefaults(
    grid_height=5,
    grid_width=5,
    start=(0, 0),  # taxi start is random but we keep a placeholder
    goal=(0, 0),
    slippery=False,
)


TOY_TEXT_DEFAULTS: Mapping[GameId, ToyTextDefaults] = {
    GameId.FROZEN_LAKE: FROZEN_LAKE_DEFAULTS,
    GameId.FROZEN_LAKE_V2: FROZEN_LAKE_V2_DEFAULTS,
    GameId.CLIFF_WALKING: CLIFF_WALKING_DEFAULTS,
    GameId.TAXI: TAXI_DEFAULTS,
}

__all__ = [
    "ToyTextDefaults",
    "TOY_TEXT_DEFAULTS",
    "FROZEN_LAKE_DEFAULTS",
    "FROZEN_LAKE_V2_DEFAULTS",
    "CLIFF_WALKING_DEFAULTS",
    "TAXI_DEFAULTS",
]
