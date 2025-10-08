"""Game-specific configuration dataclasses following separation of concerns.

Each game has its own configuration class with game-specific parameters.
These configs are separate from global application settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class FrozenLakeConfig:
    """Configuration for FrozenLake environment."""
    
    is_slippery: bool = True
    """If True, agent moves in intended direction with 33.33% probability,
    and in perpendicular directions with 33.33% probability each.
    If False, agent always moves in intended direction."""
    
    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs."""
        return {"is_slippery": self.is_slippery}


@dataclass(frozen=True)
class TaxiConfig:
    """Configuration for Taxi-v3 environment."""
    
    is_raining: bool = False
    """If True, the cab will move in intended direction with 80% probability,
    else will move left or right of target direction with 10% probability each.
    If False, cab always moves in intended direction."""
    
    fickle_passenger: bool = False
    """If True, passenger has 30% chance of changing destinations when cab
    has moved one square away from passenger's source location.
    Fickleness only triggers on first pickup and successful movement."""
    
    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs."""
        kwargs = {}
        if self.is_raining:
            kwargs["is_raining"] = True
        if self.fickle_passenger:
            kwargs["fickle_passenger"] = True
        return kwargs


@dataclass(frozen=True)
class CliffWalkingConfig:
    """Configuration for CliffWalking environment."""
    
    is_slippery: bool = False
    """If True, the cliff can be slippery so the player may move perpendicular
    to the intended direction sometimes. If False (default), player always moves
    in intended direction."""
    
    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs."""
        return {"is_slippery": self.is_slippery}


# Default configurations for each game
DEFAULT_FROZEN_LAKE_CONFIG = FrozenLakeConfig(is_slippery=True)
DEFAULT_TAXI_CONFIG = TaxiConfig(is_raining=False, fickle_passenger=False)
DEFAULT_CLIFF_WALKING_CONFIG = CliffWalkingConfig(is_slippery=False)


__all__ = [
    "FrozenLakeConfig",
    "TaxiConfig",
    "CliffWalkingConfig",
    "DEFAULT_FROZEN_LAKE_CONFIG",
    "DEFAULT_TAXI_CONFIG",
    "DEFAULT_CLIFF_WALKING_CONFIG",
]
