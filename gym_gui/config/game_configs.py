"""Game-specific configuration dataclasses following separation of concerns.

Each game has its own configuration class with game-specific parameters.
These configs are separate from global application settings.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, TypeAlias

from gym_gui.constants.constants_game import (
    CLIFF_WALKING_DEFAULTS,
    FROZEN_LAKE_DEFAULTS,
    FROZEN_LAKE_V2_DEFAULTS,
    BLACKJACK_DEFAULTS,
)
from gym_gui.core.enums import GameId


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class FrozenLakeConfig:
    """Configuration for FrozenLake environment."""
    
    is_slippery: bool = True
    """If True, agent moves in intended direction with probability specified by success_rate,
    else moves in perpendicular directions with equal probability.
    If False, agent always moves in intended direction."""
    
    success_rate: float = 1.0 / 3.0
    """Probability of moving in intended direction when is_slippery=True.
    For example, with success_rate=1/3:
    - P(intended direction) = 1/3
    - P(perpendicular direction 1) = 1/3
    - P(perpendicular direction 2) = 1/3"""
    
    reward_schedule: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Reward amounts for reaching tiles: (Goal, Hole, Frozen).
    Default (1, 0, 0) gives +1 for goal, 0 for hole/frozen."""
    
    grid_height: int = 4
    """Grid height (number of rows). Default is 4 for FrozenLake-v1, 8 for FrozenLake-v2."""
    
    grid_width: int = 4
    """Grid width (number of columns). Default is 4 for FrozenLake-v1, 8 for FrozenLake-v2."""
    
    start_position: tuple[int, int] | None = None
    """Starting position (row, col). If None, defaults to (0, 0)."""
    
    goal_position: tuple[int, int] | None = None
    """Goal position (row, col). If None, defaults to bottom-right corner."""
    
    hole_count: int | None = None
    """Number of holes. If None, uses Gymnasium default (4 for 4×4, 10 for 8×8)."""
    
    random_holes: bool = False
    """If True, holes are placed randomly. If False, uses fixed Gymnasium default map patterns.
    Only applies to FrozenLake-v2 with standard 4×4 or 8×8 grids."""
    
    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs.
        
        For FrozenLake-v1: Pass only is_slippery, success_rate, reward_schedule.
                          Do NOT pass map_name or grid dimensions.
        
        For FrozenLake-v2: Pass all parameters; custom map generation handled by adapter
                           when grid dimensions or positions deviate from defaults.
        """
        kwargs: Dict[str, Any] = {
            "is_slippery": self.is_slippery,
            "success_rate": self.success_rate,
            "reward_schedule": self.reward_schedule,
        }
        
        # Note: grid_height, grid_width, start_position, goal_position, and hole_count
        # should only be used by FrozenLakeV2Adapter._generate_map_descriptor().
        # DO NOT pass map_name to Gymnasium for v1 (causes initialization failure).
        # FrozenLakeV2Adapter handles custom map generation separately via gym_kwargs()
        # in its subclass override.
        
        return kwargs


@dataclass(frozen=True)
class TaxiConfig:
    """Configuration for Taxi-v3 environment.
    
    Note: Taxi-v3 in Gymnasium does not support is_raining or fickle_passenger
    parameters. These were present in older versions but removed in modern Gymnasium.
    The environment always uses deterministic movement.
    """
    
    is_raining: bool = False
    """[NOT SUPPORTED] If True, the cab would move in intended direction with 80% probability.
    This parameter is kept for UI compatibility but has no effect."""
    
    fickle_passenger: bool = False
    """[NOT SUPPORTED] If True, passenger would change destinations randomly.
    This parameter is kept for UI compatibility but has no effect."""
    
    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs.
        
        Note: Returns empty dict as Taxi-v3 doesn't accept custom parameters.
        """
        # Taxi-v3 doesn't support is_raining or fickle_passenger in current Gymnasium
        return {}


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


@dataclass(frozen=True)
class BlackjackConfig:
    """Configuration for Blackjack environment."""
    
    natural: bool = False
    """If True, give an additional reward for starting with a natural blackjack
    (ace and ten, sum is 21). Natural gives 1.5 reward instead of 1.0."""
    
    sab: bool = False
    """If True, follow the exact rules from Sutton and Barto's book.
    When sab=True, the natural parameter is ignored. If the player achieves
    a natural blackjack and the dealer does not, the player wins (+1 reward).
    If both get a natural, it's a draw (0 reward)."""
    
    def to_gym_kwargs(self) -> Dict[str, Any]:
        """Convert to Gymnasium environment kwargs."""
        return {"natural": self.natural, "sab": self.sab}


@dataclass(frozen=True)
class LunarLanderConfig:
    """Configuration for LunarLander environment."""

    continuous: bool = False
    gravity: float = -10.0
    enable_wind: bool = False
    wind_power: float = 15.0
    turbulence_power: float = 1.5
    max_episode_steps: int | None = None

    def to_gym_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "continuous": self.continuous,
            "gravity": _clamp(self.gravity, -12.0, 0.0),
        }
        if self.enable_wind:
            kwargs.update(
                enable_wind=True,
                wind_power=max(0.0, self.wind_power),
                turbulence_power=max(0.0, self.turbulence_power),
            )
        else:
            kwargs["enable_wind"] = False
        return kwargs

    def sanitized_step_limit(self) -> int | None:
        steps = self.max_episode_steps
        if steps is None:
            return None
        if isinstance(steps, bool):
            return None
        try:
            value = int(steps)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None


@dataclass(frozen=True)
class CarRacingConfig:
    """Configuration for CarRacing environment."""

    continuous: bool = False
    domain_randomize: bool = False
    lap_complete_percent: float = 0.95
    max_episode_steps: int | None = None
    max_episode_seconds: float | None = None

    @classmethod
    def from_env(cls) -> "CarRacingConfig":
        steps_raw = os.getenv("CAR_RACING_MAX_EPISODE_STEPS")
        seconds_raw = os.getenv("CAR_RACING_MAX_EPISODE_SECONDS")
        continuous_raw = os.getenv("CAR_RACING_CONTINUOUS")
        domain_raw = os.getenv("CAR_RACING_DOMAIN_RANDOMIZE")
        lap_raw = os.getenv("CAR_RACING_LAP_COMPLETE_PERCENT")

        steps: int | None
        seconds: float | None
        continuous = False
        domain_randomize = False
        lap_percent = 0.95

        try:
            steps = int(steps_raw) if steps_raw not in (None, "", "0") else None
        except (TypeError, ValueError):
            steps = None

        try:
            seconds = float(seconds_raw) if seconds_raw not in (None, "", "0") else None
        except (TypeError, ValueError):
            seconds = None

        if continuous_raw is not None:
            continuous = continuous_raw.strip().lower() in {"1", "true", "yes", "on"}
        if domain_raw is not None:
            domain_randomize = domain_raw.strip().lower() in {"1", "true", "yes", "on"}
        if lap_raw:
            try:
                lap_percent = float(lap_raw)
            except (TypeError, ValueError):
                lap_percent = 0.95

        return cls(
            continuous=continuous,
            domain_randomize=domain_randomize,
            lap_complete_percent=lap_percent,
            max_episode_steps=steps,
            max_episode_seconds=seconds,
        )

    def to_gym_kwargs(self) -> Dict[str, Any]:
        percent = _clamp(self.lap_complete_percent, 0.5, 1.0)
        kwargs: Dict[str, Any] = {
            "continuous": self.continuous,
            "domain_randomize": self.domain_randomize,
            "lap_complete_percent": percent,
        }
        return kwargs

    def sanitized_time_limits(self) -> tuple[int | None, float | None]:
        steps = int(self.max_episode_steps) if self.max_episode_steps and self.max_episode_steps > 0 else None
        seconds = (
            float(self.max_episode_seconds)
            if self.max_episode_seconds and self.max_episode_seconds > 0
            else None
        )
        return steps, seconds


@dataclass(frozen=True)
class BipedalWalkerConfig:
    """Configuration for BipedalWalker environment."""

    hardcore: bool = False
    max_episode_steps: int | None = None
    max_episode_seconds: float | None = None

    @classmethod
    def from_env(cls) -> "BipedalWalkerConfig":
        hardcore_raw = os.getenv("BIPEDAL_HARDCORE")
        steps_raw = os.getenv("BIPEDAL_MAX_EPISODE_STEPS")
        seconds_raw = os.getenv("BIPEDAL_MAX_EPISODE_SECONDS")

        hardcore = False
        steps: int | None
        seconds: float | None

        if hardcore_raw is not None:
            hardcore = hardcore_raw.strip().lower() in {"1", "true", "yes", "on"}

        try:
            steps = int(steps_raw) if steps_raw not in (None, "", "0") else None
        except (TypeError, ValueError):
            steps = None

        try:
            seconds = float(seconds_raw) if seconds_raw not in (None, "", "0") else None
        except (TypeError, ValueError):
            seconds = None

        return cls(
            hardcore=hardcore,
            max_episode_steps=steps,
            max_episode_seconds=seconds,
        )

    def to_gym_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"hardcore": self.hardcore}
        return kwargs

    def sanitized_time_limits(self) -> tuple[int | None, float | None]:
        steps = int(self.max_episode_steps) if self.max_episode_steps and self.max_episode_steps > 0 else None
        seconds = (
            float(self.max_episode_seconds)
            if self.max_episode_seconds and self.max_episode_seconds > 0
            else None
        )
        return steps, seconds


@dataclass(frozen=True)
class MiniGridConfig:
    """Configuration payload for MiniGrid environments."""

    env_id: str = GameId.MINIGRID_EMPTY_5x5.value
    """Gymnasium environment identifier (e.g., ``MiniGrid-Empty-5x5-v0``)."""

    partial_observation: bool = True
    """Apply :class:`RGBImgPartialObsWrapper` to expose agent-centric views."""

    image_observation: bool = True
    """Convert observations to RGB image arrays using :class:`ImgObsWrapper`."""

    reward_multiplier: float = 10.0
    """Scalar applied to environment rewards (aligned with xuance baseline)."""

    agent_view_size: int | None = None
    """Optional override for agent view size (MiniGrid default is 7)."""

    max_episode_steps: int | None = None
    """Override max episode steps; ``None`` preserves environment default."""

    seed: int | None = None
    """Default seed forwarded to :meth:`gymnasium.Env.reset`."""

    render_mode: str = "rgb_array"
    """Render mode requested during environment creation."""

    append_direction: bool = True
    """When True, append the agent's direction to the flattened observation
    vector (image.flatten() + [direction]). Matches xuance baseline behaviour."""

    def to_gym_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"render_mode": self.render_mode}
        if self.agent_view_size is not None:
            kwargs["agent_view_size"] = int(self.agent_view_size)
        if self.max_episode_steps is not None and self.max_episode_steps > 0:
            kwargs["max_episode_steps"] = int(self.max_episode_steps)
        return kwargs


GameConfig: TypeAlias = (
    FrozenLakeConfig
    | TaxiConfig
    | CliffWalkingConfig
    | LunarLanderConfig
    | CarRacingConfig
    | BipedalWalkerConfig
    | MiniGridConfig
)


# Default configurations for each game
DEFAULT_FROZEN_LAKE_CONFIG = FrozenLakeConfig(
    is_slippery=FROZEN_LAKE_DEFAULTS.slippery,
    success_rate=1.0 / 3.0,
    reward_schedule=(1.0, 0.0, 0.0),
    grid_height=FROZEN_LAKE_DEFAULTS.grid_height,
    grid_width=FROZEN_LAKE_DEFAULTS.grid_width,
    start_position=FROZEN_LAKE_DEFAULTS.start,
    goal_position=FROZEN_LAKE_DEFAULTS.goal,
    hole_count=FROZEN_LAKE_DEFAULTS.hole_count,
    random_holes=FROZEN_LAKE_DEFAULTS.random_holes,
)
DEFAULT_FROZEN_LAKE_V2_CONFIG = FrozenLakeConfig(
    is_slippery=FROZEN_LAKE_V2_DEFAULTS.slippery,
    success_rate=1.0 / 3.0,
    reward_schedule=(1.0, 0.0, 0.0),
    grid_height=FROZEN_LAKE_V2_DEFAULTS.grid_height,
    grid_width=FROZEN_LAKE_V2_DEFAULTS.grid_width,
    start_position=FROZEN_LAKE_V2_DEFAULTS.start,
    goal_position=FROZEN_LAKE_V2_DEFAULTS.goal,
    hole_count=FROZEN_LAKE_V2_DEFAULTS.hole_count,
    random_holes=FROZEN_LAKE_V2_DEFAULTS.random_holes,
)
DEFAULT_TAXI_CONFIG = TaxiConfig(is_raining=False, fickle_passenger=False)
DEFAULT_CLIFF_WALKING_CONFIG = CliffWalkingConfig(
    is_slippery=CLIFF_WALKING_DEFAULTS.slippery
)
DEFAULT_BLACKJACK_CONFIG = BlackjackConfig()
DEFAULT_LUNAR_LANDER_CONFIG = LunarLanderConfig()
DEFAULT_CAR_RACING_CONFIG = CarRacingConfig.from_env()
DEFAULT_BIPEDAL_WALKER_CONFIG = BipedalWalkerConfig.from_env()
DEFAULT_MINIGRID_EMPTY_5x5_CONFIG = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_5x5.value)
DEFAULT_MINIGRID_EMPTY_RANDOM_5x5_CONFIG = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_RANDOM_5x5.value)
DEFAULT_MINIGRID_EMPTY_6x6_CONFIG = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_6x6.value)
DEFAULT_MINIGRID_EMPTY_RANDOM_6x6_CONFIG = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_RANDOM_6x6.value)
DEFAULT_MINIGRID_EMPTY_8x8_CONFIG = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_8x8.value)
DEFAULT_MINIGRID_EMPTY_16x16_CONFIG = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_16x16.value)
DEFAULT_MINIGRID_DOORKEY_5x5_CONFIG = MiniGridConfig(
    env_id=GameId.MINIGRID_DOORKEY_5x5.value,
    agent_view_size=5,
)
DEFAULT_MINIGRID_DOORKEY_6x6_CONFIG = MiniGridConfig(
    env_id=GameId.MINIGRID_DOORKEY_6x6.value,
    agent_view_size=7,
)
DEFAULT_MINIGRID_DOORKEY_8x8_CONFIG = MiniGridConfig(
    env_id=GameId.MINIGRID_DOORKEY_8x8.value,
    agent_view_size=7,
)
DEFAULT_MINIGRID_DOORKEY_16x16_CONFIG = MiniGridConfig(
    env_id=GameId.MINIGRID_DOORKEY_16x16.value,
    agent_view_size=9,
)
DEFAULT_MINIGRID_LAVAGAP_S7_CONFIG = MiniGridConfig(
    env_id=GameId.MINIGRID_LAVAGAP_S7.value,
    agent_view_size=7,
    partial_observation=True,
)


# ---------------------------------------------------------------------------
# ALE (Atari) configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ALEConfig:
    """Configuration payload for ALE Atari environments.

    Mirrors common ALE kwargs used by Gymnasium:
    - obs_type: "rgb" | "ram" | "grayscale"
    - frameskip: int or (min, max) tuple
    - repeat_action_probability: float (aka stickiness / RAP)
    - difficulty, mode: integers selecting game flavour
    - full_action_space: request full 18-action set when True
    - render_mode: rendering mode (default "rgb_array")
    - seed: default seed used by adapter reset when caller omits one
    - env_id: Gymnasium environment id
    """

    env_id: str = GameId.ADVENTURE_V4.value
    obs_type: str = "rgb"
    frameskip: int | tuple[int, int] | None = None
    repeat_action_probability: float | None = None
    difficulty: int | None = None
    mode: int | None = None
    full_action_space: bool = False
    render_mode: str = "rgb_array"
    seed: int | None = None

    def to_gym_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "render_mode": self.render_mode,
            "obs_type": self.obs_type,
        }
        if self.frameskip is not None:
            kwargs["frameskip"] = self.frameskip
        if self.repeat_action_probability is not None:
            kwargs["repeat_action_probability"] = float(self.repeat_action_probability)
        if self.difficulty is not None:
            kwargs["difficulty"] = int(self.difficulty)
        if self.mode is not None:
            kwargs["mode"] = int(self.mode)
        if self.full_action_space:
            kwargs["full_action_space"] = True
        return kwargs


__all__ = [
    "FrozenLakeConfig",
    "TaxiConfig",
    "CliffWalkingConfig",
    "BlackjackConfig",
    "LunarLanderConfig",
    "CarRacingConfig",
    "BipedalWalkerConfig",
    "MiniGridConfig",
    "ALEConfig",
    "GameConfig",
    "DEFAULT_FROZEN_LAKE_CONFIG",
    "DEFAULT_FROZEN_LAKE_V2_CONFIG",
    "DEFAULT_TAXI_CONFIG",
    "DEFAULT_CLIFF_WALKING_CONFIG",
    "DEFAULT_BLACKJACK_CONFIG",
    "DEFAULT_LUNAR_LANDER_CONFIG",
    "DEFAULT_CAR_RACING_CONFIG",
    "DEFAULT_BIPEDAL_WALKER_CONFIG",
    "DEFAULT_MINIGRID_EMPTY_5x5_CONFIG",
    "DEFAULT_MINIGRID_EMPTY_RANDOM_5x5_CONFIG",
    "DEFAULT_MINIGRID_EMPTY_6x6_CONFIG",
    "DEFAULT_MINIGRID_EMPTY_RANDOM_6x6_CONFIG",
    "DEFAULT_MINIGRID_EMPTY_8x8_CONFIG",
    "DEFAULT_MINIGRID_EMPTY_16x16_CONFIG",
    "DEFAULT_MINIGRID_DOORKEY_5x5_CONFIG",
    "DEFAULT_MINIGRID_DOORKEY_6x6_CONFIG",
    "DEFAULT_MINIGRID_DOORKEY_8x8_CONFIG",
    "DEFAULT_MINIGRID_DOORKEY_16x16_CONFIG",
    "DEFAULT_MINIGRID_LAVAGAP_S7_CONFIG",
]
