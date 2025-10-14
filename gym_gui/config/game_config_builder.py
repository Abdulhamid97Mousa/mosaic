"""Build game configurations from UI overrides."""

from __future__ import annotations

from typing import Any, Dict

from gym_gui.config.game_configs import (
    BipedalWalkerConfig,
    CarRacingConfig,
    CliffWalkingConfig,
    FrozenLakeConfig,
    LunarLanderConfig,
    TaxiConfig,
)
from gym_gui.core.enums import GameId


class GameConfigBuilder:
    """Builds typed game configuration objects from UI override dictionaries."""

    @staticmethod
    def build_config(
        game_id: GameId,
        overrides: Dict[str, Any],
    ) -> (
        FrozenLakeConfig
        | TaxiConfig
        | CliffWalkingConfig
        | LunarLanderConfig
        | CarRacingConfig
        | BipedalWalkerConfig
        | None
    ):
        """Build game configuration from control panel overrides."""
        if game_id == GameId.FROZEN_LAKE:
            is_slippery = bool(overrides.get("is_slippery", True))
            success_rate = float(overrides.get("success_rate", 1.0 / 3.0))
            reward_schedule = overrides.get("reward_schedule", (1.0, 0.0, 0.0))
            
            # Ensure reward_schedule is a tuple of 3 floats
            if not isinstance(reward_schedule, tuple) or len(reward_schedule) != 3:
                reward_schedule = (1.0, 0.0, 0.0)
            
            return FrozenLakeConfig(
                is_slippery=is_slippery,
                success_rate=success_rate,
                reward_schedule=reward_schedule,
            )
        
        elif game_id == GameId.FROZEN_LAKE_V2:
            is_slippery = bool(overrides.get("is_slippery", True))
            success_rate = float(overrides.get("success_rate", 1.0 / 3.0))
            reward_schedule = overrides.get("reward_schedule", (1.0, 0.0, 0.0))
            
            # Ensure reward_schedule is a tuple of 3 floats
            if not isinstance(reward_schedule, tuple) or len(reward_schedule) != 3:
                reward_schedule = (1.0, 0.0, 0.0)
            
            # Extract grid dimensions
            grid_height = overrides.get("grid_height", 8)
            grid_width = overrides.get("grid_width", 8)
            
            # Sanitize dimensions
            if isinstance(grid_height, (int, float)):
                grid_height = max(4, min(20, int(grid_height)))
            else:
                grid_height = 8
                
            if isinstance(grid_width, (int, float)):
                grid_width = max(4, min(20, int(grid_width)))
            else:
                grid_width = 8
            
            # Extract positions
            start_position = overrides.get("start_position", (0, 0))
            goal_position = overrides.get("goal_position", (grid_height - 1, grid_width - 1))
            
            # Extract hole count
            hole_count = overrides.get("hole_count", 19)
            if isinstance(hole_count, (int, float)):
                hole_count = int(hole_count)
            else:
                hole_count = 19
            
            # Extract random_holes flag
            random_holes = bool(overrides.get("random_holes", False))
            
            # Validate start_position
            if not isinstance(start_position, tuple) or len(start_position) != 2:
                start_position = (0, 0)
            
            # Validate goal_position
            if goal_position is None:
                goal_position = (grid_height - 1, grid_width - 1)
            elif not isinstance(goal_position, tuple) or len(goal_position) != 2:
                goal_position = (grid_height - 1, grid_width - 1)
            
            return FrozenLakeConfig(
                is_slippery=is_slippery,
                success_rate=success_rate,
                reward_schedule=reward_schedule,
                grid_height=grid_height,
                grid_width=grid_width,
                start_position=start_position,
                goal_position=goal_position,
                hole_count=hole_count,
                random_holes=random_holes,
            )
        
        elif game_id == GameId.TAXI:
            is_raining = bool(overrides.get("is_raining", False))
            fickle_passenger = bool(overrides.get("fickle_passenger", False))
            return TaxiConfig(is_raining=is_raining, fickle_passenger=fickle_passenger)
        
        elif game_id == GameId.CLIFF_WALKING:
            is_slippery = bool(overrides.get("is_slippery", False))
            return CliffWalkingConfig(is_slippery=is_slippery)
        
        elif game_id == GameId.LUNAR_LANDER:
            continuous = bool(overrides.get("continuous", False))
            gravity = overrides.get("gravity", -10.0)
            enable_wind = bool(overrides.get("enable_wind", False))
            wind_power = overrides.get("wind_power", 15.0)
            turbulence_power = overrides.get("turbulence_power", 1.5)
            steps_override = overrides.get("max_episode_steps")
            max_steps: int | None = None
            try:
                gravity_value = float(gravity)
            except (TypeError, ValueError):
                gravity_value = -10.0
            try:
                wind_power_value = float(wind_power)
            except (TypeError, ValueError):
                wind_power_value = 15.0
            try:
                turbulence_value = float(turbulence_power)
            except (TypeError, ValueError):
                turbulence_value = 1.5
            if isinstance(steps_override, (int, float)) and int(steps_override) > 0:
                max_steps = int(steps_override)
            return LunarLanderConfig(
                continuous=continuous,
                gravity=gravity_value,
                enable_wind=enable_wind,
                wind_power=wind_power_value,
                turbulence_power=turbulence_value,
                max_episode_steps=max_steps,
            )
        
        elif game_id == GameId.CAR_RACING:
            continuous = bool(overrides.get("continuous", False))
            domain_randomize = bool(overrides.get("domain_randomize", False))
            lap_percent = overrides.get("lap_complete_percent", 0.95)
            try:
                lap_value = float(lap_percent)
            except (TypeError, ValueError):
                lap_value = 0.95
            steps_override = overrides.get("max_episode_steps")
            seconds_override = overrides.get("max_episode_seconds")
            max_steps: int | None = None
            max_seconds: float | None = None
            if isinstance(steps_override, (int, float)) and int(steps_override) > 0:
                max_steps = int(steps_override)
            if isinstance(seconds_override, (int, float)) and float(seconds_override) > 0:
                max_seconds = float(seconds_override)
            return CarRacingConfig(
                continuous=continuous,
                domain_randomize=domain_randomize,
                lap_complete_percent=lap_value,
                max_episode_steps=max_steps,
                max_episode_seconds=max_seconds,
            )
        
        elif game_id == GameId.BIPEDAL_WALKER:
            hardcore = bool(overrides.get("hardcore", False))
            steps_override = overrides.get("max_episode_steps")
            seconds_override = overrides.get("max_episode_seconds")
            max_steps: int | None = None
            max_seconds: float | None = None
            if isinstance(steps_override, (int, float)) and int(steps_override) > 0:
                max_steps = int(steps_override)
            if isinstance(seconds_override, (int, float)) and float(seconds_override) > 0:
                max_seconds = float(seconds_override)
            return BipedalWalkerConfig(
                hardcore=hardcore,
                max_episode_steps=max_steps,
                max_episode_seconds=max_seconds,
            )
        
        return None


__all__ = ["GameConfigBuilder"]
