"""Build game configurations from UI overrides."""

from __future__ import annotations

from typing import Any, Dict

from gym_gui.config.game_configs import (
    BipedalWalkerConfig,
    CarRacingConfig,
    CliffWalkingConfig,
    FrozenLakeConfig,
    LunarLanderConfig,
    MeltingPotConfig,
    MiniGridConfig,
    MultiGridConfig,
    SMACConfig,
    TaxiConfig,
)
from gym_gui.core.enums import GameId, EnvironmentFamily, ENVIRONMENT_FAMILY_BY_GAME

try:
    from gym_gui.core.adapters.vizdoom import ViZDoomConfig
except ImportError:
    ViZDoomConfig = None  # type: ignore

# ViZDoom game IDs for config builder
_VIZDOOM_GAME_IDS = (
    GameId.VIZDOOM_BASIC,
    GameId.VIZDOOM_DEADLY_CORRIDOR,
    GameId.VIZDOOM_DEFEND_THE_CENTER,
    GameId.VIZDOOM_DEFEND_THE_LINE,
    GameId.VIZDOOM_HEALTH_GATHERING,
    GameId.VIZDOOM_HEALTH_GATHERING_SUPREME,
    GameId.VIZDOOM_MY_WAY_HOME,
    GameId.VIZDOOM_PREDICT_POSITION,
    GameId.VIZDOOM_TAKE_COVER,
    GameId.VIZDOOM_DEATHMATCH,
)


class GameConfigBuilder:
    """Builds typed game configuration objects from UI override dictionaries."""

    @staticmethod
    def build_config(
        game_id: GameId,
        overrides: Dict[str, Any],
    ) -> Any:
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
                hole_count = 10
            
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

        elif game_id in (
            GameId.MINIGRID_EMPTY_5x5,
            GameId.MINIGRID_EMPTY_RANDOM_5x5,
            GameId.MINIGRID_EMPTY_6x6,
            GameId.MINIGRID_EMPTY_RANDOM_6x6,
            GameId.MINIGRID_EMPTY_8x8,
            GameId.MINIGRID_EMPTY_16x16,
            GameId.MINIGRID_DOORKEY_5x5,
            GameId.MINIGRID_DOORKEY_6x6,
            GameId.MINIGRID_DOORKEY_8x8,
            GameId.MINIGRID_DOORKEY_16x16,
            GameId.MINIGRID_LAVAGAP_S7,
            GameId.MINIGRID_REDBLUE_DOORS_6x6,
            GameId.MINIGRID_REDBLUE_DOORS_8x8,
        ):
            partial = bool(overrides.get("partial_observation", True))
            image_obs = bool(overrides.get("image_observation", True))
            reward_multiplier = overrides.get("reward_multiplier", 10.0)
            try:
                reward_multiplier_value = float(reward_multiplier)
            except (TypeError, ValueError):
                reward_multiplier_value = 10.0
            agent_view_size = overrides.get("agent_view_size")
            agent_view_value: int | None = None
            if isinstance(agent_view_size, (int, float)) and int(agent_view_size) > 0:
                agent_view_value = int(agent_view_size)
            max_steps_override = overrides.get("max_episode_steps")
            max_steps_value: int | None = None
            if isinstance(max_steps_override, (int, float)) and int(max_steps_override) > 0:
                max_steps_value = int(max_steps_override)
            seed_override = overrides.get("seed")
            seed_value: int | None = None
            if isinstance(seed_override, (int, float)):
                seed_value = int(seed_override)
            return MiniGridConfig(
                env_id=game_id.value,
                partial_observation=partial,
                image_observation=image_obs,
                reward_multiplier=reward_multiplier_value,
                agent_view_size=agent_view_value,
                max_episode_steps=max_steps_value,
                seed=seed_value,
            )

        elif game_id in _VIZDOOM_GAME_IDS and ViZDoomConfig is not None:
            # Build ViZDoomConfig from overrides
            screen_resolution = str(overrides.get("screen_resolution", "RES_640X480"))
            screen_format = str(overrides.get("screen_format", "RGB24"))
            render_hud = bool(overrides.get("render_hud", True))
            render_weapon = bool(overrides.get("render_weapon", True))
            render_crosshair = bool(overrides.get("render_crosshair", False))
            render_particles = bool(overrides.get("render_particles", True))
            render_decals = bool(overrides.get("render_decals", True))
            sound_enabled = bool(overrides.get("sound_enabled", False))
            depth_buffer = bool(overrides.get("depth_buffer", False))
            labels_buffer = bool(overrides.get("labels_buffer", False))
            automap_buffer = bool(overrides.get("automap_buffer", False))

            # Episode/reward settings
            episode_timeout = int(overrides.get("episode_timeout", 2100))
            living_reward = float(overrides.get("living_reward", 0.0))
            death_penalty = float(overrides.get("death_penalty", 100.0))

            return ViZDoomConfig(
                screen_resolution=screen_resolution,
                screen_format=screen_format,
                render_hud=render_hud,
                render_weapon=render_weapon,
                render_crosshair=render_crosshair,
                render_particles=render_particles,
                render_decals=render_decals,
                sound_enabled=sound_enabled,
                depth_buffer=depth_buffer,
                labels_buffer=labels_buffer,
                automap_buffer=automap_buffer,
                episode_timeout=episode_timeout,
                living_reward=living_reward,
                death_penalty=death_penalty,
            )

        # MultiGrid environments (MOSAIC and INI multigrid)
        family = ENVIRONMENT_FAMILY_BY_GAME.get(game_id)
        if family in (EnvironmentFamily.MOSAIC_MULTIGRID, EnvironmentFamily.INI_MULTIGRID):
            # Use the game_id value directly as env_id (e.g., "MultiGrid-BlockedUnlockPickup-v0")
            env_id = game_id.value
            num_agents = overrides.get("num_agents")
            seed = overrides.get("seed")
            highlight = bool(overrides.get("highlight", True))

            # Validate num_agents if provided
            if num_agents is not None:
                try:
                    num_agents = int(num_agents)
                except (TypeError, ValueError):
                    num_agents = None

            # Default to 2 agents for INI cooperative multi-agent environments
            # (BlockedUnlockPickup, Empty, LockedHallway, etc. are designed for multi-agent)
            # Legacy environments (Soccer=4, Collect=3) have fixed agent counts
            if num_agents is None:
                if "Soccer" in env_id:
                    num_agents = 4  # 2v2 teams
                elif "Collect" in env_id:
                    num_agents = 3  # 3 collectors
                else:
                    # INI multigrid environments default to 2 for cooperative play
                    num_agents = 2

            # Validate seed if provided
            if seed is not None:
                try:
                    seed = int(seed)
                except (TypeError, ValueError):
                    seed = None

            return MultiGridConfig(
                env_id=env_id,
                num_agents=num_agents,
                seed=seed,
                highlight=highlight,
            )

        # SMAC v1 environments (hand-designed cooperative micromanagement maps)
        if family == EnvironmentFamily.SMAC:
            # Extract map name from GameId value: "SMAC-3m-v0" -> "3m"
            raw = game_id.value
            map_name = raw.replace("SMAC-", "").replace("-v0", "")
            difficulty = str(overrides.get("difficulty", "7"))
            reward_sparse = bool(overrides.get("reward_sparse", False))
            reward_only_positive = bool(overrides.get("reward_only_positive", True))
            reward_scale = bool(overrides.get("reward_scale", True))
            reward_scale_rate = float(overrides.get("reward_scale_rate", 20.0))
            obs_own_health = bool(overrides.get("obs_own_health", True))
            obs_pathing_grid = bool(overrides.get("obs_pathing_grid", False))
            obs_terrain_height = bool(overrides.get("obs_terrain_height", False))
            seed = overrides.get("seed")
            if seed is not None:
                try:
                    seed = int(seed)
                except (TypeError, ValueError):
                    seed = None
            episode_limit = overrides.get("episode_limit")
            if episode_limit is not None:
                try:
                    episode_limit = int(episode_limit)
                    if episode_limit <= 0:
                        episode_limit = None
                except (TypeError, ValueError):
                    episode_limit = None
            sc2_path = overrides.get("sc2_path")
            if sc2_path is not None and not str(sc2_path).strip():
                sc2_path = None
            renderer = str(overrides.get("renderer", "3d"))
            return SMACConfig(
                map_name=map_name,
                difficulty=difficulty,
                reward_sparse=reward_sparse,
                reward_only_positive=reward_only_positive,
                reward_scale=reward_scale,
                reward_scale_rate=reward_scale_rate,
                obs_own_health=obs_own_health,
                obs_pathing_grid=obs_pathing_grid,
                obs_terrain_height=obs_terrain_height,
                seed=seed,
                episode_limit=episode_limit,
                sc2_path=sc2_path,
                renderer=renderer,
            )

        # SMACv2 environments (procedural unit generation)
        if family == EnvironmentFamily.SMACV2:
            # Extract map name from GameId value: "SMACv2-10gen_terran-v0" -> "10gen_terran"
            raw = game_id.value
            map_name = raw.replace("SMACv2-", "").replace("-v0", "")
            difficulty = str(overrides.get("difficulty", "7"))
            reward_sparse = bool(overrides.get("reward_sparse", False))
            reward_only_positive = bool(overrides.get("reward_only_positive", True))
            reward_scale = bool(overrides.get("reward_scale", True))
            reward_scale_rate = float(overrides.get("reward_scale_rate", 20.0))
            obs_own_health = bool(overrides.get("obs_own_health", True))
            obs_pathing_grid = bool(overrides.get("obs_pathing_grid", False))
            obs_terrain_height = bool(overrides.get("obs_terrain_height", False))
            seed = overrides.get("seed")
            if seed is not None:
                try:
                    seed = int(seed)
                except (TypeError, ValueError):
                    seed = None
            episode_limit = overrides.get("episode_limit")
            if episode_limit is not None:
                try:
                    episode_limit = int(episode_limit)
                    if episode_limit <= 0:
                        episode_limit = None
                except (TypeError, ValueError):
                    episode_limit = None
            sc2_path = overrides.get("sc2_path")
            if sc2_path is not None and not str(sc2_path).strip():
                sc2_path = None
            renderer = str(overrides.get("renderer", "3d"))
            return SMACConfig(
                map_name=map_name,
                difficulty=difficulty,
                reward_sparse=reward_sparse,
                reward_only_positive=reward_only_positive,
                reward_scale=reward_scale,
                reward_scale_rate=reward_scale_rate,
                obs_own_health=obs_own_health,
                obs_pathing_grid=obs_pathing_grid,
                obs_terrain_height=obs_terrain_height,
                seed=seed,
                episode_limit=episode_limit,
                sc2_path=sc2_path,
                renderer=renderer,
            )

        # MeltingPot environments
        if family == EnvironmentFamily.MELTINGPOT:
            # Extract substrate name from game_id value (e.g., "meltingpot/clean_up" -> "clean_up")
            substrate_name = game_id.value.replace("meltingpot/", "")
            render_scale = overrides.get("render_scale", 2)
            try:
                render_scale_value = int(render_scale)
            except (TypeError, ValueError):
                render_scale_value = 2
            return MeltingPotConfig(
                substrate_name=substrate_name,
                render_scale=render_scale_value,
            )

        # RWARE (Robotic Warehouse) environments
        if family == EnvironmentFamily.RWARE:
            from gym_gui.config.game_configs import RWAREConfig

            observation_type = overrides.get("observation_type", "flattened")
            sensor_range = int(overrides.get("sensor_range", 1))
            reward_type = overrides.get("reward_type", "individual")
            msg_bits = int(overrides.get("msg_bits", 0))
            max_steps = int(overrides.get("max_steps", 500))
            seed_val = overrides.get("seed")
            seed = int(seed_val) if seed_val is not None and int(seed_val) >= 0 else None

            return RWAREConfig(
                observation_type=observation_type,
                sensor_range=sensor_range,
                reward_type=reward_type,
                msg_bits=msg_bits,
                max_steps=max_steps,
                seed=seed,
            )

        return None


__all__ = ["GameConfigBuilder"]
