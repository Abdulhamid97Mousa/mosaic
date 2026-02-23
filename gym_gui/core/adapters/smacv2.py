"""SMACv2 (StarCraft Multi-Agent Challenge v2) adapter for the MOSAIC GUI.

SMACv2 extends SMAC v1 with procedural unit generation: team compositions
vary per episode, forcing agents to generalise rather than memorise fixed
strategies.  Uses ``StarCraftCapabilityEnvWrapper`` instead of raw
``StarCraft2Env``.

The adapter re-queries ``get_env_info()`` after each ``reset()`` because
agent count and observation/action shapes can change between episodes.

Supports three renderer modes:
- ``"3d"``: Full SC2 engine 3D rendering via EGL (GPU-accelerated).
- ``"heatmap"``: Custom 2x2 feature-layer panel (pure numpy).
- ``"classic"``: SMAC's built-in PyGame 2D renderer (colored circles).

Repository: https://github.com/oxwhirl/smacv2
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    AgentSnapshot,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import (
    ControlMode,
    GameId,
    RenderMode,
    SteppingParadigm,
)
from gym_gui.logging_config.log_constants import (
    LOG_SMAC_BATTLE_RESULT,
    LOG_SMAC_ENV_CLOSED,
    LOG_SMAC_ENV_CREATED,
    LOG_SMAC_ENV_RESET,
    LOG_SMAC_RENDER_ERROR,
    LOG_SMAC_SC2_PATH_MISSING,
    LOG_SMAC_STEP_SUMMARY,
)

_LOGGER = logging.getLogger(__name__)

# SMACv2 procedural map metadata: map_name -> (default_agents, race, description)
SMACV2_MAP_INFO: Dict[str, tuple[int, str, str]] = {
    "10gen_terran": (10, "Terran", "Random Terran composition (Marines, Marauders, Medivacs)"),
    "10gen_protoss": (10, "Protoss", "Random Protoss composition (Stalkers, Zealots, Colossi)"),
    "10gen_zerg": (10, "Zerg", "Random Zerg composition (Zerglings, Hydralisks, Banelings)"),
}

# Default capability_config per race for StarCraftCapabilityEnvWrapper.
# These define the procedural unit generation distributions.
SMACV2_CAPABILITY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "10gen_terran": {
        "n_units": 10,
        "n_enemies": 10,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
            "exception_unit_types": ["medivac"],
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": 32,
            "map_y": 32,
        },
    },
    "10gen_protoss": {
        "n_units": 10,
        "n_enemies": 10,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["stalker", "zealot", "colossus"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
            "exception_unit_types": ["colossus"],
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": 32,
            "map_y": 32,
        },
    },
    "10gen_zerg": {
        "n_units": 10,
        "n_enemies": 10,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "hydralisk", "baneling"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
            "exception_unit_types": ["baneling"],
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": 32,
            "map_y": 32,
        },
    },
}


class SMACv2Adapter(EnvironmentAdapter[List[np.ndarray], List[int]]):
    """Adapter bridging SMACv2's procedural environment to MOSAIC.

    Key difference from SMAC v1: agent count and observation shapes may
    change between episodes due to procedural unit generation.  The adapter
    re-reads ``get_env_info()`` after every ``reset()``.
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (ControlMode.AGENT_ONLY, ControlMode.MULTI_AGENT_COOP)

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: Any | None = None,
    ) -> None:
        super().__init__(context)

        from gym_gui.config.game_configs import SMACConfig

        if config is None:
            config = SMACConfig(map_name="10gen_terran")
        if not isinstance(config, SMACConfig):
            config = SMACConfig(map_name="10gen_terran")

        self._config: Any = config
        self._map_name: str = config.map_name
        self._smac_env: Any = None
        self._n_agents: int = 0
        self._n_actions: int = 0
        self._obs_shape: int = 0
        self._state_shape: int = 0
        self._episode_limit: int = 0
        self._step_counter: int = 0
        self._camera_center: tuple[float, float] | None = None
        self._camera_width: float = 24.0
        self._zoom_level: float = 1.0  # 1.0 = full FOV, 2.0 = 2x zoom in
        self._pan_offset: tuple[float, float] = (0.0, 0.0)  # world units

    @property
    def stepping_paradigm(self) -> SteppingParadigm:  # type: ignore[override]
        return SteppingParadigm.SIMULTANEOUS

    @property
    def action_space(self) -> gym.Space[Any]:
        """Per-agent action space: Discrete(n_actions).

        SMACv2 is not a Gymnasium environment, so we construct a synthetic
        space from the env_info metadata.  Note: n_actions may change
        between episodes due to procedural generation.
        """
        if self._n_actions == 0:
            return gym.spaces.Discrete(1)
        return gym.spaces.Discrete(self._n_actions)

    @property
    def observation_space(self) -> gym.Space[Any]:
        """Per-agent observation space: Box of shape (obs_shape,).

        obs_shape may change between episodes as procedural generation
        varies agent count and unit types.
        """
        if self._obs_shape == 0:
            return gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        return gym.spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_shape,), dtype=np.float32,
        )

    def load(self) -> None:
        """Instantiate the SMACv2 environment with procedural generation.

        SC2 path resolution order:
            1. ``config.sc2_path`` (UI text field)
            2. ``SC2PATH`` environment variable (`.env` or shell)
            3. ``var/data`` project-local directory (paths.VAR_SC2_DIR)
            4. Error with download instructions
        """
        from gym_gui.config.paths import VAR_SC2_DIR

        # Resolve StarCraft II installation path
        sc2_path = (
            self._config.sc2_path
            or os.environ.get("SC2PATH")
            or (str(VAR_SC2_DIR) if VAR_SC2_DIR.is_dir() else None)
        )
        if sc2_path and os.path.isdir(sc2_path):
            os.environ["SC2PATH"] = sc2_path
        elif not os.environ.get("SC2PATH"):
            self.log_constant(
                LOG_SMAC_SC2_PATH_MISSING,
                extra={"map_name": self._map_name},
            )
            raise RuntimeError(
                "StarCraft II installation not found. "
                "Set SC2PATH environment variable, provide sc2_path in config, "
                "or install into var/data/. "
                "Download from: https://github.com/Blizzard/s2client-proto#linux-packages"
            )

        # Ensure protobuf compatibility (s2clientprotocol requires pure-python impl)
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

        from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

        # Build capability_config for procedural generation
        capability_config = SMACV2_CAPABILITY_CONFIGS.get(
            self._map_name,
            SMACV2_CAPABILITY_CONFIGS["10gen_terran"],  # fallback
        )

        env_kwargs: Dict[str, Any] = {
            "capability_config": capability_config,
            "map_name": self._map_name,
            "difficulty": self._config.difficulty,
            "reward_sparse": self._config.reward_sparse,
            "reward_only_positive": self._config.reward_only_positive,
            "reward_scale": self._config.reward_scale,
            "reward_scale_rate": self._config.reward_scale_rate,
            "obs_own_health": self._config.obs_own_health,
            "obs_pathing_grid": self._config.obs_pathing_grid,
            "obs_terrain_height": self._config.obs_terrain_height,
            "use_unit_ranges": True,
            "min_attack_range": 2,
            "obs_own_pos": True,
        }
        if self._config.seed is not None:
            env_kwargs["seed"] = self._config.seed
        if self._config.episode_limit is not None:
            env_kwargs["episode_limit"] = self._config.episode_limit

        self._smac_env = StarCraftCapabilityEnvWrapper(**env_kwargs)

        # Patch _launch() for 3D GPU rendering before any reset() call.
        # SMACv2 wrapper's __getattr__ proxies to env (StarCraft2Env).
        if getattr(self._config, "renderer", "3d") == "3d":
            from gym_gui.core.adapters.smac import _patch_launch_for_3d
            inner_env = getattr(self._smac_env, "env", self._smac_env)
            _patch_launch_for_3d(inner_env)

        env_info = self._smac_env.get_env_info()
        self._n_agents = env_info["n_agents"]
        self._n_actions = env_info["n_actions"]
        self._obs_shape = env_info["obs_shape"]
        self._state_shape = env_info["state_shape"]
        self._episode_limit = env_info["episode_limit"]

        self.log_constant(
            LOG_SMAC_ENV_CREATED,
            extra={
                "map_name": self._map_name,
                "version": "v2",
                "n_agents": self._n_agents,
                "n_actions": self._n_actions,
                "obs_shape": self._obs_shape,
                "state_shape": self._state_shape,
                "episode_limit": self._episode_limit,
            },
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[List[np.ndarray]]:
        """Reset the SMACv2 environment.

        Re-queries ``get_env_info()`` because procedural generation may
        produce different agent counts and observation shapes each episode.
        """
        if self._smac_env is None:
            self.load()

        self._smac_env.reset()
        self._step_counter = 0
        self._pan_offset = (0.0, 0.0)  # Reset pan for new episode
        self._zoom_level = 1.0

        # Cache playable area after first reset (SC2 is now running)
        if not hasattr(self, "_playable_area"):
            try:
                env_inner = getattr(self._smac_env, "env", self._smac_env)
                gi = env_inner._controller.game_info()
                pa = gi.start_raw.playable_area
                self._playable_area = (pa.p0.x, pa.p0.y, pa.p1.x, pa.p1.y)
            except Exception:
                mx = getattr(getattr(self._smac_env, "env", self._smac_env), "map_x", 32)
                my = getattr(getattr(self._smac_env, "env", self._smac_env), "map_y", 32)
                self._playable_area = (0.0, 0.0, float(mx), float(my))

        # Center 3D camera on the battle area (default camera misses SMAC units)
        if getattr(self._config, "renderer", "3d") == "3d":
            from gym_gui.core.adapters.smac import _center_camera_on_units
            inner_env = getattr(self._smac_env, "env", self._smac_env)
            center = _center_camera_on_units(inner_env)
            if center is not None:
                self._camera_center = center

        # Re-query env info -- SMACv2 can change agent/action counts per episode
        env_info = self._smac_env.get_env_info()
        self._n_agents = env_info["n_agents"]
        self._n_actions = env_info["n_actions"]
        self._obs_shape = env_info["obs_shape"]
        self._state_shape = env_info["state_shape"]

        obs = self._smac_env.get_obs()
        state = self._smac_env.get_state()
        avail_actions = [
            self._smac_env.get_avail_agent_actions(i)
            for i in range(self._n_agents)
        ]

        self.log_constant(
            LOG_SMAC_ENV_RESET,
            extra={
                "map_name": self._map_name,
                "version": "v2",
                "n_agents": self._n_agents,
                "seed": seed,
            },
        )

        return self._package_step(
            observation=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={
                "num_agents": self._n_agents,
                "agent_observations": obs,
                "global_state": state,
                "avail_actions": avail_actions,
                "action_masks": avail_actions,
                "step": 0,
            },
        )

    def step(self, action: List[int]) -> AdapterStep[List[np.ndarray]]:
        """Execute simultaneous actions for all agents."""
        if self._smac_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        reward, terminated, info = self._smac_env.step(action)
        self._step_counter += 1

        obs = self._smac_env.get_obs()
        state = self._smac_env.get_state()
        avail_actions = [
            self._smac_env.get_avail_agent_actions(i)
            for i in range(self._n_agents)
        ]

        battle_won = info.get("battle_won", False)
        agent_rewards = [float(reward)] * self._n_agents

        step_info: Dict[str, Any] = {
            "num_agents": self._n_agents,
            "agent_observations": obs,
            "global_state": state,
            "avail_actions": avail_actions,
            "action_masks": avail_actions,
            "agent_rewards": agent_rewards,
            "battle_won": battle_won,
            "step": self._step_counter,
        }
        step_info.update(info)

        self.log_constant(
            LOG_SMAC_STEP_SUMMARY,
            extra={
                "step": self._step_counter,
                "reward": float(reward),
                "terminated": terminated,
                "battle_won": battle_won,
            },
        )

        if terminated:
            self.log_constant(
                LOG_SMAC_BATTLE_RESULT,
                extra={
                    "map_name": self._map_name,
                    "version": "v2",
                    "battle_won": battle_won,
                    "steps": self._step_counter,
                },
            )

        return self._package_step(
            observation=obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=False,
            info=step_info,
        )

    def render(self) -> Optional[Dict[str, Any]]:
        """Render using 3D engine, heatmap overlays, or classic PyGame.

        The renderer is chosen by ``self._config.renderer``:

        - ``"3d"``: Full SC2 engine 3D rendering via EGL (GPU-accelerated).
        - ``"heatmap"``: 2x2 panel view with terrain, health, unit type,
          and shield/energy overlays (pure numpy).
        - ``"classic"``: SMAC's built-in PyGame renderer.
        """
        if self._smac_env is None:
            return None

        renderer = getattr(self._config, "renderer", "3d")

        if renderer == "classic":
            return self._render_pygame()

        if renderer == "3d":
            result = self._render_3d()
            if result is not None:
                return result
            renderer = "heatmap"

        if renderer == "heatmap":
            return self._render_heatmap()

        return self._render_pygame()

    def _render_3d(self) -> Optional[Dict[str, Any]]:
        """Extract 3D GPU-rendered RGB frame from SC2 engine via render_data."""
        try:
            # SMACv2 wrapper proxies _obs via __getattr__ to inner env
            env_inner = getattr(self._smac_env, "env", self._smac_env)
            obs = env_inner._obs
            if obs is None:
                return None
            observation = obs.observation
            if not observation.HasField("render_data"):
                return None
            map_img = observation.render_data.map
            if len(map_img.data) == 0:
                return None
            channels = map_img.bits_per_pixel // 8
            frame = np.frombuffer(map_img.data, dtype=np.uint8).reshape(
                map_img.size.y, map_img.size.x, channels,
            )
            if channels == 4:
                frame = frame[:, :, :3]
            # Apply software pan + zoom (numpy crop, no SC2 API calls)
            if self._zoom_level > 1.0 or self._pan_offset != (0.0, 0.0):
                from gym_gui.core.adapters.smac import _apply_pan_zoom
                frame = _apply_pan_zoom(
                    frame, self._pan_offset, self._zoom_level,
                    self._camera_width,
                )
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": frame,
                "game_id": self._map_name,
                "num_agents": self._n_agents,
                "step": self._step_counter,
            }
        except Exception as exc:
            self.log_constant(
                LOG_SMAC_RENDER_ERROR,
                exc_info=exc,
                extra={"map_name": self._map_name, "renderer": "3d"},
            )
            return None

    def _render_heatmap(self) -> Optional[Dict[str, Any]]:
        """Render using custom 2x2 heatmap feature-layer panels."""
        try:
            from gym_gui.rendering.smac_heatmap import (
                SMACHeatmapRenderer,
                extract_frame_data,
            )

            if not hasattr(self, "_heatmap_renderer"):
                self._heatmap_renderer = SMACHeatmapRenderer()

            frame_data = extract_frame_data(
                self._smac_env,
                self._step_counter,
                self._map_name,
                getattr(self, "_playable_area", (0.0, 0.0, 32.0, 32.0)),
            )
            if frame_data is None:
                return self._render_pygame()

            frame = self._heatmap_renderer.render(frame_data)
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": frame,
                "game_id": self._map_name,
                "num_agents": self._n_agents,
                "step": self._step_counter,
            }
        except Exception as exc:
            self.log_constant(
                LOG_SMAC_RENDER_ERROR,
                exc_info=exc,
                extra={"map_name": self._map_name, "renderer": "heatmap"},
            )
            return self._render_pygame()

    def _render_pygame(self) -> Optional[Dict[str, Any]]:
        """Fallback: SMAC's built-in PyGame 2D renderer."""
        if self._smac_env is None:
            return None
        try:
            frame = self._smac_env.render(mode="rgb_array")
            if isinstance(frame, np.ndarray):
                return {
                    "mode": RenderMode.RGB_ARRAY.value,
                    "rgb": frame,
                    "game_id": self._map_name,
                    "num_agents": self._n_agents,
                    "step": self._step_counter,
                }
        except Exception as exc:
            self.log_constant(
                LOG_SMAC_RENDER_ERROR,
                exc_info=exc,
                extra={"map_name": self._map_name},
            )
        return None

    def close(self) -> None:
        """Close the SMACv2 environment and terminate the SC2 process."""
        if self._smac_env is not None:
            self.log_constant(
                LOG_SMAC_ENV_CLOSED,
                extra={"map_name": self._map_name, "version": "v2"},
            )
            self._smac_env.close()
            self._smac_env = None

    def save_replay(self) -> None:
        """Save a StarCraft II replay file for the current episode."""
        if self._smac_env is not None and hasattr(self._smac_env, "save_replay"):
            self._smac_env.save_replay()

    # ─────────────────────────────────────────────────────────────────
    # 3D Camera control (mouse panning in the render widget)
    # ─────────────────────────────────────────────────────────────────

    def move_camera(self, dx_world: float, dy_world: float) -> None:
        """Pan the viewport by a world-coordinate delta (pure numpy).

        Does NOT call SC2's ``ActionRaw.camera_move`` because SMAC's
        controller does not reliably support camera moves during gameplay.
        Instead, the pan offset is applied as a numpy crop in ``_render_3d()``.

        Args:
            dx_world: Pan right (positive) or left (negative) in world units.
            dy_world: Pan up (positive) or down (negative) in world units.
        """
        px, py = self._pan_offset
        max_pan = self._camera_width / 2.0
        self._pan_offset = (
            max(-max_pan, min(max_pan, px + dx_world)),
            max(-max_pan, min(max_pan, py + dy_world)),
        )

    def zoom_camera(self, direction: int) -> None:
        """Adjust the software zoom level.

        Args:
            direction: +1 to zoom in, -1 to zoom out.
        """
        step = 0.15  # ~15% per scroll notch
        if direction > 0:
            self._zoom_level = min(4.0, self._zoom_level * (1.0 + step))
        else:
            self._zoom_level = max(1.0, self._zoom_level * (1.0 - step))

    @property
    def camera_width(self) -> float:
        """Effective world units visible (base width / zoom)."""
        return self._camera_width / self._zoom_level

    def build_step_state(
        self,
        observation: Any,
        info: Any,
    ) -> StepState:
        """Construct the canonical StepState for multi-agent display."""
        agent_snapshots: List[AgentSnapshot] = []
        info_dict = dict(info) if isinstance(info, dict) else {}

        for i in range(self._n_agents):
            snapshot = AgentSnapshot(
                name=f"agent_{i}",
                role="active",
                info={
                    "reward": info_dict.get("agent_rewards", [0.0] * self._n_agents)[i]
                    if i < len(info_dict.get("agent_rewards", []))
                    else 0.0,
                },
            )
            agent_snapshots.append(snapshot)

        return StepState(
            active_agent=None,
            agents=tuple(agent_snapshots),
            metrics={
                "step_count": self._step_counter,
                "num_agents": self._n_agents,
                "battle_won": info_dict.get("battle_won", False),
            },
            environment={
                "map_name": self._map_name,
                "family": "smacv2",
                "paradigm": "simultaneous",
            },
            raw=info_dict,
        )

    def get_avail_actions(self) -> List[List[int]]:
        """Get available action masks for all agents."""
        if self._smac_env is None:
            return []
        return [
            self._smac_env.get_avail_agent_actions(i)
            for i in range(self._n_agents)
        ]

    def get_global_state(self) -> Optional[np.ndarray]:
        """Get the global state vector (for centralized training)."""
        if self._smac_env is None:
            return None
        return self._smac_env.get_state()

    @property
    def num_agents(self) -> int:
        return self._n_agents

    @property
    def num_actions(self) -> int:
        return self._n_actions


# ═══════════════════════════════════════════════════════════════════════════
# Concrete adapter subclasses for SMACv2 procedural maps
# ═══════════════════════════════════════════════════════════════════════════


class SMACv2TerranAdapter(SMACv2Adapter):
    """10 random Terran units per episode."""

    id = GameId.SMACV2_TERRAN.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="10gen_terran")
        super().__init__(context, config=config)


class SMACv2ProtossAdapter(SMACv2Adapter):
    """10 random Protoss units per episode."""

    id = GameId.SMACV2_PROTOSS.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="10gen_protoss")
        super().__init__(context, config=config)


class SMACv2ZergAdapter(SMACv2Adapter):
    """10 random Zerg units per episode."""

    id = GameId.SMACV2_ZERG.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="10gen_zerg")
        super().__init__(context, config=config)


# ═══════════════════════════════════════════════════════════════════════════
# Adapter registry for factory pattern
# ═══════════════════════════════════════════════════════════════════════════

SMACV2_ADAPTERS: Dict[GameId, type[SMACv2Adapter]] = {
    GameId.SMACV2_TERRAN: SMACv2TerranAdapter,
    GameId.SMACV2_PROTOSS: SMACv2ProtossAdapter,
    GameId.SMACV2_ZERG: SMACv2ZergAdapter,
}

__all__ = [
    "SMACv2Adapter",
    "SMACV2_ADAPTERS",
    "SMACV2_MAP_INFO",
    "SMACV2_CAPABILITY_CONFIGS",
    "SMACv2TerranAdapter",
    "SMACv2ProtossAdapter",
    "SMACv2ZergAdapter",
]
