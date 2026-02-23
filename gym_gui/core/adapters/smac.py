"""SMAC v1 (StarCraft Multi-Agent Challenge) adapter for the MOSAIC GUI.

Bridges SMAC's custom ``MultiAgentEnv`` API to MOSAIC's ``EnvironmentAdapter``
interface.  SMAC does NOT use Gymnasium -- it has its own API with methods like
``get_obs()``, ``get_state()``, and ``get_avail_agent_actions()``.

Supports three renderer modes:
- ``"3d"``: Full SC2 engine 3D rendering via EGL (GPU-accelerated).
- ``"heatmap"``: Custom 2x2 feature-layer panel (pure numpy).
- ``"classic"``: SMAC's built-in PyGame 2D renderer (colored circles).

Repository: https://github.com/oxwhirl/smac
Paper:      https://arxiv.org/abs/1902.04043
"""

from __future__ import annotations

import logging
import os
import types
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
    LOG_SMAC_ACTION_MASK_WARN,
    LOG_SMAC_BATTLE_RESULT,
    LOG_SMAC_ENV_CLOSED,
    LOG_SMAC_ENV_CREATED,
    LOG_SMAC_ENV_RESET,
    LOG_SMAC_RENDER_ERROR,
    LOG_SMAC_SC2_PATH_MISSING,
    LOG_SMAC_STEP_SUMMARY,
)

_LOGGER = logging.getLogger(__name__)

# Action names for telemetry and display
SMAC_BASE_ACTIONS: List[str] = [
    "NO-OP",       # 0 -- only valid action for dead agents
    "STOP",        # 1 -- stop current movement/attack
    "MOVE_NORTH",  # 2
    "MOVE_SOUTH",  # 3
    "MOVE_EAST",   # 4
    "MOVE_WEST",   # 5
    # indices 6+ are ATTACK_ENEMY_0, ATTACK_ENEMY_1, ...
]

# Map metadata: map_name -> (ally_count, enemy_description, difficulty_tier)
SMAC_MAP_INFO: Dict[str, tuple[int, str, str]] = {
    "3m": (3, "3 Marines", "Easy"),
    "8m": (8, "8 Marines", "Easy"),
    "2s3z": (5, "2 Stalkers + 3 Zealots", "Easy"),
    "3s5z": (8, "3 Stalkers + 5 Zealots", "Easy"),
    "5m_vs_6m": (5, "6 Marines", "Hard"),
    "MMM2": (10, "1 Medivac + 3 Marauders + 8 Marines", "Super Hard"),
}


# Default resolution for 3D GPU rendering
_3D_RENDER_SIZE = 512


def _patch_launch_for_3d(smac_env: Any, render_size: int = _3D_RENDER_SIZE) -> None:
    """Monkey-patch ``_launch()`` on a SMAC env to enable 3D GPU rendering.

    SMAC hardcodes ``want_rgb=False`` (which adds ``-headlessNoRender`` to the
    SC2 command line) and omits the ``render`` field from ``InterfaceOptions``.
    This patch overrides ``_launch()`` to pass ``want_rgb=True`` and add a
    ``SpatialCameraSetup`` so the SC2 engine returns RGB frames via EGL.
    """
    original_launch = smac_env._launch

    def _launch_with_render(self_env: Any = smac_env) -> None:
        # Ensure absl flags are parsed (pysc2 requirement)
        try:
            import sys
            from absl import flags
            if not flags.FLAGS.is_parsed():
                flags.FLAGS(sys.argv)
        except Exception:
            pass

        # Register SMAC maps with pysc2 before maps.get()
        import smac.env.starcraft2.maps  # noqa: F401

        from pysc2 import run_configs, maps
        from s2clientprotocol import sc2api_pb2 as sc_pb

        self_env._run_config = run_configs.get(version=self_env.game_version)
        # SMACv2 stores version (v1 doesn't, but setting it is harmless)
        if hasattr(self_env._run_config, "version"):
            self_env.version = self_env._run_config.version
        _map = maps.get(self_env.map_name)

        # Enable 3D rendering: want_rgb=True prevents -headlessNoRender
        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
        interface_options.render.resolution.x = render_size
        interface_options.render.resolution.y = render_size
        interface_options.render.minimap_resolution.x = render_size // 2
        interface_options.render.minimap_resolution.y = render_size // 2
        interface_options.render.width = 24.0  # World units visible in 3D camera

        self_env._sc2_proc = self_env._run_config.start(
            window_size=self_env.window_size, want_rgb=True,
        )
        self_env._controller = self_env._sc2_proc.controller

        # Create game (copied from SMAC's _launch)
        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self_env._run_config.map_data(_map.path),
            ),
            realtime=False,
            random_seed=self_env._seed,
        )
        create.player_setup.add(type=sc_pb.Participant)

        from smac.env.starcraft2.starcraft2 import races, difficulties
        create.player_setup.add(
            type=sc_pb.Computer,
            race=races[self_env._bot_race],
            difficulty=difficulties[self_env.difficulty],
        )
        self_env._controller.create_game(create)

        join = sc_pb.RequestJoinGame(
            race=races[self_env._agent_race], options=interface_options,
        )
        # SMACv2 stores the join request as self.game
        self_env.game = join
        self_env._controller.join_game(join)

        # Post-join setup (terrain, pathing, map dimensions) -- same as original
        game_info = self_env._controller.game_info()
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        # SMACv2 stores these as instance attributes
        self_env.map_play_area_min = map_play_area_min
        self_env.map_play_area_max = map_play_area_max
        self_env.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self_env.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self_env.map_x = map_info.map_size.x
        self_env.map_y = map_info.map_size.y

        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self_env.map_x, int(self_env.map_y / 8)
            )
            self_env.pathing_grid = np.transpose(
                np.array(
                    [
                        [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                        for row in vals
                    ],
                    dtype=bool,
                )
            )
        else:
            self_env.pathing_grid = np.invert(
                np.flip(
                    np.transpose(
                        np.array(
                            list(map_info.pathing_grid.data), dtype=bool
                        ).reshape(self_env.map_x, self_env.map_y)
                    ),
                    axis=1,
                )
            )

        self_env.terrain_height = (
            np.flip(
                np.transpose(
                    np.array(list(map_info.terrain_height.data)).reshape(
                        self_env.map_x, self_env.map_y
                    )
                ),
                1,
            )
            / 255
        )

    smac_env._launch = types.MethodType(lambda self: _launch_with_render(self), smac_env)


def _center_camera_on_units(smac_env: Any) -> tuple[float, float] | None:
    """Move the 3D render camera to center on the unit centroid.

    After each ``reset()``, SC2's camera reverts to its default position which
    is typically NOT over the SMAC battle area.  This function computes the
    centroid of all units from the latest observation and sends an
    ``ActionRaw.camera_move`` to reposition the camera, followed by a minimal
    ``step(1)`` so the next ``observe()`` returns the correctly-framed view.

    Returns the (cx, cy) world coordinates of the new camera center, or None.
    """
    try:
        from s2clientprotocol import sc2api_pb2 as sc_pb

        obs = smac_env._obs
        if obs is None:
            return None
        units = obs.observation.raw_data.units
        if not units:
            return None

        # Compute centroid of all units (allies + enemies)
        cx = sum(u.pos.x for u in units) / len(units)
        cy = sum(u.pos.y for u in units) / len(units)

        _move_camera_to(smac_env, cx, cy)
        return (cx, cy)
    except Exception:
        return None


def _move_camera_to(smac_env: Any, cx: float, cy: float) -> None:
    """Send an ActionRaw.camera_move to the given world coordinates."""
    try:
        from s2clientprotocol import sc2api_pb2 as sc_pb

        action = sc_pb.Action()
        action.action_raw.camera_move.center_world_space.x = cx
        action.action_raw.camera_move.center_world_space.y = cy

        smac_env._controller.act([action])

        # Advance one simulation tick so the camera move takes effect
        # on the next observe(). One tick is negligible (~0.04s game time).
        smac_env._controller.step(1)
        smac_env._obs = smac_env._controller.observe()
    except Exception:
        pass


def _apply_pan_zoom(
    frame: np.ndarray,
    pan_offset: tuple[float, float],
    zoom_level: float,
    camera_width: float,
) -> np.ndarray:
    """Crop a sub-region of *frame* based on pan offset and zoom, then resize.

    Both pan and zoom are **pure numpy operations** -- no SC2 API calls.
    The SC2 engine renders a wide FOV (``camera_width`` world units); this
    function selects which portion of that wide render to show.

    Args:
        frame:  Full rendered frame from SC2, shape (H, W, 3).
        pan_offset:  (dx, dy) pan in world units.  Positive dx = right,
                     positive dy = up (SC2 convention).
        zoom_level:  1.0 = full FOV, 2.0 = show center half, etc.
        camera_width:  World units covered by the full frame width.

    Returns:
        Cropped + resized frame at the original (H, W, 3) shape.
    """
    if zoom_level <= 1.0 and pan_offset == (0.0, 0.0):
        return frame

    h, w = frame.shape[:2]
    zoom = max(1.0, zoom_level)

    # Viewport size in pixels
    view_w = w / zoom
    view_h = h / zoom

    # Convert pan from world units to pixels
    px_per_unit = w / camera_width
    pan_px = pan_offset[0] * px_per_unit
    pan_py = -pan_offset[1] * px_per_unit  # Y-flip: SC2 Y-up vs pixel Y-down

    # Center of the crop window (default = frame center + pan)
    cx = w / 2.0 + pan_px
    cy = h / 2.0 + pan_py

    # Compute crop boundaries and clamp to frame edges
    x0 = int(max(0, min(cx - view_w / 2, w - view_w)))
    y0 = int(max(0, min(cy - view_h / 2, h - view_h)))
    x1 = int(min(w, x0 + view_w))
    y1 = int(min(h, y0 + view_h))

    if x1 <= x0 or y1 <= y0:
        return frame

    cropped = frame[y0:y1, x0:x1]

    # Nearest-neighbor resize back to (h, w)
    row_idx = np.linspace(0, cropped.shape[0] - 1, h).astype(int)
    col_idx = np.linspace(0, cropped.shape[1] - 1, w).astype(int)
    return cropped[np.ix_(row_idx, col_idx)]


class SMACAdapter(EnvironmentAdapter[List[np.ndarray], List[int]]):
    """Adapter bridging SMAC v1's ``MultiAgentEnv`` to MOSAIC's adapter interface.

    SMAC v1 uses hand-designed maps with fixed team compositions.
    All agents act simultaneously each timestep (parallel stepping).
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

        # Import config type locally to avoid circular imports
        from gym_gui.config.game_configs import SMACConfig

        if config is None:
            config = SMACConfig()
        if not isinstance(config, SMACConfig):
            config = SMACConfig()

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

        SMAC is not a Gymnasium environment, so we construct a synthetic
        space from the env_info metadata returned after load().
        """
        if self._n_actions == 0:
            return gym.spaces.Discrete(1)
        return gym.spaces.Discrete(self._n_actions)

    @property
    def observation_space(self) -> gym.Space[Any]:
        """Per-agent observation space: Box of shape (obs_shape,).

        Each agent receives a float vector containing distances, health,
        shield, unit type, and relative positions of allies/enemies.
        """
        if self._obs_shape == 0:
            return gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        return gym.spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_shape,), dtype=np.float32,
        )

    def load(self) -> None:
        """Instantiate the SMAC v1 environment.

        Validates SC2PATH and imports from the ``smac`` package.

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

        from smac.env import StarCraft2Env

        env_kwargs: Dict[str, Any] = {
            "map_name": self._map_name,
            "difficulty": self._config.difficulty,
            "reward_sparse": self._config.reward_sparse,
            "reward_only_positive": self._config.reward_only_positive,
            "reward_scale": self._config.reward_scale,
            "reward_scale_rate": self._config.reward_scale_rate,
            "obs_own_health": self._config.obs_own_health,
            "obs_pathing_grid": self._config.obs_pathing_grid,
            "obs_terrain_height": self._config.obs_terrain_height,
        }
        if self._config.seed is not None:
            env_kwargs["seed"] = self._config.seed
        if self._config.episode_limit is not None:
            env_kwargs["episode_limit"] = self._config.episode_limit

        self._smac_env = StarCraft2Env(**env_kwargs)

        # Patch _launch() for 3D GPU rendering before any reset() call
        if getattr(self._config, "renderer", "3d") == "3d":
            _patch_launch_for_3d(self._smac_env)

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
        """Reset the SMAC environment and return initial observations."""
        if self._smac_env is None:
            self.load()

        self._smac_env.reset()
        self._step_counter = 0
        self._pan_offset = (0.0, 0.0)  # Reset pan for new episode
        self._zoom_level = 1.0

        # Cache playable area after first reset (SC2 is now running)
        if not hasattr(self, "_playable_area"):
            try:
                gi = self._smac_env._controller.game_info()
                pa = gi.start_raw.playable_area
                self._playable_area = (pa.p0.x, pa.p0.y, pa.p1.x, pa.p1.y)
            except Exception:
                self._playable_area = (
                    0.0, 0.0,
                    float(self._smac_env.map_x),
                    float(self._smac_env.map_y),
                )

        # Center 3D camera on the battle area (default camera misses SMAC units)
        if getattr(self._config, "renderer", "3d") == "3d":
            center = _center_camera_on_units(self._smac_env)
            if center is not None:
                self._camera_center = center

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
        """Execute simultaneous actions for all agents.

        Args:
            action: List of integer actions, one per agent.

        Returns:
            AdapterStep with per-agent observations, shared reward, and info
            containing global_state and action_masks.
        """
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

        # SMAC returns a single shared reward (fully cooperative)
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
                    "battle_won": battle_won,
                    "steps": self._step_counter,
                    "total_reward": float(reward),
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
        - ``"classic"``: SMAC's built-in PyGame renderer (coloured circles
          with health arcs).
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
            # Fall through to heatmap if 3D fails
            renderer = "heatmap"

        if renderer == "heatmap":
            return self._render_heatmap()

        return self._render_pygame()

    def _render_3d(self) -> Optional[Dict[str, Any]]:
        """Extract 3D GPU-rendered RGB frame from SC2 engine via render_data."""
        try:
            obs = self._smac_env._obs
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
                frame = frame[:, :, :3]  # Drop alpha channel
            # Apply software pan + zoom (numpy crop, no SC2 API calls)
            if self._zoom_level > 1.0 or self._pan_offset != (0.0, 0.0):
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
        """Close the SMAC environment and terminate the SC2 process."""
        if self._smac_env is not None:
            self.log_constant(
                LOG_SMAC_ENV_CLOSED,
                extra={"map_name": self._map_name},
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
        # Clamp pan so the viewport stays within the rendered frame
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
            active_agent=None,  # simultaneous -- no single active agent
            agents=tuple(agent_snapshots),
            metrics={
                "step_count": self._step_counter,
                "num_agents": self._n_agents,
                "battle_won": info_dict.get("battle_won", False),
            },
            environment={
                "map_name": self._map_name,
                "family": "smac",
                "paradigm": "simultaneous",
            },
            raw=info_dict,
        )

    # ─────────────────────────────────────────────────────────────────
    # Multi-agent helper methods
    # ─────────────────────────────────────────────────────────────────

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
# Concrete adapter subclasses for each SMAC v1 map
# ═══════════════════════════════════════════════════════════════════════════


class SMAC3MAdapter(SMACAdapter):
    """3 Marines vs 3 Marines (Easy)."""

    id = GameId.SMAC_3M.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="3m")
        super().__init__(context, config=config)


class SMAC8MAdapter(SMACAdapter):
    """8 Marines vs 8 Marines (Easy)."""

    id = GameId.SMAC_8M.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="8m")
        super().__init__(context, config=config)


class SMAC2S3ZAdapter(SMACAdapter):
    """2 Stalkers + 3 Zealots vs same (Easy)."""

    id = GameId.SMAC_2S3Z.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="2s3z")
        super().__init__(context, config=config)


class SMAC3S5ZAdapter(SMACAdapter):
    """3 Stalkers + 5 Zealots vs same (Easy)."""

    id = GameId.SMAC_3S5Z.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="3s5z")
        super().__init__(context, config=config)


class SMAC5Mvs6MAdapter(SMACAdapter):
    """5 Marines vs 6 Marines (Hard, asymmetric)."""

    id = GameId.SMAC_5M_VS_6M.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="5m_vs_6m")
        super().__init__(context, config=config)


class SMACMMM2Adapter(SMACAdapter):
    """1 Medivac + 2 Marauders + 7 Marines vs 1M+3Ma+8Mar (Super Hard)."""

    id = GameId.SMAC_MMM2.value

    def __init__(self, context: AdapterContext | None = None, *, config: Any | None = None) -> None:
        from gym_gui.config.game_configs import SMACConfig
        if config is None:
            config = SMACConfig(map_name="MMM2")
        super().__init__(context, config=config)


# ═══════════════════════════════════════════════════════════════════════════
# Adapter registry for factory pattern
# ═══════════════════════════════════════════════════════════════════════════

SMAC_ADAPTERS: Dict[GameId, type[SMACAdapter]] = {
    GameId.SMAC_3M: SMAC3MAdapter,
    GameId.SMAC_8M: SMAC8MAdapter,
    GameId.SMAC_2S3Z: SMAC2S3ZAdapter,
    GameId.SMAC_3S5Z: SMAC3S5ZAdapter,
    GameId.SMAC_5M_VS_6M: SMAC5Mvs6MAdapter,
    GameId.SMAC_MMM2: SMACMMM2Adapter,
}

__all__ = [
    "SMACAdapter",
    "SMAC_ADAPTERS",
    "SMAC_BASE_ACTIONS",
    "SMAC_MAP_INFO",
    "SMAC3MAdapter",
    "SMAC8MAdapter",
    "SMAC2S3ZAdapter",
    "SMAC3S5ZAdapter",
    "SMAC5Mvs6MAdapter",
    "SMACMMM2Adapter",
]
