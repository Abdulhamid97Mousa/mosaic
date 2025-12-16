"""MiniHack adapters for roguelike reinforcement learning environments.

MiniHack is a sandbox framework for RL research built on the NetHack Learning
Environment (NLE). This module provides adapters for various MiniHack tasks
including navigation, skill acquisition, and exploration.

See: https://minihack.readthedocs.io/
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Sequence

import numpy as np
from gymnasium import spaces

from gym_gui.core.adapters.base import AdapterContext, AdapterStep, EnvironmentAdapter, StepState
from gym_gui.core.adapters.nle_render import render_chars_to_rgb
from gym_gui.core.enums import ControlMode, GameId, RenderMode

_LOGGER = logging.getLogger(__name__)


def _ensure_minihack():
    """Import MiniHack lazily and raise a helpful error when missing."""
    try:
        import minihack  # type: ignore
        import nle  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "MiniHack is not installed. Install via:\n"
            "  pip install -r requirements/nethack.txt\n"
            "or:\n"
            "  pip install -e '.[nethack]'\n"
            "Note: Requires system dependencies (build-essential, cmake, flex, bison)."
        ) from exc
    return minihack, nle


@dataclass(slots=True)
class MiniHackConfig:
    """Configuration knobs for MiniHack environments."""

    # Observation settings
    observation_keys: tuple[str, ...] = (
        "glyphs",
        "chars",
        "colors",
        "blstats",
        "message",
        "pixel",
    )
    obs_crop_h: int = 9  # Height of agent-centered crop
    obs_crop_w: int = 9  # Width of agent-centered crop

    # Gameplay settings
    max_episode_steps: int = 1000
    autopickup: bool = True  # Automatically pick up items
    pet: bool = False  # Disable pet for simpler gameplay

    # Rendering
    pixel_size: int = 16  # Size of each tile in pixel rendering (16x16 default)

    # Reward shaping (environment-specific, passed to gym.make)
    reward_lose: float = -1.0
    reward_win: float = 1.0


class MiniHackAdapter(EnvironmentAdapter[dict, int]):
    """Base adapter for MiniHack environments.

    MiniHack environments are turn-based roguelikes. Each step corresponds to
    a single game turn. The observation is a dictionary containing various
    representations of the game state (glyphs, chars, pixel, etc.).
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )

    # Subclasses override with their specific environment ID
    _env_id: str = "MiniHack-Room-5x5-v0"

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: MiniHackConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config = config or MiniHackConfig()
        self._env: Any = None
        self._step_counter = 0
        self._episode_return = 0.0
        self._last_observation: dict | None = None

    @property
    def id(self) -> str:
        return self._env_id

    def load(self) -> None:
        """Initialize the MiniHack environment."""
        import gymnasium as gym
        minihack, nle = _ensure_minihack()

        self._env = gym.make(
            self._env_id,
            observation_keys=self._config.observation_keys,
            obs_crop_h=self._config.obs_crop_h,
            obs_crop_w=self._config.obs_crop_w,
            max_episode_steps=self._config.max_episode_steps,
            autopickup=self._config.autopickup,
            pet=self._config.pet,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> AdapterStep[dict]:
        """Start a new episode."""
        if self._env is None:
            self.load()

        obs, info = self._env.reset(seed=seed, options=options)
        self._last_observation = obs
        self._step_counter = 0
        self._episode_return = 0.0

        return AdapterStep(
            observation=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
            render_payload=self.render(),
            state=self._build_step_state(obs, info),
        )

    def step(self, action: int) -> AdapterStep[dict]:
        """Execute action and return result.

        Args:
            action: Integer action from NLE action space.

        Returns:
            AdapterStep with observation, reward, and render payload.
        """
        if self._env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._last_observation = obs
        self._step_counter += 1
        self._episode_return += reward

        return AdapterStep(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            render_payload=self.render(),
            state=self._build_step_state(obs, info),
        )

    def render(self) -> dict[str, Any]:
        """Return render payload for the UI.

        MiniHack can render as pixel observations (336x1264x3) or we can
        fall back to rendering the TTY chars.
        """
        obs = self._last_observation or {}

        if "pixel" in obs and obs["pixel"] is not None:
            # Use pixel rendering (21*16 x 79*16 x 3 = 336x1264x3)
            frame = np.asarray(obs["pixel"], dtype=np.uint8)
        else:
            # Fallback: render TTY chars as simple image
            frame = self._render_tty_fallback(obs)

        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": frame,
            "game_id": self._env_id,
            "tty_chars": obs.get("chars"),
            "tty_colors": obs.get("colors"),
            "message": obs.get("message"),
            "blstats": obs.get("blstats"),
        }

    def _render_tty_fallback(self, obs: dict) -> np.ndarray:
        """Render TTY chars to RGB image using texture atlas."""
        chars = obs.get("chars")
        colors_arr = obs.get("colors")

        if chars is None:
            # Return a small black frame as placeholder
            return np.zeros((432, 800, 3), dtype=np.uint8)

        return render_chars_to_rgb(chars, colors_arr)

    def _build_step_state(self, obs: dict, info: Mapping[str, Any]) -> StepState:
        """Build machine-readable state snapshot."""
        blstats = obs.get("blstats", np.zeros(25))

        # Extract key stats from blstats vector
        # See: https://minihack.readthedocs.io/en/latest/getting-started/observation_spaces.html
        metrics = {
            "step": self._step_counter,
            "episode_return": float(self._episode_return),
        }

        environment = {
            "x": int(blstats[0]) if len(blstats) > 0 else 0,
            "y": int(blstats[1]) if len(blstats) > 1 else 0,
            "hp": int(blstats[10]) if len(blstats) > 10 else 0,
            "hp_max": int(blstats[11]) if len(blstats) > 11 else 0,
            "dungeon_level": int(blstats[12]) if len(blstats) > 12 else 1,
            "gold": int(blstats[14]) if len(blstats) > 14 else 0,
            "experience_level": int(blstats[18]) if len(blstats) > 18 else 1,
        }

        return StepState(
            metrics=metrics,
            environment=environment,
            raw=dict(info),
        )

    def close(self) -> None:
        """Clean up resources."""
        if self._env is not None:
            self._env.close()
            self._env = None
        self._last_observation = None

    @property
    def action_space(self) -> spaces.Space:
        """Return the action space."""
        if self._env is not None:
            return self._env.action_space
        # Default discrete action space for navigation
        return spaces.Discrete(8)

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space."""
        if self._env is not None:
            return self._env.observation_space
        # Placeholder dict space
        return spaces.Dict({
            "glyphs": spaces.Box(low=0, high=5991, shape=(21, 79), dtype=np.int32),
            "chars": spaces.Box(low=0, high=255, shape=(21, 79), dtype=np.uint8),
        })


# =============================================================================
# Concrete MiniHack Environment Adapters
# =============================================================================

# Navigation environments
class MiniHackRoom5x5Adapter(MiniHackAdapter):
    id = GameId.MINIHACK_ROOM_5X5.value
    _env_id = "MiniHack-Room-5x5-v0"


class MiniHackRoom15x15Adapter(MiniHackAdapter):
    id = GameId.MINIHACK_ROOM_15X15.value
    _env_id = "MiniHack-Room-15x15-v0"


class MiniHackCorridorR2Adapter(MiniHackAdapter):
    id = GameId.MINIHACK_CORRIDOR_R2.value
    _env_id = "MiniHack-Corridor-R2-v0"


class MiniHackCorridorR3Adapter(MiniHackAdapter):
    id = GameId.MINIHACK_CORRIDOR_R3.value
    _env_id = "MiniHack-Corridor-R3-v0"


class MiniHackCorridorR5Adapter(MiniHackAdapter):
    id = GameId.MINIHACK_CORRIDOR_R5.value
    _env_id = "MiniHack-Corridor-R5-v0"


class MiniHackMazeWalk9x9Adapter(MiniHackAdapter):
    id = GameId.MINIHACK_MAZEWALK_9X9.value
    _env_id = "MiniHack-MazeWalk-9x9-v0"


class MiniHackMazeWalk15x15Adapter(MiniHackAdapter):
    id = GameId.MINIHACK_MAZEWALK_15X15.value
    _env_id = "MiniHack-MazeWalk-15x15-v0"


class MiniHackMazeWalk45x19Adapter(MiniHackAdapter):
    id = GameId.MINIHACK_MAZEWALK_45X19.value
    _env_id = "MiniHack-MazeWalk-45x19-v0"


class MiniHackRiverAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_RIVER.value
    _env_id = "MiniHack-River-v0"


class MiniHackRiverNarrowAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_RIVER_NARROW.value
    _env_id = "MiniHack-River-Narrow-v0"


# Skill environments
class MiniHackEatAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_EAT.value
    _env_id = "MiniHack-Eat-v0"


class MiniHackWearAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_WEAR.value
    _env_id = "MiniHack-Wear-v0"


class MiniHackWieldAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_WIELD.value
    _env_id = "MiniHack-Wield-v0"


class MiniHackZapAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_ZAP.value
    _env_id = "MiniHack-Zap-v0"


class MiniHackReadAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_READ.value
    _env_id = "MiniHack-Read-v0"


class MiniHackQuaffAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_QUAFF.value
    _env_id = "MiniHack-Quaff-v0"


class MiniHackPutOnAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_PUTON.value
    _env_id = "MiniHack-PutOn-v0"


class MiniHackLavaCrossAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_LAVACROSS.value
    _env_id = "MiniHack-LavaCross-v0"


class MiniHackWoDEasyAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_WOD_EASY.value
    _env_id = "MiniHack-WoD-Easy-v0"


class MiniHackWoDMediumAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_WOD_MEDIUM.value
    _env_id = "MiniHack-WoD-Medium-v0"


class MiniHackWoDHardAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_WOD_HARD.value
    _env_id = "MiniHack-WoD-Hard-v0"


# Exploration environments
class MiniHackExploreMazeEasyAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_EXPLOREMAZE_EASY.value
    _env_id = "MiniHack-ExploreMaze-Easy-v0"


class MiniHackExploreMazeHardAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_EXPLOREMAZE_HARD.value
    _env_id = "MiniHack-ExploreMaze-Hard-v0"


class MiniHackHideNSeekAdapter(MiniHackAdapter):
    id = GameId.MINIHACK_HIDENSEEK.value
    _env_id = "MiniHack-HideNSeek-v0"


class MiniHackMementoF2Adapter(MiniHackAdapter):
    id = GameId.MINIHACK_MEMENTO_F2.value
    _env_id = "MiniHack-Memento-F2-v0"


class MiniHackMementoF4Adapter(MiniHackAdapter):
    id = GameId.MINIHACK_MEMENTO_F4.value
    _env_id = "MiniHack-Memento-F4-v0"


# =============================================================================
# Adapter Registry
# =============================================================================

MINIHACK_ADAPTERS: dict[GameId, type[MiniHackAdapter]] = {
    # Navigation
    GameId.MINIHACK_ROOM_5X5: MiniHackRoom5x5Adapter,
    GameId.MINIHACK_ROOM_15X15: MiniHackRoom15x15Adapter,
    GameId.MINIHACK_CORRIDOR_R2: MiniHackCorridorR2Adapter,
    GameId.MINIHACK_CORRIDOR_R3: MiniHackCorridorR3Adapter,
    GameId.MINIHACK_CORRIDOR_R5: MiniHackCorridorR5Adapter,
    GameId.MINIHACK_MAZEWALK_9X9: MiniHackMazeWalk9x9Adapter,
    GameId.MINIHACK_MAZEWALK_15X15: MiniHackMazeWalk15x15Adapter,
    GameId.MINIHACK_MAZEWALK_45X19: MiniHackMazeWalk45x19Adapter,
    GameId.MINIHACK_RIVER: MiniHackRiverAdapter,
    GameId.MINIHACK_RIVER_NARROW: MiniHackRiverNarrowAdapter,
    # Skills
    GameId.MINIHACK_EAT: MiniHackEatAdapter,
    GameId.MINIHACK_WEAR: MiniHackWearAdapter,
    GameId.MINIHACK_WIELD: MiniHackWieldAdapter,
    GameId.MINIHACK_ZAP: MiniHackZapAdapter,
    GameId.MINIHACK_READ: MiniHackReadAdapter,
    GameId.MINIHACK_QUAFF: MiniHackQuaffAdapter,
    GameId.MINIHACK_PUTON: MiniHackPutOnAdapter,
    GameId.MINIHACK_LAVACROSS: MiniHackLavaCrossAdapter,
    GameId.MINIHACK_WOD_EASY: MiniHackWoDEasyAdapter,
    GameId.MINIHACK_WOD_MEDIUM: MiniHackWoDMediumAdapter,
    GameId.MINIHACK_WOD_HARD: MiniHackWoDHardAdapter,
    # Exploration
    GameId.MINIHACK_EXPLOREMAZE_EASY: MiniHackExploreMazeEasyAdapter,
    GameId.MINIHACK_EXPLOREMAZE_HARD: MiniHackExploreMazeHardAdapter,
    GameId.MINIHACK_HIDENSEEK: MiniHackHideNSeekAdapter,
    GameId.MINIHACK_MEMENTO_F2: MiniHackMementoF2Adapter,
    GameId.MINIHACK_MEMENTO_F4: MiniHackMementoF4Adapter,
}
