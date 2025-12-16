"""NetHack adapters for the full roguelike game via NLE.

NetHack is a classic dungeon crawler roguelike (1987). This module provides
adapters for the full NetHack game accessed via the NetHack Learning Environment
(NLE). Unlike MiniHack (sandbox environments), these are the complete game
with various objective functions for RL training.

See: https://github.com/facebookresearch/nle
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

from gym_gui.core.adapters.base import AdapterContext, AdapterStep, EnvironmentAdapter, StepState
from gym_gui.core.adapters.nle_render import render_tty_to_rgb, render_chars_to_rgb
from gym_gui.core.enums import ControlMode, GameId, RenderMode

_LOGGER = logging.getLogger(__name__)


def _ensure_nle():
    """Import NLE lazily and raise a helpful error when missing."""
    try:
        import nle  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "NLE (NetHack Learning Environment) is not installed. Install via:\n"
            "  pip install -r requirements/nethack.txt\n"
            "or:\n"
            "  pip install -e '.[nethack]'\n"
            "Note: Requires system dependencies (build-essential, cmake, flex, bison).\n"
            "See: https://github.com/facebookresearch/nle"
        ) from exc
    return nle


@dataclass(slots=True)
class NetHackConfig:
    """Configuration knobs for NetHack environments."""

    # Observation settings
    observation_keys: tuple[str, ...] = (
        "glyphs",
        "chars",
        "colors",
        "specials",
        "blstats",
        "message",
        "inv_glyphs",
        "inv_strs",
        "inv_letters",
        "inv_oclasses",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
        "screen_descriptions",
    )

    # Gameplay settings
    max_episode_steps: int = 100_000  # Full game can be very long
    character: str = "@"  # Default character (@ = random)
    savedir: str | None = None  # Directory for save files

    # Penalty/reward settings (used by some NLE tasks)
    penalty_step: float = -0.001
    penalty_time: float = -0.0
    penalty_mode: str = "constant"  # "constant", "exp", "square", "linear"


class NetHackAdapter(EnvironmentAdapter[dict, int]):
    """Base adapter for full NetHack game environments.

    NetHack is the complete roguelike with 100+ dungeon levels, complex
    interactions, and permadeath. Each NLE task variant has different
    objective functions (score, reach staircase, find oracle, etc.).
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )

    # Subclasses override with their specific environment ID
    _env_id: str = "NetHackChallenge-v0"

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: NetHackConfig | None = None,
    ) -> None:
        super().__init__(context)
        self._config = config or NetHackConfig()
        self._env: Any = None
        self._step_counter = 0
        self._episode_return = 0.0
        self._last_observation: dict | None = None

    @property
    def id(self) -> str:
        return self._env_id

    def load(self) -> None:
        """Initialize the NetHack environment."""
        import gymnasium as gym
        nle = _ensure_nle()

        # Build kwargs based on environment type
        kwargs: dict[str, Any] = {
            "observation_keys": self._config.observation_keys,
            "max_episode_steps": self._config.max_episode_steps,
        }

        # Some NLE environments accept additional parameters
        if self._config.savedir:
            kwargs["savedir"] = self._config.savedir
        if self._config.character != "@":
            kwargs["character"] = self._config.character

        self._env = gym.make(self._env_id, **kwargs)

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
            action: Integer action from NLE action space (typically 0-112).

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

        NLE provides tty_chars/tty_colors for TTY rendering. We convert
        this to an RGB image for display in the GUI.
        """
        obs = self._last_observation or {}

        # Use TTY rendering (24x80 terminal)
        tty_chars = obs.get("tty_chars")
        tty_colors = obs.get("tty_colors")

        if tty_chars is not None:
            frame = self._render_tty_to_rgb(tty_chars, tty_colors)
        else:
            # Fallback: render glyphs/chars
            frame = self._render_glyphs_fallback(obs)

        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": frame,
            "game_id": self._env_id,
            "tty_chars": tty_chars,
            "tty_colors": tty_colors,
            "message": obs.get("message"),
            "blstats": obs.get("blstats"),
            "screen_descriptions": obs.get("screen_descriptions"),
        }

    def _render_tty_to_rgb(
        self, tty_chars: np.ndarray, tty_colors: np.ndarray | None
    ) -> np.ndarray:
        """Convert TTY chars/colors to RGB image using texture atlas.

        Args:
            tty_chars: Shape (24, 80) uint8 array of ASCII chars
            tty_colors: Shape (24, 80) int8 array of color codes

        Returns:
            RGB image array suitable for display.
        """
        return render_tty_to_rgb(tty_chars, tty_colors)

    def _render_glyphs_fallback(self, obs: dict) -> np.ndarray:
        """Render glyphs/chars to RGB image when TTY obs unavailable."""
        chars = obs.get("chars")
        colors_arr = obs.get("colors")

        if chars is None:
            # Return placeholder black frame
            return np.zeros((432, 800, 3), dtype=np.uint8)

        return render_chars_to_rgb(chars, colors_arr)

    def _build_step_state(self, obs: dict, info: Mapping[str, Any]) -> StepState:
        """Build machine-readable state snapshot."""
        blstats = obs.get("blstats", np.zeros(26))

        # Extract key stats from blstats vector
        # See: https://nethackwiki.com/wiki/Bottom_line
        metrics = {
            "step": self._step_counter,
            "episode_return": float(self._episode_return),
            "score": int(blstats[9]) if len(blstats) > 9 else 0,
        }

        environment = {
            "x": int(blstats[0]) if len(blstats) > 0 else 0,
            "y": int(blstats[1]) if len(blstats) > 1 else 0,
            "strength": int(blstats[2]) if len(blstats) > 2 else 0,
            "dexterity": int(blstats[4]) if len(blstats) > 4 else 0,
            "constitution": int(blstats[5]) if len(blstats) > 5 else 0,
            "intelligence": int(blstats[6]) if len(blstats) > 6 else 0,
            "wisdom": int(blstats[7]) if len(blstats) > 7 else 0,
            "charisma": int(blstats[8]) if len(blstats) > 8 else 0,
            "hp": int(blstats[10]) if len(blstats) > 10 else 0,
            "hp_max": int(blstats[11]) if len(blstats) > 11 else 0,
            "dungeon_level": int(blstats[12]) if len(blstats) > 12 else 1,
            "gold": int(blstats[13]) if len(blstats) > 13 else 0,
            "energy": int(blstats[14]) if len(blstats) > 14 else 0,
            "energy_max": int(blstats[15]) if len(blstats) > 15 else 0,
            "armor_class": int(blstats[16]) if len(blstats) > 16 else 10,
            "experience_level": int(blstats[18]) if len(blstats) > 18 else 1,
            "experience_points": int(blstats[19]) if len(blstats) > 19 else 0,
            "time": int(blstats[20]) if len(blstats) > 20 else 0,
            "hunger_state": int(blstats[21]) if len(blstats) > 21 else 0,
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
        # NLE default action space: 113 actions (including all commands)
        return spaces.Discrete(113)

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space."""
        if self._env is not None:
            return self._env.observation_space
        # Placeholder dict space
        return spaces.Dict({
            "glyphs": spaces.Box(low=0, high=5991, shape=(21, 79), dtype=np.int32),
            "chars": spaces.Box(low=0, high=255, shape=(21, 79), dtype=np.uint8),
            "tty_chars": spaces.Box(low=0, high=255, shape=(24, 80), dtype=np.uint8),
        })


# =============================================================================
# Concrete NetHack Environment Adapters
# =============================================================================

class NetHackChallengeAdapter(NetHackAdapter):
    """Full NetHack game - the complete challenge."""
    id = GameId.NETHACK_FULL.value
    _env_id = "NetHackChallenge-v0"


class NetHackScoreAdapter(NetHackAdapter):
    """NetHack with score-based objective."""
    id = GameId.NETHACK_SCORE.value
    _env_id = "NetHackScore-v0"


class NetHackStaircaseAdapter(NetHackAdapter):
    """NetHack objective: reach the staircase to descend."""
    id = GameId.NETHACK_STAIRCASE.value
    _env_id = "NetHackStaircase-v0"


class NetHackStaircasePetAdapter(NetHackAdapter):
    """NetHack objective: reach staircase with your pet."""
    id = GameId.NETHACK_STAIRCASE_PET.value
    _env_id = "NetHackStaircasePet-v0"


class NetHackOracleAdapter(NetHackAdapter):
    """NetHack objective: find the Oracle on dungeon level 5-9."""
    id = GameId.NETHACK_ORACLE.value
    _env_id = "NetHackOracle-v0"


class NetHackGoldAdapter(NetHackAdapter):
    """NetHack objective: collect gold."""
    id = GameId.NETHACK_GOLD.value
    _env_id = "NetHackGold-v0"


class NetHackEatAdapter(NetHackAdapter):
    """NetHack objective: eat food to survive."""
    id = GameId.NETHACK_EAT.value
    _env_id = "NetHackEat-v0"


class NetHackScoutAdapter(NetHackAdapter):
    """NetHack objective: explore and scout the dungeon."""
    id = GameId.NETHACK_SCOUT.value
    _env_id = "NetHackScout-v0"


# =============================================================================
# Adapter Registry
# =============================================================================

NETHACK_ADAPTERS: dict[GameId, type[NetHackAdapter]] = {
    GameId.NETHACK_FULL: NetHackChallengeAdapter,
    GameId.NETHACK_SCORE: NetHackScoreAdapter,
    GameId.NETHACK_STAIRCASE: NetHackStaircaseAdapter,
    GameId.NETHACK_STAIRCASE_PET: NetHackStaircasePetAdapter,
    GameId.NETHACK_ORACLE: NetHackOracleAdapter,
    GameId.NETHACK_GOLD: NetHackGoldAdapter,
    GameId.NETHACK_EAT: NetHackEatAdapter,
    GameId.NETHACK_SCOUT: NetHackScoutAdapter,
}
