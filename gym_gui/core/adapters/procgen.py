"""Procgen environment adapters for the Procgen procedural benchmark.

Procgen provides 16 procedurally-generated game-like environments designed
to measure sample efficiency and generalization in reinforcement learning.

Paper: Cobbe et al. (2019). Leveraging Procedural Generation to Benchmark RL.
Repository: https://github.com/openai/procgen
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

from gym_gui.config.game_configs import ProcgenConfig
from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode

_LOGGER = logging.getLogger(__name__)

# Try to import procgen - it's an optional dependency
# For Python 3.11+, procgen-mirror provides compatible wheels
# We use shimmy to wrap old gym environments for gymnasium compatibility
try:
    import gym  # Procgen uses old gym, not gymnasium
    from shimmy import GymV21CompatibilityV0  # For old gym API (pre-v0.26)
    try:
        from procgen import ProcgenEnv
    except ImportError:
        # Fall back to procgen-mirror for Python 3.11+
        from procgen_mirror import ProcgenEnv  # type: ignore[import-not-found]
    _PROCGEN_AVAILABLE = True
    _SHIMMY_AVAILABLE = True
except ImportError:
    _PROCGEN_AVAILABLE = False
    _SHIMMY_AVAILABLE = False
    gym = None  # type: ignore[assignment]
    GymV21CompatibilityV0 = None  # type: ignore[assignment,misc]
    ProcgenEnv = None  # type: ignore[assignment,misc]


_PROCGEN_STEP_LOG_FREQUENCY = 100

# Procgen environment names (16 total)
PROCGEN_ENV_NAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]

# Procgen action combos (15 total) - matches env.py get_combos()
PROCGEN_ACTIONS = [
    "down_left",      # 0: LEFT+DOWN
    "left",           # 1: LEFT
    "up_left",        # 2: LEFT+UP
    "down",           # 3: DOWN
    "noop",           # 4: (none)
    "up",             # 5: UP
    "down_right",     # 6: RIGHT+DOWN
    "right",          # 7: RIGHT
    "up_right",       # 8: RIGHT+UP
    "action_d",       # 9: D (fire/interact - game specific)
    "action_a",       # 10: A (secondary - game specific)
    "action_w",       # 11: W (tertiary - game specific)
    "action_s",       # 12: S (quaternary - game specific)
    "action_q",       # 13: Q (special 1 - game specific)
    "action_e",       # 14: E (special 2 - game specific)
]


def _ensure_procgen() -> None:
    """Ensure Procgen is available, raise helpful error if not."""
    if not _PROCGEN_AVAILABLE:
        raise ImportError(
            "Procgen is not installed. Install via:\n"
            "  pip install procgen\n"
            "or:\n"
            "  pip install -e '.[procgen]'\n"
            "Note: Requires Python 3.7-3.10 and CPU with AVX support."
        )


class ProcgenAdapter(EnvironmentAdapter[np.ndarray, int]):
    """Base adapter for Procgen environments with RGB observations.

    Procgen provides 64x64 RGB images as observations and has a discrete
    action space of 15 actions (button combinations).
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )

    # Subclasses override with their specific environment name
    _env_name: str = "coinrun"

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: ProcgenConfig | None = None,
    ) -> None:
        super().__init__(context)
        if config is None:
            config = ProcgenConfig(env_name=self._env_name)
        self._config = config
        self._env: Any = None
        self._step_counter = 0
        self._episode_return = 0.0
        self._last_observation: np.ndarray | None = None
        self._last_info: dict[str, Any] = {}  # Store info for high-res rgb
        self._render_warning_emitted = False

    @property
    def id(self) -> str:
        return f"procgen:procgen-{self._env_name}-v0"

    def load(self) -> None:
        """Initialize the Procgen environment."""
        _ensure_procgen()

        try:
            # Use gym.make with Procgen's registration
            # Procgen registers as "procgen:procgen-{name}-v0"
            env_id = f"procgen:procgen-{self._config.env_name}-v0"
            # Pass render_mode='rgb_array' to get high-res 512x512 in info["rgb"]
            # while observations remain 64x64 for the agent
            old_gym_env = gym.make(
                env_id,
                num_levels=self._config.num_levels,
                start_level=self._config.start_level,
                distribution_mode=self._config.distribution_mode,
                use_backgrounds=self._config.use_backgrounds,
                center_agent=self._config.center_agent,
                use_sequential_levels=self._config.use_sequential_levels,
                paint_vel_info=self._config.paint_vel_info,
                render_mode="rgb_array",  # Enables 512x512 info["rgb"]
            )

            # Wrap with shimmy for gymnasium compatibility
            # GymV21CompatibilityV0 converts old gym API (step returns 4 values, reset returns obs only)
            # to gymnasium API (step returns 5 values, reset returns (obs, info))
            if _SHIMMY_AVAILABLE and GymV21CompatibilityV0 is not None:
                self._env = GymV21CompatibilityV0(env=old_gym_env, render_mode="rgb_array")
                _LOGGER.info(
                    "Procgen environment loaded with shimmy wrapper: %s (levels=%d, mode=%s)",
                    env_id,
                    self._config.num_levels,
                    self._config.distribution_mode,
                )
            else:
                self._env = old_gym_env
                _LOGGER.warning(
                    "Shimmy not available, using raw gym env (may cause warnings): %s",
                    env_id,
                )
        except Exception as exc:
            _LOGGER.error("Failed to load Procgen environment: %s", exc)
            raise

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> AdapterStep[np.ndarray]:
        """Start a new episode."""
        if self._env is None:
            self.load()

        env = self._env
        assert env is not None

        # With shimmy wrapper, reset() returns (obs, info) tuple (gymnasium API)
        # Without shimmy, handle both old and new API formats
        reset_kwargs: dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        if options is not None:
            reset_kwargs["options"] = options

        try:
            result = env.reset(**reset_kwargs) if reset_kwargs else env.reset()
        except TypeError:
            # Old gym API doesn't support seed/options kwargs
            result = env.reset()

        if isinstance(result, tuple):
            obs = result[0]
            info = result[1] if len(result) > 1 else {}
        else:
            obs = result
            info = {}

        self._last_observation = np.asarray(obs, dtype=np.uint8)
        self._last_info = dict(info) if info else {}
        self._step_counter = 0
        self._episode_return = 0.0

        return AdapterStep(
            observation=self._last_observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=self._last_info,
            render_payload=self.render(),
            state=self._build_step_state(self._last_observation, self._last_info),
        )

    def step(self, action: int) -> AdapterStep[np.ndarray]:
        """Execute action and return result.

        Args:
            action: Integer action from Procgen action space (0-14).

        Returns:
            AdapterStep with observation, reward, and render payload.
        """
        if self._env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        # Procgen uses old gym API: (obs, reward, done, info)
        # or sometimes new API: (obs, reward, terminated, truncated, info)
        result = self._env.step(action)

        if len(result) == 4:
            # Old gym API
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            # New gymnasium API
            obs, reward, terminated, truncated, info = result

        self._last_observation = np.asarray(obs, dtype=np.uint8)
        self._last_info = dict(info) if info else {}
        self._step_counter += 1
        self._episode_return += reward

        if self._step_counter % _PROCGEN_STEP_LOG_FREQUENCY == 0:
            _LOGGER.debug(
                "Procgen step %d: action=%s, reward=%.2f, return=%.2f",
                self._step_counter,
                PROCGEN_ACTIONS[action] if 0 <= action < len(PROCGEN_ACTIONS) else action,
                reward,
                self._episode_return,
            )

        return AdapterStep(
            observation=self._last_observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=self._last_info,
            render_payload=self.render(),
            state=self._build_step_state(self._last_observation, self._last_info),
        )

    def render(self) -> dict[str, Any]:
        """Return render payload for the UI.

        When render_mode='rgb_array' is set, Procgen provides:
        - Observation: 64x64 (for agent training)
        - info["rgb"]: 512x512 (for human viewing)

        We use info["rgb"] and optionally scale up further based on render_scale.
        """
        from PIL import Image

        # Prefer high-res info["rgb"] (512x512) over observation (64x64)
        if "rgb" in self._last_info:
            frame = np.asarray(self._last_info["rgb"], dtype=np.uint8)
        elif self._last_observation is not None:
            # Fallback to observation if info["rgb"] not available
            frame = self._last_observation
        else:
            # Return placeholder black frame
            frame = np.zeros((512, 512, 3), dtype=np.uint8)

        # Apply additional scaling if render_scale > 1
        # render_scale=4 with 512x512 base = 2048x2048 output
        # NOTE: Use NEAREST for fast real-time rendering during gameplay
        # LANCZOS is too slow and causes input lag
        scale = self._config.render_scale
        if scale > 1:
            img = Image.fromarray(frame)
            new_size = (frame.shape[1] * scale, frame.shape[0] * scale)
            # Use NEAREST for fast real-time rendering (no interpolation lag)
            img = img.resize(new_size, Image.Resampling.NEAREST)
            frame = np.array(img)

        return {
            "mode": RenderMode.RGB_ARRAY.value,
            "rgb": frame,
            "game_id": self.id,
            "env_name": self._config.env_name,
        }

    def _build_step_state(self, obs: np.ndarray, info: Mapping[str, Any]) -> StepState:
        """Build machine-readable state snapshot."""
        metrics = {
            "step": self._step_counter,
            "episode_return": float(self._episode_return),
        }

        # Procgen info may contain level_complete, prev_level_seed, etc.
        environment = {
            "env_name": self._config.env_name,
            "distribution_mode": self._config.distribution_mode,
            "num_levels": self._config.num_levels,
            "start_level": self._config.start_level,
        }

        # Extract any level completion info
        if "prev_level_complete" in info:
            metrics["prev_level_complete"] = int(info["prev_level_complete"])
        if "prev_level_seed" in info:
            metrics["prev_level_seed"] = int(info["prev_level_seed"])

        return StepState(
            metrics=metrics,
            environment=environment,
            raw=dict(info),
        )

    def close(self) -> None:
        """Clean up resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass  # Ignore errors on close
            self._env = None
        self._last_observation = None

    @property
    def action_space(self) -> spaces.Space:
        """Return the action space."""
        if self._env is not None:
            return self._env.action_space
        # Procgen default action space: 15 actions
        return spaces.Discrete(15)

    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space."""
        if self._env is not None:
            return self._env.observation_space
        # Procgen default observation space: 64x64 RGB
        return spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    def get_action_name(self, action: int) -> str:
        """Get the human-readable name for an action."""
        if 0 <= action < len(PROCGEN_ACTIONS):
            return PROCGEN_ACTIONS[action]
        return f"unknown_{action}"


# =============================================================================
# Concrete Procgen Environment Adapters (16 environments)
# =============================================================================

class ProcgenBigfishAdapter(ProcgenAdapter):
    """BigFish: Eat smaller fish, grow bigger."""
    _env_name = "bigfish"


class ProcgenBossfightAdapter(ProcgenAdapter):
    """BossFight: Defeat the boss starship."""
    _env_name = "bossfight"


class ProcgenCaveflyerAdapter(ProcgenAdapter):
    """CaveFlyer: Navigate caves, destroy targets."""
    _env_name = "caveflyer"


class ProcgenChaserAdapter(ProcgenAdapter):
    """Chaser: MsPacman-inspired maze game."""
    _env_name = "chaser"


class ProcgenClimberAdapter(ProcgenAdapter):
    """Climber: Platformer, collect stars."""
    _env_name = "climber"


class ProcgenCoinrunAdapter(ProcgenAdapter):
    """CoinRun: Platformer, reach the coin."""
    _env_name = "coinrun"


class ProcgenDodgeballAdapter(ProcgenAdapter):
    """Dodgeball: Avoid and throw balls."""
    _env_name = "dodgeball"


class ProcgenFruitbotAdapter(ProcgenAdapter):
    """FruitBot: Collect fruit, avoid non-fruit."""
    _env_name = "fruitbot"


class ProcgenHeistAdapter(ProcgenAdapter):
    """Heist: Collect keys, steal the gem."""
    _env_name = "heist"


class ProcgenJumperAdapter(ProcgenAdapter):
    """Jumper: Open-world platformer."""
    _env_name = "jumper"


class ProcgenLeaperAdapter(ProcgenAdapter):
    """Leaper: Frogger-inspired crossing game."""
    _env_name = "leaper"


class ProcgenMazeAdapter(ProcgenAdapter):
    """Maze: Navigate maze to find cheese."""
    _env_name = "maze"


class ProcgenMinerAdapter(ProcgenAdapter):
    """Miner: BoulderDash-inspired digging game."""
    _env_name = "miner"


class ProcgenNinjaAdapter(ProcgenAdapter):
    """Ninja: Platformer with bombs."""
    _env_name = "ninja"


class ProcgenPlunderAdapter(ProcgenAdapter):
    """Plunder: Destroy pirate ships."""
    _env_name = "plunder"


class ProcgenStarpilotAdapter(ProcgenAdapter):
    """StarPilot: Side-scrolling shooter."""
    _env_name = "starpilot"


# =============================================================================
# Adapter Registry
# =============================================================================

PROCGEN_ADAPTERS: dict[GameId, type[ProcgenAdapter]] = {
    GameId.PROCGEN_BIGFISH: ProcgenBigfishAdapter,
    GameId.PROCGEN_BOSSFIGHT: ProcgenBossfightAdapter,
    GameId.PROCGEN_CAVEFLYER: ProcgenCaveflyerAdapter,
    GameId.PROCGEN_CHASER: ProcgenChaserAdapter,
    GameId.PROCGEN_CLIMBER: ProcgenClimberAdapter,
    GameId.PROCGEN_COINRUN: ProcgenCoinrunAdapter,
    GameId.PROCGEN_DODGEBALL: ProcgenDodgeballAdapter,
    GameId.PROCGEN_FRUITBOT: ProcgenFruitbotAdapter,
    GameId.PROCGEN_HEIST: ProcgenHeistAdapter,
    GameId.PROCGEN_JUMPER: ProcgenJumperAdapter,
    GameId.PROCGEN_LEAPER: ProcgenLeaperAdapter,
    GameId.PROCGEN_MAZE: ProcgenMazeAdapter,
    GameId.PROCGEN_MINER: ProcgenMinerAdapter,
    GameId.PROCGEN_NINJA: ProcgenNinjaAdapter,
    GameId.PROCGEN_PLUNDER: ProcgenPlunderAdapter,
    GameId.PROCGEN_STARPILOT: ProcgenStarpilotAdapter,
}


__all__ = [
    "ProcgenAdapter",
    "ProcgenBigfishAdapter",
    "ProcgenBossfightAdapter",
    "ProcgenCaveflyerAdapter",
    "ProcgenChaserAdapter",
    "ProcgenClimberAdapter",
    "ProcgenCoinrunAdapter",
    "ProcgenDodgeballAdapter",
    "ProcgenFruitbotAdapter",
    "ProcgenHeistAdapter",
    "ProcgenJumperAdapter",
    "ProcgenLeaperAdapter",
    "ProcgenMazeAdapter",
    "ProcgenMinerAdapter",
    "ProcgenNinjaAdapter",
    "ProcgenPlunderAdapter",
    "ProcgenStarpilotAdapter",
    "PROCGEN_ADAPTERS",
    "PROCGEN_ENV_NAMES",
    "PROCGEN_ACTIONS",
]
