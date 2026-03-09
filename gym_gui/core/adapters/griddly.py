"""Adapter bridging Griddly's gym API to MOSAIC's Gymnasium-based adapter interface.

Griddly is a high-performance grid world research platform with a C++ backend
and Vulkan GPU rendering capable of 30 000+ FPS (headless training).

Paper: Bamford et al. (2021). "Griddly: A Platform for AI Research in Games"
Repo:  https://github.com/Bam4d/Griddly
Docs:  https://griddly.readthedocs.io

Griddly ships a legacy ``gym`` (pre-0.26) interface.  This module wraps each
environment in :class:`_GriddlyGymnasiumWrapper` which adapts the 4-tuple step
return to the Gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``
and adds ``reset(seed=..., options=...)`` support.  No extra dependency beyond
``griddly`` itself is required.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

import gymnasium as gym
import gymnasium.spaces as gspaces
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
    LOG_GRIDDLY_ENV_CLOSED,
    LOG_GRIDDLY_ENV_CREATED,
    LOG_GRIDDLY_ENV_RESET,
    LOG_GRIDDLY_RENDER_ERROR,
    LOG_GRIDDLY_STEP_SUMMARY,
)

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gymnasium compatibility shim for Griddly's legacy gym API
# ---------------------------------------------------------------------------

class _GriddlyGymnasiumWrapper(gym.Env):
    """Wraps a Griddly gym environment to conform to the Gymnasium API.

    Griddly's built-in wrapper follows the old ``gym`` convention:
    * ``reset()`` returns ``obs`` (no info dict)
    * ``step()`` returns ``(obs, reward, done, info)`` (4-tuple, no truncated)

    This wrapper translates both to the Gymnasium 5-tuple step contract and
    the ``reset(seed=..., options=...)`` signature expected by MOSAIC.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(self, griddly_env: Any, render_mode: str = "rgb_array") -> None:
        super().__init__()
        self._env = griddly_env
        self.render_mode = render_mode
        self.observation_space: gym.Space[Any] = griddly_env.observation_space
        self.action_space: gym.Space[Any] = griddly_env.action_space

    # ------------------------------------------------------------------
    # Gymnasium protocol
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            self._env.seed(seed)
        obs = self._env.reset()
        return obs, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self._env.step(action)
        if len(result) == 5:
            # Already Gymnasium API (newer griddly build)
            obs, reward, terminated, truncated, info = result
            return obs, float(reward), bool(terminated), bool(truncated), (info or {})
        # Legacy 4-tuple: (obs, reward, done, info)
        obs, reward, done, info = result
        return obs, float(reward), bool(done), False, (info or {})

    def render(self) -> np.ndarray | None:
        try:
            # griddly may require explicit mode kwarg on older builds
            try:
                return self._env.render(mode="rgb_array")
            except TypeError:
                return self._env.render()
        except Exception:
            return None

    def close(self) -> None:
        self._env.close()


# ---------------------------------------------------------------------------
# Gymnasium compatibility shim for Griddly multi-agent environments
# ---------------------------------------------------------------------------

def _to_gymnasium_space(old_space: Any) -> gspaces.Space:
    """Convert an old gym space to a gymnasium space."""
    import gym.spaces as old_spaces
    if isinstance(old_space, old_spaces.Box):
        return gspaces.Box(
            low=old_space.low,
            high=old_space.high,
            shape=old_space.shape,
            dtype=old_space.dtype,
        )
    if isinstance(old_space, old_spaces.Discrete):
        return gspaces.Discrete(old_space.n)
    if isinstance(old_space, old_spaces.MultiDiscrete):
        return gspaces.MultiDiscrete(old_space.nvec)
    # Fallback: return as-is and hope for the best
    return old_space  # type: ignore[return-value]


class _GriddlyMultiAgentGymnasiumWrapper(gym.Env):
    """Gymnasium wrapper for multi-agent Griddly environments.

    Converts ``MultiAgentActionSpace`` / ``MultiAgentObservationSpace`` (both
    ``list`` subclasses, not ``gym.spaces.Space``) to proper
    ``gymnasium.spaces.Tuple`` so that MOSAIC's space machinery accepts them.

    Step / reset API is translated from old-gym 4-tuple to Gymnasium 5-tuple.
    Observations and rewards are returned as tuples (one entry per agent).
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(self, griddly_env: Any, render_mode: str = "rgb_array") -> None:
        super().__init__()
        self._env = griddly_env
        self.render_mode = render_mode
        self.player_count: int = griddly_env.player_count

        self.action_space: gspaces.Space[Any] = gspaces.Tuple(
            tuple(_to_gymnasium_space(s) for s in griddly_env.action_space)
        )
        self.observation_space: gspaces.Space[Any] = gspaces.Tuple(
            tuple(_to_gymnasium_space(s) for s in griddly_env.observation_space)
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if seed is not None:
            self._env.seed(seed)
        obs_list = self._env.reset()  # returns list of per-agent obs
        return tuple(obs_list), {}

    def step(
        self, action: Any
    ) -> tuple[tuple[Any, ...], float, bool, bool, dict[str, Any]]:
        actions = list(action)
        result = self._env.step(actions)
        if len(result) == 5:
            obs_list, rewards, terminated, truncated, info = result
        else:
            obs_list, rewards, done, info = result
            terminated = done
            truncated = False
        total_reward = float(sum(float(r) for r in rewards))
        return tuple(obs_list), total_reward, bool(terminated), bool(truncated), (info or {})

    def render(self) -> np.ndarray | None:
        try:
            try:
                return self._env.render(mode="rgb_array")
            except TypeError:
                return self._env.render()
        except Exception:
            return None

    def close(self) -> None:
        self._env.close()


# ---------------------------------------------------------------------------
# Lazy import + env factory
# ---------------------------------------------------------------------------

def _ensure_griddly() -> None:
    """Import griddly lazily to trigger gymnasium.register() side effects."""
    import griddly  # noqa: F401  (registers GDY-* envs)


def _make_griddly_env(env_id: str) -> gym.Env:
    """Create and return a Gymnasium-wrapped single-agent Griddly environment.

    Args:
        env_id: Registered Griddly environment ID (e.g. ``"GDY-Zelda-v0"``).

    Returns:
        A fully Gymnasium-compatible :class:`~gymnasium.Env`.

    Raises:
        RuntimeError: If griddly is not installed or the env cannot be created.
    """
    _ensure_griddly()
    try:
        import warnings

        import gym as old_gym  # griddly uses legacy gym

        # Bypass gym's passive_env_checker entirely: Griddly's MultiAgentActionSpace
        # is a list subclass (not gym.spaces.Space) so the checker raises rather
        # than warns.  We validate spaces ourselves after construction.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            raw_env = old_gym.make(env_id, disable_env_checker=True)

        return _GriddlyGymnasiumWrapper(raw_env)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create Griddly environment '{env_id}'. "
            "Ensure griddly is installed and Vulkan drivers are available. "
            f"Original error: {exc}"
        ) from exc


def _make_griddly_multi_agent_env(env_id: str) -> _GriddlyMultiAgentGymnasiumWrapper:
    """Create and return a Gymnasium-wrapped multi-agent Griddly environment.

    Instantiates ``GymWrapper`` directly (bypassing gym.make) so we control
    how the ``MultiAgentActionSpace`` and ``MultiAgentObservationSpace`` are
    exposed.

    Args:
        env_id: Registered Griddly environment ID (e.g. ``"GDY-Foragers-v0"``).

    Returns:
        A :class:`_GriddlyMultiAgentGymnasiumWrapper` exposing Gymnasium API.

    Raises:
        RuntimeError: If griddly is not installed or the env cannot be created.
    """
    _ensure_griddly()
    try:
        import warnings

        import gym as old_gym
        from griddly import GymWrapper

        spec = old_gym.envs.registry[env_id]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            raw_env = GymWrapper(**spec.kwargs, global_observations=False)

        return _GriddlyMultiAgentGymnasiumWrapper(raw_env)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create Griddly multi-agent environment '{env_id}'. "
            "Ensure griddly is installed and Vulkan drivers are available. "
            f"Original error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class GriddlyAdapter(EnvironmentAdapter[np.ndarray, int]):
    """Base adapter for single-agent Griddly environments.

    Subclasses set ``_gym_id`` and ``id`` at class level.  The adapter wraps
    Griddly's legacy gym interface in a thin Gymnasium compatibility shim so
    the rest of MOSAIC sees the standard Gymnasium API.
    """

    _gym_id: str = "GDY-Zelda-v0"

    id: str = "GDY-Zelda-v0"
    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: Any | None = None,
    ) -> None:
        super().__init__(context)

        from gym_gui.config.game_configs import GriddlyConfig

        if config is None:
            config = GriddlyConfig()
        if not isinstance(config, GriddlyConfig):
            config = GriddlyConfig()

        self._config = config
        self._env: gym.Env | None = None
        self._step_count: int = 0

    @property
    def stepping_paradigm(self) -> SteppingParadigm:  # type: ignore[override]
        return SteppingParadigm.SINGLE_AGENT

    @property
    def action_space(self) -> gym.Space[Any]:
        if self._env is None:
            from gymnasium.spaces import Discrete
            return Discrete(5)
        return self._env.action_space

    @property
    def observation_space(self) -> gym.Space[Any]:
        if self._env is None:
            from gymnasium.spaces import Box
            return Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        return self._env.observation_space

    def load(self) -> None:
        """Create the Griddly environment."""
        _LOGGER.info("Loading Griddly environment: %s", self._gym_id)
        self._env = _make_griddly_env(self._gym_id)
        self.log_constant(
            LOG_GRIDDLY_ENV_CREATED,
            extra={"env_id": self._gym_id},
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[np.ndarray]:
        if self._env is None:
            self.load()
        assert self._env is not None

        effective_seed = seed if seed is not None else self._config.seed
        obs, info = self._env.reset(seed=effective_seed, options=options)
        self._step_count = 0

        self.log_constant(
            LOG_GRIDDLY_ENV_RESET,
            extra={
                "env_id": self._gym_id,
                "seed": effective_seed if effective_seed is not None else "None",
            },
        )

        return AdapterStep(
            observation=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info if isinstance(info, Mapping) else {},
            render_payload=self._try_render(),
        )

    def step(self, action: int) -> AdapterStep[np.ndarray]:
        if self._env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1

        self.log_constant(
            LOG_GRIDDLY_STEP_SUMMARY,
            extra={
                "env_id": self._gym_id,
                "step": self._step_count,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            },
        )

        return AdapterStep(
            observation=obs,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info if isinstance(info, Mapping) else {},
            render_payload=self._try_render(),
        )

    def render(self) -> dict[str, Any] | None:
        return self._try_render()

    def _try_render(self) -> dict[str, Any] | None:
        if self._env is None:
            return None
        try:
            frame = self._env.render()
            if isinstance(frame, np.ndarray):
                return {
                    "mode": RenderMode.RGB_ARRAY.value,
                    "rgb": frame,
                    "game_id": self._gym_id,
                    "step": self._step_count,
                }
            return None
        except Exception as exc:
            self.log_constant(
                LOG_GRIDDLY_RENDER_ERROR,
                extra={"env_id": self._gym_id, "error": str(exc)},
            )
            return None

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
        self.log_constant(
            LOG_GRIDDLY_ENV_CLOSED,
            extra={"env_id": self._gym_id},
        )

    def build_step_state(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> StepState:
        return StepState(
            agents=[AgentSnapshot(name="agent_0")],
            metrics={"step": self._step_count},
        )


# ---------------------------------------------------------------------------
# Concrete adapters (one per GameId)
# ---------------------------------------------------------------------------

class GriddlyZeldaAdapter(GriddlyAdapter):
    """Zelda-like grid world: navigate rooms, collect keys, defeat enemies."""
    _gym_id = "GDY-Zelda-v0"
    id = "GDY-Zelda-v0"


class GriddlyZeldaSequentialAdapter(GriddlyAdapter):
    """Zelda variant: must pick up keys and open doors in strict order."""
    _gym_id = "GDY-Zelda-Sequential-v0"
    id = "GDY-Zelda-Sequential-v0"


class GriddlyPOZeldaAdapter(GriddlyAdapter):
    """Partially observable Zelda: same as Zelda but with limited field of view."""
    _gym_id = "GDY-Partially-Observable-Zelda-v0"
    id = "GDY-Partially-Observable-Zelda-v0"


class GriddlySokobanAdapter(GriddlyAdapter):
    """Classic Sokoban: push boxes into holes."""
    _gym_id = "GDY-Sokoban-v0"
    id = "GDY-Sokoban-v0"


class GriddlySokoban2Adapter(GriddlyAdapter):
    """Sokoban variant: push boxes onto marked spaces; boxes cannot be moved once placed."""
    _gym_id = "GDY-Sokoban---2-v0"
    id = "GDY-Sokoban---2-v0"


class GriddlyPOSokoban2Adapter(GriddlyAdapter):
    """Partially observable Sokoban 2."""
    _gym_id = "GDY-Partially-Observable-Sokoban---2-v0"
    id = "GDY-Partially-Observable-Sokoban---2-v0"


class GriddlyClustersAdapter(GriddlyAdapter):
    """Clusters: push coloured blocks against matching static coloured blocks."""
    _gym_id = "GDY-Clusters-v0"
    id = "GDY-Clusters-v0"


class GriddlyPOClustersAdapter(GriddlyAdapter):
    """Partially observable Clusters."""
    _gym_id = "GDY-Partially-Observable-Clusters-v0"
    id = "GDY-Partially-Observable-Clusters-v0"


class GriddlyBaitAdapter(GriddlyAdapter):
    """Bait: get the key, unlock the door, fill holes with blocks."""
    _gym_id = "GDY-Bait-v0"
    id = "GDY-Bait-v0"


class GriddlyBaitWithKeysAdapter(GriddlyAdapter):
    """Bait With Keys: same as Bait but the avatar visibly holds the key."""
    _gym_id = "GDY-Bait-With-Keys-v0"
    id = "GDY-Bait-With-Keys-v0"


class GriddlyPOBaitAdapter(GriddlyAdapter):
    """Partially observable Bait."""
    _gym_id = "GDY-Partially-Observable-Bait-v0"
    id = "GDY-Partially-Observable-Bait-v0"


class GriddlyZenPuzzleAdapter(GriddlyAdapter):
    """Zen Puzzle: set all tiles to the same colour without revisiting any tile."""
    _gym_id = "GDY-Zen-Puzzle-v0"
    id = "GDY-Zen-Puzzle-v0"


class GriddlyPOZenPuzzleAdapter(GriddlyAdapter):
    """Partially observable Zen Puzzle."""
    _gym_id = "GDY-Partially-Observable-Zen-Puzzle-v0"
    id = "GDY-Partially-Observable-Zen-Puzzle-v0"


class GriddlyLabyrinthAdapter(GriddlyAdapter):
    """Labyrinth: find your way out of the maze, avoid spikes."""
    _gym_id = "GDY-Labyrinth-v0"
    id = "GDY-Labyrinth-v0"


class GriddlyPOLabyrinthAdapter(GriddlyAdapter):
    """Partially observable Labyrinth."""
    _gym_id = "GDY-Partially-Observable-Labyrinth-v0"
    id = "GDY-Partially-Observable-Labyrinth-v0"


class GriddlyCookMePastaAdapter(GriddlyAdapter):
    """Cook Me Pasta: combine ingredients in the correct order to complete the meal."""
    _gym_id = "GDY-Cook-Me-Pasta-v0"
    id = "GDY-Cook-Me-Pasta-v0"


class GriddlyPOCookMePastaAdapter(GriddlyAdapter):
    """Partially observable Cook Me Pasta."""
    _gym_id = "GDY-Partially-Observable-Cook-Me-Pasta-v0"
    id = "GDY-Partially-Observable-Cook-Me-Pasta-v0"


class GriddlySpidersAdapter(GriddlyAdapter):
    """Spiders: gnome avoids ghosts to reach a gem (MiniGrid dynamic obstacles port)."""
    _gym_id = "GDY-Spiders-v0"
    id = "GDY-Spiders-v0"


class GriddlySpiderNestAdapter(GriddlyAdapter):
    """Spider Nest: same premise as Spiders with a different map layout."""
    _gym_id = "GDY-Spider-Nest-v0"
    id = "GDY-Spider-Nest-v0"


class GriddlyButterfliesAndSpidersAdapter(GriddlyAdapter):
    """Butterflies and Spiders: catch butterflies before spiders eat them."""
    _gym_id = "GDY-Butterflies-and-Spiders-v0"
    id = "GDY-Butterflies-and-Spiders-v0"


class GriddlyRandomButterfliesAdapter(GriddlyAdapter):
    """Random Butterflies: partial observability, randomly spawning butterflies and spiders."""
    _gym_id = "GDY-Random-butterflies-v0"
    id = "GDY-Random-butterflies-v0"


class GriddlyEyeballAdapter(GriddlyAdapter):
    """Eyeball: giant eye navigates 4-rooms looking for eyedrops (MiniGrid 4-rooms port)."""
    _gym_id = "GDY-Eyeball-v0"
    id = "GDY-Eyeball-v0"


class GriddlyDrunkDwarfAdapter(GriddlyAdapter):
    """Drunk Dwarf: dwarf finds keys to reach a coffin-bed (MiniGrid port)."""
    _gym_id = "GDY-Drunk-Dwarf-v0"
    id = "GDY-Drunk-Dwarf-v0"


class GriddlyDoggoAdapter(GriddlyAdapter):
    """Doggo: dog fetches a stick (MiniGrid empty room port)."""
    _gym_id = "GDY-Doggo-v0"
    id = "GDY-Doggo-v0"


# ---------------------------------------------------------------------------
# Multi-agent adapters — override stepping_paradigm to SIMULTANEOUS
# ---------------------------------------------------------------------------

class _GriddlyMultiAgentAdapter(GriddlyAdapter):
    """Base for Griddly multi-agent environments.

    Uses :func:`_make_griddly_multi_agent_env` which instantiates
    ``GymWrapper`` directly and wraps ``MultiAgentActionSpace`` /
    ``MultiAgentObservationSpace`` as ``gymnasium.spaces.Tuple``.
    """

    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.MULTI_AGENT_COOP,
    )

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: Any | None = None,
    ) -> None:
        super().__init__(context, config=config)
        self._multi_env: _GriddlyMultiAgentGymnasiumWrapper | None = None
        self._player_count: int = 0

    @property
    def stepping_paradigm(self) -> SteppingParadigm:  # type: ignore[override]
        return SteppingParadigm.SIMULTANEOUS

    @property
    def action_space(self) -> gspaces.Space[Any]:
        if self._multi_env is None:
            return gspaces.Tuple((gspaces.Discrete(5),))
        return self._multi_env.action_space

    @property
    def observation_space(self) -> gspaces.Space[Any]:
        if self._multi_env is None:
            return gspaces.Tuple((gspaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8),))
        return self._multi_env.observation_space

    def load(self) -> None:
        """Create the multi-agent Griddly environment."""
        _LOGGER.info("Loading multi-agent Griddly environment: %s", self._gym_id)
        self._multi_env = _make_griddly_multi_agent_env(self._gym_id)
        self._env = self._multi_env  # keep base class reference consistent
        self._player_count = self._multi_env.player_count
        self.log_constant(
            LOG_GRIDDLY_ENV_CREATED,
            extra={"env_id": self._gym_id, "player_count": self._player_count},
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[List[np.ndarray]]:
        if self._multi_env is None:
            self.load()
        assert self._multi_env is not None

        effective_seed = seed if seed is not None else self._config.seed
        obs_tuple, info = self._multi_env.reset(seed=effective_seed, options=options)
        self._step_count = 0

        self.log_constant(
            LOG_GRIDDLY_ENV_RESET,
            extra={
                "env_id": self._gym_id,
                "seed": effective_seed if effective_seed is not None else "None",
                "player_count": self._player_count,
            },
        )

        obs_list = list(obs_tuple)
        agent_names = [f"agent_{i}" for i in range(self._player_count)]
        reset_info: Dict[str, Any] = {
            "num_agents": self._player_count,
            "agents": agent_names,
            "env_id": self._gym_id,
        }
        reset_info.update(info if isinstance(info, Mapping) else {})
        return AdapterStep(
            observation=obs_list,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=reset_info,
            render_payload=self._try_render(),
        )

    def _is_rts_env(self) -> bool:
        """Return True if the underlying Griddly env uses RTS-style unit selection.

        RTS environments have ``has_avatar=False`` and require a 4-component
        action ``[src_x, src_y, action_type, direction]`` instead of the simple
        ``[action_type, direction]`` used by avatar-based environments.
        """
        return (
            self._multi_env is not None
            and not self._multi_env._env.has_avatar
            and len(self._multi_env._env.action_space_parts) >= 4
        )

    def _rts_expand_action(self, player_idx: int, direction: int) -> list:
        """Expand a simple direction integer into a full RTS action vector.

        Looks up the player's available units via the Griddly game API, selects
        the first unit that can 'move', and returns the full action:
        ``[src_x, src_y, move_action_type, direction]``.

        Args:
            player_idx: Zero-based player index (converted to 1-based internally).
            direction: Direction integer (0=NOOP, 1-4=LEFT/UP/RIGHT/DOWN).

        Returns:
            A 4-element list ``[x, y, action_type, direction]`` for the Griddly game.
        """
        noop = [0, 0, 0, 0]
        if direction == 0:
            return noop

        raw_env = self._multi_env._env  # type: ignore[union-attr]
        action_names: list = raw_env.action_names
        move_type = action_names.index("move") if "move" in action_names else 0

        # get_available_actions() uses 1-based player IDs
        available: dict = raw_env.game.get_available_actions(player_idx + 1)
        for (x, y), acts in available.items():
            if "move" in acts:
                return [x, y, move_type, direction]

        return noop

    def step(self, action: Any) -> AdapterStep[List[np.ndarray]]:
        if self._multi_env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        # Accept a single action (broadcast) or a list/tuple of per-agent actions
        if not isinstance(action, (list, tuple)):
            action = [action] * self._player_count

        # For RTS games (has_avatar=False) the env expects a 4-component action
        # [src_x, src_y, action_type, direction] per player.  The human-input
        # resolver only provides a simple direction integer, so we expand it here.
        if self._is_rts_env():
            action = [
                self._rts_expand_action(i, int(a)) if isinstance(a, (int, np.integer)) else a
                for i, a in enumerate(action)
            ]

        obs_tuple, reward, terminated, truncated, info = self._multi_env.step(action)
        self._step_count += 1

        self.log_constant(
            LOG_GRIDDLY_STEP_SUMMARY,
            extra={
                "env_id": self._gym_id,
                "step": self._step_count,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            },
        )

        obs_list = list(obs_tuple)
        return AdapterStep(
            observation=obs_list,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info if isinstance(info, Mapping) else {},
            render_payload=self._try_render(),
        )

    def build_step_state(
        self,
        observation: List[np.ndarray],
        info: Mapping[str, Any],
    ) -> StepState:
        agents = [
            AgentSnapshot(name=f"agent_{i}")
            for i in range(self._player_count or len(observation))
        ]
        return StepState(
            agents=agents,
            metrics={"step": self._step_count},
        )

    def close(self) -> None:
        if self._multi_env is not None:
            try:
                self._multi_env.close()
            except Exception:
                pass
            self._multi_env = None
            self._env = None
        self.log_constant(
            LOG_GRIDDLY_ENV_CLOSED,
            extra={"env_id": self._gym_id},
        )


class GriddlyRTSAdapter(_GriddlyMultiAgentAdapter):
    """GriddlyRTS: real-time strategy with aliens."""
    _gym_id = "GDY-GriddlyRTS-v0"
    id = "GDY-GriddlyRTS-v0"


class GriddlyPushManiaAdapter(_GriddlyMultiAgentAdapter):
    """Push Mania (Stratega port): push all opponents' pieces into holes."""
    _gym_id = "GDY-Push-Mania-v0"
    id = "GDY-Push-Mania-v0"


class GriddlyKillTheKingAdapter(_GriddlyMultiAgentAdapter):
    """Kill The King (Stratega port): protect your king, kill the opponent's."""
    _gym_id = "GDY-Kill-The-King-v0"
    id = "GDY-Kill-The-King-v0"


class GriddlyHealOrDieAdapter(_GriddlyMultiAgentAdapter):
    """Heal Or Die (Stratega port): healers + fighters, units lose health every turn."""
    _gym_id = "GDY-Heal-Or-Die-v0"
    id = "GDY-Heal-Or-Die-v0"


class GriddlyRobotTag4v4Adapter(_GriddlyMultiAgentAdapter):
    """Robot Tag 4v4: robots tag opponents 3 times to eliminate them."""
    _gym_id = "GDY-Robot-Tag-4v4-v0"
    id = "GDY-Robot-Tag-4v4-v0"


class GriddlyRobotTag8v8Adapter(_GriddlyMultiAgentAdapter):
    """Robot Tag 8v8: larger Robot Tag arena."""
    _gym_id = "GDY-Robot-Tag-8v8-v0"
    id = "GDY-Robot-Tag-8v8-v0"


class GriddlyRobotTag12v12Adapter(_GriddlyMultiAgentAdapter):
    """Robot Tag 12v12: large-scale Robot Tag."""
    _gym_id = "GDY-Robot-Tag-12v12-v0"
    id = "GDY-Robot-Tag-12v12-v0"


class GriddlyForagersAdapter(_GriddlyMultiAgentAdapter):
    """Foragers: agents cooperatively collect coloured potions."""
    _gym_id = "GDY-Foragers-v0"
    id = "GDY-Foragers-v0"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRIDDLY_ADAPTERS: Dict[GameId, type[GriddlyAdapter]] = {
    # Single-agent
    GameId.GRIDDLY_ZELDA: GriddlyZeldaAdapter,
    GameId.GRIDDLY_ZELDA_SEQUENTIAL: GriddlyZeldaSequentialAdapter,
    GameId.GRIDDLY_PO_ZELDA: GriddlyPOZeldaAdapter,
    GameId.GRIDDLY_SOKOBAN: GriddlySokobanAdapter,
    GameId.GRIDDLY_SOKOBAN_2: GriddlySokoban2Adapter,
    GameId.GRIDDLY_PO_SOKOBAN_2: GriddlyPOSokoban2Adapter,
    GameId.GRIDDLY_CLUSTERS: GriddlyClustersAdapter,
    GameId.GRIDDLY_PO_CLUSTERS: GriddlyPOClustersAdapter,
    GameId.GRIDDLY_BAIT: GriddlyBaitAdapter,
    GameId.GRIDDLY_BAIT_WITH_KEYS: GriddlyBaitWithKeysAdapter,
    GameId.GRIDDLY_PO_BAIT: GriddlyPOBaitAdapter,
    GameId.GRIDDLY_ZEN_PUZZLE: GriddlyZenPuzzleAdapter,
    GameId.GRIDDLY_PO_ZEN_PUZZLE: GriddlyPOZenPuzzleAdapter,
    GameId.GRIDDLY_LABYRINTH: GriddlyLabyrinthAdapter,
    GameId.GRIDDLY_PO_LABYRINTH: GriddlyPOLabyrinthAdapter,
    GameId.GRIDDLY_COOK_ME_PASTA: GriddlyCookMePastaAdapter,
    GameId.GRIDDLY_PO_COOK_ME_PASTA: GriddlyPOCookMePastaAdapter,
    GameId.GRIDDLY_SPIDERS: GriddlySpidersAdapter,
    GameId.GRIDDLY_SPIDER_NEST: GriddlySpiderNestAdapter,
    GameId.GRIDDLY_BUTTERFLIES_AND_SPIDERS: GriddlyButterfliesAndSpidersAdapter,
    GameId.GRIDDLY_RANDOM_BUTTERFLIES: GriddlyRandomButterfliesAdapter,
    GameId.GRIDDLY_EYEBALL: GriddlyEyeballAdapter,
    GameId.GRIDDLY_DRUNK_DWARF: GriddlyDrunkDwarfAdapter,
    GameId.GRIDDLY_DOGGO: GriddlyDoggoAdapter,
    # Multi-agent (RTS)
    GameId.GRIDDLY_RTS: GriddlyRTSAdapter,
    GameId.GRIDDLY_PUSH_MANIA: GriddlyPushManiaAdapter,
    GameId.GRIDDLY_KILL_THE_KING: GriddlyKillTheKingAdapter,
    GameId.GRIDDLY_HEAL_OR_DIE: GriddlyHealOrDieAdapter,
    # Multi-agent (cooperative / competitive)
    GameId.GRIDDLY_ROBOT_TAG_4V4: GriddlyRobotTag4v4Adapter,
    GameId.GRIDDLY_ROBOT_TAG_8V8: GriddlyRobotTag8v8Adapter,
    GameId.GRIDDLY_ROBOT_TAG_12V12: GriddlyRobotTag12v12Adapter,
    GameId.GRIDDLY_FORAGERS: GriddlyForagersAdapter,
}

ALL_GRIDDLY_GAME_IDS: tuple[GameId, ...] = tuple(GRIDDLY_ADAPTERS.keys())

__all__ = [
    "GriddlyAdapter",
    "GRIDDLY_ADAPTERS",
    "ALL_GRIDDLY_GAME_IDS",
]
