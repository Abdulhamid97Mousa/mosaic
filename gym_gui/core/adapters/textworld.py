"""TextWorld environment adapters for text-based game environments.

TextWorld is a Microsoft Research sandbox for training RL agents on text-based
games. It generates and simulates text-based adventure games for research in
language understanding and sequential decision making.

Paper: Cote et al. (2018). TextWorld: A Learning Environment for Text-based Games.
Repository: https://github.com/microsoft/TextWorld
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np

from gym_gui.config.game_configs import TextWorldConfig
from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    EnvironmentAdapter,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
    LOG_ENV_TEXTWORLD_BOOT,
    LOG_ENV_TEXTWORLD_STEP,
    LOG_ENV_TEXTWORLD_ERROR,
    LOG_ENV_TEXTWORLD_GAME_GENERATED,
    LOG_ENV_TEXTWORLD_COMMAND,
)

_LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - import guard exercised in integration tests
    import textworld
    import textworld.gym
    from textworld import EnvInfos
    from textworld.generator import compile_game
    from textworld.challenges import CHALLENGES

    _HAS_TEXTWORLD = True
except ImportError:  # pragma: no cover - handled gracefully at runtime
    textworld = None  # type: ignore[assignment]
    EnvInfos = None  # type: ignore[assignment,misc]
    compile_game = None  # type: ignore[assignment,misc]
    CHALLENGES = {}  # type: ignore[assignment]
    _HAS_TEXTWORLD = False

_TEXTWORLD_STEP_LOG_FREQUENCY = 50


# TextWorld challenge types
TEXTWORLD_CHALLENGES = [
    "tw-simple",
    "tw-coin_collector",
    "tw-treasure_hunter",
    "tw-cooking",
]

# Default admissible commands for basic text adventure actions
DEFAULT_COMMANDS = [
    "look",
    "inventory",
    "go north",
    "go south",
    "go east",
    "go west",
    "examine",
    "take",
    "drop",
    "open",
    "close",
]


@dataclass(slots=True)
class _TextWorldMetrics:
    """Container describing TextWorld-specific telemetry traits."""

    score: int = 0
    max_score: int = 0
    moves: int = 0
    won: bool = False
    lost: bool = False
    location: str | None = None
    inventory: str | None = None
    objective: str | None = None
    admissible_commands: list[str] | None = None


class TextWorldAdapter(EnvironmentAdapter[str, str]):
    """Adapter for TextWorld text-based game environments.

    TextWorld provides text-based observations and uses string commands as actions.
    Games can be procedurally generated with configurable difficulty and objectives.

    Unlike most RL environments, TextWorld:
    - Uses text observations (strings) instead of images/arrays
    - Uses text commands (strings) as actions instead of discrete integers
    - Supports procedural game generation
    """

    default_render_mode = RenderMode.ANSI
    supported_render_modes = (RenderMode.ANSI,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )

    DEFAULT_ENV_ID = GameId.TEXTWORLD_SIMPLE.value

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: TextWorldConfig | None = None,
    ) -> None:
        super().__init__(context)
        if config is None:
            config = TextWorldConfig(env_id=self.DEFAULT_ENV_ID)
        self._config = config
        self._env_id = config.env_id or self.DEFAULT_ENV_ID
        self._step_counter = 0
        self._gamefile: str | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._last_observation: str = ""
        self._last_infos: dict[str, Any] = {}
        self._admissible_commands: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def render(self) -> dict[str, Any]:
        """Render the current game state as ANSI text."""
        # Build a text representation of the current state
        text_parts = []

        if self._last_observation:
            text_parts.append(self._last_observation)

        if self._last_infos.get("description"):
            text_parts.append(f"\n[Location]\n{self._last_infos['description']}")

        if self._last_infos.get("inventory"):
            text_parts.append(f"\n[Inventory]\n{self._last_infos['inventory']}")

        if self._admissible_commands:
            text_parts.append(f"\n[Available Commands]\n{', '.join(self._admissible_commands[:10])}")

        ansi_text = "\n".join(text_parts) if text_parts else "No observation available"

        return {
            "mode": RenderMode.ANSI.value,
            "ansi": ansi_text,
            "game_id": self._env_id,
            "score": self._last_infos.get("score", 0),
            "moves": self._last_infos.get("moves", 0),
        }

    @property
    def id(self) -> str:  # type: ignore[override]
        return self._env_id

    def gym_kwargs(self) -> dict[str, Any]:
        kwargs = super().gym_kwargs()
        kwargs.update(self._config.to_gym_kwargs())
        return kwargs

    def load(self) -> None:
        """Load or generate a TextWorld game and create the environment."""
        if not _HAS_TEXTWORLD or textworld is None:
            raise RuntimeError(
                "TextWorld package not installed. Install with: pip install textworld"
            )

        try:
            # Create a temporary directory for generated games
            self._temp_dir = tempfile.TemporaryDirectory(prefix="textworld_")

            # Generate or load game based on config
            self._gamefile = self._generate_or_load_game()

            # Set up request_infos to get useful information
            request_infos = EnvInfos(  # type: ignore[misc]
                description=True,
                inventory=True,
                location=True,
                admissible_commands=True,
                won=True,
                lost=True,
                score=True,
                max_score=True,
                moves=True,
                objective=True,
                intermediate_reward=self._config.intermediate_reward,
            )

            # Register the game with textworld.gym
            env_id = textworld.gym.register_game(
                self._gamefile,
                request_infos=request_infos,
                max_episode_steps=self._config.max_episode_steps,
                name=f"gui-{os.path.basename(self._gamefile).replace('.ulx', '')}",
            )

            # Create the environment
            self._env = textworld.gym.make(env_id)

            _LOGGER.info(
                "TextWorld environment loaded",
                extra={
                    "env_id": self._env_id,
                    "gamefile": self._gamefile,
                    "max_steps": self._config.max_episode_steps,
                },
            )

        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.error(
                f"TextWorld load error: {exc}",
                exc_info=exc,
                extra={
                    "env_id": self._env_id,
                    "stage": "load",
                },
            )
            raise

    def _generate_or_load_game(self) -> str:
        """Generate a new game or load an existing gamefile."""
        if self._config.gamefile and os.path.exists(self._config.gamefile):
            return self._config.gamefile

        # Generate a game based on challenge type
        challenge_name = self._config.challenge_type
        level = self._config.level

        if challenge_name in CHALLENGES:
            # Use built-in challenge
            challenge = CHALLENGES[challenge_name]
            game = challenge.make({"level": level}, textworld.GameOptions())  # type: ignore[union-attr]
        else:
            # Generate a simple custom game
            options = textworld.GameOptions()  # type: ignore[union-attr]
            options.nb_rooms = self._config.nb_rooms
            options.nb_objects = self._config.nb_objects
            options.quest_length = self._config.quest_length
            if self._config.seed is not None:
                options.seeds = self._config.seed

            game = textworld.generator.make_game(options)  # type: ignore[union-attr]

        # Compile the game to a playable format
        assert self._temp_dir is not None
        gamefile = os.path.join(self._temp_dir.name, f"tw_game_{self._config.seed or 0}.ulx")
        compile_game(game, gamefile)  # type: ignore[misc]

        _LOGGER.info(
            "TextWorld game generated",
            extra={
                "gamefile": gamefile,
                "challenge": challenge_name,
                "level": level,
                "nb_rooms": self._config.nb_rooms,
            },
        )

        return gamefile

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[str]:
        """Reset the TextWorld environment."""
        env = self._require_env()

        # TextWorld reset returns (observation, infos)
        observation, infos = env.reset()

        self._last_observation = observation
        self._last_infos = dict(infos)
        self._admissible_commands = infos.get("admissible_commands", DEFAULT_COMMANDS.copy())
        self._step_counter = 0

        _LOGGER.debug(
            "TextWorld reset",
            extra={
                "env_id": self._env_id,
                "seed": seed,
                "objective": infos.get("objective", ""),
            },
        )

        return self._package_step(observation, 0.0, False, False, infos)

    def step(self, action: str) -> AdapterStep[str]:
        """Execute a text command in the TextWorld environment.

        Args:
            action: Text command to execute (e.g., "go north", "take key")

        Returns:
            AdapterStep containing the new observation, reward, and game state.
        """
        env = self._require_env()

        # TextWorld step returns (observation, score, done, infos)
        # Note: score is cumulative, not per-step reward
        # TextWorld uses old Gym API (4 returns), not new Gymnasium API (5 returns)
        result = cast(tuple[str, int, bool, dict[str, Any]], env.step(action))
        observation = result[0]
        score = result[1]
        done = result[2]
        infos = result[3]

        self._last_observation = observation
        self._last_infos = dict(infos)
        self._admissible_commands = infos.get("admissible_commands", DEFAULT_COMMANDS.copy())

        # Calculate step reward from score change
        previous_score = self._last_infos.get("_previous_score", 0)
        step_reward = float(score - previous_score) * self._config.reward_multiplier
        self._last_infos["_previous_score"] = score

        # Add intermediate reward if available
        if self._config.intermediate_reward and "intermediate_reward" in infos:
            step_reward += float(infos["intermediate_reward"]) * self._config.reward_multiplier

        # Determine termination vs truncation
        won = infos.get("won", False)
        lost = infos.get("lost", False)
        terminated = won or lost
        truncated = done and not terminated

        self._step_counter += 1

        if self._step_counter % _TEXTWORLD_STEP_LOG_FREQUENCY == 1:
            _LOGGER.debug(
                "TextWorld step",
                extra={
                    "env_id": self._env_id,
                    "step": self._step_counter,
                    "action": action,
                    "reward": step_reward,
                    "score": score,
                    "terminated": terminated,
                    "truncated": truncated,
                },
            )

        return self._package_step(observation, step_reward, terminated, truncated, infos)

    def close(self) -> None:
        """Close the TextWorld environment and clean up resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None

        _LOGGER.debug("TextWorld environment closed", extra={"env_id": self.id})

    # ------------------------------------------------------------------
    # TextWorld-specific helpers
    # ------------------------------------------------------------------

    def get_admissible_commands(self) -> list[str]:
        """Get the list of currently valid commands."""
        return self._admissible_commands.copy()

    def get_objective(self) -> str:
        """Get the current game objective."""
        return self._last_infos.get("objective", "")

    def get_score(self) -> tuple[int, int]:
        """Get current score and max possible score."""
        return (
            self._last_infos.get("score", 0),
            self._last_infos.get("max_score", 0),
        )

    # ------------------------------------------------------------------
    # Action space helpers for human input
    # ------------------------------------------------------------------

    @property
    def action_space_size(self) -> int:
        """Return the number of currently admissible commands."""
        return len(self._admissible_commands)

    def action_index_to_command(self, index: int) -> str:
        """Convert an action index to a text command."""
        if 0 <= index < len(self._admissible_commands):
            return self._admissible_commands[index]
        return "look"  # Default safe action

    def command_to_action_index(self, command: str) -> int:
        """Convert a text command to an action index."""
        try:
            return self._admissible_commands.index(command)
        except ValueError:
            return 0  # Default to first command


# Convenience aliases for specific challenge types
class TextWorldSimpleAdapter(TextWorldAdapter):
    """Adapter for simple TextWorld games."""
    DEFAULT_ENV_ID = GameId.TEXTWORLD_SIMPLE.value


class TextWorldCoinCollectorAdapter(TextWorldAdapter):
    """Adapter for TextWorld Coin Collector challenge."""
    DEFAULT_ENV_ID = GameId.TEXTWORLD_COIN_COLLECTOR.value


class TextWorldTreasureHunterAdapter(TextWorldAdapter):
    """Adapter for TextWorld Treasure Hunter challenge."""
    DEFAULT_ENV_ID = GameId.TEXTWORLD_TREASURE_HUNTER.value


class TextWorldCookingAdapter(TextWorldAdapter):
    """Adapter for TextWorld Cooking challenge."""
    DEFAULT_ENV_ID = GameId.TEXTWORLD_COOKING.value


# Adapter registry for TextWorld environments
TEXTWORLD_ADAPTERS: dict[GameId, type[TextWorldAdapter]] = {
    GameId.TEXTWORLD_SIMPLE: TextWorldSimpleAdapter,
    GameId.TEXTWORLD_COIN_COLLECTOR: TextWorldCoinCollectorAdapter,
    GameId.TEXTWORLD_TREASURE_HUNTER: TextWorldTreasureHunterAdapter,
    GameId.TEXTWORLD_COOKING: TextWorldCookingAdapter,
}


__all__ = [
    "TextWorldAdapter",
    "TextWorldSimpleAdapter",
    "TextWorldCoinCollectorAdapter",
    "TextWorldTreasureHunterAdapter",
    "TextWorldCookingAdapter",
    "TEXTWORLD_ADAPTERS",
    "TEXTWORLD_CHALLENGES",
]
