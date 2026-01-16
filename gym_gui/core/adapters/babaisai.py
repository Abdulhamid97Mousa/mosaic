"""BabaIsAI environment adapter for the MOSAIC GUI.

BabaIsAI is an AI-friendly version of "Baba Is You" puzzle game where agents
manipulate rules by pushing word blocks. Excellent for testing LLM reasoning
and compositional generalization.

Paper: Cloos et al. (2024). "Baba Is AI: Break the Rules to Beat the Benchmark"
       ICML 2024 Workshop on LLMs and Cognition
Repository: https://github.com/nacloos/baba-is-ai
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from gym_gui.core.adapters.base import (
    AdapterContext,
    AdapterStep,
    EnvironmentAdapter,
    StepState,
)
from gym_gui.core.enums import ControlMode, GameId, RenderMode
from gym_gui.logging_config.log_constants import (
    LOG_ADAPTER_ENV_CLOSED,
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
)

try:  # pragma: no cover - import guard
    import baba
    from baba.grid import BabaIsYouEnv
except ImportError:  # pragma: no cover
    baba = None  # type: ignore[assignment]
    BabaIsYouEnv = None  # type: ignore[assignment, misc]


# BabaIsAI action names
BABAISAI_ACTIONS: List[str] = ["UP", "DOWN", "LEFT", "RIGHT", "IDLE"]

# Log frequency for step events
_BABAISAI_STEP_LOG_FREQUENCY = 50


@dataclass
class BabaIsAIConfig:
    """Configuration for BabaIsAI environment.

    Attributes:
        env_id: The specific puzzle environment ID (e.g., "two_room-break_stop-make_win")
        add_ruleset: Whether to include active rules in text observation
        seed: Random seed for reproducibility
    """

    env_id: str = "two_room-break_stop-make_win"
    add_ruleset: bool = True
    seed: Optional[int] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "env_id": self.env_id,
            "add_ruleset": self.add_ruleset,
            "seed": self.seed,
            "env_kwargs": self.env_kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BabaIsAIConfig":
        """Create config from dictionary."""
        return cls(
            env_id=data.get("env_id", "two_room-break_stop-make_win"),
            add_ruleset=data.get("add_ruleset", True),
            seed=data.get("seed"),
            env_kwargs=data.get("env_kwargs", {}),
        )


class BabaIsAIAdapter(EnvironmentAdapter[Dict[str, Any], str]):
    """Adapter for BabaIsAI puzzle environments.

    BabaIsAI provides both RGB images and text observations describing
    object positions relative to the player and active rules.

    The action space is discrete with 5 actions: UP, DOWN, LEFT, RIGHT, IDLE.
    Actions are passed as strings to match the BALROG wrapper convention.
    """

    default_render_mode = RenderMode.RGB_ARRAY
    supported_render_modes = (RenderMode.RGB_ARRAY,)
    supported_control_modes = (
        ControlMode.HUMAN_ONLY,
        ControlMode.AGENT_ONLY,
        ControlMode.HYBRID_TURN_BASED,
    )

    DEFAULT_ENV_ID = "two_room-break_stop-make_win"

    def __init__(
        self,
        context: AdapterContext | None = None,
        *,
        config: BabaIsAIConfig | None = None,
    ) -> None:
        """Initialize the BabaIsAI adapter.

        Args:
            context: Adapter context with settings and control mode
            config: BabaIsAI configuration
        """
        super().__init__(context)
        if config is None:
            config = BabaIsAIConfig(env_id=self.DEFAULT_ENV_ID)
        self._config = config
        self._env_id = config.env_id or self.DEFAULT_ENV_ID
        self._step_counter = 0
        self._current_rules: List[str] = []
        self._wrapper: Any = None  # BALROG-style wrapper if needed

    @property
    def id(self) -> str:  # type: ignore[override]
        """Return the environment identifier."""
        return f"BabaIsAI-{self._env_id}"

    def load(self) -> None:
        """Instantiate the BabaIsAI environment."""
        if baba is None:
            raise RuntimeError(
                "BabaIsAI package not installed. "
                "Install with: pip install git+https://github.com/nacloos/baba-is-ai"
            )
        try:
            # Create environment using baba.make() with render_mode to avoid deprecation warnings
            # This uses the newer gym API: render_mode during make() instead of mode in render()
            env = baba.make(f"env/{self._env_id}", render_mode='rgb_array', **self._config.env_kwargs)
            self._env = env
            self.log_constant(
                LOG_ADAPTER_ENV_CREATED,
                extra={
                    "env_id": self._env_id,
                    "add_ruleset": self._config.add_ruleset,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create BabaIsAI environment '{self._env_id}': {exc}"
            ) from exc

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> AdapterStep[Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Optional random seed (not directly supported by baba)
            options: Additional reset options

        Returns:
            Initial step result with observation
        """
        env = self._require_env()

        # BabaIsAI uses old gym API: reset() returns observation only
        raw_obs = env.reset()

        # Process observation to include text description
        observation = self._process_observation(raw_obs)
        self._step_counter = 0

        info: Dict[str, Any] = {
            "_babaisai_raw_observation": raw_obs,
            "active_rules": self._current_rules,
            "env_id": self._env_id,
        }

        self.log_constant(
            LOG_ADAPTER_ENV_RESET,
            extra={
                "env_id": self._env_id,
                "seed": seed if seed is not None else "None",
            },
        )

        return self._package_step(observation, 0.0, False, False, info)

    def step(self, action: str | int) -> AdapterStep[Dict[str, Any]]:
        """Execute an action in the environment.

        Args:
            action: Action to execute (string name or integer index)

        Returns:
            Step result with observation, reward, termination flags, and info
        """
        env = self._require_env()

        # Convert action to integer if string
        if isinstance(action, str):
            action_str = action.upper()
            if action_str in BABAISAI_ACTIONS:
                action_int = BABAISAI_ACTIONS.index(action_str)
            else:
                # Default to IDLE for unknown actions
                action_int = BABAISAI_ACTIONS.index("IDLE")
        else:
            action_int = int(action)

        # BabaIsAI uses old gym API: step returns (obs, reward, done, info)
        raw_obs, reward, done, info = env.step(action_int)

        observation = self._process_observation(raw_obs)
        info = dict(info) if info else {}
        info["_babaisai_raw_observation"] = raw_obs
        info["active_rules"] = self._current_rules
        info["action_name"] = BABAISAI_ACTIONS[action_int] if 0 <= action_int < len(BABAISAI_ACTIONS) else str(action_int)

        # Old gym API uses single 'done' flag - treat as termination
        terminated = done
        truncated = False

        self._step_counter += 1

        if self._step_counter % _BABAISAI_STEP_LOG_FREQUENCY == 1:
            self.log_constant(
                LOG_ADAPTER_STEP_SUMMARY,
                extra={
                    "env_id": self._env_id,
                    "step": self._step_counter,
                    "action": info["action_name"],
                    "reward": float(reward),
                    "terminated": terminated,
                },
            )

        return self._package_step(observation, float(reward), terminated, truncated, info)

    def render(self) -> Dict[str, Any]:
        """Render the environment.

        Returns:
            Dictionary with RGB array and metadata
        """
        env = self._require_env()
        try:
            # render() without mode argument (uses render_mode from initialization)
            frame = env.render()
            array = np.asarray(frame)
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": array,
                "game_id": self._env_id,
                "active_rules": self._current_rules,
            }
        except Exception:
            # Return empty frame if render fails
            return {
                "mode": RenderMode.RGB_ARRAY.value,
                "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
                "game_id": self._env_id,
            }

    def close(self) -> None:
        """Close the environment."""
        if self._env is not None:
            self.log_constant(
                LOG_ADAPTER_ENV_CLOSED,
                extra={"env_id": self.id},
            )
            # BabaIsAI may not have close() method
            if hasattr(self._env, "close"):
                self._env.close()
            self._env = None

    def _process_observation(self, raw_obs: Any) -> Dict[str, Any]:
        """Process raw observation into structured format.

        Args:
            raw_obs: Raw observation from environment

        Returns:
            Processed observation with text and image components
        """
        env = self._require_env()
        observation: Dict[str, Any] = {}

        # Get RGB image
        try:
            # render() without mode argument (uses render_mode from initialization)
            image = env.render()
            observation["image"] = np.asarray(image)
        except Exception:
            observation["image"] = None

        # Get text observation (object positions relative to player)
        text_obs = self._get_text_observation()
        observation["text"] = text_obs

        # Get active rules
        self._current_rules = self._get_active_rules()
        observation["rules"] = self._current_rules

        # Build full text prompt
        prompt_parts = []
        if self._config.add_ruleset and self._current_rules:
            prompt_parts.append(f"Active rules:\n" + "\n".join(self._current_rules))
        if text_obs:
            prompt_parts.append(f"Objects on the map:\n{text_obs}")

        observation["prompt"] = "\n\n".join(prompt_parts)

        return observation

    def _get_active_rules(self) -> List[str]:
        """Extract active rules from the environment.

        Returns:
            List of active rule strings (e.g., ["baba is you", "flag is win"])
        """
        env = self._require_env()
        rules = []

        try:
            if hasattr(env, "grid") and hasattr(env.grid, "_ruleset"):
                for rule in env.grid._ruleset.get("_rule_", []):
                    if "object" not in rule or "property" not in rule:
                        continue
                    obj_name = rule["object"].removeprefix("f")
                    # Try to get property name mapping
                    prop = rule["property"]
                    try:
                        from baba.world_object import name_mapping
                        prop_name = name_mapping.get(prop, prop)
                    except ImportError:
                        prop_name = prop
                    rules.append(f"{obj_name} is {prop_name}")
        except Exception:
            pass

        return rules

    def _get_text_observation(self) -> str:
        """Generate text description of object positions.

        Returns:
            Text description of objects relative to player position
        """
        env = self._require_env()

        try:
            # Find the "you" object (player)
            you_obj = None
            for rule in env.grid._ruleset.get("_rule_", []):
                if rule.get("property") == "you" or (
                    hasattr(rule.get("property"), "__str__") and "you" in str(rule.get("property"))
                ):
                    you_obj = rule.get("object")
                    break

            if you_obj is None:
                return "No controllable object found"

            # Find player position
            player_pos = None
            for j in range(env.height):
                for i in range(env.width):
                    cell = env.grid.get(i, j)
                    if cell is not None and cell.type == you_obj:
                        player_pos = (i, j)
                        break
                if player_pos:
                    break

            if player_pos is None:
                return "Player position not found"

            # Find other objects and describe positions
            descriptions = []
            for j in range(env.height):
                for i in range(env.width):
                    if (i, j) == player_pos:
                        continue
                    cell = env.grid.get(i, j)
                    if cell is None:
                        continue

                    # Get object name
                    name = cell.type
                    if hasattr(cell, "name"):
                        name = cell.name

                    # Calculate relative position
                    dx, dy = i - player_pos[0], j - player_pos[1]
                    parts = []
                    if dx > 0:
                        parts.append(f"{dx} step{'s' if dx > 1 else ''} right")
                    elif dx < 0:
                        parts.append(f"{-dx} step{'s' if -dx > 1 else ''} left")
                    if dy > 0:
                        parts.append(f"{dy} step{'s' if dy > 1 else ''} down")
                    elif dy < 0:
                        parts.append(f"{-dy} step{'s' if -dy > 1 else ''} up")

                    if parts:
                        descriptions.append(f"{name} {' and '.join(parts)}")

            return "\n".join(descriptions) if descriptions else "No objects nearby"

        except Exception as e:
            return f"Could not generate text observation: {e}"

    def build_step_state(
        self,
        observation: Dict[str, Any],
        info: Mapping[str, Any],
    ) -> StepState:
        """Construct the canonical StepState for the current step."""
        return StepState(
            active_agent=None,
            agents=(),
            metrics={
                "step_count": self._step_counter,
                "num_rules": len(self._current_rules),
            },
            environment={
                "env_id": self._env_id,
                "active_rules": self._current_rules,
            },
            raw=dict(info) if isinstance(info, Mapping) else {},
        )

    def get_action_meanings(self) -> List[str]:
        """Get human-readable action names.

        Returns:
            List of action names
        """
        return BABAISAI_ACTIONS.copy()

    def sample_action(self) -> str:
        """Sample a random action.

        Returns:
            Random action name
        """
        import random
        return random.choice(BABAISAI_ACTIONS)


# Adapter registry for factory pattern
BABAISAI_ADAPTERS: Dict[GameId, type[BabaIsAIAdapter]] = {
    GameId.BABAISAI_DEFAULT: BabaIsAIAdapter,
}


def create_babaisai_adapter(
    env_id: str = "two_room-break_stop-make_win",
    context: AdapterContext | None = None,
    config: BabaIsAIConfig | None = None,
) -> BabaIsAIAdapter:
    """Factory function to create a BabaIsAI adapter.

    Args:
        env_id: Specific puzzle environment ID
        context: Adapter context
        config: Optional configuration

    Returns:
        BabaIsAI adapter instance
    """
    if config is None:
        config = BabaIsAIConfig(env_id=env_id)
    return BabaIsAIAdapter(context, config=config)


__all__ = [
    "BabaIsAIConfig",
    "BabaIsAIAdapter",
    "BABAISAI_ACTIONS",
    "BABAISAI_ADAPTERS",
    "create_babaisai_adapter",
]
