"""Environment wrappers and factory functions for MOSAIC LLM Worker.

This module provides:
- Environment factory functions for creating gym/pettingzoo environments
- Text description generators for multi-agent observations
- Environment wrappers for compatibility fixes

MultiGrid functions are now in environments/multigrid/ - this module re-exports
them for backwards compatibility.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import gymnasium as gym

logger = logging.getLogger(__name__)


# =============================================================================
# Re-export MultiGrid functions from new location
# =============================================================================

from .environments.multigrid import (
    # Constants
    MULTIGRID_OBJECT_NAMES,
    MULTIGRID_COLOR_NAMES,
    MULTIGRID_DIRECTION_NAMES,
    # Observation descriptions
    describe_observation_egocentric,
    describe_observation_with_teammates,
    extract_visible_teammates,
    generate_multigrid_description,
    # Environment factory
    make_multigrid_env,
    # Prompts
    get_instruction_prompt as get_multigrid_instruction_prompt,
)


# =============================================================================
# Environment Names
# =============================================================================

ENV_NAMES = (
    "multigrid",
    "pettingzoo",
    "gymnasium",
    "minigrid",
    "babyai",
)


# =============================================================================
# BabyAI Object and Color Constants
# =============================================================================

BABYAI_OBJECT_NAMES = {
    1: "wall",
    2: "floor",
    3: "door",
    4: "key",
    5: "ball",
    6: "box",
    7: "goal",
    8: "lava",
    10: "agent",
}

BABYAI_COLOR_NAMES = {
    0: "red",
    1: "green",
    2: "blue",
    3: "purple",
    4: "yellow",
    5: "grey",
}


# =============================================================================
# BabyAI Text Description Generator
# =============================================================================

def generate_babyai_descriptions(obs: Dict[str, Any]) -> List[str]:
    """Generate text descriptions from BabyAI observation grid."""
    descriptions = []
    image = obs.get("image")

    logger.debug(f"LOG1050 generate_babyai_descriptions obs keys: {list(obs.keys()) if isinstance(obs, dict) else 'not a dict'}")
    logger.debug(f"LOG1051 image shape: {image.shape if image is not None else 'None'}")

    if image is None:
        return ["You see nothing special."]

    agent_pos = (3, 6)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            obj_type = int(image[i, j, 0])
            obj_color = int(image[i, j, 1])

            if obj_type <= 2 or obj_type not in BABYAI_OBJECT_NAMES:
                continue

            obj_name = BABYAI_OBJECT_NAMES[obj_type]
            color_name = BABYAI_COLOR_NAMES.get(obj_color, "")

            dx = i - agent_pos[0]
            dy = agent_pos[1] - j

            parts = []
            if dy > 0:
                parts.append(f"{dy} step{'s' if dy > 1 else ''} ahead")
            elif dy < 0:
                parts.append(f"{-dy} step{'s' if -dy > 1 else ''} behind")
            if dx > 0:
                parts.append(f"{dx} step{'s' if dx > 1 else ''} to the right")
            elif dx < 0:
                parts.append(f"{-dx} step{'s' if -dx > 1 else ''} to the left")

            if parts:
                direction = " and ".join(parts)
                if color_name:
                    descriptions.append(f"You see a {color_name} {obj_name} {direction}")
                else:
                    descriptions.append(f"You see a {obj_name} {direction}")

    result = descriptions if descriptions else ["You see nothing special."]
    logger.debug(f"LOG1052 generated {len(result)} descriptions: {result[:3]}...")
    return result


# =============================================================================
# BabyAI Description Wrapper
# =============================================================================

class BabyAIDescriptionWrapper:
    """Wrapper that adds 'descriptions' to info dict for BabyAI environments."""

    def __init__(self, env: "gym.Env"):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self.unwrapped = getattr(env, "unwrapped", env)

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        info["descriptions"] = generate_babyai_descriptions(obs)
        logger.debug(f"LOG1053 BabyAIDescriptionWrapper.reset - descriptions added: {len(info['descriptions'])} items")
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["descriptions"] = generate_babyai_descriptions(obs)
        logger.debug(f"LOG1054 BabyAIDescriptionWrapper.step - descriptions added: {len(info['descriptions'])} items")
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        return self.env.render()

    def close(self) -> None:
        self.env.close()


# =============================================================================
# PettingZoo Instruction Prompts
# =============================================================================

def get_pettingzoo_instruction_prompt(game_name: str, player_id: str) -> str:
    """Get instruction prompt for a PettingZoo game."""
    prompts = {
        "chess_v6": (
            f"You are playing chess as {player_id}. "
            "Analyze the board position and select your next move. "
            "Output ONLY the move in UCI format (e.g., 'e2e4', 'g1f3'). "
            "The legal moves will be provided in the observation."
        ),
        "chess": (
            f"You are playing chess as {player_id}. "
            "Output ONLY the move in UCI format."
        ),
        "connect_four_v3": (
            f"You are playing Connect Four as {player_id}. "
            "Select which column (0-6) to drop your piece. "
            "Output ONLY a single number."
        ),
        "connect_four": (
            f"You are playing Connect Four as {player_id}. "
            "Output ONLY a single number (0-6)."
        ),
        "go_v5": (
            f"You are playing Go as {player_id}. "
            "Output your move as coordinates (row, col) or 'pass'."
        ),
        "tictactoe_v3": (
            f"You are playing Tic-Tac-Toe as {player_id}. "
            "Select a position (0-8). Output ONLY a single number."
        ),
    }

    if game_name in prompts:
        return prompts[game_name]

    base_name = game_name.rsplit("_v", 1)[0] if "_v" in game_name else game_name
    if base_name in prompts:
        return prompts[base_name]

    return (
        f"You are playing {game_name} as {player_id}. "
        "Analyze the game state and select your action. "
        "Output ONLY the action, no explanation."
    )


# =============================================================================
# Environment Factory Functions
# =============================================================================

def make_babyai_env(
    task: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create a BabyAI environment with BALROG-compatible observation format.

    Wrapping chain:
    1. gym.make() - base MiniGrid/BabyAI environment
    2. BabyAIDescriptionWrapper - adds info["descriptions"] from observation grid
    3. BabyAITextCleanLangWrapper - transforms obs to include obs["text"]["long_term_context"]

    This matches BALROG's approach where obs["text"]["long_term_context"] contains
    the text description for the LLM.
    """
    import gymnasium as gym
    from .environments.babyai_text import BabyAITextCleanLangWrapper

    try:
        import minigrid
        if "MiniGrid-Empty-5x5-v0" not in gym.envs.registry:
            minigrid.register_minigrid_envs()
    except ImportError:
        raise ImportError(
            "MiniGrid not installed. Install with: pip install minigrid>=2.0.0"
        )

    if task.startswith("BabyAI-MixedTrainLocal-v0/"):
        base_task, goal = task.split("/")
        while True:
            env = gym.make(base_task, render_mode=render_mode, **kwargs)
            if env.unwrapped.action_kinds[0].replace(" ", "_") == goal:
                break
            env.close()
    else:
        env = gym.make(task, render_mode=render_mode, **kwargs)

    # Chain wrappers like BALROG:
    # 1. Add descriptions to info dict
    env = BabyAIDescriptionWrapper(env)
    # 2. Transform obs to include obs["text"]["long_term_context"] (BALROG format)
    env = BabyAITextCleanLangWrapper(env)

    return env


def make_pettingzoo_env(
    task: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create a PettingZoo environment."""
    if "_v" in task:
        env_name = task.rsplit("_v", 1)[0]
    else:
        env_name = task

    env = None

    try:
        from pettingzoo import classic
        env_map = {
            "chess": "chess_v6",
            "connect_four": "connect_four_v3",
            "go": "go_v5",
            "tictactoe": "tictactoe_v3",
            "backgammon": "backgammon_v3",
            "checkers": "checkers_v3",
        }

        if env_name in env_map:
            module_name = env_map[env_name]
            if hasattr(classic, module_name):
                env_module = getattr(classic, module_name)
                env = env_module.env(render_mode=render_mode, **kwargs)
    except ImportError:
        pass

    if env is None:
        try:
            from pettingzoo import mpe
            mpe_envs = {
                "simple_spread": "simple_spread_v3",
                "simple_adversary": "simple_adversary_v3",
                "simple_tag": "simple_tag_v3",
            }

            if env_name in mpe_envs:
                module_name = mpe_envs[env_name]
                if hasattr(mpe, module_name):
                    env_module = getattr(mpe, module_name)
                    env = env_module.env(render_mode=render_mode, **kwargs)
        except ImportError:
            pass

    if env is None:
        raise ValueError(f"Unknown PettingZoo environment: {task}")

    return env


def make_env(
    env_name: str,
    task: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create an environment based on environment family."""
    env_name = env_name.lower()

    if env_name == "multigrid" or "MultiGrid" in task:
        return make_multigrid_env(task, render_mode=render_mode, **kwargs)
    elif env_name in ("babyai", "minigrid"):
        return make_babyai_env(task, render_mode=render_mode, **kwargs)
    elif env_name == "pettingzoo":
        return make_pettingzoo_env(task, render_mode=render_mode, **kwargs)
    elif env_name == "gymnasium":
        import gymnasium as gym
        return gym.make(task, render_mode=render_mode, **kwargs)
    else:
        raise ValueError(f"Unknown environment family: {env_name}. Supported: {ENV_NAMES}")


# =============================================================================
# Utility Functions
# =============================================================================

def sanitize_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize info dict for JSON serialization."""
    result = {}
    for key, value in info.items():
        try:
            json.dumps(value)
            result[key] = value
        except (TypeError, ValueError):
            result[key] = str(value)[:100]
    return result


def obs_to_str(obs: Any, env_name: str = "unknown") -> str:
    """Convert observation to string for telemetry."""
    if isinstance(obs, str):
        return obs

    if isinstance(obs, dict):
        if "text" in obs:
            return str(obs["text"])
        if "message" in obs:
            return str(obs["message"])
        if "mission" in obs:
            return str(obs["mission"])
        return json.dumps({k: str(v)[:100] for k, v in obs.items()})

    if isinstance(obs, np.ndarray):
        return f"<ndarray shape={obs.shape} dtype={obs.dtype}>"

    return str(obs)[:500]


def check_action_validity(action_str: str, valid_actions: List[str]) -> Optional[int]:
    """Check if action string matches a valid action."""
    action_str = action_str.strip().lower()

    for i, action in enumerate(valid_actions):
        if action_str == action.lower():
            return i

    for i, action in enumerate(valid_actions):
        if action.lower() in action_str or action_str in action.lower():
            return i

    return None


__all__ = [
    # Environment names
    "ENV_NAMES",
    # Factory functions
    "make_env",
    "make_multigrid_env",
    "make_babyai_env",
    "make_pettingzoo_env",
    # BabyAI
    "BabyAIDescriptionWrapper",
    "generate_babyai_descriptions",
    "BABYAI_OBJECT_NAMES",
    "BABYAI_COLOR_NAMES",
    # MultiGrid (re-exported from environments.multigrid)
    "MULTIGRID_OBJECT_NAMES",
    "MULTIGRID_COLOR_NAMES",
    "MULTIGRID_DIRECTION_NAMES",
    "generate_multigrid_description",
    "describe_observation_egocentric",
    "describe_observation_with_teammates",
    "extract_visible_teammates",
    "get_multigrid_instruction_prompt",
    # PettingZoo
    "get_pettingzoo_instruction_prompt",
    # Utilities
    "sanitize_info",
    "obs_to_str",
    "check_action_validity",
]
