"""BabyAI/MiniGrid environment creation and wrapping."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import minigrid
import numpy as np

from llm_worker.environments.babyai_text import BabyAITextCleanLangWrapper

logger = logging.getLogger(__name__)

minigrid.register_minigrid_envs()

# see discussion starting here: https://github.com/Farama-Foundation/Minigrid/pull/381#issuecomment-1646800992
broken_bonus_envs = {
    "BabyAI-PutNextS5N2Carrying-v0",
    "BabyAI-PutNextS6N3Carrying-v0",
    "BabyAI-PutNextS7N4Carrying-v0",
    "BabyAI-KeyInBox-v0",
}


# get all babyai envs (except the broken ones)
BABYAI_ENVS = []
for env_spec in gym.envs.registry:
    id = env_spec
    if id.split("-")[0] == "BabyAI":
        if id not in broken_bonus_envs:
            BABYAI_ENVS.append(id)

BABYAI_ENVS += [
    "BabyAI-MixedTrainLocal-v0/goto",
    "BabyAI-MixedTrainLocal-v0/pickup",
    "BabyAI-MixedTrainLocal-v0/open",
    "BabyAI-MixedTrainLocal-v0/putnext",
    "BabyAI-MixedTrainLocal-v0/pick_up_seq_go_to",
]


# =============================================================================
# BabyAI Object and Color Mappings
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
    """Generate text descriptions from BabyAI observation grid.

    The MiniGrid observation grid is (7, 7, 3) where each cell has:
    - [0]: object type (see BABYAI_OBJECT_NAMES)
    - [1]: color index (see BABYAI_COLOR_NAMES)
    - [2]: state (e.g., door open/closed)

    The agent is always at position (3, 6) in its own view, facing "up" (towards row 0).
    """
    descriptions = []
    image = obs.get("image")

    logger.debug(f"generate_babyai_descriptions obs keys: {list(obs.keys()) if isinstance(obs, dict) else 'not a dict'}")
    logger.debug(f"image shape: {image.shape if image is not None else 'None'}")

    if image is None:
        return ["You see nothing special."]

    # Agent is at (3, 6) in the 7x7 egocentric view, facing "up"
    agent_pos = (3, 6)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            obj_type = int(image[i, j, 0])
            obj_color = int(image[i, j, 1])

            # Skip empty/wall/floor (types 0, 1, 2)
            if obj_type <= 2 or obj_type not in BABYAI_OBJECT_NAMES:
                continue

            obj_name = BABYAI_OBJECT_NAMES[obj_type]
            color_name = BABYAI_COLOR_NAMES.get(obj_color, "")

            # Calculate relative position from agent
            dx = i - agent_pos[0]  # left/right
            dy = agent_pos[1] - j  # ahead/behind (inverted because row 0 is ahead)

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
    logger.debug(f"generated {len(result)} descriptions: {result[:3]}...")
    return result


# =============================================================================
# BabyAI Description Wrapper
# =============================================================================

class BabyAIDescriptionWrapper(gym.Wrapper):
    """Wrapper that adds 'descriptions' to info dict for BabyAI environments.

    This wrapper intercepts reset() and step() calls to add a 'descriptions'
    key to the info dict. The descriptions are generated from the observation
    grid and provide text-based spatial information about visible objects.

    This is required by BabyAITextCleanLangWrapper which expects
    info["descriptions"] to be present.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        info["descriptions"] = generate_babyai_descriptions(obs)
        logger.debug(f"BabyAIDescriptionWrapper.reset - descriptions added: {len(info['descriptions'])} items")
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["descriptions"] = generate_babyai_descriptions(obs)
        logger.debug(f"BabyAIDescriptionWrapper.step - descriptions added: {len(info['descriptions'])} items")
        return obs, reward, terminated, truncated, info


# =============================================================================
# Environment Factory
# =============================================================================

def make_babyai_env(env_name, task, config, render_mode: Optional[str] = None):
    """Create a BabyAI/MiniGrid environment with proper text wrappers.

    The wrapping chain:
    1. gym.make() - base environment
    2. BabyAIDescriptionWrapper - adds info["descriptions"] from observation grid
    3. BabyAITextCleanLangWrapper - transforms obs to include obs["text"]["long_term_context"]
    """
    # Get babyai_kwargs from config if available
    babyai_kwargs = {}
    if hasattr(config, "envs") and hasattr(config.envs, "babyai_kwargs"):
        babyai_kwargs = dict(config.envs.babyai_kwargs)

    if task.startswith("BabyAI-MixedTrainLocal-v0/"):
        base_task, goal = task.split("/")
        while True:
            env = gym.make(base_task, render_mode=render_mode, **babyai_kwargs)
            if env.unwrapped.action_kinds[0].replace(" ", "_") == goal:
                break
            env.close()
    else:
        # Standard environment creation
        env = gym.make(task, render_mode=render_mode, **babyai_kwargs)

    # Add description wrapper to provide info["descriptions"]
    env = BabyAIDescriptionWrapper(env)

    # Wrap with text clean lang wrapper for obs["text"]["long_term_context"]
    env = BabyAITextCleanLangWrapper(env, **babyai_kwargs)

    return env
