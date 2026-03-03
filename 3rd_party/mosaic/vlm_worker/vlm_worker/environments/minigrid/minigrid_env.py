"""MiniGrid environment creation and wrapping.

This module provides factory functions and wrappers for MiniGrid environments.
MiniGrid is a minimalistic gridworld environment for reinforcement learning.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import minigrid
import numpy as np

from .clean_lang_wrapper import MiniGridTextCleanLangWrapper

logger = logging.getLogger(__name__)

# Register MiniGrid environments
minigrid.register_minigrid_envs()


# =============================================================================
# MiniGrid Object and Color Mappings (from minigrid.core.constants)
# =============================================================================

# Object type indices - these are from minigrid.core.constants.OBJECT_TO_IDX
MINIGRID_OBJECT_NAMES = {
    0: "unseen",    # Not visible (outside field of view)
    1: "empty",     # Empty cell
    2: "wall",      # Wall (impassable)
    3: "floor",     # Floor tile
    4: "door",      # Door (can be opened with toggle)
    5: "key",       # Key (can be picked up)
    6: "ball",      # Ball (can be picked up)
    7: "box",       # Box (can contain items)
    8: "goal",      # Goal square (target destination)
    9: "lava",      # Lava (dangerous, ends episode)
    10: "agent",    # Agent position
}

# Color indices - from minigrid.core.constants.COLOR_TO_IDX
MINIGRID_COLOR_NAMES = {
    0: "red",
    1: "green",
    2: "blue",
    3: "purple",
    4: "yellow",
    5: "grey",
}


# =============================================================================
# MiniGrid Text Description Generator
# =============================================================================

def generate_minigrid_descriptions(obs: Dict[str, Any]) -> List[str]:
    """Generate text descriptions from MiniGrid observation grid.

    The MiniGrid observation grid is (7, 7, 3) where each cell has:
    - [0]: object type (see MINIGRID_OBJECT_NAMES)
    - [1]: color index (see MINIGRID_COLOR_NAMES)
    - [2]: state (e.g., door open/closed/locked: 0/1/2)

    The agent is always at position (3, 6) in its egocentric 7x7 view,
    facing "up" (towards row 0).

    Args:
        obs: Observation dict with 'image' key containing (7,7,3) array

    Returns:
        List of description strings
    """
    descriptions = []
    image = obs.get("image")

    if image is None:
        logger.warning("No 'image' key in observation")
        return ["You see nothing special."]

    logger.debug(f"generate_minigrid_descriptions - image shape: {image.shape}")

    # Agent is at (3, 6) in the 7x7 egocentric view, facing "up" (row 0)
    agent_col = 3
    agent_row = 6

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            obj_type = int(image[row, col, 0])
            obj_color = int(image[row, col, 1])
            obj_state = int(image[row, col, 2])

            # Skip non-interesting objects
            # unseen(0), empty(1), wall(2), floor(3) are not worth mentioning
            if obj_type <= 3:
                continue

            # Skip if not in our mapping (shouldn't happen)
            if obj_type not in MINIGRID_OBJECT_NAMES:
                continue

            obj_name = MINIGRID_OBJECT_NAMES[obj_type]
            color_name = MINIGRID_COLOR_NAMES.get(obj_color, "")

            # Calculate relative position from agent
            # Agent faces "up" (row 0), so:
            # - dy > 0 means object is ahead (lower row number)
            # - dx > 0 means object is to the right (higher col number)
            dy = agent_row - row  # Steps ahead (positive = ahead)
            dx = col - agent_col  # Steps right (positive = right)

            # Build direction description
            parts = []
            if dy > 0:
                parts.append(f"{dy} step{'s' if dy > 1 else ''} ahead")
            elif dy < 0:
                parts.append(f"{-dy} step{'s' if -dy > 1 else ''} behind")

            if dx > 0:
                parts.append(f"{dx} step{'s' if dx > 1 else ''} to the right")
            elif dx < 0:
                parts.append(f"{-dx} step{'s' if -dx > 1 else ''} to the left")

            # Add state information for doors
            state_str = ""
            if obj_name == "door":
                if obj_state == 0:
                    state_str = " (open)"
                elif obj_state == 1:
                    state_str = " (closed)"
                elif obj_state == 2:
                    state_str = " (locked)"

            if parts:
                direction = " and ".join(parts)
                if color_name:
                    descriptions.append(f"You see a {color_name} {obj_name}{state_str} {direction}")
                else:
                    descriptions.append(f"You see a {obj_name}{state_str} {direction}")

    result = descriptions if descriptions else ["You see nothing special."]
    logger.debug(f"Generated {len(result)} descriptions")
    return result


# =============================================================================
# MiniGrid Description Wrapper
# =============================================================================

class MiniGridDescriptionWrapper(gym.Wrapper):
    """Wrapper that adds 'descriptions' to info dict for MiniGrid environments.

    This wrapper intercepts reset() and step() calls to add a 'descriptions'
    key to the info dict. The descriptions are generated from the observation
    grid and provide text-based spatial information about visible objects.

    This is required by MiniGridTextCleanLangWrapper which expects
    info["descriptions"] to be present.
    """

    def __init__(self, env: gym.Env):
        """Initialize the wrapper.

        Args:
            env: The MiniGrid environment to wrap
        """
        super().__init__(env)

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment and add descriptions to info."""
        obs, info = self.env.reset(**kwargs)
        info["descriptions"] = generate_minigrid_descriptions(obs)
        logger.debug(f"MiniGridDescriptionWrapper.reset - {len(info['descriptions'])} descriptions")
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute step and add descriptions to info."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["descriptions"] = generate_minigrid_descriptions(obs)
        logger.debug(f"MiniGridDescriptionWrapper.step - {len(info['descriptions'])} descriptions")
        return obs, reward, terminated, truncated, info


# =============================================================================
# Environment Factory
# =============================================================================

def make_minigrid_env(
    env_name: str,
    task: str,
    config: Any,
    render_mode: Optional[str] = None
) -> gym.Env:
    """Create a MiniGrid environment with proper text wrappers.

    The wrapping chain:
    1. gym.make() - base MiniGrid environment
    2. MiniGridDescriptionWrapper - adds info["descriptions"] from observation
    3. MiniGridTextCleanLangWrapper - transforms obs to include obs["text"]

    Args:
        env_name: Environment category (should be "minigrid")
        task: Specific environment ID (e.g., "MiniGrid-Empty-8x8-v0")
        config: Configuration object (may contain minigrid_kwargs)
        render_mode: Rendering mode ("rgb_array", "human", etc.)

    Returns:
        Wrapped MiniGrid environment ready for LLM agents
    """
    # Get minigrid-specific kwargs from config if available
    minigrid_kwargs = {}
    if hasattr(config, "envs") and hasattr(config.envs, "minigrid_kwargs"):
        minigrid_kwargs = dict(config.envs.minigrid_kwargs)

    logger.info(f"Creating MiniGrid environment: {task}")

    # Create base environment
    env = gym.make(task, render_mode=render_mode, **minigrid_kwargs)

    # Add description wrapper to provide info["descriptions"]
    env = MiniGridDescriptionWrapper(env)

    # Wrap with text clean lang wrapper for obs["text"]["long_term_context"]
    env = MiniGridTextCleanLangWrapper(env, **minigrid_kwargs)

    logger.info(f"MiniGrid environment created with wrappers")
    return env
