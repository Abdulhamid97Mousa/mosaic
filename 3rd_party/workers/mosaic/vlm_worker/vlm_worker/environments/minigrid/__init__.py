"""MiniGrid environment module for LLM Worker.

This module provides MiniGrid-specific wrappers and utilities for
navigation-focused environments like MiniGrid-Empty, MiniGrid-DoorKey, etc.

MiniGrid environments are simpler navigation tasks compared to BabyAI,
focused on reaching goals, avoiding obstacles, and basic object interaction.
"""

from .clean_lang_wrapper import MiniGridTextCleanLangWrapper
from .minigrid_env import (
    make_minigrid_env,
    MiniGridDescriptionWrapper,
    MINIGRID_OBJECT_NAMES,
    MINIGRID_COLOR_NAMES,
    generate_minigrid_descriptions,
)

# MiniGrid action space (same as BabyAI but may have different semantics)
ACTIONS = {
    "turn left": "rotate 90 degrees to the left",
    "turn right": "rotate 90 degrees to the right",
    "go forward": "move one cell forward in your facing direction",
    "pick up": "pick up an object at your current position",
    "drop": "drop the object you are carrying",
    "toggle": "interact with the object directly in front of you (open doors, etc.)",
}


def get_instruction_prompt(env=None, mission=None):
    """Generate instruction prompt for MiniGrid environments.

    Args:
        env: The MiniGrid environment (optional, for future customization)
        mission: The mission string (e.g., "get to the green goal square")

    Returns:
        str: The instruction prompt for the LLM
    """
    action_strings = ",\n".join(f"{action}: {description}" for action, description in ACTIONS.items())

    mission_text = mission if mission else "navigate the grid environment"

    instruction_prompt = f"""
You are an agent navigating a grid world. Your mission is to {mission_text}.

The following are the possible actions you can take, followed by a description:

{action_strings}.

Important tips:
- You have a limited field of view (7x7 grid in front of you)
- "nothing special" means no notable objects are visible - keep exploring!
- When you see "a green goal X steps ahead", navigate towards it
- Use 'turn left' and 'turn right' to change direction and explore
- Use 'toggle' to open doors or interact with objects in front of you
- Avoid lava (stepping on it ends the episode)

PLAY!
""".strip()

    return instruction_prompt


__all__ = [
    "MiniGridTextCleanLangWrapper",
    "MiniGridDescriptionWrapper",
    "make_minigrid_env",
    "get_instruction_prompt",
    "generate_minigrid_descriptions",
    "ACTIONS",
    "MINIGRID_OBJECT_NAMES",
    "MINIGRID_COLOR_NAMES",
]
