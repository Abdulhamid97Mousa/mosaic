"""Environment wrappers for BALROG Worker.

This module provides wrappers that fix compatibility issues between
BALROG and standard Gymnasium environments. Instead of modifying the
upstream BALROG code, we wrap environments here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# BabyAI Text Description Generator
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


def generate_babyai_descriptions(obs: Dict[str, Any]) -> list[str]:
    """Generate text descriptions from BabyAI observation grid.

    BabyAI observations contain a 7x7 image grid where:
    - Agent is at position (3, 6) facing forward (up)
    - Each cell is (object_type, color, state)

    Args:
        obs: Observation dict with 'image' key containing the grid

    Returns:
        List of description strings like "You see a red ball 2 steps ahead"
    """
    descriptions = []
    image = obs.get("image")

    if image is None:
        return ["You see nothing special."]

    agent_pos = (3, 6)  # Agent is at bottom center of 7x7 partial view

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            obj_type = int(image[i, j, 0])
            obj_color = int(image[i, j, 1])

            # Skip empty, unseen, wall, and floor tiles (types 0, 1, 2)
            if obj_type <= 2 or obj_type not in BABYAI_OBJECT_NAMES:
                continue

            obj_name = BABYAI_OBJECT_NAMES[obj_type]
            color_name = BABYAI_COLOR_NAMES.get(obj_color, "")

            # Calculate relative position from agent perspective
            # In the grid: i is column (left-right), j is row (top-bottom)
            # Agent faces up (negative j direction)
            dx = i - agent_pos[0]  # Positive = right
            dy = agent_pos[1] - j  # Positive = ahead (up in grid)

            parts = []

            if dy > 0:
                steps = "step" if dy == 1 else "steps"
                parts.append(f"{dy} {steps} ahead")
            elif dy < 0:
                steps = "step" if -dy == 1 else "steps"
                parts.append(f"{-dy} {steps} behind")

            if dx > 0:
                steps = "step" if dx == 1 else "steps"
                parts.append(f"{dx} {steps} to the right")
            elif dx < 0:
                steps = "step" if -dx == 1 else "steps"
                parts.append(f"{-dx} {steps} to the left")

            if parts:
                direction = " and ".join(parts)
                if color_name:
                    desc = f"You see a {color_name} {obj_name} {direction}"
                else:
                    desc = f"You see a {obj_name} {direction}"
                descriptions.append(desc)

    if not descriptions:
        descriptions = ["You see nothing special."]

    return descriptions


# =============================================================================
# Environment Wrapper that adds descriptions to info
# =============================================================================

class BabyAIDescriptionWrapper(gym.Wrapper):
    """Wrapper that adds 'descriptions' to info dict for BabyAI environments.

    BALROG's BabyAITextCleanLangWrapper expects info['descriptions'] but
    standard BabyAI/MiniGrid environments don't provide it. This wrapper
    generates descriptions from the observation grid.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        info["descriptions"] = generate_babyai_descriptions(obs)
        return obs, info

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["descriptions"] = generate_babyai_descriptions(obs)
        return obs, reward, terminated, truncated, info


# =============================================================================
# MultiGrid Text Description Generator (MOSAIC Extension)
# =============================================================================

def generate_multigrid_description(
    obs: np.ndarray,
    agent_id: int,
    env: Any,
    observation_mode: str = "visible_teammates",
    agent_direction: int = 0,
    carrying: Optional[str] = None,
) -> str:
    """Generate text description from MultiGrid observation using MOSAIC extension.

    Args:
        obs: Encoded grid observation (view_size, view_size, 6)
        agent_id: Current agent index
        env: MultiGrid environment instance
        observation_mode: "egocentric" or "visible_teammates"
        agent_direction: Agent's facing direction (0-3)
        carrying: What agent is carrying

    Returns:
        Natural language description of observation
    """
    # Import MOSAIC extension (now in llm_worker.environments.multigrid)
    from llm_worker.environments.multigrid import observations

    if observation_mode == "egocentric":
        return observations.describe_observation_egocentric(
            obs, agent_direction, carrying
        )
    else:  # visible_teammates
        # Extract teammate info
        agent_team = agent_id // 2  # Soccer: 0,1=team0, 2,3=team1
        visible_teammates = observations.extract_visible_teammates(
            env, agent_id, agent_team
        )
        return observations.describe_observation_with_teammates(
            obs, agent_id, visible_teammates, agent_direction, carrying
        )


def get_multigrid_instruction_prompt(
    agent_id: int,
    env_id: str,
    coordination_level: int = 1,
    role: Optional[str] = None,
) -> str:
    """Get MultiGrid instruction prompt using MOSAIC extension.

    Args:
        agent_id: Agent index
        env_id: Environment identifier (e.g., "MultiGrid-Soccer-v0")
        coordination_level: 1=Emergent, 2=Basic Hints, 3=Role-Based
        role: Agent role for Level 3 (e.g., "forward", "defender")

    Returns:
        Instruction prompt for the specified coordination level
    """
    # Import MOSAIC extension (now in llm_worker.environments.multigrid)
    from llm_worker.environments.multigrid import prompts

    # Determine team (Soccer: agents 0,1=team0, 2,3=team1)
    team = agent_id // 2 if "Soccer" in env_id else agent_id

    if coordination_level == 1:
        return prompts.get_instruction_prompt_level1(agent_id, team, env_id)
    elif coordination_level == 2:
        return prompts.get_instruction_prompt_level2(agent_id, team, env_id)
    elif coordination_level == 3:
        return prompts.get_instruction_prompt_level3(
            agent_id, team, role or "forward", env_id
        )
    else:
        logger.warning(f"Unknown coordination_level {coordination_level}, using Level 1")
        return prompts.get_instruction_prompt_level1(agent_id, team, env_id)


# =============================================================================
# Environment Factory
# =============================================================================

def make_babyai_env(
    task: str,
    render_mode: Optional[str] = None,
    **kwargs,
) -> gym.Env:
    """Create a BabyAI environment with description support.

    Args:
        task: Environment ID like "BabyAI-GoToRedBall-v0"
        render_mode: Rendering mode (None, "human", "rgb_array")
        **kwargs: Additional arguments for gym.make

    Returns:
        Wrapped environment with descriptions in info
    """
    import minigrid
    # Only register if not already in registry
    if "MiniGrid-Empty-5x5-v0" not in gym.envs.registry:
        minigrid.register_minigrid_envs()

    # Handle special MixedTrainLocal tasks
    if task.startswith("BabyAI-MixedTrainLocal-v0/"):
        base_task, goal = task.split("/")
        while True:
            env = gym.make(base_task, render_mode=render_mode, **kwargs)
            if env.unwrapped.action_kinds[0].replace(" ", "_") == goal:
                break
            env.close()
    else:
        env = gym.make(task, render_mode=render_mode, **kwargs)

    # Add our description wrapper
    env = BabyAIDescriptionWrapper(env)

    return env


def make_env(
    env_name: str,
    task: str,
    config: Any,
    render_mode: Optional[str] = None,
) -> Any:
    """Create a BALROG-compatible environment with our fixes.

    This function wraps BALROG's make_env but applies our fixes first
    for environments that need them.

    Args:
        env_name: Environment type (babyai, minigrid, minihack, etc.)
        task: Specific task/level
        config: OmegaConf config object
        render_mode: Rendering mode

    Returns:
        Wrapped environment ready for BALROG agents
    """
    import sys
    from pathlib import Path

    if env_name in ("babyai", "minigrid"):
        # Use our fixed BabyAI environment creation
        babyai_kwargs = {}
        if hasattr(config, "envs") and hasattr(config.envs, "babyai_kwargs"):
            babyai_kwargs = dict(config.envs.babyai_kwargs)

        base_env = make_babyai_env(task, render_mode=render_mode, **babyai_kwargs)

        # Now wrap with text wrapper (local llm_worker version)
        from llm_worker.environments.babyai_text import BabyAITextCleanLangWrapper
        from llm_worker.environments.env_wrapper import EnvWrapper

        wrapped_env = BabyAITextCleanLangWrapper(base_env, **babyai_kwargs)
        return EnvWrapper(wrapped_env, env_name, task)
    else:
        # For other environments, use llm_worker's make_env
        from llm_worker.environments import make_env as llm_worker_make_env
        return llm_worker_make_env(env_name, task, config, render_mode=render_mode)


__all__ = [
    "make_env",
    "make_babyai_env",
    "BabyAIDescriptionWrapper",
    "generate_babyai_descriptions",
]
