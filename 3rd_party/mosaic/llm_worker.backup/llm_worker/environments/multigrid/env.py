"""MultiGrid Environment Helpers.

This module provides:
- Environment creation for MultiGrid
- Observation description generators (egocentric and with teammates)
- Theory of Mind teammate extraction
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Object and Color Constants
# =============================================================================

MULTIGRID_OBJECT_NAMES = {
    0: "unseen",
    1: "empty",
    2: "wall",
    3: "floor",
    4: "door",
    5: "key",
    6: "ball",
    7: "box",
    8: "goal",
    9: "lava",
    10: "agent",
}

MULTIGRID_COLOR_NAMES = {
    0: "red",
    1: "green",
    2: "blue",
    3: "purple",
    4: "yellow",
    5: "grey",
}

MULTIGRID_DIRECTION_NAMES = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
}


# =============================================================================
# Observation Description Generators
# =============================================================================

def describe_observation_egocentric(
    obs: np.ndarray,
    agent_direction: int = 0,
    carrying: Optional[str] = None,
) -> str:
    """Generate egocentric description from MultiGrid observation.

    Args:
        obs: Observation array from MultiGrid environment.
        agent_direction: Direction agent is facing (0-3).
        carrying: Description of carried object, if any.

    Returns:
        Text description of what the agent sees.
    """
    if obs is None or not hasattr(obs, 'shape'):
        return "You cannot see anything."

    descriptions = []

    if carrying:
        descriptions.append(f"You are carrying a {carrying}.")

    direction_name = MULTIGRID_DIRECTION_NAMES.get(agent_direction, "unknown")
    descriptions.append(f"You are facing {direction_name}.")

    view_size = obs.shape[0]
    agent_pos = (view_size // 2, view_size - 1)

    objects_seen = []
    for i in range(view_size):
        for j in range(view_size):
            obj_type = int(obs[i, j, 0])

            if obj_type in (0, 1, 3):
                continue

            obj_name = MULTIGRID_OBJECT_NAMES.get(obj_type, f"object_{obj_type}")
            color_idx = int(obs[i, j, 1]) if obs.shape[2] > 1 else 0
            color_name = MULTIGRID_COLOR_NAMES.get(color_idx, "")

            dx = i - agent_pos[0]
            dy = agent_pos[1] - j

            position_parts = []
            if dy > 0:
                position_parts.append(f"{dy} step{'s' if dy > 1 else ''} ahead")
            elif dy < 0:
                position_parts.append(f"{-dy} step{'s' if -dy > 1 else ''} behind")
            if dx > 0:
                position_parts.append(f"{dx} step{'s' if dx > 1 else ''} to the right")
            elif dx < 0:
                position_parts.append(f"{-dx} step{'s' if -dx > 1 else ''} to the left")

            if position_parts:
                position = " and ".join(position_parts)
                if color_name:
                    objects_seen.append(f"a {color_name} {obj_name} {position}")
                else:
                    objects_seen.append(f"a {obj_name} {position}")

    if objects_seen:
        descriptions.append("You see: " + "; ".join(objects_seen) + ".")
    else:
        descriptions.append("You see nothing notable in front of you.")

    return " ".join(descriptions)


def extract_visible_teammates(
    env: Any,
    agent_id: int,
    agent_team: int,
) -> List[Dict[str, Any]]:
    """Extract visible teammates from environment state.

    Args:
        env: MultiGrid environment instance.
        agent_id: Current agent's ID.
        agent_team: Current agent's team.

    Returns:
        List of teammate info dicts with id, position, direction.
    """
    visible_teammates = []

    try:
        agents = getattr(env, "agents", [])
        if not agents:
            agents = getattr(env.unwrapped, "agents", [])

        for i, agent in enumerate(agents):
            if i == agent_id:
                continue

            other_team = i // 2
            if other_team != agent_team:
                continue

            pos = getattr(agent, "pos", None)
            if pos is not None:
                visible_teammates.append({
                    "id": i,
                    "position": f"at ({pos[0]}, {pos[1]})",
                    "direction": getattr(agent, "dir", 0),
                })
    except Exception as e:
        logger.debug(f"Could not extract teammates: {e}")

    return visible_teammates


def describe_observation_with_teammates(
    obs: np.ndarray,
    agent_id: int,
    visible_teammates: List[Dict[str, Any]],
    agent_direction: int = 0,
    carrying: Optional[str] = None,
) -> str:
    """Generate description with Theory of Mind teammate awareness.

    Args:
        obs: Observation array.
        agent_id: Current agent's ID.
        visible_teammates: List of visible teammate info.
        agent_direction: Direction agent is facing.
        carrying: Description of carried object.

    Returns:
        Text description including teammate information.
    """
    base_description = describe_observation_egocentric(obs, agent_direction, carrying)

    teammate_descriptions = []
    for teammate in visible_teammates:
        teammate_id = teammate.get("id", "unknown")
        teammate_pos = teammate.get("position", "nearby")
        teammate_action = teammate.get("last_action")

        desc = f"Teammate {teammate_id} is {teammate_pos}"
        if teammate_action:
            desc += f" (last action: {teammate_action})"
        teammate_descriptions.append(desc)

    if teammate_descriptions:
        return base_description + " " + " ".join(teammate_descriptions) + "."

    return base_description


def generate_multigrid_description(
    obs: np.ndarray,
    agent_id: int,
    env: Optional[Any] = None,
    observation_mode: str = "visible_teammates",
    agent_direction: int = 0,
    carrying: Optional[str] = None,
) -> str:
    """Generate text description from MultiGrid observation.

    Args:
        obs: Observation array from MultiGrid.
        agent_id: Current agent's ID.
        env: Environment instance (for teammate extraction).
        observation_mode: "egocentric" or "visible_teammates".
        agent_direction: Direction agent is facing (0-3).
        carrying: Description of carried object.

    Returns:
        Text description of observation.
    """
    if observation_mode == "egocentric":
        return describe_observation_egocentric(obs, agent_direction, carrying)
    else:
        visible_teammates = []
        if env is not None:
            agent_team = agent_id // 2
            visible_teammates = extract_visible_teammates(env, agent_id, agent_team)

        return describe_observation_with_teammates(
            obs, agent_id, visible_teammates, agent_direction, carrying
        )


# =============================================================================
# Environment Factory
# =============================================================================

def make_multigrid_env(
    task: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create a MultiGrid environment.

    Args:
        task: Environment ID (e.g., "MultiGrid-Soccer-v0").
        render_mode: Rendering mode ("human", "rgb_array", None).
        **kwargs: Additional environment kwargs.

    Returns:
        MultiGrid environment instance.

    Raises:
        ImportError: If MultiGrid is not installed.
    """
    import gymnasium as gym

    try:
        from multigrid.envs import register_all
        register_all()
    except ImportError:
        raise ImportError(
            "MultiGrid not installed. Install with: pip install -e 3rd_party/gym-multigrid/"
        )

    return gym.make(task, render_mode=render_mode, **kwargs)


__all__ = [
    # Constants
    "MULTIGRID_OBJECT_NAMES",
    "MULTIGRID_COLOR_NAMES",
    "MULTIGRID_DIRECTION_NAMES",
    # Observation descriptions
    "describe_observation_egocentric",
    "describe_observation_with_teammates",
    "extract_visible_teammates",
    "generate_multigrid_description",
    # Environment factory
    "make_multigrid_env",
]
