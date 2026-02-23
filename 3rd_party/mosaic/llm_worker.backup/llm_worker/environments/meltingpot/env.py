"""MeltingPot Environment Helpers.

MeltingPot is a suite of test scenarios for multi-agent reinforcement learning
developed by Google DeepMind. It assesses generalization to novel social situations.

Repository: https://github.com/google-deepmind/meltingpot
Shimmy: https://shimmy.farama.org/environments/meltingpot/

This module provides:
- Environment creation for MeltingPot via Shimmy
- Observation description generators
- Action space definitions

NOTE: Linux/macOS only (Windows NOT supported)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Action Space
# =============================================================================

# MeltingPot action meanings (from dm_env spec)
MELTINGPOT_ACTION_NAMES: List[str] = [
    "NOOP",       # 0 - Do nothing
    "FORWARD",    # 1 - Move forward
    "BACKWARD",   # 2 - Move backward
    "LEFT",       # 3 - Strafe left
    "RIGHT",      # 4 - Strafe right
    "TURN_LEFT",  # 5 - Turn left
    "TURN_RIGHT", # 6 - Turn right
    "INTERACT",   # 7 - Interact/use
]

MELTINGPOT_ACTION_DESCRIPTIONS: Dict[str, str] = {
    "NOOP": "do nothing (stay in place)",
    "FORWARD": "move one step forward",
    "BACKWARD": "move one step backward",
    "LEFT": "strafe one step to the left",
    "RIGHT": "strafe one step to the right",
    "TURN_LEFT": "turn 90 degrees counter-clockwise",
    "TURN_RIGHT": "turn 90 degrees clockwise",
    "INTERACT": "interact with object/agent in front",
}


# =============================================================================
# Observation Descriptions
# =============================================================================

def describe_observation(
    obs: Dict[str, Any],
    agent_id: str,
    substrate_name: str = "",
) -> str:
    """Generate text description from MeltingPot observation.

    Args:
        obs: Observation dict with RGB and possibly COLLECTIVE_REWARD.
        agent_id: Agent identifier.
        substrate_name: Name of the substrate/scenario.

    Returns:
        Text description of the observation.
    """
    descriptions = [f"You are {agent_id} in a {substrate_name} scenario."]

    # Check for collective reward info
    if "COLLECTIVE_REWARD" in obs:
        collective_reward = obs["COLLECTIVE_REWARD"]
        if isinstance(collective_reward, (int, float)):
            descriptions.append(f"Collective reward: {collective_reward:.2f}")

    # RGB observations are images - describe what we can infer
    if "RGB" in obs:
        rgb = obs["RGB"]
        if hasattr(rgb, "shape"):
            h, w = rgb.shape[:2]
            descriptions.append(f"You see a {w}x{h} view of the environment.")

    descriptions.append("Analyze the visual information to decide your next action.")

    return " ".join(descriptions)


def describe_collective_reward(obs: Dict[str, Any]) -> str:
    """Extract and describe collective reward information.

    Args:
        obs: Observation dict.

    Returns:
        Description of collective reward.
    """
    if "COLLECTIVE_REWARD" in obs:
        reward = obs["COLLECTIVE_REWARD"]
        if reward > 0:
            return f"The group earned a collective reward of {reward:.2f}!"
        elif reward < 0:
            return f"The group received a collective penalty of {reward:.2f}."
    return ""


# =============================================================================
# Environment Factory
# =============================================================================

def make_meltingpot_env(
    substrate_name: str,
    render_mode: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create a MeltingPot environment via Shimmy.

    Args:
        substrate_name: Name of the MeltingPot substrate/scenario.
        render_mode: Rendering mode ("rgb_array", None).
        **kwargs: Additional environment kwargs.

    Returns:
        MeltingPot environment wrapped with Shimmy.

    Raises:
        ImportError: If Shimmy or MeltingPot is not installed.
    """
    try:
        from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0
    except ImportError:
        raise ImportError(
            "MeltingPot/Shimmy not installed. Install with:\n"
            "  pip install shimmy[meltingpot]\n"
            "Note: Linux/macOS only, Windows is NOT supported."
        )

    return MeltingPotCompatibilityV0(
        substrate_name=substrate_name,
        render_mode=render_mode,
        **kwargs,
    )


def list_substrates() -> List[str]:
    """List available MeltingPot substrates.

    Returns:
        List of substrate names.
    """
    try:
        from meltingpot.configs import substrates as substrate_configs
        return list(substrate_configs.SUBSTRATES.keys())
    except ImportError:
        logger.warning("MeltingPot not installed, cannot list substrates")
        return []


__all__ = [
    # Action space
    "MELTINGPOT_ACTION_NAMES",
    "MELTINGPOT_ACTION_DESCRIPTIONS",
    # Observations
    "describe_observation",
    "describe_collective_reward",
    # Environment factory
    "make_meltingpot_env",
    "list_substrates",
]
