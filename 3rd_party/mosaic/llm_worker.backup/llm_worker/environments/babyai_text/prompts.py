"""Prompt generation for BabyAI/MiniGrid environments.

This module provides LLM prompts compatible with BALROG benchmark style,
adapted for MOSAIC LLM Worker's single-agent MiniGrid support.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# =============================================================================
# BabyAI Action Space (BALROG-compatible)
# =============================================================================

# Action space matching BALROG's BabyAI implementation
BABYAI_ACTION_SPACE = [
    "turn left",
    "turn right",
    "forward",  # Note: BALROG uses "go forward" but we simplify
    "pickup",
    "drop",
    "toggle",
]

# Action descriptions for instruction prompts
BABYAI_ACTION_DESCRIPTIONS = {
    "turn left": "rotate 90 degrees counter-clockwise",
    "turn right": "rotate 90 degrees clockwise",
    "forward": "move one step in facing direction",
    "pickup": "pick up object in front of you",
    "drop": "drop held object",
    "toggle": "interact with object in front (open door, etc.)",
}


# =============================================================================
# Instruction Prompts
# =============================================================================

def get_babyai_system_prompt(mission: str = "") -> str:
    """Generate system prompt for BabyAI/MiniGrid environments.

    Args:
        mission: The mission/goal text from the environment.

    Returns:
        System prompt string for the LLM.
    """
    action_list = "\n".join(
        f"- {action}: {desc}"
        for action, desc in BABYAI_ACTION_DESCRIPTIONS.items()
    )

    prompt = f"""You are an agent navigating a grid world. Your goal: {mission if mission else "reach the green goal square"}.

## Available Actions
{action_list}

## Observation Format
You will receive text descriptions of what you can see from your current position.

## Output Format
Respond with ONLY the action name. Examples:
- forward
- turn left
- turn right
- pickup

Do NOT include explanations, just output the action."""

    return prompt


def get_babyai_instruction_prompt() -> str:
    """Get the basic instruction for BabyAI environments."""
    return """Navigate the grid to complete your mission. You see objects relative to your position and facing direction.

Available actions: turn left, turn right, forward, pickup, drop, toggle

Respond with ONLY the action name."""


# =============================================================================
# Observation Formatting
# =============================================================================

def format_babyai_observation(
    observation: Any,
    info: Dict[str, Any] = None,
    include_mission: bool = True,
) -> str:
    """Format a BabyAI/MiniGrid observation for the LLM.

    This creates a text description from the observation and info dict,
    similar to BALROG's approach.

    Args:
        observation: The observation dict from MiniGrid.
        info: The info dict containing 'descriptions' from BabyAIDescriptionWrapper.
        include_mission: Whether to include mission text.

    Returns:
        Formatted text observation for the LLM.
    """
    parts = []

    # Add mission if available
    if include_mission:
        mission = None
        if isinstance(observation, dict) and "mission" in observation:
            mission = observation["mission"]
        if mission:
            parts.append(f"Goal: {mission}")

    # Add descriptions from info (from BabyAIDescriptionWrapper)
    if info and "descriptions" in info:
        descriptions = info["descriptions"]
        if descriptions:
            # Format descriptions nicely
            desc_text = "\n".join(f"- {d}" for d in descriptions)
            parts.append(f"You see:\n{desc_text}")
    elif isinstance(observation, dict):
        # Fallback: try to extract from observation text
        if "text" in observation:
            text = observation["text"]
            if isinstance(text, dict):
                if "long_term_context" in text:
                    parts.append(f"Surroundings: {text['long_term_context']}")
            else:
                parts.append(f"Surroundings: {text}")

    if not parts:
        # Minimal fallback
        if isinstance(observation, dict) and "mission" in observation:
            parts.append(observation["mission"])
        else:
            parts.append("You see nothing special.")

    return "\n\n".join(parts)


# =============================================================================
# Action Parsing
# =============================================================================

def parse_babyai_action(llm_output: str) -> int:
    """Parse LLM output to BabyAI action index.

    Args:
        llm_output: Raw text from LLM.

    Returns:
        Action index (0-5), defaults to 2 (forward) on parse failure.
    """
    output_lower = llm_output.lower().strip()

    # Direct match
    for i, action in enumerate(BABYAI_ACTION_SPACE):
        if action == output_lower:
            logger.debug(f"LOG1041 MOSAIC BabyAI action parsed: exact match '{action}'")
            return i

    # Partial match
    for i, action in enumerate(BABYAI_ACTION_SPACE):
        if action in output_lower:
            logger.debug(f"LOG1041 MOSAIC BabyAI action parsed: partial match '{action}'")
            return i

    # Handle "go forward" -> "forward"
    if "go forward" in output_lower or "move forward" in output_lower:
        logger.debug("LOG1041 MOSAIC BabyAI action parsed: 'forward' from go/move forward")
        return 2  # forward

    # Handle "left"/"right" without "turn"
    if "left" in output_lower and "turn" not in output_lower:
        logger.debug("LOG1041 MOSAIC BabyAI action parsed: 'turn left' from 'left'")
        return 0  # turn left
    if "right" in output_lower and "turn" not in output_lower:
        logger.debug("LOG1041 MOSAIC BabyAI action parsed: 'turn right' from 'right'")
        return 1  # turn right

    # Default to forward (most common exploration action)
    logger.warning(f"Could not parse action from: '{llm_output[:50]}', defaulting to forward")
    return 2  # forward


# =============================================================================
# Prompt Generator Class
# =============================================================================

class BabyAIPromptGenerator:
    """Prompt generator for BabyAI/MiniGrid environments.

    Compatible with MOSAIC LLM Worker's prompt generator interface.
    """

    def __init__(self, task: str = "", **kwargs):
        """Initialize BabyAI prompt generator.

        Args:
            task: Task/environment name (e.g., "MiniGrid-Empty-8x8-v0").
        """
        self.task = task
        self._mission = ""
        self._info: Dict[str, Any] = {}

    def get_system_prompt(
        self,
        agent_id: int = 0,
        coordination_level: int = 1,
        role: str = None,
        **kwargs,
    ) -> str:
        """Generate system prompt for an agent.

        Args:
            agent_id: Agent ID (always 0 for single-agent).
            coordination_level: Ignored for single-agent.
            role: Optional role description.

        Returns:
            System prompt string.
        """
        return get_babyai_system_prompt(self._mission)

    def format_observation(
        self,
        observation: Any,
        agent_id: int = 0,
        info: Dict[str, Any] = None,
        **kwargs,
    ) -> str:
        """Convert observation to text for LLM.

        Args:
            observation: MiniGrid observation dict.
            agent_id: Agent ID (always 0 for single-agent).
            info: Info dict with 'descriptions' from wrapper.

        Returns:
            Formatted observation text.
        """
        # Store for later use
        if info:
            self._info = info
        if isinstance(observation, dict) and "mission" in observation:
            self._mission = observation["mission"]

        return format_babyai_observation(observation, info or self._info)

    def parse_action(self, llm_output: str) -> int:
        """Parse LLM output to action index.

        Args:
            llm_output: Raw LLM text output.

        Returns:
            Action index (0-5).
        """
        return parse_babyai_action(llm_output)

    @property
    def action_space(self) -> List[str]:
        """Return list of action names."""
        return BABYAI_ACTION_SPACE

    @property
    def action_descriptions(self) -> Dict[str, str]:
        """Return action descriptions."""
        return BABYAI_ACTION_DESCRIPTIONS


__all__ = [
    "BABYAI_ACTION_SPACE",
    "BABYAI_ACTION_DESCRIPTIONS",
    "get_babyai_system_prompt",
    "get_babyai_instruction_prompt",
    "format_babyai_observation",
    "parse_babyai_action",
    "BabyAIPromptGenerator",
]
