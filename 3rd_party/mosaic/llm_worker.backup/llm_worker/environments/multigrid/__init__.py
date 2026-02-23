"""MultiGrid Environment Module.

This module provides complete support for MultiGrid multi-agent environments:
- Environment creation and helpers
- Observation description generators
- Instruction prompts with 3 coordination levels
- Action parsing
"""

from .env import (
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
)

from .prompts import (
    # Base classes
    BasePromptGenerator,
    GENERIC_ACTION_SPACE,
    format_action_list,
    # Action space
    MULTIGRID_ACTION_SPACE,
    MULTIGRID_ACTIONS,
    # Action parsing
    parse_action,
    # Instruction prompts
    get_instruction_prompt_level1,
    get_instruction_prompt_level2,
    get_instruction_prompt_level3,
    # Prompt generator
    MultiGridPromptGenerator,
)

# Convenience function for instruction prompts
def get_instruction_prompt(
    agent_id: int,
    env_id: str,
    coordination_level: int = 1,
    role: str = None,
) -> str:
    """Get MultiGrid instruction prompt for specified coordination level.

    Args:
        agent_id: Agent index.
        env_id: Environment identifier (e.g., "MultiGrid-Soccer-v0").
        coordination_level: 1 (emergent), 2 (hints), or 3 (role-based).
        role: Role for level 3 ("forward" or "defender").

    Returns:
        Instruction prompt string.
    """
    team = agent_id // 2 if "Soccer" in env_id else 0

    if coordination_level == 1:
        return get_instruction_prompt_level1(agent_id, team, env_id)
    elif coordination_level == 2:
        return get_instruction_prompt_level2(agent_id, team, env_id)
    elif coordination_level == 3:
        return get_instruction_prompt_level3(agent_id, team, role or "forward", env_id)
    else:
        return get_instruction_prompt_level1(agent_id, team, env_id)


__all__ = [
    # Environment
    "make_multigrid_env",
    "MULTIGRID_OBJECT_NAMES",
    "MULTIGRID_COLOR_NAMES",
    "MULTIGRID_DIRECTION_NAMES",
    # Observations
    "describe_observation_egocentric",
    "describe_observation_with_teammates",
    "extract_visible_teammates",
    "generate_multigrid_description",
    # Prompts
    "BasePromptGenerator",
    "MultiGridPromptGenerator",
    "get_instruction_prompt",
    "get_instruction_prompt_level1",
    "get_instruction_prompt_level2",
    "get_instruction_prompt_level3",
    # Actions
    "MULTIGRID_ACTION_SPACE",
    "MULTIGRID_ACTIONS",
    "parse_action",
    "format_action_list",
    "GENERIC_ACTION_SPACE",
]
