"""MeltingPot Environment Module.

MeltingPot is a suite of test scenarios for multi-agent reinforcement learning
developed by Google DeepMind for studying social behavior and coordination.

This module provides:
- Environment creation via Shimmy wrapper
- Observation description generators
- Instruction prompts with 3 coordination levels
- Action parsing

NOTE: Linux/macOS only (Windows NOT supported)
"""

from .env import (
    # Action space
    MELTINGPOT_ACTION_NAMES,
    MELTINGPOT_ACTION_DESCRIPTIONS,
    # Observations
    describe_observation,
    describe_collective_reward,
    # Environment factory
    make_meltingpot_env,
    list_substrates,
)

from .prompts import (
    # Utilities
    format_action_list,
    parse_action,
    # Substrate categories
    COOPERATIVE_SUBSTRATES,
    MIXED_MOTIVE_SUBSTRATES,
    COMPETITIVE_SUBSTRATES,
    get_substrate_category,
    # Instruction prompts
    get_instruction_prompt_level1,
    get_instruction_prompt_level2,
    get_instruction_prompt_level3,
    # Prompt generator
    MeltingPotPromptGenerator,
)


# Convenience function for instruction prompts
def get_instruction_prompt(
    agent_id: str,
    substrate_name: str,
    coordination_level: int = 1,
    role: str = None,
    num_agents: int = 2,
) -> str:
    """Get MeltingPot instruction prompt for specified coordination level.

    Args:
        agent_id: Agent identifier.
        substrate_name: Name of the substrate/scenario.
        coordination_level: 1 (emergent), 2 (hints), or 3 (role-based).
        role: Role for level 3 coordination.
        num_agents: Number of agents in the environment.

    Returns:
        Instruction prompt string.
    """
    if coordination_level == 1:
        return get_instruction_prompt_level1(agent_id, substrate_name, num_agents)
    elif coordination_level == 2:
        return get_instruction_prompt_level2(agent_id, substrate_name, num_agents)
    elif coordination_level == 3:
        return get_instruction_prompt_level3(
            agent_id, substrate_name, role or "helper", num_agents
        )
    else:
        return get_instruction_prompt_level1(agent_id, substrate_name, num_agents)


__all__ = [
    # Environment
    "make_meltingpot_env",
    "list_substrates",
    # Action space
    "MELTINGPOT_ACTION_NAMES",
    "MELTINGPOT_ACTION_DESCRIPTIONS",
    # Observations
    "describe_observation",
    "describe_collective_reward",
    # Prompts
    "MeltingPotPromptGenerator",
    "get_instruction_prompt",
    "get_instruction_prompt_level1",
    "get_instruction_prompt_level2",
    "get_instruction_prompt_level3",
    # Substrate categories
    "COOPERATIVE_SUBSTRATES",
    "MIXED_MOTIVE_SUBSTRATES",
    "COMPETITIVE_SUBSTRATES",
    "get_substrate_category",
    # Action parsing
    "parse_action",
    "format_action_list",
]
