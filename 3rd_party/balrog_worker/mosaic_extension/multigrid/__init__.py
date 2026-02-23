"""MOSAIC MultiGrid Extension for BALROG.

Multi-agent coordination strategies and Theory of Mind observations for LLMs.

This extension enables research on:
1. Multi-agent LLM coordination (3 levels: Emergent, Basic Hints, Role-Based)
2. Theory of Mind in multi-agent scenarios (Egocentric vs Visible Teammates)
3. Role-based team strategies (Forward/Defender assignments)

Environments supported:
- MultiGrid-Soccer-v0: 4 agents (2v2), zero-sum team game
- MultiGrid-Collect-v0: 3 agents, competitive ball collection

Example usage:
    from mosaic_extension.multigrid import prompts, observations

    # Level 1: Emergent coordination
    prompt = prompts.get_instruction_prompt_level1(agent_id=0, team=0, env_id="MultiGrid-Soccer-v0")

    # Egocentric observation
    obs_text = observations.describe_observation_egocentric(obs, agent_direction=0, carrying=None)
"""

from .prompts import (
    MULTIGRID_ACTION_SPACE,
    MULTIGRID_ACTIONS,
    parse_action,
    get_instruction_prompt_level1,
    get_instruction_prompt_level2,
    get_instruction_prompt_level3,
)

from .observations import (
    describe_observation_egocentric,
    describe_observation_with_teammates,
    extract_visible_teammates,
)

__all__ = [
    # Prompts
    "MULTIGRID_ACTION_SPACE",
    "MULTIGRID_ACTIONS",
    "parse_action",
    "get_instruction_prompt_level1",
    "get_instruction_prompt_level2",
    "get_instruction_prompt_level3",
    # Observations
    "describe_observation_egocentric",
    "describe_observation_with_teammates",
    "extract_visible_teammates",
]
