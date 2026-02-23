"""MeltingPot Instruction Prompts - Multi-Agent Social Scenarios.

MeltingPot tests multi-agent coordination in social dilemmas. This module
provides prompts for different scenario types:

- Cooperative: Agents must work together (e.g., collaborative cooking)
- Mixed-motive: Agents have aligned and conflicting goals
- Competitive: Agents compete for resources

Coordination Levels (matching MultiGrid pattern):
- Level 1 (Emergent): Minimal guidance, test emergent social behavior
- Level 2 (Basic Hints): Add cooperation/competition tips
- Level 3 (Role-Based): Explicit roles and social strategies
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .env import MELTINGPOT_ACTION_NAMES, MELTINGPOT_ACTION_DESCRIPTIONS

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Utilities
# =============================================================================

def format_action_list(actions: Dict[str, str]) -> str:
    """Format action dictionary as a readable list."""
    return "\n".join(
        f"  {action}: {description}"
        for action, description in actions.items()
    )


# =============================================================================
# Action Parser
# =============================================================================

def parse_action(llm_output: str) -> int:
    """Parse LLM text output to extract action index.

    Args:
        llm_output: Raw text output from LLM containing action name.

    Returns:
        Action index (0-7) or 0 (NOOP) if parsing fails.
    """
    llm_lower = llm_output.lower().strip()

    # Try exact match first
    for i, action in enumerate(MELTINGPOT_ACTION_NAMES):
        if action.lower() in llm_lower:
            logger.debug(f"Parsed action '{action}' (index {i}) from: {llm_output[:50]}")
            return i

    # Try common variations
    if "forward" in llm_lower or "ahead" in llm_lower:
        return 1
    if "backward" in llm_lower or "back" in llm_lower:
        return 2
    if "strafe left" in llm_lower:
        return 3
    if "strafe right" in llm_lower:
        return 4
    if "turn left" in llm_lower or "rotate left" in llm_lower:
        return 5
    if "turn right" in llm_lower or "rotate right" in llm_lower:
        return 6
    if "interact" in llm_lower or "use" in llm_lower or "activate" in llm_lower:
        return 7
    if "wait" in llm_lower or "nothing" in llm_lower or "stay" in llm_lower:
        return 0

    # Default to NOOP if no action found
    logger.warning(f"Failed to parse action, defaulting to 'NOOP': {llm_output[:100]}")
    return 0


# =============================================================================
# Substrate Categories
# =============================================================================

COOPERATIVE_SUBSTRATES = [
    "collaborative_cooking__asymmetric",
    "collaborative_cooking__circuit",
    "collaborative_cooking__crowded",
    "collaborative_cooking__figure_eight",
    "collaborative_cooking__forced",
    "collaborative_cooking__ring",
]

MIXED_MOTIVE_SUBSTRATES = [
    "prisoners_dilemma_in_the_matrix__arena",
    "prisoners_dilemma_in_the_matrix__repeated",
    "stag_hunt_in_the_matrix__arena",
    "clean_up",
    "territory__rooms",
]

COMPETITIVE_SUBSTRATES = [
    "capture_the_flag__arena",
    "king_of_the_hill",
    "paintball__capture_the_flag",
    "paintball__king_of_the_hill",
]


def get_substrate_category(substrate_name: str) -> str:
    """Determine the category of a substrate.

    Args:
        substrate_name: Name of the substrate.

    Returns:
        Category: "cooperative", "mixed_motive", or "competitive".
    """
    substrate_lower = substrate_name.lower()

    for coop in COOPERATIVE_SUBSTRATES:
        if coop in substrate_lower or substrate_lower in coop:
            return "cooperative"

    for comp in COMPETITIVE_SUBSTRATES:
        if comp in substrate_lower or substrate_lower in comp:
            return "competitive"

    return "mixed_motive"


# =============================================================================
# Level 1: Emergent Coordination
# =============================================================================

def get_instruction_prompt_level1(
    agent_id: str,
    substrate_name: str,
    num_agents: int = 2,
) -> str:
    """Generate Level 1 (Emergent) instruction prompt.

    Minimal guidance - let LLMs discover social strategies naturally.
    """
    action_strings = format_action_list(MELTINGPOT_ACTION_DESCRIPTIONS)
    category = get_substrate_category(substrate_name)

    if category == "cooperative":
        scenario_desc = "This is a cooperative scenario where working together benefits everyone."
    elif category == "competitive":
        scenario_desc = "This is a competitive scenario where you compete against others."
    else:
        scenario_desc = "This is a social scenario with both cooperative and competitive elements."

    return f"""You are {agent_id} in a multi-agent scenario: {substrate_name}

{scenario_desc}

There are {num_agents} agents in this environment.

Actions:
{action_strings}

Based on what you observe, respond with ONE action word (e.g., "FORWARD", "INTERACT", "TURN_LEFT")."""


# =============================================================================
# Level 2: Basic Hints
# =============================================================================

def get_instruction_prompt_level2(
    agent_id: str,
    substrate_name: str,
    num_agents: int = 2,
) -> str:
    """Generate Level 2 (Basic Hints) instruction prompt.

    Add social strategy hints based on scenario type.
    """
    action_strings = format_action_list(MELTINGPOT_ACTION_DESCRIPTIONS)
    category = get_substrate_category(substrate_name)

    if category == "cooperative":
        strategy_tips = """Tips for Cooperation:
- Coordinate actions with other agents
- Help teammates when they're stuck
- Share resources and opportunities
- Take turns if there's a bottleneck
- Collective success benefits everyone"""

    elif category == "competitive":
        strategy_tips = """Tips for Competition:
- Secure resources before opponents
- Control strategic positions
- Block opponents when advantageous
- Prioritize high-value objectives
- Watch opponent movements"""

    else:  # mixed_motive
        strategy_tips = """Tips for Mixed-Motive Scenarios:
- Balance self-interest with group benefit
- Build trust through consistent cooperation
- Retaliate against defection, but forgive
- Long-term cooperation often beats short-term gains
- Watch for free-riders"""

    return f"""You are {agent_id} in a multi-agent scenario: {substrate_name}

There are {num_agents} agents in this environment.

Actions:
{action_strings}

{strategy_tips}

Based on what you observe, respond with ONE action word (e.g., "FORWARD", "INTERACT", "TURN_LEFT")."""


# =============================================================================
# Level 3: Role-Based
# =============================================================================

def get_instruction_prompt_level3(
    agent_id: str,
    substrate_name: str,
    role: str,
    num_agents: int = 2,
) -> str:
    """Generate Level 3 (Role-Based) instruction prompt.

    Assign explicit roles with detailed social strategies.
    """
    action_strings = format_action_list(MELTINGPOT_ACTION_DESCRIPTIONS)
    category = get_substrate_category(substrate_name)

    role = role.lower()

    if category == "cooperative":
        if role == "leader":
            role_strategy = """Your Role: LEADER
- Coordinate team actions
- Direct teammates to tasks
- Handle complex decisions
- Ensure everyone contributes
- Optimize team efficiency"""
        else:  # helper
            role_strategy = """Your Role: HELPER
- Follow leader's coordination
- Support teammates in their tasks
- Fill gaps in team coverage
- Communicate through positioning
- Maximize collective output"""

    elif category == "competitive":
        if role == "aggressor":
            role_strategy = """Your Role: AGGRESSOR
- Actively pursue objectives
- Contest resources with opponents
- Apply pressure to enemies
- Take calculated risks
- Maintain offensive positioning"""
        else:  # defender
            role_strategy = """Your Role: DEFENDER
- Protect team resources
- Block opponent advances
- Maintain defensive positions
- Support aggressive teammates
- Control key areas"""

    else:  # mixed_motive
        if role == "cooperator":
            role_strategy = """Your Role: COOPERATOR
- Prioritize group benefit
- Build trust with others
- Forgive occasional defection
- Lead by example
- Encourage collective action"""
        else:  # pragmatist
            role_strategy = """Your Role: PRAGMATIST
- Balance self and group interest
- Cooperate when beneficial
- Defect against defectors
- Adapt to others' strategies
- Optimize personal outcome"""

    return f"""You are {agent_id} in a multi-agent scenario: {substrate_name}

There are {num_agents} agents in this environment.

{role_strategy}

Actions:
{action_strings}

Based on your role and observations, respond with ONE action word (e.g., "FORWARD", "INTERACT", "TURN_LEFT")."""


# =============================================================================
# Prompt Generator Class
# =============================================================================

class MeltingPotPromptGenerator:
    """Prompt generator for MeltingPot environments."""

    def __init__(self, substrate_name: str, num_agents: int = 2):
        """Initialize MeltingPot prompt generator.

        Args:
            substrate_name: Name of the substrate/scenario.
            num_agents: Number of agents in the environment.
        """
        self.substrate_name = substrate_name
        self.num_agents = num_agents
        self.category = get_substrate_category(substrate_name)

    def get_system_prompt(
        self,
        agent_id: str,
        coordination_level: int = 1,
        role: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate system prompt based on coordination level.

        Args:
            agent_id: Agent identifier.
            coordination_level: 1 (emergent), 2 (hints), or 3 (role-based).
            role: Role for level 3 coordination.
            **kwargs: Ignored.

        Returns:
            System prompt string.
        """
        if coordination_level == 1:
            return get_instruction_prompt_level1(
                agent_id, self.substrate_name, self.num_agents
            )
        elif coordination_level == 2:
            return get_instruction_prompt_level2(
                agent_id, self.substrate_name, self.num_agents
            )
        elif coordination_level == 3:
            if role is None:
                # Default role based on category
                if self.category == "cooperative":
                    role = "helper"
                elif self.category == "competitive":
                    role = "defender"
                else:
                    role = "pragmatist"
            return get_instruction_prompt_level3(
                agent_id, self.substrate_name, role, self.num_agents
            )
        else:
            logger.warning(f"Unknown coordination level {coordination_level}, using Level 1")
            return get_instruction_prompt_level1(
                agent_id, self.substrate_name, self.num_agents
            )

    def parse_action(self, llm_output: str) -> int:
        """Parse LLM output to action index."""
        return parse_action(llm_output)

    @property
    def action_space(self) -> List[str]:
        """Return MeltingPot action names."""
        return MELTINGPOT_ACTION_NAMES

    @property
    def action_descriptions(self) -> Dict[str, str]:
        """Return MeltingPot action descriptions."""
        return MELTINGPOT_ACTION_DESCRIPTIONS


__all__ = [
    # Utilities
    "format_action_list",
    "parse_action",
    # Substrate categories
    "COOPERATIVE_SUBSTRATES",
    "MIXED_MOTIVE_SUBSTRATES",
    "COMPETITIVE_SUBSTRATES",
    "get_substrate_category",
    # Instruction prompts
    "get_instruction_prompt_level1",
    "get_instruction_prompt_level2",
    "get_instruction_prompt_level3",
    # Prompt generator
    "MeltingPotPromptGenerator",
]
