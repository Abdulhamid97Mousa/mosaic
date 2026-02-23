"""MultiGrid Instruction Prompts - 3 Coordination Levels.

This module implements MOSAIC's research on multi-agent LLM coordination strategies.
Three levels are provided to study how explicit coordination guidance affects performance:

- Level 1 (Emergent): Minimal guidance, test emergent coordination
- Level 2 (Basic Hints): Add cooperation tips, balance emergence and guidance
- Level 3 (Role-Based): Explicit roles with detailed strategies

Research Questions:
- RQ1: How do coordination levels affect multi-agent performance?
- RQ2: Can LLMs coordinate effectively without explicit guidance?
- RQ3: Do role-based strategies improve team-based game outcomes?
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import structured logging from gym_gui (optional)
try:
    from gym_gui.logging_config.helpers import log_constant
    from gym_gui.logging_config.log_constants import (
        LOG_WORKER_MOSAIC_PROMPT_GENERATED,
        LOG_WORKER_MOSAIC_ACTION_PARSED,
        LOG_WORKER_MOSAIC_ACTION_PARSE_FAILED,
    )
    _HAS_STRUCTURED_LOGGING = True
except ImportError:
    log_constant = None
    _HAS_STRUCTURED_LOGGING = False


# =============================================================================
# Base Classes
# =============================================================================

# Generic action space for unknown environments
GENERIC_ACTION_SPACE = [
    "action_0",
    "action_1",
    "action_2",
    "action_3",
]


def format_action_list(actions: Dict[str, str]) -> str:
    """Format action dictionary as a readable list.

    Args:
        actions: Dictionary mapping action names to descriptions.

    Returns:
        Formatted string with each action on its own line.
    """
    return ",\n".join(
        f"  {action}: {description}"
        for action, description in actions.items()
    )


class BasePromptGenerator(ABC):
    """Abstract base class for prompt generators.

    Subclasses should implement environment-specific prompt generation
    for different coordination levels.
    """

    def __init__(self, env_id: str, num_agents: int = 2):
        """Initialize prompt generator.

        Args:
            env_id: Environment identifier.
            num_agents: Number of agents in the environment.
        """
        self.env_id = env_id
        self.num_agents = num_agents

    @abstractmethod
    def get_system_prompt(
        self,
        agent_id: int,
        coordination_level: int,
        role: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate system prompt for an agent."""
        pass

    @abstractmethod
    def format_observation(
        self,
        observation: Any,
        agent_id: int,
        **kwargs: Any,
    ) -> str:
        """Convert observation to text for LLM."""
        pass

    @abstractmethod
    def parse_action(self, llm_output: str) -> int:
        """Parse LLM output to action index."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> List[str]:
        """Return list of action names."""
        pass

    @property
    @abstractmethod
    def action_descriptions(self) -> Dict[str, str]:
        """Return dictionary mapping action names to descriptions."""
        pass


# =============================================================================
# MultiGrid Action Space
# =============================================================================

# MultiGrid action space (8 actions)
# Note: MultiGrid extends MiniGrid by adding STILL action at index 0
MULTIGRID_ACTION_SPACE = [
    "still",    # 0
    "left",     # 1
    "right",    # 2
    "forward",  # 3
    "pickup",   # 4
    "drop",     # 5
    "toggle",   # 6
    "done",     # 7
]

# Action descriptions for LLM instruction prompts
MULTIGRID_ACTIONS = {
    "still": "do nothing (wait in place)",
    "left": "turn left 90 degrees counter-clockwise",
    "right": "turn right 90 degrees clockwise",
    "forward": "move one step in facing direction",
    "pickup": "pick up ball (or steal from agent carrying ball)",
    "drop": "drop held ball (scores if at goal)",
    "toggle": "interact with object in front",
    "done": "signal completion (rarely used)",
}


# =============================================================================
# Action Parser
# =============================================================================

def parse_action(llm_output: str) -> int:
    """Parse LLM text output to extract action index.

    Args:
        llm_output: Raw text output from LLM containing action name.

    Returns:
        Action index (0-7) or 0 (still) if parsing fails.
    """
    llm_lower = llm_output.lower().strip()

    # Try exact match first
    for i, action in enumerate(MULTIGRID_ACTION_SPACE):
        if action in llm_lower:
            if _HAS_STRUCTURED_LOGGING and log_constant:
                log_constant(
                    logger,
                    LOG_WORKER_MOSAIC_ACTION_PARSED,
                    extra={
                        "action_name": action,
                        "action_index": i,
                        "llm_output": llm_output[:50],
                    },
                )
            else:
                logger.debug(f"Parsed action '{action}' (index {i}) from: {llm_output[:50]}")
            return i

    # Try common variations
    if "turn left" in llm_lower or "rotate left" in llm_lower:
        return 1
    if "turn right" in llm_lower or "rotate right" in llm_lower:
        return 2
    if "move forward" in llm_lower or "go forward" in llm_lower or "move ahead" in llm_lower:
        return 3
    if "pick up" in llm_lower or "pick-up" in llm_lower or "grab" in llm_lower:
        return 4
    if "wait" in llm_lower or "stay" in llm_lower or "nothing" in llm_lower:
        return 0

    # Default to still if no action found
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_ACTION_PARSE_FAILED,
            extra={"llm_output": llm_output[:100]},
        )
    else:
        logger.warning(f"Failed to parse action, defaulting to 'still': {llm_output[:100]}")
    return 0


# =============================================================================
# Level 1: Emergent Coordination
# =============================================================================

def get_instruction_prompt_level1(agent_id: int, team: int, env_id: str) -> str:
    """Generate Level 1 (Emergent) instruction prompt.

    Minimal guidance - let LLMs discover coordination naturally.
    """
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_PROMPT_GENERATED,
            extra={"coordination_level": 1, "agent_id": agent_id, "team": team, "env_id": env_id},
        )

    action_strings = format_action_list(MULTIGRID_ACTIONS)
    is_soccer = "Soccer" in env_id
    is_collect = "Collect" in env_id

    if is_soccer:
        team_name = "Red" if team == 0 else "Green"
        opponent_name = "Green" if team == 0 else "Red"

        return f"""You are Agent {agent_id} playing a 2v2 soccer game.

Your team: {team_name} (Agents {team * 2}, {team * 2 + 1})
Opponent team: {opponent_name}

Actions:
{action_strings}

Goal: Score by carrying the ball to the opponent's goal.

Game Mechanics:
- Pick up the ball by walking to it and using "pickup"
- You can steal the ball from opponents
- Score by dropping the ball at the opponent's goal
- Pass to teammate by dropping the ball near them

Based on your observation, respond with ONE action word (e.g., "forward", "left", "pickup")."""

    elif is_collect:
        colors = ["Red", "Green", "Blue"]
        color = colors[agent_id] if agent_id < 3 else "Unknown"

        return f"""You are Agent {agent_id} ({color}) playing a ball collection game.

Your color: {color}
Opponents: {', '.join(c for i, c in enumerate(colors) if i != agent_id)}

Actions:
{action_strings}

Goal: Collect as many balls as possible before your opponents.

Game Mechanics:
- Walk to balls and use "pickup" to collect them
- Balls disappear once collected
- Race against opponents

Based on your observation, respond with ONE action word (e.g., "forward", "left", "pickup")."""

    else:
        return f"""You are Agent {agent_id} in a multi-agent grid world.

Actions:
{action_strings}

Based on your observation, respond with ONE action word (e.g., "forward", "left", "pickup")."""


# =============================================================================
# Level 2: Basic Hints
# =============================================================================

def get_instruction_prompt_level2(agent_id: int, team: int, env_id: str) -> str:
    """Generate Level 2 (Basic Hints) instruction prompt.

    Add cooperation tips to guide LLMs toward teamwork.
    """
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_PROMPT_GENERATED,
            extra={"coordination_level": 2, "agent_id": agent_id, "team": team, "env_id": env_id},
        )

    action_strings = format_action_list(MULTIGRID_ACTIONS)
    is_soccer = "Soccer" in env_id
    is_collect = "Collect" in env_id

    if is_soccer:
        team_name = "Red" if team == 0 else "Green"
        opponent_name = "Green" if team == 0 else "Red"

        return f"""You are Agent {agent_id} playing a 2v2 soccer game.

Your team: {team_name} (Agents {team * 2}, {team * 2 + 1})
Opponent team: {opponent_name}

Actions:
{action_strings}

Goal: Score by carrying the ball to the opponent's goal.

Game Mechanics:
- Pick up the ball by walking to it and using "pickup"
- You can steal the ball from opponents
- Score by dropping the ball at the opponent's goal
- Pass to teammate by dropping the ball near them

Tips for Coordination:
- Spread out to cover more area - don't cluster together
- If your teammate has the ball, position yourself to receive a pass
- If an opponent has the ball, try to intercept or block their path
- One player should focus on attacking, the other on supporting
- Don't all chase the ball - maintain team positioning

Based on your observation, respond with ONE action word (e.g., "forward", "left", "pickup")."""

    elif is_collect:
        colors = ["Red", "Green", "Blue"]
        color = colors[agent_id] if agent_id < 3 else "Unknown"

        return f"""You are Agent {agent_id} ({color}) playing a ball collection game.

Your color: {color}
Opponents: {', '.join(c for i, c in enumerate(colors) if i != agent_id)}

Actions:
{action_strings}

Goal: Collect as many balls as possible before your opponents.

Game Mechanics:
- Walk to balls and use "pickup" to collect them
- Balls disappear once collected
- Race against opponents

Tips for Competition:
- Target the nearest uncollected ball
- Watch opponent movements and predict their targets
- If an opponent is closer to a ball, switch to a different target
- Move quickly and decisively

Based on your observation, respond with ONE action word (e.g., "forward", "left", "pickup")."""

    else:
        return f"""You are Agent {agent_id} in a multi-agent grid world.

Actions:
{action_strings}

Tips:
- Coordinate with visible teammates when possible
- Avoid repeating the same action if observation doesn't change

Based on your observation, respond with ONE action word (e.g., "forward", "left", "pickup")."""


# =============================================================================
# Level 3: Role-Based
# =============================================================================

def get_instruction_prompt_level3(agent_id: int, team: int, role: str, env_id: str) -> str:
    """Generate Level 3 (Role-Based) instruction prompt.

    Assign explicit roles (Forward/Defender) with detailed strategies.
    """
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_PROMPT_GENERATED,
            extra={"coordination_level": 3, "agent_id": agent_id, "team": team, "role": role, "env_id": env_id},
        )

    action_strings = format_action_list(MULTIGRID_ACTIONS)
    is_soccer = "Soccer" in env_id

    if not is_soccer:
        return get_instruction_prompt_level2(agent_id, team, env_id)

    team_name = "Red" if team == 0 else "Green"
    opponent_name = "Green" if team == 0 else "Red"
    teammate_id = 1 if agent_id == 0 else 0 if agent_id in [0, 1] else 3 if agent_id == 2 else 2
    teammate_role = "DEFENDER" if role.lower() == "forward" else "FORWARD"

    if role.lower() == "forward":
        role_strategy = """Your Role: FORWARD (Offensive)
- Priority: Score goals, pressure opponents
- If you have ball: advance toward opponent goal
- If teammate has ball: move to open space ahead for pass
- If opponent has ball: pressure the ball carrier
- Stay in attacking half when possible
- Take risks to create scoring opportunities"""
    else:
        role_strategy = """Your Role: DEFENDER (Defensive)
- Priority: Protect goal, support attacks
- If you have ball: pass forward or advance carefully
- If teammate has ball: cover defensive position
- If opponent has ball: fall back, block goal path
- Stay between ball and your goal
- Be conservative - don't leave goal undefended"""

    return f"""You are Agent {agent_id} playing a 2v2 soccer game.

Your team: {team_name} (Agents {team * 2}, {team * 2 + 1})
Opponent team: {opponent_name}

{role_strategy}

Actions:
{action_strings}

Goal: Score by carrying the ball to the opponent's goal.

Game Mechanics:
- Pick up the ball by walking to it and using "pickup"
- You can steal the ball from opponents
- Score by dropping the ball at the opponent's goal
- Pass to teammate by dropping the ball near them

Coordination Protocol:
- You are the {role.upper()}, your teammate (Agent {teammate_id}) is the {teammate_role}
- Trust your teammate to handle their role
- Adjust your position based on visible teammate location
- Don't overlap roles - maintain positional discipline

Based on your observation, respond with ONE action word (e.g., "forward", "left", "pickup")."""


# =============================================================================
# Prompt Generator Class
# =============================================================================

class MultiGridPromptGenerator(BasePromptGenerator):
    """Prompt generator for MultiGrid environments."""

    def __init__(self, env_id: str, num_agents: int = 4):
        """Initialize MultiGrid prompt generator.

        Args:
            env_id: Environment identifier (e.g., "MultiGrid-Soccer-v0").
            num_agents: Number of agents (default 4 for Soccer).
        """
        super().__init__(env_id, num_agents)
        self._team_map = self._compute_team_map()

    def _compute_team_map(self) -> Dict[int, int]:
        """Compute agent-to-team mapping."""
        if "Soccer" in self.env_id:
            return {0: 0, 1: 0, 2: 1, 3: 1}
        return {i: i for i in range(self.num_agents)}

    def get_team(self, agent_id: int) -> int:
        """Get team for an agent."""
        return self._team_map.get(agent_id, agent_id)

    def get_system_prompt(
        self,
        agent_id: int,
        coordination_level: int,
        role: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate system prompt based on coordination level."""
        team = self.get_team(agent_id)

        if coordination_level == 1:
            return get_instruction_prompt_level1(agent_id, team, self.env_id)
        elif coordination_level == 2:
            return get_instruction_prompt_level2(agent_id, team, self.env_id)
        elif coordination_level == 3:
            if role is None:
                role = "forward" if (agent_id % 2) == 0 else "defender"
            return get_instruction_prompt_level3(agent_id, team, role, self.env_id)
        else:
            logger.warning(f"Unknown coordination level {coordination_level}, using Level 1")
            return get_instruction_prompt_level1(agent_id, team, self.env_id)

    def format_observation(
        self,
        observation: Any,
        agent_id: int,
        **kwargs: Any,
    ) -> str:
        """Format observation as text."""
        from .env import describe_observation_egocentric
        return describe_observation_egocentric(
            observation,
            agent_direction=kwargs.get("agent_direction", 0),
            carrying=kwargs.get("carrying"),
        )

    def parse_action(self, llm_output: str) -> int:
        """Parse LLM output to action index."""
        return parse_action(llm_output)

    @property
    def action_space(self) -> List[str]:
        """Return MultiGrid action names."""
        return MULTIGRID_ACTION_SPACE

    @property
    def action_descriptions(self) -> Dict[str, str]:
        """Return MultiGrid action descriptions."""
        return MULTIGRID_ACTIONS


__all__ = [
    # Base classes
    "BasePromptGenerator",
    "GENERIC_ACTION_SPACE",
    "format_action_list",
    # MultiGrid specific
    "MULTIGRID_ACTION_SPACE",
    "MULTIGRID_ACTIONS",
    "parse_action",
    "get_instruction_prompt_level1",
    "get_instruction_prompt_level2",
    "get_instruction_prompt_level3",
    "MultiGridPromptGenerator",
]
