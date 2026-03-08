"""MOSAIC MultiGrid Instruction Prompts - 3 Coordination Levels.

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

# MultiGrid action space (8 actions)
# Note: MultiGrid extends MiniGrid by adding STILL action at index 0
# These are the EXACT action names that MultiGrid expects
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
    "left": "turn left 90° counter-clockwise",
    "right": "turn right 90° clockwise",
    "forward": "move one step in facing direction",
    "pickup": "pick up ball (or steal from agent carrying ball)",
    "drop": "drop held ball (scores if at goal)",
    "toggle": "interact with object in front",
    "done": "signal completion (rarely used)",
}


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
                logger.debug(f"MOSAIC: Parsed action '{action}' (index {i}) from LLM output: {llm_output[:50]}")
            return i

    # Try common variations
    def _log_action(action_name: str, action_index: int):
        """Helper to log action parsing with structured logging if available."""
        if _HAS_STRUCTURED_LOGGING and log_constant:
            log_constant(
                logger,
                LOG_WORKER_MOSAIC_ACTION_PARSED,
                extra={
                    "action_name": action_name,
                    "action_index": action_index,
                    "llm_output": llm_output[:50],
                },
            )
        else:
            logger.debug(f"MOSAIC: Parsed action '{action_name}' (index {action_index}) from LLM output: {llm_output[:50]}")

    if "turn left" in llm_lower or "rotate left" in llm_lower:
        _log_action("left", 1)
        return 1
    if "turn right" in llm_lower or "rotate right" in llm_lower:
        _log_action("right", 2)
        return 2
    if "move forward" in llm_lower or "go forward" in llm_lower or "move ahead" in llm_lower:
        _log_action("forward", 3)
        return 3
    if "pick up" in llm_lower or "pick-up" in llm_lower or "grab" in llm_lower:
        _log_action("pickup", 4)
        return 4
    if "wait" in llm_lower or "stay" in llm_lower or "nothing" in llm_lower:
        _log_action("still", 0)
        return 0

    # Default to still if no action found
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_ACTION_PARSE_FAILED,
            extra={"llm_output": llm_output[:100]},
        )
    else:
        logger.warning(f"MOSAIC: Failed to parse action from LLM output, defaulting to 'still': {llm_output[:100]}")
    return 0


def get_instruction_prompt_level1(agent_id: int, team: int, env_id: str) -> str:
    """Generate Level 1 (Emergent) instruction prompt.

    Minimal guidance - let LLMs discover coordination naturally.
    Tests emergent multi-agent behavior without explicit cooperation strategies.

    Args:
        agent_id: Agent index (0-3 for Soccer, 0-2 for Collect).
        team: Team index (0 or 1 for Soccer).
        env_id: Environment identifier.

    Returns:
        Level 1 instruction prompt.
    """
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_PROMPT_GENERATED,
            extra={
                "coordination_level": 1,
                "agent_id": agent_id,
                "team": team,
                "env_id": env_id,
            },
        )
    else:
        logger.debug(
            f"MOSAIC: Generating Level 1 prompt | agent={agent_id} team={team} env={env_id}"
        )

    action_strings = ",\n".join(
        f"  {action}: {description}"
        for action, description in MULTIGRID_ACTIONS.items()
    )

    is_soccer = "Soccer" in env_id
    is_collect = "Collect" in env_id

    if is_soccer:
        team_name = "Red" if team == 0 else "Green"
        opponent_name = "Green" if team == 0 else "Red"

        return f"""
You are Agent {agent_id} playing a 2v2 soccer game.

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

PLAY!
""".strip()

    elif is_collect:
        colors = ["Red", "Green", "Blue"]
        color = colors[agent_id] if agent_id < 3 else "Unknown"

        return f"""
You are Agent {agent_id} ({color}) playing a ball collection game.

Your color: {color}
Opponents: {', '.join(c for i, c in enumerate(colors) if i != agent_id)}

Actions:
{action_strings}

Goal: Collect as many balls as possible before your opponents.

Game Mechanics:
- Walk to balls and use "pickup" to collect them
- Balls disappear once collected
- Race against opponents

PLAY!
""".strip()

    else:
        return f"""
You are Agent {agent_id} in a multi-agent grid world.

Actions:
{action_strings}

PLAY!
""".strip()


def get_instruction_prompt_level2(agent_id: int, team: int, env_id: str) -> str:
    """Generate Level 2 (Basic Hints) instruction prompt.

    Add cooperation tips to guide LLMs toward teamwork without being prescriptive.
    Balances emergent behavior with helpful coordination guidance.

    Args:
        agent_id: Agent index.
        team: Team index.
        env_id: Environment identifier.

    Returns:
        Level 2 instruction prompt.
    """
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_PROMPT_GENERATED,
            extra={
                "coordination_level": 2,
                "agent_id": agent_id,
                "team": team,
                "env_id": env_id,
            },
        )
    else:
        logger.debug(
            f"MOSAIC: Generating Level 2 prompt | agent={agent_id} team={team} env={env_id}"
        )

    action_strings = ",\n".join(
        f"  {action}: {description}"
        for action, description in MULTIGRID_ACTIONS.items()
    )

    is_soccer = "Soccer" in env_id
    is_collect = "Collect" in env_id

    if is_soccer:
        team_name = "Red" if team == 0 else "Green"
        opponent_name = "Green" if team == 0 else "Red"

        return f"""
You are Agent {agent_id} playing a 2v2 soccer game.

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

PLAY!
""".strip()

    elif is_collect:
        colors = ["Red", "Green", "Blue"]
        color = colors[agent_id] if agent_id < 3 else "Unknown"

        return f"""
You are Agent {agent_id} ({color}) playing a ball collection game.

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

PLAY!
""".strip()

    else:
        return f"""
You are Agent {agent_id} in a multi-agent grid world.

Actions:
{action_strings}

Tips:
- Coordinate with visible teammates when possible
- Avoid repeating the same action if observation doesn't change

PLAY!
""".strip()


def get_instruction_prompt_level3(agent_id: int, team: int, role: str, env_id: str) -> str:
    """Generate Level 3 (Role-Based) instruction prompt.

    Assign explicit roles (Forward/Defender) with detailed role-specific strategies.
    Tests whether explicit role division improves team coordination and performance.

    Args:
        agent_id: Agent index.
        team: Team index.
        role: Assigned role ("forward" or "defender").
        env_id: Environment identifier.

    Returns:
        Level 3 instruction prompt with role-specific strategies.
    """
    if _HAS_STRUCTURED_LOGGING and log_constant:
        log_constant(
            logger,
            LOG_WORKER_MOSAIC_PROMPT_GENERATED,
            extra={
                "coordination_level": 3,
                "agent_id": agent_id,
                "team": team,
                "role": role,
                "env_id": env_id,
            },
        )
    else:
        logger.debug(
            f"MOSAIC: Generating Level 3 prompt | agent={agent_id} team={team} role={role} env={env_id}"
        )

    action_strings = ",\n".join(
        f"  {action}: {description}"
        for action, description in MULTIGRID_ACTIONS.items()
    )

    is_soccer = "Soccer" in env_id

    if not is_soccer:
        # Level 3 only applies to Soccer (roles don't make sense for Collect)
        return get_instruction_prompt_level2(agent_id, team, env_id)

    team_name = "Red" if team == 0 else "Green"
    opponent_name = "Green" if team == 0 else "Red"
    teammate_id = 1 if agent_id == 0 else 0 if agent_id in [0, 1] else 3 if agent_id == 2 else 2
    teammate_role = "DEFENDER" if role.lower() == "forward" else "FORWARD"

    # Role-specific strategies
    if role.lower() == "forward":
        role_strategy = """
Your Role: FORWARD (Offensive)
- Priority: Score goals, pressure opponents
- If you have ball → advance toward opponent goal
- If teammate has ball → move to open space ahead for pass
- If opponent has ball → pressure the ball carrier
- Stay in attacking half when possible
- Take risks to create scoring opportunities"""
    else:  # defender
        role_strategy = """
Your Role: DEFENDER (Defensive)
- Priority: Protect goal, support attacks
- If you have ball → pass forward or advance carefully
- If teammate has ball → cover defensive position
- If opponent has ball → fall back, block goal path
- Stay between ball and your goal
- Be conservative - don't leave goal undefended"""

    return f"""
You are Agent {agent_id} playing a 2v2 soccer game.

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

PLAY!
""".strip()


__all__ = [
    "MULTIGRID_ACTION_SPACE",
    "MULTIGRID_ACTIONS",
    "parse_action",
    "get_instruction_prompt_level1",
    "get_instruction_prompt_level2",
    "get_instruction_prompt_level3",
]
