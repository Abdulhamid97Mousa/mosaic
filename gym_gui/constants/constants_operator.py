"""Operator-related constants for the MOSAIC Operator abstraction.

This module defines constants for the Operator system, which manages action
selection entities (Human, Worker-based LLM agents, RL policies, etc.).

Operator Categories:
-------------------
- HUMAN: Human keyboard/mouse input operators
- LLM: Large Language Model-based operators (via subprocess workers)
- RL: Reinforcement Learning policy operators
- HYBRID: Combined human + agent operators

Usage:
------
    from gym_gui.constants import (
        OPERATOR_CATEGORY_HUMAN,
        OPERATOR_CATEGORY_LLM,
        OPERATOR_DEFAULTS,
    )

    # Check operator category
    if operator.category == OPERATOR_CATEGORY_LLM:
        # Handle LLM operator
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# ================================================================
# Operator Category Constants
# ================================================================

OPERATOR_CATEGORY_HUMAN: str = "human"
OPERATOR_CATEGORY_LLM: str = "llm"
OPERATOR_CATEGORY_RL: str = "rl"
OPERATOR_CATEGORY_HYBRID: str = "hybrid"

# All valid operator categories
OPERATOR_CATEGORIES: Tuple[str, ...] = (
    OPERATOR_CATEGORY_HUMAN,
    OPERATOR_CATEGORY_LLM,
    OPERATOR_CATEGORY_RL,
    OPERATOR_CATEGORY_HYBRID,
)

# ================================================================
# Default Operator IDs
# ================================================================

DEFAULT_OPERATOR_ID_HUMAN: str = "human"
DEFAULT_OPERATOR_ID_BALROG_LLM: str = "balrog_llm"

# ================================================================
# Worker Operator Configuration
# ================================================================

# Worker ID for BALROG LLM worker (subprocess)
WORKER_ID_BALROG: str = "balrog_worker"

# Display names for operators
OPERATOR_DISPLAY_NAME_HUMAN: str = "Human (Keyboard)"
OPERATOR_DISPLAY_NAME_BALROG_LLM: str = "BALROG LLM Agent"

# Descriptions for operator tooltips
OPERATOR_DESCRIPTION_HUMAN: str = (
    "Human operator using keyboard/mouse input. "
    "Actions are selected via the GUI input system."
)
OPERATOR_DESCRIPTION_BALROG_LLM: str = (
    "LLM-based agent using BALROG benchmark framework. "
    "Supports OpenAI, Claude, Gemini, and vLLM backends."
)

# ================================================================
# BALROG Worker Specific Constants
# ================================================================

# Supported environment families for BALROG LLM worker
# Note: "minigrid" uses the same wrapper as "babyai" (BabyAI is built on MiniGrid)
BALROG_SUPPORTED_ENVS: Tuple[str, ...] = (
    "babyai",
    "minigrid",
    "minihack",
    "crafter",
    "nle",
    "textworld",
    "toytext",
)

# Supported LLM clients
BALROG_SUPPORTED_CLIENTS: Tuple[str, ...] = (
    "openai",
    "anthropic",
    "google",
    "vllm",
)

# Supported agent reasoning types
BALROG_AGENT_TYPES: Tuple[str, ...] = (
    "naive",
    "cot",
    "robust_naive",
    "robust_cot",
    "few_shot",
    "dummy",
)

# Default BALROG configuration values
BALROG_DEFAULT_ENV: str = "babyai"
BALROG_DEFAULT_TASK: str = "BabyAI-GoToRedBall-v0"
BALROG_DEFAULT_CLIENT: str = "openai"
BALROG_DEFAULT_MODEL: str = "gpt-4o-mini"
BALROG_DEFAULT_AGENT_TYPE: str = "naive"
BALROG_DEFAULT_NUM_EPISODES: int = 5
BALROG_DEFAULT_MAX_STEPS: int = 100
BALROG_DEFAULT_TEMPERATURE: float = 0.7

# ================================================================
# Structured Operator Defaults (Dataclass)
# ================================================================


@dataclass(frozen=True)
class OperatorCategoryDefaults:
    """Default values for operator categories."""

    human: str = OPERATOR_CATEGORY_HUMAN
    llm: str = OPERATOR_CATEGORY_LLM
    rl: str = OPERATOR_CATEGORY_RL
    hybrid: str = OPERATOR_CATEGORY_HYBRID
    all_categories: Tuple[str, ...] = OPERATOR_CATEGORIES


@dataclass(frozen=True)
class BarlogDefaults:
    """Default configuration values for BALROG LLM worker."""

    supported_envs: Tuple[str, ...] = BALROG_SUPPORTED_ENVS
    supported_clients: Tuple[str, ...] = BALROG_SUPPORTED_CLIENTS
    agent_types: Tuple[str, ...] = BALROG_AGENT_TYPES
    default_env: str = BALROG_DEFAULT_ENV
    default_task: str = BALROG_DEFAULT_TASK
    default_client: str = BALROG_DEFAULT_CLIENT
    default_model: str = BALROG_DEFAULT_MODEL
    default_agent_type: str = BALROG_DEFAULT_AGENT_TYPE
    default_num_episodes: int = BALROG_DEFAULT_NUM_EPISODES
    default_max_steps: int = BALROG_DEFAULT_MAX_STEPS
    default_temperature: float = BALROG_DEFAULT_TEMPERATURE


@dataclass(frozen=True)
class OperatorDefaults:
    """Aggregated operator defaults."""

    categories: OperatorCategoryDefaults = field(
        default_factory=OperatorCategoryDefaults
    )
    balrog: BarlogDefaults = field(default_factory=BarlogDefaults)
    default_operator_id: str = DEFAULT_OPERATOR_ID_HUMAN
    worker_id_balrog: str = WORKER_ID_BALROG


OPERATOR_DEFAULTS = OperatorDefaults()


__all__ = [
    # Category constants
    "OPERATOR_CATEGORY_HUMAN",
    "OPERATOR_CATEGORY_LLM",
    "OPERATOR_CATEGORY_RL",
    "OPERATOR_CATEGORY_HYBRID",
    "OPERATOR_CATEGORIES",
    # Default operator IDs
    "DEFAULT_OPERATOR_ID_HUMAN",
    "DEFAULT_OPERATOR_ID_BALROG_LLM",
    # Worker IDs
    "WORKER_ID_BALROG",
    # Display names
    "OPERATOR_DISPLAY_NAME_HUMAN",
    "OPERATOR_DISPLAY_NAME_BALROG_LLM",
    # Descriptions
    "OPERATOR_DESCRIPTION_HUMAN",
    "OPERATOR_DESCRIPTION_BALROG_LLM",
    # BALROG specific
    "BALROG_SUPPORTED_ENVS",
    "BALROG_SUPPORTED_CLIENTS",
    "BALROG_AGENT_TYPES",
    "BALROG_DEFAULT_ENV",
    "BALROG_DEFAULT_TASK",
    "BALROG_DEFAULT_CLIENT",
    "BALROG_DEFAULT_MODEL",
    "BALROG_DEFAULT_AGENT_TYPE",
    "BALROG_DEFAULT_NUM_EPISODES",
    "BALROG_DEFAULT_MAX_STEPS",
    "BALROG_DEFAULT_TEMPERATURE",
    # Structured defaults
    "OperatorCategoryDefaults",
    "BarlogDefaults",
    "OperatorDefaults",
    "OPERATOR_DEFAULTS",
]
