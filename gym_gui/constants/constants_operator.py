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
DEFAULT_OPERATOR_ID_BARLOG_LLM: str = "barlog_llm"

# ================================================================
# Worker Operator Configuration
# ================================================================

# Worker ID for BARLOG LLM worker (subprocess)
WORKER_ID_BARLOG: str = "barlog_worker"

# Display names for operators
OPERATOR_DISPLAY_NAME_HUMAN: str = "Human (Keyboard)"
OPERATOR_DISPLAY_NAME_BARLOG_LLM: str = "BARLOG LLM Agent"

# Descriptions for operator tooltips
OPERATOR_DESCRIPTION_HUMAN: str = (
    "Human operator using keyboard/mouse input. "
    "Actions are selected via the GUI input system."
)
OPERATOR_DESCRIPTION_BARLOG_LLM: str = (
    "LLM-based agent using BALROG benchmark framework. "
    "Supports OpenAI, Claude, Gemini, and vLLM backends."
)

# ================================================================
# BARLOG Worker Specific Constants
# ================================================================

# Supported environments for BARLOG LLM worker
# Note: "minigrid" uses the same wrapper as "babyai" (BabyAI is built on MiniGrid)
BARLOG_SUPPORTED_ENVS: Tuple[str, ...] = (
    "babyai",
    "minigrid",
    "minihack",
    "crafter",
    "nle",
    "textworld",
)

# Supported LLM clients
BARLOG_SUPPORTED_CLIENTS: Tuple[str, ...] = (
    "openai",
    "anthropic",
    "google",
    "vllm",
)

# Supported agent reasoning types
BARLOG_AGENT_TYPES: Tuple[str, ...] = (
    "naive",
    "cot",
    "robust_naive",
    "robust_cot",
    "few_shot",
    "dummy",
)

# Default BARLOG configuration values
BARLOG_DEFAULT_ENV: str = "babyai"
BARLOG_DEFAULT_TASK: str = "BabyAI-GoToRedBall-v0"
BARLOG_DEFAULT_CLIENT: str = "openai"
BARLOG_DEFAULT_MODEL: str = "gpt-4o-mini"
BARLOG_DEFAULT_AGENT_TYPE: str = "naive"
BARLOG_DEFAULT_NUM_EPISODES: int = 5
BARLOG_DEFAULT_MAX_STEPS: int = 100
BARLOG_DEFAULT_TEMPERATURE: float = 0.7

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
    """Default configuration values for BARLOG LLM worker."""

    supported_envs: Tuple[str, ...] = BARLOG_SUPPORTED_ENVS
    supported_clients: Tuple[str, ...] = BARLOG_SUPPORTED_CLIENTS
    agent_types: Tuple[str, ...] = BARLOG_AGENT_TYPES
    default_env: str = BARLOG_DEFAULT_ENV
    default_task: str = BARLOG_DEFAULT_TASK
    default_client: str = BARLOG_DEFAULT_CLIENT
    default_model: str = BARLOG_DEFAULT_MODEL
    default_agent_type: str = BARLOG_DEFAULT_AGENT_TYPE
    default_num_episodes: int = BARLOG_DEFAULT_NUM_EPISODES
    default_max_steps: int = BARLOG_DEFAULT_MAX_STEPS
    default_temperature: float = BARLOG_DEFAULT_TEMPERATURE


@dataclass(frozen=True)
class OperatorDefaults:
    """Aggregated operator defaults."""

    categories: OperatorCategoryDefaults = field(
        default_factory=OperatorCategoryDefaults
    )
    barlog: BarlogDefaults = field(default_factory=BarlogDefaults)
    default_operator_id: str = DEFAULT_OPERATOR_ID_HUMAN
    worker_id_barlog: str = WORKER_ID_BARLOG


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
    "DEFAULT_OPERATOR_ID_BARLOG_LLM",
    # Worker IDs
    "WORKER_ID_BARLOG",
    # Display names
    "OPERATOR_DISPLAY_NAME_HUMAN",
    "OPERATOR_DISPLAY_NAME_BARLOG_LLM",
    # Descriptions
    "OPERATOR_DESCRIPTION_HUMAN",
    "OPERATOR_DESCRIPTION_BARLOG_LLM",
    # BARLOG specific
    "BARLOG_SUPPORTED_ENVS",
    "BARLOG_SUPPORTED_CLIENTS",
    "BARLOG_AGENT_TYPES",
    "BARLOG_DEFAULT_ENV",
    "BARLOG_DEFAULT_TASK",
    "BARLOG_DEFAULT_CLIENT",
    "BARLOG_DEFAULT_MODEL",
    "BARLOG_DEFAULT_AGENT_TYPE",
    "BARLOG_DEFAULT_NUM_EPISODES",
    "BARLOG_DEFAULT_MAX_STEPS",
    "BARLOG_DEFAULT_TEMPERATURE",
    # Structured defaults
    "OperatorCategoryDefaults",
    "BarlogDefaults",
    "OperatorDefaults",
    "OPERATOR_DEFAULTS",
]
