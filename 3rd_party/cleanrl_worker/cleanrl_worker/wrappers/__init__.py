"""
Wrappers for CleanRL environments.

This module provides algorithm-agnostic wrappers that can be used with any
CleanRL algorithm (PPO, DQN, SAC, etc.) by operating at the environment level.

Available wrappers:
    - Curriculum Learning: Automatically progress through increasingly difficult environments
    - Procedural Generation: Control whether environments use random or fixed layouts
"""

from cleanrl_worker.wrappers.procedural_generation import ProceduralGenerationWrapper
from cleanrl_worker.wrappers.curriculum import (
    BabyAITaskWrapper,
    make_babyai_curriculum,
    make_curriculum_env,
    BABYAI_GOTO_CURRICULUM,
    BABYAI_DOORKEY_CURRICULUM,
)

__all__ = [
    # Curriculum Learning
    "BabyAITaskWrapper",
    "make_babyai_curriculum",
    "make_curriculum_env",
    "BABYAI_GOTO_CURRICULUM",
    "BABYAI_DOORKEY_CURRICULUM",
    # Procedural Generation
    "ProceduralGenerationWrapper",
]
