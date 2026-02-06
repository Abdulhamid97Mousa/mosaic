"""XuanCe Worker Wrappers.

This module provides environment wrappers for XuanCe training,
including curriculum learning support via Syllabus-RL.
"""

from .curriculum import (
    BabyAITaskWrapper,
    make_babyai_curriculum,
    make_curriculum_env,
    BABYAI_GOTO_CURRICULUM,
    BABYAI_DOORKEY_CURRICULUM,
)

__all__ = [
    "BabyAITaskWrapper",
    "make_babyai_curriculum",
    "make_curriculum_env",
    "BABYAI_GOTO_CURRICULUM",
    "BABYAI_DOORKEY_CURRICULUM",
]
