"""
Wrappers for CleanRL environments.

This module provides algorithm-agnostic wrappers that can be used with any
CleanRL algorithm (PPO, DQN, SAC, etc.) by operating at the environment level.

Available wrappers:
    - Curriculum Learning: Automatically progress through increasingly difficult environments (requires syllabus package)
    - Procedural Generation: Control whether environments use random or fixed layouts
    - MOSAIC MultiGrid: Wrappers for MOSAIC MultiGrid environments
"""

from cleanrl_worker.wrappers.procedural_generation import ProceduralGenerationWrapper
from cleanrl_worker.wrappers.mosaic_multigrid import (
    is_mosaic_env,
    make_mosaic_env,
    make_mosaic_envs,
)

# Optional curriculum learning imports (requires syllabus package)
try:
    from cleanrl_worker.wrappers.curriculum import (
        BabyAITaskWrapper,
        make_babyai_curriculum,
        make_curriculum_env,
        BABYAI_GOTO_CURRICULUM,
        BABYAI_DOORKEY_CURRICULUM,
    )
    _CURRICULUM_AVAILABLE = True
except ImportError:
    _CURRICULUM_AVAILABLE = False
    BabyAITaskWrapper = None
    make_babyai_curriculum = None
    make_curriculum_env = None
    BABYAI_GOTO_CURRICULUM = None
    BABYAI_DOORKEY_CURRICULUM = None

__all__ = [
    # Procedural Generation
    "ProceduralGenerationWrapper",
    # MOSAIC MultiGrid
    "is_mosaic_env",
    "make_mosaic_env",
    "make_mosaic_envs",
]

# Add curriculum learning to __all__ if available
if _CURRICULUM_AVAILABLE:
    __all__.extend([
        "BabyAITaskWrapper",
        "make_babyai_curriculum",
        "make_curriculum_env",
        "BABYAI_GOTO_CURRICULUM",
        "BABYAI_DOORKEY_CURRICULUM",
    ])
