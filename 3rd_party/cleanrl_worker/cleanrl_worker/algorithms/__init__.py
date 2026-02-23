"""
CleanRL Algorithm Implementations with Shared Infrastructure.

Architecture:
    Instead of duplicating code for each algorithm (ppo.py, dqn.py, sac.py...),
    we provide SHARED infrastructure that any algorithm can use:

    - cleanrl_worker.agents: MinigridAgent, MLPAgent (algorithm-agnostic)
    - cleanrl_worker.wrappers: make_env, is_minigrid_env (algorithm-agnostic)
    - cleanrl_worker.save: save_checkpoint, load_checkpoint (algorithm-agnostic)

    Currently implemented:
    - ppo.py: PPO with full MiniGrid/BabyAI support

    For other algorithms, runtime falls back to original cleanrl:
    - "dqn" → cleanrl.dqn
    - "sac" → cleanrl.sac_continuous_action
    - etc.

To add a new algorithm with MiniGrid support:
    1. Copy from cleanrl (e.g., cleanrl/dqn.py)
    2. Import from this module: `from . import MinigridAgent, make_env, ...`
    3. Add agent selection logic (like in ppo.py)
    4. Update runtime.py ALGO_REGISTRY

Usage:
    python -m cleanrl_worker.algorithms.ppo --env-id BabyAI-GoToRedBall-v0
"""

# Re-export shared utilities for algorithm implementations
from cleanrl_worker.agents import MinigridAgent, MLPAgent
from cleanrl_worker.wrappers.minigrid import is_minigrid_env, make_env
from cleanrl_worker.save import save_checkpoint, load_checkpoint

__all__ = [
    "MinigridAgent",
    "MLPAgent",
    "is_minigrid_env",
    "make_env",
    "save_checkpoint",
    "load_checkpoint",
]
