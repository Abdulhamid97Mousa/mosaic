"""Registry describing how to evaluate CleanRL algorithms under MOSAIC."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Type

__all__ = ["EvalEntry", "EVAL_REGISTRY", "get_eval_entry", "unified_evaluate"]


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalEntry:
    """Metadata describing how to evaluate a CleanRL algorithm."""

    agent_path: str
    make_env_path: str
    eval_path: str
    accepts_gamma: bool = True

    @cached_property
    def agent_cls(self) -> type:
        return _load_attr(self.agent_path)

    @cached_property
    def make_env(self) -> Callable[..., Any]:
        return _load_attr(self.make_env_path)

    @cached_property
    def evaluate(self) -> Callable[..., Any]:
        return _load_attr(self.eval_path)


def unified_evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: Type,
    device: Any = "cpu",
    capture_video: bool = True,
    gamma: float = 0.99,
    algo: str = "ppo",
) -> List[float]:
    """Unified evaluation function compatible with ppo_eval.evaluate signature.

    This function wraps the unified_eval system to provide a single evaluation
    function that works with ANY CleanRL algorithm, while maintaining backward
    compatibility with the ppo_eval.evaluate interface.

    Args:
        model_path: Path to the .cleanrl_model checkpoint
        make_env: Environment factory function
        env_id: Environment ID (e.g., "CartPole-v1")
        eval_episodes: Number of episodes to evaluate
        run_name: Name for this evaluation run
        Model: Agent class (e.g., Agent from ppo.py or QNetwork from dqn.py)
        device: Device to run evaluation on ("cpu" or "cuda")
        capture_video: Whether to capture video
        gamma: Discount factor
        algo: Algorithm name for selecting the correct adapter

    Returns:
        List of episodic returns
    """
    import gymnasium as gym

    from .unified_eval import evaluate as unified_eval_fn
    from .unified_eval.registry import get_adapter

    # Get adapter for this algorithm
    adapter = get_adapter(algo)
    if adapter is None:
        LOGGER.warning("No unified adapter for '%s', falling back to PPO adapter", algo)
        from .unified_eval.adapters import PPOSelector
        adapter = PPOSelector()

    # Create environment
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])

    try:
        # Load model using adapter
        adapter.load(model_path, envs, str(device), model_cls=Model)

        # Run evaluation using unified evaluator
        result = unified_eval_fn(adapter, envs, eval_episodes)

        return result.returns
    finally:
        envs.close()
        adapter.close()


# Algorithm to module path mapping for dynamic loading
# Maps algorithm name to (agent_path, make_env_path, accepts_gamma)
_ALGO_MODULES: Dict[str, tuple] = {
    # PPO family
    "ppo": ("cleanrl_worker.agents.MLPAgent", "cleanrl_worker.wrappers.minigrid.make_env", True),
    "ppo_continuous_action": ("cleanrl.ppo_continuous_action.Agent", "cleanrl.ppo_continuous_action.make_env", True),
    "ppo_atari": ("cleanrl.ppo_atari.Agent", "cleanrl.ppo_atari.make_env", False),
    "ppo_atari_lstm": ("cleanrl.ppo_atari_lstm.Agent", "cleanrl.ppo_atari_lstm.make_env", False),
    "ppo_atari_envpool": ("cleanrl.ppo_atari_envpool.Agent", "cleanrl.ppo_atari_envpool.make_env", False),
    "ppo_procgen": ("cleanrl.ppo_procgen.Agent", "cleanrl.ppo_procgen.make_env", False),

    # DQN family
    "dqn": ("cleanrl.dqn.QNetwork", "cleanrl.dqn.make_env", True),
    "dqn_atari": ("cleanrl.dqn_atari.QNetwork", "cleanrl.dqn_atari.make_env", False),

    # C51 (Distributional DQN)
    "c51": ("cleanrl.c51.QNetwork", "cleanrl.c51.make_env", True),
    "c51_atari": ("cleanrl.c51_atari.QNetwork", "cleanrl.c51_atari.make_env", False),

    # DDPG
    "ddpg_continuous_action": ("cleanrl.ddpg_continuous_action.Actor", "cleanrl.ddpg_continuous_action.make_env", True),

    # TD3
    "td3_continuous_action": ("cleanrl.td3_continuous_action.Actor", "cleanrl.td3_continuous_action.make_env", True),

    # SAC
    "sac_continuous_action": ("cleanrl.sac_continuous_action.Actor", "cleanrl.sac_continuous_action.make_env", True),
}


def _create_unified_eval_entry(algo: str, agent_path: str, make_env_path: str, accepts_gamma: bool) -> EvalEntry:
    """Create an EvalEntry that uses the unified evaluator.

    The eval_path points to a dynamically created evaluate function that
    wraps unified_evaluate with the correct algo parameter.
    """
    # For unified entries, we store the algo name and use a factory
    return _UnifiedEvalEntry(
        agent_path=agent_path,
        make_env_path=make_env_path,
        eval_path="",  # Not used, evaluate property is overridden
        accepts_gamma=accepts_gamma,
        algo_name=algo,
    )


@dataclass(frozen=True)
class _UnifiedEvalEntry(EvalEntry):
    """EvalEntry that uses the unified evaluation system."""

    algo_name: str = ""

    @cached_property
    def evaluate(self) -> Callable[..., Any]:
        """Return an evaluate function that uses unified_evaluate."""
        algo = self.algo_name

        def _evaluate(
            model_path: str,
            make_env: Callable,
            env_id: str,
            eval_episodes: int,
            run_name: str,
            Model: Type,
            device: Any = "cpu",
            capture_video: bool = True,
            gamma: float = 0.99,
        ) -> List[float]:
            return unified_evaluate(
                model_path=model_path,
                make_env=make_env,
                env_id=env_id,
                eval_episodes=eval_episodes,
                run_name=run_name,
                Model=Model,
                device=device,
                capture_video=capture_video,
                gamma=gamma,
                algo=algo,
            )

        return _evaluate


# Build the registry from algorithm modules
EVAL_REGISTRY: Dict[str, EvalEntry] = {}

for algo, (agent_path, make_env_path, accepts_gamma) in _ALGO_MODULES.items():
    EVAL_REGISTRY[algo] = _create_unified_eval_entry(algo, agent_path, make_env_path, accepts_gamma)


def get_eval_entry(algo: str) -> Optional[EvalEntry]:
    """Return the evaluation entry for a CleanRL algorithm, if registered.

    This now supports all algorithms in the unified_eval registry.
    """
    return EVAL_REGISTRY.get(algo)


def _load_attr(path: str) -> Any:
    module_name, attr_name = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)
