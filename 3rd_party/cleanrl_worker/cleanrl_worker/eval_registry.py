"""Registry describing how to evaluate CleanRL algorithms under MOSAIC."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from typing import Any, Callable, Dict, Optional

__all__ = ["EvalEntry", "EVAL_REGISTRY", "get_eval_entry"]


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


EVAL_REGISTRY: Dict[str, EvalEntry] = {
    "ppo": EvalEntry(
        agent_path="cleanrl_worker.algorithms.ppo_with_save.Agent",
        make_env_path="cleanrl_worker.algorithms.ppo_with_save.make_env",
        eval_path="cleanrl_worker.cleanrl_utils.evals.ppo_eval.evaluate",
        accepts_gamma=False,
    ),
    "ppo_continuous_action": EvalEntry(
        agent_path="cleanrl.ppo_continuous_action.Agent",
        make_env_path="cleanrl.ppo_continuous_action.make_env",
        eval_path="cleanrl_utils.evals.ppo_eval.evaluate",
        accepts_gamma=True,
    ),
    # Additional algorithms can be registered here (ppo_atari, dqn, c51, etc.)
}


def get_eval_entry(algo: str) -> Optional[EvalEntry]:
    """Return the evaluation entry for a CleanRL algorithm, if registered."""

    return EVAL_REGISTRY.get(algo)


def _load_attr(path: str) -> Any:
    module_name, attr_name = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)
