"""Algorithm-specific documentation for the CleanRL worker."""

from __future__ import annotations

from typing import Dict

from .ppo import ALGO_DOCS as PPO_DOCS, DEFAULT_PPO_DOC
from .dqn import ALGO_DOCS as DQN_DOCS, DEFAULT_DQN_DOC

ALGO_DOCS: Dict[str, str] = {}
ALGO_DOCS.update(PPO_DOCS)
ALGO_DOCS.update(DQN_DOCS)

DEFAULT_DOC = (
    "<h3>CleanRL Algorithm</h3>"
    "<p>Select an algorithm to view recommended hyperparameters and configuration tips.</p>"
    "<p>The form exposes the most common extras; any additional CleanRL flag can be added via the\n"
    "Algorithm Parameters section.</p>"
)


def get_algo_doc(algo: str) -> str:
    """Return HTML help text for the requested algorithm."""

    if algo in ALGO_DOCS:
        return ALGO_DOCS[algo]
    if algo.startswith("ppo"):
        return DEFAULT_PPO_DOC
    if algo in {"dqn", "c51"}:
        return DEFAULT_DQN_DOC
    return DEFAULT_DOC


__all__ = ["ALGO_DOCS", "DEFAULT_DOC", "get_algo_doc"]
