"""Crafter game documentation module.

Crafter is an open world survival game benchmark for reinforcement learning
that evaluates a wide range of agent capabilities within a single environment.

Environments:
    - CrafterReward-v1: With reward signals
    - CrafterNoReward-v1: Reward-free variant for unsupervised learning

Reference:
    Hafner, D. (2022). Benchmarking the Spectrum of Agent Capabilities. ICLR 2022.
    https://github.com/danijar/crafter
"""

from __future__ import annotations

from .CrafterEnv import (
    CRAFTER_HTML,
    CRAFTER_NO_REWARD_HTML,
    CRAFTER_REWARD_HTML,
    get_crafter_html,
)

__all__ = [
    "CRAFTER_HTML",
    "CRAFTER_REWARD_HTML",
    "CRAFTER_NO_REWARD_HTML",
    "get_crafter_html",
]
