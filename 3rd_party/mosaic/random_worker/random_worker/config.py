"""Configuration for Random Worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RandomWorkerConfig:
    """Configuration for the random action-selector worker.

    Attributes:
        run_id: Unique identifier for this run (from GUI).
        env_name: Environment family (mosaic_multigrid, babyai, etc.).
        task: Specific task / gymnasium ID.
        seed: Random seed for reproducibility.
        behavior: Action selection strategy (random, noop, cycling).
    """

    run_id: str = ""
    env_name: str = ""
    task: str = ""
    seed: Optional[int] = None
    behavior: str = "random"
