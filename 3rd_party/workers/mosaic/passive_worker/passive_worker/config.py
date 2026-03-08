"""Configuration for Passive Worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PassiveWorkerConfig:
    """Configuration for the passive (noop) baseline worker.

    Attributes:
        run_id: Unique identifier for this run (from GUI).
        env_name: Environment family (mosaic_multigrid, babyai, etc.).
        task: Specific task / gymnasium ID.
        seed: Random seed for reproducibility (used for env resets).
    """

    run_id: str = ""
    env_name: str = ""
    task: str = ""
    seed: Optional[int] = None
