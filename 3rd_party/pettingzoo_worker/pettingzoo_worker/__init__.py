"""PettingZoo Worker - Multi-agent environment wrapper for Mosaic.

This package provides utilities for working with PettingZoo multi-agent
environments in both AEC (turn-based) and Parallel modes.
"""

from .config import PettingZooConfig
from .env_wrapper import (
    PettingZooWrapper,
    create_aec_env,
    create_parallel_env,
)

__all__ = [
    "PettingZooConfig",
    "PettingZooWrapper",
    "create_aec_env",
    "create_parallel_env",
]
