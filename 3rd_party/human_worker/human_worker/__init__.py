"""Human Worker - Human-in-the-loop action selection for multi-agent games."""

__version__ = "0.1.0"

from .runtime import HumanWorkerRuntime
from .config import HumanWorkerConfig


def get_worker_metadata() -> dict:
    """Return worker metadata for MOSAIC registry."""
    return {
        "name": "human_worker",
        "version": __version__,
        "description": "Human-in-the-loop action selection via GUI clicks",
        "supported_envs": ["pettingzoo", "gymnasium"],
        "supported_tasks": ["chess_v6", "go_v5", "connect_four_v3", "tictactoe_v3"],
        "entry_point": "human_worker.cli:main",
        "runtime_class": "human_worker.runtime:HumanWorkerRuntime",
        "config_class": "human_worker.config:HumanWorkerConfig",
    }


__all__ = ["HumanWorkerRuntime", "HumanWorkerConfig", "__version__", "get_worker_metadata"]
