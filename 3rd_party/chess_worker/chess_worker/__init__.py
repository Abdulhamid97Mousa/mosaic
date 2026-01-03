"""Chess Worker - LLM-based chess player using llm_chess prompting style."""

__version__ = "0.1.0"

from .runtime import ChessWorkerRuntime
from .config import ChessWorkerConfig


def get_worker_metadata() -> dict:
    """Return worker metadata for MOSAIC registry."""
    return {
        "name": "chess_worker",
        "version": __version__,
        "description": "LLM-based chess player using llm_chess prompting style",
        "supported_envs": ["pettingzoo"],
        "supported_tasks": ["chess_v6"],
        "entry_point": "chess_worker.cli:main",
        "runtime_class": "chess_worker.runtime:ChessWorkerRuntime",
        "config_class": "chess_worker.config:ChessWorkerConfig",
    }


__all__ = ["ChessWorkerRuntime", "ChessWorkerConfig", "__version__", "get_worker_metadata"]
