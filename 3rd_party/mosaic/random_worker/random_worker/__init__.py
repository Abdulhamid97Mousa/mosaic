"""Random action-selector worker for MOSAIC multi-agent environments.

Provides a lightweight subprocess that responds to the init_agent / select_action
JSON protocol used by the MOSAIC GUI for multi-agent games.  Each agent gets its
own subprocess (and therefore its own per-agent log file).
"""

__version__ = "0.1.0"

from random_worker.config import RandomWorkerConfig
from random_worker.runtime import RandomWorkerRuntime

__all__ = [
    "RandomWorkerConfig",
    "RandomWorkerRuntime",
]
