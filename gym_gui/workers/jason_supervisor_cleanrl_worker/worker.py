from __future__ import annotations

"""Jason supervisor CleanRL worker stub.

The original Jason supervisor integration registered a bespoke CleanRL worker
actor that polled supervisor state and attempted to stream control updates back
into the trainer. That coupling has been removed; the CleanRL worker now runs
without any Jason-specific hooks. This module remains only to avoid import
errors from legacy references and simply aliases the standard CleanRL worker.
"""

import logging
from typing import Any

from gym_gui.services.actor import CleanRLWorkerActor

LOGGER = logging.getLogger("gym_gui.workers.jason_supervisor_cleanrl_worker")


class JasonSupervisorCleanRLWorkerActor(CleanRLWorkerActor):
    """Alias of ``CleanRLWorkerActor`` kept for backward compatibility."""

    # Re-use the canonical CleanRL worker identifier so no duplicate catalog entry
    id: str = CleanRLWorkerActor.id

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        LOGGER.warning(
            "JasonSupervisorCleanRLWorkerActor is deprecated; using CleanRLWorkerActor instead."
        )
        super().__init__()


__all__ = ["JasonSupervisorCleanRLWorkerActor"]
