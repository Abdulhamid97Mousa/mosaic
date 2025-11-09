from __future__ import annotations

"""Jason Supervisor CleanRL worker package.

This package provides a placeholder Actor that represents a CleanRL worker
running under Jason Supervisor oversight, without modifying the existing
CleanRL wiring. It allows the GUI to list/select this worker independently
from the standard CleanRL worker actor.

Registration is intentionally left to the application bootstrap or callers
who want to opt into this worker (to avoid changing current defaults).
"""

from .worker import JasonSupervisorCleanRLWorkerActor

__all__ = ["JasonSupervisorCleanRLWorkerActor"]
