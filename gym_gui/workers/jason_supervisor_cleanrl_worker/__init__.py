from __future__ import annotations

"""Jason Supervisor CleanRL worker stub.

The Jason supervisor integration no longer controls CleanRL workers. This
package now exposes a compatibility alias so legacy imports continue to
resolve, while the underlying behaviour defers to ``CleanRLWorkerActor``.
"""

from .worker import JasonSupervisorCleanRLWorkerActor

__all__ = ["JasonSupervisorCleanRLWorkerActor"]
