from __future__ import annotations
"""Actor representing a CleanRL worker supervised by the Jason Supervisor.

This actor does not directly select actions (delegated to external CleanRL
process) but offers lifecycle integration points to surface supervisor-related
metadata and potential future control updates.

Design goals:
- Avoid modifying existing CleanRL wiring (pure addition).
- Provide a distinct actor id: ``jason_supervisor_cleanrl_worker``.
- Allow optional registration via environment variable without editing bootstrap.
- Query ``JasonSupervisorService`` for snapshot info on step/episode boundaries.

Future extension points:
- Forward validated control updates to trainer RPC once available.
- Annotate telemetry with supervisor decision metadata.
- Emit structured logs tying supervisor actions to episode outcomes.
"""
from dataclasses import dataclass
from typing import Optional
import os
import logging

from gym_gui.services.actor import StepSnapshot, EpisodeSummary, Actor
from gym_gui.services.service_locator import get_service_locator
from gym_gui.services.jason_supervisor import JasonSupervisorService
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_SERVICE_SUPERVISOR_EVENT,
)

LOGGER = logging.getLogger("gym_gui.workers.jason_supervisor_cleanrl_worker")


@dataclass(slots=True)
class JasonSupervisorCleanRLWorkerActor(LogConstantMixin):
    """Placeholder actor wrapper for a supervised CleanRL worker.

    The worker itself runs out-of-process; this actor surfaces supervisor
    status for UI selection and future coordination. Action selection is
    delegated externally (returns ``None``).
    """

    id: str = "jason_supervisor_cleanrl_worker"

    def select_action(self, step: StepSnapshot) -> Optional[int]:  # pragma: no cover - external policy
        # No direct action; CleanRL worker selects actions independently.
        return None

    def on_step(self, step: StepSnapshot) -> None:
        """Hook invoked after each environment step.

        We opportunistically poll supervisor state for potential lightweight
        instrumentation (e.g., last_action). Currently logs sampled state
        every Nth step to avoid noise.
        """
        if step.step_index % 128 == 0:  # low-frequency sampling
            supervisor = get_service_locator().resolve(JasonSupervisorService)
            if supervisor:
                snap = supervisor.snapshot()
                self.log_constant(
                    LOG_SERVICE_SUPERVISOR_EVENT,
                    message="supervised_step_sample",
                    extra={
                        "active": snap.get("active"),
                        "safety_on": snap.get("safety_on"),
                        "last_action": snap.get("last_action"),
                        "actions_emitted": snap.get("actions_emitted"),
                    },
                )

    def on_episode_end(self, summary: EpisodeSummary) -> None:
        """Episode boundary hook: attach supervisor snapshot metadata.

        Future implementation could push annotation events or control feedback.
        Currently emits a structured log for observability.
        """
        supervisor = get_service_locator().resolve(JasonSupervisorService)
        if supervisor:
            snap = supervisor.snapshot()
            self.log_constant(
                LOG_SERVICE_SUPERVISOR_EVENT,
                message="supervised_episode_end",
                extra={
                    "episode_index": summary.episode_index,
                    "total_reward": summary.total_reward,
                    "steps": summary.steps,
                    "active": snap.get("active"),
                    "safety_on": snap.get("safety_on"),
                    "last_action": snap.get("last_action"),
                    "actions_emitted": snap.get("actions_emitted"),
                },
            )


def _auto_register() -> None:  # pragma: no cover - optional runtime hook
    """Optionally register the actor with the global ActorService.

    Controlled by environment variable ``ENABLE_JASON_SUPERVISOR_CLEANRL_WORKER``.
    This avoids modifying existing bootstrap wiring while allowing quick
    experimentation.
    """
    if os.getenv("ENABLE_JASON_SUPERVISOR_CLEANRL_WORKER") != "1":
        return
    try:
        from gym_gui.services.actor import ActorService  # local import to avoid circulars
        locator = get_service_locator()
        actors = locator.resolve(ActorService)
        if actors is None:
            return
        # Avoid duplicate registration
        if "jason_supervisor_cleanrl_worker" in actors.available_actor_ids():
            return
        actor = JasonSupervisorCleanRLWorkerActor()
        actors.register_actor(
            actor,
            display_name="Jason Supervisor CleanRL Worker",
            description="CleanRL worker under Jason Supervisor observational oversight.",
            policy_label="External CleanRL policy (supervised)",
            backend_label="Trainer-managed worker + supervisor",
        )
    except Exception:  # pragma: no cover - defensive
        LOGGER.exception("Auto-registration of JasonSupervisorCleanRLWorkerActor failed")


# Attempt auto-registration on import (safe no-op if env var not set)
_auto_register()

__all__ = ["JasonSupervisorCleanRLWorkerActor"]
