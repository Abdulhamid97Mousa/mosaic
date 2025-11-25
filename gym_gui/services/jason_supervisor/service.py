from __future__ import annotations

"""Minimal Jason Supervisor service for control-plane integration.

This service maintains lightweight supervisor state for UI overlays and
provides a narrow API to validate and emit control updates. It does NOT
speak to workers directly; translation to trainer-side mechanisms remains
pluggable.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging

from gym_gui.constants import DEFAULT_SUPERVISOR, SupervisorDefaults
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_SERVICE_JASON_WORKER_EVENT,
    LOG_SERVICE_JASON_WORKER_WARNING,
)
from gym_gui.validations.validations_telemetry import ValidationService


@dataclass(slots=True)
class SupervisorState:
    active: bool = False
    safety_on: bool = True
    last_action: str = "—"
    actions_emitted: int = 0
    last_error: Optional[str] = None


class JasonSupervisorService(LogConstantMixin):
    """Tracks supervisor state and validates control updates.

    UI components may poll this service for status labels; presenters can
    call ``apply_control_update`` to emit structured updates after they
    are validated. The actual delivery to trainer/worker is delegated to
    callers for now (future: integrate TrainerClient control RPC when ready).
    """

    def __init__(
        self,
        *,
        defaults: SupervisorDefaults = DEFAULT_SUPERVISOR,
        validator: Optional[ValidationService] = None,
    ) -> None:
        self._logger = logging.getLogger("gym_gui.services.supervisor")
        self._defaults = defaults
        self._state = SupervisorState()
        self._validator = validator or ValidationService(strict_mode=False)

    # ------------------------------------------------------------------
    # State API
    # ------------------------------------------------------------------
    def is_active(self) -> bool:
        return self._state.active

    def set_active(self, active: bool) -> None:
        if self._state.active != active:
            self._state.active = active
            self.log_constant(
                LOG_SERVICE_JASON_WORKER_EVENT,
                message="active_changed",
                extra={"active": active},
            )

    def safety_enabled(self) -> bool:
        return self._state.safety_on

    def set_safety(self, enabled: bool) -> None:
        if self._state.safety_on != enabled:
            self._state.safety_on = enabled
            self.log_constant(
                LOG_SERVICE_JASON_WORKER_EVENT,
                message="safety_changed",
                extra={"safety_on": enabled},
            )

    def last_action(self) -> str:
        return self._state.last_action

    # ------------------------------------------------------------------
    # Control updates
    # ------------------------------------------------------------------
    def apply_control_update(self, update: Dict[str, Any]) -> bool:
        """Validate and register a supervisor control update.

        Returns True if valid and accepted; False otherwise. Delivery to
        trainer/worker is outside the scope of this minimal service.
        """
        model = self._validator.validate_trainer_control_update(update)
        if model is None:
            self._state.last_error = "validation_failed"
            self.log_constant(
                LOG_SERVICE_JASON_WORKER_WARNING,
                extra={"reason": "validation_failed", "update_preview": list(update.keys())[:6]},
            )
            return False

        # Apply conservative credit/backpressure awareness via caller-provided hints
        available_credits = int(update.get("available_credits", self._defaults.min_available_credits))
        if available_credits < self._defaults.min_available_credits:
            self.log_constant(
                LOG_SERVICE_JASON_WORKER_WARNING,
                message="insufficient_credits",
                extra={"available_credits": available_credits},
            )
            return False

        # Mark last_action and emit applied event
        self._state.last_action = model.reason or "control_update"
        self._state.actions_emitted += 1
        self.log_constant(
            LOG_SERVICE_JASON_WORKER_EVENT,
            extra={
                "run_id": model.run_id,
                "source": model.source,
                "params_keys": sorted(list(model.params.keys())),
            },
        )
        return True

    def record_rollback(self, *, reason: str) -> None:
        self._state.last_action = f"rollback: {reason}"
        self.log_constant(
            LOG_SERVICE_JASON_WORKER_EVENT,
            extra={"reason": reason},
        )

    # ------------------------------------------------------------------
    # Introspection for UI overlays
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "active": self._state.active,
            "safety_on": self._state.safety_on,
            "last_action": self._state.last_action,
            "actions_emitted": self._state.actions_emitted,
            "last_error": self._state.last_error,
        }
