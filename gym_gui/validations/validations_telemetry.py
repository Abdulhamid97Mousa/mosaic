"""Telemetry validation helpers used across services and controllers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import ValidationError

from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_SERVICE_VALIDATION_DEBUG,
    LOG_SERVICE_VALIDATION_WARNING,
    LOG_SERVICE_VALIDATION_ERROR,
)
from gym_gui.validations.validations_pydantic import (
    TelemetryEventBase,
    TrainingConfig,
    validate_telemetry_event,
)


class ValidationService(LogConstantMixin):
    """Service object for validating telemetry payloads and trainer configs."""

    def __init__(self, *, strict_mode: bool = False) -> None:
        import logging

        self.strict_mode = strict_mode
        self._validation_errors: list[str] = []
        self._validation_warnings: list[str] = []
        self._logger = logging.getLogger("gym_gui.services.validation")

    # ------------------------------------------------------------------
    # Telemetry events
    # ------------------------------------------------------------------
    def validate_telemetry_event(self, event_data: Dict[str, Any]) -> Optional[TelemetryEventBase]:
        try:
            event = validate_telemetry_event(event_data)
            self.log_constant(
                LOG_SERVICE_VALIDATION_DEBUG,
                message="telemetry_event_valid",
                extra={"event_type": event.type},
            )
            return event
        except ValidationError as exc:
            error_msg = f"Telemetry validation error: {exc}"
            self._validation_errors.append(error_msg)
            self.log_constant(
                LOG_SERVICE_VALIDATION_ERROR,
                message=error_msg,
                extra={"phase": "telemetry_event"},
            )
            if self.strict_mode:
                raise
            return None
        except Exception as exc:  # pragma: no cover - unexpected path
            error_msg = f"Unexpected validation error: {exc}"
            self._validation_errors.append(error_msg)
            self.log_constant(
                LOG_SERVICE_VALIDATION_ERROR,
                message=error_msg,
                extra={"phase": "telemetry_event", "error_type": type(exc).__name__},
                exc_info=exc,
            )
            if self.strict_mode:
                raise
            return None

    # ------------------------------------------------------------------
    # Trainer configuration
    # ------------------------------------------------------------------
    def validate_training_config(self, config_data: Dict[str, Any]) -> Optional[TrainingConfig]:
        try:
            config = TrainingConfig(**config_data)
            self.log_constant(
                LOG_SERVICE_VALIDATION_DEBUG,
                message="training_config_valid",
                extra={"run_id": config.run_id},
            )
            return config
        except ValidationError as exc:
            error_msg = f"Config validation error: {exc}"
            self._validation_errors.append(error_msg)
            self.log_constant(
                LOG_SERVICE_VALIDATION_ERROR,
                message=error_msg,
                extra={"phase": "training_config"},
            )
            if self.strict_mode:
                raise
            return None
        except Exception as exc:  # pragma: no cover - unexpected path
            error_msg = f"Unexpected config validation error: {exc}"
            self._validation_errors.append(error_msg)
            self.log_constant(
                LOG_SERVICE_VALIDATION_ERROR,
                message=error_msg,
                extra={"phase": "training_config", "error_type": type(exc).__name__},
                exc_info=exc,
            )
            if self.strict_mode:
                raise
            return None

    # ------------------------------------------------------------------
    # Lightweight step validation
    # ------------------------------------------------------------------
    def validate_step_data(
        self,
        *,
        episode: int,
        step: int,
        action: int,
        reward: float,
        state: int,
        next_state: int,
    ) -> bool:
        try:
            if episode < 0:
                raise ValueError("episode must be non-negative")
            if step < 0:
                raise ValueError("step must be non-negative")
            if action < 0:
                raise ValueError("action must be non-negative")
            if state < 0:
                raise ValueError("state must be non-negative")
            if next_state < 0:
                raise ValueError("next_state must be non-negative")

            self.log_constant(
                LOG_SERVICE_VALIDATION_DEBUG,
                message="step_data_valid",
                extra={
                    "episode": episode,
                    "step": step,
                    "action": action,
                    "reward": reward,
                    "state": state,
                    "next_state": next_state,
                },
            )
            return True
        except ValueError as exc:
            warning_msg = f"Step data validation warning: {exc}"
            self._validation_warnings.append(warning_msg)
            self.log_constant(
                LOG_SERVICE_VALIDATION_WARNING,
                message=warning_msg,
                extra={"phase": "step_validation"},
            )
            return False

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def get_validation_errors(self) -> list[str]:
        return self._validation_errors.copy()

    def get_validation_warnings(self) -> list[str]:
        return self._validation_warnings.copy()

    def clear_errors(self) -> None:
        self._validation_errors.clear()

    def clear_warnings(self) -> None:
        self._validation_warnings.clear()


__all__ = ["ValidationService"]
