"""Validation service for telemetry data and configuration."""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from pydantic import ValidationError

from gym_gui.core.validation import (
    validate_telemetry_event,
    TrainingConfig,
    TelemetryEventBase,
)
from gym_gui.logging_config.helpers import LogConstantMixin
from gym_gui.logging_config.log_constants import (
    LOG_SERVICE_VALIDATION_DEBUG,
    LOG_SERVICE_VALIDATION_WARNING,
    LOG_SERVICE_VALIDATION_ERROR,
)

if TYPE_CHECKING:  # pragma: no cover
    pass


class ValidationService(LogConstantMixin):
    """Service for validating telemetry data and configuration."""

    def __init__(self, strict_mode: bool = False) -> None:
        """Initialize validation service.
        
        Args:
            strict_mode: If True, raise exceptions on validation errors.
                        If False, log warnings and continue.
        """
        import logging

        self.strict_mode = strict_mode
        self._validation_errors: list[str] = []
        self._validation_warnings: list[str] = []
        self._logger = logging.getLogger("gym_gui.services.validation")

    def validate_telemetry_event(self, event_data: Dict[str, Any]) -> Optional[TelemetryEventBase]:
        """Validate a telemetry event.
        
        Args:
            event_data: Raw event data
            
        Returns:
            Validated event or None if validation failed
        """
        try:
            event = validate_telemetry_event(event_data)
            self.log_constant(
                LOG_SERVICE_VALIDATION_DEBUG,
                message="telemetry_event_valid",
                extra={"event_type": event.type},
            )
            return event
        except ValidationError as e:
            error_msg = f"Telemetry validation error: {e}"
            self._validation_errors.append(error_msg)
            self.log_constant(
                LOG_SERVICE_VALIDATION_ERROR,
                message=error_msg,
                extra={"phase": "telemetry_event"},
            )
            
            if self.strict_mode:
                raise
            return None
        except Exception as e:
            error_msg = f"Unexpected validation error: {e}"
            self._validation_errors.append(error_msg)
            self.log_constant(
                LOG_SERVICE_VALIDATION_ERROR,
                message=error_msg,
                extra={"phase": "telemetry_event", "error_type": type(e).__name__},
                exc_info=e,
            )
            
            if self.strict_mode:
                raise
            return None

    def validate_training_config(self, config_data: Dict[str, Any]) -> Optional[TrainingConfig]:
        """Validate training configuration.
        
        Args:
            config_data: Raw configuration data
            
        Returns:
            Validated config or None if validation failed
        """
        try:
            config = TrainingConfig(**config_data)
            self.log_constant(
                LOG_SERVICE_VALIDATION_DEBUG,
                message="training_config_valid",
                extra={"run_id": config.run_id},
            )
            return config
        except ValidationError as e:
            error_msg = f"Config validation error: {e}"
            self._validation_errors.append(error_msg)
            self.log_constant(
                LOG_SERVICE_VALIDATION_ERROR,
                message=error_msg,
                extra={"phase": "training_config"},
            )
            
            if self.strict_mode:
                raise
            return None
        except Exception as e:
            error_msg = f"Unexpected config validation error: {e}"
            self._validation_errors.append(error_msg)
            self.log_constant(
                LOG_SERVICE_VALIDATION_ERROR,
                message=error_msg,
                extra={"phase": "training_config", "error_type": type(e).__name__},
                exc_info=e,
            )
            
            if self.strict_mode:
                raise
            return None

    def validate_step_data(
        self,
        episode: int,
        step: int,
        action: int,
        reward: float,
        state: int,
        next_state: int,
    ) -> bool:
        """Quick validation of step data without full event creation.
        
        Args:
            episode: Episode number
            step: Step index
            action: Action taken
            reward: Reward received
            state: Current state
            next_state: Next state
            
        Returns:
            True if valid, False otherwise
        """
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
        except ValueError as e:
            warning_msg = f"Step data validation warning: {e}"
            self._validation_warnings.append(warning_msg)
            self.log_constant(
                LOG_SERVICE_VALIDATION_WARNING,
                message=warning_msg,
                extra={"phase": "step_validation"},
            )
            return False

    def get_validation_errors(self) -> list[str]:
        """Get list of validation errors."""
        return self._validation_errors.copy()

    def get_validation_warnings(self) -> list[str]:
        """Get list of validation warnings."""
        return self._validation_warnings.copy()

    def clear_errors(self) -> None:
        """Clear validation error history."""
        self._validation_errors.clear()

    def clear_warnings(self) -> None:
        """Clear validation warning history."""
        self._validation_warnings.clear()

    def reset(self) -> None:
        """Reset all validation state."""
        self.clear_errors()
        self.clear_warnings()

    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "error_count": len(self._validation_errors),
            "warning_count": len(self._validation_warnings),
            "strict_mode": self.strict_mode,
            "errors": self._validation_errors,
            "warnings": self._validation_warnings,
        }


__all__ = ["ValidationService"]
