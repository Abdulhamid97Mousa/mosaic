"""Telemetry schema validation for different agent types.

Validates incoming telemetry records against expected agent schemas
and logs mismatches with detailed error information.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import logging

from gym_gui.core.agent_config import get_agent_config, AgentConfig

_LOGGER = logging.getLogger(__name__)


class TelemetrySchemaValidator:
    """Validates telemetry records against agent-specific schemas."""

    def __init__(self, agent_type: str = "Headless"):
        """Initialize validator for given agent type.
        
        Args:
            agent_type: Type of agent ("BDI", "Headless", etc.)
        """
        self.agent_type = agent_type
        self.agent_config = get_agent_config(agent_type)
        self.validation_count = 0
        self.validation_errors = 0
        self.validation_warnings = 0

        _LOGGER.debug(
            f"TelemetrySchemaValidator initialized",
            extra={
                "agent_type": agent_type,
                "config_class": self.agent_config.__class__.__name__,
                "required_fields": sorted(self.agent_config.get_required_fields()),
                "optional_fields": sorted(self.agent_config.get_optional_fields()),
            },
        )

    def validate_step_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a step record and return validation result with metadata.
        
        Args:
            record: StepRecord dict to validate
            
        Returns:
            Dict with keys:
            - is_valid: bool, True if all required fields present
            - issues: list[str], list of validation issues
            - agent_type: str, agent type being validated against
            - validation_count: int, cumulative validation count
            - required_fields_found: list[str], required fields that were present
            - required_fields_missing: list[str], required fields that were missing
            - optional_fields_found: list[str], optional fields that were present
        """
        self.validation_count += 1
        is_valid, issues = self.agent_config.validate_step_record(record)

        record_keys = set(record.keys())
        required = self.agent_config.get_required_fields()
        optional = self.agent_config.get_optional_fields()

        required_found = required & record_keys
        required_missing = required - record_keys
        optional_found = optional & record_keys

        result = {
            "is_valid": is_valid,
            "issues": issues,
            "agent_type": self.agent_type,
            "validation_count": self.validation_count,
            "required_fields_found": sorted(required_found),
            "required_fields_missing": sorted(required_missing),
            "optional_fields_found": sorted(optional_found),
        }

        if not is_valid:
            self.validation_errors += 1
            _LOGGER.error(
                f"StepRecord validation FAILED for {self.agent_type}",
                extra={
                    **result,
                    "all_issues": issues,
                    "total_errors": self.validation_errors,
                    "total_validations": self.validation_count,
                    "error_rate": f"{(self.validation_errors / self.validation_count * 100):.1f}%",
                },
            )
        elif issues:
            self.validation_warnings += 1
            _LOGGER.warning(
                f"StepRecord validation WARNING for {self.agent_type}",
                extra={
                    **result,
                    "all_issues": issues,
                    "total_warnings": self.validation_warnings,
                    "total_validations": self.validation_count,
                },
            )

        return result

    def validate_episode_rollup(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an episode rollup record.
        
        Similar to step record validation but for aggregate episode data.
        """
        result = self.agent_config.validate_episode_rollup(record)
        is_valid = result[0]
        issues = result[1]

        record_keys = set(record.keys())
        required = self.agent_config.get_required_fields()
        optional = self.agent_config.get_optional_fields()

        required_found = required & record_keys
        required_missing = required - record_keys
        optional_found = optional & record_keys

        result_dict = {
            "is_valid": is_valid,
            "issues": issues,
            "agent_type": self.agent_type,
            "validation_count": self.validation_count,
            "required_fields_found": sorted(required_found),
            "required_fields_missing": sorted(required_missing),
            "optional_fields_found": sorted(optional_found),
        }

        if not is_valid:
            self.validation_errors += 1
            _LOGGER.error(
                f"EpisodeRollup validation FAILED for {self.agent_type}",
                extra={**result_dict, "all_issues": issues},
            )

        return result_dict

    def get_schema_info(self) -> Dict[str, Any]:
        """Return schema information for this agent type."""
        return {
            "agent_type": self.agent_type,
            "schema": self.agent_config.get_telemetry_schema(),
            "required_fields": sorted(self.agent_config.get_required_fields()),
            "optional_fields": sorted(self.agent_config.get_optional_fields()),
            "validation_count": self.validation_count,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
        }


def get_telemetry_validator(agent_type: str = "Headless") -> TelemetrySchemaValidator:
    """Factory function to get validator for agent type."""
    return TelemetrySchemaValidator(agent_type)


__all__ = [
    "TelemetrySchemaValidator",
    "get_telemetry_validator",
]
