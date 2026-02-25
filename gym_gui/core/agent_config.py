"""Agent configuration abstraction layer with schema validation.

Provides base class and implementations for different agent types (Headless, etc.)
with telemetry schema definitions and validation logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Set
import logging

_LOGGER = logging.getLogger(__name__)


class AgentConfig(ABC):
    """Base abstraction for agent types with schema validation.
    
    Defines interface for agent-specific telemetry schema validation.
    Supports multiple agent implementations (Q-Learning, DQN, etc.)
    """

    @abstractmethod
    def get_agent_type(self) -> str:
        """Return agent type identifier (e.g., 'Headless')."""
        pass

    @abstractmethod
    def get_telemetry_schema(self) -> Dict[str, Any]:
        """Return expected telemetry schema fields with categories.
        
        Returns dict mapping category names to lists of expected field names.
        Categories: core_metrics, agent_state, positions, environment, etc.
        """
        pass

    @abstractmethod
    def get_required_fields(self) -> Set[str]:
        """Return set of required field names in telemetry records.
        
        These fields MUST be present in every StepRecord/EpisodeRollup,
        or validation will fail with error logging.
        """
        pass

    @abstractmethod
    def get_optional_fields(self) -> Set[str]:
        """Return set of optional field names in telemetry records.
        
        These fields may or may not be present. Their absence is not an error,
        but presence will be logged for debugging.
        """
        pass

    def validate_step_record(self, record: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate step record matches schema.
        
        Args:
            record: StepRecord dict to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
            is_valid=True means all required fields present
            list_of_issues contains descriptions of any problems found
        """
        required = self.get_required_fields()
        optional = self.get_optional_fields()
        record_keys = set(record.keys())
        issues: list[str] = []

        # Check for missing required fields
        missing = required - record_keys
        if missing:
            msg = f"Missing required fields for {self.get_agent_type()}: {missing}"
            issues.append(msg)
            _LOGGER.error(
                msg,
                extra={
                    "agent_type": self.get_agent_type(),
                    "missing_fields": sorted(missing),
                    "expected_required": sorted(required),
                    "record_keys": sorted(record_keys),
                },
            )

        # Check for unexpected fields (extra fields not in schema)
        schema_fields = required | optional
        extra = record_keys - schema_fields
        if extra:
            msg = f"Unexpected fields in {self.get_agent_type()} record: {extra}"
            issues.append(msg)
            _LOGGER.warning(
                msg,
                extra={
                    "agent_type": self.get_agent_type(),
                    "extra_fields": sorted(extra),
                    "expected_schema": sorted(schema_fields),
                },
            )

        if not issues:
            _LOGGER.debug(
                f"StepRecord validation passed for {self.get_agent_type()}",
                extra={
                    "agent_type": self.get_agent_type(),
                    "required_fields": sorted(required),
                    "optional_fields": sorted(optional),
                    "record_keys": sorted(record_keys),
                },
            )

        return (len(issues) == 0 or not missing, issues)

    def validate_episode_rollup(self, record: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate episode rollup record matches schema.
        
        Episode rollups have similar but slightly different schema than step records.
        Default implementation reuses step validation; subclasses can override.
        """
        return self.validate_step_record(record)


class HeadlessAgentConfig(AgentConfig):
    """Configuration for headless (Q-Learning, DQN, etc.) agents.

    Standard RL algorithms.
    Telemetry includes algorithm-specific metrics like Q-values, epsilon, etc.
    """

    def get_agent_type(self) -> str:
        return "Headless"

    def get_telemetry_schema(self) -> Dict[str, Any]:
        """Headless telemetry schema for standard RL algorithms."""
        return {
            "core_metrics": [
                "episode",
                "step",
                "reward",
                "done",
                "observation",
                "action",
                "timestamp",  # Add timestamp to schema
                "ts",  # Also support 'ts' field from JSONL emitter
            ],
            "algorithm": [
                "q_value",
                "learning_rate",
                "epsilon",
                "td_error",
            ],
            "positions": [
                "x",
                "y",
                "position",
            ],
            "environment": [
                "grid_size",
                "goal",
                "game_id",
            ],
        }

    def get_required_fields(self) -> Set[str]:
        """Headless agents must provide core metrics."""
        return {
            "episode",
            "step",
            "reward",
            "done",
            "observation",
            "action",
        }

    def get_optional_fields(self) -> Set[str]:
        """Headless agents may provide algorithm metrics and positions."""
        return {
            "q_value",
            "learning_rate",
            "epsilon",
            "td_error",
            "x",
            "y",
            "position",
            "grid_size",
            "goal",
            "game_id",
            "timestamp",  # Timestamp is optional (may come from protobuf or JSONL)
            "ts",  # Alternative timestamp field from JSONL emitter
        }


def get_agent_config(agent_type: str) -> AgentConfig:
    """Factory to get agent config instance by type.
    
    Args:
        agent_type: Type identifier ("Headless", "Q-Learning", etc.)
        
    Returns:
        AgentConfig instance for the given type
        
    Raises:
        ValueError if agent_type is not recognized (logs warning, returns default)
    """
    agent_type_lower = agent_type.lower()

    if agent_type_lower in ["headless", "q-learning", "dqn", "ppo", "a2c"]:
        _LOGGER.info(f"Loading Headless agent configuration for: {agent_type}")
        return HeadlessAgentConfig()
    else:
        _LOGGER.warning(
            f"Unknown agent type, defaulting to Headless",
            extra={"agent_type": agent_type},
        )
        return HeadlessAgentConfig()


__all__ = [
    "AgentConfig",
    "HeadlessAgentConfig",
    "get_agent_config",
]
