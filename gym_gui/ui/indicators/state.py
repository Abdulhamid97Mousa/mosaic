from __future__ import annotations

"""Shared indicator state primitives used across UI surfaces."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, MutableMapping, Optional


class IndicatorSeverity(str, Enum):
    """Severity ladder for UI indicators (info → critical)."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def escalates(self, other: "IndicatorSeverity") -> bool:
        order = (self.INFO, self.WARNING, self.ERROR, self.CRITICAL)
        return order.index(other) > order.index(self)


@dataclass(slots=True)
class IndicatorState:
    """Describes the status of a run/agent for indicator rendering."""

    run_id: str
    severity: IndicatorSeverity
    message: str
    details: Optional[str] = None
    badges: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_payload(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "run_id": self.run_id,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        if self.badges:
            payload["badges"] = self.badges
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def escalate(self, *, severity: IndicatorSeverity, message: Optional[str] = None) -> "IndicatorState":
        if severity.escalates(self.severity):
            return IndicatorState(
                run_id=self.run_id,
                severity=severity,
                message=message or self.message,
                details=self.details,
                badges=self.badges,
                metadata=self.metadata,
            )
        return self
