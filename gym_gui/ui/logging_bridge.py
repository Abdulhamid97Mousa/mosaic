from __future__ import annotations

"""Utilities for streaming Python logging records into Qt widgets."""

import logging
from dataclasses import dataclass
from typing import Optional

from qtpy import QtCore


@dataclass(slots=True)
class LogRecordPayload:
    level: str
    name: str
    message: str
    created: float
    component: str
    subcomponent: str
    log_code: str | None


class LogEmitter(QtCore.QObject):
    """Qt object that emits logging payloads across threads safely."""

    record_emitted = QtCore.Signal(LogRecordPayload)  # type: ignore[attr-defined]


class QtLogHandler(logging.Handler):
    """Logging handler that forwards records to a Qt signal."""

    def __init__(self, *, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__()
        self.emitter = LogEmitter(parent)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI only
        try:
            msg = self.format(record)
        except Exception:  # pragma: no cover - formatting errors
            msg = record.getMessage()
        component = getattr(record, "component", "Unknown")
        subcomponent = getattr(record, "subcomponent", "-")
        log_code = getattr(record, "log_code", None)
        if log_code is not None:
            log_code = str(log_code)

        payload = LogRecordPayload(
            level=record.levelname,
            name=record.name,
            message=msg,
            created=record.created,
            component=component,
            subcomponent=subcomponent,
            log_code=log_code,
        )
        self.emitter.record_emitted.emit(payload)


__all__ = ["QtLogHandler", "LogEmitter", "LogRecordPayload"]
