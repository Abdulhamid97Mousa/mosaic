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
        payload = LogRecordPayload(
            level=record.levelname,
            name=record.name,
            message=msg,
            created=record.created,
        )
        self.emitter.record_emitted.emit(payload)


__all__ = ["QtLogHandler", "LogEmitter", "LogRecordPayload"]
