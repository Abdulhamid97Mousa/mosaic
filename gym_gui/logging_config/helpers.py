"""Convenience helpers for emitting structured log constants."""

from __future__ import annotations

from typing import Any, Mapping
import logging

from .log_constants import LogConstant


def log_constant(
    logger: logging.Logger,
    constant: LogConstant,
    *,
    message: str | None = None,
    extra: Mapping[str, Any] | None = None,
    exc_info: BaseException | tuple | None = None,
) -> None:
    """Log a :class:`LogConstant` with shared structured metadata.

    Parameters
    ----------
    logger:
        Target logger instance.
    constant:
        Structured log constant describing level/component/subcomponent.
    message:
        Optional additional detail appended after the constant's base message.
    extra:
        Mapping merged into the log record's ``extra`` context.
    exc_info:
        Exception info to attach for stack traces.
    """

    # Reserved LogRecord attributes that cannot be overwritten via extra
    _RESERVED_LOG_KEYS = {
        "name", "msg", "args", "created", "filename", "funcName", "levelname",
        "levelno", "lineno", "module", "msecs", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "exc_info", "exc_text",
        "thread", "threadName", "taskName", "message", "asctime",
    }

    payload: dict[str, Any] = {
        "log_code": constant.code,
        "component": constant.component,
        "subcomponent": constant.subcomponent,
        "tags": ",".join(constant.tags),
    }
    if extra:
        # Filter out reserved keys to avoid LogRecord conflicts
        safe_extra = {k: v for k, v in extra.items() if k not in _RESERVED_LOG_KEYS}
        payload.update(safe_extra)

    text = constant.message if message is None else f"{constant.message} | {message}"
    level = getattr(logging, constant.level) if isinstance(constant.level, str) else constant.level
    logger.log(level, "%s %s", constant.code, text, extra=payload, exc_info=exc_info)


class LogConstantMixin:
    """Mixin providing ``log_constant`` convenience on ``self._logger``."""

    _logger: logging.Logger

    def log_constant(
        self,
        constant: LogConstant,
        *,
        message: str | None = None,
        extra: Mapping[str, Any] | None = None,
        exc_info: BaseException | tuple | None = None,
    ) -> None:
        log_constant(self._logger, constant, message=message, extra=extra, exc_info=exc_info)


__all__ = ["log_constant", "LogConstantMixin"]

