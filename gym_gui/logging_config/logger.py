"""Central logging utilities for the Gym GUI project."""

from __future__ import annotations

import logging
from typing import Any, Optional

from gym_gui.config.paths import VAR_LOGS_DIR, ensure_var_directories

ensure_var_directories()
LOG_DIR = VAR_LOGS_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Format includes correlation IDs for tracing across components
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | [run=%(run_id)s agent=%(agent_id)s] | %(message)s"

# Custom formatter that includes optional fields from extra dict
class _CustomFormatter(logging.Formatter):
    """Formatter that includes optional fields from the extra dict."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record, including optional fields."""
        # Add optional fields to the message if they exist
        line = getattr(record, 'line', None)
        if line:
            record.msg = f"{record.msg} | line={line}"
        return super().format(record)


class _CorrelationIdFilter(logging.Filter):
    """
    Logging filter that injects default correlation IDs into log records.

    This ensures that ALL log records have run_id and agent_id fields,
    preventing KeyError when the formatter tries to use them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Inject default correlation IDs if not already present."""
        if not hasattr(record, 'run_id'):
            record.run_id = 'unknown'
        if not hasattr(record, 'agent_id'):
            record.agent_id = 'unknown'
        return True


class CorrelationIdAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that injects correlation IDs (run_id, agent_id) into all log records.

    This enables tracing of a single training run across GUI, daemon, and worker processes.

    Usage:
        logger = logging.getLogger(__name__)
        adapter = CorrelationIdAdapter(logger, {"run_id": "run_123", "agent_id": "agent_1"})
        adapter.info("Training started")  # Logs with correlation IDs
    """

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Inject correlation IDs into the extra dict of every log record."""
        extra = kwargs.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}

        # Merge adapter context with per-call extra dict
        # Per-call extra takes precedence over adapter context
        merged_extra: dict[str, Any] = {}
        if isinstance(self.extra, dict):
            merged_extra.update(self.extra)
        merged_extra.update(extra)

        # Ensure required correlation fields exist
        if "run_id" not in merged_extra:
            merged_extra["run_id"] = "unknown"
        if "agent_id" not in merged_extra:
            merged_extra["agent_id"] = "unknown"

        kwargs["extra"] = merged_extra
        return msg, kwargs


class _GrpcBlockingIOFilter(logging.Filter):
    """Filter out non-fatal gRPC BlockingIOError warnings from asyncio logger."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Suppress "BlockingIOError: [Errno 11] Resource temporarily unavailable" 
        # from PollerCompletionQueue - these are harmless gRPC/asyncio interactions
        if record.name == "asyncio" and record.levelno == logging.ERROR:
            msg = record.getMessage()
            if "BlockingIOError" in msg and "PollerCompletionQueue" in msg:
                return False
        return True


def configure_logging(level: int = logging.INFO, stream: bool = True, *, log_to_file: bool = True) -> None:
    """
    Configure root logging handlers for the application.

    Args:
        level: Logging level (default: logging.INFO)
        stream: Whether to log to stdout/stderr (default: True)
        log_to_file: Whether to log to rotating file (default: True)
    """
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger()
    root.handlers.clear()

    formatter = _CustomFormatter(_DEFAULT_FORMAT)
    handlers: list[logging.Handler] = []

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    if log_to_file:
        log_file = LOG_DIR / "gym_gui.log"
        # Use RotatingFileHandler: 10 MB per file, keep 5 backups (50 MB total)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger with handlers
    root.setLevel(level)
    for handler in handlers:
        root.addHandler(handler)

    # Add correlation ID filter to ALL handlers to inject defaults
    correlation_filter = _CorrelationIdFilter()
    for handler in root.handlers:
        handler.addFilter(correlation_filter)

    # Add filter to suppress non-fatal gRPC asyncio warnings
    grpc_filter = _GrpcBlockingIOFilter()
    asyncio_logger = logging.getLogger("asyncio")
    for handler in asyncio_logger.handlers or root.handlers:
        handler.addFilter(grpc_filter)


__all__ = ["configure_logging", "CorrelationIdAdapter", "LOG_DIR"]
