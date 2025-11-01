"""Central logging utilities for the Gym GUI project."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from logging.config import dictConfig
from typing import Any, Dict, Iterable, Tuple

from gym_gui.config.paths import VAR_LOGS_DIR, ensure_var_directories

ensure_var_directories()
LOG_DIR = VAR_LOGS_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Format includes correlation IDs and component metadata for tracing
_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s | "
    "[comp=%(component)s sub=%(subcomponent)s run=%(run_id)s agent=%(agent_id)s code=%(log_code)s tags=%(tags)s] | %(message)s"
)


@dataclass(frozen=True)
class RegisteredLogger:
    name: str
    component: str
    description: str | None = None


class _ComponentRegistry:
    """Tracks logger → component mappings and observed severities."""

    def __init__(self) -> None:
        self._prefix_map: Dict[str, RegisteredLogger] = {}
        self._observed_components: Dict[str, set[str]] = {}
        self._default_component = "Unknown"
        self._register_default_prefixes()

    def _register_default_prefixes(self) -> None:
        defaults = {
            "gym_gui.ui": "UI",
            "gym_gui.controllers": "Controller",
            "gym_gui.core.adapters": "Adapter",
            "gym_gui.core": "Core",
            "gym_gui.services": "Service",
            "gym_gui.telemetry": "Telemetry",
            "gym_gui.logging": "Logging",
            "spade_bdi_worker": "Worker",
        }
        for prefix, component in defaults.items():
            self.register_prefix(prefix, component)

    def register_prefix(self, prefix: str, component: str, *, description: str | None = None) -> None:
        self._prefix_map[prefix] = RegisteredLogger(prefix, component, description)

    def resolve(self, logger_name: str) -> str:
        best_match_len = -1
        component = self._default_component
        for prefix, entry in self._prefix_map.items():
            if logger_name.startswith(prefix) and len(prefix) > best_match_len:
                component = entry.component
                best_match_len = len(prefix)
        return component

    def observe(self, component: str, severity: str) -> None:
        self._observed_components.setdefault(component, set()).add(severity)

    def snapshot(self) -> Dict[str, tuple[str, ...]]:
        return {
            component: tuple(sorted(levels))
            for component, levels in sorted(self._observed_components.items())
        }

    def observed_components(self) -> tuple[str, ...]:
        keys = set(self._observed_components.keys()) | {entry.component for entry in self._prefix_map.values()}
        return tuple(sorted(keys))

    def reset_observations(self) -> None:
        self._observed_components.clear()


COMPONENT_REGISTRY = _ComponentRegistry()


def register_component_prefix(prefix: str, component: str, *, description: str | None = None) -> None:
    """Expose registry for modules that declare new logger namespaces."""

    COMPONENT_REGISTRY.register_prefix(prefix, component, description=description)


def get_component_snapshot() -> Dict[str, tuple[str, ...]]:
    """Return observed component → severity mappings for GUI filters."""

    return COMPONENT_REGISTRY.snapshot()


def list_known_components() -> tuple[str, ...]:
    """Return all known component labels (observed or default)."""

    return COMPONENT_REGISTRY.observed_components()


class CustomFormatter(logging.Formatter):
    """Formatter that includes optional fields supplied via ``extra``."""

    def format(self, record: logging.LogRecord) -> str:
        line = getattr(record, "line", None)
        if line:
            record.msg = f"{record.msg} | line={line}"
        return super().format(record)


class CorrelationIdFilter(logging.Filter):
    """Ensure ``run_id`` and ``agent_id`` keys exist on every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = "unknown"
        if not hasattr(record, "agent_id"):
            record.agent_id = "unknown"
        if not hasattr(record, "log_code"):
            record.log_code = None  # Used by log inspectors in UI
        if not hasattr(record, "tags"):
            record.tags = "-"
        return True


class ComponentFilter(logging.Filter):
    """Annotate records with component/subcomponent metadata and track usage."""

    def filter(self, record: logging.LogRecord) -> bool:
        component = getattr(record, "component", None)
        if component is None:
            component = COMPONENT_REGISTRY.resolve(record.name)
            record.component = component
        else:
            COMPONENT_REGISTRY.register_prefix(record.name, component)

        subcomponent = getattr(record, "subcomponent", None)
        if subcomponent is None:
            subcomponent = getattr(record, "subcategory", None) or "-"
            record.subcomponent = subcomponent

        severity = getattr(record, "levelname", "INFO")
        COMPONENT_REGISTRY.observe(component, severity)
        return True


class GrpcBlockingIOFilter(logging.Filter):
    """Filter out noisy gRPC BlockingIOError warnings from asyncio."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "asyncio" and record.levelno == logging.ERROR:
            msg = record.getMessage()
            if "BlockingIOError" in msg and "PollerCompletionQueue" in msg:
                return False
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

_ACTIVE_CONFIG: Tuple[bool, bool] | None = None


def _level_name(level: int) -> str:
    return logging.getLevelName(level) if isinstance(level, int) else str(level)


def _project_loggers() -> Iterable[str]:
    return ("gym_gui", "spade_bdi_worker")


def configure_logging(
    level: int = logging.INFO,
    *,
    stream: bool = True,
    log_to_file: bool = True,
    force: bool = False,
) -> None:
    """Configure logging once per process using dictConfig.

    ``level`` controls the verbosity of Gym GUI packages while the root logger
    remains conservative (WARNING) unless ``level`` is DEBUG.
    """

    global _ACTIVE_CONFIG

    debug_enabled = level <= logging.DEBUG
    root_level = logging.DEBUG if debug_enabled else logging.WARNING
    project_level_name = _level_name(level)

    config_key = (stream, log_to_file)
    if _ACTIVE_CONFIG == config_key and not force:
        # Update project logger level without rebuilding handlers
        for name in _project_loggers():
            logging.getLogger(name).setLevel(level)
        return

    COMPONENT_REGISTRY.reset_observations()

    handlers: Dict[str, Dict[str, Any]] = {}
    root_handlers: list[str] = []

    if stream:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": project_level_name,
            "formatter": "structured",
            "filters": ["correlation", "component"],
        }
        root_handlers.append("console")

    if log_to_file:
        log_file = LOG_DIR / "gym_gui.log"
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": project_level_name,
            "formatter": "structured",
            "filters": ["correlation", "component"],
            "filename": str(log_file),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
        }
        root_handlers.append("file")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": "gym_gui.logging_config.logger.CustomFormatter",
                "format": _DEFAULT_FORMAT,
            }
        },
        "filters": {
            "correlation": {
                "()": "gym_gui.logging_config.logger.CorrelationIdFilter",
            },
            "grpc_blocking_io": {
                "()": "gym_gui.logging_config.logger.GrpcBlockingIOFilter",
            },
            "component": {
                "()": "gym_gui.logging_config.logger.ComponentFilter",
            },
        },
        "handlers": handlers,
        "loggers": {
            "asyncio": {
                "level": "WARNING",
                "propagate": True,
                "filters": ["grpc_blocking_io"],
            },
        },
        "root": {
            "level": _level_name(root_level),
            "handlers": root_handlers,
        },
    }

    dictConfig(config)

    for name in _project_loggers():
        logging.getLogger(name).setLevel(level)

    _ACTIVE_CONFIG = config_key


__all__ = [
    "configure_logging",
    "CorrelationIdAdapter",
    "CorrelationIdFilter",
    "ComponentFilter",
    "CustomFormatter",
    "COMPONENT_REGISTRY",
    "register_component_prefix",
    "get_component_snapshot",
    "list_known_components",
    "GrpcBlockingIOFilter",
    "LOG_DIR",
]
