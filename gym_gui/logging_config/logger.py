"""Central logging utilities for the Gym GUI project."""

from __future__ import annotations

import logging

from gym_gui.config.paths import VAR_LOGS_DIR, ensure_var_directories

ensure_var_directories()
LOG_DIR = VAR_LOGS_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, stream: bool = True, *, log_to_file: bool = True) -> None:
    """Configure root logging handlers for the application."""

    root = logging.getLogger()
    root.handlers.clear()

    handlers: list[logging.Handler] = []
    if stream:
        handlers.append(logging.StreamHandler())

    if log_to_file:
        log_file = LOG_DIR / "gym_gui.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=_DEFAULT_FORMAT, handlers=handlers)


__all__ = ["configure_logging", "LOG_DIR"]
