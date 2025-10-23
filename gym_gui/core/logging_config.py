"""Centralized logging configuration for the application."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


class LoggingConfig:
    """Centralized logging configuration."""

    # Log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    # Default configuration
    DEFAULT_LEVEL = logging.INFO
    DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def setup_root_logger(
        level: int = DEFAULT_LEVEL,
        log_file: Optional[Path] = None,
        format_string: str = DEFAULT_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
    ) -> logging.Logger:
        """Setup root logger with console and optional file handlers.
        
        Args:
            level: Logging level
            log_file: Optional path to log file
            format_string: Log message format
            date_format: Date format for log messages
            
        Returns:
            Configured root logger
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter(format_string, datefmt=date_format)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        return root_logger

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger with the given name.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)

    @staticmethod
    def set_level(level: int, logger_name: Optional[str] = None) -> None:
        """Set logging level for a specific logger or root logger.
        
        Args:
            level: Logging level
            logger_name: Logger name (None for root logger)
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)

    @staticmethod
    def suppress_logger(logger_name: str) -> None:
        """Suppress logging for a specific logger.
        
        Args:
            logger_name: Logger name to suppress
        """
        logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

    @staticmethod
    def enable_debug_logging() -> None:
        """Enable debug logging for all loggers."""
        logging.getLogger().setLevel(logging.DEBUG)

    @staticmethod
    def disable_debug_logging() -> None:
        """Disable debug logging (set to INFO)."""
        logging.getLogger().setLevel(logging.INFO)


__all__ = ["LoggingConfig"]

