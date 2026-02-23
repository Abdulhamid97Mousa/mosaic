"""Centralized error handling utilities."""

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable, Optional, TypeVar

from qtpy import QtWidgets

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorHandler:
    """Centralized error handling for the application."""

    @staticmethod
    def handle_exception(
        exc: Exception,
        context: str = "Unknown",
        show_dialog: bool = True,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Handle an exception with logging and optional dialog.
        
        Args:
            exc: Exception to handle
            context: Context description for logging
            show_dialog: Whether to show error dialog
            parent: Parent widget for dialog
        """
        error_msg = f"{context}: {exc}"
        logger.error(error_msg, exc_info=True)

        if show_dialog and parent:
            QtWidgets.QMessageBox.critical(
                parent,
                "Error",
                f"{context}\n\n{str(exc)}",
            )

    @staticmethod
    def handle_warning(
        message: str,
        context: str = "Warning",
        show_dialog: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Handle a warning with logging and optional dialog.
        
        Args:
            message: Warning message
            context: Context description
            show_dialog: Whether to show warning dialog
            parent: Parent widget for dialog
        """
        logger.warning(f"{context}: {message}")

        if show_dialog and parent:
            QtWidgets.QMessageBox.warning(
                parent,
                context,
                message,
            )

    @staticmethod
    def handle_info(
        message: str,
        context: str = "Info",
        show_dialog: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Handle an info message with logging and optional dialog.
        
        Args:
            message: Info message
            context: Context description
            show_dialog: Whether to show info dialog
            parent: Parent widget for dialog
        """
        logger.info(f"{context}: {message}")

        if show_dialog and parent:
            QtWidgets.QMessageBox.information(
                parent,
                context,
                message,
            )

    @staticmethod
    def safe_call(
        func: Callable[..., T],
        *args: Any,
        context: str = "Function call",
        default: Optional[T] = None,
        show_dialog: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
        **kwargs: Any,
    ) -> Optional[T]:
        """Safely call a function with error handling.
        
        Args:
            func: Function to call
            *args: Positional arguments
            context: Context description
            default: Default return value on error
            show_dialog: Whether to show error dialog
            parent: Parent widget for dialog
            **kwargs: Keyword arguments
            
        Returns:
            Function result or default value
        """
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            ErrorHandler.handle_exception(
                exc,
                context=context,
                show_dialog=show_dialog,
                parent=parent,
            )
            return default

    @staticmethod
    def get_traceback_string(exc: Exception) -> str:
        """Get formatted traceback string for an exception.
        
        Args:
            exc: Exception to format
            
        Returns:
            Formatted traceback string
        """
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    @staticmethod
    def log_traceback(exc: Exception, context: str = "Exception") -> None:
        """Log full traceback for an exception.
        
        Args:
            exc: Exception to log
            context: Context description
        """
        tb_str = ErrorHandler.get_traceback_string(exc)
        logger.error(f"{context}:\n{tb_str}")


class ErrorContext:
    """Context manager for error handling."""

    def __init__(
        self,
        context: str = "Operation",
        show_dialog: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
        reraise: bool = False,
    ) -> None:
        """Initialize error context.
        
        Args:
            context: Context description
            show_dialog: Whether to show error dialog
            parent: Parent widget for dialog
            reraise: Whether to reraise exception
        """
        self.context = context
        self.show_dialog = show_dialog
        self.parent = parent
        self.reraise = reraise
        self.exception: Optional[Exception] = None

    def __enter__(self) -> ErrorContext:
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context and handle exception if occurred."""
        if exc_type is not None and issubclass(exc_type, Exception):
            self.exception = exc_val
            ErrorHandler.handle_exception(
                exc_val,
                context=self.context,
                show_dialog=self.show_dialog,
                parent=self.parent,
            )
            return not self.reraise
        return True


__all__ = ["ErrorHandler", "ErrorContext"]

