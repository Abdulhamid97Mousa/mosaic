"""Filtered QWebEnginePage to suppress noisy JavaScript console warnings.

This module provides a custom QWebEnginePage that filters out known harmless
JavaScript warnings from embedded web views (TensorBoard, WANDB, etc.) while
still allowing important errors through for debugging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

try:
    from PyQt6.QtWebEngineCore import QWebEnginePage
except ImportError:
    QWebEnginePage = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from PyQt6.QtWebEngineCore import QWebEnginePage as QWebEnginePageType

_LOGGER = logging.getLogger(__name__)

# Known harmless warnings to suppress from embedded web content
_SUPPRESSED_PATTERNS: frozenset[str] = frozenset({
    # Three.js bundling issues in TensorBoard - harmless, just noisy
    "Multiple instances of Three.js",
    "THREE.WebGLRenderer",
    # Common Chromium rendering fallback messages
    "GBM is not supported",
    "Fallback to Vulkan",
})


class FilteredWebEnginePage(QWebEnginePage):  # type: ignore[misc]
    """QWebEnginePage subclass that filters noisy JavaScript console messages.

    By default, Qt's QWebEngineView forwards all JavaScript console.log(),
    console.warn(), and console.error() calls to the application's stdout
    with a 'js:' prefix. This clutters the terminal with harmless warnings
    from third-party libraries like Three.js used by TensorBoard.

    This class intercepts those messages and:
    - Always passes through JavaScript errors (they might indicate real bugs)
    - Suppresses known harmless warnings matching _SUPPRESSED_PATTERNS
    - Optionally logs filtered messages at DEBUG level for troubleshooting
    """

    def __init__(self, parent=None, *, log_filtered: bool = False) -> None:
        """Initialize the filtered page.

        Args:
            parent: Parent QObject (typically the QWebEngineView).
            log_filtered: If True, log filtered messages at DEBUG level.
        """
        super().__init__(parent)
        self._log_filtered = log_filtered

    def javaScriptConsoleMessage(
        self,
        level: "QWebEnginePageType.JavaScriptConsoleMessageLevel",
        message: str,
        line: int,
        source: str,
    ) -> None:
        """Handle JavaScript console messages with filtering.

        Args:
            level: Message severity (Info, Warning, Error).
            message: The console message text.
            line: Line number in the source file.
            source: Source file URL.
        """
        # Always let errors through - they might indicate real problems
        if QWebEnginePage is not None:
            error_level = QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel
            if level == error_level:
                super().javaScriptConsoleMessage(level, message, line, source)
                return

        # Check if this message matches any suppressed pattern
        for pattern in _SUPPRESSED_PATTERNS:
            if pattern in message:
                if self._log_filtered:
                    _LOGGER.debug(
                        "Filtered JS console message: %s (source: %s:%d)",
                        message[:100],
                        source,
                        line,
                    )
                return

        # Let everything else through
        super().javaScriptConsoleMessage(level, message, line, source)


# Only export if QWebEnginePage is available
if QWebEnginePage is not None:
    __all__ = ["FilteredWebEnginePage"]
else:
    __all__ = []
