from __future__ import annotations

"""Utility helpers that interact with Qt primitives safely."""

from contextlib import contextmanager
from typing import Iterator

@contextmanager
def busy_cursor(app) -> Iterator[None]:
    """Temporarily set a busy cursor during long-running operations."""

    from qtpy.QtGui import QCursor  # type: ignore
    from qtpy.QtCore import Qt  # type: ignore

    previous = app.overrideCursor()
    # Qt6 uses Qt.CursorShape.WaitCursor, Qt5 uses Qt.WaitCursor
    try:
        wait_cursor = Qt.CursorShape.WaitCursor
    except AttributeError:
        wait_cursor = Qt.WaitCursor  # type: ignore
    
    app.setOverrideCursor(QCursor(wait_cursor))
    try:
        yield
    finally:
        app.restoreOverrideCursor()
        if previous is not None:
            app.setOverrideCursor(previous)


__all__ = ["busy_cursor"]
