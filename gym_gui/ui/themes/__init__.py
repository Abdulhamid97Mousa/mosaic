"""Theme management for MOSAIC GUI.

Provides functions to load and apply Qt stylesheets (QSS).

Usage:
    >>> from gym_gui.ui.themes import load_theme, apply_theme, DARK_THEME, LIGHT_THEME
    >>> apply_theme(DARK_THEME)  # Apply dark theme
    >>> apply_theme(LIGHT_THEME)  # Reset to system default
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from qtpy import QtWidgets

_LOGGER = logging.getLogger(__name__)

# Theme directory
THEMES_DIR = Path(__file__).parent

# Theme constants
DARK_THEME = "dark"
LIGHT_THEME = "light"


def get_theme_path(theme_name: str) -> Path:
    """Get the path to a theme file.

    Args:
        theme_name: Name of the theme (e.g., 'dark', 'light').

    Returns:
        Path to the .qss file.
    """
    return THEMES_DIR / f"{theme_name}.qss"


def load_theme(theme_name: str) -> Optional[str]:
    """Load a theme stylesheet from file.

    Args:
        theme_name: Name of the theme to load.

    Returns:
        Stylesheet content as string, or None if not found.
    """
    theme_path = get_theme_path(theme_name)

    if not theme_path.exists():
        _LOGGER.warning("Theme file not found: %s", theme_path)
        return None

    try:
        return theme_path.read_text(encoding="utf-8")
    except Exception as e:
        _LOGGER.error("Failed to load theme %s: %s", theme_name, e)
        return None


def apply_theme(theme_name: str) -> bool:
    """Apply a theme to the application.

    Args:
        theme_name: Name of the theme ('dark', 'light', or custom).

    Returns:
        True if theme was applied successfully.
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        _LOGGER.error("No QApplication instance found")
        return False

    if theme_name == LIGHT_THEME:
        # Light theme = reset to system default
        app.setStyleSheet("")
        _LOGGER.info("Applied light theme (system default)")
        return True

    stylesheet = load_theme(theme_name)
    if stylesheet is None:
        _LOGGER.warning("Could not load theme: %s", theme_name)
        return False

    app.setStyleSheet(stylesheet)
    _LOGGER.info("Applied theme: %s", theme_name)
    return True


__all__ = [
    "THEMES_DIR",
    "DARK_THEME",
    "LIGHT_THEME",
    "get_theme_path",
    "load_theme",
    "apply_theme",
]
