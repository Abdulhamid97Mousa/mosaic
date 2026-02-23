"""QML integration module for Qt Quick support.

This module contains:
- FastLaneItem: QQuickPaintedItem for fast frame rendering
- FastLaneView.qml: QML declarative view using FastLaneItem
"""

from gym_gui.ui.qml.fastlane_item import FastLaneItem

__all__ = ["FastLaneItem"]
