"""Minimal test to check if LiveTelemetryTab actually disables rendering."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets
from gym_gui.ui.widgets.live_telemetry_tab import LiveTelemetryTab

def test_rendering_disabled():
    """Test that when live_render_enabled=False, no render group is created."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    
    # Create tab with rendering DISABLED
    tab = LiveTelemetryTab(
        run_id="run-test",
        agent_id="agent-test",
        live_render_enabled=False,
    )
    
    # Check if render_container is None (meaning rendering was disabled)
    assert tab._render_container is None, "render_container should be None when rendering is disabled!"
    assert tab._render_layout is None, "render_layout should be None when rendering is disabled!"
    assert tab._render_placeholder is None, "render_placeholder should be None when rendering is disabled!"
    
    tab.deleteLater()
    print("✓ Test passed: Rendering correctly disabled when live_render_enabled=False")

def test_rendering_enabled():
    """Test that when live_render_enabled=True, render group IS created."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    
    # Create tab with rendering ENABLED (default)
    tab = LiveTelemetryTab(
        run_id="run-test",
        agent_id="agent-test",
        live_render_enabled=True,
    )
    
    # Check if render_container is NOT None (meaning rendering was enabled)
    assert tab._render_container is not None, "render_container should NOT be None when rendering is enabled!"
    assert tab._render_layout is not None, "render_layout should NOT be None when rendering is enabled!"
    assert tab._render_placeholder is not None, "render_placeholder should NOT be None when rendering is enabled!"
    
    tab.deleteLater()
    print("✓ Test passed: Rendering correctly enabled when live_render_enabled=True")

if __name__ == "__main__":
    test_rendering_disabled()
    test_rendering_enabled()
    print("\n✅ All tests passed!")

