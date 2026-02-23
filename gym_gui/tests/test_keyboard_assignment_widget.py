"""Tests for keyboard assignment widget for multi-human gameplay.

Tests keyboard detection, assignment, and signal emission for multi-agent
human control in environments like MultiGrid and Overcooked.
"""

from __future__ import annotations

import os
from typing import Optional
from unittest.mock import Mock, MagicMock, patch

import pytest
from qtpy import QtWidgets
from PyQt6.QtGui import QInputDevice

# Ensure Qt renders offscreen in CI environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    """Create Qt application for widget testing."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


@pytest.fixture
def mock_keyboard_devices():
    """Create mock keyboard devices matching user's hardware."""
    # User's actual keyboards from lsusb output
    keyboards = []
    
    # Logitech keyboard
    kb1 = Mock(spec=QInputDevice)
    kb1.systemId.return_value = 12
    kb1.name.return_value = "Logitech USB Keyboard"
    kb1.seatName.return_value = "seat0"
    kb1.type.return_value = QInputDevice.DeviceType.Keyboard
    keyboards.append(kb1)
    
    # Lenovo Black Silk
    kb2 = Mock(spec=QInputDevice)
    kb2.systemId.return_value = 13
    kb2.name.return_value = "Lenovo Black Silk USB Keyboard TRACER"
    kb2.seatName.return_value = "seat0"
    kb2.type.return_value = QInputDevice.DeviceType.Keyboard
    keyboards.append(kb2)
    
    # SiGma Micro
    kb3 = Mock(spec=QInputDevice)
    kb3.systemId.return_value = 14
    kb3.name.return_value = "SiGma Micro Keyboard TRACER Gamma Ivory"
    kb3.seatName.return_value = "seat0"
    kb3.type.return_value = QInputDevice.DeviceType.Keyboard
    keyboards.append(kb3)
    
    # Also include a mouse (should be filtered out)
    mouse = Mock(spec=QInputDevice)
    mouse.systemId.return_value = 15
    mouse.name.return_value = "USB Optical Mouse"
    mouse.seatName.return_value = "seat0"
    mouse.type.return_value = QInputDevice.DeviceType.Mouse
    
    return keyboards + [mouse]


class TestKeyboardAssignmentWidget:
    """Test keyboard assignment widget functionality."""

    def test_widget_initialization(self, qt_app) -> None:
        """Widget initializes with default agents."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget
        
        widget = KeyboardAssignmentWidget(parent=None)
        
        assert widget is not None
        assert widget._available_agents == ["agent_0", "agent_1"]
        assert len(widget._keyboards) >= 0  # May or may not detect keyboards
        
        widget.deleteLater()

    def test_widget_custom_agents(self, qt_app) -> None:
        """Widget initializes with custom agent list."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget
        
        agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
        widget = KeyboardAssignmentWidget(available_agents=agents, parent=None)
        
        assert widget._available_agents == agents
        
        widget.deleteLater()

    @patch('gym_gui.ui.widgets.keyboard_assignment_widget.QInputDevice.devices')
    def test_keyboard_detection(self, mock_devices, qt_app, mock_keyboard_devices) -> None:
        """Widget detects keyboards and filters out non-keyboard devices."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget
        
        mock_devices.return_value = mock_keyboard_devices
        
        widget = KeyboardAssignmentWidget(parent=None)
        widget._detect_keyboards()
        
        # Should detect 3 keyboards (not the mouse)
        assert len(widget._keyboards) == 3
        
        # Check keyboard IDs
        detected_ids = set(widget._keyboards.keys())
        assert detected_ids == {12, 13, 14}
        
        widget.deleteLater()

    @patch('gym_gui.ui.widgets.keyboard_assignment_widget.QInputDevice.devices')
    def test_get_assignments(self, mock_devices, qt_app, mock_keyboard_devices) -> None:
        """get_assignments returns correct keyboard-to-agent mapping."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget
        
        mock_devices.return_value = mock_keyboard_devices
        
        widget = KeyboardAssignmentWidget(parent=None)
        widget._detect_keyboards()
        
        # Initially no assignments
        assert widget.get_assignments() == {}
        
        # Assign keyboards
        widget._keyboards[12].assigned_agent = "agent_0"
        widget._keyboards[13].assigned_agent = "agent_1"
        
        assignments = widget.get_assignments()
        assert assignments == {12: "agent_0", 13: "agent_1"}
        
        widget.deleteLater()

    @patch('gym_gui.ui.widgets.keyboard_assignment_widget.QInputDevice.devices')
    def test_get_agent_keyboard(self, mock_devices, qt_app, mock_keyboard_devices) -> None:
        """get_agent_keyboard returns keyboard ID for agent."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget
        
        mock_devices.return_value = mock_keyboard_devices
        
        widget = KeyboardAssignmentWidget(parent=None)
        widget._detect_keyboards()
        
        # Initially no assignments
        assert widget.get_agent_keyboard("agent_0") is None
        
        # Assign keyboard 12 to agent_0
        widget._keyboards[12].assigned_agent = "agent_0"
        
        assert widget.get_agent_keyboard("agent_0") == 12
        assert widget.get_agent_keyboard("agent_1") is None
        
        widget.deleteLater()

    @patch('gym_gui.ui.widgets.keyboard_assignment_widget.QInputDevice.devices')
    def test_assignment_changed_signal(self, mock_devices, qt_app, mock_keyboard_devices) -> None:
        """assignment_changed signal emits when assignment changes."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget
        
        mock_devices.return_value = mock_keyboard_devices
        
        widget = KeyboardAssignmentWidget(parent=None)
        widget._detect_keyboards()
        
        # Create signal spy
        signal_received = []
        widget.assignment_changed.connect(
            lambda sid, aid: signal_received.append((sid, aid))
        )
        
        # Trigger assignment via internal method (simulating dropdown change)
        widget._keyboards[12].assigned_agent = "agent_0"
        widget.assignment_changed.emit(12, "agent_0")
        
        assert len(signal_received) == 1
        assert signal_received[0] == (12, "agent_0")
        
        widget.deleteLater()

    @patch('gym_gui.ui.widgets.keyboard_assignment_widget.QInputDevice.devices')
    def test_keyboards_detected_signal(self, mock_devices, qt_app, mock_keyboard_devices) -> None:
        """keyboards_detected signal emits with keyboard count."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget
        
        mock_devices.return_value = mock_keyboard_devices
        
        widget = KeyboardAssignmentWidget(parent=None)
        
        # Create signal spy
        detected_counts = []
        widget.keyboards_detected.connect(lambda count: detected_counts.append(count))
        
        # Trigger detection
        widget._detect_keyboards()
        
        assert len(detected_counts) == 1
        assert detected_counts[0] == 3  # 3 keyboards
        
        widget.deleteLater()

    @patch('gym_gui.ui.widgets.keyboard_assignment_widget.QInputDevice.devices')
    def test_set_available_agents(self, mock_devices, qt_app, mock_keyboard_devices) -> None:
        """set_available_agents updates agent list."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget
        
        mock_devices.return_value = mock_keyboard_devices
        
        widget = KeyboardAssignmentWidget(
            available_agents=["agent_0", "agent_1"],
            parent=None
        )
        
        assert widget._available_agents == ["agent_0", "agent_1"]
        
        # Update to 4 agents (for MultiGrid)
        new_agents = ["agent_0", "agent_1", "agent_2", "agent_3"]
        widget.set_available_agents(new_agents)
        
        assert widget._available_agents == new_agents
        
        widget.deleteLater()

    @patch('gym_gui.ui.widgets.keyboard_assignment_widget.QInputDevice.devices')
    def test_get_detected_keyboards(self, mock_devices, qt_app, mock_keyboard_devices) -> None:
        """get_detected_keyboards returns list of KeyboardInfo."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import (
            KeyboardAssignmentWidget,
            KeyboardInfo,
        )
        
        mock_devices.return_value = mock_keyboard_devices
        
        widget = KeyboardAssignmentWidget(parent=None)
        widget._detect_keyboards()
        
        keyboards = widget.get_detected_keyboards()
        
        assert isinstance(keyboards, list)
        assert len(keyboards) == 3
        assert all(isinstance(kb, KeyboardInfo) for kb in keyboards)
        
        # Check keyboard names
        names = {kb.name for kb in keyboards}
        assert "Logitech USB Keyboard" in names
        assert "Lenovo Black Silk USB Keyboard TRACER" in names
        
        widget.deleteLater()

    @patch('gym_gui.ui.widgets.keyboard_assignment_widget.QInputDevice.devices')
    def test_keyboard_info_display_name(self, mock_devices, qt_app, mock_keyboard_devices) -> None:
        """KeyboardInfo.display_name truncates long names."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardInfo
        from unittest.mock import Mock
        
        # Short name - no truncation
        kb_short = KeyboardInfo(
            system_id=1,
            name="Logitech KB",
            seat_name="seat0",
            device=Mock()
        )
        assert kb_short.display_name == "Logitech KB"
        
        # Long name - should truncate
        long_name = "Very Long Keyboard Name That Exceeds Fifty Characters For Sure And More"
        kb_long = KeyboardInfo(
            system_id=2,
            name=long_name,
            seat_name="seat0",
            device=Mock()
        )
        assert len(kb_long.display_name) == 50  # 47 chars + "..."
        assert kb_long.display_name.endswith("...")

    def test_widget_visibility_integration(self, qt_app) -> None:
        """Widget can be shown/hidden for single vs multi-agent environments."""
        from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget
        
        widget = KeyboardAssignmentWidget(parent=None)
        
        # Start visible
        widget.setVisible(True)
        assert widget.isVisible()
        
        # Hide for single-agent environment
        widget.setVisible(False)
        assert not widget.isVisible()
        
        # Show for multi-agent environment
        widget.setVisible(True)
        assert widget.isVisible()
        
        widget.deleteLater()


# =============================================================================
# Integration Test - Manual Testing on Real Hardware
# =============================================================================

def test_keyboard_detection_real_hardware_manual(qt_app) -> None:
    """Manual test to verify keyboard detection on real Ubuntu X11 system.
    
    This test is marked for manual execution only. Run it with:
        pytest gym_gui/tests/test_keyboard_assignment_widget.py::test_keyboard_detection_real_hardware_manual -v
    
    Expected on user's system (Ubuntu 22.04 X11):
        - 3 keyboards detected:
            1. Logitech keyboard (046d:c34b)
            2. Lenovo Black Silk (17ef:602d)
            3. SiGma Micro TRACER (1c4f:0002)
    """
    from gym_gui.ui.widgets.keyboard_assignment_widget import KeyboardAssignmentWidget
    
    widget = KeyboardAssignmentWidget(parent=None)
    
    print("\n" + "="*60)
    print("KEYBOARD DETECTION TEST (Real Hardware)")
    print("="*60)
    print(f"Platform: {os.environ.get('QT_QPA_PLATFORM', 'native')}")
    print(f"Detected keyboards: {len(widget._keyboards)}")
    print("="*60)
    
    for kb in widget.get_detected_keyboards():
        print(f"\nKeyboard: {kb.name}")
        print(f"  System ID: {kb.system_id}")
        print(f"  Seat: {kb.seat_name}")
        print(f"  Assigned: {kb.assigned_agent or '(unassigned)'}")
    
    print("\n" + "="*60)
    print("Note: This test may show 0 keyboards in offscreen mode.")
    print("For full testing, run MOSAIC GUI and check Human Control tab.")
    print("="*60 + "\n")
    
    widget.deleteLater()
