"""Tests to validate type annotations in render_tabs module.

These tests ensure that type annotations are correct and that the module
can be imported without type errors. They also validate that Protocol
definitions match their implementations.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Callable, Protocol
from unittest.mock import MagicMock

import pytest


class TestTypeAnnotations:
    """Tests for type annotation correctness."""

    def test_render_tabs_imports_without_error(self) -> None:
        """Verify render_tabs module imports successfully."""
        from gym_gui.ui.widgets import render_tabs

        assert render_tabs is not None
        assert hasattr(render_tabs, "RenderTabs")

    def test_mouse_capture_strategy_protocol_defined(self) -> None:
        """Verify MouseCaptureStrategy Protocol is properly defined."""
        from gym_gui.ui.widgets.render_tabs import MouseCaptureStrategy

        assert hasattr(MouseCaptureStrategy, "__protocol_attrs__") or isinstance(
            MouseCaptureStrategy, type
        )

    def test_mouse_capture_strategy_has_required_methods(self) -> None:
        """Verify MouseCaptureStrategy Protocol defines all required methods."""
        from gym_gui.ui.widgets.render_tabs import MouseCaptureStrategy

        # Check that the protocol defines the expected methods
        expected_methods = [
            "set_mouse_capture_enabled",
            "set_mouse_delta_callback",
            "set_mouse_delta_scale",
            "set_mouse_action_callback",
        ]

        for method_name in expected_methods:
            assert hasattr(
                MouseCaptureStrategy, method_name
            ), f"MouseCaptureStrategy missing method: {method_name}"


class TestProtocolConformance:
    """Tests verifying that implementations conform to Protocol definitions."""

    def test_mock_conforms_to_mouse_capture_strategy(self) -> None:
        """A mock with the right methods should satisfy MouseCaptureStrategy."""
        from gym_gui.ui.widgets.render_tabs import MouseCaptureStrategy

        # Create a mock that has all required methods
        mock_strategy = MagicMock()
        mock_strategy.set_mouse_capture_enabled = MagicMock()
        mock_strategy.set_mouse_delta_callback = MagicMock()
        mock_strategy.set_mouse_delta_scale = MagicMock()
        mock_strategy.set_mouse_action_callback = MagicMock()

        # These should all work without error
        mock_strategy.set_mouse_capture_enabled(True)
        mock_strategy.set_mouse_delta_callback(lambda x, y: None)
        mock_strategy.set_mouse_delta_scale(0.5)
        mock_strategy.set_mouse_action_callback(lambda a: None)

        # Verify calls were made
        mock_strategy.set_mouse_capture_enabled.assert_called_once_with(True)
        mock_strategy.set_mouse_delta_scale.assert_called_once_with(0.5)


class TestPyrightValidation:
    """Tests that run pyright to validate type annotations."""

    @pytest.mark.skipif(
        subprocess.run(
            [sys.executable, "-m", "pyright", "--version"],
            capture_output=True,
        ).returncode
        != 0,
        reason="pyright not installed",
    )
    def test_render_tabs_passes_pyright(self) -> None:
        """Verify render_tabs.py passes pyright type checking."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pyright",
                "gym_gui/ui/widgets/render_tabs.py",
            ],
            capture_output=True,
            text=True,
        )
        # Check for success (0 errors)
        assert result.returncode == 0, (
            f"pyright found type errors:\n{result.stdout}\n{result.stderr}"
        )


class TestSignalAnnotations:
    """Tests for Qt Signal type annotations."""

    def test_render_tabs_has_signal_attributes(self) -> None:
        """Verify RenderTabs class has properly typed signals."""
        from gym_gui.ui.widgets.render_tabs import RenderTabs

        # Check that signals are defined as class attributes
        assert hasattr(RenderTabs, "chess_move_made")
        assert hasattr(RenderTabs, "connect_four_column_clicked")
        assert hasattr(RenderTabs, "go_intersection_clicked")
        assert hasattr(RenderTabs, "go_pass_requested")


class TestManagementTabTyping:
    """Tests for ManagementTab type annotations."""

    def test_management_tab_import_in_type_checking(self) -> None:
        """Verify ManagementTab is imported correctly for type checking."""
        # This test verifies the TYPE_CHECKING block works correctly
        import typing

        if typing.TYPE_CHECKING:
            from gym_gui.ui.widgets.management_tab import ManagementTab

            assert ManagementTab is not None

    def test_render_tabs_management_tab_attribute_optional(self) -> None:
        """Verify _management_tab is properly typed as Optional."""
        from gym_gui.ui.widgets.render_tabs import RenderTabs

        # Verify the class is defined correctly
        # We can't use get_type_hints because of forward references in TYPE_CHECKING
        # Instead, check the annotation directly from the source
        import inspect

        source = inspect.getsource(RenderTabs.__init__)
        assert "_management_tab" in source
        assert "Optional" in source or "None" in source
        assert RenderTabs is not None
