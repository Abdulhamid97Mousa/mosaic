"""Test the Disable Live Rendering toggle functionality."""

import os
from PyQt6 import QtWidgets

import pytest

from gym_gui.ui.widgets.spade_bdi_train_form import SpadeBdiTrainForm
from gym_gui.core.enums import GameId

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    """Create a QApplication for Qt testing."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_disable_live_rendering_checkbox_unchecked_by_default(qt_app):
    """Test that the disable live rendering checkbox is unchecked by default."""
    form = SpadeBdiTrainForm(default_game=GameId.FROZEN_LAKE_V2)
    assert form._disable_live_render_checkbox.isChecked() is False
    form.deleteLater()


def test_live_rendering_enabled_when_checkbox_unchecked(qt_app):
    """Test that live_rendering_enabled is True when checkbox is unchecked."""
    form = SpadeBdiTrainForm(default_game=GameId.FROZEN_LAKE_V2)
    # Checkbox is unchecked by default
    assert form._disable_live_render_checkbox.isChecked() is False

    # live_rendering_enabled should be True (NOT False = NOT False = True)
    live_rendering_enabled = not form._disable_live_render_checkbox.isChecked()
    assert live_rendering_enabled is True
    form.deleteLater()


def test_live_rendering_disabled_when_checkbox_checked(qt_app):
    """Test that live_rendering_enabled is False when checkbox is checked."""
    form = SpadeBdiTrainForm(default_game=GameId.FROZEN_LAKE_V2)
    # Check the checkbox
    form._disable_live_render_checkbox.setChecked(True)

    # live_rendering_enabled should be False (NOT True = False)
    live_rendering_enabled = not form._disable_live_render_checkbox.isChecked()
    assert live_rendering_enabled is False
    form.deleteLater()


def test_config_includes_live_rendering_flag_when_enabled(qt_app):
    """Test that the config includes live_rendering_enabled=True when checkbox is unchecked."""
    form = SpadeBdiTrainForm(default_game=GameId.FROZEN_LAKE_V2)
    form._episodes_spin.setValue(10)
    form._max_steps_spin.setValue(25)
    form._seed_spin.setValue(1)

    # Leave checkbox unchecked (default)
    form._disable_live_render_checkbox.setChecked(False)

    # Build config
    config = form._build_base_config()

    # Verify the flag is in metadata["ui"]
    assert "metadata" in config
    assert "ui" in config["metadata"]
    assert "live_rendering_enabled" in config["metadata"]["ui"]
    assert config["metadata"]["ui"]["live_rendering_enabled"] is True
    form.deleteLater()


def test_config_includes_live_rendering_disabled_when_checkbox_checked(qt_app):
    """Test that the config includes live_rendering_enabled=False when checkbox is checked."""
    form = SpadeBdiTrainForm(default_game=GameId.FROZEN_LAKE_V2)
    form._episodes_spin.setValue(10)
    form._max_steps_spin.setValue(25)
    form._seed_spin.setValue(1)

    # Check the checkbox to disable rendering
    form._disable_live_render_checkbox.setChecked(True)

    # Build config
    config = form._build_base_config()

    # Verify the flag is in metadata["ui"] and set to False
    assert "metadata" in config
    assert "ui" in config["metadata"]
    assert "live_rendering_enabled" in config["metadata"]["ui"]
    assert config["metadata"]["ui"]["live_rendering_enabled"] is False
    form.deleteLater()


def test_toggle_updates_render_control_states(qt_app):
    """Test that toggling the checkbox updates render control states."""
    form = SpadeBdiTrainForm(default_game=GameId.FROZEN_LAKE_V2)

    # Initially unchecked - controls should be enabled
    assert form._ui_rendering_throttle_slider.isEnabled() is True
    assert form._render_delay_slider.isEnabled() is True

    # Check the checkbox to disable rendering
    form._disable_live_render_checkbox.setChecked(True)

    # Controls should now be disabled
    assert form._ui_rendering_throttle_slider.isEnabled() is False
    assert form._render_delay_slider.isEnabled() is False

    # Uncheck the checkbox
    form._disable_live_render_checkbox.setChecked(False)

    # Controls should be enabled again
    assert form._ui_rendering_throttle_slider.isEnabled() is True
    assert form._render_delay_slider.isEnabled() is True
    form.deleteLater()


def test_environment_has_live_rendering_flag(qt_app):
    """Test that the environment variables include the live rendering flag when disabled."""
    form = SpadeBdiTrainForm(default_game=GameId.FROZEN_LAKE_V2)
    form._episodes_spin.setValue(10)
    form._max_steps_spin.setValue(25)
    form._seed_spin.setValue(1)

    form._disable_live_render_checkbox.setChecked(True)

    config = form._build_base_config()

    assert "environment" in config
    # UI_LIVE_RENDERING_ENABLED should be "0" when disabled
    assert config["environment"]["UI_LIVE_RENDERING_ENABLED"] == "0"
    form.deleteLater()


def test_environment_has_live_rendering_enabled(qt_app):
    """Test that the environment variables have live rendering enabled by default."""
    form = SpadeBdiTrainForm(default_game=GameId.FROZEN_LAKE_V2)
    form._episodes_spin.setValue(10)
    form._max_steps_spin.setValue(25)
    form._seed_spin.setValue(1)

    form._disable_live_render_checkbox.setChecked(False)

    config = form._build_base_config()

    assert "environment" in config
    # UI_LIVE_RENDERING_ENABLED should be "1" when enabled
    assert config["environment"]["UI_LIVE_RENDERING_ENABLED"] == "1"
    form.deleteLater()


