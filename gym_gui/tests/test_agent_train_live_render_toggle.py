"""Tests covering live rendering toggle behaviour in the training form."""

from __future__ import annotations

import os

import pytest

from PyQt6 import QtWidgets

from gym_gui.ui.widgets.spade_bdi_train_form import SpadeBdiTrainForm
from gym_gui.ui.widgets.live_telemetry_tab import LiveTelemetryTab


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_live_render_toggle_disables_ui_path(qt_app):
    form = SpadeBdiTrainForm()
    form._episodes_spin.setValue(10)
    form._max_steps_spin.setValue(25)
    form._seed_spin.setValue(1)

    # Disable live rendering via toggle
    form._disable_live_render_checkbox.setChecked(True)
    form._on_accept()

    config = form.get_config()
    assert config is not None

    ui_only = config["metadata"]["ui"]["path_config"]["ui_only"]
    assert ui_only["live_rendering_enabled"] is False
    assert config["environment"]["UI_LIVE_RENDERING_ENABLED"] == "0"
    assert config["environment"]["UI_RENDER_DELAY_MS"] == "0"

    form.deleteLater()


def test_live_render_toggle_prevents_regulator(qt_app):
    tab = LiveTelemetryTab(
        run_id="run-123",
        agent_id="agent-A",
        live_render_enabled=False,
    )

    # When live rendering is disabled, regulator should not be created
    assert tab._render_regulator is None

    # Cleanup to avoid lingering Qt objects
    tab.deleteLater()
