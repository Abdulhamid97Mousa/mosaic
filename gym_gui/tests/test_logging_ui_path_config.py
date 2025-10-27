"""Tests for UI-side logging of dual-path configuration diagnostics."""

from __future__ import annotations

import logging
from datetime import datetime as real_datetime

import pytest

qtpy = pytest.importorskip("qtpy", reason="QtPy is required for UI logging tests")
from qtpy import QtWidgets

import gym_gui.ui.widgets.spade_bdi_train_form as train_form_mod
from gym_gui.logging_config.log_constants import (
    LOG_UI_TRAIN_FORM_TELEMETRY_PATH,
    LOG_UI_TRAIN_FORM_UI_PATH,
)
from gym_gui.ui.widgets.spade_bdi_train_form import SpadeBdiTrainForm


@pytest.fixture(scope="function")
def qt_app():
    """Provide a Qt application instance for the duration of the test."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def _install_fixed_datetime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force deterministic timestamps so run_name assertions remain stable."""

    class FixedDatetime(real_datetime):
        @classmethod
        def utcnow(cls) -> real_datetime:
            return cls(2025, 10, 26, 12, 0, 0)

    monkeypatch.setattr(train_form_mod, "datetime", FixedDatetime)


def _collect_log_codes(caplog: pytest.LogCaptureFixture) -> set[str]:
    """Extract log codes produced by log_constant helper."""
    return {
        record.__dict__["log_code"]
        for record in caplog.records
        if "log_code" in record.__dict__
    }


def test_train_form_emits_dual_path_logs(qt_app, caplog, monkeypatch):
    """SpadeBdiTrainForm should log both UI-only and durable telemetry settings."""
    _install_fixed_datetime(monkeypatch)

    form = SpadeBdiTrainForm()
    try:
        # Adjust knobs to non-default values so payload is easy to assert.
        form._training_telemetry_throttle_slider.setValue(3)
        form._ui_rendering_throttle_slider.setValue(4)
        form._render_delay_slider.setValue(150)
        form._ui_training_speed_slider.setValue(20)  # -> 200 ms
        form._telemetry_buffer_spin.setValue(4096)
        form._episode_buffer_spin.setValue(512)

        caplog.set_level(logging.INFO, logger="gym_gui.ui.spade_bdi_train_form")

        config = form._build_base_config()

        codes = _collect_log_codes(caplog)
        assert LOG_UI_TRAIN_FORM_UI_PATH.code in codes
        assert LOG_UI_TRAIN_FORM_TELEMETRY_PATH.code in codes

        ui_meta = config["metadata"]["ui"]["path_config"]
        assert ui_meta["ui_only"]["render_delay_ms"] == 150
        assert ui_meta["ui_only"]["step_delay_ms"] == 200
        assert ui_meta["telemetry_durable"]["telemetry_buffer_size"] == 4096
        assert ui_meta["telemetry_durable"]["episode_buffer_size"] == 512

        worker_path = config["metadata"]["worker"]["config"]["path_config"]
        assert worker_path["telemetry_durable"]["telemetry_buffer_size"] == 4096
        assert worker_path["ui_only"]["render_delay_ms"] == 150
    finally:
        form.deleteLater()
