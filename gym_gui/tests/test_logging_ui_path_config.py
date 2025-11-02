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

        worker_config = config["metadata"]["worker"]["config"]
        assert worker_config["schema_id"] == "telemetry.step.default"
        assert worker_config["schema_version"] == 1
        assert isinstance(worker_config["schema_definition"], dict)
        ui_meta_root = config["metadata"]["ui"]
        assert ui_meta_root["schema_id"] == "telemetry.step.default"
        assert ui_meta_root["schema_version"] == 1
    finally:
        form.deleteLater()


def test_fast_training_mode_clears_ui_and_telemetry_paths(qt_app, monkeypatch):
    """Fast training mode must disable UI rendering and mark telemetry as disabled."""
    _install_fixed_datetime(monkeypatch)

    form = SpadeBdiTrainForm()
    try:
        # Pre-configure non-zero values to ensure fast mode overrides them.
        form._render_delay_slider.setValue(150)
        form._ui_training_speed_slider.setValue(50)
        form._telemetry_buffer_spin.setValue(2048)
        form._episode_buffer_spin.setValue(256)

        form._fast_training_checkbox.setChecked(True)
        config = form._build_base_config()

        environment = config["environment"]
        assert environment["DISABLE_TELEMETRY"] == "1"
        assert environment["UI_LIVE_RENDERING_ENABLED"] == "0"
        assert environment["UI_RENDER_DELAY_MS"] == "0"

        worker_config = config["metadata"]["worker"]["config"]
        assert worker_config["step_delay"] == 0.0
        assert worker_config["extra"]["disable_telemetry"] is True
        assert worker_config["schema_id"] == "telemetry.step.default"
        assert worker_config["schema_version"] == 1
        assert isinstance(worker_config["schema_definition"], dict)

        ui_path = worker_config["path_config"]["ui_only"]
        assert ui_path["live_rendering_enabled"] is False
        assert ui_path["render_delay_ms"] == 0
        assert ui_path["step_delay_ms"] == 0
        assert ui_path["headless_only"] is True

        telemetry_path = worker_config["path_config"]["telemetry_durable"]
        assert telemetry_path["disabled"] is True
        assert telemetry_path["telemetry_buffer_size"] == 0
        assert telemetry_path["episode_buffer_size"] == 0
        assert telemetry_path["hub_buffer_size"] == 0

        ui_metadata = config["metadata"]["ui"]
        assert ui_metadata["live_rendering_enabled"] is False
        assert ui_metadata["telemetry_buffer_size"] == 0
        assert ui_metadata["episode_buffer_size"] == 0
        assert ui_metadata["schema_id"] == "telemetry.step.default"
        assert ui_metadata["schema_version"] == 1
    finally:
        form.deleteLater()
