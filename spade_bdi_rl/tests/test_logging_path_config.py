"""Worker-level logging tests validating dual-path diagnostics."""

from __future__ import annotations

import logging

import pytest

from gym_gui.logging_config.log_constants import (
    LOG_WORKER_CONFIG_DURABLE_PATH,
    LOG_WORKER_CONFIG_EVENT,
    LOG_WORKER_CONFIG_UI_PATH,
    LOG_WORKER_CONFIG_WARNING,
)
from spade_bdi_rl.core.config import RunConfig


def _collect_log_records(caplog: pytest.LogCaptureFixture) -> dict[str, list[logging.LogRecord]]:
    """Group captured records by log code."""
    grouped: dict[str, list[logging.LogRecord]] = {}
    for record in caplog.records:
        code = record.__dict__.get("log_code")
        if not code:
            continue
        grouped.setdefault(code, []).append(record)
    return grouped


def test_runconfig_logs_ui_vs_durable_settings(caplog):
    """RunConfig should announce both UI-only and durable-path configuration."""
    payload = {
        "run_id": "run-test",
        "game_id": "ExampleGame",
        "seed": 5,
        "max_episodes": 3,
        "max_steps_per_episode": 20,
        "policy_strategy": "train_and_save",
        "agent_id": "agent-alpha",
        "step_delay": 0.5,  # 500 ms applied delay
        "telemetry_buffer_size": 2048,
        "episode_buffer_size": 128,
        "path_config": {
            "ui_only": {
                "live_rendering_enabled": False,
                "ui_rendering_throttle": 4,
                "render_delay_ms": 120,
                "step_delay_ms": 750,  # mismatched against applied 500 ms
            },
            "telemetry_durable": {
                "training_telemetry_throttle": 3,
                "telemetry_buffer_size": 4096,
                "episode_buffer_size": 256,
            },
        },
    }

    caplog.set_level(logging.INFO, logger="spade_bdi_rl.core.config")
    config = RunConfig.from_dict(payload)

    grouped = _collect_log_records(caplog)

    assert LOG_WORKER_CONFIG_EVENT.code in grouped
    assert LOG_WORKER_CONFIG_UI_PATH.code in grouped
    assert LOG_WORKER_CONFIG_DURABLE_PATH.code in grouped
    # Mismatch between requested 750 ms vs applied 500 ms should raise warning.
    assert LOG_WORKER_CONFIG_WARNING.code in grouped

    ui_record = grouped[LOG_WORKER_CONFIG_UI_PATH.code][0]
    assert ui_record.live_rendering_enabled is False
    assert ui_record.step_delay_mismatch is True
    assert ui_record.requested_step_delay_ms == 750
    assert ui_record.applied_step_delay_ms == 500

    durable_record = grouped[LOG_WORKER_CONFIG_DURABLE_PATH.code][0]
    assert durable_record.telemetry_buffer_requested == 4096
    assert durable_record.telemetry_buffer_applied == 2048
    assert durable_record.episode_buffer_requested == 256
    assert durable_record.episode_buffer_applied == 128

    # Verify extra metadata persisted for downstream consumers.
    assert config.extra["path_config"]["ui_only"]["step_delay_ms"] == 750
    assert config.step_delay == pytest.approx(0.5)
