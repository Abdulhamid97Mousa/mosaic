"""Tests for RunConfig live rendering metadata propagation."""

from __future__ import annotations

from spade_bdi_worker.core.config import RunConfig


def test_run_config_respects_live_rendering_flag():
    payload = {
        "run_id": "01K8TESTRUNID0000000000000",
        "game_id": "FrozenLake-v1",
        "seed": 1,
        "max_episodes": 5,
        "max_steps_per_episode": 20,
        "policy_strategy": "train_and_save",
        "headless": True,
        "path_config": {
            "ui_only": {
                "live_rendering_enabled": False,
                "render_delay_ms": 0,
                "ui_rendering_throttle": 1,
                "step_delay_ms": 0,
            },
            "telemetry_durable": {
                "training_telemetry_throttle": 1,
                "telemetry_buffer_size": 256,
                "episode_buffer_size": 64,
            },
        },
    }

    config = RunConfig.from_dict(payload)

    ui_config = config.extra["path_config"]["ui_only"]
    assert ui_config["live_rendering_enabled"] is False
    assert config.max_episodes == 5
    assert config.worker_id is None
