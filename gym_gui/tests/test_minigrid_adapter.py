"""Regression tests for the MiniGrid adapter integration."""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("minigrid")

from gym_gui.core.adapters.base import AdapterContext
from gym_gui.core.adapters.minigrid import MiniGridAdapter
from gym_gui.config.game_configs import MiniGridConfig
from gym_gui.core.enums import ControlMode, GameId
from gym_gui.logging_config.log_constants import (
    LOG_ENV_MINIGRID_BOOT,
    LOG_ENV_MINIGRID_STEP,
)


def _make_adapter(**overrides: object) -> MiniGridAdapter:
    config = MiniGridConfig(env_id=GameId.MINIGRID_EMPTY_5x5.value, **overrides)
    context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
    adapter = MiniGridAdapter(context, config=config)
    adapter.load()
    return adapter


def test_minigrid_adapter_boot_logs_and_seed(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="gym_gui.core.adapters.base")
    adapter = _make_adapter(partial_observation=True, image_observation=True)
    try:
        boot_codes = [record.log_code for record in caplog.records if hasattr(record, "log_code")]
        assert LOG_ENV_MINIGRID_BOOT.code in boot_codes

        first = adapter.reset(seed=123)
        second = adapter.reset(seed=123)
        np.testing.assert_array_equal(first.observation, second.observation)
        assert first.state.environment["env_id"] == GameId.MINIGRID_EMPTY_5x5.value

        render_payload = first.render_payload
        assert isinstance(render_payload, dict)
        assert render_payload.get("mode") == "rgb_array"
        assert isinstance(render_payload.get("rgb"), np.ndarray)
    finally:
        adapter.close()


def test_minigrid_adapter_step_metadata(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="gym_gui.core.adapters.base")
    adapter = _make_adapter()
    try:
        _ = adapter.reset(seed=99)
        action = int(adapter.action_space.sample())
        step = adapter.step(action)

        raw = step.info.get("_minigrid_raw_observation")
        assert isinstance(raw, dict)
        assert "image" in raw
        assert step.observation.dtype == np.uint8
        assert step.observation.ndim == 1
        assert step.observation.shape[0] == raw["image"].size + 1
        assert step.state.environment["env_id"] == GameId.MINIGRID_EMPTY_5x5.value
        if step.state.metrics:
            assert "direction" in step.state.metrics

        step_codes = [record.log_code for record in caplog.records if hasattr(record, "log_code")]
        assert LOG_ENV_MINIGRID_STEP.code in step_codes

        render_payload = step.render_payload
        assert isinstance(render_payload, dict)
        assert render_payload.get("mode") == "rgb_array"
        assert isinstance(render_payload.get("rgb"), np.ndarray)
    finally:
        adapter.close()
