"""Unit tests for worker disable telemetry flag resolution."""

from __future__ import annotations

import pytest

from spade_bdi_worker import worker


@pytest.mark.parametrize(
    "env_value",
    ["1", "true", "TRUE", "Yes", "on", "ON"],
)
def test_env_truthy_values_disable_telemetry(env_value: str) -> None:
    assert worker._resolve_disable_telemetry(False, env_value, None) is True


@pytest.mark.parametrize(
    "env_value",
    ["0", "false", "", "no", "off", "Off"],
)
def test_env_non_truthy_values_do_not_disable(env_value: str) -> None:
    assert worker._resolve_disable_telemetry(False, env_value, None) is False


def test_env_none_defaults_to_config_value() -> None:
    assert worker._resolve_disable_telemetry(False, None, True) is True
    assert worker._resolve_disable_telemetry(False, None, "yes") is True
    assert worker._resolve_disable_telemetry(False, None, "0") is False


def test_cli_flag_always_wins() -> None:
    assert worker._resolve_disable_telemetry(True, None, False) is True
    assert worker._resolve_disable_telemetry(True, "0", False) is True


def test_coerce_bool_handles_mixed_inputs() -> None:
    assert worker._coerce_bool(True) is True
    assert worker._coerce_bool(False) is False
    assert worker._coerce_bool(1) is True
    assert worker._coerce_bool(0) is False
    assert worker._coerce_bool("On") is True
    assert worker._coerce_bool("off") is False
    assert worker._coerce_bool(None) is False
