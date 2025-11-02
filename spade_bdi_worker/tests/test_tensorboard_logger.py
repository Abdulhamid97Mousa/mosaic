"""Tests for the TensorBoard logger integration used by the SPADE-BDI worker."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from spade_bdi_worker.core.config import RunConfig
from spade_bdi_worker.core import tensorboard_logger as tb_module
from spade_bdi_worker.core.tensorboard_logger import TensorboardLogger


class _DummyWriter:
    """In-memory SummaryWriter replacement used for tests."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = Path(log_dir)
        self.scalars: List[Dict[str, Any]] = []
        self.texts: List[Dict[str, Any]] = []
        self.flush_count = 0
        self.closed = False

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append({"tag": tag, "value": value, "step": step})

    def add_text(self, tag: str, text: str, step: int | None = None) -> None:
        self.texts.append({"tag": tag, "text": text, "step": step})

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture(name="dummy_writer_factory")
def dummy_writer_factory_fixture() -> Dict[str, Any]:
    writers: List[_DummyWriter] = []

    def factory(*, log_dir: str) -> _DummyWriter:
        writer = _DummyWriter(log_dir)
        writers.append(writer)
        return writer

    return {"factory": factory, "writers": writers}


def test_tensorboard_logger_initializes_and_records_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dummy_writer_factory: Dict[str, Any]
) -> None:
    # Patch paths so the logger writes under the temp directory.
    var_runs_dir = tmp_path / "runs"

    def ensure_var_dirs() -> None:
        var_runs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(tb_module, "VAR_TENSORBOARD_DIR", var_runs_dir)
    monkeypatch.setattr(tb_module, "ensure_var_directories", ensure_var_dirs)
    monkeypatch.setattr(tb_module, "_resolve_summary_writer", lambda: dummy_writer_factory["factory"])

    config = RunConfig(run_id="test-run", game_id="dummy", extra={})

    logger = TensorboardLogger.from_run_config(config)
    assert logger is not None

    expected_log_dir = (var_runs_dir / config.run_id / "tensorboard").resolve()
    assert logger.log_dir == expected_log_dir
    assert expected_log_dir.exists()

    tensorboard_meta = config.extra.get("tensorboard")
    assert tensorboard_meta is not None and tensorboard_meta["enabled"] is True
    assert tensorboard_meta["log_dir"] == str(expected_log_dir)
    assert tensorboard_meta["relative_path"] == f"var/trainer/runs/{config.run_id}/tensorboard"

    logger.on_run_started({"game": config.game_id})
    logger.log_episode(episode_number=1, reward=42.0, steps=10, epsilon=0.2, success=True)
    logger.log_run_summary(
        [
            SimpleNamespace(total_reward=42.0, steps=10, success=True),
            SimpleNamespace(total_reward=21.0, steps=8, success=False),
        ]
    )
    logger.on_run_completed(status="finished")
    logger.close()

    writers: List[_DummyWriter] = dummy_writer_factory["writers"]
    assert len(writers) == 1
    writer = writers[0]

    assert any(item["tag"] == "episode/return" for item in writer.scalars)
    assert any(item["tag"] == "run/avg_return" for item in writer.scalars)
    assert any(entry["tag"] == "run/status" and entry["text"] == "finished" for entry in writer.texts)
    assert writer.flush_count > 0
    assert writer.closed is True


def test_tensorboard_logger_respects_disabled_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected() -> None:
        raise AssertionError("SummaryWriter should not be resolved when TensorBoard is disabled")

    monkeypatch.setattr(tb_module, "_resolve_summary_writer", _unexpected)

    for extra in ({"tensorboard": {"disabled": True}}, {"disable_tensorboard": True}):
        config = RunConfig(run_id="disabled", game_id="dummy", extra=dict(extra))
        assert TensorboardLogger.from_run_config(config) is None


def test_tensorboard_logger_requires_summary_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_module, "_resolve_summary_writer", lambda: None)

    config = RunConfig(run_id="missing-writer", game_id="dummy", extra={})

    with pytest.raises(RuntimeError):
        TensorboardLogger.from_run_config(config)
