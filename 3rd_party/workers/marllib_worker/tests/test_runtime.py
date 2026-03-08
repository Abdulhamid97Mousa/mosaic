"""Tests for MARLlib worker runtime (dry-run and helpers only).

Full integration tests require MARLlib + Ray 1.8 to be installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marllib_worker.config import MARLlibWorkerConfig
from marllib_worker.runtime import MARLlibWorkerRuntime


def _make_config(**overrides) -> MARLlibWorkerConfig:
    defaults = dict(
        run_id="test-runtime",
        algo="mappo",
        environment_name="mpe",
        map_name="simple_spread",
        seed=42,
    )
    defaults.update(overrides)
    return MARLlibWorkerConfig(**defaults)


class TestDryRun:
    """Verify dry-run mode works without MARLlib installed."""

    def test_dry_run_returns_summary(self):
        cfg = _make_config()
        runtime = MARLlibWorkerRuntime(cfg, dry_run=True)
        result = runtime.run()

        assert result["status"] == "dry-run"
        assert result["algo"] == "mappo"
        assert result["environment"] == "mpe"
        assert result["map_name"] == "simple_spread"

    def test_dry_run_includes_algo_type(self):
        cfg = _make_config(algo="qmix")
        runtime = MARLlibWorkerRuntime(cfg, dry_run=True)
        result = runtime.run()

        assert result["algo_type"] == "VD"

    def test_dry_run_cc_algo(self):
        cfg = _make_config(algo="coma")
        runtime = MARLlibWorkerRuntime(cfg, dry_run=True)
        result = runtime.run()

        assert result["algo_type"] == "CC"

    def test_dry_run_il_algo(self):
        cfg = _make_config(algo="ippo")
        runtime = MARLlibWorkerRuntime(cfg, dry_run=True)
        result = runtime.run()

        assert result["algo_type"] == "IL"

    def test_dry_run_includes_share_policy(self):
        cfg = _make_config(share_policy="group")
        runtime = MARLlibWorkerRuntime(cfg, dry_run=True)
        result = runtime.run()

        assert result["share_policy"] == "group"

    def test_dry_run_includes_full_config(self):
        cfg = _make_config()
        runtime = MARLlibWorkerRuntime(cfg, dry_run=True)
        result = runtime.run()

        assert "config" in result
        assert result["config"]["seed"] == 42


class TestFindRayTuneOutput:
    """Verify Ray Tune output directory discovery."""

    def test_finds_nested_trial_dir(self, tmp_path: Path):
        trial = tmp_path / "mappo_mlp_simple_spread" / "trial_0001"
        trial.mkdir(parents=True)
        (trial / "progress.csv").write_text("step,reward\n1,0.5\n")

        cfg = _make_config()
        runtime = MARLlibWorkerRuntime(cfg)
        found = runtime._find_ray_tune_output(tmp_path)

        assert found is not None
        assert found.name == "trial_0001"

    def test_finds_direct_trial_dir(self, tmp_path: Path):
        trial = tmp_path / "experiment_dir"
        trial.mkdir()
        (trial / "progress.csv").write_text("step,reward\n1,0.5\n")

        cfg = _make_config()
        runtime = MARLlibWorkerRuntime(cfg)
        found = runtime._find_ray_tune_output(tmp_path)

        assert found is not None
        assert found.name == "experiment_dir"

    def test_returns_none_when_empty(self, tmp_path: Path):
        cfg = _make_config()
        runtime = MARLlibWorkerRuntime(cfg)
        found = runtime._find_ray_tune_output(tmp_path)

        assert found is None

    def test_ignores_hidden_and_logs(self, tmp_path: Path):
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "logs").mkdir()

        cfg = _make_config()
        runtime = MARLlibWorkerRuntime(cfg)
        found = runtime._find_ray_tune_output(tmp_path)

        assert found is None
