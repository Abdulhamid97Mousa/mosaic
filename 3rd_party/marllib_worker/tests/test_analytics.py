"""Tests for MARLlib worker analytics manifest generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from marllib_worker.config import MARLlibWorkerConfig


def _skip_if_no_analytics():
    try:
        from gym_gui.core.worker import WorkerAnalyticsManifest  # noqa: F401

        return False
    except ImportError:
        return True


_SKIP = _skip_if_no_analytics()


def _make_config(**overrides) -> MARLlibWorkerConfig:
    defaults = dict(
        run_id="test-analytics",
        algo="mappo",
        environment_name="mpe",
        map_name="simple_spread",
    )
    defaults.update(overrides)
    return MARLlibWorkerConfig(**defaults)


@pytest.mark.skipif(_SKIP, reason="gym_gui not available")
class TestBuildManifest:
    """Verify manifest construction from Ray Tune output."""

    def test_with_ray_tune_output(self, tmp_path: Path):
        """Manifest discovers TensorBoard events and checkpoints."""
        from marllib_worker.analytics import build_manifest

        # Simulate Ray Tune trial directory
        trial = tmp_path / "mappo_mlp_simple_spread" / "trial_0001"
        trial.mkdir(parents=True)
        (trial / "progress.csv").write_text("step,reward\n1,0.5\n")
        (trial / "events.out.tfevents.12345.host").write_text("")
        (trial / "checkpoint_100").mkdir()
        (trial / "checkpoint_200").mkdir()

        cfg = _make_config()
        manifest = build_manifest(tmp_path, trial, cfg)

        assert manifest.run_id == "test-analytics"
        assert manifest.worker_type == "marllib"

        # TensorBoard
        assert manifest.artifacts.tensorboard is not None
        assert manifest.artifacts.tensorboard.enabled is True

        # Checkpoints
        assert manifest.artifacts.checkpoints is not None
        assert len(manifest.artifacts.checkpoints.files) == 2
        assert manifest.artifacts.checkpoints.format == "ray_rllib"
        assert "checkpoint_200" in manifest.artifacts.checkpoints.final_checkpoint

    def test_no_ray_output(self, tmp_path: Path):
        """Manifest handles missing Ray Tune output gracefully."""
        from marllib_worker.analytics import build_manifest

        cfg = _make_config()
        manifest = build_manifest(tmp_path, None, cfg)

        assert manifest.artifacts.tensorboard is None
        assert manifest.artifacts.checkpoints is None

    def test_metadata_fields(self, tmp_path: Path):
        """Manifest metadata includes algo/env info."""
        from marllib_worker.analytics import build_manifest

        cfg = _make_config(algo="qmix", environment_name="smac", map_name="3m")
        manifest = build_manifest(tmp_path, None, cfg)

        assert manifest.metadata["algo"] == "qmix"
        assert manifest.metadata["algo_type"] == "VD"
        assert manifest.metadata["environment"] == "smac"
        assert manifest.metadata["map_name"] == "3m"

    def test_relative_paths(self, tmp_path: Path):
        """TensorBoard and checkpoint paths are relative to run_dir."""
        from marllib_worker.analytics import build_manifest

        trial = tmp_path / "exp" / "trial"
        trial.mkdir(parents=True)
        (trial / "progress.csv").write_text("")
        (trial / "events.out.tfevents.1").write_text("")
        (trial / "checkpoint_1").mkdir()

        cfg = _make_config()
        manifest = build_manifest(tmp_path, trial, cfg)

        tb_dir = manifest.artifacts.tensorboard.log_dir
        assert not Path(tb_dir).is_absolute(), f"Expected relative path, got {tb_dir}"
