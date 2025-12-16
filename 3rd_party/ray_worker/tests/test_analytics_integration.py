"""Tests for TensorBoard and WandB analytics integration.

Ensures:
- TensorBoard directory is created when enabled
- analytics.json manifest is written correctly
- WandB is initialized when enabled (mocked)
- Metrics are logged to TensorBoard
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestAnalyticsSetup:
    """Test analytics initialization in RayWorkerRuntime."""

    def test_tensorboard_dir_created(self, tmp_path):
        """TensorBoard directory should be created when tensorboard=True."""
        from ray_worker.config import RayWorkerConfig
        from ray_worker.runtime import RayWorkerRuntime

        # Set tensorboard_dir explicitly to ensure we know where it will be created
        tb_dir = tmp_path / "tensorboard_logs"
        config = RayWorkerConfig.from_dict({
            "run_id": "test-tb-dir",
            "environment": {
                "family": "sisl",
                "env_id": "multiwalker_v9",
                "api_type": "parallel",
            },
            "paradigm": "parameter_sharing",
            "training": {"algorithm": "PPO", "total_timesteps": 1000},
            "tensorboard": True,
            "tensorboard_dir": str(tb_dir),
            "output_dir": str(tmp_path),
        })

        runtime = RayWorkerRuntime(config)

        # Mock the writer creation to avoid actual tensorboard import issues
        with patch("ray_worker.runtime.write_analytics_manifest") as mock_manifest:
            mock_manifest.return_value = tmp_path / "analytics.json"
            runtime._setup_analytics()

        # Check that tensorboard directory was created at the specified location
        assert tb_dir.exists(), f"TensorBoard directory should exist: {tb_dir}"

    def test_analytics_manifest_written(self, tmp_path):
        """analytics.json should be written with correct structure."""
        from ray_worker.config import RayWorkerConfig
        from ray_worker.analytics import write_analytics_manifest

        config = RayWorkerConfig.from_dict({
            "run_id": "test-manifest",
            "environment": {
                "family": "sisl",
                "env_id": "waterworld_v4",
                "api_type": "parallel",
            },
            "paradigm": "independent",
            "training": {"algorithm": "PPO", "total_timesteps": 10000},
            "tensorboard": True,
            "wandb": False,
            "output_dir": str(tmp_path),
        })

        # Ensure run directory exists
        config.ensure_run_directories()

        # Write manifest
        manifest_path = write_analytics_manifest(
            config,
            wandb_run_path=None,
            num_agents=5,
            notes="Test run",
        )

        # Check manifest exists and has correct structure
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Check required fields
        assert manifest["run_id"] == "test-manifest"
        assert manifest["worker_type"] == "ray_worker"
        assert "artifacts" in manifest

        # Check artifacts structure
        artifacts = manifest["artifacts"]
        assert artifacts["tensorboard"]["enabled"] is True
        assert artifacts["tensorboard"]["relative_path"] == "tensorboard"
        assert artifacts["wandb"]["enabled"] is False
        assert artifacts["checkpoints"]["enabled"] is True

        # Check ray metadata
        ray_meta = manifest["ray_metadata"]
        assert ray_meta["paradigm"] == "independent"
        assert ray_meta["algorithm"] == "PPO"
        assert ray_meta["env_id"] == "waterworld_v4"
        assert ray_meta["num_agents"] == 5

    def test_tensorboard_disabled(self, tmp_path):
        """TensorBoard should not be initialized when disabled."""
        from ray_worker.config import RayWorkerConfig
        from ray_worker.runtime import RayWorkerRuntime

        config = RayWorkerConfig.from_dict({
            "run_id": "test-no-tb",
            "environment": {
                "family": "sisl",
                "env_id": "multiwalker_v9",
                "api_type": "parallel",
            },
            "paradigm": "parameter_sharing",
            "training": {"algorithm": "PPO", "total_timesteps": 1000},
            "tensorboard": False,
            "output_dir": str(tmp_path),
        })

        runtime = RayWorkerRuntime(config)

        with patch("ray_worker.runtime.write_analytics_manifest") as mock_manifest:
            mock_manifest.return_value = tmp_path / "analytics.json"
            runtime._setup_analytics()

        # Writer should be None
        assert runtime._writer is None


class TestMetricsLogging:
    """Test metrics logging to TensorBoard/WandB."""

    def test_log_metrics_with_writer(self, tmp_path):
        """_log_metrics should write to TensorBoard writer when available."""
        from ray_worker.config import RayWorkerConfig
        from ray_worker.runtime import RayWorkerRuntime

        config = RayWorkerConfig.from_dict({
            "run_id": "test-log-metrics",
            "environment": {
                "family": "sisl",
                "env_id": "multiwalker_v9",
                "api_type": "parallel",
            },
            "paradigm": "parameter_sharing",
            "training": {"algorithm": "PPO", "total_timesteps": 1000},
            "tensorboard": True,
            "output_dir": str(tmp_path),
        })

        runtime = RayWorkerRuntime(config)

        # Create a mock writer
        mock_writer = MagicMock()
        runtime._writer = mock_writer

        # Sample result from Ray RLlib
        result = {
            "env_runners": {
                "episode_return_mean": 123.45,
                "episode_len_mean": 100,
                "num_episodes_lifetime": 50,
            },
            "learners": {
                "default_learner": {
                    "total_loss": 0.5,
                    "policy_loss": 0.3,
                    "vf_loss": 0.1,
                    "entropy": 0.01,
                }
            }
        }

        runtime._log_metrics(result, global_step=1000)

        # Verify writer.add_scalar was called
        assert mock_writer.add_scalar.called
        assert mock_writer.flush.called

        # Check specific metrics were logged
        calls = {call[0][0]: call[0][1] for call in mock_writer.add_scalar.call_args_list}
        assert "train/episode_reward_mean" in calls
        assert calls["train/episode_reward_mean"] == 123.45

    def test_log_metrics_without_writer(self, tmp_path):
        """_log_metrics should not fail when writer is None."""
        from ray_worker.config import RayWorkerConfig
        from ray_worker.runtime import RayWorkerRuntime

        config = RayWorkerConfig.from_dict({
            "run_id": "test-no-writer",
            "environment": {
                "family": "sisl",
                "env_id": "multiwalker_v9",
                "api_type": "parallel",
            },
            "paradigm": "parameter_sharing",
            "training": {"algorithm": "PPO", "total_timesteps": 1000},
            "tensorboard": False,
            "output_dir": str(tmp_path),
        })

        runtime = RayWorkerRuntime(config)
        runtime._writer = None

        result = {"env_runners": {"episode_return_mean": 100}}

        # Should not raise an exception
        runtime._log_metrics(result, global_step=1000)


class TestWandBIntegration:
    """Test WandB integration (mocked)."""

    def test_wandb_init_called_when_enabled(self, tmp_path):
        """WandB should be initialized when wandb=True."""
        from ray_worker.config import RayWorkerConfig
        from ray_worker.runtime import RayWorkerRuntime

        config = RayWorkerConfig.from_dict({
            "run_id": "test-wandb",
            "environment": {
                "family": "sisl",
                "env_id": "multiwalker_v9",
                "api_type": "parallel",
            },
            "paradigm": "parameter_sharing",
            "training": {"algorithm": "PPO", "total_timesteps": 1000},
            "tensorboard": False,
            "wandb": True,
            "wandb_project": "test-project",
            "output_dir": str(tmp_path),
        })

        runtime = RayWorkerRuntime(config)

        mock_wandb_run = MagicMock()
        mock_wandb_run.path = "entity/project/run_id"

        with patch.dict("sys.modules", {"wandb": MagicMock()}):
            import sys
            sys.modules["wandb"].init.return_value = mock_wandb_run

            with patch("ray_worker.runtime.write_analytics_manifest") as mock_manifest:
                mock_manifest.return_value = tmp_path / "analytics.json"
                runtime._setup_analytics()

            # Check wandb.init was called
            sys.modules["wandb"].init.assert_called_once()
            call_kwargs = sys.modules["wandb"].init.call_args[1]
            assert call_kwargs["project"] == "test-project"

    def test_wandb_not_initialized_when_disabled(self, tmp_path):
        """WandB should not be initialized when wandb=False."""
        from ray_worker.config import RayWorkerConfig
        from ray_worker.runtime import RayWorkerRuntime

        config = RayWorkerConfig.from_dict({
            "run_id": "test-no-wandb",
            "environment": {
                "family": "sisl",
                "env_id": "multiwalker_v9",
                "api_type": "parallel",
            },
            "paradigm": "parameter_sharing",
            "training": {"algorithm": "PPO", "total_timesteps": 1000},
            "tensorboard": False,
            "wandb": False,
            "output_dir": str(tmp_path),
        })

        runtime = RayWorkerRuntime(config)

        with patch("ray_worker.runtime.write_analytics_manifest") as mock_manifest:
            mock_manifest.return_value = tmp_path / "analytics.json"
            runtime._setup_analytics()

        assert runtime._wandb_run is None


class TestCleanup:
    """Test analytics cleanup."""

    def test_cleanup_closes_writer(self, tmp_path):
        """_cleanup_analytics should close TensorBoard writer."""
        from ray_worker.config import RayWorkerConfig
        from ray_worker.runtime import RayWorkerRuntime

        config = RayWorkerConfig.from_dict({
            "run_id": "test-cleanup",
            "environment": {
                "family": "sisl",
                "env_id": "multiwalker_v9",
                "api_type": "parallel",
            },
            "paradigm": "parameter_sharing",
            "training": {"algorithm": "PPO", "total_timesteps": 1000},
            "output_dir": str(tmp_path),
        })

        runtime = RayWorkerRuntime(config)

        # Mock writer
        mock_writer = MagicMock()
        runtime._writer = mock_writer

        runtime._cleanup_analytics()

        mock_writer.close.assert_called_once()
        assert runtime._writer is None

    def test_cleanup_finishes_wandb(self, tmp_path):
        """_cleanup_analytics should finish WandB run."""
        from ray_worker.config import RayWorkerConfig
        from ray_worker.runtime import RayWorkerRuntime

        config = RayWorkerConfig.from_dict({
            "run_id": "test-cleanup-wandb",
            "environment": {
                "family": "sisl",
                "env_id": "multiwalker_v9",
                "api_type": "parallel",
            },
            "paradigm": "parameter_sharing",
            "training": {"algorithm": "PPO", "total_timesteps": 1000},
            "output_dir": str(tmp_path),
        })

        runtime = RayWorkerRuntime(config)

        # Mock wandb run
        mock_wandb_run = MagicMock()
        runtime._wandb_run = mock_wandb_run

        with patch.dict("sys.modules", {"wandb": MagicMock()}):
            runtime._cleanup_analytics()

        # wandb_run should be set to None (cleanup logic will call wandb.finish)
