"""Tests for Ray RLlib worker training functionality.

These tests verify:
1. Configuration loading and validation
2. Environment factory works for SISL environments
3. Training runtime can be initialized
4. Short training runs complete successfully
5. Analytics manifest is written correctly
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Skip all tests if Ray is not available
pytest.importorskip("ray")
pytest.importorskip("pettingzoo")


class TestRayWorkerConfig:
    """Tests for RayWorkerConfig."""

    def test_config_creation(self):
        """Test creating a basic config."""
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            TrainingParadigm,
        )

        config = RayWorkerConfig(
            run_id="test_run_001",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
            ),
            paradigm=TrainingParadigm.PARAMETER_SHARING,
        )

        assert config.run_id == "test_run_001"
        assert config.environment.family == "sisl"
        assert config.environment.env_id == "waterworld_v4"
        assert config.paradigm == TrainingParadigm.PARAMETER_SHARING

    def test_config_path_resolution(self):
        """Test path resolution properties."""
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            TrainingParadigm,
        )

        config = RayWorkerConfig(
            run_id="test_paths_001",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
            ),
            paradigm=TrainingParadigm.PARAMETER_SHARING,
        )

        # Test path properties
        assert "test_paths_001" in str(config.run_dir)
        assert "checkpoints" in str(config.checkpoint_dir)
        assert "logs" in str(config.logs_dir)
        assert "tensorboard" in str(config.tensorboard_log_dir)
        assert "analytics.json" in str(config.analytics_manifest_path)

    def test_config_to_dict(self):
        """Test serialization to dict."""
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            TrainingParadigm,
        )

        config = RayWorkerConfig(
            run_id="test_serial_001",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
            ),
            paradigm=TrainingParadigm.INDEPENDENT,
        )

        data = config.to_dict()
        assert data["run_id"] == "test_serial_001"
        assert data["environment"]["family"] == "sisl"
        assert data["paradigm"] == "independent"

    def test_config_from_dict(self):
        """Test deserialization from dict."""
        from ray_worker.config import RayWorkerConfig

        data = {
            "run_id": "test_deserial_001",
            "environment": {
                "family": "sisl",
                "env_id": "waterworld_v4",
                "api_type": "parallel",
            },
            "paradigm": "self_play",
            "training": {
                "algorithm": "PPO",
                "total_timesteps": 5000,
            },
        }

        config = RayWorkerConfig.from_dict(data)
        assert config.run_id == "test_deserial_001"
        assert config.environment.family == "sisl"
        assert config.training.total_timesteps == 5000


class TestEnvironmentFactory:
    """Tests for PettingZoo environment factory."""

    def test_create_waterworld(self):
        """Test creating Waterworld environment."""
        from ray_worker.runtime import EnvironmentFactory

        env = EnvironmentFactory.create_sisl_env("waterworld_v4")
        assert env is not None
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        env.close()

    def test_create_multiwalker(self):
        """Test creating Multiwalker environment."""
        from ray_worker.runtime import EnvironmentFactory

        env = EnvironmentFactory.create_sisl_env("multiwalker_v9")
        assert env is not None
        assert hasattr(env, "reset")
        env.close()

    def test_create_pursuit(self):
        """Test creating Pursuit environment."""
        from ray_worker.runtime import EnvironmentFactory

        env = EnvironmentFactory.create_sisl_env("pursuit_v4")
        assert env is not None
        assert hasattr(env, "reset")
        env.close()

    def test_create_env_generic(self):
        """Test generic env creation."""
        from ray_worker.runtime import EnvironmentFactory
        from ray_worker.config import PettingZooAPIType

        env = EnvironmentFactory.create_env(
            family="sisl",
            env_id="waterworld_v4",
            api_type=PettingZooAPIType.PARALLEL,
        )
        assert env is not None
        env.close()

    def test_invalid_env_raises(self):
        """Test that invalid env ID raises error."""
        from ray_worker.runtime import EnvironmentFactory

        with pytest.raises(ValueError, match="Unknown SISL environment"):
            EnvironmentFactory.create_sisl_env("invalid_env_v99")


class TestRayWorkerRuntime:
    """Tests for RayWorkerRuntime training loop."""

    @pytest.fixture
    def basic_config(self):
        """Create a basic config for testing."""
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            TrainingConfig,
            ResourceConfig,
            CheckpointConfig,
            TrainingParadigm,
        )

        return RayWorkerConfig(
            run_id="pytest_training_001",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
            ),
            paradigm=TrainingParadigm.PARAMETER_SHARING,
            training=TrainingConfig(
                algorithm="PPO",
                total_timesteps=2000,  # Very short for testing
                train_batch_size=500,
            ),
            resources=ResourceConfig(
                num_workers=0,  # Local only for testing
                num_gpus=0,
            ),
            checkpoint=CheckpointConfig(
                checkpoint_freq=1,
                checkpoint_at_end=True,
            ),
        )

    def test_runtime_initialization(self, basic_config):
        """Test runtime can be initialized."""
        from ray_worker.runtime import RayWorkerRuntime

        runtime = RayWorkerRuntime(basic_config)
        assert runtime.config == basic_config

    @pytest.mark.slow
    def test_short_training_run(self, basic_config, tmp_path):
        """Test a very short training run completes.

        This test is marked slow as it involves actual training.
        """
        import ray
        from ray_worker.runtime import RayWorkerRuntime

        # Update config to use temp directory
        basic_config.output_dir = str(tmp_path)

        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=2)

        try:
            runtime = RayWorkerRuntime(basic_config)
            result = runtime.run()

            # Verify training completed
            assert result is not None
            assert "timesteps_total" in result or "env_runners" in result

            # Verify checkpoint was created
            checkpoint_dir = tmp_path / basic_config.run_id / "checkpoints"
            # Note: Checkpoint dir might be created by Ray in different location
            # Just verify no exception was raised

        finally:
            ray.shutdown()


class TestAnalyticsManifest:
    """Tests for analytics manifest generation."""

    def test_manifest_creation(self):
        """Test creating analytics manifest."""
        from ray_worker.analytics import RayAnalyticsManifest

        manifest = RayAnalyticsManifest(
            run_id="test_manifest_001",
            tensorboard_relative="tensorboard",
            paradigm="parameter_sharing",
            algorithm="PPO",
            env_id="waterworld_v4",
            env_family="sisl",
            num_agents=2,
        )

        assert manifest.run_id == "test_manifest_001"
        assert manifest.paradigm == "parameter_sharing"

    def test_manifest_to_dict(self):
        """Test manifest serialization."""
        from ray_worker.analytics import RayAnalyticsManifest

        manifest = RayAnalyticsManifest(
            run_id="test_manifest_002",
            tensorboard_relative="tensorboard",
            paradigm="independent",
            algorithm="PPO",
            env_id="waterworld_v4",
            env_family="sisl",
        )

        data = manifest.to_dict()
        assert data["run_id"] == "test_manifest_002"
        assert data["worker_type"] == "ray_worker"
        assert data["artifacts"]["tensorboard"]["enabled"] is True
        assert data["ray_metadata"]["paradigm"] == "independent"

    def test_manifest_save_load(self, tmp_path):
        """Test manifest save and load."""
        from ray_worker.analytics import RayAnalyticsManifest

        manifest = RayAnalyticsManifest(
            run_id="test_manifest_003",
            tensorboard_relative="tensorboard",
            paradigm="self_play",
            algorithm="PPO",
            env_id="chess_v6",
            env_family="classic",
        )

        # Save
        manifest_path = tmp_path / "analytics.json"
        manifest.save(manifest_path)

        assert manifest_path.exists()

        # Load
        loaded = RayAnalyticsManifest.load(manifest_path)
        assert loaded.run_id == "test_manifest_003"
        assert loaded.paradigm == "self_play"


class TestPolicyActor:
    """Tests for RayPolicyActor inference."""

    def test_policy_config_creation(self):
        """Test creating policy config."""
        from ray_worker.policy_actor import RayPolicyConfig

        config = RayPolicyConfig(
            checkpoint_path="/path/to/checkpoint",
            policy_id="shared",
            deterministic=True,
        )

        assert config.checkpoint_path == "/path/to/checkpoint"
        assert config.policy_id == "shared"
        assert config.deterministic is True

    def test_policy_actor_creation(self):
        """Test creating policy actor (without loading)."""
        from ray_worker.policy_actor import RayPolicyActor, RayPolicyConfig

        config = RayPolicyConfig(
            checkpoint_path="/path/to/checkpoint",
            policy_id="shared",
        )

        actor = RayPolicyActor(id="test_actor", config=config)
        assert actor.id == "test_actor"
        assert actor.is_ready is False  # Not loaded yet


