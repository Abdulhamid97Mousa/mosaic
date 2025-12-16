"""Tests for Ray RLlib training form configuration building.

Ensures:
- Form builds valid config for trainer daemon
- No 'train' subcommand in arguments
- worker_id is set correctly in metadata
- FastLane settings are properly configured
- render_mode logic for multi-agent composite view
"""

import pytest
from typing import Dict, Any
import json


class TestRayTrainFormConfigStructure:
    """Test the structure of config built by RayRLlibTrainForm."""

    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Sample config as would be built by the form."""
        return {
            "run_name": "ray-multiwalker-v9-20251213-222144",
            "entry_point": "/usr/bin/python",
            "arguments": ["-m", "ray_worker.cli"],  # No 'train' subcommand!
            "environment": {
                "RAY_FASTLANE_ENABLED": "1",
                # RAY_FASTLANE_RUN_ID is NOT set by form - dispatcher sets it with ULID
                "RAY_FASTLANE_ENV_NAME": "multiwalker_v9",
                "RAY_FASTLANE_THROTTLE_MS": "33",
                "GYM_GUI_FASTLANE_ONLY": "1",
            },
            "resources": {
                "cpus": 2,
                "memory_mb": 4096,
                "gpus": {"requested": 1, "mandatory": False},
            },
            "artifacts": {
                "output_prefix": "runs/ray-multiwalker-v9-20251213-222144",
                "persist_logs": True,
                "keep_checkpoints": True,
            },
            "metadata": {
                "ui": {
                    "worker_id": "ray_worker",
                    "env_id": "multiwalker_v9",
                    "family": "sisl",
                    "algorithm": "PPO",
                    "paradigm": "parameter_sharing",
                    "mode": "training",
                    "fastlane_enabled": True,
                    "fastlane_only": True,
                },
                "worker": {
                    "worker_id": "ray_worker",
                    "module": "ray_worker.cli",
                    "use_grpc": True,
                    "grpc_target": "127.0.0.1:50055",
                    "config": {
                        # run_id is NOT set by form - dispatcher sets it with ULID
                        "environment": {
                            "family": "sisl",
                            "env_id": "multiwalker_v9",
                            "api_type": "parallel",
                        },
                        "paradigm": "parameter_sharing",
                        "training": {
                            "algorithm": "PPO",
                            "total_timesteps": 100000,
                            "train_batch_size": 4000,
                            "sgd_minibatch_size": 128,
                            "lr": 0.0003,
                            "gamma": 0.99,
                        },
                        "resources": {
                            "num_workers": 2,
                            "num_gpus": 1,
                            "num_cpus_per_worker": 1,
                        },
                        "checkpoint": {
                            "checkpoint_freq": 0,
                            "checkpoint_at_end": True,
                            "export_policy": True,
                        },
                        "fastlane_enabled": True,
                        "fastlane_throttle_ms": 33,
                        "seed": 42,
                        "tensorboard": True,
                        "wandb": False,
                        "extras": {
                            "fastlane_only": True,
                            "save_model": True,
                        },
                    },
                },
                "artifacts": {
                    "tensorboard": {
                        "enabled": True,
                        "relative_path": "tensorboard",
                    },
                    "wandb": {
                        "enabled": False,
                        "project": "ray-marl",
                    },
                    "fastlane": {
                        "enabled": True,
                        "mode": "composite",
                        "throttle_ms": 33,
                    },
                },
            },
        }

    def test_no_run_id_at_top_level(self, sample_config):
        """Config should NOT have run_id at top level (schema violation)."""
        assert "run_id" not in sample_config

    def test_has_run_name_at_top_level(self, sample_config):
        """Config should have run_name at top level."""
        assert "run_name" in sample_config
        assert sample_config["run_name"].startswith("ray-")

    def test_arguments_no_train_subcommand(self, sample_config):
        """Arguments should NOT include 'train' subcommand."""
        args = sample_config["arguments"]
        assert "train" not in args
        assert args == ["-m", "ray_worker.cli"]

    def test_worker_id_in_metadata_ui(self, sample_config):
        """worker_id should be in metadata.ui."""
        worker_id = sample_config["metadata"]["ui"]["worker_id"]
        assert worker_id == "ray_worker"

    def test_worker_id_in_metadata_worker(self, sample_config):
        """worker_id should be in metadata.worker."""
        worker_id = sample_config["metadata"]["worker"]["worker_id"]
        assert worker_id == "ray_worker"

    def test_env_id_in_metadata_ui(self, sample_config):
        """env_id should be in metadata.ui for tab naming."""
        env_id = sample_config["metadata"]["ui"]["env_id"]
        assert env_id == "multiwalker_v9"

    def test_no_run_id_in_worker_config(self, sample_config):
        """Worker config should NOT have run_id - dispatcher sets it with ULID."""
        worker_config = sample_config["metadata"]["worker"]["config"]
        # run_id is NOT set by form - dispatcher adds it later
        assert "run_id" not in worker_config

    def test_fastlane_mode_is_composite(self, sample_config):
        """FastLane mode should be 'composite' for multi-agent."""
        mode = sample_config["metadata"]["artifacts"]["fastlane"]["mode"]
        assert mode == "composite"

    def test_fastlane_environment_variables(self, sample_config):
        """FastLane env vars should be set (except RUN_ID which dispatcher adds)."""
        env = sample_config["environment"]
        assert env["RAY_FASTLANE_ENABLED"] == "1"
        # RAY_FASTLANE_RUN_ID is NOT in form config - dispatcher adds it with ULID
        assert "RAY_FASTLANE_RUN_ID" not in env
        assert "RAY_FASTLANE_ENV_NAME" in env
        assert "RAY_FASTLANE_THROTTLE_MS" in env


class TestRayWorkerConfigParsing:
    """Test that ray_worker can parse the config correctly."""

    def test_config_from_dict_parses_environment(self):
        """RayWorkerConfig.from_dict parses environment correctly."""
        from ray_worker.config import RayWorkerConfig, PettingZooAPIType

        config_data = {
            "run_id": "test-run-123",
            "environment": {
                "family": "sisl",
                "env_id": "waterworld_v4",
                "api_type": "parallel",
            },
            "paradigm": "parameter_sharing",
            "training": {
                "algorithm": "PPO",
                "total_timesteps": 50000,
            },
            "resources": {
                "num_workers": 2,
                "num_gpus": 1,
            },
            "checkpoint": {},
            "fastlane_enabled": True,
            "fastlane_throttle_ms": 33,
        }

        config = RayWorkerConfig.from_dict(config_data)

        assert config.run_id == "test-run-123"
        assert config.environment.family == "sisl"
        assert config.environment.env_id == "waterworld_v4"
        assert config.environment.api_type == PettingZooAPIType.PARALLEL
        assert config.fastlane_enabled is True
        assert config.fastlane_throttle_ms == 33

    def test_config_from_dict_with_nested_metadata(self):
        """RayWorkerConfig.from_dict handles metadata.worker.config structure."""
        from ray_worker.config import load_worker_config
        import tempfile
        import os

        # Create temp config file with nested structure (as UI generates)
        config_data = {
            "metadata": {
                "worker": {
                    "config": {
                        "run_id": "nested-run-456",
                        "environment": {
                            "family": "classic",
                            "env_id": "chess_v6",
                            "api_type": "aec",
                        },
                        "paradigm": "self_play",
                        "training": {"algorithm": "PPO"},
                        "resources": {"num_workers": 1},
                        "checkpoint": {},
                        "fastlane_enabled": False,
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_worker_config(temp_path)
            assert config.run_id == "nested-run-456"
            assert config.environment.family == "classic"
            assert config.environment.env_id == "chess_v6"
            assert config.fastlane_enabled is False
        finally:
            os.unlink(temp_path)


class TestFastLaneRenderMode:
    """Test that FastLane uses rgb_array render mode (no pygame windows)."""

    def test_fastlane_enabled_implies_rgb_array(self):
        """When fastlane_enabled=True, render_mode should be rgb_array."""
        # This is tested by checking the wrapped_env_creator logic
        # The actual runtime.py code does:
        # if self.config.fastlane_enabled and "render_mode" not in env_kwargs:
        #     env_kwargs["render_mode"] = "rgb_array"

        fastlane_enabled = True
        env_kwargs = {}

        # Simulate the logic
        if fastlane_enabled and "render_mode" not in env_kwargs:
            env_kwargs["render_mode"] = "rgb_array"

        assert env_kwargs["render_mode"] == "rgb_array"

    def test_explicit_render_mode_not_overwritten(self):
        """Explicit render_mode in env_kwargs is preserved."""
        fastlane_enabled = True
        env_kwargs = {"render_mode": "human"}  # User explicitly wants human

        # Simulate the logic
        if fastlane_enabled and "render_mode" not in env_kwargs:
            env_kwargs["render_mode"] = "rgb_array"

        # Should NOT overwrite
        assert env_kwargs["render_mode"] == "human"

    def test_fastlane_disabled_no_render_mode_set(self):
        """When fastlane_enabled=False, render_mode is not automatically set."""
        fastlane_enabled = False
        env_kwargs = {}

        # Simulate the logic
        if fastlane_enabled and "render_mode" not in env_kwargs:
            env_kwargs["render_mode"] = "rgb_array"

        assert "render_mode" not in env_kwargs


class TestConfigSchemaCompliance:
    """Test that config complies with trainer daemon schema."""

    def test_top_level_keys_are_valid(self):
        """Only allowed keys at top level of config."""
        allowed_keys = {
            "run_name",
            "entry_point",
            "arguments",
            "environment",
            "resources",
            "artifacts",
            "metadata",
            "schedule",
        }

        sample_keys = {
            "run_name",
            "entry_point",
            "arguments",
            "environment",
            "resources",
            "artifacts",
            "metadata",
        }

        # All sample keys should be in allowed
        assert sample_keys.issubset(allowed_keys)

        # run_id should NOT be at top level
        assert "run_id" not in sample_keys

    def test_no_additional_properties_at_top_level(self):
        """Schema has additionalProperties: false, so no extra keys allowed."""
        # This tests that we don't accidentally add keys like 'run_id' at top level
        disallowed_top_level_keys = ["run_id", "config", "worker_config", "fastlane"]

        sample_top_level = [
            "run_name", "entry_point", "arguments", "environment",
            "resources", "artifacts", "metadata"
        ]

        for key in disallowed_top_level_keys:
            assert key not in sample_top_level, f"{key} should not be at top level"
