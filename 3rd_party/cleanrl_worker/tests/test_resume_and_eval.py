"""Tests for resume training and policy evaluation functionality."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cleanrl_worker.config import WorkerConfig
from cleanrl_worker.runtime import CleanRLWorkerRuntime


def _make_config(
    *,
    extras: dict[str, Any] | None = None,
    algo: str = "ppo",
    env_id: str = "CartPole-v1",
) -> WorkerConfig:
    """Create a WorkerConfig for testing."""
    return WorkerConfig(
        run_id="test-run-id",
        algo=algo,
        env_id=env_id,
        total_timesteps=1000,
        seed=42,
        extras=dict(extras or {}),
    )


class TestResumeTraining:
    """Tests for the resume training checkpoint auto-loading feature."""

    def test_sitecustomize_patches_module_to(self) -> None:
        """Verify sitecustomize patches nn.Module.to for checkpoint loading."""
        import torch.nn as nn

        # The patched function should have been installed by sitecustomize
        # Check that to() is callable and the module loads sitecustomize
        import cleanrl_worker.sitecustomize  # noqa: F401

        # Verify nn.Module.to still works
        model = nn.Linear(10, 5)
        model = model.to("cpu")
        assert model is not None

    def test_checkpoint_loading_with_matching_keys(self, tmp_path: Path) -> None:
        """Test that checkpoint is loaded when state_dict keys match."""
        import torch
        import torch.nn as nn

        # Create a simple model and save its state
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        checkpoint_path = tmp_path / "test_model.cleanrl_model"

        # Initialize with specific weights we can verify
        with torch.no_grad():
            model[0].weight.fill_(1.0)
            model[0].bias.fill_(0.5)

        torch.save(model.state_dict(), checkpoint_path)

        # Create a new model with different weights
        new_model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        with torch.no_grad():
            new_model[0].weight.fill_(0.0)
            new_model[0].bias.fill_(0.0)

        # Set environment variable and reload sitecustomize to reset flag
        import cleanrl_worker.sitecustomize as sc
        sc._RESUME_CHECKPOINT_LOADED = False  # Reset the flag

        with patch.dict(os.environ, {"CLEANRL_RESUME_PATH": str(checkpoint_path)}):
            # Move to device should trigger checkpoint loading
            new_model = new_model.to("cpu")

        # Verify weights were loaded
        assert torch.allclose(new_model[0].weight, torch.ones_like(new_model[0].weight))
        assert torch.allclose(new_model[0].bias, torch.full_like(new_model[0].bias, 0.5))

    def test_checkpoint_not_loaded_when_keys_mismatch(self, tmp_path: Path) -> None:
        """Test that checkpoint is NOT loaded when state_dict keys don't match."""
        import torch
        import torch.nn as nn

        # Create and save a model with different architecture
        original_model = nn.Linear(10, 5)
        checkpoint_path = tmp_path / "mismatched_model.cleanrl_model"
        torch.save(original_model.state_dict(), checkpoint_path)

        # Create a model with different architecture
        different_model = nn.Sequential(
            nn.Linear(4, 8),
            nn.Linear(8, 2),
        )
        original_weights = different_model[0].weight.clone()

        import cleanrl_worker.sitecustomize as sc
        sc._RESUME_CHECKPOINT_LOADED = False

        with patch.dict(os.environ, {"CLEANRL_RESUME_PATH": str(checkpoint_path)}):
            different_model = different_model.to("cpu")

        # Weights should NOT have changed (keys don't match)
        assert torch.allclose(different_model[0].weight, original_weights)

    def test_checkpoint_loaded_only_once(self, tmp_path: Path) -> None:
        """Test that checkpoint is loaded only once even with multiple .to() calls."""
        import torch
        import torch.nn as nn

        model = nn.Linear(4, 2)
        checkpoint_path = tmp_path / "single_load_model.cleanrl_model"

        with torch.no_grad():
            model.weight.fill_(1.0)
        torch.save(model.state_dict(), checkpoint_path)

        import cleanrl_worker.sitecustomize as sc
        sc._RESUME_CHECKPOINT_LOADED = False

        load_count = 0
        original_load = torch.load

        def counting_load(*args, **kwargs):
            nonlocal load_count
            load_count += 1
            return original_load(*args, **kwargs)

        new_model = nn.Linear(4, 2)

        with patch.dict(os.environ, {"CLEANRL_RESUME_PATH": str(checkpoint_path)}):
            with patch("torch.load", counting_load):
                new_model = new_model.to("cpu")
                new_model = new_model.to("cpu")  # Second call
                new_model = new_model.to("cpu")  # Third call

        # Should only have loaded once
        assert load_count == 1

    def test_no_loading_without_env_var(self, tmp_path: Path) -> None:
        """Test that no checkpoint loading occurs without CLEANRL_RESUME_PATH."""
        import torch
        import torch.nn as nn

        model = nn.Linear(4, 2)
        original_weights = model.weight.clone()

        import cleanrl_worker.sitecustomize as sc
        sc._RESUME_CHECKPOINT_LOADED = False

        # Ensure env var is not set
        env = os.environ.copy()
        env.pop("CLEANRL_RESUME_PATH", None)

        with patch.dict(os.environ, env, clear=True):
            model = model.to("cpu")

        # Weights should remain unchanged
        assert torch.allclose(model.weight, original_weights)


class TestPolicyEvaluation:
    """Tests for policy evaluation mode."""

    def test_policy_eval_mode_requires_policy_path(self) -> None:
        """Test that policy_eval mode requires policy_path extra."""
        config = _make_config(extras={"mode": "policy_eval"})
        runtime = CleanRLWorkerRuntime(
            config,
            use_grpc=False,
            grpc_target="127.0.0.1:50055",
            dry_run=False,
        )

        # Should raise because policy_path is missing
        with pytest.raises(ValueError, match="policy_path"):
            runtime.run()

    def test_policy_eval_mode_with_missing_file(self, tmp_path: Path, monkeypatch) -> None:
        """Test that policy_eval raises FileNotFoundError for missing checkpoint."""
        config = _make_config(
            extras={
                "mode": "policy_eval",
                "policy_path": str(tmp_path / "nonexistent.cleanrl_model"),
            }
        )
        runtime = CleanRLWorkerRuntime(
            config,
            use_grpc=False,
            grpc_target="127.0.0.1:50055",
            dry_run=False,
        )

        monkeypatch.setattr(
            CleanRLWorkerRuntime, "_register_with_trainer", lambda self: None
        )
        monkeypatch.setattr(
            "cleanrl_worker.runtime.VAR_TRAINER_DIR",
            tmp_path,
            raising=False,
        )
        monkeypatch.setattr(
            "cleanrl_worker.runtime.ensure_var_directories",
            lambda: None,
        )

        with pytest.raises(FileNotFoundError):
            runtime.run()

    def test_policy_eval_config_parsing(self) -> None:
        """Test that policy_eval extras are properly parsed."""
        config = _make_config(
            extras={
                "mode": "policy_eval",
                "policy_path": "/path/to/model.cleanrl_model",
                "eval_episodes": 10,
                "eval_capture_video": True,
                "eval_gamma": 0.95,
            }
        )

        assert config.extras["mode"] == "policy_eval"
        assert config.extras["eval_episodes"] == 10
        assert config.extras["eval_capture_video"] is True
        assert config.extras["eval_gamma"] == 0.95

    def test_dry_run_returns_summary_for_policy_eval(self) -> None:
        """Test that dry run works for policy_eval mode."""
        config = _make_config(
            extras={
                "mode": "policy_eval",
                "policy_path": "/path/to/model.cleanrl_model",
            }
        )
        runtime = CleanRLWorkerRuntime(
            config,
            use_grpc=False,
            grpc_target="127.0.0.1:50055",
            dry_run=True,
        )

        summary = runtime.run()
        assert summary.status == "dry-run"
        assert summary.extras["mode"] == "policy_eval"


class TestResumeTrainingConfig:
    """Tests for resume training configuration handling."""

    def test_resume_training_mode_in_extras(self) -> None:
        """Test that resume_training mode is properly configured."""
        config = _make_config(
            extras={
                "mode": "resume_training",
                "checkpoint_path": "/path/to/checkpoint.cleanrl_model",
            }
        )

        assert config.extras["mode"] == "resume_training"
        assert config.extras["checkpoint_path"] == "/path/to/checkpoint.cleanrl_model"

    def test_resume_training_dry_run(self) -> None:
        """Test dry run with resume training config."""
        config = _make_config(
            extras={
                "mode": "resume_training",
                "checkpoint_path": "/path/to/checkpoint.cleanrl_model",
                "fastlane_only": True,
            }
        )
        runtime = CleanRLWorkerRuntime(
            config,
            use_grpc=False,
            grpc_target="127.0.0.1:50055",
            dry_run=True,
        )

        summary = runtime.run()
        assert summary.status == "dry-run"
        assert summary.config["algo"] == "ppo"

    def test_learning_rate_override_env_var(self, monkeypatch) -> None:
        """Test that CLEANRL_LEARNING_RATE env var is recognized."""
        # This tests the env var mechanism, actual usage is in sitecustomize
        monkeypatch.setenv("CLEANRL_LEARNING_RATE", "0.001")

        lr = os.environ.get("CLEANRL_LEARNING_RATE")
        assert lr == "0.001"
        assert float(lr) == 0.001


class TestEnvironmentVariableFlow:
    """Tests for environment variable propagation."""

    def test_cleanrl_resume_path_env_var_format(self, tmp_path: Path) -> None:
        """Test that CLEANRL_RESUME_PATH is properly formatted."""
        checkpoint_path = tmp_path / "model.cleanrl_model"
        checkpoint_path.touch()

        env_value = str(checkpoint_path)
        assert Path(env_value).exists()
        assert env_value.endswith(".cleanrl_model")

    def test_environment_variables_for_resume(self) -> None:
        """Test the expected environment variables for resume training."""
        # These are the env vars that cleanrl_resume_form.py sets
        expected_vars = {
            "CLEANRL_RUN_ID": "test-run-123",
            "CLEANRL_AGENT_ID": "cleanrl_resume",
            "CLEANRL_RESUME_PATH": "/path/to/checkpoint.cleanrl_model",
            "TRACK_TENSORBOARD": "1",
            "TRACK_WANDB": "0",
        }

        for key, value in expected_vars.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            # Verify they can be set as env vars
            with patch.dict(os.environ, {key: value}):
                assert os.environ[key] == value
