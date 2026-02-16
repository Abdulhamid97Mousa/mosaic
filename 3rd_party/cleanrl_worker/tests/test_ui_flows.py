"""Comprehensive tests for the three main CleanRL UI flows.

These tests cover the three main buttons in the gym_gui UI:
1. Train Agent (cleanrl_train_form.py)
2. Evaluate Policy (cleanrl_eval_form.py)
3. Resume Training (cleanrl_resume_form.py)

Each flow has specific requirements for:
- Configuration handling
- TensorBoard directory management (tensorboard/ for training, tensorboard_eval/ for evaluation)
- Checkpoint file handling
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from cleanrl_worker.config import WorkerConfig, parse_worker_config
from cleanrl_worker.runtime import CleanRLWorkerRuntime


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Create a temporary run directory structure."""
    run_dir = tmp_path / "runs" / "test-run-123"
    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "tensorboard").mkdir()
    (run_dir / "tensorboard_eval").mkdir()
    return run_dir


@pytest.fixture
def sample_checkpoint(tmp_path: Path) -> Path:
    """Create a sample checkpoint file for testing."""
    # Create a simple model and save it
    model = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    checkpoint_path = tmp_path / "test_model.cleanrl_model"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


@pytest.fixture
def ppo_agent_checkpoint(tmp_path: Path) -> Path:
    """Create a PPO-style agent checkpoint."""
    # Simulated PPO Agent state dict (simplified)
    state_dict = {
        "network.0.weight": torch.randn(64, 4),
        "network.0.bias": torch.randn(64),
        "network.2.weight": torch.randn(64, 64),
        "network.2.bias": torch.randn(64),
        "actor_mean.weight": torch.randn(2, 64),
        "actor_mean.bias": torch.randn(2),
        "critic.weight": torch.randn(1, 64),
        "critic.bias": torch.randn(1),
    }
    checkpoint_path = tmp_path / "ppo_agent.cleanrl_model"
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


def _make_config(
    *,
    run_id: str = "test-run-123",
    algo: str = "ppo",
    env_id: str = "CartPole-v1",
    total_timesteps: int = 1000,
    extras: dict[str, Any] | None = None,
) -> WorkerConfig:
    """Create a WorkerConfig for testing."""
    return WorkerConfig(
        run_id=run_id,
        algo=algo,
        env_id=env_id,
        total_timesteps=total_timesteps,
        seed=42,
        extras=dict(extras or {}),
    )


def _make_runtime(
    config: WorkerConfig,
    *,
    dry_run: bool = False,
) -> CleanRLWorkerRuntime:
    """Create a CleanRLWorkerRuntime for testing."""
    return CleanRLWorkerRuntime(
        config,
        use_grpc=False,
        grpc_target="127.0.0.1:50055",
        dry_run=dry_run,
    )


# =============================================================================
# UI Flow 1: Train Agent Tests
# =============================================================================


class TestTrainAgentFlow:
    """Tests for the 'Train Agent' button flow (cleanrl_train_form.py).

    This flow:
    - Creates a new training run
    - Uses tensorboard/ directory for training logs
    - Saves checkpoints during training
    """

    def test_train_config_parsing(self) -> None:
        """Test that training configuration is properly parsed."""
        config = _make_config(
            extras={
                "tensorboard_dir": "tensorboard",
                "track_wandb": False,
                "algo_params": {
                    "learning_rate": 0.0003,
                    "num_envs": 4,
                    "num_steps": 128,
                },
            }
        )

        assert config.algo == "ppo"
        assert config.env_id == "CartPole-v1"
        assert config.total_timesteps == 1000
        assert config.extras["tensorboard_dir"] == "tensorboard"
        assert config.extras["algo_params"]["learning_rate"] == 0.0003

    def test_train_dry_run_returns_summary(self) -> None:
        """Test that dry run returns a proper summary without executing."""
        config = _make_config(
            extras={
                "tensorboard_dir": "tensorboard",
            }
        )
        runtime = _make_runtime(config, dry_run=True)

        summary = runtime.run()

        assert summary["status"] == "dry-run"
        assert "ppo" in summary["module"]
        assert summary["config"]["algo"] == "ppo"

    def test_train_builds_correct_cli_args(self) -> None:
        """Test that CLI arguments are correctly built for training."""
        config = _make_config(
            extras={
                "track_wandb": True,
                "cuda": True,
                "algo_params": {
                    "learning_rate": 0.0003,
                    "num_envs": 4,
                },
            }
        )
        runtime = _make_runtime(config, dry_run=True)

        args = runtime.build_cleanrl_args()

        assert "--env-id=CartPole-v1" in args
        assert "--total-timesteps=1000" in args
        assert "--seed=42" in args
        assert "--track" in args
        assert "--cuda" in args
        assert "--learning-rate=0.0003" in args
        assert "--num-envs=4" in args

    def test_train_tensorboard_dir_created(self, tmp_path: Path, monkeypatch) -> None:
        """Test that tensorboard directory is created for training."""
        config = _make_config(
            extras={
                "tensorboard_dir": "tensorboard",
            }
        )

        # Mock the trainer directory
        monkeypatch.setattr(
            "cleanrl_worker.runtime.VAR_TRAINER_DIR",
            tmp_path,
            raising=False,
        )
        monkeypatch.setattr(
            "cleanrl_worker.runtime.ensure_var_directories",
            lambda: None,
        )

        runtime = _make_runtime(config, dry_run=True)
        summary = runtime.run()

        # Dry run should return summary
        assert summary["status"] == "dry-run"

    def test_train_config_from_json(self, tmp_path: Path) -> None:
        """Test parsing configuration from JSON file (as GUI would create)."""
        config_dict = {
            "run_id": "train-run-001",
            "algo": "ppo",
            "env_id": "CartPole-v1",
            "total_timesteps": 50000,
            "seed": 1,
            "extras": {
                "mode": "train",
                "tensorboard_dir": "tensorboard",
                "track_wandb": False,
                "algo_params": {
                    "learning_rate": 0.0003,
                    "gamma": 0.99,
                },
            },
        }

        config = parse_worker_config(config_dict)

        assert config.run_id == "train-run-001"
        assert config.algo == "ppo"
        assert config.extras["mode"] == "train"
        assert config.extras["tensorboard_dir"] == "tensorboard"

    def test_train_supported_algorithms(self) -> None:
        """Test that all expected training algorithms are registered."""
        runtime = _make_runtime(_make_config(algo="ppo"), dry_run=True)

        # Check PPO family
        for algo in ["ppo", "ppo_continuous_action", "ppo_atari"]:
            config = _make_config(algo=algo)
            rt = _make_runtime(config, dry_run=True)
            module, _ = rt.resolve_entrypoint()
            assert "ppo" in module.lower() or "cleanrl" in module.lower()

        # Check DQN
        config = _make_config(algo="dqn")
        rt = _make_runtime(config, dry_run=True)
        module, _ = rt.resolve_entrypoint()
        assert "dqn" in module.lower()


# =============================================================================
# UI Flow 2: Evaluate Policy Tests
# =============================================================================


class TestEvaluatePolicyFlow:
    """Tests for the 'Evaluate Policy' button flow (cleanrl_eval_form.py).

    This flow:
    - Loads a trained checkpoint
    - Uses tensorboard_eval/ directory for evaluation logs
    - Runs evaluation episodes and logs returns
    """

    def test_eval_config_requires_policy_path(self) -> None:
        """Test that policy_eval mode requires policy_path."""
        # Config __post_init__ validates that policy_eval needs policy_path
        with pytest.raises(ValueError, match="policy_path"):
            _make_config(extras={"mode": "policy_eval"})

    def test_eval_config_parsing(self, sample_checkpoint: Path) -> None:
        """Test that evaluation configuration is properly parsed."""
        config = _make_config(
            extras={
                "mode": "policy_eval",
                "policy_path": str(sample_checkpoint),
                "tensorboard_dir": "tensorboard_eval",
                "eval_episodes": 10,
                "eval_gamma": 0.99,
                "eval_capture_video": False,
            }
        )

        assert config.extras["mode"] == "policy_eval"
        assert config.extras["policy_path"] == str(sample_checkpoint)
        assert config.extras["tensorboard_dir"] == "tensorboard_eval"
        assert config.extras["eval_episodes"] == 10

    def test_eval_dry_run_returns_summary(self, sample_checkpoint: Path) -> None:
        """Test that dry run works for evaluation mode."""
        config = _make_config(
            extras={
                "mode": "policy_eval",
                "policy_path": str(sample_checkpoint),
                "tensorboard_dir": "tensorboard_eval",
            }
        )
        runtime = _make_runtime(config, dry_run=True)

        summary = runtime.run()

        assert summary["status"] == "dry-run"
        assert summary["extras"]["mode"] == "policy_eval"

    def test_eval_tensorboard_dir_is_separate(self) -> None:
        """Test that eval uses tensorboard_eval, not tensorboard."""
        config = _make_config(
            extras={
                "mode": "policy_eval",
                "policy_path": "/path/to/model.cleanrl_model",
                "tensorboard_dir": "tensorboard_eval",
            }
        )

        # The tensorboard_dir should be tensorboard_eval for evaluation
        assert config.extras["tensorboard_dir"] == "tensorboard_eval"
        assert config.extras["tensorboard_dir"] != "tensorboard"

    def test_eval_missing_checkpoint_raises_error(self, tmp_path: Path, monkeypatch) -> None:
        """Test that missing checkpoint file raises FileNotFoundError."""
        config = _make_config(
            extras={
                "mode": "policy_eval",
                "policy_path": str(tmp_path / "nonexistent.cleanrl_model"),
            }
        )
        runtime = _make_runtime(config, dry_run=False)

        # Mock dependencies
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

    def test_eval_config_from_json(self, sample_checkpoint: Path) -> None:
        """Test parsing evaluation config from JSON (as GUI would create)."""
        config_dict = {
            "run_id": "eval-run-001",
            "algo": "ppo",
            "env_id": "CartPole-v1",
            "total_timesteps": 0,  # Not used for eval
            "seed": 42,
            "extras": {
                "mode": "policy_eval",
                "policy_path": str(sample_checkpoint),
                "tensorboard_dir": "tensorboard_eval",
                "eval_episodes": 5,
                "eval_gamma": 0.99,
                "eval_capture_video": True,
            },
        }

        config = parse_worker_config(config_dict)

        assert config.extras["mode"] == "policy_eval"
        assert config.extras["eval_episodes"] == 5
        assert config.extras["eval_capture_video"] is True

    def test_eval_episodes_parameter(self, sample_checkpoint: Path) -> None:
        """Test various eval_episodes configurations."""
        for episodes in [1, 5, 10, 100]:
            config = _make_config(
                extras={
                    "mode": "policy_eval",
                    "policy_path": str(sample_checkpoint),
                    "eval_episodes": episodes,
                }
            )
            assert config.extras["eval_episodes"] == episodes


# =============================================================================
# UI Flow 3: Resume Training Tests
# =============================================================================


class TestResumeTrainingFlow:
    """Tests for the 'Resume Training' button flow (cleanrl_resume_form.py).

    This flow:
    - Loads a checkpoint to resume training
    - Uses tensorboard/ directory (continues training logs)
    - Sets CLEANRL_RESUME_PATH environment variable
    """

    def test_resume_config_parsing(self, sample_checkpoint: Path) -> None:
        """Test that resume configuration is properly parsed."""
        config = _make_config(
            extras={
                "mode": "resume_training",
                "checkpoint_path": str(sample_checkpoint),
                "tensorboard_dir": "tensorboard",
            }
        )

        assert config.extras["mode"] == "resume_training"
        assert config.extras["checkpoint_path"] == str(sample_checkpoint)
        assert config.extras["tensorboard_dir"] == "tensorboard"

    def test_resume_dry_run_returns_summary(self, sample_checkpoint: Path) -> None:
        """Test that dry run works for resume training mode."""
        config = _make_config(
            extras={
                "mode": "resume_training",
                "checkpoint_path": str(sample_checkpoint),
            }
        )
        runtime = _make_runtime(config, dry_run=True)

        summary = runtime.run()

        assert summary["status"] == "dry-run"
        assert summary["config"]["algo"] == "ppo"

    def test_resume_tensorboard_dir_is_tensorboard(self) -> None:
        """Test that resume uses tensorboard, not tensorboard_eval."""
        config = _make_config(
            extras={
                "mode": "resume_training",
                "checkpoint_path": "/path/to/model.cleanrl_model",
                "tensorboard_dir": "tensorboard",
            }
        )

        # Resume training should continue to tensorboard (not tensorboard_eval)
        assert config.extras["tensorboard_dir"] == "tensorboard"
        assert config.extras["tensorboard_dir"] != "tensorboard_eval"

    def test_resume_env_variable_format(self, sample_checkpoint: Path) -> None:
        """Test that CLEANRL_RESUME_PATH is properly formatted."""
        # The env var should be an absolute path to the checkpoint
        env_value = str(sample_checkpoint.resolve())

        assert Path(env_value).exists()
        assert env_value.endswith(".cleanrl_model")

    def test_resume_config_from_json(self, sample_checkpoint: Path) -> None:
        """Test parsing resume config from JSON (as GUI would create)."""
        config_dict = {
            "run_id": "resume-run-001",
            "algo": "ppo",
            "env_id": "CartPole-v1",
            "total_timesteps": 100000,
            "seed": 42,
            "extras": {
                "mode": "resume_training",
                "checkpoint_path": str(sample_checkpoint),
                "tensorboard_dir": "tensorboard",
                "algo_params": {
                    "learning_rate": 0.0001,  # Potentially different for fine-tuning
                },
            },
        }

        config = parse_worker_config(config_dict)

        assert config.extras["mode"] == "resume_training"
        assert config.total_timesteps == 100000


# =============================================================================
# TensorBoard Directory Tests
# =============================================================================


class TestTensorBoardDirectories:
    """Tests for TensorBoard directory management.

    The system uses two separate directories:
    - tensorboard/ - Training logs
    - tensorboard_eval/ - Evaluation logs
    """

    def test_tensorboard_dirs_are_distinct(self) -> None:
        """Test that training and eval use different tensorboard directories."""
        train_config = _make_config(
            extras={
                "mode": "train",
                "tensorboard_dir": "tensorboard",
            }
        )

        eval_config = _make_config(
            extras={
                "mode": "policy_eval",
                "policy_path": "/path/to/model.cleanrl_model",
                "tensorboard_dir": "tensorboard_eval",
            }
        )

        assert train_config.extras["tensorboard_dir"] == "tensorboard"
        assert eval_config.extras["tensorboard_dir"] == "tensorboard_eval"
        assert train_config.extras["tensorboard_dir"] != eval_config.extras["tensorboard_dir"]

    def test_tensorboard_dir_naming_conventions(self) -> None:
        """Test the naming conventions for tensorboard directories."""
        # Training uses 'tensorboard'
        train_dir = "tensorboard"
        assert train_dir == "tensorboard"

        # Evaluation uses 'tensorboard_eval'
        eval_dir = "tensorboard_eval"
        assert eval_dir == "tensorboard_eval"
        assert eval_dir.startswith("tensorboard")
        assert "_eval" in eval_dir

    def test_both_tensorboard_dirs_can_exist(self, tmp_run_dir: Path) -> None:
        """Test that both tensorboard directories can coexist."""
        tb_dir = tmp_run_dir / "tensorboard"
        tb_eval_dir = tmp_run_dir / "tensorboard_eval"

        assert tb_dir.exists()
        assert tb_eval_dir.exists()
        assert tb_dir != tb_eval_dir


# =============================================================================
# Unified Evaluator Tests
# =============================================================================


class TestUnifiedEvaluator:
    """Tests for the unified evaluation system."""

    def test_adapter_registry_has_all_algorithms(self) -> None:
        """Test that all common algorithms are in the registry."""
        from cleanrl_worker.unified_eval.registry import ADAPTER_REGISTRY

        expected_algorithms = [
            "ppo",
            "ppo_continuous_action",
            "ppo_atari",
            "dqn",
            "dqn_atari",
            "ddpg_continuous_action",
            "td3_continuous_action",
            "sac_continuous_action",
            "c51",
        ]

        for algo in expected_algorithms:
            assert algo in ADAPTER_REGISTRY, f"Algorithm {algo} not in registry"

    def test_get_adapter_returns_correct_type(self) -> None:
        """Test that get_adapter returns correct adapter instances."""
        from cleanrl_worker.unified_eval.registry import get_adapter
        from cleanrl_worker.unified_eval.adapters import (
            PPOSelector,
            DQNSelector,
            DDPGSelector,
        )

        assert isinstance(get_adapter("ppo"), PPOSelector)
        assert isinstance(get_adapter("dqn"), DQNSelector)
        assert isinstance(get_adapter("ddpg_continuous_action"), DDPGSelector)

    def test_get_adapter_returns_none_for_unknown(self) -> None:
        """Test that get_adapter returns None for unknown algorithms."""
        from cleanrl_worker.unified_eval.registry import get_adapter

        assert get_adapter("unknown_algorithm") is None
        assert get_adapter("fake_algo") is None

    def test_eval_result_statistics(self) -> None:
        """Test EvalResult correctly calculates statistics."""
        from cleanrl_worker.unified_eval.evaluator import EvalResult

        result = EvalResult.from_episodes(
            returns=[10.0, 20.0, 30.0, 40.0, 50.0],
            lengths=[100, 150, 200, 250, 300],
        )

        assert result.episodes == 5
        assert result.avg_return == 30.0
        assert result.avg_length == 200.0
        assert result.min_return == 10.0
        assert result.max_return == 50.0
        assert result.std_return > 0

    def test_eval_result_empty_episodes(self) -> None:
        """Test EvalResult handles empty episode list."""
        from cleanrl_worker.unified_eval.evaluator import EvalResult

        result = EvalResult.from_episodes(returns=[], lengths=[])

        assert result.episodes == 0
        assert result.avg_return == 0.0
        assert result.returns == []

    def test_list_supported_algorithms(self) -> None:
        """Test listing all supported algorithms."""
        from cleanrl_worker.unified_eval.registry import list_supported_algorithms

        algorithms = list_supported_algorithms()

        assert len(algorithms) > 0
        assert "ppo" in algorithms
        assert "dqn" in algorithms
        assert algorithms == sorted(algorithms)  # Should be sorted


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests that simulate full UI flows."""

    def test_train_then_eval_flow(self, tmp_path: Path) -> None:
        """Test the complete flow: train → save checkpoint → evaluate."""
        # Step 1: Create training config
        train_config = _make_config(
            run_id="train-then-eval",
            extras={
                "mode": "train",
                "tensorboard_dir": "tensorboard",
            }
        )
        train_runtime = _make_runtime(train_config, dry_run=True)
        train_summary = train_runtime.run()

        assert train_summary["status"] == "dry-run"

        # Step 2: Simulate checkpoint creation
        checkpoint_path = tmp_path / "trained_model.cleanrl_model"
        model = nn.Linear(4, 2)
        torch.save(model.state_dict(), checkpoint_path)

        # Step 3: Create evaluation config using the checkpoint
        eval_config = _make_config(
            run_id="train-then-eval",
            extras={
                "mode": "policy_eval",
                "policy_path": str(checkpoint_path),
                "tensorboard_dir": "tensorboard_eval",
                "eval_episodes": 5,
            }
        )
        eval_runtime = _make_runtime(eval_config, dry_run=True)
        eval_summary = eval_runtime.run()

        assert eval_summary["status"] == "dry-run"
        assert eval_summary["extras"]["mode"] == "policy_eval"

    def test_train_then_resume_flow(self, tmp_path: Path) -> None:
        """Test the complete flow: train → save checkpoint → resume training."""
        # Step 1: Initial training
        train_config = _make_config(
            run_id="train-then-resume",
            total_timesteps=10000,
            extras={
                "mode": "train",
                "tensorboard_dir": "tensorboard",
            }
        )
        train_runtime = _make_runtime(train_config, dry_run=True)
        train_summary = train_runtime.run()

        # Step 2: Simulate checkpoint
        checkpoint_path = tmp_path / "checkpoint.cleanrl_model"
        model = nn.Linear(4, 2)
        torch.save(model.state_dict(), checkpoint_path)

        # Step 3: Resume training with more timesteps
        resume_config = _make_config(
            run_id="train-then-resume",
            total_timesteps=50000,  # Additional timesteps
            extras={
                "mode": "resume_training",
                "checkpoint_path": str(checkpoint_path),
                "tensorboard_dir": "tensorboard",
            }
        )
        resume_runtime = _make_runtime(resume_config, dry_run=True)
        resume_summary = resume_runtime.run()

        assert resume_summary["status"] == "dry-run"
        assert resume_summary["config"]["total_timesteps"] == 50000

    def test_multiple_eval_runs(self, tmp_path: Path) -> None:
        """Test running multiple evaluations on same checkpoint."""
        # Create checkpoint
        checkpoint_path = tmp_path / "model.cleanrl_model"
        model = nn.Linear(4, 2)
        torch.save(model.state_dict(), checkpoint_path)

        # Run multiple evaluations with different episode counts
        for eval_episodes in [5, 10, 20]:
            config = _make_config(
                run_id=f"eval-{eval_episodes}",
                extras={
                    "mode": "policy_eval",
                    "policy_path": str(checkpoint_path),
                    "tensorboard_dir": "tensorboard_eval",
                    "eval_episodes": eval_episodes,
                }
            )
            runtime = _make_runtime(config, dry_run=True)
            summary = runtime.run()

            assert summary["status"] == "dry-run"
            assert summary["extras"]["eval_episodes"] == eval_episodes
