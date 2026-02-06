"""Comprehensive tests for XuanCe worker UI flows.

These tests cover the main UI flows for the xuance_train_form.py widget:
1. Train Agent flow - training with selected backend and algorithm
2. Test/Evaluate flow - loading model and evaluating

Each flow validates:
- Configuration handling
- Backend selection (PyTorch, TensorFlow, MindSpore)
- Algorithm filtering by paradigm (single-agent, multi-agent)
- Parameter configuration
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from xuance_worker.config import XuanCeWorkerConfig
from xuance_worker.runtime import XuanCeWorkerRuntime
from xuance_worker.algorithm_registry import (
    get_algorithms,
    get_algorithm_choices,
    get_algorithms_by_category,
    is_algorithm_available,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def _make_config(
    *,
    run_id: str = "test-run-123",
    method: str = "ppo",
    env: str = "classic_control",
    env_id: str = "CartPole-v1",
    dl_toolbox: str = "torch",
    running_steps: int = 1000,
    seed: int | None = 42,
    device: str = "cpu",
    parallels: int = 8,
    test_mode: bool = False,
    extras: dict[str, Any] | None = None,
) -> XuanCeWorkerConfig:
    """Create a XuanCeWorkerConfig for testing."""
    return XuanCeWorkerConfig(
        run_id=run_id,
        method=method,
        env=env,
        env_id=env_id,
        dl_toolbox=dl_toolbox,
        running_steps=running_steps,
        seed=seed,
        device=device,
        parallels=parallels,
        test_mode=test_mode,
        extras=dict(extras or {}),
    )


def _make_runtime(
    config: XuanCeWorkerConfig,
    *,
    dry_run: bool = True,
) -> XuanCeWorkerRuntime:
    """Create a XuanCeWorkerRuntime for testing."""
    return XuanCeWorkerRuntime(config, dry_run=dry_run)


# =============================================================================
# UI Flow 1: Train Agent Tests
# =============================================================================


class TestTrainAgentFlow:
    """Tests for the 'Train Agent' button flow.

    This flow:
    - Selects a deep learning backend (PyTorch, TensorFlow, MindSpore)
    - Selects algorithm paradigm (single-agent or multi-agent)
    - Selects specific algorithm from filtered list
    - Configures training parameters
    - Executes training
    """

    def test_train_config_parsing(self) -> None:
        """Test that training configuration is properly parsed."""
        config = _make_config(
            method="PPO_Clip",
            env="classic_control",
            env_id="CartPole-v1",
            dl_toolbox="torch",
            running_steps=100000,
            extras={
                "learning_rate": 0.0003,
                "gamma": 0.99,
            },
        )

        assert config.method == "PPO_Clip"
        assert config.env_id == "CartPole-v1"
        assert config.dl_toolbox == "torch"
        assert config.running_steps == 100000
        assert config.extras["learning_rate"] == 0.0003

    def test_train_dry_run_returns_summary(self) -> None:
        """Test that dry run returns a proper summary without executing."""
        config = _make_config(
            method="PPO_Clip",
            dl_toolbox="torch",
        )
        runtime = _make_runtime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == "PPO_Clip"
        assert summary.config["dl_toolbox"] == "torch"

    def test_train_with_pytorch_backend(self) -> None:
        """Test training configuration with PyTorch backend."""
        config = _make_config(
            method="DreamerV3",  # PyTorch-only algorithm
            dl_toolbox="torch",
        )
        runtime = _make_runtime(config, dry_run=True)

        summary = runtime.run()

        assert summary.config["dl_toolbox"] == "torch"
        assert is_algorithm_available("DreamerV3", "torch")

    def test_train_with_tensorflow_backend(self) -> None:
        """Test training configuration with TensorFlow backend."""
        config = _make_config(
            method="PPO_Clip",
            dl_toolbox="tensorflow",
        )
        runtime = _make_runtime(config, dry_run=True)

        summary = runtime.run()

        assert summary.config["dl_toolbox"] == "tensorflow"

    def test_train_with_mindspore_backend(self) -> None:
        """Test training configuration with MindSpore backend."""
        config = _make_config(
            method="DQN",
            dl_toolbox="mindspore",
        )
        runtime = _make_runtime(config, dry_run=True)

        summary = runtime.run()

        assert summary.config["dl_toolbox"] == "mindspore"

    def test_train_config_from_json(self, tmp_path: Path) -> None:
        """Test parsing configuration from JSON file (as GUI would create)."""
        config_dict = {
            "run_id": "train-run-001",
            "method": "SAC",
            "env": "mujoco",
            "env_id": "HalfCheetah-v4",
            "dl_toolbox": "torch",
            "running_steps": 1000000,
            "seed": 42,
            "device": "cuda:0",
            "parallels": 8,
            "extras": {
                "algo_params": {
                    "learning_rate": 0.0003,
                    "gamma": 0.99,
                    "tau": 0.005,
                },
            },
        }

        config = XuanCeWorkerConfig.from_dict(config_dict)

        assert config.run_id == "train-run-001"
        assert config.method == "SAC"
        assert config.env == "mujoco"
        assert config.dl_toolbox == "torch"

    def test_train_single_agent_algorithms(self) -> None:
        """Test that single-agent algorithms work in training."""
        single_agent_algos = ["DQN", "PPO_Clip", "SAC", "TD3", "DDPG"]

        for algo in single_agent_algos:
            config = _make_config(method=algo)
            runtime = _make_runtime(config, dry_run=True)
            summary = runtime.run()

            assert summary.status == "dry-run"
            assert summary.method == algo

    def test_train_multi_agent_algorithms(self) -> None:
        """Test that multi-agent algorithms work in training."""
        multi_agent_algos = ["MAPPO", "QMIX", "VDN", "MADDPG"]

        for algo in multi_agent_algos:
            config = _make_config(
                method=algo,
                env="mpe",
                env_id="simple_spread_v3",
            )
            runtime = _make_runtime(config, dry_run=True)
            summary = runtime.run()

            assert summary.status == "dry-run"
            assert summary.method == algo


# =============================================================================
# UI Flow 2: Test/Evaluate Mode Tests
# =============================================================================


class TestEvaluatePolicyFlow:
    """Tests for the 'Test/Evaluate' mode flow.

    This flow:
    - Sets test_mode=True
    - Loads a trained model
    - Runs evaluation episodes
    """

    def test_eval_config_with_test_mode(self) -> None:
        """Test that test_mode is properly set."""
        config = _make_config(
            test_mode=True,
        )

        assert config.test_mode is True

    def test_eval_dry_run_returns_summary(self) -> None:
        """Test that dry run works for evaluation mode."""
        config = _make_config(
            method="PPO_Clip",
            test_mode=True,
        )
        runtime = _make_runtime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.config["test_mode"] is True

    def test_eval_with_model_path(self) -> None:
        """Test evaluation with model path in extras."""
        config = _make_config(
            method="DQN",
            test_mode=True,
            extras={
                "model_dir_load": "/path/to/trained/model",
                "test_episode": 10,
            },
        )

        assert config.extras["model_dir_load"] == "/path/to/trained/model"
        assert config.extras["test_episode"] == 10


# =============================================================================
# Backend Selection Tests
# =============================================================================


class TestBackendSelection:
    """Tests for backend selection in UI."""

    def test_backend_affects_available_algorithms(self) -> None:
        """Test that backend selection affects available algorithms."""
        torch_algos = get_algorithms("torch", "single_agent")
        tf_algos = get_algorithms("tensorflow", "single_agent")

        # PyTorch should have more algorithms
        assert len(torch_algos) >= len(tf_algos)

        # DreamerV3 only in PyTorch
        assert "DreamerV3" in torch_algos
        assert "DreamerV3" not in tf_algos

    def test_algorithm_choices_for_ui_dropdown(self) -> None:
        """Test getting algorithm choices for UI dropdown."""
        choices = get_algorithm_choices("torch", "single_agent")

        assert isinstance(choices, list)
        assert len(choices) > 0

        # Each choice should be (key, display_name)
        for key, display_name in choices:
            assert isinstance(key, str)
            assert isinstance(display_name, str)

    def test_algorithm_categories_for_ui(self) -> None:
        """Test getting algorithms by category for organized UI."""
        categories = get_algorithms_by_category("torch", "single_agent")

        assert isinstance(categories, dict)
        assert "Policy Optimization" in categories
        assert "Value-based" in categories

        # PPO should be in Policy Optimization
        policy_opt_keys = [a.key for a in categories["Policy Optimization"]]
        assert "PPO_Clip" in policy_opt_keys


# =============================================================================
# Paradigm Selection Tests
# =============================================================================


class TestParadigmSelection:
    """Tests for paradigm selection (single-agent vs multi-agent)."""

    def test_single_agent_paradigm_filters_correctly(self) -> None:
        """Test that single-agent paradigm filters correctly."""
        algos = get_algorithms("torch", "single_agent")

        # Should include single-agent algorithms
        assert "DQN" in algos
        assert "PPO_Clip" in algos

        # Should NOT include multi-agent algorithms
        assert "MAPPO" not in algos
        assert "QMIX" not in algos

    def test_multi_agent_paradigm_filters_correctly(self) -> None:
        """Test that multi-agent paradigm filters correctly."""
        algos = get_algorithms("torch", "multi_agent")

        # Should include multi-agent algorithms
        assert "MAPPO" in algos
        assert "QMIX" in algos

        # Should NOT include single-agent algorithms
        assert "DQN" not in algos
        assert "PPO_Clip" not in algos


# =============================================================================
# Environment Selection Tests
# =============================================================================


class TestEnvironmentSelection:
    """Tests for environment selection in UI."""

    def test_single_agent_environments(self) -> None:
        """Test configuration for single-agent environments."""
        environments = [
            ("classic_control", "CartPole-v1"),
            ("atari", "Pong-v5"),
            ("mujoco", "HalfCheetah-v4"),
            ("box2d", "LunarLander-v2"),
        ]

        for env, env_id in environments:
            config = _make_config(
                method="PPO_Clip",
                env=env,
                env_id=env_id,
            )

            assert config.env == env
            assert config.env_id == env_id

    def test_multi_agent_environments(self) -> None:
        """Test configuration for multi-agent environments."""
        environments = [
            ("mpe", "simple_spread_v3"),
            ("smac", "3m"),
            ("mpe", "simple_adversary_v3"),
        ]

        for env, env_id in environments:
            config = _make_config(
                method="MAPPO",
                env=env,
                env_id=env_id,
            )

            assert config.env == env
            assert config.env_id == env_id


# =============================================================================
# Parameter Configuration Tests
# =============================================================================


class TestParameterConfiguration:
    """Tests for algorithm parameter configuration."""

    def test_learning_rate_configuration(self) -> None:
        """Test learning rate parameter configuration."""
        config = _make_config(
            extras={"learning_rate": 0.0001},
        )

        assert config.extras["learning_rate"] == 0.0001

    def test_network_architecture_configuration(self) -> None:
        """Test network architecture parameter configuration."""
        config = _make_config(
            extras={
                "representation_hidden_size": [256, 256],
                "actor_hidden_size": [256, 256],
                "critic_hidden_size": [256, 256],
            },
        )

        assert config.extras["representation_hidden_size"] == [256, 256]

    def test_training_parameters_configuration(self) -> None:
        """Test training parameter configuration."""
        config = _make_config(
            running_steps=500000,
            parallels=16,
            seed=123,
            extras={
                "batch_size": 256,
                "n_epoch": 10,
                "gamma": 0.99,
            },
        )

        assert config.running_steps == 500000
        assert config.parallels == 16
        assert config.seed == 123
        assert config.extras["batch_size"] == 256

    def test_ppo_specific_parameters(self) -> None:
        """Test PPO-specific parameter configuration."""
        config = _make_config(
            method="PPO_Clip",
            extras={
                "clip_range": 0.2,
                "vf_coef": 0.5,
                "ent_coef": 0.01,
                "use_gae": True,
                "gae_lambda": 0.95,
            },
        )

        assert config.extras["clip_range"] == 0.2
        assert config.extras["vf_coef"] == 0.5
        assert config.extras["use_gae"] is True

    def test_dqn_specific_parameters(self) -> None:
        """Test DQN-specific parameter configuration."""
        config = _make_config(
            method="DQN",
            extras={
                "buffer_size": 100000,
                "start_greedy": 1.0,
                "end_greedy": 0.05,
                "sync_frequency": 500,
            },
        )

        assert config.extras["buffer_size"] == 100000
        assert config.extras["start_greedy"] == 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests simulating full UI flows."""

    def test_complete_single_agent_flow(self) -> None:
        """Test complete single-agent training flow."""
        # Step 1: Select backend
        backend = "torch"
        assert is_algorithm_available("PPO_Clip", backend)

        # Step 2: Get algorithm choices for single-agent
        choices = get_algorithm_choices(backend, "single_agent")
        algo_keys = [c[0] for c in choices]
        assert "PPO_Clip" in algo_keys

        # Step 3: Create configuration
        config = XuanCeWorkerConfig(
            run_id="single-agent-flow",
            method="PPO_Clip",
            env="classic_control",
            env_id="CartPole-v1",
            dl_toolbox=backend,
            running_steps=100000,
            seed=42,
            extras={
                "learning_rate": 0.0003,
                "clip_range": 0.2,
            },
        )

        # Step 4: Execute (dry-run)
        runtime = XuanCeWorkerRuntime(config, dry_run=True)
        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == "PPO_Clip"
        assert summary.config["dl_toolbox"] == "torch"

    def test_complete_multi_agent_flow(self) -> None:
        """Test complete multi-agent training flow."""
        # Step 1: Select backend
        backend = "torch"

        # Step 2: Get algorithm choices for multi-agent
        choices = get_algorithm_choices(backend, "multi_agent")
        algo_keys = [c[0] for c in choices]
        assert "MAPPO" in algo_keys

        # Step 3: Create configuration
        config = XuanCeWorkerConfig(
            run_id="multi-agent-flow",
            method="MAPPO",
            env="mpe",
            env_id="simple_spread_v3",
            dl_toolbox=backend,
            running_steps=1000000,
            parallels=128,
            extras={
                "n_agents": 3,
                "share_policy": True,
            },
        )

        # Step 4: Execute (dry-run)
        runtime = XuanCeWorkerRuntime(config, dry_run=True)
        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == "MAPPO"

    def test_backend_switch_flow(self) -> None:
        """Test switching between backends in UI flow."""
        # Start with PyTorch
        torch_algos = get_algorithms("torch", "single_agent")
        assert "DreamerV3" in torch_algos

        # Switch to TensorFlow
        tf_algos = get_algorithms("tensorflow", "single_agent")
        assert "DreamerV3" not in tf_algos
        assert "PPO_Clip" in tf_algos  # Core algorithms still available

        # Configuration should reflect backend
        config = _make_config(
            method="PPO_Clip",
            dl_toolbox="tensorflow",
        )
        runtime = _make_runtime(config, dry_run=True)
        summary = runtime.run()

        assert summary.config["dl_toolbox"] == "tensorflow"

    def test_paradigm_switch_flow(self) -> None:
        """Test switching between paradigms in UI flow."""
        # Start with single-agent
        sa_algos = get_algorithms("torch", "single_agent")
        assert "DQN" in sa_algos
        assert "MAPPO" not in sa_algos

        # Switch to multi-agent
        ma_algos = get_algorithms("torch", "multi_agent")
        assert "MAPPO" in ma_algos
        assert "DQN" not in ma_algos

        # Configuration should work for both
        sa_config = _make_config(method="DQN")
        ma_config = _make_config(method="MAPPO", env="mpe", env_id="simple_spread_v3")

        sa_summary = _make_runtime(sa_config, dry_run=True).run()
        ma_summary = _make_runtime(ma_config, dry_run=True).run()

        assert sa_summary.method == "DQN"
        assert ma_summary.method == "MAPPO"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in UI flows."""

    def test_runtime_error_for_missing_xuance(self) -> None:
        """Test that RuntimeError is raised when XuanCe is not installed."""
        config = _make_config()
        runtime = XuanCeWorkerRuntime(config, dry_run=False)

        with pytest.raises(RuntimeError, match="XuanCe is not installed"):
            runtime.run()

    def test_config_file_not_found(self, tmp_path: Path) -> None:
        """Test error when config file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            XuanCeWorkerConfig.from_json_file(nonexistent)

    def test_config_invalid_json(self, tmp_path: Path) -> None:
        """Test error when config file contains invalid JSON."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ not valid }")

        with pytest.raises(json.JSONDecodeError):
            XuanCeWorkerConfig.from_json_file(invalid_file)
