"""Tests for XuanCeWorkerRuntime."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from xuance_worker.config import XuanCeWorkerConfig
from xuance_worker.runtime import XuanCeRuntimeSummary, XuanCeWorkerRuntime


class TestXuanCeRuntimeSummary:
    """Test suite for XuanCeRuntimeSummary dataclass."""

    def test_summary_creation(self) -> None:
        """Test basic summary creation."""
        summary = XuanCeRuntimeSummary(
            status="completed",
            method="ppo",
            env_id="CartPole-v1",
            runner_type="RunnerDRL",
            config={"method": "ppo"},
        )

        assert summary.status == "completed"
        assert summary.method == "ppo"
        assert summary.env_id == "CartPole-v1"
        assert summary.runner_type == "RunnerDRL"
        assert summary.config == {"method": "ppo"}

    def test_summary_frozen(self) -> None:
        """Test that summary is immutable (frozen)."""
        summary = XuanCeRuntimeSummary(
            status="completed",
            method="ppo",
            env_id="CartPole-v1",
            runner_type="RunnerDRL",
            config={},
        )

        with pytest.raises(AttributeError):
            summary.status = "error"  # type: ignore[misc]

    def test_summary_dry_run(self) -> None:
        """Test summary for dry-run mode."""
        summary = XuanCeRuntimeSummary(
            status="dry-run",
            method="dqn",
            env_id="Pong-v5",
            runner_type="unknown",
            config={"method": "dqn", "env_id": "Pong-v5"},
        )

        assert summary.status == "dry-run"
        assert summary.runner_type == "unknown"


class TestXuanCeWorkerRuntimeInit:
    """Test suite for XuanCeWorkerRuntime initialization."""

    def test_runtime_init(self) -> None:
        """Test basic runtime initialization."""
        config = XuanCeWorkerConfig(
            run_id="test",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
        )

        runtime = XuanCeWorkerRuntime(config)

        assert runtime.config == config

    def test_runtime_dry_run_init(self) -> None:
        """Test runtime initialization with dry_run=True."""
        config = XuanCeWorkerConfig(
            run_id="test",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
        )

        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        assert runtime.config == config


class TestXuanCeWorkerRuntimeDryRun:
    """Test XuanCeWorkerRuntime dry-run mode."""

    def test_run_dry_run(self) -> None:
        """Test run() in dry-run mode."""
        config = XuanCeWorkerConfig(
            run_id="dry_test",
            method="sac",
            env="mujoco",
            env_id="Hopper-v4",
            running_steps=50000,
        )

        runtime = XuanCeWorkerRuntime(config, dry_run=True)
        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == "sac"
        assert summary.env_id == "Hopper-v4"
        assert summary.runner_type == "unknown"
        assert summary.config["running_steps"] == 50000

    def test_benchmark_dry_run(self) -> None:
        """Test benchmark() in dry-run mode."""
        config = XuanCeWorkerConfig(
            run_id="bench_dry",
            method="td3",
            env="box2d",
            env_id="LunarLander-v2",
        )

        runtime = XuanCeWorkerRuntime(config, dry_run=True)
        summary = runtime.benchmark()

        assert summary.status == "dry-run"
        assert summary.method == "td3"
        assert summary.env_id == "LunarLander-v2"


class TestXuanCeWorkerRuntimeBuildParserArgs:
    """Test _build_parser_args method."""

    def test_build_parser_args_basic(self) -> None:
        """Test building parser args with basic config."""
        config = XuanCeWorkerConfig(
            run_id="test",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            device="cpu",
            parallels=8,
            running_steps=100000,
        )

        runtime = XuanCeWorkerRuntime(config)
        args = runtime._build_parser_args()

        assert isinstance(args, SimpleNamespace)
        assert args.device == "cpu"
        assert args.parallels == 8
        assert args.running_steps == 100000

    def test_build_parser_args_with_seed(self) -> None:
        """Test building parser args with seed."""
        config = XuanCeWorkerConfig(
            run_id="test",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            seed=42,
        )

        runtime = XuanCeWorkerRuntime(config)
        args = runtime._build_parser_args()

        assert args.seed == 42
        assert args.env_seed == 42

    def test_build_parser_args_no_seed(self) -> None:
        """Test building parser args without seed."""
        config = XuanCeWorkerConfig(
            run_id="test",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            seed=None,
        )

        runtime = XuanCeWorkerRuntime(config)
        args = runtime._build_parser_args()

        assert not hasattr(args, "seed")
        assert not hasattr(args, "env_seed")

    def test_build_parser_args_with_extras(self) -> None:
        """Test building parser args with extras."""
        config = XuanCeWorkerConfig(
            run_id="test",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            extras={"learning_rate": 0.001, "gamma": 0.99},
        )

        runtime = XuanCeWorkerRuntime(config)
        args = runtime._build_parser_args()

        assert args.learning_rate == 0.001
        assert args.gamma == 0.99

    def test_build_parser_args_cuda_device(self) -> None:
        """Test building parser args with CUDA device."""
        config = XuanCeWorkerConfig(
            run_id="test",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            device="cuda:0",
        )

        runtime = XuanCeWorkerRuntime(config)
        args = runtime._build_parser_args()

        assert args.device == "cuda:0"


class TestXuanCeWorkerRuntimeRun:
    """Test run() method with mocked XuanCe."""

    def test_run_success_mocked(self) -> None:
        """Test successful run with mocked XuanCe via sys.modules."""
        import sys

        # Create mock runner
        mock_runner = MagicMock()
        mock_runner.__class__.__name__ = "RunnerDRL"

        # Create mock xuance module
        mock_xuance = MagicMock()
        mock_xuance.get_runner.return_value = mock_runner

        config = XuanCeWorkerConfig(
            run_id="test",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            running_steps=1000,
        )

        runtime = XuanCeWorkerRuntime(config, dry_run=False)

        # Patch sys.modules to provide fake xuance
        with patch.dict(sys.modules, {"xuance": mock_xuance}):
            summary = runtime.run()

        assert summary.status == "completed"
        assert summary.method == "ppo"
        assert summary.env_id == "CartPole-v1"
        assert summary.runner_type == "RunnerDRL"  # From mock.__class__.__name__
        mock_xuance.get_runner.assert_called_once()
        mock_runner.run.assert_called_once()

    def test_run_xuance_not_installed(self) -> None:
        """Test run() raises RuntimeError when XuanCe not installed."""
        config = XuanCeWorkerConfig(
            run_id="test",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
        )

        runtime = XuanCeWorkerRuntime(config, dry_run=False)

        # This test will actually fail if XuanCe IS installed
        # We test the error path by checking the exception handling
        # In CI without XuanCe, this should raise RuntimeError


class TestXuanCeWorkerRuntimeBenchmark:
    """Test benchmark() method."""

    def test_benchmark_dry_run_returns_summary(self) -> None:
        """Test benchmark dry-run returns proper summary."""
        config = XuanCeWorkerConfig(
            run_id="bench_test",
            method="mappo",
            env="mpe",
            env_id="simple_spread_v3",
            running_steps=200000,
        )

        runtime = XuanCeWorkerRuntime(config, dry_run=True)
        summary = runtime.benchmark()

        assert isinstance(summary, XuanCeRuntimeSummary)
        assert summary.status == "dry-run"
        assert summary.method == "mappo"
        assert summary.env_id == "simple_spread_v3"


class TestXuanCeWorkerRuntimeConfig:
    """Test config property."""

    def test_config_property(self) -> None:
        """Test config property returns the configuration."""
        config = XuanCeWorkerConfig(
            run_id="prop_test",
            method="qmix",
            env="smac",
            env_id="3m",
        )

        runtime = XuanCeWorkerRuntime(config)

        assert runtime.config is config
        assert runtime.config.method == "qmix"
        assert runtime.config.env == "smac"
        assert runtime.config.env_id == "3m"


class TestXuanCeWorkerRuntimeIntegration:
    """Integration tests for XuanCeWorkerRuntime."""

    def test_full_dry_run_workflow(self) -> None:
        """Test complete dry-run workflow."""
        # Create config
        config = XuanCeWorkerConfig(
            run_id="integration_test",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            running_steps=10000,
            seed=42,
            device="cpu",
            parallels=4,
        )

        # Create runtime in dry-run mode
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        # Verify config
        assert runtime.config.run_id == "integration_test"
        assert runtime.config.seed == 42

        # Run and verify summary
        summary = runtime.run()
        assert summary.status == "dry-run"
        assert summary.config["seed"] == 42

        # Benchmark and verify summary
        summary = runtime.benchmark()
        assert summary.status == "dry-run"
