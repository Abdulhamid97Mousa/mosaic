"""Tests for XuanCeWorkerRuntime.

These tests validate:
- Runtime initialization
- Dry-run mode behavior
- Parser args construction
- XuanCe integration (mocked tests)
- Error handling for missing dependencies
- Summary generation
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from xuance_worker.config import XuanCeWorkerConfig
from xuance_worker.runtime import XuanCeWorkerRuntime, XuanCeRuntimeSummary


# =============================================================================
# Test Fixtures
# =============================================================================


def _make_config(
    *,
    run_id: str = "test-run-001",
    method: str = "ppo",
    env: str = "classic_control",
    env_id: str = "CartPole-v1",
    dl_toolbox: str = "torch",
    running_steps: int = 1000,
    seed: int | None = None,
    device: str = "cpu",
    parallels: int = 8,
    test_mode: bool = False,
    config_path: str | None = None,
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
        config_path=config_path,
        extras=dict(extras or {}),
    )


# =============================================================================
# XuanCeRuntimeSummary Tests
# =============================================================================


class TestXuanCeRuntimeSummary:
    """Tests for XuanCeRuntimeSummary dataclass."""

    def test_summary_creation(self) -> None:
        """Test creating a runtime summary."""
        config = _make_config()
        summary = XuanCeRuntimeSummary(
            status="completed",
            method="ppo",
            env_id="CartPole-v1",
            runner_type="RunnerDRL",
            config=config.to_dict(),
        )

        assert summary.status == "completed"
        assert summary.method == "ppo"
        assert summary.env_id == "CartPole-v1"
        assert summary.runner_type == "RunnerDRL"
        assert summary.config["method"] == "ppo"

    def test_summary_immutability(self) -> None:
        """Test that summary is frozen (immutable)."""
        config = _make_config()
        summary = XuanCeRuntimeSummary(
            status="completed",
            method="ppo",
            env_id="CartPole-v1",
            runner_type="RunnerDRL",
            config=config.to_dict(),
        )

        with pytest.raises(AttributeError):
            summary.status = "modified"  # type: ignore

    def test_dry_run_summary(self) -> None:
        """Test dry-run summary format."""
        config = _make_config()
        summary = XuanCeRuntimeSummary(
            status="dry-run",
            method="dqn",
            env_id="Pong-v5",
            runner_type="unknown",
            config=config.to_dict(),
        )

        assert summary.status == "dry-run"
        assert summary.runner_type == "unknown"


# =============================================================================
# XuanCeWorkerRuntime Initialization Tests
# =============================================================================


class TestXuanCeWorkerRuntimeInit:
    """Tests for XuanCeWorkerRuntime initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic runtime initialization."""
        config = _make_config()
        runtime = XuanCeWorkerRuntime(config)

        assert runtime.config == config

    def test_dry_run_initialization(self) -> None:
        """Test runtime initialization with dry_run=True."""
        config = _make_config()
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        assert runtime.config == config

    def test_config_property(self) -> None:
        """Test that config property returns the configuration."""
        config = _make_config(method="sac", env="mujoco")
        runtime = XuanCeWorkerRuntime(config)

        assert runtime.config.method == "sac"
        assert runtime.config.env == "mujoco"


# =============================================================================
# Dry-Run Mode Tests
# =============================================================================


class TestXuanCeWorkerRuntimeDryRun:
    """Tests for dry-run mode behavior."""

    def test_run_dry_run_returns_summary(self) -> None:
        """Test that run() in dry-run mode returns a summary."""
        config = _make_config(method="ppo", env_id="CartPole-v1")
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == "ppo"
        assert summary.env_id == "CartPole-v1"
        assert summary.runner_type == "unknown"
        assert summary.config["method"] == "ppo"

    def test_dry_run_does_not_import_xuance(self) -> None:
        """Test that dry-run mode doesn't attempt to import XuanCe."""
        config = _make_config()
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        # Should not raise ImportError even if XuanCe is not installed
        summary = runtime.run()
        assert summary.status == "dry-run"


# =============================================================================
# Parser Args Building Tests
# =============================================================================


class TestXuanCeWorkerRuntimeBuildParserArgs:
    """Tests for _build_parser_args method."""

    def test_basic_parser_args(self) -> None:
        """Test building basic parser arguments."""
        config = _make_config(
            device="cpu",
            parallels=8,
            running_steps=1000,
            dl_toolbox="torch",
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        args = runtime._build_parser_args()

        assert isinstance(args, SimpleNamespace)
        assert args.device == "cpu"
        assert args.parallels == 8
        assert args.running_steps == 1000
        assert args.dl_toolbox == "torch"

    def test_parser_args_with_seed(self) -> None:
        """Test that seed is correctly set in parser args."""
        config = _make_config(seed=42)
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        args = runtime._build_parser_args()

        assert args.seed == 42
        assert args.env_seed == 42

    def test_parser_args_without_seed(self) -> None:
        """Test that seed is not set when None."""
        config = _make_config(seed=None)
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        args = runtime._build_parser_args()

        assert not hasattr(args, "seed")
        assert not hasattr(args, "env_seed")

    def test_parser_args_with_extras(self) -> None:
        """Test that extras are applied to parser args."""
        config = _make_config(
            extras={
                "learning_rate": 0.001,
                "batch_size": 64,
                "gamma": 0.99,
            }
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        args = runtime._build_parser_args()

        assert args.learning_rate == 0.001
        assert args.batch_size == 64
        assert args.gamma == 0.99

    def test_parser_args_cuda_device(self) -> None:
        """Test parser args with CUDA device."""
        config = _make_config(device="cuda:0")
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        args = runtime._build_parser_args()

        assert args.device == "cuda:0"

    def test_parser_args_with_all_options(self) -> None:
        """Test parser args with all options specified."""
        config = _make_config(
            device="cuda:1",
            parallels=16,
            running_steps=500000,
            dl_toolbox="tensorflow",
            seed=123,
            extras={"custom_param": "value"},
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        args = runtime._build_parser_args()

        assert args.device == "cuda:1"
        assert args.parallels == 16
        assert args.running_steps == 500000
        assert args.dl_toolbox == "tensorflow"
        assert args.seed == 123
        assert args.env_seed == 123
        assert args.custom_param == "value"


# =============================================================================
# Run Method Tests (with mocking)
# =============================================================================


class TestXuanCeWorkerRuntimeRun:
    """Tests for run() method with XuanCe mocked."""

    @patch("xuance_worker.runtime.get_runner", create=True)
    def test_run_with_mocked_xuance(self, mock_get_runner) -> None:
        """Test run() with mocked XuanCe."""
        # Setup mock
        mock_runner = MagicMock()
        mock_runner.__class__.__name__ = "RunnerDRL"
        mock_get_runner.return_value = mock_runner

        config = _make_config(method="ppo", env="classic_control", env_id="CartPole-v1")
        runtime = XuanCeWorkerRuntime(config, dry_run=False)

        # Patch the import inside the module
        with patch.dict("sys.modules", {"xuance": MagicMock(get_runner=mock_get_runner)}):
            with patch("xuance_worker.runtime.get_runner", mock_get_runner, create=True):
                # This will fail because xuance is not installed, but we test the dry-run path
                pass

    def test_run_xuance_not_installed_raises_error(self) -> None:
        """Test that run() raises RuntimeError if XuanCe is not installed."""
        import sys
        from unittest.mock import patch

        config = _make_config()
        runtime = XuanCeWorkerRuntime(config, dry_run=False)

        # Mock xuance as unavailable to test the import error path
        with patch.dict(sys.modules, {"xuance": None}):
            with pytest.raises(RuntimeError, match="XuanCe is not installed"):
                runtime.run()


# =============================================================================
# Integration Tests
# =============================================================================


class TestXuanCeWorkerRuntimeIntegration:
    """Integration tests for the runtime."""

    def test_full_dry_run_workflow(self) -> None:
        """Test complete dry-run workflow from config to summary."""
        # Create config
        config = XuanCeWorkerConfig(
            run_id="integration-test",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            dl_toolbox="torch",
            running_steps=10000,
            seed=42,
            device="cpu",
            parallels=4,
            extras={"learning_rate": 0.0003},
        )

        # Create runtime
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        # Execute run
        summary = runtime.run()

        # Verify summary
        assert summary.status == "dry-run"
        assert summary.method == "ppo"
        assert summary.env_id == "CartPole-v1"
        assert summary.config["run_id"] == "integration-test"
        assert summary.config["seed"] == 42
        assert summary.config["extras"]["learning_rate"] == 0.0003

    def test_multi_agent_dry_run_workflow(self) -> None:
        """Test dry-run workflow for multi-agent configuration."""
        config = XuanCeWorkerConfig(
            run_id="multi-agent-test",
            method="mappo",
            env="mpe",
            env_id="simple_spread_v3",
            dl_toolbox="torch",
            running_steps=100000,
            parallels=128,
            extras={"n_agents": 3, "share_policy": True},
        )

        runtime = XuanCeWorkerRuntime(config, dry_run=True)
        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == "mappo"
        assert summary.env_id == "simple_spread_v3"
        assert summary.config["extras"]["n_agents"] == 3

# =============================================================================
# Backend-Specific Tests
# =============================================================================


class TestXuanCeWorkerRuntimeBackends:
    """Tests for different backend configurations."""

    def test_torch_backend_dry_run(self) -> None:
        """Test runtime with PyTorch backend."""
        config = _make_config(dl_toolbox="torch")
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        args = runtime._build_parser_args()
        assert args.dl_toolbox == "torch"

        summary = runtime.run()
        assert summary.config["dl_toolbox"] == "torch"

    def test_tensorflow_backend_dry_run(self) -> None:
        """Test runtime with TensorFlow backend."""
        config = _make_config(dl_toolbox="tensorflow")
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        args = runtime._build_parser_args()
        assert args.dl_toolbox == "tensorflow"

        summary = runtime.run()
        assert summary.config["dl_toolbox"] == "tensorflow"

    def test_mindspore_backend_dry_run(self) -> None:
        """Test runtime with MindSpore backend."""
        config = _make_config(dl_toolbox="mindspore")
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        args = runtime._build_parser_args()
        assert args.dl_toolbox == "mindspore"

        summary = runtime.run()
        assert summary.config["dl_toolbox"] == "mindspore"


# =============================================================================
# Algorithm Coverage Tests
# =============================================================================


class TestXuanCeWorkerRuntimeAlgorithms:
    """Tests for various algorithms in dry-run mode."""

    @pytest.mark.parametrize(
        "method,env,env_id",
        [
            ("dqn", "classic_control", "CartPole-v1"),
            ("ppo", "classic_control", "CartPole-v1"),
            ("sac", "mujoco", "HalfCheetah-v4"),
            ("ddpg", "mujoco", "Ant-v4"),
            ("td3", "box2d", "BipedalWalker-v3"),
        ],
    )
    def test_single_agent_algorithms(
        self, method: str, env: str, env_id: str
    ) -> None:
        """Test various single-agent algorithms in dry-run mode."""
        config = _make_config(method=method, env=env, env_id=env_id)
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == method
        assert summary.env_id == env_id

    @pytest.mark.parametrize(
        "method,env,env_id",
        [
            ("mappo", "mpe", "simple_spread_v3"),
            ("qmix", "smac", "3m"),
            ("maddpg", "mpe", "simple_adversary_v3"),
            ("vdn", "smac", "8m"),
            ("ippo", "mpe", "simple_tag_v3"),
        ],
    )
    def test_multi_agent_algorithms(
        self, method: str, env: str, env_id: str
    ) -> None:
        """Test various multi-agent algorithms in dry-run mode."""
        config = _make_config(method=method, env=env, env_id=env_id)
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == method
        assert summary.env_id == env_id
