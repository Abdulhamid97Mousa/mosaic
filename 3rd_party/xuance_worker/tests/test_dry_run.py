"""Dry-run validation tests for XuanCe worker.

These tests validate the dry-run functionality which allows
configuration validation without actually executing training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from xuance_worker.config import XuanCeWorkerConfig
from xuance_worker.runtime import XuanCeWorkerRuntime
from xuance_worker.cli import main


# =============================================================================
# Helper Functions
# =============================================================================


def _minimal_config() -> dict[str, Any]:
    """Create a minimal configuration dictionary."""
    return {
        "run_id": "dry-run-test",
        "method": "ppo",
        "env": "classic_control",
        "env_id": "CartPole-v1",
        "dl_toolbox": "torch",
        "running_steps": 1000,
        "seed": 42,
        "device": "cpu",
        "parallels": 8,
        "extras": {},
    }


def _multi_agent_config() -> dict[str, Any]:
    """Create a multi-agent configuration dictionary."""
    return {
        "run_id": "dry-run-multi-agent",
        "method": "mappo",
        "env": "mpe",
        "env_id": "simple_spread_v3",
        "dl_toolbox": "torch",
        "running_steps": 10000,
        "seed": 123,
        "device": "cpu",
        "parallels": 128,
        "extras": {
            "n_agents": 3,
            "share_policy": True,
        },
    }


def _parse_config_from_output(stdout: str) -> dict:
    """Parse config output from CLI dry-run.

    The CLI may emit lifecycle events (JSON lines) followed by the main config.
    This function finds the config JSON by looking for an object with 'method' key.

    Args:
        stdout: Captured standard output from the CLI.

    Returns:
        Parsed config dictionary.

    Raises:
        ValueError: If no valid config could be parsed.
    """
    lines = stdout.strip().split('\n')

    # First, try to find multi-line pretty-printed JSON at the end
    json_buffer: list[str] = []
    brace_count = 0
    in_json = False

    for line in lines:
        stripped = line.strip()
        if not in_json and stripped.startswith('{'):
            # Check if it's a single-line JSON first
            try:
                output = json.loads(stripped)
                if isinstance(output, dict) and "method" in output:
                    json_buffer = [stripped]
                    in_json = False
                    continue
            except json.JSONDecodeError:
                pass
            # Start of multi-line JSON
            in_json = True
            json_buffer = [line]
            brace_count = line.count('{') - line.count('}')
        elif in_json:
            json_buffer.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                try:
                    output = json.loads('\n'.join(json_buffer))
                    if isinstance(output, dict) and "method" in output:
                        return output
                except json.JSONDecodeError:
                    pass
                json_buffer = []
                in_json = False

    # If we collected something, try parsing it
    if json_buffer:
        try:
            output = json.loads('\n'.join(json_buffer))
            if isinstance(output, dict) and "method" in output:
                return output
        except json.JSONDecodeError:
            pass

    # Fallback: try each line in reverse as single-line JSON
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            output = json.loads(stripped)
            if isinstance(output, dict) and "method" in output:
                return output
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse config from output: {stdout}")


# =============================================================================
# CLI Dry-Run Tests
# =============================================================================


class TestCLIDryRun:
    """Tests for CLI dry-run functionality."""

    def test_cli_dry_run_success(self, capsys) -> None:
        """Test that CLI dry-run succeeds and outputs valid JSON."""
        exit_code = main([
            "--method", "ppo",
            "--env", "classic_control",
            "--env-id", "CartPole-v1",
            "--dry-run",
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)

        assert output["method"] == "ppo"
        assert output["env"] == "classic_control"
        assert output["env_id"] == "CartPole-v1"

    def test_cli_dry_run_with_seed(self, capsys) -> None:
        """Test dry-run with seed parameter."""
        exit_code = main([
            "--method", "dqn",
            "--seed", "42",
            "--dry-run",
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)

        assert output["seed"] == 42

    def test_cli_dry_run_with_cuda_device(self, capsys) -> None:
        """Test dry-run with CUDA device."""
        exit_code = main([
            "--method", "sac",
            "--device", "cuda:0",
            "--dry-run",
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)

        assert output["device"] == "cuda:0"

    def test_cli_dry_run_tensorflow_backend(self, capsys) -> None:
        """Test dry-run with TensorFlow backend."""
        exit_code = main([
            "--method", "ppo",
            "--dl-toolbox", "tensorflow",
            "--dry-run",
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)

        assert output["dl_toolbox"] == "tensorflow"

    def test_cli_dry_run_from_config_file(self, tmp_path: Path, capsys) -> None:
        """Test dry-run loading from config file."""
        config = _minimal_config()
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        exit_code = main([
            "--config", str(config_file),
            "--dry-run",
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)

        assert output["run_id"] == "dry-run-test"


# =============================================================================
# Runtime Dry-Run Tests
# =============================================================================


class TestRuntimeDryRun:
    """Tests for runtime dry-run functionality."""

    def test_runtime_dry_run_run_method(self) -> None:
        """Test runtime.run() in dry-run mode."""
        config = XuanCeWorkerConfig.from_dict(_minimal_config())
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == "ppo"
        assert summary.env_id == "CartPole-v1"
        assert summary.runner_type == "unknown"

    def test_runtime_dry_run_preserves_config(self) -> None:
        """Test that dry-run preserves full configuration."""
        config_dict = _minimal_config()
        config_dict["extras"] = {
            "learning_rate": 0.001,
            "gamma": 0.99,
        }
        config = XuanCeWorkerConfig.from_dict(config_dict)
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.config["extras"]["learning_rate"] == 0.001
        assert summary.config["extras"]["gamma"] == 0.99


# =============================================================================
# Multi-Agent Dry-Run Tests
# =============================================================================


class TestMultiAgentDryRun:
    """Tests for multi-agent configuration dry-run."""

    def test_mappo_dry_run(self) -> None:
        """Test MAPPO algorithm dry-run."""
        config = XuanCeWorkerConfig.from_dict(_multi_agent_config())
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == "mappo"
        assert summary.env_id == "simple_spread_v3"

    def test_qmix_dry_run(self) -> None:
        """Test QMIX algorithm dry-run."""
        config = XuanCeWorkerConfig(
            run_id="qmix-test",
            method="qmix",
            env="smac",
            env_id="3m",
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == "qmix"
        assert summary.env_id == "3m"

    def test_maddpg_dry_run(self) -> None:
        """Test MADDPG algorithm dry-run."""
        config = XuanCeWorkerConfig(
            run_id="maddpg-test",
            method="maddpg",
            env="mpe",
            env_id="simple_adversary_v3",
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == "maddpg"


# =============================================================================
# Single-Agent Algorithm Dry-Run Tests
# =============================================================================


class TestSingleAgentDryRun:
    """Tests for single-agent algorithm dry-run."""

    @pytest.mark.parametrize(
        "method,env,env_id",
        [
            ("DQN", "classic_control", "CartPole-v1"),
            ("DDQN", "classic_control", "CartPole-v1"),
            ("PPO_Clip", "classic_control", "CartPole-v1"),
            ("PPO_KL", "classic_control", "CartPole-v1"),
            ("A2C", "classic_control", "CartPole-v1"),
            ("SAC", "mujoco", "HalfCheetah-v4"),
            ("DDPG", "mujoco", "Ant-v4"),
            ("TD3", "mujoco", "Walker2d-v4"),
        ],
    )
    def test_algorithm_dry_run(
        self, method: str, env: str, env_id: str
    ) -> None:
        """Test dry-run for various single-agent algorithms."""
        config = XuanCeWorkerConfig(
            run_id=f"test-{method.lower()}",
            method=method,
            env=env,
            env_id=env_id,
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.method == method
        assert summary.env_id == env_id


# =============================================================================
# Backend Dry-Run Tests
# =============================================================================


class TestBackendDryRun:
    """Tests for different backend dry-runs."""

    def test_pytorch_backend_dry_run(self) -> None:
        """Test PyTorch backend dry-run."""
        config = XuanCeWorkerConfig(
            run_id="torch-test",
            method="PPO_Clip",
            env="classic_control",
            env_id="CartPole-v1",
            dl_toolbox="torch",
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.config["dl_toolbox"] == "torch"

    def test_tensorflow_backend_dry_run(self) -> None:
        """Test TensorFlow backend dry-run."""
        config = XuanCeWorkerConfig(
            run_id="tf-test",
            method="PPO_Clip",
            env="classic_control",
            env_id="CartPole-v1",
            dl_toolbox="tensorflow",
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.config["dl_toolbox"] == "tensorflow"

    def test_mindspore_backend_dry_run(self) -> None:
        """Test MindSpore backend dry-run."""
        config = XuanCeWorkerConfig(
            run_id="ms-test",
            method="PPO_Clip",
            env="classic_control",
            env_id="CartPole-v1",
            dl_toolbox="mindspore",
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.config["dl_toolbox"] == "mindspore"


# =============================================================================
# Configuration Validation Tests
# =============================================================================


class TestConfigValidation:
    """Tests for configuration validation during dry-run."""

    def test_dry_run_with_all_parameters(self) -> None:
        """Test dry-run with all configuration parameters."""
        config = XuanCeWorkerConfig(
            run_id="full-config-test",
            method="SAC",
            env="mujoco",
            env_id="HalfCheetah-v4",
            dl_toolbox="torch",
            running_steps=1000000,
            seed=42,
            device="cuda:0",
            parallels=8,
            test_mode=False,
            config_path="/custom/config.yaml",
            worker_id="worker-001",
            extras={
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "tau": 0.005,
            },
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.config["run_id"] == "full-config-test"
        assert summary.config["running_steps"] == 1000000
        assert summary.config["seed"] == 42
        assert summary.config["device"] == "cuda:0"
        assert summary.config["config_path"] == "/custom/config.yaml"
        assert summary.config["extras"]["learning_rate"] == 0.0003

    def test_dry_run_with_test_mode(self) -> None:
        """Test dry-run with test_mode enabled."""
        config = XuanCeWorkerConfig(
            run_id="test-mode-test",
            method="DQN",
            env="atari",
            env_id="Pong-v5",
            test_mode=True,
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.config["test_mode"] is True


# =============================================================================
# JSON Roundtrip Tests
# =============================================================================


class TestJsonRoundtrip:
    """Tests for JSON serialization in dry-run context."""

    def test_config_to_json_dry_run(self, tmp_path: Path) -> None:
        """Test config serialization and dry-run."""
        original_config = XuanCeWorkerConfig(
            run_id="json-test",
            method="PPO_Clip",
            env="classic_control",
            env_id="CartPole-v1",
            dl_toolbox="torch",
            running_steps=50000,
            seed=123,
            extras={"learning_rate": 0.0003},
        )

        # Serialize to JSON
        json_file = tmp_path / "config.json"
        json_file.write_text(original_config.to_json())

        # Load and run dry-run
        loaded_config = XuanCeWorkerConfig.from_json_file(json_file)
        runtime = XuanCeWorkerRuntime(loaded_config, dry_run=True)
        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.config["run_id"] == "json-test"
        assert summary.config["running_steps"] == 50000
        assert summary.config["seed"] == 123

    def test_dry_run_output_is_valid_json(self, capsys) -> None:
        """Test that CLI dry-run output is valid JSON."""
        exit_code = main([
            "--method", "ppo",
            "--env", "classic_control",
            "--env-id", "CartPole-v1",
            "--running-steps", "10000",
            "--seed", "42",
            "--device", "cpu",
            "--parallels", "4",
            "--dl-toolbox", "torch",
            "--run-id", "json-output-test",
            "--dry-run",
        ])

        assert exit_code == 0

        captured = capsys.readouterr()

        # Should be valid JSON
        output = _parse_config_from_output(captured.out)

        # Should contain all fields
        assert "run_id" in output
        assert "method" in output
        assert "env" in output
        assert "env_id" in output
        assert "dl_toolbox" in output
        assert "running_steps" in output
        assert "seed" in output
        assert "device" in output
        assert "parallels" in output


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDryRunEdgeCases:
    """Tests for edge cases in dry-run mode."""

    def test_dry_run_with_empty_extras(self) -> None:
        """Test dry-run with empty extras dictionary."""
        config = XuanCeWorkerConfig(
            run_id="empty-extras",
            method="DQN",
            env="classic_control",
            env_id="CartPole-v1",
            extras={},
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.config["extras"] == {}

    def test_dry_run_with_seed_zero(self) -> None:
        """Test dry-run with seed=0 (edge case)."""
        config = XuanCeWorkerConfig(
            run_id="seed-zero",
            method="PPO_Clip",
            env="classic_control",
            env_id="CartPole-v1",
            seed=0,
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.config["seed"] == 0

    def test_dry_run_with_none_seed(self) -> None:
        """Test dry-run with seed=None."""
        config = XuanCeWorkerConfig(
            run_id="none-seed",
            method="PPO_Clip",
            env="classic_control",
            env_id="CartPole-v1",
            seed=None,
        )
        runtime = XuanCeWorkerRuntime(config, dry_run=True)

        summary = runtime.run()

        assert summary.status == "dry-run"
        assert summary.config["seed"] is None
