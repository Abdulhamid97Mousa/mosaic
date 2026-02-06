"""Tests for XuanCe worker CLI.

These tests validate:
- Argument parsing with various combinations
- Config file loading
- Dry-run functionality
- Error handling (missing files, invalid JSON)
- Exit code generation
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from xuance_worker.cli import parse_args, main


def _parse_config_from_output(stdout: str) -> dict:
    """Parse config output from CLI dry-run.

    The CLI may emit lifecycle events (JSON lines) followed by the main config.
    This function finds the config JSON by looking for an object with 'method' key.

    The output can be:
    1. Single-line JSON (lifecycle events)
    2. Multi-line pretty-printed JSON (config output)

    Args:
        stdout: Captured standard output from the CLI.

    Returns:
        Parsed config dictionary.

    Raises:
        ValueError: If no valid config could be parsed.
    """
    lines = stdout.strip().split('\n')

    # First, try to find multi-line pretty-printed JSON at the end
    # Look for a line starting with '{' and collect until we find matching '}'
    json_buffer = []
    brace_count = 0
    in_json = False

    for line in lines:
        stripped = line.strip()
        if not in_json and stripped.startswith('{'):
            # Check if it's a single-line JSON first
            try:
                output = json.loads(stripped)
                if isinstance(output, dict) and "method" in output:
                    # Found a single-line config, but keep looking for pretty-printed one
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
                # Complete JSON object found
                try:
                    output = json.loads('\n'.join(json_buffer))
                    if isinstance(output, dict) and "method" in output:
                        return output
                except json.JSONDecodeError:
                    pass
                # Reset for next potential JSON
                json_buffer = []
                in_json = False

    # If we collected something but didn't parse it, try parsing it
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
# Argument Parsing Tests
# =============================================================================


class TestParseArgs:
    """Tests for parse_args function."""

    def test_default_values(self) -> None:
        """Test that default values are correctly set."""
        args = parse_args([])

        assert args.config is None
        assert args.method == "ppo"
        assert args.env == "classic_control"
        assert args.env_id == "CartPole-v1"
        assert args.running_steps == 1_000_000
        assert args.seed is None
        assert args.device == "cpu"
        assert args.parallels == 8
        assert args.dl_toolbox == "torch"
        assert args.test is False
        assert args.dry_run is False
        assert args.run_id is None
        assert args.config_path is None
        assert args.log_level == "INFO"

    def test_method_argument(self) -> None:
        """Test --method argument."""
        args = parse_args(["--method", "dqn"])
        assert args.method == "dqn"

        args = parse_args(["--method", "mappo"])
        assert args.method == "mappo"

    def test_env_argument(self) -> None:
        """Test --env argument."""
        args = parse_args(["--env", "atari"])
        assert args.env == "atari"

        args = parse_args(["--env", "mpe"])
        assert args.env == "mpe"

    def test_env_id_argument(self) -> None:
        """Test --env-id argument."""
        args = parse_args(["--env-id", "Pong-v5"])
        assert args.env_id == "Pong-v5"

        args = parse_args(["--env-id", "simple_spread_v3"])
        assert args.env_id == "simple_spread_v3"

    def test_running_steps_argument(self) -> None:
        """Test --running-steps argument."""
        args = parse_args(["--running-steps", "500000"])
        assert args.running_steps == 500_000

        args = parse_args(["--running-steps", "100"])
        assert args.running_steps == 100

    def test_seed_argument(self) -> None:
        """Test --seed argument."""
        args = parse_args(["--seed", "42"])
        assert args.seed == 42

        args = parse_args(["--seed", "0"])
        assert args.seed == 0

    def test_device_argument(self) -> None:
        """Test --device argument with valid choices."""
        for device in ["cpu", "cuda", "cuda:0", "cuda:1"]:
            args = parse_args(["--device", device])
            assert args.device == device

    def test_parallels_argument(self) -> None:
        """Test --parallels argument."""
        args = parse_args(["--parallels", "16"])
        assert args.parallels == 16

        args = parse_args(["--parallels", "1"])
        assert args.parallels == 1

    def test_dl_toolbox_argument(self) -> None:
        """Test --dl-toolbox argument with valid choices."""
        for backend in ["torch", "tensorflow", "mindspore"]:
            args = parse_args(["--dl-toolbox", backend])
            assert args.dl_toolbox == backend

    def test_test_flag(self) -> None:
        """Test --test flag."""
        args = parse_args(["--test"])
        assert args.test is True

        args = parse_args([])
        assert args.test is False

    def test_dry_run_flag(self) -> None:
        """Test --dry-run flag."""
        args = parse_args(["--dry-run"])
        assert args.dry_run is True

        args = parse_args([])
        assert args.dry_run is False

    def test_run_id_argument(self) -> None:
        """Test --run-id argument."""
        args = parse_args(["--run-id", "custom-run-123"])
        assert args.run_id == "custom-run-123"

    def test_config_path_argument(self) -> None:
        """Test --config-path argument."""
        args = parse_args(["--config-path", "/custom/config.yaml"])
        assert args.config_path == "/custom/config.yaml"

    def test_log_level_argument(self) -> None:
        """Test --log-level argument with valid choices."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            args = parse_args(["--log-level", level])
            assert args.log_level == level

    def test_config_file_argument(self, tmp_path: Path) -> None:
        """Test --config argument with file path."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        args = parse_args(["--config", str(config_file)])
        assert args.config == config_file

    def test_combined_arguments(self) -> None:
        """Test multiple arguments combined."""
        args = parse_args([
            "--method", "sac",
            "--env", "mujoco",
            "--env-id", "Ant-v4",
            "--running-steps", "2000000",
            "--seed", "123",
            "--device", "cuda:0",
            "--parallels", "4",
            "--dl-toolbox", "torch",
            "--run-id", "combined-test",
            "--log-level", "DEBUG",
        ])

        assert args.method == "sac"
        assert args.env == "mujoco"
        assert args.env_id == "Ant-v4"
        assert args.running_steps == 2_000_000
        assert args.seed == 123
        assert args.device == "cuda:0"
        assert args.parallels == 4
        assert args.dl_toolbox == "torch"
        assert args.run_id == "combined-test"
        assert args.log_level == "DEBUG"


# =============================================================================
# Main Function Tests
# =============================================================================


class TestMainFunction:
    """Tests for main function."""

    def test_dry_run_mode(self, capsys) -> None:
        """Test that dry-run mode prints config and exits 0."""
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

    def test_dry_run_with_all_parameters(self, capsys) -> None:
        """Test dry-run with all parameters specified."""
        exit_code = main([
            "--method", "dqn",
            "--env", "atari",
            "--env-id", "Pong-v5",
            "--running-steps", "100000",
            "--seed", "42",
            "--device", "cuda:0",
            "--parallels", "16",
            "--dl-toolbox", "torch",
            "--run-id", "test-run",
            "--dry-run",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)
        assert output["method"] == "dqn"
        assert output["env"] == "atari"
        assert output["env_id"] == "Pong-v5"
        assert output["running_steps"] == 100_000
        assert output["seed"] == 42
        assert output["device"] == "cuda:0"
        assert output["parallels"] == 16
        assert output["dl_toolbox"] == "torch"
        assert output["run_id"] == "test-run"

    def test_config_file_mode(self, tmp_path: Path, capsys) -> None:
        """Test loading config from JSON file."""
        config_data = {
            "run_id": "json-config-test",
            "method": "td3",
            "env": "box2d",
            "env_id": "BipedalWalker-v3",
            "running_steps": 50000,
            "seed": 99,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        exit_code = main(["--config", str(config_file), "--dry-run"])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)
        assert output["run_id"] == "json-config-test"
        assert output["method"] == "td3"
        assert output["env"] == "box2d"

    def test_config_file_not_found(self, tmp_path: Path) -> None:
        """Test error when config file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.json"

        exit_code = main(["--config", str(nonexistent)])

        assert exit_code == 1

    def test_config_file_invalid_json(self, tmp_path: Path) -> None:
        """Test error when config file contains invalid JSON."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ not valid json }")

        exit_code = main(["--config", str(invalid_file)])

        assert exit_code == 1

    def test_auto_generated_run_id(self, capsys) -> None:
        """Test that run_id is auto-generated when not provided."""
        exit_code = main([
            "--method", "ppo",
            "--dry-run",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)

        assert output["run_id"] != ""
        assert len(output["run_id"]) == 8  # UUID[:8]

    def test_custom_run_id_preserved(self, capsys) -> None:
        """Test that custom run_id is preserved."""
        exit_code = main([
            "--method", "ppo",
            "--run-id", "my-custom-id",
            "--dry-run",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)

        assert output["run_id"] == "my-custom-id"

    @patch("xuance_worker.cli.XuanCeWorkerRuntime")
    def test_runtime_error_handling(self, mock_runtime_cls) -> None:
        """Test that RuntimeError is handled properly."""
        mock_runtime = MagicMock()
        mock_runtime.run.side_effect = RuntimeError("XuanCe not installed")
        mock_runtime_cls.return_value = mock_runtime

        exit_code = main([
            "--method", "ppo",
            "--env", "classic_control",
            "--env-id", "CartPole-v1",
        ])

        assert exit_code == 1

    @patch("xuance_worker.cli.XuanCeWorkerRuntime")
    def test_generic_exception_handling(self, mock_runtime_cls) -> None:
        """Test that generic exceptions are handled properly."""
        mock_runtime = MagicMock()
        mock_runtime.run.side_effect = ValueError("Some error")
        mock_runtime_cls.return_value = mock_runtime

        exit_code = main([
            "--method", "ppo",
            "--env", "classic_control",
            "--env-id", "CartPole-v1",
        ])

        assert exit_code == 1

    @patch("xuance_worker.cli.XuanCeWorkerRuntime")
    def test_normal_run_mode_called(self, mock_runtime_cls) -> None:
        """Test that normal mode calls runtime.run()."""
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = MagicMock(
            status="completed", runner_type="RunnerDRL"
        )
        mock_runtime_cls.return_value = mock_runtime

        exit_code = main([
            "--method", "ppo",
        ])

        assert exit_code == 0
        mock_runtime.run.assert_called_once()


# =============================================================================
# Backend Selection Tests
# =============================================================================


class TestBackendSelection:
    """Tests for deep learning backend selection."""

    def test_torch_backend_dry_run(self, capsys) -> None:
        """Test PyTorch backend selection in dry-run."""
        exit_code = main([
            "--method", "ppo",
            "--dl-toolbox", "torch",
            "--dry-run",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)
        assert output["dl_toolbox"] == "torch"

    def test_tensorflow_backend_dry_run(self, capsys) -> None:
        """Test TensorFlow backend selection in dry-run."""
        exit_code = main([
            "--method", "ppo",
            "--dl-toolbox", "tensorflow",
            "--dry-run",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)
        assert output["dl_toolbox"] == "tensorflow"

    def test_mindspore_backend_dry_run(self, capsys) -> None:
        """Test MindSpore backend selection in dry-run."""
        exit_code = main([
            "--method", "ppo",
            "--dl-toolbox", "mindspore",
            "--dry-run",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)
        assert output["dl_toolbox"] == "mindspore"


# =============================================================================
# Test Mode Tests
# =============================================================================


class TestTestMode:
    """Tests for test/evaluation mode."""

    def test_test_mode_flag_in_config(self, capsys) -> None:
        """Test that --test flag sets test_mode in config."""
        exit_code = main([
            "--method", "ppo",
            "--test",
            "--dry-run",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)
        assert output["test_mode"] is True

    def test_no_test_mode_flag(self, capsys) -> None:
        """Test that test_mode is False when --test not specified."""
        exit_code = main([
            "--method", "ppo",
            "--dry-run",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)
        assert output["test_mode"] is False


# =============================================================================
# Multi-Agent Tests
# =============================================================================


class TestMOSAICIntegration:
    """Tests for MOSAIC dispatcher integration arguments."""

    def test_grpc_flag(self) -> None:
        """Test --grpc flag."""
        args = parse_args(["--grpc"])
        assert args.grpc is True

        args = parse_args([])
        assert args.grpc is False

    def test_grpc_target_argument(self) -> None:
        """Test --grpc-target argument."""
        args = parse_args(["--grpc-target", "192.168.1.1:50055"])
        assert args.grpc_target == "192.168.1.1:50055"

        # Default value
        args = parse_args([])
        assert args.grpc_target == "127.0.0.1:50055"

    def test_worker_id_argument(self) -> None:
        """Test --worker-id argument."""
        args = parse_args(["--worker-id", "4"])
        assert args.worker_id == "4"

        args = parse_args([])
        assert args.worker_id is None

    def test_combined_mosaic_args(self) -> None:
        """Test all MOSAIC dispatcher arguments combined."""
        args = parse_args([
            "--grpc",
            "--grpc-target", "127.0.0.1:50055",
            "--worker-id", "worker-1",
            "--config", "/path/to/config.json",
        ])

        assert args.grpc is True
        assert args.grpc_target == "127.0.0.1:50055"
        assert args.worker_id == "worker-1"

    def test_dry_run_with_mosaic_args(self, capsys) -> None:
        """Test dry-run with MOSAIC dispatcher arguments."""
        exit_code = main([
            "--method", "ppo",
            "--grpc",
            "--grpc-target", "127.0.0.1:50055",
            "--worker-id", "test-worker",
            "--dry-run",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)

        assert output["method"] == "ppo"
        assert output["worker_id"] == "test-worker"


class TestMultiAgentCLI:
    """Tests for multi-agent algorithm CLI usage."""

    def test_mappo_mpe_config(self, capsys) -> None:
        """Test MAPPO with MPE environment."""
        exit_code = main([
            "--method", "mappo",
            "--env", "mpe",
            "--env-id", "simple_spread_v3",
            "--dry-run",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)

        assert output["method"] == "mappo"
        assert output["env"] == "mpe"
        assert output["env_id"] == "simple_spread_v3"

    def test_qmix_smac_config(self, capsys) -> None:
        """Test QMIX with SMAC environment."""
        exit_code = main([
            "--method", "qmix",
            "--env", "smac",
            "--env-id", "3m",
            "--dry-run",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)

        assert output["method"] == "qmix"
        assert output["env"] == "smac"
        assert output["env_id"] == "3m"

    def test_maddpg_mpe_config(self, capsys) -> None:
        """Test MADDPG with MPE environment."""
        exit_code = main([
            "--method", "maddpg",
            "--env", "mpe",
            "--env-id", "simple_adversary_v3",
            "--dry-run",
        ])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = _parse_config_from_output(captured.out)

        assert output["method"] == "maddpg"
        assert output["env"] == "mpe"
