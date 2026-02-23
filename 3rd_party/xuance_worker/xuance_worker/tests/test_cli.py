"""Tests for XuanCe Worker CLI."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xuance_worker.cli import main, parse_args


def _extract_last_json(text: str) -> dict:
    """Extract the last JSON object from stdout that may contain lifecycle events.

    The lifecycle emitter writes single-line JSON events before the pretty-printed
    config JSON.  We try parsing from the end of stdout backwards until we find
    a valid JSON object.
    """
    # Try parsing the entire output first (works if only one JSON object)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Multiple JSON objects: split by lines, find the last valid one.
    # The config is pretty-printed (multi-line), so collect lines from end.
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        candidate = "\n".join(lines[i:])
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No valid JSON found in output:\n{text[:500]}")


class TestParseArgs:
    """Test suite for CLI argument parsing."""

    def test_default_arguments(self) -> None:
        """Test default argument values."""
        args = parse_args([])

        assert args.config is None
        assert args.method == "ppo"
        assert args.env == "classic_control"
        assert args.env_id == "CartPole-v1"
        assert args.dl_toolbox == "torch"  # default backend
        assert args.running_steps == 1_000_000
        assert args.seed is None
        assert args.device == "cpu"
        assert args.parallels == 8
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

    def test_env_arguments(self) -> None:
        """Test --env and --env-id arguments."""
        args = parse_args(["--env", "atari", "--env-id", "Pong-v5"])

        assert args.env == "atari"
        assert args.env_id == "Pong-v5"

    def test_running_steps(self) -> None:
        """Test --running-steps argument."""
        args = parse_args(["--running-steps", "50000"])
        assert args.running_steps == 50000

    def test_seed_argument(self) -> None:
        """Test --seed argument."""
        args = parse_args(["--seed", "42"])
        assert args.seed == 42

    def test_device_argument(self) -> None:
        """Test --device argument."""
        args = parse_args(["--device", "cuda:0"])
        assert args.device == "cuda:0"

        args = parse_args(["--device", "cuda:1"])
        assert args.device == "cuda:1"

    def test_parallels_argument(self) -> None:
        """Test --parallels argument."""
        args = parse_args(["--parallels", "16"])
        assert args.parallels == 16

    def test_dl_toolbox_argument(self) -> None:
        """Test --dl-toolbox argument."""
        args = parse_args(["--dl-toolbox", "tensorflow"])
        assert args.dl_toolbox == "tensorflow"

        args = parse_args(["--dl-toolbox", "mindspore"])
        assert args.dl_toolbox == "mindspore"

        args = parse_args(["--dl-toolbox", "torch"])
        assert args.dl_toolbox == "torch"

    def test_test_flag(self) -> None:
        """Test --test flag."""
        args = parse_args(["--test"])
        assert args.test is True

    def test_dry_run_flag(self) -> None:
        """Test --dry-run flag."""
        args = parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_run_id_argument(self) -> None:
        """Test --run-id argument."""
        args = parse_args(["--run-id", "my_custom_run"])
        assert args.run_id == "my_custom_run"

    def test_config_path_argument(self) -> None:
        """Test --config-path argument."""
        args = parse_args(["--config-path", "/custom/config.yaml"])
        assert args.config_path == "/custom/config.yaml"

    def test_config_file_argument(self) -> None:
        """Test --config argument for JSON config file."""
        args = parse_args(["--config", "/path/to/config.json"])
        assert args.config == Path("/path/to/config.json")

    def test_log_level_argument(self) -> None:
        """Test --log-level argument."""
        args = parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

        args = parse_args(["--log-level", "WARNING"])
        assert args.log_level == "WARNING"

    def test_combined_arguments(self) -> None:
        """Test multiple arguments combined."""
        args = parse_args([
            "--method", "sac",
            "--env", "mujoco",
            "--env-id", "Hopper-v4",
            "--running-steps", "100000",
            "--seed", "123",
            "--device", "cuda:0",
            "--parallels", "4",
            "--run-id", "test_combined",
            "--log-level", "DEBUG",
        ])

        assert args.method == "sac"
        assert args.env == "mujoco"
        assert args.env_id == "Hopper-v4"
        assert args.running_steps == 100000
        assert args.seed == 123
        assert args.device == "cuda:0"
        assert args.parallels == 4
        assert args.run_id == "test_combined"
        assert args.log_level == "DEBUG"


class TestMainFunction:
    """Test suite for main() entry point."""

    def test_dry_run_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test dry-run mode prints config and exits 0."""
        exit_code = main([
            "--method", "ppo",
            "--env", "classic_control",
            "--env-id", "CartPole-v1",
            "--dry-run",
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        # Lifecycle emitter may write JSON events before the config line;
        # the config is the last valid JSON object on stdout.
        output = _extract_last_json(captured.out)

        assert output["method"] == "ppo"
        assert output["env"] == "classic_control"
        assert output["env_id"] == "CartPole-v1"

    def test_dry_run_with_seed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test dry-run with seed argument."""
        exit_code = main([
            "--method", "dqn",
            "--seed", "42",
            "--dry-run",
        ])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = _extract_last_json(captured.out)

        assert output["method"] == "dqn"
        assert output["seed"] == 42

    def test_config_file_not_found(self) -> None:
        """Test error when config file doesn't exist."""
        exit_code = main(["--config", "/nonexistent/config.json"])

        assert exit_code == 1

    def test_config_file_load(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test loading config from JSON file."""
        config_data = {
            "run_id": "file_config_test",
            "method": "td3",
            "env": "box2d",
            "env_id": "LunarLander-v2",
            "running_steps": 75000,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            exit_code = main(["--config", temp_path, "--dry-run"])

            assert exit_code == 0

            captured = capsys.readouterr()
            output = _extract_last_json(captured.out)

            assert output["run_id"] == "file_config_test"
            assert output["method"] == "td3"
            assert output["env"] == "box2d"
            assert output["env_id"] == "LunarLander-v2"
            assert output["running_steps"] == 75000
        finally:
            Path(temp_path).unlink()

    def test_config_file_invalid_json(self) -> None:
        """Test error on invalid JSON config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            exit_code = main(["--config", temp_path])
            assert exit_code == 1
        finally:
            Path(temp_path).unlink()

    @patch("xuance_worker.cli.XuanCeWorkerRuntime")
    def test_run_mode(self, mock_runtime_class: MagicMock) -> None:
        """Test normal run mode (non-dry-run)."""
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = MagicMock(
            status="completed",
            runner_type="RunnerDRL",
        )
        mock_runtime_class.return_value = mock_runtime

        exit_code = main([
            "--method", "ppo",
            "--running-steps", "1000",
        ])

        assert exit_code == 0
        mock_runtime.run.assert_called_once()

    @patch("xuance_worker.cli.XuanCeWorkerRuntime")
    def test_runtime_error(self, mock_runtime_class: MagicMock) -> None:
        """Test handling of RuntimeError from runtime."""
        mock_runtime = MagicMock()
        mock_runtime.run.side_effect = RuntimeError("XuanCe not installed")
        mock_runtime_class.return_value = mock_runtime

        exit_code = main(["--method", "ppo"])

        assert exit_code == 1

    @patch("xuance_worker.cli.XuanCeWorkerRuntime")
    def test_generic_exception(self, mock_runtime_class: MagicMock) -> None:
        """Test handling of generic exceptions."""
        mock_runtime = MagicMock()
        mock_runtime.run.side_effect = ValueError("Unexpected error")
        mock_runtime_class.return_value = mock_runtime

        exit_code = main(["--method", "ppo"])

        assert exit_code == 1


class TestMainFunctionRunId:
    """Test run_id generation in main()."""

    def test_auto_generated_run_id(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that run_id is auto-generated when not provided."""
        exit_code = main(["--dry-run"])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = _extract_last_json(captured.out)

        # Auto-generated run_id should be 8 characters (UUID prefix)
        assert len(output["run_id"]) == 8
        assert output["run_id"].isalnum()

    def test_custom_run_id_preserved(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that custom run_id is preserved."""
        exit_code = main(["--run-id", "my_custom_run", "--dry-run"])

        assert exit_code == 0

        captured = capsys.readouterr()
        output = _extract_last_json(captured.out)

        assert output["run_id"] == "my_custom_run"
