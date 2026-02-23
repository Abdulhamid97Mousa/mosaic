"""Tests for balrog-worker CLI."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from balrog_worker.cli import create_parser, build_config_from_args, main


class TestCLIParser(unittest.TestCase):
    """Test CLI argument parsing."""

    def setUp(self) -> None:
        self.parser = create_parser()

    def test_parse_minimal_args(self) -> None:
        """Should parse minimal required arguments."""
        args = self.parser.parse_args(["--run-id", "test123"])

        self.assertEqual(args.run_id, "test123")
        self.assertEqual(args.env_name, "babyai")
        self.assertEqual(args.client_name, "openai")

    def test_parse_full_args(self) -> None:
        """Should parse all arguments."""
        args = self.parser.parse_args([
            "--run-id", "full_test",
            "--env", "minihack",
            "--task", "MiniHack-Room-5x5-v0",
            "--client", "anthropic",
            "--model", "claude-3-5-sonnet-20241022",
            "--agent-type", "cot",
            "--num-episodes", "10",
            "--max-steps", "200",
            "--temperature", "0.5",
            "--seed", "42",
            "--telemetry-dir", "/tmp/telemetry",
            "--verbose",
        ])

        self.assertEqual(args.run_id, "full_test")
        self.assertEqual(args.env_name, "minihack")
        self.assertEqual(args.task, "MiniHack-Room-5x5-v0")
        self.assertEqual(args.client_name, "anthropic")
        self.assertEqual(args.model_id, "claude-3-5-sonnet-20241022")
        self.assertEqual(args.agent_type, "cot")
        self.assertEqual(args.num_episodes, 10)
        self.assertEqual(args.max_steps, 200)
        self.assertEqual(args.temperature, 0.5)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.telemetry_dir, "/tmp/telemetry")
        self.assertTrue(args.verbose)

    def test_parse_config_file(self) -> None:
        """Should parse --config argument."""
        args = self.parser.parse_args(["--config", "/path/to/config.json"])

        self.assertEqual(args.config, "/path/to/config.json")

    def test_parse_no_jsonl(self) -> None:
        """Should parse --no-jsonl flag."""
        args = self.parser.parse_args(["--run-id", "test", "--no-jsonl"])

        self.assertTrue(args.no_jsonl)

    def test_env_choices(self) -> None:
        """Should only accept valid environment choices."""
        # Valid
        args = self.parser.parse_args(["--run-id", "t", "--env", "babyai"])
        self.assertEqual(args.env_name, "babyai")

        args = self.parser.parse_args(["--run-id", "t", "--env", "minihack"])
        self.assertEqual(args.env_name, "minihack")

        args = self.parser.parse_args(["--run-id", "t", "--env", "crafter"])
        self.assertEqual(args.env_name, "crafter")

    def test_client_choices(self) -> None:
        """Should only accept valid client choices."""
        for client in ["openai", "anthropic", "google", "vllm"]:
            args = self.parser.parse_args(["--run-id", "t", "--client", client])
            self.assertEqual(args.client_name, client)

    def test_agent_type_choices(self) -> None:
        """Should only accept valid agent type choices."""
        for agent in ["naive", "cot", "robust_naive", "robust_cot", "few_shot", "dummy"]:
            args = self.parser.parse_args(["--run-id", "t", "--agent-type", agent])
            self.assertEqual(args.agent_type, agent)


class TestBuildConfigFromArgs(unittest.TestCase):
    """Test building config from parsed arguments."""

    def setUp(self) -> None:
        self.parser = create_parser()

    def test_build_from_cli_args(self) -> None:
        """Should build config from CLI arguments."""
        args = self.parser.parse_args([
            "--run-id", "cli_test",
            "--env", "crafter",
            "--num-episodes", "3",
        ])

        config = build_config_from_args(args)

        self.assertEqual(config.run_id, "cli_test")
        self.assertEqual(config.env_name, "crafter")
        self.assertEqual(config.num_episodes, 3)

    def test_build_from_config_file(self) -> None:
        """Should build config from JSON file."""
        config_data = {
            "run_id": "file_config_test",
            "env_name": "minihack",
            "task": "MiniHack-Corridor-R3-v0",
            "num_episodes": 7,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            args = self.parser.parse_args(["--config", temp_path])
            config = build_config_from_args(args)

            self.assertEqual(config.run_id, "file_config_test")
            self.assertEqual(config.env_name, "minihack")
            self.assertEqual(config.num_episodes, 7)
        finally:
            Path(temp_path).unlink()

    def test_missing_run_id(self) -> None:
        """Should raise error when run_id is missing without config file."""
        args = self.parser.parse_args([])

        with self.assertRaises(ValueError) as ctx:
            build_config_from_args(args)
        self.assertIn("--run-id is required", str(ctx.exception))


def _has_runtime_deps() -> bool:
    """Check if runtime dependencies are available."""
    try:
        from omegaconf import OmegaConf  # noqa: F401
        return True
    except ImportError:
        return False


class TestMainFunction(unittest.TestCase):
    """Test main entry point."""

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_main_success(self) -> None:
        """Main should return 0 on success."""
        # Import runtime module first so we can patch it
        import balrog_worker.runtime as runtime_module

        mock_runtime = MagicMock()
        with patch.object(runtime_module, "BarlogWorkerRuntime", return_value=mock_runtime):
            result = main(["--run-id", "test_main"])

        self.assertEqual(result, 0)
        mock_runtime.run.assert_called_once()

    @unittest.skipUnless(_has_runtime_deps(), "Requires omegaconf")
    def test_main_runtime_error(self) -> None:
        """Main should return 1 on runtime error."""
        import balrog_worker.runtime as runtime_module

        mock_runtime = MagicMock()
        mock_runtime.run.side_effect = RuntimeError("Test error")
        with patch.object(runtime_module, "BarlogWorkerRuntime", return_value=mock_runtime):
            result = main(["--run-id", "test_error"])

        self.assertEqual(result, 1)

    def test_main_config_error(self) -> None:
        """Main should return 1 on config error."""
        # Missing required run_id
        result = main([])

        self.assertEqual(result, 1)

    def test_main_missing_config_file(self) -> None:
        """Main should return 1 when config file not found."""
        result = main(["--config", "/nonexistent/config.json"])

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
