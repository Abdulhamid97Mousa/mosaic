"""Tests for BarlogWorkerConfig."""

import json
import tempfile
import unittest
from pathlib import Path

from balrog_worker.config import (
    AGENT_TYPES,
    CLIENT_NAMES,
    ENV_NAMES,
    BarlogWorkerConfig,
)


class TestBarlogWorkerConfig(unittest.TestCase):
    """Test BarlogWorkerConfig dataclass."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = BarlogWorkerConfig(run_id="test123")

        self.assertEqual(config.run_id, "test123")
        self.assertEqual(config.env_name, "babyai")
        self.assertEqual(config.task, "BabyAI-GoToRedBall-v0")
        self.assertEqual(config.client_name, "openai")
        self.assertEqual(config.model_id, "gpt-4o-mini")
        self.assertEqual(config.agent_type, "naive")
        self.assertEqual(config.num_episodes, 5)
        self.assertEqual(config.max_steps, 100)
        self.assertEqual(config.temperature, 0.7)

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = BarlogWorkerConfig(
            run_id="custom_run",
            env_name="minihack",
            task="MiniHack-Room-5x5-v0",
            client_name="anthropic",
            model_id="claude-3-5-sonnet-20241022",
            agent_type="cot",
            num_episodes=10,
            max_steps=200,
            temperature=0.5,
        )

        self.assertEqual(config.env_name, "minihack")
        self.assertEqual(config.task, "MiniHack-Room-5x5-v0")
        self.assertEqual(config.client_name, "anthropic")
        self.assertEqual(config.model_id, "claude-3-5-sonnet-20241022")
        self.assertEqual(config.agent_type, "cot")

    def test_invalid_env_name(self) -> None:
        """Should raise ValueError for invalid env_name."""
        with self.assertRaises(ValueError) as ctx:
            BarlogWorkerConfig(run_id="test", env_name="invalid")
        self.assertIn("Invalid env_name", str(ctx.exception))

    def test_invalid_client_name(self) -> None:
        """Should raise ValueError for invalid client_name."""
        with self.assertRaises(ValueError) as ctx:
            BarlogWorkerConfig(run_id="test", client_name="invalid")
        self.assertIn("Invalid client_name", str(ctx.exception))

    def test_invalid_agent_type(self) -> None:
        """Should raise ValueError for invalid agent_type."""
        with self.assertRaises(ValueError) as ctx:
            BarlogWorkerConfig(run_id="test", agent_type="invalid")
        self.assertIn("Invalid agent_type", str(ctx.exception))

    def test_invalid_num_episodes(self) -> None:
        """Should raise ValueError for invalid num_episodes."""
        with self.assertRaises(ValueError) as ctx:
            BarlogWorkerConfig(run_id="test", num_episodes=0)
        self.assertIn("num_episodes must be >= 1", str(ctx.exception))

    def test_invalid_max_steps(self) -> None:
        """Should raise ValueError for invalid max_steps."""
        with self.assertRaises(ValueError) as ctx:
            BarlogWorkerConfig(run_id="test", max_steps=0)
        self.assertIn("max_steps must be >= 1", str(ctx.exception))

    def test_invalid_temperature(self) -> None:
        """Should raise ValueError for invalid temperature."""
        with self.assertRaises(ValueError) as ctx:
            BarlogWorkerConfig(run_id="test", temperature=3.0)
        self.assertIn("temperature must be between", str(ctx.exception))

    def test_to_dict(self) -> None:
        """to_dict should return serializable dict."""
        config = BarlogWorkerConfig(run_id="test", api_key="secret")
        data = config.to_dict()

        self.assertEqual(data["run_id"], "test")
        self.assertEqual(data["api_key"], "***")  # Masked

    def test_to_json(self) -> None:
        """to_json should return valid JSON string."""
        config = BarlogWorkerConfig(run_id="test")
        json_str = config.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed["run_id"], "test")

    def test_from_dict(self) -> None:
        """from_dict should create config from dict."""
        data = {
            "run_id": "from_dict_test",
            "env_name": "crafter",
            "num_episodes": 20,
        }
        config = BarlogWorkerConfig.from_dict(data)

        self.assertEqual(config.run_id, "from_dict_test")
        self.assertEqual(config.env_name, "crafter")
        self.assertEqual(config.num_episodes, 20)

    def test_from_json_file(self) -> None:
        """from_json_file should load config from file."""
        data = {
            "run_id": "file_test",
            "env_name": "minihack",
            "task": "MiniHack-Corridor-R3-v0",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            config = BarlogWorkerConfig.from_json_file(temp_path)
            self.assertEqual(config.run_id, "file_test")
            self.assertEqual(config.env_name, "minihack")
            self.assertEqual(config.task, "MiniHack-Corridor-R3-v0")
        finally:
            Path(temp_path).unlink()

    def test_to_balrog_config(self) -> None:
        """to_balrog_config should create BALROG-compatible config."""
        config = BarlogWorkerConfig(
            run_id="test",
            client_name="openai",
            model_id="gpt-4o",
            agent_type="cot",
            temperature=0.5,
        )
        balrog = config.to_balrog_config()

        self.assertEqual(balrog["client"]["client_name"], "openai")
        self.assertEqual(balrog["client"]["model_id"], "gpt-4o")
        self.assertEqual(balrog["agent"]["type"], "cot")
        self.assertEqual(balrog["client"]["generate_kwargs"]["temperature"], 0.5)


class TestConfigConstants(unittest.TestCase):
    """Test configuration constants."""

    def test_env_names(self) -> None:
        """ENV_NAMES should contain expected environments."""
        self.assertIn("babyai", ENV_NAMES)
        self.assertIn("minihack", ENV_NAMES)
        self.assertIn("crafter", ENV_NAMES)

    def test_client_names(self) -> None:
        """CLIENT_NAMES should contain expected clients."""
        self.assertIn("openai", CLIENT_NAMES)
        self.assertIn("anthropic", CLIENT_NAMES)
        self.assertIn("google", CLIENT_NAMES)
        self.assertIn("vllm", CLIENT_NAMES)

    def test_agent_types(self) -> None:
        """AGENT_TYPES should contain expected agent types."""
        self.assertIn("naive", AGENT_TYPES)
        self.assertIn("cot", AGENT_TYPES)
        self.assertIn("robust_naive", AGENT_TYPES)


if __name__ == "__main__":
    unittest.main()
