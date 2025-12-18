"""Tests for XuanCeWorkerConfig."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from xuance_worker.config import XuanCeWorkerConfig


class TestXuanCeWorkerConfig:
    """Test suite for XuanCeWorkerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = XuanCeWorkerConfig(
            run_id="test_run",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
        )

        assert config.run_id == "test_run"
        assert config.method == "ppo"
        assert config.env == "classic_control"
        assert config.env_id == "CartPole-v1"
        assert config.dl_toolbox == "torch"  # default backend
        assert config.running_steps == 1_000_000
        assert config.seed is None
        assert config.device == "cpu"
        assert config.parallels == 8
        assert config.test_mode is False
        assert config.config_path is None
        assert config.worker_id is None
        assert config.extras == {}

    def test_custom_values(self) -> None:
        """Test config with custom values."""
        config = XuanCeWorkerConfig(
            run_id="custom_run",
            method="dqn",
            env="atari",
            env_id="Pong-v5",
            dl_toolbox="tensorflow",
            running_steps=50000,
            seed=42,
            device="cuda:0",
            parallels=16,
            test_mode=True,
            config_path="/custom/config.yaml",
            worker_id="worker_1",
            extras={"learning_rate": 0.001},
        )

        assert config.run_id == "custom_run"
        assert config.method == "dqn"
        assert config.env == "atari"
        assert config.env_id == "Pong-v5"
        assert config.dl_toolbox == "tensorflow"
        assert config.running_steps == 50000
        assert config.seed == 42
        assert config.device == "cuda:0"
        assert config.parallels == 16
        assert config.test_mode is True
        assert config.config_path == "/custom/config.yaml"
        assert config.worker_id == "worker_1"
        assert config.extras == {"learning_rate": 0.001}

    def test_frozen_dataclass(self) -> None:
        """Test that config is immutable (frozen)."""
        config = XuanCeWorkerConfig(
            run_id="test",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
        )

        with pytest.raises(AttributeError):
            config.method = "dqn"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = XuanCeWorkerConfig(
            run_id="test_run",
            method="sac",
            env="mujoco",
            env_id="Hopper-v4",
            dl_toolbox="mindspore",
            running_steps=100000,
            seed=123,
        )

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["run_id"] == "test_run"
        assert result["method"] == "sac"
        assert result["env"] == "mujoco"
        assert result["env_id"] == "Hopper-v4"
        assert result["dl_toolbox"] == "mindspore"
        assert result["running_steps"] == 100000
        assert result["seed"] == 123

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        config = XuanCeWorkerConfig(
            run_id="json_test",
            method="td3",
            env="box2d",
            env_id="LunarLander-v2",
        )

        json_str = config.to_json()
        parsed = json.loads(json_str)

        assert parsed["run_id"] == "json_test"
        assert parsed["method"] == "td3"
        assert parsed["env"] == "box2d"
        assert parsed["env_id"] == "LunarLander-v2"

    def test_to_json_compact(self) -> None:
        """Test compact JSON serialization."""
        config = XuanCeWorkerConfig(
            run_id="compact",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
        )

        json_str = config.to_json(indent=None)

        # Compact JSON should have no newlines
        assert "\n" not in json_str

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "run_id": "from_dict_test",
            "method": "mappo",
            "env": "mpe",
            "env_id": "simple_spread_v3",
            "dl_toolbox": "tensorflow",
            "running_steps": 200000,
            "seed": 456,
            "device": "cuda:1",
            "parallels": 4,
        }

        config = XuanCeWorkerConfig.from_dict(data)

        assert config.run_id == "from_dict_test"
        assert config.method == "mappo"
        assert config.env == "mpe"
        assert config.env_id == "simple_spread_v3"
        assert config.dl_toolbox == "tensorflow"
        assert config.running_steps == 200000
        assert config.seed == 456
        assert config.device == "cuda:1"
        assert config.parallels == 4

    def test_from_dict_cleanrl_style_keys(self) -> None:
        """Test from_dict with CleanRL-style key aliases."""
        data = {
            "run_id": "cleanrl_style",
            "algo": "dqn",  # CleanRL uses 'algo' instead of 'method'
            "env": "classic_control",
            "env_id": "CartPole-v1",
            "total_timesteps": 500000,  # CleanRL uses 'total_timesteps'
        }

        config = XuanCeWorkerConfig.from_dict(data)

        assert config.method == "dqn"  # 'algo' mapped to 'method'
        assert config.running_steps == 500000  # 'total_timesteps' mapped

    def test_from_dict_defaults(self) -> None:
        """Test from_dict with minimal data uses defaults."""
        data = {"run_id": "minimal"}

        config = XuanCeWorkerConfig.from_dict(data)

        assert config.run_id == "minimal"
        assert config.method == "dqn"  # default
        assert config.env == "classic_control"  # default
        assert config.env_id == "CartPole-v1"  # default

    def test_from_dict_extras(self) -> None:
        """Test from_dict with extras field."""
        data = {
            "run_id": "extras_test",
            "method": "ppo",
            "env": "classic_control",
            "env_id": "CartPole-v1",
            "extras": {"custom_param": 123, "another": "value"},
        }

        config = XuanCeWorkerConfig.from_dict(data)

        assert config.extras == {"custom_param": 123, "another": "value"}

    def test_from_json_file(self) -> None:
        """Test loading config from JSON file."""
        data = {
            "run_id": "file_test",
            "method": "qmix",
            "env": "smac",
            "env_id": "3m",
            "running_steps": 300000,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            config = XuanCeWorkerConfig.from_json_file(temp_path)

            assert config.run_id == "file_test"
            assert config.method == "qmix"
            assert config.env == "smac"
            assert config.env_id == "3m"
            assert config.running_steps == 300000
        finally:
            temp_path.unlink()

    def test_from_json_file_not_found(self) -> None:
        """Test from_json_file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            XuanCeWorkerConfig.from_json_file(Path("/nonexistent/config.json"))

    def test_roundtrip_dict(self) -> None:
        """Test roundtrip conversion through dict."""
        original = XuanCeWorkerConfig(
            run_id="roundtrip",
            method="maddpg",
            env="mpe",
            env_id="simple_adversary_v3",
            dl_toolbox="mindspore",
            running_steps=150000,
            seed=789,
            device="cpu",
            parallels=12,
            test_mode=False,
            config_path="/path/to/config.yaml",
            worker_id="worker_x",
            extras={"gamma": 0.99},
        )

        data = original.to_dict()
        restored = XuanCeWorkerConfig.from_dict(data)

        assert restored.run_id == original.run_id
        assert restored.method == original.method
        assert restored.env == original.env
        assert restored.env_id == original.env_id
        assert restored.dl_toolbox == original.dl_toolbox
        assert restored.running_steps == original.running_steps
        assert restored.seed == original.seed
        assert restored.device == original.device
        assert restored.parallels == original.parallels
        assert restored.test_mode == original.test_mode
        assert restored.config_path == original.config_path
        assert restored.worker_id == original.worker_id
        assert restored.extras == original.extras


class TestXuanCeWorkerConfigEdgeCases:
    """Edge case tests for XuanCeWorkerConfig."""

    def test_empty_run_id(self) -> None:
        """Test handling of empty run_id."""
        config = XuanCeWorkerConfig(
            run_id="",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
        )

        assert config.run_id == ""

    def test_seed_zero(self) -> None:
        """Test that seed=0 is preserved (not treated as None)."""
        config = XuanCeWorkerConfig(
            run_id="seed_zero",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            seed=0,
        )

        assert config.seed == 0
        assert config.seed is not None

    def test_empty_extras(self) -> None:
        """Test that empty extras dict is handled."""
        config = XuanCeWorkerConfig(
            run_id="empty_extras",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            extras={},
        )

        assert config.extras == {}
        assert isinstance(config.extras, dict)
