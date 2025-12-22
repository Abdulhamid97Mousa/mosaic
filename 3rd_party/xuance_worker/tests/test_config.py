"""Tests for XuanCeWorkerConfig dataclass.

These tests validate:
- Default value initialization
- Custom value setting
- Immutability (frozen dataclass)
- Dictionary conversion (to_dict, from_dict)
- JSON serialization (to_json, from_json_file)
- CleanRL-style key aliases
- Extras field handling
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from xuance_worker.config import XuanCeWorkerConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def minimal_config() -> XuanCeWorkerConfig:
    """Create a minimal valid configuration."""
    return XuanCeWorkerConfig(
        run_id="test-run-001",
        method="ppo",
        env="classic_control",
        env_id="CartPole-v1",
    )


@pytest.fixture
def full_config() -> XuanCeWorkerConfig:
    """Create a configuration with all fields specified."""
    return XuanCeWorkerConfig(
        run_id="full-run-001",
        method="dqn",
        env="atari",
        env_id="Pong-v5",
        dl_toolbox="torch",
        running_steps=500_000,
        seed=42,
        device="cuda:0",
        parallels=16,
        test_mode=True,
        config_path="/custom/config.yaml",
        worker_id="worker-001",
        extras={"learning_rate": 0.001, "batch_size": 64},
    )


@pytest.fixture
def config_json_file(tmp_path: Path) -> Path:
    """Create a JSON configuration file for testing."""
    config_data = {
        "run_id": "json-run-001",
        "method": "sac",
        "env": "mujoco",
        "env_id": "HalfCheetah-v4",
        "dl_toolbox": "torch",
        "running_steps": 1_000_000,
        "seed": 123,
        "device": "cuda:0",
        "parallels": 8,
        "test_mode": False,
        "extras": {"gamma": 0.99},
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    return config_file


# =============================================================================
# Basic Configuration Tests
# =============================================================================


class TestXuanCeWorkerConfigBasic:
    """Basic tests for XuanCeWorkerConfig."""

    def test_minimal_config_creation(self, minimal_config: XuanCeWorkerConfig) -> None:
        """Test creating config with only required fields."""
        assert minimal_config.run_id == "test-run-001"
        assert minimal_config.method == "ppo"
        assert minimal_config.env == "classic_control"
        assert minimal_config.env_id == "CartPole-v1"

    def test_default_values(self, minimal_config: XuanCeWorkerConfig) -> None:
        """Test that default values are correctly set."""
        assert minimal_config.dl_toolbox == "torch"
        assert minimal_config.running_steps == 1_000_000
        assert minimal_config.seed is None
        assert minimal_config.device == "cpu"
        assert minimal_config.parallels == 8
        assert minimal_config.test_mode is False
        assert minimal_config.config_path is None
        assert minimal_config.worker_id is None
        assert minimal_config.extras == {}

    def test_full_config_creation(self, full_config: XuanCeWorkerConfig) -> None:
        """Test creating config with all fields specified."""
        assert full_config.run_id == "full-run-001"
        assert full_config.method == "dqn"
        assert full_config.env == "atari"
        assert full_config.env_id == "Pong-v5"
        assert full_config.dl_toolbox == "torch"
        assert full_config.running_steps == 500_000
        assert full_config.seed == 42
        assert full_config.device == "cuda:0"
        assert full_config.parallels == 16
        assert full_config.test_mode is True
        assert full_config.config_path == "/custom/config.yaml"
        assert full_config.worker_id == "worker-001"
        assert full_config.extras == {"learning_rate": 0.001, "batch_size": 64}

    def test_config_is_frozen(self, minimal_config: XuanCeWorkerConfig) -> None:
        """Test that config is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            minimal_config.run_id = "modified"  # type: ignore

        with pytest.raises(AttributeError):
            minimal_config.method = "dqn"  # type: ignore


# =============================================================================
# Dictionary Conversion Tests
# =============================================================================


class TestXuanCeWorkerConfigDict:
    """Tests for dictionary conversion methods."""

    def test_to_dict(self, full_config: XuanCeWorkerConfig) -> None:
        """Test converting config to dictionary."""
        d = full_config.to_dict()

        assert isinstance(d, dict)
        assert d["run_id"] == "full-run-001"
        assert d["method"] == "dqn"
        assert d["env"] == "atari"
        assert d["env_id"] == "Pong-v5"
        assert d["dl_toolbox"] == "torch"
        assert d["running_steps"] == 500_000
        assert d["seed"] == 42
        assert d["device"] == "cuda:0"
        assert d["parallels"] == 16
        assert d["test_mode"] is True
        assert d["config_path"] == "/custom/config.yaml"
        assert d["worker_id"] == "worker-001"
        assert d["extras"] == {"learning_rate": 0.001, "batch_size": 64}

    def test_from_dict_minimal(self) -> None:
        """Test creating config from minimal dictionary."""
        data = {
            "run_id": "dict-run",
            "method": "ppo",
            "env": "classic_control",
            "env_id": "CartPole-v1",
        }
        config = XuanCeWorkerConfig.from_dict(data)

        assert config.run_id == "dict-run"
        assert config.method == "ppo"
        assert config.env == "classic_control"
        assert config.env_id == "CartPole-v1"
        # Check defaults
        assert config.dl_toolbox == "torch"
        assert config.running_steps == 1_000_000

    def test_from_dict_full(self) -> None:
        """Test creating config from full dictionary."""
        data = {
            "run_id": "full-dict-run",
            "method": "td3",
            "env": "box2d",
            "env_id": "LunarLander-v2",
            "dl_toolbox": "tensorflow",
            "running_steps": 250_000,
            "seed": 99,
            "device": "cuda:1",
            "parallels": 4,
            "test_mode": True,
            "config_path": "/path/to/config.yaml",
            "worker_id": "worker-xyz",
            "extras": {"tau": 0.005},
        }
        config = XuanCeWorkerConfig.from_dict(data)

        assert config.run_id == "full-dict-run"
        assert config.method == "td3"
        assert config.env == "box2d"
        assert config.env_id == "LunarLander-v2"
        assert config.dl_toolbox == "tensorflow"
        assert config.running_steps == 250_000
        assert config.seed == 99
        assert config.device == "cuda:1"
        assert config.parallels == 4
        assert config.test_mode is True
        assert config.config_path == "/path/to/config.yaml"
        assert config.worker_id == "worker-xyz"
        assert config.extras == {"tau": 0.005}

    def test_to_dict_from_dict_roundtrip(self, full_config: XuanCeWorkerConfig) -> None:
        """Test roundtrip conversion to and from dictionary."""
        d = full_config.to_dict()
        restored = XuanCeWorkerConfig.from_dict(d)

        assert restored.run_id == full_config.run_id
        assert restored.method == full_config.method
        assert restored.env == full_config.env
        assert restored.env_id == full_config.env_id
        assert restored.dl_toolbox == full_config.dl_toolbox
        assert restored.running_steps == full_config.running_steps
        assert restored.seed == full_config.seed
        assert restored.device == full_config.device
        assert restored.parallels == full_config.parallels
        assert restored.test_mode == full_config.test_mode
        assert restored.config_path == full_config.config_path
        assert restored.worker_id == full_config.worker_id
        assert restored.extras == full_config.extras


# =============================================================================
# JSON Serialization Tests
# =============================================================================


class TestXuanCeWorkerConfigJson:
    """Tests for JSON serialization methods."""

    def test_to_json(self, minimal_config: XuanCeWorkerConfig) -> None:
        """Test converting config to JSON string."""
        json_str = minimal_config.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["run_id"] == "test-run-001"
        assert data["method"] == "ppo"

    def test_to_json_compact(self, minimal_config: XuanCeWorkerConfig) -> None:
        """Test compact JSON serialization."""
        json_str = minimal_config.to_json(indent=None)

        assert isinstance(json_str, str)
        assert "\n" not in json_str  # Compact format

    def test_from_json_file(self, config_json_file: Path) -> None:
        """Test loading config from JSON file."""
        config = XuanCeWorkerConfig.from_json_file(config_json_file)

        assert config.run_id == "json-run-001"
        assert config.method == "sac"
        assert config.env == "mujoco"
        assert config.env_id == "HalfCheetah-v4"
        assert config.seed == 123
        assert config.extras == {"gamma": 0.99}

    def test_from_json_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            XuanCeWorkerConfig.from_json_file(tmp_path / "nonexistent.json")

    def test_from_json_file_invalid_json(self, tmp_path: Path) -> None:
        """Test that JSONDecodeError is raised for invalid JSON."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            XuanCeWorkerConfig.from_json_file(invalid_file)

    def test_to_json_from_json_roundtrip(
        self, full_config: XuanCeWorkerConfig, tmp_path: Path
    ) -> None:
        """Test roundtrip conversion to and from JSON file."""
        json_file = tmp_path / "roundtrip.json"
        json_file.write_text(full_config.to_json())
        restored = XuanCeWorkerConfig.from_json_file(json_file)

        assert restored.run_id == full_config.run_id
        assert restored.method == full_config.method
        assert restored.running_steps == full_config.running_steps


# =============================================================================
# CleanRL Compatibility Tests
# =============================================================================


class TestXuanCeWorkerConfigCleanRLCompat:
    """Tests for CleanRL-style key compatibility."""

    def test_algo_key_alias(self) -> None:
        """Test that 'algo' is accepted as alias for 'method'."""
        data = {
            "run_id": "alias-test",
            "algo": "ddpg",  # CleanRL-style
            "env": "mujoco",
            "env_id": "Ant-v4",
        }
        config = XuanCeWorkerConfig.from_dict(data)

        assert config.method == "ddpg"

    def test_total_timesteps_key_alias(self) -> None:
        """Test that 'total_timesteps' is accepted as alias for 'running_steps'."""
        data = {
            "run_id": "alias-test",
            "method": "ppo",
            "env": "classic_control",
            "env_id": "CartPole-v1",
            "total_timesteps": 50_000,  # CleanRL-style
        }
        config = XuanCeWorkerConfig.from_dict(data)

        assert config.running_steps == 50_000

    def test_method_takes_precedence_over_algo(self) -> None:
        """Test that 'method' takes precedence over 'algo' if both present."""
        data = {
            "run_id": "precedence-test",
            "method": "sac",
            "algo": "ppo",  # Should be ignored
            "env": "mujoco",
            "env_id": "Walker2d-v4",
        }
        config = XuanCeWorkerConfig.from_dict(data)

        assert config.method == "sac"

    def test_running_steps_takes_precedence(self) -> None:
        """Test that 'running_steps' takes precedence over 'total_timesteps'."""
        data = {
            "run_id": "precedence-test",
            "method": "ppo",
            "env": "classic_control",
            "env_id": "CartPole-v1",
            "running_steps": 100_000,
            "total_timesteps": 50_000,  # Should be ignored
        }
        config = XuanCeWorkerConfig.from_dict(data)

        assert config.running_steps == 100_000


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestXuanCeWorkerConfigEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_run_id(self) -> None:
        """Test handling of empty run_id."""
        config = XuanCeWorkerConfig(
            run_id="",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
        )
        assert config.run_id == ""

    def test_seed_zero_preserved(self) -> None:
        """Test that seed=0 is preserved (not treated as None)."""
        data = {
            "run_id": "seed-zero",
            "method": "ppo",
            "env": "classic_control",
            "env_id": "CartPole-v1",
            "seed": 0,
        }
        config = XuanCeWorkerConfig.from_dict(data)

        assert config.seed == 0

    def test_empty_extras(self) -> None:
        """Test handling of empty extras dictionary."""
        config = XuanCeWorkerConfig(
            run_id="empty-extras",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            extras={},
        )
        assert config.extras == {}

    def test_nested_extras(self) -> None:
        """Test handling of nested extras dictionary."""
        nested_extras = {
            "algo_params": {
                "learning_rate": 0.001,
                "network": {"hidden_sizes": [256, 256]},
            },
            "env_params": {"max_steps": 1000},
        }
        config = XuanCeWorkerConfig(
            run_id="nested-extras",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            extras=nested_extras,
        )

        assert config.extras["algo_params"]["learning_rate"] == 0.001
        assert config.extras["algo_params"]["network"]["hidden_sizes"] == [256, 256]

    def test_various_backends(self) -> None:
        """Test all supported backend options."""
        for backend in ["torch", "tensorflow", "mindspore"]:
            config = XuanCeWorkerConfig(
                run_id=f"backend-{backend}",
                method="ppo",
                env="classic_control",
                env_id="CartPole-v1",
                dl_toolbox=backend,
            )
            assert config.dl_toolbox == backend

    def test_various_devices(self) -> None:
        """Test various device configurations."""
        for device in ["cpu", "cuda:0", "cuda:1", "cuda"]:
            config = XuanCeWorkerConfig(
                run_id=f"device-{device}",
                method="ppo",
                env="classic_control",
                env_id="CartPole-v1",
                device=device,
            )
            assert config.device == device

    def test_from_dict_missing_optional_fields(self) -> None:
        """Test from_dict with missing optional fields uses defaults."""
        data: dict[str, Any] = {}  # Empty dict
        config = XuanCeWorkerConfig.from_dict(data)

        # Should use all defaults
        assert config.run_id == ""
        assert config.method == "dqn"
        assert config.env == "classic_control"
        assert config.env_id == "CartPole-v1"
        assert config.dl_toolbox == "torch"
        assert config.running_steps == 1_000_000


# =============================================================================
# Multi-Agent Configuration Tests
# =============================================================================


class TestXuanCeWorkerConfigMultiAgent:
    """Tests for multi-agent configuration scenarios."""

    def test_multi_agent_mpe_config(self) -> None:
        """Test configuration for MPE environments."""
        config = XuanCeWorkerConfig(
            run_id="mpe-test",
            method="mappo",
            env="mpe",
            env_id="simple_spread_v3",
            extras={"n_agents": 3, "share_policy": True},
        )

        assert config.method == "mappo"
        assert config.env == "mpe"
        assert config.extras["n_agents"] == 3

    def test_multi_agent_smac_config(self) -> None:
        """Test configuration for SMAC environments."""
        config = XuanCeWorkerConfig(
            run_id="smac-test",
            method="qmix",
            env="smac",
            env_id="3m",
            extras={"mixing_hidden_dim": 32},
        )

        assert config.method == "qmix"
        assert config.env == "smac"
        assert config.env_id == "3m"

    def test_multi_agent_pettingzoo_config(self) -> None:
        """Test configuration for PettingZoo environments."""
        config = XuanCeWorkerConfig(
            run_id="pettingzoo-test",
            method="maddpg",
            env="mpe",
            env_id="simple_adversary_v3",
        )

        assert config.method == "maddpg"
        assert config.env == "mpe"
