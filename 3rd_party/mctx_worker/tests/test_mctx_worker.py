"""Tests for MCTX Worker configuration and basic functionality.

These tests verify:
1. Configuration loading and validation
2. Environment enumeration
3. Algorithm enumeration
4. Basic network creation (if JAX available)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestMCTXWorkerConfig:
    """Tests for MCTXWorkerConfig."""

    def test_config_creation(self):
        """Test creating a basic config."""
        from mctx_worker.config import (
            MCTXWorkerConfig,
            MCTXAlgorithm,
            PGXEnvironment,
        )

        config = MCTXWorkerConfig(
            run_id="test_run_001",
            env_id="chess",
            algorithm=MCTXAlgorithm.ALPHAZERO,
        )

        assert config.run_id == "test_run_001"
        assert config.env_id == "chess"
        assert config.algorithm == MCTXAlgorithm.ALPHAZERO

    def test_config_validation(self):
        """Test configuration validation."""
        from mctx_worker.config import MCTXWorkerConfig

        config = MCTXWorkerConfig(
            run_id="test_validation",
            env_id="chess",
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_config_invalid_env(self):
        """Test validation fails for invalid env_id."""
        from mctx_worker.config import MCTXWorkerConfig

        config = MCTXWorkerConfig(
            run_id="test_invalid",
            env_id="invalid_game_v99",
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any("env_id" in e for e in errors)

    def test_config_to_dict(self):
        """Test serialization to dict."""
        from mctx_worker.config import (
            MCTXWorkerConfig,
            MCTXAlgorithm,
        )

        config = MCTXWorkerConfig(
            run_id="test_serial_001",
            env_id="go_9x9",
            algorithm=MCTXAlgorithm.GUMBEL_MUZERO,
        )

        data = config.to_dict()
        assert data["run_id"] == "test_serial_001"
        assert data["env_id"] == "go_9x9"
        assert data["algorithm"] == "gumbel_muzero"

    def test_config_from_dict(self):
        """Test deserialization from dict."""
        from mctx_worker.config import MCTXWorkerConfig

        data = {
            "run_id": "test_deserial_001",
            "env_id": "tic_tac_toe",
            "algorithm": "alphazero",
            "seed": 123,
            "mcts": {
                "num_simulations": 400,
                "dirichlet_alpha": 0.5,
            },
        }

        config = MCTXWorkerConfig.from_dict(data)
        assert config.run_id == "test_deserial_001"
        assert config.env_id == "tic_tac_toe"
        assert config.seed == 123
        assert config.mcts.num_simulations == 400
        assert config.mcts.dirichlet_alpha == 0.5

    def test_config_save_load(self):
        """Test config save and load roundtrip."""
        from mctx_worker.config import (
            MCTXWorkerConfig,
            load_worker_config,
            save_worker_config,
        )

        config = MCTXWorkerConfig(
            run_id="test_roundtrip",
            env_id="connect_four",
            seed=42,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            save_worker_config(config, config_path)

            loaded = load_worker_config(config_path)
            assert loaded.run_id == "test_roundtrip"
            assert loaded.env_id == "connect_four"
            assert loaded.seed == 42


class TestPGXEnvironments:
    """Tests for PGX environment enumeration."""

    def test_environment_enum(self):
        """Test PGXEnvironment enum values."""
        from mctx_worker.config import PGXEnvironment

        # Board games
        assert PGXEnvironment.CHESS.value == "chess"
        assert PGXEnvironment.GO_9X9.value == "go_9x9"
        assert PGXEnvironment.GO_19X19.value == "go_19x19"
        assert PGXEnvironment.SHOGI.value == "shogi"

        # Simple games
        assert PGXEnvironment.TIC_TAC_TOE.value == "tic_tac_toe"
        assert PGXEnvironment.CONNECT_FOUR.value == "connect_four"
        assert PGXEnvironment.OTHELLO.value == "othello"

    def test_all_envs_have_values(self):
        """Test all environment enum members have string values."""
        from mctx_worker.config import PGXEnvironment

        for env in PGXEnvironment:
            assert isinstance(env.value, str)
            assert len(env.value) > 0


class TestMCTXAlgorithms:
    """Tests for MCTS algorithm enumeration."""

    def test_algorithm_enum(self):
        """Test MCTXAlgorithm enum values."""
        from mctx_worker.config import MCTXAlgorithm

        assert MCTXAlgorithm.ALPHAZERO.value == "alphazero"
        assert MCTXAlgorithm.MUZERO.value == "muzero"
        assert MCTXAlgorithm.GUMBEL_MUZERO.value == "gumbel_muzero"

    def test_algorithm_from_string(self):
        """Test creating algorithm from string."""
        from mctx_worker.config import MCTXAlgorithm

        algo = MCTXAlgorithm("alphazero")
        assert algo == MCTXAlgorithm.ALPHAZERO


class TestNetworkConfig:
    """Tests for neural network configuration."""

    def test_network_config_defaults(self):
        """Test NetworkConfig default values."""
        from mctx_worker.config import NetworkConfig

        cfg = NetworkConfig()
        assert cfg.num_res_blocks == 8
        assert cfg.channels == 128
        assert cfg.hidden_dims == (256, 256)

    def test_network_config_custom(self):
        """Test NetworkConfig with custom values."""
        from mctx_worker.config import NetworkConfig

        cfg = NetworkConfig(
            num_res_blocks=16,
            channels=256,
            hidden_dims=(512, 512, 256),
        )
        assert cfg.num_res_blocks == 16
        assert cfg.channels == 256
        assert cfg.hidden_dims == (512, 512, 256)


class TestMCTSConfig:
    """Tests for MCTS configuration."""

    def test_mcts_config_defaults(self):
        """Test MCTSConfig default values."""
        from mctx_worker.config import MCTSConfig

        cfg = MCTSConfig()
        assert cfg.num_simulations == 800
        assert cfg.dirichlet_alpha == 0.3
        assert cfg.temperature == 1.0

    def test_mcts_config_custom(self):
        """Test MCTSConfig with custom values."""
        from mctx_worker.config import MCTSConfig

        cfg = MCTSConfig(
            num_simulations=1600,
            dirichlet_alpha=0.5,
            temperature=0.5,
        )
        assert cfg.num_simulations == 1600
        assert cfg.dirichlet_alpha == 0.5
        assert cfg.temperature == 0.5


class TestTrainingConfig:
    """Tests for training configuration."""

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        from mctx_worker.config import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.learning_rate == 2e-4
        assert cfg.batch_size == 256
        assert cfg.replay_buffer_size == 100_000

    def test_training_config_custom(self):
        """Test TrainingConfig with custom values."""
        from mctx_worker.config import TrainingConfig

        cfg = TrainingConfig(
            learning_rate=1e-3,
            batch_size=512,
            replay_buffer_size=500_000,
        )
        assert cfg.learning_rate == 1e-3
        assert cfg.batch_size == 512
        assert cfg.replay_buffer_size == 500_000


# JAX-dependent tests (skip if JAX not available)
@pytest.mark.skipif(
    not pytest.importorskip("jax", reason="JAX not installed"),
    reason="JAX required for runtime tests"
)
class TestJAXDependentFeatures:
    """Tests that require JAX."""

    def test_pgx_environment_creation(self):
        """Test creating a PGX environment."""
        import pgx

        env = pgx.make("tic_tac_toe")
        assert env is not None
        assert hasattr(env, "init")
        assert hasattr(env, "step")

    def test_replay_buffer(self):
        """Test ReplayBuffer functionality."""
        import numpy as np
        from mctx_worker.runtime import ReplayBuffer

        buffer = ReplayBuffer(
            capacity=100,
            obs_shape=(3, 3, 2),
            num_actions=9,
        )

        # Add some data
        for _ in range(50):
            obs = np.random.randn(3, 3, 2).astype(np.float32)
            policy = np.random.randn(9).astype(np.float32)
            policy = np.exp(policy) / np.exp(policy).sum()  # Softmax
            value = np.random.uniform(-1, 1)
            buffer.add(obs, policy, value)

        assert buffer.size == 50

        # Sample batch
        rng = np.random.default_rng(42)
        obs, policies, values = buffer.sample(16, rng)

        assert obs.shape == (16, 3, 3, 2)
        assert policies.shape == (16, 9)
        assert values.shape == (16,)
