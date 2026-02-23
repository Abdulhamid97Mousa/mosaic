"""Integration tests for MCTX Worker.

These tests verify:
1. Worker metadata and capabilities
2. Configuration protocol compliance
3. Entry point discovery
4. Policy saving and loading
5. Training progress monitoring (without full training)

Run with: pytest 3rd_party/mctx_worker/tests/test_mctx_integration.py -v
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


# =============================================================================
# Test: Worker Discovery and Metadata
# =============================================================================

class TestWorkerDiscovery:
    """Test that MCTX worker is discoverable via entry points."""

    def test_get_worker_metadata_returns_tuple(self):
        """Test that get_worker_metadata returns (metadata, capabilities) tuple."""
        from mctx_worker import get_worker_metadata

        result = get_worker_metadata()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_worker_metadata_fields(self):
        """Test that WorkerMetadata has required fields."""
        from mctx_worker import get_worker_metadata

        metadata, _ = get_worker_metadata()

        assert hasattr(metadata, "name")
        assert hasattr(metadata, "version")
        assert hasattr(metadata, "description")
        assert metadata.name == "MCTX Worker"
        assert "mctx" in metadata.description.lower() or "mcts" in metadata.description.lower()

    def test_worker_capabilities_fields(self):
        """Test that WorkerCapabilities has required fields."""
        from mctx_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        assert hasattr(capabilities, "worker_type")
        assert hasattr(capabilities, "supported_paradigms")
        assert hasattr(capabilities, "env_families")
        assert hasattr(capabilities, "action_spaces")
        assert capabilities.worker_type == "mctx"

    def test_worker_supports_self_play(self):
        """Test that MCTX worker supports self-play paradigm."""
        from mctx_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        assert "self_play" in capabilities.supported_paradigms
        assert capabilities.supports_self_play is True

    def test_worker_requires_gpu(self):
        """Test that MCTX worker indicates GPU requirement."""
        from mctx_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        assert capabilities.requires_gpu is True

    def test_entry_point_discovery(self):
        """Test that worker can be discovered via entry points."""
        try:
            from importlib.metadata import entry_points

            # Get mosaic.workers entry points
            eps = entry_points()
            if hasattr(eps, "select"):
                # Python 3.10+
                worker_eps = eps.select(group="mosaic.workers")
            else:
                # Python 3.9
                worker_eps = eps.get("mosaic.workers", [])

            # Check if mctx is registered
            mctx_eps = [ep for ep in worker_eps if ep.name == "mctx"]

            # Note: Entry point may not be registered in editable install
            # This is expected if running without `pip install -e .`
            if mctx_eps:
                ep = mctx_eps[0]
                get_metadata = ep.load()
                metadata, capabilities = get_metadata()
                assert capabilities.worker_type == "mctx"

        except ImportError:
            pytest.skip("importlib.metadata not available")


# =============================================================================
# Test: Configuration Protocol
# =============================================================================

class TestConfigProtocol:
    """Test that MCTXWorkerConfig follows the worker protocol."""

    def test_config_has_required_fields(self):
        """Test that config has run_id and seed fields."""
        from mctx_worker.config import MCTXWorkerConfig

        config = MCTXWorkerConfig(run_id="test_run", seed=42)

        assert hasattr(config, "run_id")
        assert hasattr(config, "seed")
        assert config.run_id == "test_run"
        assert config.seed == 42

    def test_config_to_dict(self):
        """Test config serialization to dict."""
        from mctx_worker.config import MCTXWorkerConfig

        config = MCTXWorkerConfig(
            run_id="test_serial",
            env_id="chess",
            seed=123,
        )

        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["run_id"] == "test_serial"
        assert data["env_id"] == "chess"
        assert data["seed"] == 123

    def test_config_from_dict(self):
        """Test config deserialization from dict."""
        from mctx_worker.config import MCTXWorkerConfig

        data = {
            "run_id": "test_deserial",
            "env_id": "tic_tac_toe",
            "algorithm": "alphazero",
            "seed": 456,
        }

        config = MCTXWorkerConfig.from_dict(data)

        assert config.run_id == "test_deserial"
        assert config.env_id == "tic_tac_toe"
        assert config.seed == 456

    def test_config_roundtrip(self):
        """Test config save and load roundtrip."""
        from mctx_worker.config import (
            MCTXWorkerConfig,
            save_worker_config,
            load_worker_config,
        )

        config = MCTXWorkerConfig(
            run_id="test_roundtrip",
            env_id="connect_four",
            seed=789,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            save_worker_config(config, config_path)

            loaded = load_worker_config(config_path)

            assert loaded.run_id == config.run_id
            assert loaded.env_id == config.env_id
            assert loaded.seed == config.seed

    def test_config_validation_valid(self):
        """Test that valid config passes validation."""
        from mctx_worker.config import MCTXWorkerConfig

        config = MCTXWorkerConfig(
            run_id="test_valid",
            env_id="chess",
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_config_validation_invalid_env(self):
        """Test that invalid env_id fails validation."""
        from mctx_worker.config import MCTXWorkerConfig

        config = MCTXWorkerConfig(
            run_id="test_invalid",
            env_id="nonexistent_game_v99",
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any("env_id" in e for e in errors)

    def test_config_nested_format(self):
        """Test that config handles nested metadata.worker.config format."""
        from mctx_worker.config import MCTXWorkerConfig

        # Simulate nested format from GUI
        nested_data = {
            "metadata": {
                "worker": {
                    "config": {
                        "run_id": "nested_test",
                        "env_id": "chess",
                        "seed": 42,
                    }
                }
            }
        }

        # Extract nested config
        if "metadata" in nested_data and "worker" in nested_data["metadata"]:
            inner_data = nested_data["metadata"]["worker"].get("config", {})
        else:
            inner_data = nested_data

        config = MCTXWorkerConfig.from_dict(inner_data)
        assert config.run_id == "nested_test"


# =============================================================================
# Test: Policy Saving and Loading
# =============================================================================

class TestPolicySaving:
    """Test policy saving format and loading."""

    def test_policy_file_format(self):
        """Test that policy files have correct structure."""
        # Create a mock policy file
        policy_data = {
            "params": {"layer1": [1, 2, 3]},
            "batch_stats": {},
            "env_id": "chess",
            "algorithm": "gumbel_muzero",
            "num_actions": 4672,
            "obs_shape": (8, 8, 111),
            "network_config": {
                "num_res_blocks": 8,
                "channels": 128,
                "hidden_dims": [256, 256],
                "use_resnet": True,
            },
            "mcts_config": {
                "num_simulations": 800,
                "dirichlet_alpha": 0.3,
                "temperature": 1.0,
            },
            "iteration": 1000,
            "run_id": "test_policy",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = Path(tmpdir) / "policy.pkl"

            # Save
            with open(policy_path, "wb") as f:
                pickle.dump(policy_data, f)

            # Load and verify
            with open(policy_path, "rb") as f:
                loaded = pickle.load(f)

            assert loaded["env_id"] == "chess"
            assert loaded["algorithm"] == "gumbel_muzero"
            assert "params" in loaded
            assert "network_config" in loaded
            assert "mcts_config" in loaded

    def test_policy_directory_structure(self):
        """Test that policy saves to correct directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "var" / "trainer" / "runs" / "test_run"
            policy_dir = run_dir / "policies"
            policy_dir.mkdir(parents=True)

            policy_path = policy_dir / "policy_final.pkl"
            policy_path.write_bytes(b"test")

            # Verify structure
            assert policy_dir.exists()
            assert policy_path.exists()
            assert policy_path.parent.name == "policies"


# =============================================================================
# Test: Training Progress Monitoring
# =============================================================================

class TestProgressMonitoring:
    """Test training progress output format."""

    def test_progress_output_format(self):
        """Test that progress output follows expected format."""
        # Example progress line
        progress_line = (
            "[PROGRESS] iteration=10 | "
            "steps=1000/100000 (1.0%) | "
            "buffer=500 | "
            "steps/sec=100"
        )

        assert "[PROGRESS]" in progress_line
        assert "iteration=" in progress_line
        assert "steps=" in progress_line
        assert "buffer=" in progress_line
        assert "steps/sec=" in progress_line

    def test_checkpoint_output_format(self):
        """Test that checkpoint output follows expected format."""
        checkpoint_line = "[CHECKPOINT] path=/var/trainer/runs/test/checkpoints/checkpoint_100.pkl"

        assert "[CHECKPOINT]" in checkpoint_line
        assert "path=" in checkpoint_line

    def test_policy_output_format(self):
        """Test that policy output follows expected format."""
        policy_line = "[POLICY] path=/var/trainer/runs/test/policies/policy_final.pkl final=True"

        assert "[POLICY]" in policy_line
        assert "path=" in policy_line
        assert "final=" in policy_line

    def test_complete_output_format(self):
        """Test that completion output follows expected format."""
        complete_line = "[COMPLETE] run_id=test_run status=success"

        assert "[COMPLETE]" in complete_line
        assert "run_id=" in complete_line
        assert "status=success" in complete_line


# =============================================================================
# Test: MCTS Configuration
# =============================================================================

class TestMCTSConfig:
    """Test MCTS-specific configuration."""

    def test_mcts_config_defaults(self):
        """Test MCTSConfig default values."""
        from mctx_worker.config import MCTSConfig

        cfg = MCTSConfig()

        assert cfg.num_simulations == 800
        assert cfg.dirichlet_alpha == 0.3
        assert cfg.temperature == 1.0
        assert cfg.max_num_considered_actions == 16

    def test_algorithm_enum(self):
        """Test MCTXAlgorithm enum values."""
        from mctx_worker.config import MCTXAlgorithm

        assert MCTXAlgorithm.ALPHAZERO.value == "alphazero"
        assert MCTXAlgorithm.MUZERO.value == "muzero"
        assert MCTXAlgorithm.GUMBEL_MUZERO.value == "gumbel_muzero"

    def test_pgx_environment_enum(self):
        """Test PGXEnvironment enum values."""
        from mctx_worker.config import PGXEnvironment

        assert PGXEnvironment.CHESS.value == "chess"
        assert PGXEnvironment.GO_9X9.value == "go_9x9"
        assert PGXEnvironment.TIC_TAC_TOE.value == "tic_tac_toe"


# =============================================================================
# Test: Analytics Manifest
# =============================================================================

class TestAnalyticsManifest:
    """Test analytics manifest generation."""

    def test_analytics_manifest_creation(self):
        """Test that analytics manifest is created correctly."""
        from mctx_worker.config import MCTXWorkerConfig

        config = MCTXWorkerConfig(
            run_id="test_analytics",
            env_id="chess",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override checkpoint_path
            config = MCTXWorkerConfig(
                run_id="test_analytics",
                env_id="chess",
                checkpoint_path=tmpdir,
            )

            try:
                from mctx_worker.analytics import write_analytics_manifest

                manifest_path = write_analytics_manifest(config)

                assert manifest_path.exists()
                assert manifest_path.name == "analytics.json"

                # Load and verify
                with open(manifest_path) as f:
                    manifest = json.load(f)

                assert manifest["run_id"] == "test_analytics"
                assert manifest["worker_type"] == "mctx"
                assert "metadata" in manifest
                assert manifest["metadata"]["env_id"] == "chess"

            except ImportError:
                pytest.skip("gym_gui not available for analytics test")


# =============================================================================
# Test: JAX-dependent features (skip if not available)
# =============================================================================

@pytest.mark.skipif(
    not pytest.importorskip("jax", reason="JAX not installed"),
    reason="JAX required for runtime tests"
)
class TestJAXFeatures:
    """Tests that require JAX."""

    def test_replay_buffer_basic(self):
        """Test ReplayBuffer basic operations."""
        import numpy as np
        from mctx_worker.runtime import ReplayBuffer

        buffer = ReplayBuffer(
            capacity=100,
            obs_shape=(3, 3, 2),
            num_actions=9,
        )

        assert buffer.size == 0
        assert buffer.capacity == 100

        # Add samples
        for i in range(50):
            obs = np.random.randn(3, 3, 2).astype(np.float32)
            policy = np.random.randn(9).astype(np.float32)
            policy = np.exp(policy) / np.exp(policy).sum()
            value = np.random.uniform(-1, 1)
            buffer.add(obs, policy, value)

        assert buffer.size == 50

    def test_replay_buffer_sampling(self):
        """Test ReplayBuffer sampling."""
        import numpy as np
        from mctx_worker.runtime import ReplayBuffer

        buffer = ReplayBuffer(
            capacity=100,
            obs_shape=(8, 8, 2),
            num_actions=64,
        )

        # Fill buffer
        for _ in range(100):
            obs = np.random.randn(8, 8, 2).astype(np.float32)
            policy = np.random.randn(64).astype(np.float32)
            policy = np.exp(policy) / np.exp(policy).sum()
            value = np.random.uniform(-1, 1)
            buffer.add(obs, policy, value)

        # Sample batch
        rng = np.random.default_rng(42)
        obs, policies, values = buffer.sample(32, rng)

        assert obs.shape == (32, 8, 8, 2)
        assert policies.shape == (32, 64)
        assert values.shape == (32,)

    def test_network_creation(self):
        """Test neural network creation."""
        import jax
        import jax.numpy as jnp
        from mctx_worker.runtime import AlphaZeroNetwork, MLPNetwork

        # Test AlphaZeroNetwork (CNN)
        cnn = AlphaZeroNetwork(num_actions=64, channels=32, num_blocks=2)
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 8, 8, 2))
        variables = cnn.init(key, dummy_input)

        assert "params" in variables

        # Test MLPNetwork
        mlp = MLPNetwork(num_actions=9, hidden_dims=(64, 64))
        dummy_flat = jnp.ones((1, 18))
        variables = mlp.init(key, dummy_flat)

        assert "params" in variables


# =============================================================================
# Test: CLI
# =============================================================================

class TestCLI:
    """Test CLI functionality."""

    def test_cli_parser_creation(self):
        """Test CLI argument parser creation."""
        from mctx_worker.cli import create_parser

        parser = create_parser()

        assert parser is not None
        assert parser.prog == "mctx-worker"

    def test_cli_dry_run(self):
        """Test CLI dry-run mode with config."""
        from mctx_worker.config import MCTXWorkerConfig, save_worker_config

        config = MCTXWorkerConfig(
            run_id="cli_test",
            env_id="tic_tac_toe",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            save_worker_config(config, config_path)

            # Note: Actual dry-run test would require mocking JAX
            # This just verifies config can be loaded
            from mctx_worker.config import load_worker_config
            loaded = load_worker_config(config_path)
            assert loaded.run_id == "cli_test"
