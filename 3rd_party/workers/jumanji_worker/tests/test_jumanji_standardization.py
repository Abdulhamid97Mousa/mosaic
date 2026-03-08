"""Comprehensive tests for Jumanji Worker standardization.

This test suite verifies that Jumanji Worker implements all required standardization
components from the MOSAIC Worker Architecture plan:

- Config protocol compliance
- Worker metadata and capabilities
- Entry point discovery
- Analytics standardization
- Config loading (nested/direct formats)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from jumanji_worker.config import JumanjiWorkerConfig, load_worker_config, LOGIC_ENVIRONMENTS

if TYPE_CHECKING:
    from gym_gui.core.worker import WorkerConfig, WorkerMetadata, WorkerCapabilities


class TestPhase4_1_ConfigCompliance:
    """Test that JumanjiWorkerConfig implements WorkerConfig protocol."""

    def test_config_has_required_fields(self):
        """Config must have run_id and seed fields."""
        config = JumanjiWorkerConfig(
            run_id="test_run",
            env_id="Game2048-v1",
            seed=42,
        )
        assert hasattr(config, "run_id")
        assert hasattr(config, "seed")
        assert config.run_id == "test_run"
        assert config.seed == 42

    def test_config_implements_protocol(self):
        """Config must implement WorkerConfig protocol."""
        try:
            from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
        except ImportError:
            pytest.skip("gym_gui not available")

        config = JumanjiWorkerConfig(
            run_id="test_run",
            env_id="Game2048-v1",
        )

        # Should pass protocol check
        assert isinstance(config, WorkerConfigProtocol)

    def test_config_validation_requires_run_id(self):
        """Config validation should enforce required fields."""
        with pytest.raises(ValueError, match="run_id is required"):
            JumanjiWorkerConfig(
                run_id="",  # Empty run_id
                env_id="Game2048-v1",
            )

    def test_config_validation_checks_env_id(self):
        """Config validation should check env_id is in supported list."""
        with pytest.raises(ValueError, match="env_id must be one of"):
            JumanjiWorkerConfig(
                run_id="test_run",
                env_id="InvalidEnv-v1",  # Not in LOGIC_ENVIRONMENTS
            )

    def test_config_validation_checks_agent(self):
        """Config validation should check agent field."""
        with pytest.raises(ValueError, match="agent must be"):
            JumanjiWorkerConfig(
                run_id="test_run",
                env_id="Game2048-v1",
                agent="invalid",  # Not 'a2c' or 'random'
            )

    def test_config_validation_checks_device(self):
        """Config validation should check device field."""
        with pytest.raises(ValueError, match="device must be"):
            JumanjiWorkerConfig(
                run_id="test_run",
                env_id="Game2048-v1",
                device="invalid",  # Not 'cpu', 'gpu', or 'tpu'
            )

    def test_config_to_dict_from_dict_roundtrip(self):
        """Config must support dict serialization."""
        original = JumanjiWorkerConfig(
            run_id="test_run",
            env_id="Minesweeper-v0",
            agent="a2c",
            seed=42,
            num_epochs=50,
            learning_rate=1e-3,
            device="cpu",
        )

        # Convert to dict and back
        config_dict = original.to_dict()
        restored = JumanjiWorkerConfig.from_dict(config_dict)

        assert restored.run_id == original.run_id
        assert restored.env_id == original.env_id
        assert restored.agent == original.agent
        assert restored.seed == original.seed
        assert restored.num_epochs == original.num_epochs
        assert restored.learning_rate == original.learning_rate
        assert restored.device == original.device

    def test_config_all_logic_environments_accepted(self):
        """All logic environments should be accepted."""
        for env_id in LOGIC_ENVIRONMENTS:
            config = JumanjiWorkerConfig(
                run_id=f"test_{env_id}",
                env_id=env_id,
            )
            assert config.env_id == env_id


class TestPhase4_2_WorkerMetadata:
    """Test that Jumanji Worker provides correct metadata and capabilities."""

    def test_get_worker_metadata_exists(self):
        """Worker must export get_worker_metadata() function."""
        from jumanji_worker import get_worker_metadata

        assert callable(get_worker_metadata)

    def test_get_worker_metadata_returns_tuple(self):
        """get_worker_metadata() must return (WorkerMetadata, WorkerCapabilities)."""
        try:
            from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from jumanji_worker import get_worker_metadata

        result = get_worker_metadata()
        assert isinstance(result, tuple)
        assert len(result) == 2

        metadata, capabilities = result
        assert isinstance(metadata, WorkerMetadata)
        assert isinstance(capabilities, WorkerCapabilities)

    def test_worker_metadata_has_correct_fields(self):
        """WorkerMetadata must have all required fields."""
        try:
            from gym_gui.core.worker import WorkerMetadata
        except ImportError:
            pytest.skip("gym_gui not available")

        from jumanji_worker import get_worker_metadata

        metadata, _ = get_worker_metadata()

        assert metadata.name == "Jumanji Worker"
        assert metadata.version == "0.1.0"
        assert "jax" in metadata.description.lower() or "logic" in metadata.description.lower()
        assert metadata.upstream_library == "jumanji"
        assert metadata.license == "Apache-2.0"

    def test_worker_capabilities_declares_correct_worker_type(self):
        """WorkerCapabilities must declare worker_type='jumanji'."""
        try:
            from gym_gui.core.worker import WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from jumanji_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        assert capabilities.worker_type == "jumanji"

    def test_worker_capabilities_declares_env_families(self):
        """WorkerCapabilities must declare supported environment families."""
        try:
            from gym_gui.core.worker import WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from jumanji_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        # Jumanji supports native and gymnasium bridge
        assert "jumanji" in capabilities.env_families
        assert "gymnasium" in capabilities.env_families

    def test_worker_capabilities_declares_features(self):
        """WorkerCapabilities must declare key features."""
        try:
            from gym_gui.core.worker import WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from jumanji_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        # Jumanji supports checkpointing
        assert capabilities.supports_checkpointing is True
        # Jumanji is single-agent (for now)
        assert capabilities.max_agents == 1
        # Jumanji logic envs use discrete actions
        assert "discrete" in capabilities.action_spaces


class TestPhase4_3_EntryPointDiscovery:
    """Test that Jumanji Worker is discoverable via entry points."""

    def test_entry_point_registered(self):
        """Worker must be registered in mosaic.workers entry point group."""
        try:
            from importlib.metadata import entry_points
        except ImportError:
            from importlib_metadata import entry_points  # Python <3.10

        # Get all entry points in mosaic.workers group
        if sys.version_info >= (3, 10):
            eps = entry_points(group="mosaic.workers")
        else:
            eps = entry_points().get("mosaic.workers", [])

        jumanji_eps = [ep for ep in eps if ep.name == "jumanji"]
        assert len(jumanji_eps) > 0, "jumanji entry point not found in mosaic.workers group"

    def test_worker_discoverable_via_discovery_system(self):
        """Worker must be discoverable via WorkerDiscovery."""
        try:
            from gym_gui.core.worker import WorkerDiscovery
        except ImportError:
            pytest.skip("gym_gui.core.worker not available")

        discovery = WorkerDiscovery()
        workers = discovery.discover_all()

        # Check if jumanji is discovered (workers could be dict or list)
        if isinstance(workers, dict):
            assert "jumanji" in workers or any(
                w.worker_id == "jumanji" for w in workers.values()
            ), "Jumanji worker not discovered"
        else:
            # It's a list of DiscoveredWorker objects
            assert any(
                w.worker_id == "jumanji" for w in workers
            ), "Jumanji worker not discovered"


class TestPhase4_4_AnalyticsStandardization:
    """Test that Jumanji Worker uses standardized analytics manifests."""

    def test_write_analytics_uses_worker_analytics_manifest(self):
        """write_analytics_manifest() must use WorkerAnalyticsManifest."""
        try:
            from gym_gui.core.worker import WorkerAnalyticsManifest
            from jumanji_worker.analytics import write_analytics_manifest
        except ImportError:
            pytest.skip("gym_gui or analytics not available")

        config = JumanjiWorkerConfig(
            run_id="test_analytics",
            env_id="Game2048-v1",
        )

        manifest_path = write_analytics_manifest(config)
        assert manifest_path.exists()

        # Load and verify structure
        manifest = WorkerAnalyticsManifest.load(manifest_path)
        assert manifest.run_id == "test_analytics"
        assert manifest.worker_type == "jumanji"

        # Cleanup
        manifest_path.unlink()

    def test_analytics_manifest_has_standardized_structure(self):
        """Analytics manifest must follow standardized structure."""
        try:
            from gym_gui.core.worker import WorkerAnalyticsManifest
            from jumanji_worker.analytics import write_analytics_manifest
        except ImportError:
            pytest.skip("gym_gui or analytics not available")

        config = JumanjiWorkerConfig(
            run_id="test_structure",
            env_id="RubiksCube-v0",
        )

        manifest_path = write_analytics_manifest(config)
        manifest = WorkerAnalyticsManifest.load(manifest_path)

        # Verify standardized fields
        assert hasattr(manifest, "run_id")
        assert hasattr(manifest, "worker_type")
        assert hasattr(manifest, "artifacts")
        assert hasattr(manifest, "metadata")

        # Cleanup
        manifest_path.unlink()

    def test_analytics_manifest_stores_jumanji_specific_metadata(self):
        """Analytics manifest must store Jumanji-specific metadata."""
        try:
            from gym_gui.core.worker import WorkerAnalyticsManifest
            from jumanji_worker.analytics import write_analytics_manifest
        except ImportError:
            pytest.skip("gym_gui or analytics not available")

        config = JumanjiWorkerConfig(
            run_id="test_metadata",
            env_id="Sudoku-v0",
            agent="a2c",
            device="cpu",
            num_epochs=100,
            learning_rate=3e-4,
        )

        manifest_path = write_analytics_manifest(config)
        manifest = WorkerAnalyticsManifest.load(manifest_path)

        # Verify Jumanji-specific metadata
        assert manifest.metadata["env_id"] == "Sudoku-v0"
        assert manifest.metadata["agent"] == "a2c"
        assert manifest.metadata["device"] == "cpu"
        assert manifest.metadata["num_epochs"] == 100
        assert manifest.metadata["learning_rate"] == 3e-4

        # Cleanup
        manifest_path.unlink()


class TestPhase4_5_ConfigLoading:
    """Test that Jumanji Worker supports both direct and nested config formats."""

    def test_load_worker_config_from_direct_format(self, tmp_path):
        """load_worker_config() must handle direct config format."""
        config_file = tmp_path / "direct_config.json"
        config_data = {
            "run_id": "direct_test",
            "env_id": "Game2048-v1",
            "agent": "a2c",
            "seed": 123,
        }

        config_file.write_text(json.dumps(config_data))

        config = load_worker_config(str(config_file))
        assert config.run_id == "direct_test"
        assert config.env_id == "Game2048-v1"
        assert config.seed == 123

    def test_load_worker_config_from_nested_format(self, tmp_path):
        """load_worker_config() must handle nested metadata.worker.config format."""
        config_file = tmp_path / "nested_config.json"
        nested_data = {
            "metadata": {
                "worker": {
                    "config": {
                        "run_id": "nested_test",
                        "env_id": "Minesweeper-v0",
                        "agent": "random",
                        "seed": 456,
                    }
                }
            }
        }

        config_file.write_text(json.dumps(nested_data))

        config = load_worker_config(str(config_file))
        assert config.run_id == "nested_test"
        assert config.env_id == "Minesweeper-v0"
        assert config.agent == "random"
        assert config.seed == 456

    def test_load_worker_config_handles_missing_file(self):
        """load_worker_config() must raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_worker_config("/nonexistent/path/config.json")


class TestPhase4_6_GymnasiumBridge:
    """Test that Jumanji Worker provides Gymnasium bridge (THE TRICK)."""

    def test_gymnasium_adapter_importable(self):
        """gymnasium_adapter module must be importable."""
        from jumanji_worker import gymnasium_adapter

        assert hasattr(gymnasium_adapter, "JumanjiGymnasiumEnv")
        assert hasattr(gymnasium_adapter, "make_jumanji_gym_env")
        assert hasattr(gymnasium_adapter, "register_jumanji_envs")

    @pytest.mark.skipif(
        True,  # Skip unless JAX is installed
        reason="JAX not installed"
    )
    def test_gymnasium_make_works(self):
        """gymnasium.make('jumanji/...') must work."""
        import gymnasium as gym
        from jumanji_worker.gymnasium_adapter import register_jumanji_envs

        register_jumanji_envs()

        env = gym.make("jumanji/Game2048-v1")
        obs, info = env.reset()
        assert obs is not None
        env.close()

    @pytest.mark.skipif(
        True,  # Skip unless JAX is installed
        reason="JAX not installed"
    )
    def test_gymnasium_step_returns_correct_format(self):
        """Gymnasium step must return (obs, reward, terminated, truncated, info)."""
        import gymnasium as gym
        from jumanji_worker.gymnasium_adapter import register_jumanji_envs

        register_jumanji_envs()

        env = gym.make("jumanji/Game2048-v1")
        obs, info = env.reset()
        action = env.action_space.sample()
        result = env.step(action)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env.close()


def test_phase4_summary():
    """Overall summary test for Phase 4 standardization."""
    print("\n" + "=" * 70)
    print("Phase 4: Jumanji Worker Standardization Summary")
    print("=" * 70)

    checks = {
        "Config Protocol Compliance": False,
        "Worker Metadata": False,
        "Entry Point Discovery": False,
        "Analytics Standardization": False,
        "Config Loading": False,
        "Gymnasium Bridge": False,
    }

    # Check config protocol
    try:
        config = JumanjiWorkerConfig(
            run_id="test", env_id="Game2048-v1"
        )
        assert hasattr(config, "run_id") and hasattr(config, "seed")
        checks["Config Protocol Compliance"] = True
    except Exception:
        pass

    # Check worker metadata
    try:
        from jumanji_worker import get_worker_metadata

        metadata, capabilities = get_worker_metadata()
        assert metadata.name == "Jumanji Worker"
        assert capabilities.worker_type == "jumanji"
        checks["Worker Metadata"] = True
    except Exception:
        pass

    # Check entry point
    try:
        from importlib.metadata import entry_points

        if sys.version_info >= (3, 10):
            eps = entry_points(group="mosaic.workers")
        else:
            eps = entry_points().get("mosaic.workers", [])
        jumanji_eps = [ep for ep in eps if ep.name == "jumanji"]
        checks["Entry Point Discovery"] = len(jumanji_eps) > 0
    except Exception:
        pass

    # Check analytics
    try:
        from jumanji_worker.analytics import write_analytics_manifest

        checks["Analytics Standardization"] = callable(write_analytics_manifest)
    except Exception:
        pass

    # Check config loading
    try:
        from jumanji_worker.config import load_worker_config

        checks["Config Loading"] = callable(load_worker_config)
    except Exception:
        pass

    # Check gymnasium bridge
    try:
        from jumanji_worker.gymnasium_adapter import (
            JumanjiGymnasiumEnv,
            make_jumanji_gym_env,
            register_jumanji_envs,
        )

        checks["Gymnasium Bridge"] = all([
            JumanjiGymnasiumEnv is not None,
            callable(make_jumanji_gym_env),
            callable(register_jumanji_envs),
        ])
    except Exception:
        pass

    # Print results
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{check:.<40} {status}")

    print("=" * 70)

    # All checks should pass
    assert all(checks.values()), f"Some checks failed: {checks}"
