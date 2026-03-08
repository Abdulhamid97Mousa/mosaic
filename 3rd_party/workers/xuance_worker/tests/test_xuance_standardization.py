"""Comprehensive tests for XuanCe Worker standardization (Phase 4).

This test suite verifies that XuanCe Worker implements all required standardization
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

from xuance_worker.config import XuanCeWorkerConfig, load_worker_config

if TYPE_CHECKING:
    from gym_gui.core.worker import WorkerConfig, WorkerMetadata, WorkerCapabilities


class TestPhase4_1_ConfigCompliance:
    """Test that XuanCeWorkerConfig implements WorkerConfig protocol."""

    def test_config_has_required_fields(self):
        """Config must have run_id and seed fields."""
        config = XuanCeWorkerConfig(
            run_id="test_run",
            method="dqn",
            env="classic_control",
            env_id="CartPole-v1",
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

        config = XuanCeWorkerConfig(
            run_id="test_run",
            method="dqn",
            env="classic_control",
            env_id="CartPole-v1",
        )

        # Should pass protocol check
        assert isinstance(config, WorkerConfigProtocol)

    def test_config_validation_requires_run_id(self):
        """Config validation should enforce required fields."""
        with pytest.raises(ValueError, match="run_id is required"):
            XuanCeWorkerConfig(
                run_id="",  # Empty run_id
                method="dqn",
                env="classic_control",
                env_id="CartPole-v1",
            )

    def test_config_validation_checks_method(self):
        """Config validation should check method field."""
        with pytest.raises(ValueError, match="method is required"):
            XuanCeWorkerConfig(
                run_id="test_run",
                method="",  # Empty method
                env="classic_control",
                env_id="CartPole-v1",
            )

    def test_config_validation_checks_env(self):
        """Config validation should check env field."""
        with pytest.raises(ValueError, match="env is required"):
            XuanCeWorkerConfig(
                run_id="test_run",
                method="dqn",
                env="",  # Empty env
                env_id="CartPole-v1",
            )

    def test_config_validation_checks_env_id(self):
        """Config validation should check env_id field."""
        with pytest.raises(ValueError, match="env_id is required"):
            XuanCeWorkerConfig(
                run_id="test_run",
                method="dqn",
                env="classic_control",
                env_id="",  # Empty env_id
            )

    def test_config_to_dict_from_dict_roundtrip(self):
        """Config must support dict serialization."""
        original = XuanCeWorkerConfig(
            run_id="test_run",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
            seed=42,
            running_steps=100000,
        )

        # Convert to dict and back
        config_dict = original.to_dict()
        restored = XuanCeWorkerConfig.from_dict(config_dict)

        assert restored.run_id == original.run_id
        assert restored.method == original.method
        assert restored.env == original.env
        assert restored.env_id == original.env_id
        assert restored.seed == original.seed
        assert restored.running_steps == original.running_steps


class TestPhase4_2_WorkerMetadata:
    """Test that XuanCe Worker provides correct metadata and capabilities."""

    def test_get_worker_metadata_exists(self):
        """Worker must export get_worker_metadata() function."""
        from xuance_worker import get_worker_metadata

        assert callable(get_worker_metadata)

    def test_get_worker_metadata_returns_tuple(self):
        """get_worker_metadata() must return (WorkerMetadata, WorkerCapabilities)."""
        try:
            from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from xuance_worker import get_worker_metadata

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

        from xuance_worker import get_worker_metadata

        metadata, _ = get_worker_metadata()

        assert metadata.name == "XuanCe Worker"
        assert metadata.version == "0.1.0"
        assert "comprehensive" in metadata.description.lower() or "46+" in metadata.description
        assert metadata.upstream_library == "xuance"
        assert metadata.license == "MIT"

    def test_worker_capabilities_declares_correct_worker_type(self):
        """WorkerCapabilities must declare worker_type='xuance'."""
        try:
            from gym_gui.core.worker import WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from xuance_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        assert capabilities.worker_type == "xuance"

    def test_worker_capabilities_declares_env_families(self):
        """WorkerCapabilities must declare supported environment families."""
        try:
            from gym_gui.core.worker import WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from xuance_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        # XuanCe supports many env families
        assert "gymnasium" in capabilities.env_families
        assert "classic_control" in capabilities.env_families
        # Should support multi-agent envs
        assert "pettingzoo" in capabilities.env_families or "mpe" in capabilities.env_families

    def test_worker_capabilities_declares_features(self):
        """WorkerCapabilities must declare key features."""
        try:
            from gym_gui.core.worker import WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from xuance_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        # XuanCe supports checkpointing
        assert capabilities.supports_checkpointing is True
        # XuanCe supports multi-agent
        assert capabilities.max_agents > 1
        # XuanCe supports continuous actions
        assert "continuous" in capabilities.action_spaces


class TestPhase4_3_EntryPointDiscovery:
    """Test that XuanCe Worker is discoverable via entry points."""

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

        xuance_eps = [ep for ep in eps if ep.name == "xuance"]
        assert len(xuance_eps) > 0, "xuance entry point not found in mosaic.workers group"

    def test_worker_discoverable_via_discovery_system(self):
        """Worker must be discoverable via WorkerDiscovery."""
        try:
            from gym_gui.core.worker import WorkerDiscovery
        except ImportError:
            pytest.skip("gym_gui.core.worker not available")

        discovery = WorkerDiscovery()
        workers = discovery.discover_all()

        # Check if xuance is discovered (workers could be dict or list)
        if isinstance(workers, dict):
            assert "xuance" in workers or any(
                w.worker_id == "xuance" for w in workers.values()
            ), "XuanCe worker not discovered"
        else:
            # It's a list of DiscoveredWorker objects
            assert any(
                w.worker_id == "xuance" for w in workers
            ), "XuanCe worker not discovered"


class TestPhase4_4_AnalyticsStandardization:
    """Test that XuanCe Worker uses standardized analytics manifests."""

    def test_write_analytics_uses_worker_analytics_manifest(self):
        """write_analytics_manifest() must use WorkerAnalyticsManifest."""
        try:
            from gym_gui.core.worker import WorkerAnalyticsManifest
            from xuance_worker.analytics import write_analytics_manifest
        except ImportError:
            pytest.skip("gym_gui or analytics not available")

        config = XuanCeWorkerConfig(
            run_id="test_analytics",
            method="dqn",
            env="classic_control",
            env_id="CartPole-v1",
        )

        manifest_path = write_analytics_manifest(config)
        assert manifest_path.exists()

        # Load and verify structure
        manifest = WorkerAnalyticsManifest.load(manifest_path)
        assert manifest.run_id == "test_analytics"
        assert manifest.worker_type == "xuance"

        # Cleanup
        manifest_path.unlink()

    def test_analytics_manifest_has_standardized_structure(self):
        """Analytics manifest must follow standardized structure."""
        try:
            from gym_gui.core.worker import WorkerAnalyticsManifest
            from xuance_worker.analytics import write_analytics_manifest
        except ImportError:
            pytest.skip("gym_gui or analytics not available")

        config = XuanCeWorkerConfig(
            run_id="test_structure",
            method="ppo",
            env="classic_control",
            env_id="CartPole-v1",
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

    def test_analytics_manifest_stores_xuance_specific_metadata(self):
        """Analytics manifest must store XuanCe-specific metadata."""
        try:
            from gym_gui.core.worker import WorkerAnalyticsManifest
            from xuance_worker.analytics import write_analytics_manifest
        except ImportError:
            pytest.skip("gym_gui or analytics not available")

        config = XuanCeWorkerConfig(
            run_id="test_metadata",
            method="sac",
            env="mujoco",
            env_id="HalfCheetah-v4",
            dl_toolbox="torch",
            running_steps=500000,
        )

        manifest_path = write_analytics_manifest(config)
        manifest = WorkerAnalyticsManifest.load(manifest_path)

        # Verify XuanCe-specific metadata
        assert manifest.metadata["method"] == "sac"
        assert manifest.metadata["env"] == "mujoco"
        assert manifest.metadata["env_id"] == "HalfCheetah-v4"
        assert manifest.metadata["dl_toolbox"] == "torch"
        assert manifest.metadata["running_steps"] == 500000

        # Cleanup
        manifest_path.unlink()


class TestPhase4_5_ConfigLoading:
    """Test that XuanCe Worker supports both direct and nested config formats."""

    def test_load_worker_config_from_direct_format(self, tmp_path):
        """load_worker_config() must handle direct config format."""
        config_file = tmp_path / "direct_config.json"
        config_data = {
            "run_id": "direct_test",
            "method": "dqn",
            "env": "classic_control",
            "env_id": "CartPole-v1",
            "seed": 123,
        }

        config_file.write_text(json.dumps(config_data))

        config = load_worker_config(str(config_file))
        assert config.run_id == "direct_test"
        assert config.method == "dqn"
        assert config.seed == 123

    def test_load_worker_config_from_nested_format(self, tmp_path):
        """load_worker_config() must handle nested metadata.worker.config format."""
        config_file = tmp_path / "nested_config.json"
        nested_data = {
            "metadata": {
                "worker": {
                    "config": {
                        "run_id": "nested_test",
                        "method": "ppo",
                        "env": "atari",
                        "env_id": "Pong-v5",
                        "seed": 456,
                    }
                }
            }
        }

        config_file.write_text(json.dumps(nested_data))

        config = load_worker_config(str(config_file))
        assert config.run_id == "nested_test"
        assert config.method == "ppo"
        assert config.env == "atari"
        assert config.seed == 456

    def test_load_worker_config_handles_missing_file(self):
        """load_worker_config() must raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_worker_config("/nonexistent/path/config.json")


class TestPhase4_6_MultiAgentSupport:
    """Test that XuanCe Worker declares multi-agent capabilities."""

    def test_supports_multiple_paradigms(self):
        """XuanCe should support sequential, parameter_sharing, and independent paradigms."""
        try:
            from gym_gui.core.worker import WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from xuance_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        # XuanCe supports multiple paradigms
        assert "sequential" in capabilities.supported_paradigms
        # Should support at least one multi-agent paradigm
        assert (
            "parameter_sharing" in capabilities.supported_paradigms
            or "independent" in capabilities.supported_paradigms
        )

    def test_max_agents_greater_than_one(self):
        """XuanCe should support multi-agent scenarios."""
        try:
            from gym_gui.core.worker import WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from xuance_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        assert capabilities.max_agents > 1, "XuanCe should support multi-agent"


def test_phase4_summary():
    """Overall summary test for Phase 4 standardization."""
    print("\n" + "=" * 70)
    print("Phase 4: XuanCe Worker Standardization Summary")
    print("=" * 70)

    checks = {
        "Config Protocol Compliance": False,
        "Worker Metadata": False,
        "Entry Point Discovery": False,
        "Analytics Standardization": False,
        "Config Loading": False,
        "Multi-Agent Support": False,
    }

    # Check config protocol
    try:
        config = XuanCeWorkerConfig(
            run_id="test", method="dqn", env="classic_control", env_id="CartPole-v1"
        )
        assert hasattr(config, "run_id") and hasattr(config, "seed")
        checks["Config Protocol Compliance"] = True
    except Exception:
        pass

    # Check worker metadata
    try:
        from xuance_worker import get_worker_metadata

        metadata, capabilities = get_worker_metadata()
        assert metadata.name == "XuanCe Worker"
        assert capabilities.worker_type == "xuance"
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
        xuance_eps = [ep for ep in eps if ep.name == "xuance"]
        checks["Entry Point Discovery"] = len(xuance_eps) > 0
    except Exception:
        pass

    # Check analytics
    try:
        from xuance_worker.analytics import write_analytics_manifest

        checks["Analytics Standardization"] = callable(write_analytics_manifest)
    except Exception:
        pass

    # Check config loading
    try:
        from xuance_worker.config import load_worker_config

        checks["Config Loading"] = callable(load_worker_config)
    except Exception:
        pass

    # Check multi-agent support
    try:
        from xuance_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()
        checks["Multi-Agent Support"] = capabilities.max_agents > 1
    except Exception:
        pass

    # Print results
    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check:.<40} {status}")

    print("=" * 70)

    # All checks should pass
    assert all(checks.values()), f"Some checks failed: {checks}"
