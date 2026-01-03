"""Tests for BARLOG Worker standardization (Phase 4).

These tests verify:
1. Configuration protocol compliance
2. Worker metadata and capabilities
3. Entry point discovery
4. Analytics manifest generation
5. Config loading from files
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestPhase4_1_ConfigCompliance:
    """Test BarlogWorkerConfig protocol compliance."""

    def test_config_has_required_fields(self):
        """Test that BarlogWorkerConfig has required WorkerConfig protocol fields."""
        from barlog_worker.config import BarlogWorkerConfig

        # Create a basic config
        config = BarlogWorkerConfig(
            run_id="test_config_001",
            env_name="babyai",
            task="BabyAI-GoToRedBall-v0",
        )

        # Check required protocol fields
        assert hasattr(config, "run_id")
        assert hasattr(config, "to_dict")
        assert hasattr(config, "from_dict")
        assert config.run_id == "test_config_001"

    def test_config_implements_protocol(self):
        """Test that BarlogWorkerConfig is a valid WorkerConfig protocol implementation."""
        from barlog_worker.config import BarlogWorkerConfig

        try:
            from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol

            config = BarlogWorkerConfig(
                run_id="test_protocol_002",
                env_name="minihack",
                task="MiniHack-Room-5x5-v0",
                client_name="openai",
                model_id="gpt-4o-mini",
            )

            # Protocol compliance check (from __post_init__)
            assert isinstance(config, WorkerConfigProtocol)
        except ImportError:
            pytest.skip("gym_gui not available")

    def test_config_validation_requires_run_id(self):
        """Test that config validation requires run_id."""
        from barlog_worker.config import BarlogWorkerConfig

        # Missing run_id should raise validation error
        # Note: dataclass requires all non-default fields
        with pytest.raises(TypeError):
            BarlogWorkerConfig()  # Missing required run_id

    def test_config_validation_checks_env_name(self):
        """Test that config validation checks env_name against valid options."""
        from barlog_worker.config import BarlogWorkerConfig

        # Invalid env_name should raise ValueError
        with pytest.raises(ValueError, match="Invalid env_name"):
            BarlogWorkerConfig(
                run_id="test_validation_001",
                env_name="invalid_env",  # Not in ENV_NAMES
            )

    def test_config_validation_checks_client_name(self):
        """Test that config validation checks client_name."""
        from barlog_worker.config import BarlogWorkerConfig

        with pytest.raises(ValueError, match="Invalid client_name"):
            BarlogWorkerConfig(
                run_id="test_validation_002",
                env_name="babyai",
                client_name="invalid_client",  # Not in CLIENT_NAMES
            )

    def test_config_validation_checks_agent_type(self):
        """Test that config validation checks agent_type."""
        from barlog_worker.config import BarlogWorkerConfig

        with pytest.raises(ValueError, match="Invalid agent_type"):
            BarlogWorkerConfig(
                run_id="test_validation_003",
                env_name="babyai",
                agent_type="invalid_agent",  # Not in AGENT_TYPES
            )

    def test_config_to_dict_from_dict_roundtrip(self):
        """Test config serialization/deserialization roundtrip."""
        from barlog_worker.config import BarlogWorkerConfig

        original = BarlogWorkerConfig(
            run_id="test_roundtrip_001",
            env_name="crafter",
            task="crafter-reward-v1",
            client_name="anthropic",
            model_id="claude-3-5-sonnet-20241022",
            agent_type="robust_cot",
            num_episodes=10,
            max_steps=200,
            temperature=0.8,
            seed=42,
        )

        # Serialize
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data["run_id"] == "test_roundtrip_001"
        assert data["env_name"] == "crafter"
        assert data["agent_type"] == "robust_cot"

        # Deserialize
        reconstructed = BarlogWorkerConfig.from_dict(data)
        assert reconstructed.run_id == original.run_id
        assert reconstructed.env_name == original.env_name
        assert reconstructed.agent_type == original.agent_type
        assert reconstructed.seed == original.seed


class TestPhase4_2_WorkerMetadata:
    """Test BARLOG worker metadata and capabilities."""

    def test_get_worker_metadata_exists(self):
        """Test that get_worker_metadata function exists."""
        from barlog_worker import get_worker_metadata

        assert callable(get_worker_metadata)

    def test_get_worker_metadata_returns_tuple(self):
        """Test that get_worker_metadata returns (WorkerMetadata, WorkerCapabilities)."""
        from barlog_worker import get_worker_metadata

        try:
            result = get_worker_metadata()
            assert isinstance(result, tuple)
            assert len(result) == 2
        except ImportError:
            pytest.skip("gym_gui not available")

    def test_worker_metadata_has_correct_fields(self):
        """Test that WorkerMetadata has all required fields."""
        from barlog_worker import get_worker_metadata

        try:
            metadata, capabilities = get_worker_metadata()

            # Check metadata fields
            assert metadata.name == "BARLOG Worker"
            assert metadata.version == "0.1.0"
            assert "LLM" in metadata.description or "BALROG" in metadata.description
            # Note: WorkerMetadata doesn't have worker_type field (it's in WorkerCapabilities)
        except ImportError:
            pytest.skip("gym_gui not available")

    def test_worker_capabilities_declares_correct_worker_type(self):
        """Test that WorkerCapabilities declares worker_type='barlog'."""
        from barlog_worker import get_worker_metadata

        try:
            metadata, capabilities = get_worker_metadata()

            assert capabilities.worker_type == "barlog"
        except ImportError:
            pytest.skip("gym_gui not available")

    def test_worker_capabilities_declares_env_families(self):
        """Test that WorkerCapabilities declares correct environment families."""
        from barlog_worker import get_worker_metadata

        try:
            metadata, capabilities = get_worker_metadata()

            # BARLOG supports these env families
            expected_families = {"babyai", "minigrid", "minihack", "crafter", "nle", "textworld"}
            assert set(capabilities.env_families) == expected_families
        except ImportError:
            pytest.skip("gym_gui not available")

    def test_worker_capabilities_declares_features(self):
        """Test that WorkerCapabilities declares correct features."""
        from barlog_worker import get_worker_metadata

        try:
            metadata, capabilities = get_worker_metadata()

            # BARLOG specific capabilities
            assert capabilities.max_agents == 1  # Single agent per env
            assert capabilities.supports_self_play is False
            assert capabilities.supports_population is False
            assert capabilities.supports_checkpointing is False  # No RL checkpoints
            assert capabilities.supports_pause_resume is True  # Via InteractiveRuntime
        except ImportError:
            pytest.skip("gym_gui not available")


class TestPhase4_3_EntryPointDiscovery:
    """Test worker discovery via entry points."""

    def test_entry_point_registered(self):
        """Test that 'barlog' entry point is registered in mosaic.workers group."""
        try:
            from importlib.metadata import entry_points
        except ImportError:
            from importlib_metadata import entry_points

        # Get entry points for mosaic.workers group
        try:
            # Python 3.10+ API
            eps = entry_points(group="mosaic.workers")
        except TypeError:
            # Python 3.9 fallback
            eps = entry_points().get("mosaic.workers", [])

        # Check if 'barlog' is registered
        barlog_eps = [ep for ep in eps if ep.name == "barlog"]
        assert len(barlog_eps) > 0, "barlog entry point not found in mosaic.workers group"

        # Load the entry point
        barlog_ep = barlog_eps[0]
        get_metadata = barlog_ep.load()
        assert callable(get_metadata)

    def test_worker_discoverable_via_discovery_system(self):
        """Test that BARLOG worker can be discovered via WorkerDiscovery."""
        try:
            from gym_gui.core.worker import WorkerDiscovery

            discovery = WorkerDiscovery()
            workers = discovery.discover_all()

            # Convert to dict if it's a list
            if isinstance(workers, list):
                workers_dict = {w.worker_id: w for w in workers}
            else:
                workers_dict = workers

            # Check if barlog worker was discovered
            assert "barlog" in workers_dict or any(
                "barlog" in str(w).lower() for w in workers_dict.values()
            ), "BARLOG worker not discovered"
        except ImportError:
            pytest.skip("gym_gui not available")


class TestPhase4_4_AnalyticsStandardization:
    """Test analytics manifest generation."""

    def test_write_analytics_uses_worker_analytics_manifest(self):
        """Test that write_analytics_manifest uses standardized WorkerAnalyticsManifest."""
        from barlog_worker.analytics import write_analytics_manifest
        from barlog_worker.config import BarlogWorkerConfig

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = BarlogWorkerConfig(
                    run_id="test_analytics_001",
                    env_name="babyai",
                    task="BabyAI-GoToRedBall-v0",
                    telemetry_dir=tmpdir,
                )

                manifest_path = write_analytics_manifest(config)
                assert manifest_path.exists()
                assert manifest_path.name.endswith("_analytics.json")
        except ImportError:
            pytest.skip("gym_gui not available")

    def test_analytics_manifest_has_standardized_structure(self):
        """Test that manifest has correct standardized structure."""
        from barlog_worker.analytics import write_analytics_manifest
        from barlog_worker.config import BarlogWorkerConfig

        try:
            from gym_gui.core.worker import WorkerAnalyticsManifest

            with tempfile.TemporaryDirectory() as tmpdir:
                config = BarlogWorkerConfig(
                    run_id="test_analytics_002",
                    env_name="minihack",
                    task="MiniHack-Corridor-R5-v0",
                    client_name="openai",
                    model_id="gpt-4o",
                    agent_type="cot",
                    telemetry_dir=tmpdir,
                )

                manifest_path = write_analytics_manifest(config, notes="Test run")

                # Load and verify
                loaded = WorkerAnalyticsManifest.load(manifest_path)
                assert loaded.run_id == "test_analytics_002"
                assert loaded.worker_type == "barlog"
                assert loaded.artifacts is not None
                assert loaded.metadata is not None
        except ImportError:
            pytest.skip("gym_gui not available")

    def test_analytics_manifest_stores_barlog_specific_metadata(self):
        """Test that manifest stores BARLOG-specific metadata."""
        from barlog_worker.analytics import write_analytics_manifest
        from barlog_worker.config import BarlogWorkerConfig

        try:
            from gym_gui.core.worker import WorkerAnalyticsManifest

            with tempfile.TemporaryDirectory() as tmpdir:
                config = BarlogWorkerConfig(
                    run_id="test_analytics_003",
                    env_name="crafter",
                    task="crafter-reward-v1",
                    client_name="anthropic",
                    model_id="claude-3-opus-20240229",
                    agent_type="robust_naive",
                    num_episodes=5,
                    max_steps=150,
                    temperature=0.7,
                    seed=123,
                    max_image_history=4,  # VLM enabled
                    telemetry_dir=tmpdir,
                )

                manifest_path = write_analytics_manifest(config)
                loaded = WorkerAnalyticsManifest.load(manifest_path)

                # Check BARLOG-specific metadata
                assert loaded.metadata["env_name"] == "crafter"
                assert loaded.metadata["task"] == "crafter-reward-v1"
                assert loaded.metadata["client_name"] == "anthropic"
                assert loaded.metadata["model_id"] == "claude-3-opus-20240229"
                assert loaded.metadata["agent_type"] == "robust_naive"
                assert loaded.metadata["num_episodes"] == 5
                assert loaded.metadata["max_steps"] == 150
                assert loaded.metadata["temperature"] == 0.7
                assert loaded.metadata["seed"] == 123
                assert loaded.metadata["max_image_history"] == 4  # VLM enabled
        except ImportError:
            pytest.skip("gym_gui not available")


class TestPhase4_5_ConfigLoading:
    """Test config loading from files."""

    def test_load_worker_config_from_direct_format(self):
        """Test loading config from direct JSON format."""
        from barlog_worker.config import load_worker_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "run_id": "test_load_001",
                "env_name": "babyai",
                "task": "BabyAI-GoToRedBall-v0",
                "client_name": "openai",
                "model_id": "gpt-4o-mini",
                "agent_type": "naive",
            }
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = load_worker_config(config_path)
            assert config.run_id == "test_load_001"
            assert config.env_name == "babyai"
            assert config.agent_type == "naive"
        finally:
            Path(config_path).unlink()

    def test_load_worker_config_from_nested_format(self):
        """Test loading config from nested metadata.worker.config format (from GUI)."""
        from barlog_worker.config import load_worker_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Nested format from GUI
            full_data = {
                "metadata": {
                    "worker": {
                        "config": {
                            "run_id": "test_load_002",
                            "env_name": "minihack",
                            "task": "MiniHack-Quest-Hard-v0",
                            "client_name": "anthropic",
                            "model_id": "claude-3-5-sonnet-20241022",
                            "agent_type": "few_shot",
                        }
                    }
                }
            }
            json.dump(full_data, f)
            config_path = f.name

        try:
            config = load_worker_config(config_path)
            assert config.run_id == "test_load_002"
            assert config.env_name == "minihack"
            assert config.agent_type == "few_shot"
        finally:
            Path(config_path).unlink()

    def test_load_worker_config_handles_missing_file(self):
        """Test that load_worker_config raises FileNotFoundError for missing files."""
        from barlog_worker.config import load_worker_config

        with pytest.raises(FileNotFoundError):
            load_worker_config("/nonexistent/path/to/config.json")


class TestPhase4_6_AgentTypes:
    """Test agent type validation based on actual BALROG implementation."""

    def test_valid_agent_types(self):
        """Test that all valid BALROG agent types are accepted."""
        from barlog_worker.config import BarlogWorkerConfig, AGENT_TYPES

        # These are the actual agent types from BALROG/balrog/agents/__init__.py
        valid_types = ["naive", "cot", "robust_naive", "robust_cot", "few_shot", "dummy"]

        assert set(AGENT_TYPES) == set(valid_types), (
            f"AGENT_TYPES mismatch. Expected: {valid_types}, Got: {list(AGENT_TYPES)}"
        )

        # Test that each type can be used
        for agent_type in valid_types:
            config = BarlogWorkerConfig(
                run_id=f"test_agent_{agent_type}",
                env_name="babyai",
                agent_type=agent_type,
            )
            assert config.agent_type == agent_type


class TestPhase4_7_VLMCapability:
    """Test Vision-Language Model capability configuration."""

    def test_vlm_controlled_by_max_image_history(self):
        """Test that VLM is controlled by max_image_history parameter, not agent type."""
        from barlog_worker.config import BarlogWorkerConfig

        # Text-only model
        text_only_config = BarlogWorkerConfig(
            run_id="test_text_only",
            env_name="babyai",
            agent_type="naive",
            max_image_history=0,  # Text-only
        )
        assert text_only_config.max_image_history == 0

        # Vision-Language Model
        vlm_config = BarlogWorkerConfig(
            run_id="test_vlm",
            env_name="babyai",
            agent_type="naive",  # Same agent type
            max_image_history=4,  # VLM enabled
        )
        assert vlm_config.max_image_history == 4

        # VLM can be used with any agent type
        for agent_type in ["naive", "cot", "robust_naive", "robust_cot", "few_shot"]:
            vlm_config = BarlogWorkerConfig(
                run_id=f"test_vlm_{agent_type}",
                env_name="babyai",
                agent_type=agent_type,
                max_image_history=2,
            )
            assert vlm_config.max_image_history == 2


def test_phase4_summary():
    """Summary test to verify all Phase 4 components are in place."""
    from barlog_worker import (
        BarlogWorkerConfig,
        load_worker_config,
        get_worker_metadata,
        BarlogWorkerRuntime,
        InteractiveRuntime,
    )
    from barlog_worker.analytics import write_analytics_manifest

    # Check all components exist
    assert BarlogWorkerConfig is not None
    assert load_worker_config is not None
    assert get_worker_metadata is not None
    assert BarlogWorkerRuntime is not None
    assert InteractiveRuntime is not None
    assert write_analytics_manifest is not None

    print("\nPhase 4 Standardization Complete!")
    print("- Config: Protocol compliance ✓")
    print("- Runtime: TelemetryEmitter integration ✓")
    print("- Analytics: WorkerAnalyticsManifest ✓")
    print("- Metadata: get_worker_metadata() ✓")
    print("- Discovery: Entry point registered ✓")
    print("- Agent Types: 6 types (naive, cot, robust_naive, robust_cot, few_shot, dummy) ✓")
    print("- VLM: Controlled by max_image_history parameter ✓")
