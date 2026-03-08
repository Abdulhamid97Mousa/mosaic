"""Phase 3 Standardization Tests for Ray Worker.

This test suite validates that the Ray worker has been successfully
standardized according to the MOSAIC worker architecture plan.

Test Coverage:
1. Protocol compliance (WorkerConfig, WorkerRuntime)
2. TelemetryEmitter integration with log constants
3. Standardized analytics manifest (WorkerAnalyticsManifest)
4. Worker discovery (entry points, get_worker_metadata)
5. Log constants (LOG446-460)
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict

import pytest


class TestPhase3_1_ProtocolCompliance:
    """Phase 3.1-3.2: Test that RayWorkerConfig implements WorkerConfig protocol."""

    def test_config_has_required_protocol_fields(self):
        """Test that RayWorkerConfig has all required protocol fields."""
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            PolicyConfiguration,
            PettingZooAPIType,
        )

        config = RayWorkerConfig(
            run_id="test_protocol_001",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
                api_type=PettingZooAPIType.PARALLEL,
            ),
            policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
            seed=42,
        )

        # Protocol requires: run_id, seed, to_dict(), from_dict()
        assert hasattr(config, "run_id")
        assert hasattr(config, "seed")
        assert hasattr(config, "to_dict")
        assert callable(config.to_dict)
        assert hasattr(RayWorkerConfig, "from_dict")
        assert callable(RayWorkerConfig.from_dict)

        assert config.run_id == "test_protocol_001"
        assert config.seed == 42

    def test_config_implements_protocol(self):
        """Test that RayWorkerConfig is a valid WorkerConfig protocol implementation."""
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            PolicyConfiguration,
            PettingZooAPIType,
        )
        from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol

        config = RayWorkerConfig(
            run_id="test_protocol_002",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
                api_type=PettingZooAPIType.PARALLEL,
            ),
            policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
        )

        # Protocol compliance check (from __post_init__)
        assert isinstance(config, WorkerConfigProtocol)

    def test_config_validation_requires_run_id(self):
        """Test that __post_init__ validation requires run_id."""
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            PolicyConfiguration,
            PettingZooAPIType,
        )

        with pytest.raises(ValueError, match="run_id"):
            RayWorkerConfig(
                run_id="",  # Empty run_id should fail
                environment=EnvironmentConfig(
                    family="sisl",
                    env_id="waterworld_v4",
                    api_type=PettingZooAPIType.PARALLEL,
                ),
                policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
            )

    def test_config_validation_requires_environment_fields(self):
        """Test that __post_init__ validation requires environment.family and env_id."""
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            PolicyConfiguration,
            PettingZooAPIType,
        )

        with pytest.raises(ValueError, match="environment.family"):
            RayWorkerConfig(
                run_id="test_protocol_003",
                environment=EnvironmentConfig(
                    family="",  # Empty family should fail
                    env_id="waterworld_v4",
                    api_type=PettingZooAPIType.PARALLEL,
                ),
                policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
            )

        with pytest.raises(ValueError, match="environment.env_id"):
            RayWorkerConfig(
                run_id="test_protocol_004",
                environment=EnvironmentConfig(
                    family="sisl",
                    env_id="",  # Empty env_id should fail
                    api_type=PettingZooAPIType.PARALLEL,
                ),
                policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
            )

    def test_config_to_dict_from_dict_roundtrip(self):
        """Test that config can be serialized and deserialized."""
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            PolicyConfiguration,
            PettingZooAPIType,
        )

        original = RayWorkerConfig(
            run_id="test_protocol_005",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
                api_type=PettingZooAPIType.PARALLEL,
            ),
            policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
            seed=123,
        )

        # Serialize
        data = original.to_dict()
        assert isinstance(data, dict)
        assert data["run_id"] == "test_protocol_005"
        assert data["seed"] == 123

        # Deserialize
        restored = RayWorkerConfig.from_dict(data)
        assert restored.run_id == original.run_id
        assert restored.seed == original.seed
        assert restored.policy_configuration == original.policy_configuration


class TestPhase3_3_TelemetryIntegration:
    """Phase 3.3: Test TelemetryEmitter integration with log constants."""

    def test_runtime_creates_telemetry_emitter(self):
        """Test that RayWorkerRuntime creates a TelemetryEmitter."""
        from ray_worker.runtime import RayWorkerRuntime
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            PolicyConfiguration,
            PettingZooAPIType,
        )
        from gym_gui.core.worker import TelemetryEmitter

        config = RayWorkerConfig(
            run_id="test_telemetry_001",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
                api_type=PettingZooAPIType.PARALLEL,
            ),
            policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
        )

        runtime = RayWorkerRuntime(config)

        assert hasattr(runtime, "_emitter")
        assert isinstance(runtime._emitter, TelemetryEmitter)
        assert runtime._emitter._run_id == "test_telemetry_001"

    def test_lifecycle_events_use_log_constants(self):
        """Test that lifecycle events use the correct log constants."""
        from gym_gui.logging_config.log_constants import (
            LOG_WORKER_RAY_RUNTIME_STARTED,
            LOG_WORKER_RAY_RUNTIME_COMPLETED,
            LOG_WORKER_RAY_RUNTIME_FAILED,
            LOG_WORKER_RAY_HEARTBEAT,
        )

        # Verify log constants exist and have correct structure
        assert LOG_WORKER_RAY_RUNTIME_STARTED.code == "LOG446"
        assert LOG_WORKER_RAY_RUNTIME_STARTED.component == "Worker"
        assert LOG_WORKER_RAY_RUNTIME_STARTED.subcomponent == "RayRuntime"
        assert "lifecycle" in LOG_WORKER_RAY_RUNTIME_STARTED.tags

        assert LOG_WORKER_RAY_RUNTIME_COMPLETED.code == "LOG447"
        assert LOG_WORKER_RAY_RUNTIME_FAILED.code == "LOG448"
        assert LOG_WORKER_RAY_HEARTBEAT.code == "LOG459"

    def test_emitter_emits_jsonl_to_stdout(self, monkeypatch):
        """Test that TelemetryEmitter emits JSONL events to stdout."""
        from gym_gui.core.worker import TelemetryEmitter
        from gym_gui.logging_config.log_constants import LOG_WORKER_RAY_RUNTIME_STARTED

        # Capture stdout
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        emitter = TelemetryEmitter(run_id="test_telemetry_002")
        emitter.run_started(
            {"worker_type": "ray", "algo": "PPO"},
            constant=LOG_WORKER_RAY_RUNTIME_STARTED,
        )

        # Check JSONL output
        output = captured_output.getvalue()
        lines = [line for line in output.strip().split("\n") if line]
        assert len(lines) == 1

        event = json.loads(lines[0])
        assert event["event"] == "run_started"
        assert event["run_id"] == "test_telemetry_002"
        # worker_type is in payload, not top-level
        assert event["payload"]["worker_type"] == "ray"

    def test_log_constants_range_446_to_460(self):
        """Test that all Ray worker log constants are defined in LOG446-460 range."""
        from gym_gui.logging_config.log_constants import (
            LOG_WORKER_RAY_RUNTIME_STARTED,
            LOG_WORKER_RAY_RUNTIME_COMPLETED,
            LOG_WORKER_RAY_RUNTIME_FAILED,
            LOG_WORKER_RAY_HEARTBEAT,
            LOG_WORKER_RAY_TENSORBOARD_ENABLED,
            LOG_WORKER_RAY_WANDB_ENABLED,
            LOG_WORKER_RAY_CHECKPOINT_SAVED,
            LOG_WORKER_RAY_ANALYTICS_MANIFEST_CREATED,
        )

        constants = [
            LOG_WORKER_RAY_RUNTIME_STARTED,
            LOG_WORKER_RAY_RUNTIME_COMPLETED,
            LOG_WORKER_RAY_RUNTIME_FAILED,
            LOG_WORKER_RAY_HEARTBEAT,
            LOG_WORKER_RAY_TENSORBOARD_ENABLED,
            LOG_WORKER_RAY_WANDB_ENABLED,
            LOG_WORKER_RAY_CHECKPOINT_SAVED,
            LOG_WORKER_RAY_ANALYTICS_MANIFEST_CREATED,
        ]

        # All should be in LOG446-460 range
        for const in constants:
            code_num = int(const.code.replace("LOG", ""))
            assert 446 <= code_num <= 460, f"{const.code} outside LOG446-460 range"


class TestPhase3_4_AnalyticsStandardization:
    """Phase 3.4: Test standardized analytics manifest using WorkerAnalyticsManifest."""

    def test_write_analytics_uses_worker_analytics_manifest(self, tmp_path):
        """Test that write_analytics_manifest uses WorkerAnalyticsManifest."""
        from ray_worker.analytics import write_analytics_manifest
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            PolicyConfiguration,
            PettingZooAPIType,
        )
        from gym_gui.core.worker import WorkerAnalyticsManifest

        config = RayWorkerConfig(
            run_id="test_analytics_001",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
                api_type=PettingZooAPIType.PARALLEL,
            ),
            policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
            output_dir=str(tmp_path),
        )

        manifest_path = write_analytics_manifest(config)

        # Verify it can be loaded as WorkerAnalyticsManifest
        loaded = WorkerAnalyticsManifest.load(manifest_path)
        assert isinstance(loaded, WorkerAnalyticsManifest)
        assert loaded.run_id == "test_analytics_001"
        assert loaded.worker_type == "ray"

    def test_analytics_manifest_has_standardized_artifacts(self, tmp_path):
        """Test that manifest uses standardized artifact metadata."""
        from ray_worker.analytics import write_analytics_manifest
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            PolicyConfiguration,
            PettingZooAPIType,
        )
        from gym_gui.core.worker import WorkerAnalyticsManifest

        config = RayWorkerConfig(
            run_id="test_analytics_002",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
                api_type=PettingZooAPIType.PARALLEL,
            ),
            policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
            output_dir=str(tmp_path),
            tensorboard=True,
            wandb=True,  # Enable WandB
        )

        manifest_path = write_analytics_manifest(
            config,
            wandb_run_path="entity/project/run123",
            num_agents=3,
        )

        loaded = WorkerAnalyticsManifest.load(manifest_path)

        # Check artifacts structure
        assert loaded.artifacts is not None
        assert loaded.artifacts.tensorboard is not None
        assert loaded.artifacts.tensorboard.enabled is True
        assert loaded.artifacts.tensorboard.log_dir == "tensorboard"

        assert loaded.artifacts.wandb is not None
        assert loaded.artifacts.wandb.enabled is True
        assert loaded.artifacts.wandb.project == "ray-marl"

        assert loaded.artifacts.checkpoints is not None
        assert loaded.artifacts.checkpoints.directory == "checkpoints"
        assert loaded.artifacts.checkpoints.format == "ray_rllib"

    def test_analytics_manifest_stores_ray_specific_metadata(self, tmp_path):
        """Test that Ray-specific metadata is stored in metadata dict."""
        from ray_worker.analytics import write_analytics_manifest
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            PolicyConfiguration,
            PettingZooAPIType,
        )
        from gym_gui.core.worker import WorkerAnalyticsManifest

        config = RayWorkerConfig(
            run_id="test_analytics_003",
            environment=EnvironmentConfig(
                family="mpe",
                env_id="simple_spread_v3",
                api_type=PettingZooAPIType.PARALLEL,
            ),
            policy_configuration=PolicyConfiguration.INDEPENDENT,
            output_dir=str(tmp_path),
            seed=999,
        )

        manifest_path = write_analytics_manifest(config, num_agents=5)

        loaded = WorkerAnalyticsManifest.load(manifest_path)

        # Check Ray-specific metadata
        assert loaded.metadata["paradigm"] == "independent"
        assert loaded.metadata["algorithm"] == "PPO"
        assert loaded.metadata["env_family"] == "mpe"
        assert loaded.metadata["env_id"] == "simple_spread_v3"
        assert loaded.metadata["num_agents"] == 5
        assert loaded.metadata["seed"] == 999


class TestPhase3_5_WorkerMetadata:
    """Phase 3.5: Test get_worker_metadata() function."""

    def test_get_worker_metadata_exists(self):
        """Test that get_worker_metadata() function exists."""
        from ray_worker import get_worker_metadata

        assert callable(get_worker_metadata)

    def test_get_worker_metadata_returns_tuple(self):
        """Test that get_worker_metadata() returns (WorkerMetadata, WorkerCapabilities)."""
        from ray_worker import get_worker_metadata
        from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities

        result = get_worker_metadata()

        assert isinstance(result, tuple)
        assert len(result) == 2

        metadata, capabilities = result
        assert isinstance(metadata, WorkerMetadata)
        assert isinstance(capabilities, WorkerCapabilities)

    def test_worker_metadata_has_correct_fields(self):
        """Test that WorkerMetadata has correct values."""
        from ray_worker import get_worker_metadata

        metadata, _ = get_worker_metadata()

        assert metadata.name == "Ray RLlib Worker"
        assert "ray" in metadata.upstream_library.lower()
        assert metadata.license == "Apache-2.0"

    def test_worker_capabilities_declares_paradigms(self):
        """Test that WorkerCapabilities declares supported paradigms."""
        from ray_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        assert capabilities.worker_type == "ray"
        assert "parameter_sharing" in capabilities.supported_paradigms
        assert "independent" in capabilities.supported_paradigms
        assert "self_play" in capabilities.supported_paradigms
        assert "shared_value_function" in capabilities.supported_paradigms

    def test_worker_capabilities_declares_env_families(self):
        """Test that WorkerCapabilities declares env families (no 'pettingzoo')."""
        from ray_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        # Should have PettingZoo families, not "pettingzoo" itself
        assert "sisl" in capabilities.env_families
        assert "classic" in capabilities.env_families
        assert "butterfly" in capabilities.env_families
        assert "mpe" in capabilities.env_families
        # Should NOT have "pettingzoo" (it's the library, not a family)
        assert "pettingzoo" not in capabilities.env_families

    def test_worker_capabilities_declares_features(self):
        """Test that WorkerCapabilities declares key features."""
        from ray_worker import get_worker_metadata

        _, capabilities = get_worker_metadata()

        assert capabilities.supports_self_play is True
        assert capabilities.supports_checkpointing is True
        assert capabilities.supports_pause_resume is True
        assert capabilities.max_agents == 100


class TestPhase3_6_EntryPointDiscovery:
    """Phase 3.6: Test entry point registration and discovery."""

    def test_entry_point_registered(self):
        """Test that Ray worker entry point is registered."""
        try:
            from importlib.metadata import entry_points
        except ImportError:
            from importlib_metadata import entry_points

        # Get all mosaic.workers entry points
        eps = entry_points()
        if hasattr(eps, "select"):
            # Python 3.10+ API
            worker_eps = eps.select(group="mosaic.workers")
        else:
            # Python 3.9 API
            worker_eps = eps.get("mosaic.workers", [])

        # Find ray entry point
        ray_ep = None
        for ep in worker_eps:
            if ep.name == "ray":
                ray_ep = ep
                break

        assert ray_ep is not None, "Ray worker entry point not found"
        assert ray_ep.value == "ray_worker:get_worker_metadata"

    def test_worker_discoverable_via_discovery_system(self):
        """Test that Ray worker is discoverable via WorkerDiscovery."""
        from gym_gui.core.worker import WorkerDiscovery

        discovery = WorkerDiscovery()
        workers = discovery.discover_all()

        # discover_all() returns a list of DiscoveredWorker
        # Convert to dict for easy lookup
        if isinstance(workers, list):
            workers_dict = {w.worker_id: w for w in workers}
        else:
            workers_dict = workers

        # Find ray worker
        ray_worker = workers_dict.get("ray")

        assert ray_worker is not None, f"Ray worker not discovered. Found: {list(workers_dict.keys())}"
        assert ray_worker.metadata.name == "Ray RLlib Worker"
        assert ray_worker.capabilities.worker_type == "ray"


class TestPhase3_Integration:
    """Integration tests for Phase 3 - all components working together."""

    def test_end_to_end_config_telemetry_analytics(self, tmp_path, monkeypatch):
        """Test full flow: config → telemetry → analytics."""
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            PolicyConfiguration,
            PettingZooAPIType,
        )
        from ray_worker.analytics import write_analytics_manifest
        from gym_gui.core.worker import TelemetryEmitter, WorkerAnalyticsManifest
        from gym_gui.logging_config.log_constants import LOG_WORKER_RAY_RUNTIME_STARTED

        # 1. Create config (protocol compliance)
        config = RayWorkerConfig(
            run_id="test_integration_001",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
                api_type=PettingZooAPIType.PARALLEL,
            ),
            policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
            output_dir=str(tmp_path),
            seed=42,
        )

        # 2. Create telemetry emitter
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        emitter = TelemetryEmitter(run_id=config.run_id)
        emitter.run_started(
            {"worker_type": "ray", "paradigm": config.policy_configuration.value},
            constant=LOG_WORKER_RAY_RUNTIME_STARTED,
        )

        # Verify JSONL emission
        output = captured_output.getvalue()
        assert "run_started" in output
        assert config.run_id in output

        # 3. Write analytics manifest
        manifest_path = write_analytics_manifest(config, num_agents=3)

        # 4. Verify manifest
        manifest = WorkerAnalyticsManifest.load(manifest_path)
        assert manifest.run_id == config.run_id
        assert manifest.worker_type == "ray"
        assert manifest.metadata["paradigm"] == "parameter_sharing"
        assert manifest.metadata["seed"] == 42
        assert manifest.metadata["num_agents"] == 3

    def test_protocol_compliance_assertion_in_config(self):
        """Test that protocol compliance is checked in __post_init__."""
        from ray_worker.config import (
            RayWorkerConfig,
            EnvironmentConfig,
            PolicyConfiguration,
            PettingZooAPIType,
        )
        from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol

        # This should not raise (protocol assertion passes)
        config = RayWorkerConfig(
            run_id="test_integration_002",
            environment=EnvironmentConfig(
                family="sisl",
                env_id="waterworld_v4",
                api_type=PettingZooAPIType.PARALLEL,
            ),
            policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
        )

        # Verify it's a protocol instance
        assert isinstance(config, WorkerConfigProtocol)


# Test Summary
def test_phase3_summary():
    """Summary test - verify all Phase 3 components are in place."""
    components = {
        "Config Protocol": False,
        "TelemetryEmitter": False,
        "WorkerAnalyticsManifest": False,
        "get_worker_metadata": False,
        "Entry Point": False,
        "Log Constants": False,
    }

    # 1. Check config protocol
    try:
        from ray_worker.config import RayWorkerConfig, EnvironmentConfig, PolicyConfiguration, PettingZooAPIType
        from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
        cfg = RayWorkerConfig(
            run_id="test",
            environment=EnvironmentConfig(family="sisl", env_id="waterworld_v4", api_type=PettingZooAPIType.PARALLEL),
            policy_configuration=PolicyConfiguration.PARAMETER_SHARING,
        )
        components["Config Protocol"] = isinstance(cfg, WorkerConfigProtocol)
    except Exception:
        pass

    # 2. Check TelemetryEmitter
    try:
        from ray_worker.runtime import RayWorkerRuntime
        from gym_gui.core.worker import TelemetryEmitter
        runtime = RayWorkerRuntime(cfg)
        components["TelemetryEmitter"] = isinstance(runtime._emitter, TelemetryEmitter)
    except Exception:
        pass

    # 3. Check WorkerAnalyticsManifest
    try:
        from ray_worker.analytics import write_analytics_manifest
        components["WorkerAnalyticsManifest"] = True
    except Exception:
        pass

    # 4. Check get_worker_metadata
    try:
        from ray_worker import get_worker_metadata
        metadata, capabilities = get_worker_metadata()
        components["get_worker_metadata"] = True
    except Exception:
        pass

    # 5. Check entry point
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        if hasattr(eps, "select"):
            worker_eps = eps.select(group="mosaic.workers")
        else:
            worker_eps = eps.get("mosaic.workers", [])
        components["Entry Point"] = any(ep.name == "ray" for ep in worker_eps)
    except Exception:
        pass

    # 6. Check log constants
    try:
        from gym_gui.logging_config.log_constants import (
            LOG_WORKER_RAY_RUNTIME_STARTED,
            LOG_WORKER_RAY_HEARTBEAT,
        )
        components["Log Constants"] = True
    except Exception:
        pass

    print("\n=== Phase 3 Standardization Summary ===")
    for component, status in components.items():
        symbol = "[PASS]" if status else "[FAIL]"
        print(f"{symbol} {component}")

    # All must pass
    assert all(components.values()), f"Missing components: {[k for k, v in components.items() if not v]}"
