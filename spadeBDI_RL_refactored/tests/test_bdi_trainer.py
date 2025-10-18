"""Tests for BDITrainer integration with RL adapters.

This test suite verifies that:
1. BDITrainer can be instantiated with all adapter types
2. BDI-specific configuration is properly set
3. The trainer falls back to RL-only mode when SPADE is unavailable
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from spadeBDI_RL_refactored.adapters import create_adapter
from spadeBDI_RL_refactored.core import BDITrainer, RunConfig, TelemetryEmitter


class TestBDITrainerInstantiation:
    """Test that BDITrainer can be created with different adapter types."""

    @pytest.fixture
    def run_config(self, tmp_path: Path) -> RunConfig:
        """Create a minimal run configuration for testing."""
        config_dict = {
            "run_id": "test_bdi_run_001",
            "agent_id": "bdi_test_agent",
            "env_id": "FrozenLake-v2",
            "seed": 42,
            "max_episodes": 2,
            "max_steps_per_episode": 10,
            "policy_strategy": "train",
            "policy_path": str(tmp_path / "policies"),
            "headless": True,
        }
        return RunConfig.from_dict(config_dict)

    @pytest.fixture
    def emitter(self) -> TelemetryEmitter:
        """Create a telemetry emitter for testing."""
        return TelemetryEmitter()

    def test_bdi_trainer_with_frozenlake_v1(self, run_config: RunConfig, emitter: TelemetryEmitter):
        """Test BDITrainer instantiation with FrozenLake-v1."""
        adapter = create_adapter("FrozenLake-v1")
        trainer = BDITrainer(
            adapter,
            run_config,
            emitter,
            jid="test@localhost",
            password="test123",
        )
        
        assert trainer.jid == "test@localhost"
        assert trainer.password == "test123"
        assert trainer.asl_file is None
        assert trainer.bdi_agent is None  # Not started yet
        assert trainer.adapter == adapter

    def test_bdi_trainer_with_frozenlake_v2(self, run_config: RunConfig, emitter: TelemetryEmitter):
        """Test BDITrainer instantiation with FrozenLake-v2."""
        adapter = create_adapter("FrozenLake-v2", grid_height=5, grid_width=5)
        trainer = BDITrainer(
            adapter,
            run_config,
            emitter,
            jid="agent@localhost",
            password="secret",
        )
        
        assert trainer.jid == "agent@localhost"
        assert trainer.config.env_id == "FrozenLake-v2"

    def test_bdi_trainer_with_cliffwalking(self, run_config: RunConfig, emitter: TelemetryEmitter):
        """Test BDITrainer instantiation with CliffWalking-v1."""
        run_config.env_id = "CliffWalking-v1"
        adapter = create_adapter("CliffWalking-v1")
        trainer = BDITrainer(
            adapter,
            run_config,
            emitter,
            jid="cliff_agent@localhost",
            password="cliff123",
        )
        
        assert trainer.jid == "cliff_agent@localhost"
        assert trainer.adapter == adapter

    def test_bdi_trainer_with_taxi(self, run_config: RunConfig, emitter: TelemetryEmitter):
        """Test BDITrainer instantiation with Taxi-v3."""
        run_config.env_id = "Taxi-v3"
        adapter = create_adapter("Taxi-v3")
        trainer = BDITrainer(
            adapter,
            run_config,
            emitter,
            jid="taxi_agent@localhost",
            password="taxi123",
        )
        
        assert trainer.jid == "taxi_agent@localhost"
        assert trainer.adapter == adapter

    def test_bdi_trainer_with_custom_asl(self, run_config: RunConfig, emitter: TelemetryEmitter):
        """Test BDITrainer with custom ASL file path."""
        adapter = create_adapter("FrozenLake-v2")
        custom_asl = "/path/to/custom.asl"
        
        trainer = BDITrainer(
            adapter,
            run_config,
            emitter,
            jid="custom_agent@localhost",
            password="custom123",
            asl_file=custom_asl,
        )
        
        assert trainer.asl_file == custom_asl


class TestBDITrainerConfiguration:
    """Test BDI-specific configuration handling."""

    @pytest.fixture
    def trainer(self, tmp_path: Path) -> BDITrainer:
        """Create a BDITrainer instance for testing."""
        config_dict = {
            "run_id": "test_config_run",
            "agent_id": "config_test",
            "env_id": "FrozenLake-v2",
            "seed": 100,
            "max_episodes": 1,
            "max_steps_per_episode": 5,
            "policy_strategy": "train",
            "policy_path": str(tmp_path / "policies"),
            "headless": True,
        }
        config = RunConfig.from_dict(config_dict)
        adapter = create_adapter("FrozenLake-v2", grid_height=4, grid_width=4)
        emitter = TelemetryEmitter()
        
        return BDITrainer(
            adapter,
            config,
            emitter,
            jid="config@localhost",
            password="config123",
            asl_file="/custom/plan.asl",
        )

    def test_config_payload_includes_bdi_fields(self, trainer: BDITrainer):
        """Test that config payload includes BDI-specific fields."""
        payload = trainer._build_config_payload()
        
        assert payload["bdi_enabled"] is True
        assert payload["bdi_jid"] == "config@localhost"
        assert payload["asl_file"] == "/custom/plan.asl"
        assert "env_id" in payload
        assert "run_id" in payload

    def test_episode_metadata_includes_bdi_fields(self, trainer: BDITrainer):
        """Test that episode metadata includes BDI-specific fields."""
        from spadeBDI_RL_refactored.core.runtime import EpisodeMetrics
        
        metrics = EpisodeMetrics(episode=0, total_reward=1.0, steps=5, success=True)
        metadata = trainer._build_episode_metadata(0, metrics)
        
        assert metadata["control_mode"] == "bdi_agent"
        assert metadata["bdi_enabled"] is True
        assert metadata["bdi_jid"] == "config@localhost"
        assert metadata["success"] is True
        assert metadata["episode_index"] == 0

    def test_policy_metadata_includes_bdi_fields(self, trainer: BDITrainer):
        """Test that policy save metadata includes BDI-specific fields."""
        metadata = trainer._build_policy_metadata()
        
        assert metadata["bdi_enabled"] is True
        assert metadata["bdi_jid"] == "config@localhost"
        assert metadata["asl_file"] == "/custom/plan.asl"
        assert "run_id" in metadata
        assert "strategy" in metadata


class TestBDITrainerWorkerIntegration:
    """Test worker.py integration with BDITrainer."""

    def test_worker_imports_bdi_trainer(self):
        """Test that worker module can import BDITrainer."""
        from spadeBDI_RL_refactored.worker import BDITrainer as WorkerBDITrainer
        
        assert WorkerBDITrainer is not None
        assert WorkerBDITrainer.__name__ == "BDITrainer"

    def test_worker_main_with_bdi_flag(self, tmp_path: Path, monkeypatch):
        """Test worker main() function with --bdi flag."""
        from spadeBDI_RL_refactored import worker
        
        # Create a test config file
        config_file = tmp_path / "test_config.json"
        config_data = {
            "run_id": "worker_bdi_test",
            "agent_id": "worker_agent",
            "env_id": "FrozenLake-v2",
            "seed": 42,
            "max_episodes": 1,
            "max_steps_per_episode": 5,
            "policy_strategy": "train",
            "policy_path": str(tmp_path / "policies"),
            "headless": True,
        }
        config_file.write_text(json.dumps(config_data))
        
        # Mock sys.stdout to capture JSONL output
        import io
        mock_stdout = io.StringIO()
        monkeypatch.setattr("sys.stdout", mock_stdout)
        
        # Run worker with BDI flag
        argv = [
            "--config", str(config_file),
            "--bdi",
            "--bdi-jid", "test_worker@localhost",
            "--bdi-password", "test_pass",
        ]
        
        result = worker.main(argv)
        
        # Should complete successfully (BDI trainer falls back to RL mode)
        assert result == 0
        
        # Verify JSONL output contains run events
        output = mock_stdout.getvalue()
        assert "run_started" in output or '"type":"run_started"' in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
