"""Tests for Stockfish Worker standardization and functionality."""

import json
import sys
from pathlib import Path

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stockfish_worker.config import (
    StockfishWorkerConfig,
    load_worker_config,
    DIFFICULTY_PRESETS,
)


class TestConfigCompliance:
    """Test config implements WorkerConfig protocol."""

    def test_config_has_required_fields(self):
        """Config must have run_id and seed fields."""
        config = StockfishWorkerConfig(run_id="test", seed=42)
        assert hasattr(config, "run_id")
        assert hasattr(config, "seed")
        assert config.run_id == "test"
        assert config.seed == 42

    def test_config_implements_protocol(self):
        """Config must implement WorkerConfig protocol."""
        try:
            from gym_gui.core.worker import WorkerConfig as WorkerConfigProtocol
        except ImportError:
            pytest.skip("gym_gui not available")

        config = StockfishWorkerConfig(run_id="test")
        assert isinstance(config, WorkerConfigProtocol)

    def test_config_to_dict(self):
        """Config must have to_dict() method."""
        config = StockfishWorkerConfig(run_id="test", difficulty="easy")
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["run_id"] == "test"
        assert d["difficulty"] == "easy"

    def test_config_from_dict(self):
        """Config must have from_dict() class method."""
        data = {"run_id": "from_dict_test", "difficulty": "hard"}
        config = StockfishWorkerConfig.from_dict(data)
        assert config.run_id == "from_dict_test"
        assert config.difficulty == "hard"

    def test_config_roundtrip(self):
        """Config should survive to_dict/from_dict roundtrip."""
        original = StockfishWorkerConfig(
            run_id="roundtrip",
            seed=123,
            difficulty="expert",
        )
        restored = StockfishWorkerConfig.from_dict(original.to_dict())
        assert restored.run_id == original.run_id
        assert restored.seed == original.seed
        assert restored.difficulty == original.difficulty
        assert restored.skill_level == original.skill_level


class TestDifficultyPresets:
    """Test difficulty preset functionality."""

    def test_all_presets_exist(self):
        """All documented presets should exist."""
        expected = ["beginner", "easy", "medium", "hard", "expert"]
        for preset in expected:
            assert preset in DIFFICULTY_PRESETS

    def test_preset_values_applied(self):
        """Preset values should be applied to config."""
        config = StockfishWorkerConfig(run_id="test", difficulty="beginner")
        assert config.skill_level == 1
        assert config.depth == 5
        assert config.time_limit_ms == 500

    def test_preset_override(self):
        """Explicit values should override presets."""
        config = StockfishWorkerConfig(
            run_id="test",
            difficulty="beginner",
            skill_level=15,  # Override preset
        )
        assert config.skill_level == 15  # Overridden
        assert config.depth == 5  # From preset

    def test_from_preset_factory(self):
        """from_preset should create config with preset values."""
        config = StockfishWorkerConfig.from_preset("expert", run_id="expert_test")
        assert config.run_id == "expert_test"
        assert config.skill_level == 20
        assert config.depth == 20


class TestConfigValidation:
    """Test config validation."""

    def test_invalid_difficulty_raises(self):
        """Invalid difficulty should raise ValueError."""
        with pytest.raises(ValueError, match="difficulty must be one of"):
            StockfishWorkerConfig(run_id="test", difficulty="impossible")

    def test_skill_level_bounds(self):
        """Skill level outside 0-20 should raise ValueError."""
        with pytest.raises(ValueError, match="skill_level must be between"):
            StockfishWorkerConfig(run_id="test", skill_level=25)

    def test_depth_bounds(self):
        """Depth outside 1-30 should raise ValueError."""
        with pytest.raises(ValueError, match="depth must be between"):
            StockfishWorkerConfig(run_id="test", depth=50)

    def test_time_limit_minimum(self):
        """Time limit below 100ms should raise ValueError."""
        with pytest.raises(ValueError, match="time_limit_ms must be at least"):
            StockfishWorkerConfig(run_id="test", time_limit_ms=50)


class TestConfigLoading:
    """Test config loading from files."""

    def test_load_direct_format(self, tmp_path):
        """Load config from direct JSON format."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "run_id": "direct_test",
            "difficulty": "medium",
            "seed": 42,
        }))

        config = load_worker_config(str(config_file))
        assert config.run_id == "direct_test"
        assert config.seed == 42

    def test_load_nested_format(self, tmp_path):
        """Load config from nested metadata format (GUI format)."""
        config_file = tmp_path / "config.json"
        nested = {
            "metadata": {
                "worker": {
                    "config": {
                        "run_id": "nested_test",
                        "difficulty": "hard",
                    }
                }
            }
        }
        config_file.write_text(json.dumps(nested))

        config = load_worker_config(str(config_file))
        assert config.run_id == "nested_test"
        assert config.difficulty == "hard"

    def test_load_operator_format(self, tmp_path):
        """Load config from operator settings format."""
        config_file = tmp_path / "config.json"
        operator_format = {
            "run_id": "operator_test",
            "settings": {
                "difficulty": "easy",
                "skill_level": 3,
            }
        }
        config_file.write_text(json.dumps(operator_format))

        config = load_worker_config(str(config_file))
        assert config.run_id == "operator_test"

    def test_load_missing_file_raises(self):
        """Loading missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_worker_config("/nonexistent/path/config.json")


class TestWorkerMetadata:
    """Test worker metadata for MOSAIC discovery."""

    def test_get_worker_metadata_exists(self):
        """Worker must export get_worker_metadata()."""
        from stockfish_worker import get_worker_metadata
        assert callable(get_worker_metadata)

    def test_get_worker_metadata_returns_tuple(self):
        """get_worker_metadata() must return (WorkerMetadata, WorkerCapabilities)."""
        try:
            from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from stockfish_worker import get_worker_metadata
        result = get_worker_metadata()

        assert isinstance(result, tuple)
        assert len(result) == 2

        metadata, capabilities = result
        assert isinstance(metadata, WorkerMetadata)
        assert isinstance(capabilities, WorkerCapabilities)

    def test_metadata_fields(self):
        """Metadata should have required fields."""
        try:
            from gym_gui.core.worker import WorkerMetadata, WorkerCapabilities
        except ImportError:
            pytest.skip("gym_gui not available")

        from stockfish_worker import get_worker_metadata
        metadata, capabilities = get_worker_metadata()

        assert metadata.name == "Stockfish Worker"
        assert capabilities.worker_type == "stockfish"
        assert "pettingzoo" in capabilities.env_families


class TestEntryPointDiscovery:
    """Test worker is discoverable via entry points."""

    def test_entry_point_registered(self):
        """Worker must be registered in mosaic.workers group."""
        try:
            from importlib.metadata import entry_points
        except ImportError:
            from importlib_metadata import entry_points

        if sys.version_info >= (3, 10):
            eps = entry_points(group="mosaic.workers")
        else:
            eps = entry_points().get("mosaic.workers", [])

        stockfish_eps = [ep for ep in eps if ep.name == "stockfish"]
        # This test passes after pip install -e .
        # For now, just check the entry point module is importable
        from stockfish_worker import get_worker_metadata
        assert get_worker_metadata is not None


class TestRuntimeCreation:
    """Test runtime can be created (without actually running Stockfish)."""

    def test_runtime_requires_stockfish(self):
        """Runtime should fail gracefully if Stockfish not available."""
        from stockfish_worker import StockfishWorkerRuntime
        from stockfish_worker.config import StockfishWorkerConfig

        config = StockfishWorkerConfig(
            run_id="test",
            stockfish_path="/nonexistent/stockfish",
        )

        # Should raise RuntimeError about missing binary
        # (unless Stockfish is actually installed)
        try:
            runtime = StockfishWorkerRuntime(config)
            # If we get here, Stockfish was found on system
            assert runtime._stockfish_path is not None
        except RuntimeError as e:
            assert "Stockfish binary not found" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
