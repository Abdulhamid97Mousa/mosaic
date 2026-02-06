"""Tests for XuanCe FastLane integration.

These tests validate:
- FastLane configuration from UI metadata
- Tab creation with paradigm-aware naming
- Environment variable setup
- Presenter behavior when FastLane is enabled/disabled
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def single_agent_metadata() -> Dict[str, Any]:
    """Create metadata for single-agent training with FastLane enabled."""
    return {
        "ui": {
            "worker_id": "xuance_worker",
            "method": "PPO_Clip",
            "env": "classic_control",
            "env_id": "CartPole-v1",
            "backend": "torch",
            "paradigm": "single_agent",
            "fastlane_enabled": True,
            "fastlane_only": False,
            "fastlane_slot": 0,
            "fastlane_video_mode": "single",
            "fastlane_grid_limit": 4,
        },
        "worker": {
            "worker_id": "xuance_worker",
            "module": "xuance_worker.cli",
            "config": {
                "method": "PPO_Clip",
                "env": "classic_control",
                "env_id": "CartPole-v1",
                "parallels": 8,
            },
        },
    }


@pytest.fixture
def multi_agent_metadata() -> Dict[str, Any]:
    """Create metadata for multi-agent training with FastLane enabled."""
    return {
        "ui": {
            "worker_id": "xuance_worker",
            "method": "MAPPO",
            "env": "mpe",
            "env_id": "simple_spread_v3",
            "backend": "torch",
            "paradigm": "multi_agent",
            "fastlane_enabled": True,
            "fastlane_only": False,
            "fastlane_slot": 0,
            "fastlane_video_mode": "single",
            "fastlane_grid_limit": 4,
        },
        "worker": {
            "worker_id": "xuance_worker",
            "module": "xuance_worker.cli",
            "config": {
                "method": "MAPPO",
                "env": "mpe",
                "env_id": "simple_spread_v3",
                "parallels": 16,
            },
        },
    }


@pytest.fixture
def fastlane_disabled_metadata() -> Dict[str, Any]:
    """Create metadata with FastLane disabled."""
    return {
        "ui": {
            "worker_id": "xuance_worker",
            "method": "DQN",
            "env": "atari",
            "env_id": "Pong-v5",
            "backend": "torch",
            "paradigm": "single_agent",
            "fastlane_enabled": False,  # Disabled!
            "fastlane_only": False,
            "fastlane_slot": 0,
            "fastlane_video_mode": "single",
            "fastlane_grid_limit": 4,
        },
        "worker": {
            "worker_id": "xuance_worker",
            "module": "xuance_worker.cli",
            "config": {
                "method": "DQN",
                "env": "atari",
                "env_id": "Pong-v5",
                "parallels": 4,
            },
        },
    }


# =============================================================================
# Presenter Tests
# =============================================================================


class TestXuanCeWorkerPresenter:
    """Test XuanCeWorkerPresenter tab creation functionality."""

    def test_presenter_imports(self):
        """Test that XuanCeWorkerPresenter is importable."""
        from gym_gui.ui.presenters.workers.xuance_worker_presenter import XuanCeWorkerPresenter

        assert XuanCeWorkerPresenter is not None

    def test_presenter_id(self):
        """Test presenter ID is 'xuance_worker'."""
        from gym_gui.ui.presenters.workers.xuance_worker_presenter import XuanCeWorkerPresenter

        presenter = XuanCeWorkerPresenter()
        assert presenter.id == "xuance_worker"

    def test_create_tabs_single_agent_fastlane_enabled(self, single_agent_metadata):
        """Test tab creation for single-agent with FastLane enabled."""
        from gym_gui.ui.presenters.workers.xuance_worker_presenter import XuanCeWorkerPresenter

        presenter = XuanCeWorkerPresenter()

        # Mock first_payload with metadata
        first_payload = {"metadata": single_agent_metadata}

        # Mock parent widget
        mock_parent = Mock()

        # Mock FastLaneTab import (patch where it's imported, not where it's defined)
        with patch('gym_gui.ui.widgets.fastlane_tab.FastLaneTab') as MockFastLaneTab:
            mock_tab = Mock()
            MockFastLaneTab.return_value = mock_tab

            tabs = presenter.create_tabs(
                run_id="test-run-001",
                agent_id="agent-001",
                first_payload=first_payload,
                parent=mock_parent,
            )

            # Should create 1 tab
            assert len(tabs) == 1

            # Unpack tab tuple
            tab_name, tab_widget = tabs[0]

            # Tab name should be: XuanCe-SA-Live-CartPole-v1-{run_id[:8]}
            assert tab_name.startswith("XuanCe-SA-Live-CartPole-v1-")
            assert "test-run" in tab_name

            # FastLaneTab should be created with correct args
            MockFastLaneTab.assert_called_once_with(
                run_id="test-run-001",
                agent_id="agent-001",
                mode_label="Fast lane",
                run_mode="train",
                parent=mock_parent,
            )

    def test_create_tabs_multi_agent_fastlane_enabled(self, multi_agent_metadata):
        """Test tab creation for multi-agent with FastLane enabled."""
        from gym_gui.ui.presenters.workers.xuance_worker_presenter import XuanCeWorkerPresenter

        presenter = XuanCeWorkerPresenter()
        first_payload = {"metadata": multi_agent_metadata}
        mock_parent = Mock()

        with patch('gym_gui.ui.widgets.fastlane_tab.FastLaneTab') as MockFastLaneTab:
            mock_tab = Mock()
            MockFastLaneTab.return_value = mock_tab

            tabs = presenter.create_tabs(
                run_id="test-ma-run-002",
                agent_id="agent-002",
                first_payload=first_payload,
                parent=mock_parent,
            )

            # Should create 1 tab
            assert len(tabs) == 1

            tab_name, tab_widget = tabs[0]

            # Tab name should be: XuanCe-MA-Live-simple_spread_v3-{run_id[:8]}
            assert tab_name.startswith("XuanCe-MA-Live-simple_spread_v3-")
            assert "test-ma-" in tab_name

    def test_create_tabs_fastlane_disabled_returns_empty(self, fastlane_disabled_metadata):
        """Test that no tabs are created when FastLane is disabled."""
        from gym_gui.ui.presenters.workers.xuance_worker_presenter import XuanCeWorkerPresenter

        presenter = XuanCeWorkerPresenter()
        first_payload = {"metadata": fastlane_disabled_metadata}
        mock_parent = Mock()

        tabs = presenter.create_tabs(
            run_id="test-disabled-003",
            agent_id="agent-003",
            first_payload=first_payload,
            parent=mock_parent,
        )

        # Should return empty list when FastLane disabled
        assert tabs == []

    def test_create_tabs_missing_metadata_returns_empty(self):
        """Test that no tabs are created when metadata is missing."""
        from gym_gui.ui.presenters.workers.xuance_worker_presenter import XuanCeWorkerPresenter

        presenter = XuanCeWorkerPresenter()
        first_payload = {}  # No metadata!
        mock_parent = Mock()

        tabs = presenter.create_tabs(
            run_id="test-no-meta-004",
            agent_id="agent-004",
            first_payload=first_payload,
            parent=mock_parent,
        )

        # Should return empty list
        assert tabs == []

    def test_create_tabs_import_error_returns_empty(self, single_agent_metadata):
        """Test that tab creation handles import errors gracefully."""
        from gym_gui.ui.presenters.workers.xuance_worker_presenter import XuanCeWorkerPresenter

        presenter = XuanCeWorkerPresenter()
        first_payload = {"metadata": single_agent_metadata}
        mock_parent = Mock()

        # Mock FastLaneTab import to raise ImportError
        with patch('gym_gui.ui.widgets.fastlane_tab.FastLaneTab', side_effect=ImportError("FastLane not available")):
            tabs = presenter.create_tabs(
                run_id="test-import-err-005",
                agent_id="agent-005",
                first_payload=first_payload,
                parent=mock_parent,
            )

            # Should return empty list on import error
            assert tabs == []


# =============================================================================
# Tab Naming Tests
# =============================================================================


class TestTabNaming:
    """Test paradigm-aware tab naming conventions."""

    @pytest.mark.parametrize("paradigm,expected_prefix", [
        ("single_agent", "SA"),
        ("multi_agent", "MA"),
    ])
    def test_paradigm_prefix_mapping(self, paradigm, expected_prefix):
        """Test that paradigms map to correct prefixes."""
        from gym_gui.ui.presenters.workers.xuance_worker_presenter import XuanCeWorkerPresenter

        presenter = XuanCeWorkerPresenter()

        metadata = {
            "ui": {
                "env_id": "TestEnv-v0",
                "paradigm": paradigm,
                "fastlane_enabled": True,
            },
        }

        first_payload = {"metadata": metadata}
        mock_parent = Mock()

        with patch('gym_gui.ui.widgets.fastlane_tab.FastLaneTab') as MockFastLaneTab:
            MockFastLaneTab.return_value = Mock()

            tabs = presenter.create_tabs(
                run_id="test-paradigm-001",
                agent_id="agent-001",
                first_payload=first_payload,
                parent=mock_parent,
            )

            assert len(tabs) == 1
            tab_name, _ = tabs[0]

            # Should contain correct paradigm prefix
            assert f"XuanCe-{expected_prefix}-Live-TestEnv-v0" in tab_name

    def test_tab_naming_format(self):
        """Test complete tab naming format."""
        from gym_gui.ui.presenters.workers.xuance_worker_presenter import XuanCeWorkerPresenter

        presenter = XuanCeWorkerPresenter()

        metadata = {
            "ui": {
                "env_id": "HalfCheetah-v4",
                "paradigm": "single_agent",
                "fastlane_enabled": True,
            },
        }

        first_payload = {"metadata": metadata}

        with patch('gym_gui.ui.widgets.fastlane_tab.FastLaneTab') as MockFastLaneTab:
            MockFastLaneTab.return_value = Mock()

            tabs = presenter.create_tabs(
                run_id="01ABCDEFGH1234567890ABCD",
                agent_id="agent-001",
                first_payload=first_payload,
                parent=Mock(),
            )

            tab_name, _ = tabs[0]

            # Format: XuanCe-{Paradigm}-Live-{env_id}-{run_id[:8]}
            assert tab_name == "XuanCe-SA-Live-HalfCheetah-v4-01ABCDEF"


# =============================================================================
# Configuration Tests
# =============================================================================


class TestFastLaneConfiguration:
    """Test FastLane configuration parsing and validation."""

    def test_fastlane_enabled_in_metadata(self, single_agent_metadata):
        """Test that fastlane_enabled flag is correctly extracted."""
        ui_meta = single_agent_metadata["ui"]

        assert ui_meta["fastlane_enabled"] is True
        assert ui_meta["fastlane_video_mode"] == "single"
        assert ui_meta["fastlane_slot"] == 0
        assert ui_meta["fastlane_grid_limit"] == 4

    def test_fastlane_disabled_in_metadata(self, fastlane_disabled_metadata):
        """Test metadata when FastLane is disabled."""
        ui_meta = fastlane_disabled_metadata["ui"]

        assert ui_meta["fastlane_enabled"] is False

    @pytest.mark.parametrize("video_mode", ["single", "grid"])
    def test_video_mode_options(self, video_mode):
        """Test that video mode options are preserved in metadata."""
        metadata = {
            "ui": {
                "env_id": "TestEnv-v0",
                "paradigm": "single_agent",
                "fastlane_enabled": True,
                "fastlane_video_mode": video_mode,
            },
        }

        assert metadata["ui"]["fastlane_video_mode"] == video_mode


# =============================================================================
# Metadata Extraction Tests
# =============================================================================


class TestMetadataExtraction:
    """Test metadata extraction from first_payload."""

    def test_extract_env_id(self, single_agent_metadata):
        """Test env_id extraction."""
        env_id = single_agent_metadata["ui"]["env_id"]
        assert env_id == "CartPole-v1"

    def test_extract_paradigm(self, single_agent_metadata, multi_agent_metadata):
        """Test paradigm extraction."""
        sa_paradigm = single_agent_metadata["ui"]["paradigm"]
        ma_paradigm = multi_agent_metadata["ui"]["paradigm"]

        assert sa_paradigm == "single_agent"
        assert ma_paradigm == "multi_agent"

    def test_extract_worker_id(self, single_agent_metadata):
        """Test worker_id extraction."""
        worker_id = single_agent_metadata["ui"]["worker_id"]
        assert worker_id == "xuance_worker"

    def test_extract_backend(self, single_agent_metadata):
        """Test backend extraction."""
        backend = single_agent_metadata["ui"]["backend"]
        assert backend == "torch"


# =============================================================================
# Edge Cases & Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_ui_metadata(self):
        """Test handling of missing ui metadata."""
        from gym_gui.ui.presenters.workers.xuance_worker_presenter import XuanCeWorkerPresenter

        presenter = XuanCeWorkerPresenter()
        first_payload = {"metadata": {}}  # Missing 'ui' key

        tabs = presenter.create_tabs(
            run_id="test-001",
            agent_id="agent-001",
            first_payload=first_payload,
            parent=Mock(),
        )

        # Should handle gracefully and return empty
        assert tabs == []

    def test_default_paradigm_fallback(self):
        """Test that paradigm defaults to 'single_agent' if missing."""
        from gym_gui.ui.presenters.workers.xuance_worker_presenter import XuanCeWorkerPresenter

        presenter = XuanCeWorkerPresenter()

        metadata = {
            "ui": {
                "env_id": "TestEnv-v0",
                # paradigm missing!
                "fastlane_enabled": True,
            },
        }

        first_payload = {"metadata": metadata}

        with patch('gym_gui.ui.widgets.fastlane_tab.FastLaneTab') as MockFastLaneTab:
            MockFastLaneTab.return_value = Mock()

            tabs = presenter.create_tabs(
                run_id="test-default-001",
                agent_id="agent-001",
                first_payload=first_payload,
                parent=Mock(),
            )

            tab_name, _ = tabs[0]

            # Should default to SA (single_agent)
            assert "XuanCe-SA-" in tab_name

    def test_default_env_id_fallback(self):
        """Test that env_id defaults to 'env' if missing."""
        from gym_gui.ui.presenters.workers.xuance_worker_presenter import XuanCeWorkerPresenter

        presenter = XuanCeWorkerPresenter()

        metadata = {
            "ui": {
                # env_id missing!
                "paradigm": "single_agent",
                "fastlane_enabled": True,
            },
        }

        first_payload = {"metadata": metadata}

        with patch('gym_gui.ui.widgets.fastlane_tab.FastLaneTab') as MockFastLaneTab:
            MockFastLaneTab.return_value = Mock()

            tabs = presenter.create_tabs(
                run_id="test-default-002",
                agent_id="agent-002",
                first_payload=first_payload,
                parent=Mock(),
            )

            tab_name, _ = tabs[0]

            # Should default to "env"
            assert "XuanCe-SA-Live-env-" in tab_name


# =============================================================================
# Integration Tests (Future)
# =============================================================================


class TestFastLaneIntegration:
    """Integration tests for FastLane functionality."""

    def test_environment_wrapping(self, monkeypatch):
        """Test that environments are wrapped with FastLane telemetry."""
        # Set FastLane environment variables
        monkeypatch.setenv("GYM_GUI_FASTLANE_ONLY", "1")
        monkeypatch.setenv("XUANCE_RUN_ID", "test-run-wrap")
        monkeypatch.setenv("GYM_GUI_FASTLANE_VIDEO_MODE", "single")
        monkeypatch.setenv("MOSAIC_FASTLANE_ENABLED", "1")

        # Import after setting env vars to get fresh config
        from xuance_worker.fastlane import maybe_wrap_env, reload_fastlane_config, FastLaneTelemetryWrapper

        # Force reload config with new env vars
        reload_fastlane_config()

        # Create a mock environment
        class MockEnv:
            def render(self):
                return None

            def step(self, action):
                return None, 0.0, False, False, {}

            def reset(self):
                return None, {}

            def close(self):
                pass

        env = MockEnv()
        wrapped = maybe_wrap_env(env)

        # Verify wrapping
        assert isinstance(wrapped, FastLaneTelemetryWrapper), f"Expected FastLaneTelemetryWrapper, got {type(wrapped)}"
        assert wrapped.env is env

    def test_config_reload_on_env_var_change(self, monkeypatch):
        """Test that config is reloaded when env vars change."""
        # Start with FastLane disabled
        monkeypatch.delenv("GYM_GUI_FASTLANE_ONLY", raising=False)
        monkeypatch.delenv("MOSAIC_FASTLANE_ENABLED", raising=False)

        from xuance_worker.fastlane import (
            is_fastlane_enabled,
            reload_fastlane_config,
            maybe_wrap_env,
            _CONFIG,
        )

        # Reload to get disabled state
        reload_fastlane_config()

        # Verify FastLane is initially disabled
        assert not is_fastlane_enabled()

        # Now enable FastLane via env vars
        monkeypatch.setenv("GYM_GUI_FASTLANE_ONLY", "1")
        monkeypatch.setenv("XUANCE_RUN_ID", "test-reload")

        # is_fastlane_enabled should now return True (reads env vars dynamically)
        assert is_fastlane_enabled()

        # Create a mock environment - maybe_wrap_env should detect the change and reload config
        class MockEnv:
            def render(self):
                return None

            def step(self, action):
                return None, 0.0, False, False, {}

            def reset(self):
                return None, {}

            def close(self):
                pass

        env = MockEnv()
        wrapped = maybe_wrap_env(env)

        # The config should have been reloaded and the env should be wrapped
        from xuance_worker.fastlane import FastLaneTelemetryWrapper

        assert isinstance(wrapped, FastLaneTelemetryWrapper), "Config should have been reloaded"

    @pytest.mark.skip(reason="Requires shared memory access - run manually")
    def test_frame_emission(self):
        """Test that frames are emitted to FastLane shared memory.

        Note: This test requires /dev/shm access and is skipped by default.
        Run manually with: pytest -k test_frame_emission --run-shm-tests
        """
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
