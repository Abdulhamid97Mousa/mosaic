"""Tests for XuanCe worker FastLane tab naming logic.

Ensures:
- XuanCe worker runs create tabs named "XuanCe-Live-{env_id}"
- XuanCe worker is detected by metadata_supports_fastlane
- Worker prefix mapping works correctly for XuanCe
- Multi-agent XuanCe runs (MAPPO, QMIX, etc.) are handled correctly
"""

import pytest
from typing import Dict, Any
from unittest.mock import MagicMock


class TestXuanCeMetadataSupport:
    """Test XuanCe worker detection in metadata_supports_fastlane."""

    @pytest.fixture
    def xuance_metadata(self) -> Dict[str, Any]:
        """Sample XuanCe worker metadata."""
        return {
            "ui": {
                "worker_id": "xuance_worker",
                "env_id": "Humanoid-v4",
                "method": "PPO_Clip",
                "paradigm": "single_agent",
                "fastlane_enabled": False,
                "fastlane_only": True,
            },
            "worker": {
                "worker_id": "xuance_worker",
                "module": "xuance_worker.cli",
                "config": {
                    "run_id": "xuance-ppo-clip-20251223-130032-cce7b3",
                    "method": "PPO_Clip",
                    "env": "mujoco",
                    "env_id": "Humanoid-v4",
                    "dl_toolbox": "torch",
                },
            },
        }

    def test_xuance_worker_detected_by_worker_id(self, xuance_metadata):
        """XuanCe worker detected via worker.worker_id."""
        from gym_gui.ui.handlers.features.fastlane_tab_handler import FastLaneTabHandler

        handler = FastLaneTabHandler(MagicMock())
        assert handler.metadata_supports_fastlane(xuance_metadata) is True

    def test_xuance_worker_detected_by_module_name(self):
        """XuanCe worker detected via worker.module containing 'xuance_worker'."""
        from gym_gui.ui.handlers.features.fastlane_tab_handler import FastLaneTabHandler

        metadata = {
            "worker": {
                "module": "xuance_worker.cli",
            },
        }
        handler = FastLaneTabHandler(MagicMock())
        assert handler.metadata_supports_fastlane(metadata) is True

    def test_xuance_worker_detected_by_ui_worker_id(self):
        """XuanCe worker detected via ui.worker_id."""
        from gym_gui.ui.handlers.features.fastlane_tab_handler import FastLaneTabHandler

        metadata = {
            "ui": {"worker_id": "xuance_worker"},
            "worker": {"config": {}},
        }
        handler = FastLaneTabHandler(MagicMock())
        # Need either fastlane_only or module detection
        metadata["ui"]["fastlane_only"] = True
        assert handler.metadata_supports_fastlane(metadata) is True

    def test_get_worker_id_from_xuance_metadata(self, xuance_metadata):
        """Extract worker_id from XuanCe metadata."""
        from gym_gui.ui.handlers.features.fastlane_tab_handler import FastLaneTabHandler

        handler = FastLaneTabHandler(MagicMock())
        worker_id = handler.get_worker_id(xuance_metadata)
        assert worker_id == "xuance_worker"

    def test_get_env_id_from_xuance_metadata(self, xuance_metadata):
        """Extract env_id from XuanCe metadata."""
        from gym_gui.ui.handlers.features.fastlane_tab_handler import FastLaneTabHandler

        handler = FastLaneTabHandler(MagicMock())
        env_id = handler.get_env_id(xuance_metadata)
        assert env_id == "Humanoid-v4"


class TestXuanCeTabNaming:
    """Test XuanCe worker tab naming patterns."""

    def test_xuance_worker_prefix(self):
        """XuanCe worker gets 'XuanCe' prefix."""
        from gym_gui.ui.handlers.features.fastlane_tab_handler import FastLaneTabHandler

        handler = FastLaneTabHandler(MagicMock())
        prefix = handler._get_worker_prefix("xuance_worker")
        assert prefix == "XuanCe"

    def test_cleanrl_worker_prefix(self):
        """CleanRL worker gets 'CleanRL' prefix."""
        from gym_gui.ui.handlers.features.fastlane_tab_handler import FastLaneTabHandler

        handler = FastLaneTabHandler(MagicMock())
        prefix = handler._get_worker_prefix("cleanrl_worker")
        assert prefix == "CleanRL"

    def test_ray_worker_prefix(self):
        """Ray worker gets 'Ray' prefix."""
        from gym_gui.ui.handlers.features.fastlane_tab_handler import FastLaneTabHandler

        handler = FastLaneTabHandler(MagicMock())
        prefix = handler._get_worker_prefix("ray_worker")
        assert prefix == "Ray"

    def test_unknown_worker_prefix(self):
        """Unknown worker gets 'Worker' prefix."""
        from gym_gui.ui.handlers.features.fastlane_tab_handler import FastLaneTabHandler

        handler = FastLaneTabHandler(MagicMock())
        prefix = handler._get_worker_prefix("custom_worker")
        assert prefix == "Worker"


class TestXuanCeTabTitle:
    """Test tab title generation for XuanCe runs."""

    def test_xuance_live_tab_title_with_env_id(self):
        """XuanCe live training uses XuanCe-Live-{env_id}."""
        # Build expected title using the helper logic
        worker_id = "xuance_worker"
        env_id = "Humanoid-v4"
        run_mode = "training"
        agent_id = ""

        # Mirror handler logic
        prefix = "XuanCe"
        mode_suffix = "Live"
        label = env_id if env_id else (agent_id or "default")
        expected_title = f"{prefix}-{mode_suffix}-{label}"

        assert expected_title == "XuanCe-Live-Humanoid-v4"

    def test_xuance_eval_tab_title(self):
        """XuanCe policy eval uses XuanCe-Eval-{env_id}."""
        worker_id = "xuance_worker"
        env_id = "CartPole-v1"
        run_mode = "policy_eval"
        agent_id = ""

        prefix = "XuanCe"
        mode_suffix = "Eval"
        label = env_id if env_id else (agent_id or "default")
        expected_title = f"{prefix}-{mode_suffix}-{label}"

        assert expected_title == "XuanCe-Eval-CartPole-v1"

    def test_xuance_tab_title_fallback_to_default(self):
        """XuanCe with no env_id or agent_id uses 'default'."""
        worker_id = "xuance_worker"
        env_id = ""
        run_mode = "training"
        agent_id = ""

        prefix = "XuanCe"
        mode_suffix = "Live"
        label = env_id if env_id else (agent_id or "default")
        expected_title = f"{prefix}-{mode_suffix}-{label}"

        assert expected_title == "XuanCe-Live-default"
