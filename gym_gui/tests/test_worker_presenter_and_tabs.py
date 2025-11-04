"""Comprehensive tests for worker presenter, registry, and tab factory.

Tests cover:
1. WorkerPresenterRegistry - registration, lookup, and lifecycle
2. WorkerPresenter protocol compliance
3. SpadeBdiWorkerPresenter - train request building, tab creation, metadata extraction
4. TabFactory - tab instantiation, environment detection, conditional creation
5. Analytics tabs wiring (TensorBoard + W&B)
6. Integration with main_window.py - registry usage in UI tab creation
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any
import os

from gym_gui.ui.presenters.workers.registry import (
    WorkerPresenter,
    WorkerPresenterRegistry,
)
from gym_gui.ui.presenters.workers.spade_bdi_worker_presenter import SpadeBdiWorkerPresenter
from gym_gui.ui.presenters.workers.cleanrl_worker_presenter import CleanRlWorkerPresenter
from gym_gui.ui.presenters.workers import (
    get_worker_presenter_registry,
    SpadeBdiWorkerPresenter as ExportedSpadePresenter,
    CleanRlWorkerPresenter as ExportedCleanPresenter,
)
from gym_gui.ui.widgets.spade_bdi_worker_tabs.factory import TabFactory
from gym_gui.core.enums import GameId
from gym_gui.ui.panels.analytics_tabs import AnalyticsTabManager
from gym_gui.ui.widgets.tensorboard_artifact_tab import TensorboardArtifactTab
from gym_gui.ui.widgets.wandb_artifact_tab import WandbArtifactTab
from qtpy import QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestWorkerPresenterRegistry(unittest.TestCase):
    """Test WorkerPresenterRegistry registration, lookup, and lifecycle."""

    def setUp(self) -> None:
        """Create a fresh registry for each test."""
        self.registry = WorkerPresenterRegistry()
        self.mock_presenter = Mock(spec=WorkerPresenter)
        self.mock_presenter.id = "test_worker"

    def test_register_presenter(self) -> None:
        """Test registering a presenter."""
        self.registry.register("test_worker", self.mock_presenter)
        self.assertTrue("test_worker" in self.registry)

    def test_register_duplicate_raises_error(self) -> None:
        """Test that registering the same worker twice raises ValueError."""
        self.registry.register("test_worker", self.mock_presenter)
        with self.assertRaises(ValueError) as ctx:
            self.registry.register("test_worker", self.mock_presenter)
        self.assertIn("already registered", str(ctx.exception))

    def test_get_presenter(self) -> None:
        """Test retrieving a registered presenter."""
        self.registry.register("test_worker", self.mock_presenter)
        retrieved = self.registry.get("test_worker")
        self.assertIs(retrieved, self.mock_presenter)

    def test_get_nonexistent_presenter(self) -> None:
        """Test that getting a non-existent presenter returns None."""
        retrieved = self.registry.get("nonexistent")
        self.assertIsNone(retrieved)

    def test_available_workers(self) -> None:
        """Test listing available workers."""
        presenter1 = Mock(spec=WorkerPresenter)
        presenter1.id = "worker1"
        presenter2 = Mock(spec=WorkerPresenter)
        presenter2.id = "worker2"

        self.registry.register("worker1", presenter1)
        self.registry.register("worker2", presenter2)

        workers = self.registry.available_workers()
        self.assertEqual(sorted(workers), ["worker1", "worker2"])

    def test_contains_operator(self) -> None:
        """Test __contains__ operator for presence check."""
        self.registry.register("test_worker", self.mock_presenter)
        self.assertTrue("test_worker" in self.registry)
        self.assertFalse("nonexistent" in self.registry)


class TestGlobalRegistry(unittest.TestCase):
    """Test global singleton registry behavior."""

    def test_get_worker_presenter_registry_returns_singleton(self) -> None:
        """Test that get_worker_presenter_registry returns the same instance."""
        registry1 = get_worker_presenter_registry()
        registry2 = get_worker_presenter_registry()
        self.assertIs(registry1, registry2)

    def test_global_registry_has_spade_presenter(self) -> None:
        """Test that the global registry has SPADE-BDI presenter registered."""
        registry = get_worker_presenter_registry()
        self.assertTrue("spade_bdi_worker" in registry)
        presenter = registry.get("spade_bdi_worker")
        self.assertIsNotNone(presenter)
        if presenter is not None:
            self.assertEqual(presenter.id, "spade_bdi_worker")
        exported = ExportedSpadePresenter()
        self.assertEqual(exported.id, "spade_bdi_worker")

    def test_global_registry_has_cleanrl_presenter(self) -> None:
        """Test that the global registry has CleanRL presenter registered."""
        registry = get_worker_presenter_registry()
        self.assertTrue("cleanrl_worker" in registry)
        presenter = registry.get("cleanrl_worker")
        self.assertIsNotNone(presenter)
        if presenter is not None:
            self.assertEqual(presenter.id, "cleanrl_worker")
        exported = ExportedCleanPresenter()
        self.assertEqual(exported.id, "cleanrl_worker")

class TestSpadeBdiWorkerPresenterBasics(unittest.TestCase):
    """Test SpadeBdiWorkerPresenter basic properties and protocol compliance."""

    def setUp(self) -> None:
        """Create a presenter instance for testing."""
        self.presenter = SpadeBdiWorkerPresenter()

    def test_presenter_id(self) -> None:
        """Test that presenter has correct ID."""
        self.assertEqual(self.presenter.id, "spade_bdi_worker")

    def test_presenter_implements_protocol(self) -> None:
        """Test that presenter implements WorkerPresenter protocol."""
        self.assertTrue(hasattr(self.presenter, "id"))
        self.assertTrue(callable(getattr(self.presenter, "build_train_request")))
        self.assertTrue(callable(getattr(self.presenter, "create_tabs")))

    def test_extract_agent_id_from_config(self) -> None:
        """Test extract_agent_id utility method."""
        config = {
            "metadata": {
                "worker": {
                    "agent_id": "test_agent_123",
                    "config": {"game_id": "FrozenLake-v1"},
                }
            }
        }
        agent_id = SpadeBdiWorkerPresenter.extract_agent_id(config)
        self.assertEqual(agent_id, "test_agent_123")

    def test_extract_agent_id_fallback(self) -> None:
        """Test extract_agent_id fallback to worker config."""
        config = {
            "metadata": {
                "worker": {
                    "config": {"agent_id": "fallback_agent"},
                }
            }
        }
        agent_id = SpadeBdiWorkerPresenter.extract_agent_id(config)
        self.assertEqual(agent_id, "fallback_agent")

    def test_extract_agent_id_missing(self) -> None:
        """Test extract_agent_id returns None when not found."""
        config = {"metadata": {}}
        agent_id = SpadeBdiWorkerPresenter.extract_agent_id(config)
        self.assertIsNone(agent_id)

    def test_extract_agent_id_handles_exceptions(self) -> None:
        """Test extract_agent_id handles malformed config gracefully."""
        config: Any = "not_a_dict"  # type: ignore
        agent_id = SpadeBdiWorkerPresenter.extract_agent_id(config)
        self.assertIsNone(agent_id)


class TestBuildTrainRequest(unittest.TestCase):
    """Test train request building from policy files."""

    def setUp(self) -> None:
        """Create presenter and temp directory for policy files."""
        self.presenter = SpadeBdiWorkerPresenter()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up temp directory."""
        self.temp_dir.cleanup()

    def _create_policy_file(self, metadata: dict[str, Any] | None = None) -> Path:
        """Helper to create a test policy file."""
        if metadata is None:
            metadata = {
                "game_id": "FrozenLake-v1",
                "seed": 42,
                "eval_episodes": 5,
                "max_steps": 100,
                "algorithm": "QLearning",
                "agent_id": "test_agent",
            }
        policy_file = self.temp_path / "test_policy.json"
        policy_file.write_text(json.dumps({"metadata": metadata}))
        return policy_file

    def test_build_train_request_success(self) -> None:
        """Test successful train request building."""
        policy_file = self._create_policy_file()
        config = self.presenter.build_train_request(policy_file, current_game=None)

        self.assertIsNotNone(config)
        self.assertIn("run_name", config)
        self.assertIn("metadata", config)
        self.assertIn("environment", config)
        self.assertIn("resources", config)
        self.assertIn("artifacts", config)

        # Verify metadata payload
        metadata = config["metadata"]
        self.assertIn("ui", metadata)
        self.assertIn("worker", metadata)
        self.assertEqual(metadata["ui"]["algorithm"], "QLearning")
        self.assertEqual(metadata["worker"]["agent_id"], "test_agent")
        self.assertEqual(metadata["worker"]["schema_id"], "telemetry.step.default")
        self.assertEqual(metadata["worker"]["schema_version"], 1)
        worker_config = metadata["worker"]["config"]
        self.assertEqual(worker_config["schema_id"], "telemetry.step.default")
        self.assertEqual(worker_config["schema_version"], 1)
        self.assertIsInstance(worker_config["schema_definition"], dict)
        self.assertEqual(metadata["ui"]["schema_id"], "telemetry.step.default")
        self.assertEqual(metadata["ui"]["schema_version"], 1)

    def test_build_train_request_missing_file(self) -> None:
        """Test train request fails with missing policy file."""
        missing_file = self.temp_path / "nonexistent.json"
        with self.assertRaises(FileNotFoundError):
            self.presenter.build_train_request(missing_file, current_game=None)

    def test_build_train_request_missing_game_id(self) -> None:
        """Test train request fails when game_id cannot be determined."""
        metadata = {"seed": 42}  # No game_id
        policy_file = self._create_policy_file(metadata)

        with self.assertRaises(ValueError) as ctx:
            self.presenter.build_train_request(policy_file, current_game=None)
        self.assertIn("Game environment", str(ctx.exception))

    def test_build_train_request_uses_current_game_fallback(self) -> None:
        """Test that current_game parameter is used as fallback for game_id."""
        metadata = {
            "seed": 42,
            "eval_episodes": 5,
            "max_steps": 100,
            "algorithm": "QLearning",
        }
        policy_file = self._create_policy_file(metadata)

        # Mock GameId enum
        mock_game = Mock()
        mock_game.value = "CliffWalking-v1"

        config = self.presenter.build_train_request(policy_file, current_game=mock_game)
        self.assertIsNotNone(config)
        worker_config = config["metadata"]["worker"]["config"]
        self.assertEqual(worker_config["game_id"], "CliffWalking-v1")
        self.assertEqual(worker_config["schema_id"], "telemetry.step.default")
        self.assertEqual(config["metadata"]["worker"]["schema_id"], "telemetry.step.default")

    def test_extract_metadata_from_config(self) -> None:
        """Test extract_metadata utility method."""
        config = {
            "run_name": "test_run_123",
            "metadata": {
                "ui": {"algorithm": "QLearning", "source_policy": "/path/to/policy.json"},
                "worker": {
                    "module": "spade_bdi_worker.worker",
                    "agent_id": "agent_1",
                    "config": {
                        "game_id": "FrozenLake-v1",
                        "run_id": "test_run_123",
                    },
                },
            },
        }

        extracted = self.presenter.extract_metadata(config)
        self.assertEqual(extracted["agent_id"], "agent_1")
        self.assertEqual(extracted["game_id"], "FrozenLake-v1")
        self.assertEqual(extracted["run_id"], "test_run_123")
        self.assertEqual(extracted["algorithm"], "QLearning")


class TestTabFactory(unittest.TestCase):
    """Test TabFactory for creating worker UI tabs."""

    def setUp(self) -> None:
        """Create factory instance."""
        self.factory = TabFactory()

    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentReplayTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineGridTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineRawTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineVideoTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.get_service_locator")
    def test_create_tabs_toytext_environment(
        self,
        mock_get_locator: Any,
        mock_video_tab: Any,
        mock_raw_tab: Any,
        mock_grid_tab: Any,
        mock_replay_tab: Any,
        mock_online_tab: Any,
    ) -> None:
        """Test tab creation for ToyText environment (no video tab)."""
        mock_locator = Mock()
        mock_renderer = Mock()
        mock_locator.resolve.return_value = mock_renderer
        mock_get_locator.return_value = mock_locator

        mock_online_tab.return_value = Mock()
        mock_replay_tab.return_value = Mock()
        mock_grid_tab.return_value = Mock()
        mock_raw_tab.return_value = Mock()

        payload = {
            "game_id": "FrozenLake-v1",
            "episode_index": 1,
            "step_index": 0,
        }

        mock_parent = Mock()
        tabs = self.factory.create_tabs("run_1", "agent_1", payload, mock_parent)

        # ToyText should create 4 tabs (no video)
        self.assertEqual(len(tabs), 4)
        mock_video_tab.assert_not_called()

    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentReplayTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineGridTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineRawTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineVideoTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.get_service_locator")
    def test_create_tabs_visual_environment(
        self,
        mock_get_locator: Any,
        mock_video_tab: Any,
        mock_raw_tab: Any,
        mock_grid_tab: Any,
        mock_replay_tab: Any,
        mock_online_tab: Any,
    ) -> None:
        """Test tab creation for visual environment (includes video tab)."""
        mock_locator = Mock()
        mock_renderer = Mock()
        mock_locator.resolve.return_value = mock_renderer
        mock_get_locator.return_value = mock_locator

        mock_online_tab.return_value = Mock()
        mock_replay_tab.return_value = Mock()
        mock_grid_tab.return_value = Mock()
        mock_raw_tab.return_value = Mock()
        mock_video_tab.return_value = Mock()

        payload = {
            "game_id": "Atari-v1",  # Visual environment
            "episode_index": 1,
            "step_index": 0,
        }

        mock_parent = Mock()
        tabs = self.factory.create_tabs("run_1", "agent_1", payload, mock_parent)

        # Visual should create 5 tabs (including video)
        self.assertEqual(len(tabs), 5)
        mock_video_tab.assert_called_once()

    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentReplayTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineGridTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineRawTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineVideoTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.get_service_locator")
    def test_create_tabs_handles_invalid_game_id(
        self,
        mock_get_locator: Any,
        mock_video_tab: Any,
        mock_raw_tab: Any,
        mock_grid_tab: Any,
        mock_replay_tab: Any,
        mock_online_tab: Any,
    ) -> None:
        """Test that invalid game_id is handled gracefully."""
        mock_locator = Mock()
        mock_renderer = Mock()
        mock_locator.resolve.return_value = mock_renderer
        mock_get_locator.return_value = mock_locator

        mock_online_tab.return_value = Mock()
        mock_replay_tab.return_value = Mock()
        mock_grid_tab.return_value = Mock()
        mock_raw_tab.return_value = Mock()
        mock_video_tab.return_value = Mock()

        payload = {
            "game_id": "InvalidGame-v999",
            "episode_index": 1,
            "step_index": 0,
        }

        mock_parent = Mock()
        # Should not raise, should default to visual (5 tabs)
        tabs = self.factory.create_tabs("run_1", "agent_1", payload, mock_parent)
        self.assertEqual(len(tabs), 5)

    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentReplayTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineGridTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineRawTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.AgentOnlineVideoTab")
    @patch("gym_gui.ui.widgets.spade_bdi_worker_tabs.factory.get_service_locator")
    def test_create_tabs_case_insensitive_game_id(
        self,
        mock_get_locator: Any,
        mock_video_tab: Any,
        mock_raw_tab: Any,
        mock_grid_tab: Any,
        mock_replay_tab: Any,
        mock_online_tab: Any,
    ) -> None:
        """Test that game_id detection is case-insensitive."""
        mock_locator = Mock()
        mock_renderer = Mock()
        mock_locator.resolve.return_value = mock_renderer
        mock_get_locator.return_value = mock_locator

        mock_online_tab.return_value = Mock()
        mock_replay_tab.return_value = Mock()
        mock_grid_tab.return_value = Mock()
        mock_raw_tab.return_value = Mock()

        payload = {
            "game_id": "CLIFFWALKING-v1",  # Uppercase
            "episode_index": 1,
            "step_index": 0,
        }

        mock_parent = Mock()
        tabs = self.factory.create_tabs("run_1", "agent_1", payload, mock_parent)
        # Should detect as ToyText and create 4 tabs
        self.assertEqual(len(tabs), 4)
        mock_video_tab.assert_not_called()


class TestCreateTabsIntegration(unittest.TestCase):
    """Integration tests for tab creation via presenter."""

    def setUp(self) -> None:
        """Set up presenter for integration tests."""
        self.presenter = SpadeBdiWorkerPresenter()

    @patch.object(TabFactory, "create_tabs")
    def test_presenter_create_tabs_delegates_to_factory(
        self, mock_factory_create: Any
    ) -> None:
        """Test that presenter delegates tab creation to TabFactory."""
        mock_tabs = [Mock(), Mock(), Mock(), Mock()]
        mock_factory_create.return_value = mock_tabs

        mock_parent = Mock()
        payload = {"game_id": "FrozenLake-v1"}

        tabs = self.presenter.create_tabs("run_1", "agent_1", payload, mock_parent)

        # Verify factory was called with correct args
        mock_factory_create.assert_called_once_with(
            "run_1", "agent_1", payload, mock_parent
        )
        self.assertEqual(tabs, mock_tabs)


class TestWorkerPresenterProtocolCompliance(unittest.TestCase):
    """Test that SpadeBdiWorkerPresenter fully implements WorkerPresenter protocol."""

    def test_protocol_compliance(self) -> None:
        """Test that SpadeBdiWorkerPresenter implements all required methods."""
        presenter = SpadeBdiWorkerPresenter()

        # Check required attributes
        self.assertTrue(hasattr(presenter, "id"))
        self.assertEqual(isinstance(presenter.id, str), True)

        # Check required methods
        self.assertTrue(callable(presenter.build_train_request))
        self.assertTrue(callable(presenter.create_tabs))

        # For protocols with runtime_checkable, we can verify adherence
        self.assertIsInstance(presenter, WorkerPresenter)

    def test_cleanrl_presenter_protocol(self) -> None:
        """Test that CleanRL presenter satisfies protocol requirements."""
        presenter = CleanRlWorkerPresenter()
        self.assertEqual(presenter.id, "cleanrl_worker")
        self.assertTrue(callable(presenter.build_train_request))
        self.assertTrue(callable(presenter.create_tabs))
        self.assertIsInstance(presenter, WorkerPresenter)
        with self.assertRaises(NotImplementedError):
            presenter.build_train_request(policy_path=None, current_game=None)
        self.assertEqual(presenter.create_tabs("run", "agent", {}, None), [])


class TestRegistryIntegration(unittest.TestCase):
    """Integration tests for registry usage patterns."""

    def test_registry_workflow(self) -> None:
        """Test typical registry workflow: register -> lookup -> use."""
        registry = WorkerPresenterRegistry()
        presenter = SpadeBdiWorkerPresenter()

        # Register
        registry.register(presenter.id, presenter)

        # Lookup
        retrieved = registry.get(presenter.id)
        self.assertIsNotNone(retrieved)
        if retrieved is not None:
            self.assertEqual(retrieved.id, "spade_bdi_worker")
            # Use
            self.assertEqual(retrieved.id, presenter.id)

    def test_multiple_presenters_in_registry(self) -> None:
        """Test managing multiple presenters in registry."""
        registry = WorkerPresenterRegistry()

        presenter1 = SpadeBdiWorkerPresenter()
        presenter2 = Mock(spec=WorkerPresenter)
        presenter2.id = "future_worker"

        registry.register(presenter1.id, presenter1)
        registry.register(presenter2.id, presenter2)

        self.assertEqual(len(registry.available_workers()), 2)
        p1 = registry.get("spade_bdi_worker")
        p2 = registry.get("future_worker")
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        if p1 is not None:
            self.assertEqual(p1.id, "spade_bdi_worker")
        if p2 is not None:
            self.assertEqual(p2.id, "future_worker")


class _RecordingRenderTabs:
    """Minimal stand-in for RenderTabs that records dynamic tab additions."""

    def __init__(self) -> None:
        self._agent_tabs: dict[str, dict[str, QtWidgets.QWidget]] = {}

    def add_dynamic_tab(self, run_id: str, name: str, widget: QtWidgets.QWidget) -> None:
        self._agent_tabs.setdefault(run_id, {})[name] = widget


class TestAnalyticsTabManager(unittest.TestCase):
    """Validate TensorBoard and W&B tabs are created with expected widgets."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._qt_app = QtWidgets.QApplication.instance()
        if cls._qt_app is None:
            cls._qt_app = QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.render_tabs = _RecordingRenderTabs()
        self.parent = QtWidgets.QWidget()
        self.manager = AnalyticsTabManager(self.render_tabs, self.parent)

    def tearDown(self) -> None:
        self.parent.deleteLater()

    def test_tensorboard_tab_added(self) -> None:
        metadata = {
            "artifacts": {
                "tensorboard": {
                    "enabled": True,
                    "log_dir": "/tmp/tb-demo",
                }
            }
        }

        self.manager.ensure_tensorboard_tab("run-tb", "agent-1", metadata)

        agent_tabs = self.render_tabs._agent_tabs.get("run-tb", {})
        self.assertIn("TensorBoard-Agent-agent-1", agent_tabs)
        widget = agent_tabs["TensorBoard-Agent-agent-1"]
        self.assertIsInstance(widget, TensorboardArtifactTab)

    def test_wandb_tab_added(self) -> None:
        metadata = {
            "artifacts": {
                "wandb": {
                    "enabled": True,
                    "run_path": "abdulhamid97mousa/MOSAIC/runs/test123",
                }
            }
        }

        self.manager.ensure_wandb_tab("run-wandb", "agent-2", metadata)

        agent_tabs = self.render_tabs._agent_tabs.get("run-wandb", {})
        self.assertIn("WAB-Agent-agent-2", agent_tabs)
        widget = agent_tabs["WAB-Agent-agent-2"]
        self.assertIsInstance(widget, WandbArtifactTab)
        widget.append_status_line("wandb login succeeded")


if __name__ == "__main__":
    unittest.main()
