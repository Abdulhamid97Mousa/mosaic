"""Unit tests for the Multi-Operator system (Phase 6).

Tests cover:
- OperatorConfig dataclass validation
- MultiOperatorService operator management
- MultiOperatorService state management
- MultiOperatorService run_id management
- MultiOperatorService lifecycle helpers (start_all, stop_all)
"""

import unittest
from typing import Any, Dict

from gym_gui.services.operator import (
    MultiOperatorService,
    OperatorConfig,
)


class TestOperatorConfig(unittest.TestCase):
    """Test OperatorConfig dataclass."""

    def test_create_llm_config(self) -> None:
        """Should create a valid LLM operator config."""
        config = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="GPT-4 LLM",
            env_name="babyai",
            task="BabyAI-GoToRedBall-v0",
        )
        self.assertEqual(config.operator_id, "op_0")
        self.assertEqual(config.operator_type, "llm")
        self.assertEqual(config.worker_id, "barlog_worker")
        self.assertEqual(config.display_name, "GPT-4 LLM")
        self.assertEqual(config.env_name, "babyai")
        self.assertEqual(config.task, "BabyAI-GoToRedBall-v0")
        self.assertIsNone(config.run_id)

    def test_create_rl_config(self) -> None:
        """Should create a valid RL operator config."""
        config = OperatorConfig(
            operator_id="op_1",
            operator_type="rl",
            worker_id="cleanrl_worker",
            display_name="PPO Agent",
            env_name="CartPole-v1",
            task="CartPole-v1",
            settings={"algorithm": "ppo", "learning_rate": 0.0003},
        )
        self.assertEqual(config.operator_type, "rl")
        self.assertEqual(config.worker_id, "cleanrl_worker")
        self.assertEqual(config.settings["algorithm"], "ppo")

    def test_invalid_operator_type(self) -> None:
        """Should raise ValueError for invalid operator_type."""
        with self.assertRaises(ValueError) as ctx:
            OperatorConfig(
                operator_id="op_bad",
                operator_type="invalid",
                worker_id="test",
                display_name="Bad Op",
            )
        self.assertIn("operator_type must be 'llm' or 'rl'", str(ctx.exception))

    def test_with_run_id(self) -> None:
        """Should create a copy with the run_id set."""
        config = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="GPT-4 LLM",
        )
        new_config = config.with_run_id("run_abc123")

        # Original should be unchanged
        self.assertIsNone(config.run_id)

        # New config should have run_id
        self.assertEqual(new_config.run_id, "run_abc123")
        self.assertEqual(new_config.operator_id, config.operator_id)
        self.assertEqual(new_config.operator_type, config.operator_type)

    def test_default_env_and_task(self) -> None:
        """Should have sensible defaults for env_name and task."""
        config = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="Test LLM",
        )
        self.assertEqual(config.env_name, "babyai")
        self.assertEqual(config.task, "BabyAI-GoToRedBall-v0")

    def test_settings_default_to_empty_dict(self) -> None:
        """Settings should default to empty dict."""
        config = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="Test LLM",
        )
        self.assertEqual(config.settings, {})


class TestMultiOperatorServiceBasic(unittest.TestCase):
    """Test MultiOperatorService basic functionality."""

    def setUp(self) -> None:
        self.service = MultiOperatorService()

    def test_empty_service(self) -> None:
        """New service should have no operators."""
        self.assertEqual(self.service.operator_count, 0)
        self.assertEqual(self.service.get_active_operators(), {})
        self.assertEqual(self.service.get_operator_ids(), [])

    def test_add_operator(self) -> None:
        """Should add an operator successfully."""
        config = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="GPT-4 LLM",
        )
        self.service.add_operator(config)

        self.assertEqual(self.service.operator_count, 1)
        self.assertIn("op_0", self.service.get_operator_ids())

        retrieved = self.service.get_operator("op_0")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.display_name, "GPT-4 LLM")

    def test_add_multiple_operators(self) -> None:
        """Should manage multiple operators."""
        config1 = OperatorConfig(
            operator_id="llm_1",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="LLM Agent 1",
        )
        config2 = OperatorConfig(
            operator_id="rl_1",
            operator_type="rl",
            worker_id="cleanrl_worker",
            display_name="RL Agent 1",
        )

        self.service.add_operator(config1)
        self.service.add_operator(config2)

        self.assertEqual(self.service.operator_count, 2)
        self.assertIn("llm_1", self.service.get_operator_ids())
        self.assertIn("rl_1", self.service.get_operator_ids())

    def test_remove_operator(self) -> None:
        """Should remove an operator."""
        config = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="Test",
        )
        self.service.add_operator(config)
        self.assertEqual(self.service.operator_count, 1)

        self.service.remove_operator("op_0")

        self.assertEqual(self.service.operator_count, 0)
        self.assertIsNone(self.service.get_operator("op_0"))

    def test_remove_nonexistent_operator(self) -> None:
        """Should not raise when removing nonexistent operator."""
        # Should not raise
        self.service.remove_operator("nonexistent")
        self.assertEqual(self.service.operator_count, 0)

    def test_update_operator(self) -> None:
        """Should update an existing operator."""
        config1 = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="Original Name",
        )
        self.service.add_operator(config1)

        config2 = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="Updated Name",
        )
        self.service.update_operator(config2)

        retrieved = self.service.get_operator("op_0")
        self.assertEqual(retrieved.display_name, "Updated Name")

    def test_clear_operators(self) -> None:
        """Should clear all operators."""
        config1 = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="LLM 1",
        )
        config2 = OperatorConfig(
            operator_id="op_1",
            operator_type="rl",
            worker_id="cleanrl_worker",
            display_name="RL 1",
        )

        self.service.add_operator(config1)
        self.service.add_operator(config2)
        self.assertEqual(self.service.operator_count, 2)

        self.service.clear_operators()

        self.assertEqual(self.service.operator_count, 0)
        self.assertEqual(self.service.get_active_operators(), {})


class TestMultiOperatorServiceIdGeneration(unittest.TestCase):
    """Test MultiOperatorService operator ID generation."""

    def setUp(self) -> None:
        self.service = MultiOperatorService()

    def test_generate_unique_ids(self) -> None:
        """Should generate unique operator IDs."""
        id1 = self.service.generate_operator_id()
        id2 = self.service.generate_operator_id()
        id3 = self.service.generate_operator_id()

        self.assertNotEqual(id1, id2)
        self.assertNotEqual(id2, id3)
        self.assertNotEqual(id1, id3)

    def test_id_format(self) -> None:
        """Generated IDs should follow expected format."""
        id1 = self.service.generate_operator_id()
        self.assertTrue(id1.startswith("operator_"))

        # Verify it contains a number
        parts = id1.split("_")
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[1].isdigit())


class TestMultiOperatorServiceStateManagement(unittest.TestCase):
    """Test MultiOperatorService state management."""

    def setUp(self) -> None:
        self.service = MultiOperatorService()
        self.config = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="Test",
        )
        self.service.add_operator(self.config)

    def test_initial_state_is_pending(self) -> None:
        """New operators should start in pending state."""
        state = self.service.get_operator_state("op_0")
        self.assertEqual(state, "pending")

    def test_set_operator_state(self) -> None:
        """Should set operator state."""
        self.service.set_operator_state("op_0", "running")
        self.assertEqual(self.service.get_operator_state("op_0"), "running")

        self.service.set_operator_state("op_0", "stopped")
        self.assertEqual(self.service.get_operator_state("op_0"), "stopped")

    def test_invalid_state_raises(self) -> None:
        """Setting invalid state should raise ValueError."""
        with self.assertRaises(ValueError):
            self.service.set_operator_state("op_0", "invalid_state")

    def test_get_running_operators(self) -> None:
        """Should return only running operators."""
        config2 = OperatorConfig(
            operator_id="op_1",
            operator_type="rl",
            worker_id="cleanrl_worker",
            display_name="RL Agent",
        )
        config3 = OperatorConfig(
            operator_id="op_2",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="LLM 2",
        )

        self.service.add_operator(config2)
        self.service.add_operator(config3)

        # Set some to running
        self.service.set_operator_state("op_0", "running")
        self.service.set_operator_state("op_2", "running")
        # op_1 stays pending

        running = self.service.get_running_operators()

        self.assertEqual(len(running), 2)
        self.assertIn("op_0", running)
        self.assertIn("op_2", running)
        self.assertNotIn("op_1", running)


class TestMultiOperatorServiceRunIdManagement(unittest.TestCase):
    """Test MultiOperatorService run_id management."""

    def setUp(self) -> None:
        self.service = MultiOperatorService()
        self.config = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="Test",
        )
        self.service.add_operator(self.config)

    def test_assign_run_id(self) -> None:
        """Should assign run_id to operator."""
        self.service.assign_run_id("op_0", "run_abc123")

        self.assertEqual(self.service.get_run_id("op_0"), "run_abc123")

        # Config should also be updated
        updated_config = self.service.get_operator("op_0")
        self.assertEqual(updated_config.run_id, "run_abc123")

    def test_get_operator_by_run_id(self) -> None:
        """Should retrieve operator by run_id."""
        self.service.assign_run_id("op_0", "run_abc123")

        config = self.service.get_operator_by_run_id("run_abc123")

        self.assertIsNotNone(config)
        self.assertEqual(config.operator_id, "op_0")

    def test_get_operator_by_unknown_run_id(self) -> None:
        """Should return None for unknown run_id."""
        config = self.service.get_operator_by_run_id("nonexistent_run")
        self.assertIsNone(config)


class TestMultiOperatorServiceLifecycle(unittest.TestCase):
    """Test MultiOperatorService lifecycle helpers."""

    def setUp(self) -> None:
        self.service = MultiOperatorService()
        # Add multiple operators
        for i in range(3):
            config = OperatorConfig(
                operator_id=f"op_{i}",
                operator_type="llm" if i % 2 == 0 else "rl",
                worker_id="barlog_worker" if i % 2 == 0 else "cleanrl_worker",
                display_name=f"Agent {i}",
            )
            self.service.add_operator(config)

    def test_start_all_returns_pending(self) -> None:
        """start_all should return all pending operators."""
        to_start = self.service.start_all()

        self.assertEqual(len(to_start), 3)
        self.assertIn("op_0", to_start)
        self.assertIn("op_1", to_start)
        self.assertIn("op_2", to_start)

    def test_start_all_with_some_running(self) -> None:
        """start_all should only return pending operators."""
        # Set one to running
        self.service.set_operator_state("op_1", "running")

        to_start = self.service.start_all()

        self.assertEqual(len(to_start), 2)
        self.assertIn("op_0", to_start)
        self.assertIn("op_2", to_start)
        self.assertNotIn("op_1", to_start)

    def test_stop_all_running(self) -> None:
        """stop_all should stop all running operators."""
        # Set all to running
        for i in range(3):
            self.service.set_operator_state(f"op_{i}", "running")

        stopped = self.service.stop_all()

        self.assertEqual(len(stopped), 3)

        # All should now be stopped
        for i in range(3):
            self.assertEqual(
                self.service.get_operator_state(f"op_{i}"),
                "stopped"
            )

    def test_stop_all_with_mixed_states(self) -> None:
        """stop_all should only stop running operators."""
        # Mixed states
        self.service.set_operator_state("op_0", "running")
        self.service.set_operator_state("op_1", "stopped")
        self.service.set_operator_state("op_2", "running")

        stopped = self.service.stop_all()

        self.assertEqual(len(stopped), 2)
        self.assertIn("op_0", stopped)
        self.assertIn("op_2", stopped)
        self.assertNotIn("op_1", stopped)


class TestMultiOperatorServiceRemovalCleansState(unittest.TestCase):
    """Test that removing an operator cleans up all associated state."""

    def setUp(self) -> None:
        self.service = MultiOperatorService()
        config = OperatorConfig(
            operator_id="op_0",
            operator_type="llm",
            worker_id="barlog_worker",
            display_name="Test",
        )
        self.service.add_operator(config)
        self.service.assign_run_id("op_0", "run_abc123")
        self.service.set_operator_state("op_0", "running")

    def test_removal_cleans_run_id(self) -> None:
        """Removing operator should clean up run_id mapping."""
        self.service.remove_operator("op_0")

        self.assertIsNone(self.service.get_run_id("op_0"))
        self.assertIsNone(self.service.get_operator_by_run_id("run_abc123"))

    def test_removal_cleans_state(self) -> None:
        """Removing operator should clean up state."""
        self.service.remove_operator("op_0")

        self.assertIsNone(self.service.get_operator_state("op_0"))


if __name__ == "__main__":
    unittest.main()
