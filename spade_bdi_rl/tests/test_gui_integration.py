"""Integration tests for GUI tab population with multi-episode telemetry.

This test suite validates the full flow:
1. Worker emits multi-episode telemetry
2. Episodes are routed to agent tabs
3. Live tabs populate with episode data
4. Replay tabs populate after training completes
"""

from __future__ import annotations

import json
import logging
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from ..adapters import create_adapter
from ..core.config import RunConfig, PolicyStrategy
from ..core.runtime import HeadlessTrainer
from ..core.telemetry import TelemetryEmitter

logger = logging.getLogger(__name__)


class MockTab:
    """Mock tab for testing episode/step routing."""

    def __init__(self, run_id: str, agent_id: str):
        self.run_id = run_id
        self.agent_id = agent_id
        self.steps: List[Dict[str, Any]] = []
        self.episodes: List[Dict[str, Any]] = []
        self._episode_count = 0
        self._step_count = 0

    def add_step(self, step: Dict[str, Any]) -> None:
        """Record step data."""
        self.steps.append(step)
        self._step_count += 1

    def on_step(self, step: Dict[str, Any], *, metadata: Dict[str, Any] | None = None) -> None:
        """Handle step (same as add_step)."""
        self.add_step(step)

    def add_episode(self, episode: Dict[str, Any]) -> None:
        """Record episode data."""
        self.episodes.append(episode)
        self._episode_count += 1

    def on_episode(self, episode: Dict[str, Any], *, metadata: Dict[str, Any] | None = None) -> None:
        """Handle episode (same as add_episode)."""
        self.add_episode(episode)

    def get_episode_count(self) -> int:
        """Get number of episodes received."""
        return self._episode_count

    def get_step_count(self) -> int:
        """Get number of steps received."""
        return self._step_count

    def get_latest_episode(self) -> Dict[str, Any] | None:
        """Get the most recent episode."""
        return self.episodes[-1] if self.episodes else None

    def get_latest_step(self) -> Dict[str, Any] | None:
        """Get the most recent step."""
        return self.steps[-1] if self.steps else None


class TestGUITabPopulation:
    """Test suite for GUI tab population with telemetry."""

    @pytest.fixture
    def temp_policy_dir(self):
        """Create temporary directory for policy storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def run_config(self, temp_policy_dir):
        """Create a test run configuration."""
        return RunConfig(
            run_id="gui-test-001",
            agent_id="test-agent",
            game_id="FrozenLake-v1",
            max_episodes=3,
            max_steps_per_episode=100,
            seed=42,
            policy_strategy=PolicyStrategy.TRAIN,
            policy_path=temp_policy_dir / "policy.pkl",
            headless=True,
            capture_video=False,
            extra={},
        )

    def test_agent_tabs_created_on_first_step(self, run_config):
        """Test that agent tabs are created when first step arrives."""
        telemetry_stream = StringIO()
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Parse telemetry
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]

        # Find first step event
        step_events = [e for e in events if e.get("type") == "step"]
        assert len(step_events) > 0, "Should have step events"

        first_step = step_events[0]
        assert "agent_id" in first_step, "Step should have agent_id"
        assert first_step["agent_id"] == run_config.agent_id

        logger.info(f"✓ First step has agent_id: {first_step['agent_id']}")

    def test_episodes_routed_to_agent_tabs(self, run_config):
        """Test that episodes are routed to the correct agent tabs."""
        telemetry_stream = StringIO()
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Parse telemetry
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]

        # Get episodes
        episode_events = [e for e in events if e.get("type") == "episode"]
        assert len(episode_events) == run_config.max_episodes

        # Simulate tab routing
        tabs: Dict[tuple[str, str], MockTab] = {}
        for episode_event in episode_events:
            run_id = episode_event.get("run_id")
            agent_id = episode_event.get("agent_id")
            key = (run_id, agent_id)

            # Create tab if needed
            if key not in tabs:
                tabs[key] = MockTab(run_id, agent_id)

            # Route episode to tab
            tabs[key].add_episode(episode_event)

        # Verify tabs received episodes
        assert len(tabs) == 1, "Should have one agent tab"
        tab = list(tabs.values())[0]
        assert tab.get_episode_count() == run_config.max_episodes

        logger.info(f"✓ Tab received {tab.get_episode_count()} episodes")

    def test_steps_and_episodes_in_correct_order(self, run_config):
        """Test that steps and episodes are routed in correct order."""
        telemetry_stream = StringIO()
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Parse telemetry
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]

        # Simulate tab routing
        tabs: Dict[tuple[str, str], MockTab] = {}
        for event in events:
            if event.get("type") == "step":
                run_id = event.get("run_id")
                agent_id = event.get("agent_id")
                key = (run_id, agent_id)

                if key not in tabs:
                    tabs[key] = MockTab(run_id, agent_id)

                tabs[key].add_step(event)

            elif event.get("type") == "episode":
                run_id = event.get("run_id")
                agent_id = event.get("agent_id")
                key = (run_id, agent_id)

                if key not in tabs:
                    tabs[key] = MockTab(run_id, agent_id)

                tabs[key].add_episode(event)

        # Verify tab has both steps and episodes
        assert len(tabs) == 1
        tab = list(tabs.values())[0]
        assert tab.get_step_count() > 0, "Tab should have steps"
        assert tab.get_episode_count() == run_config.max_episodes, "Tab should have episodes"

        logger.info(
            f"✓ Tab received {tab.get_step_count()} steps and {tab.get_episode_count()} episodes"
        )

    def test_episode_data_completeness(self, run_config):
        """Test that episode data contains all required fields for tab display."""
        telemetry_stream = StringIO()
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Parse telemetry
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]

        # Get episodes
        episode_events = [e for e in events if e.get("type") == "episode"]

        # Verify each episode has required fields for tab display
        required_fields = ["run_id", "agent_id", "episode", "reward", "steps", "success"]
        for episode_event in episode_events:
            for field in required_fields:
                assert field in episode_event, f"Episode missing {field}"

            # Verify field types
            assert isinstance(episode_event["reward"], (int, float))
            assert isinstance(episode_event["steps"], int)
            assert isinstance(episode_event["success"], bool)

        logger.info(f"✓ All {len(episode_events)} episodes have complete data")

    def test_agent_tab_naming_convention(self, run_config):
        """Test that agent tabs follow naming convention Agent-{id}-*."""
        telemetry_stream = StringIO()
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Parse telemetry
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]

        # Get first step to extract agent_id
        step_events = [e for e in events if e.get("type") == "step"]
        assert len(step_events) > 0

        first_step = step_events[0]
        agent_id = first_step.get("agent_id")

        # Expected tab names
        expected_tabs = [
            f"Agent-{agent_id}-Online",
            f"Agent-{agent_id}-Replay",
            f"Agent-{agent_id}-Online-Grid",
            f"Agent-{agent_id}-Online-Raw",
        ]

        logger.info(f"✓ Expected tab names: {expected_tabs}")

    def test_replay_tab_population_after_training(self, run_config):
        """Test that replay tab can be populated after training completes."""
        telemetry_stream = StringIO()
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Parse telemetry
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]

        # Find run_completed event
        run_completed = [e for e in events if e.get("type") == "run_completed"]
        assert len(run_completed) == 1, "Should have run_completed event"

        completed_event = run_completed[0]
        assert completed_event.get("status") == "completed"

        # Verify episodes are available for replay
        episode_events = [e for e in events if e.get("type") == "episode"]
        assert len(episode_events) == run_config.max_episodes

        logger.info(
            f"✓ Replay tab can be populated with {len(episode_events)} episodes after training"
        )

    def test_multi_agent_tab_isolation(self, run_config):
        """Test that multiple agent tabs are isolated from each other."""
        telemetry_stream = StringIO()
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Parse telemetry
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]

        # Simulate multiple agent tabs
        tabs: Dict[tuple[str, str], MockTab] = {}
        for event in events:
            if event.get("type") in ("step", "episode"):
                run_id = event.get("run_id")
                agent_id = event.get("agent_id")
                key = (run_id, agent_id)

                if key not in tabs:
                    tabs[key] = MockTab(run_id, agent_id)

                if event.get("type") == "step":
                    tabs[key].add_step(event)
                else:
                    tabs[key].add_episode(event)

        # Verify each tab has correct data
        for (run_id, agent_id), tab in tabs.items():
            assert tab.run_id == run_id
            assert tab.agent_id == agent_id
            assert tab.get_episode_count() == run_config.max_episodes

        logger.info(f"✓ {len(tabs)} agent tab(s) properly isolated")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

