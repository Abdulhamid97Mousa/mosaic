"""Comprehensive tests for multi-episode telemetry validation.

This test suite validates that:
1. Episode signals are emitted after each episode completes
2. Multi-episode telemetry is correctly logged
3. Episode statistics are accurate (reward, steps, success)
4. Telemetry events are properly sequenced
5. Worker-side validation of episode data
"""

from __future__ import annotations

import json
import logging
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ..adapters import create_adapter
from ..core.config import RunConfig, PolicyStrategy
from ..core.runtime import HeadlessTrainer
from ..core.telemetry_worker import TelemetryEmitter

logger = logging.getLogger(__name__)


class TestMultiEpisodeTelemetry:
    """Test suite for multi-episode telemetry validation."""

    @pytest.fixture
    def temp_policy_dir(self):
        """Create temporary directory for policy storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def run_config(self, temp_policy_dir):
        """Create a test run configuration."""
        return RunConfig(
            run_id="test-run-001",
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

    @pytest.fixture
    def telemetry_stream(self):
        """Create a StringIO stream to capture telemetry."""
        return StringIO()

    def test_episode_telemetry_emission(self, run_config, telemetry_stream):
        """Test that episode telemetry is emitted after each episode."""
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0, "Training should complete successfully"

        # Parse telemetry
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]

        # Validate events
        assert len(events) > 0, "Should have emitted telemetry events"

        # Find episode events
        episode_events = [e for e in events if e.get("type") == "episode"]
        assert len(episode_events) == run_config.max_episodes, (
            f"Should have {run_config.max_episodes} episode events, "
            f"got {len(episode_events)}"
        )

        logger.info(f"✓ Emitted {len(episode_events)} episode events")

    def test_episode_statistics_accuracy(self, run_config, telemetry_stream):
        """Test that episode statistics are accurate."""
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Parse telemetry
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]

        # Validate episode events have required fields
        episode_events = [e for e in events if e.get("type") == "episode"]
        for episode_event in episode_events:
            assert "run_id" in episode_event, "Episode event missing run_id"
            assert "episode" in episode_event, "Episode event missing episode"
            assert "episode_index" in episode_event, "Episode event missing episode_index"
            assert "agent_id" in episode_event, "Episode event missing agent_id"
            assert "reward" in episode_event, "Episode event missing reward"
            assert "steps" in episode_event, "Episode event missing steps"
            assert "success" in episode_event, "Episode event missing success"
            assert "ts" in episode_event, "Episode event missing timestamp"

            # Validate field types
            assert isinstance(episode_event["run_id"], str)
            assert isinstance(episode_event["episode"], int)
            assert isinstance(episode_event["reward"], (int, float))
            assert isinstance(episode_event["steps"], int)
            assert isinstance(episode_event["success"], bool)

            logger.info(
                f"✓ Episode {episode_event['episode']}: "
                f"reward={episode_event['reward']}, "
                f"steps={episode_event['steps']}, "
                f"success={episode_event['success']}"
            )

    def test_step_and_episode_sequencing(self, run_config, telemetry_stream):
        """Test that steps and episodes are properly sequenced."""
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Parse telemetry
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]

        # Validate sequencing
        step_events = [e for e in events if e.get("type") == "step"]
        episode_events = [e for e in events if e.get("type") == "episode"]

        assert len(step_events) > 0, "Should have step events"
        assert len(episode_events) > 0, "Should have episode events"

        # Group steps by episode_index (0-based)
        steps_by_episode: Dict[int, List[Dict[str, Any]]] = {}
        for step_event in step_events:
            # Use episode_index (0-based) instead of episode (display value)
            episode_idx = step_event.get("episode_index", step_event.get("episode"))
            if episode_idx not in steps_by_episode:
                steps_by_episode[episode_idx] = []
            steps_by_episode[episode_idx].append(step_event)

        # Validate each episode has steps
        for episode_idx in range(run_config.max_episodes):
            assert episode_idx in steps_by_episode, (
                f"Episode {episode_idx} has no steps. Available keys: {list(steps_by_episode.keys())}"
            )
            steps = steps_by_episode[episode_idx]
            assert len(steps) > 0, f"Episode {episode_idx} should have steps"

            # Validate step indices are sequential
            step_indices = [s.get("step_index") for s in steps]
            assert step_indices == list(range(len(steps))), (
                f"Episode {episode_idx} steps not sequential: {step_indices}"
            )

            logger.info(
                f"✓ Episode {episode_idx}: {len(steps)} steps with sequential indices"
            )

    def test_multi_episode_run_completion(self, run_config, telemetry_stream):
        """Test that multi-episode run completes with proper lifecycle."""
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Parse telemetry
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]

        # Validate lifecycle
        run_started = [e for e in events if e.get("type") == "run_started"]
        run_completed = [e for e in events if e.get("type") == "run_completed"]

        assert len(run_started) == 1, "Should have exactly one run_started event"
        assert len(run_completed) == 1, "Should have exactly one run_completed event"

        # Validate run_completed status
        completed_event = run_completed[0]
        assert completed_event.get("status") == "completed", (
            f"Run should complete with status 'completed', "
            f"got '{completed_event.get('status')}'"
        )

        logger.info("✓ Multi-episode run completed successfully")

    def test_episode_reward_accumulation(self, run_config, telemetry_stream):
        """Test that episode rewards are properly accumulated."""
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Parse telemetry
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]

        # Get step and episode events
        step_events = [e for e in events if e.get("type") == "step"]
        episode_events = [e for e in events if e.get("type") == "episode"]

        # Group steps by episode and calculate rewards
        for episode_event in episode_events:
            episode_idx = episode_event.get("episode")
            episode_reward = episode_event.get("reward")

            # Find all steps for this episode
            episode_steps = [
                s for s in step_events if s.get("episode") == episode_idx
            ]

            # Calculate accumulated reward from steps
            accumulated_reward = sum(s.get("reward", 0) for s in episode_steps)

            # Validate reward matches
            assert abs(accumulated_reward - episode_reward) < 0.001, (
                f"Episode {episode_idx} reward mismatch: "
                f"accumulated={accumulated_reward}, episode={episode_reward}"
            )

            logger.info(
                f"✓ Episode {episode_idx}: reward={episode_reward} "
                f"(accumulated from {len(episode_steps)} steps)"
            )


class TestWorkerSideValidation:
    """Test suite for worker-side telemetry validation."""

    @pytest.fixture
    def temp_policy_dir(self):
        """Create temporary directory for policy storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def run_config(self, temp_policy_dir):
        """Create a test run configuration."""
        return RunConfig(
            run_id="worker-test-001",
            agent_id="worker-agent",
            game_id="FrozenLake-v1",
            max_episodes=2,
            max_steps_per_episode=50,
            seed=123,
            policy_strategy=PolicyStrategy.TRAIN,
            policy_path=temp_policy_dir / "policy.pkl",
            headless=True,
            capture_video=False,
            extra={},
        )

    def test_worker_telemetry_format(self, run_config):
        """Test that worker emits properly formatted telemetry."""
        telemetry_stream = StringIO()
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Parse and validate telemetry format
        telemetry_stream.seek(0)
        for line_num, line in enumerate(telemetry_stream, 1):
            if not line.strip():
                continue

            try:
                event = json.loads(line)
                assert isinstance(event, dict), f"Line {line_num}: Event must be dict"
                assert "type" in event, f"Line {line_num}: Missing 'type' field"
                assert "ts" in event, f"Line {line_num}: Missing 'ts' field"
            except json.JSONDecodeError as e:
                pytest.fail(f"Line {line_num}: Invalid JSON: {e}")

        logger.info("✓ All telemetry events are properly formatted JSON")

    def test_worker_episode_count_validation(self, run_config):
        """Test that worker emits correct number of episodes."""
        telemetry_stream = StringIO()
        emitter = TelemetryEmitter(telemetry_stream)
        adapter = create_adapter(run_config.game_id)
        trainer = HeadlessTrainer(adapter, run_config, emitter)

        # Run training
        result = trainer.run()
        assert result == 0

        # Count episodes
        telemetry_stream.seek(0)
        events = [json.loads(line) for line in telemetry_stream if line.strip()]
        episode_count = len([e for e in events if e.get("type") == "episode"])

        assert episode_count == run_config.max_episodes, (
            f"Expected {run_config.max_episodes} episodes, got {episode_count}"
        )

        logger.info(f"✓ Worker emitted {episode_count} episodes as expected")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

