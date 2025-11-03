"""Test Blackjack replay functionality and human episode filtering.

This test verifies that:
1. Telemetry correctly records Blackjack episodes
2. Replay loader can load Blackjack episodes
3. Human episode filtering works correctly for Blackjack
4. Episode metadata includes control_mode for filtering
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from gym_gui.core.enums import GameId, ControlMode
from gym_gui.core.adapters.toy_text import BlackjackAdapter
from gym_gui.config.game_configs import DEFAULT_BLACKJACK_CONFIG
from gym_gui.core.data_model import EpisodeRollup
from gym_gui.replays.loader import EpisodeReplayLoader
from gym_gui.telemetry.sqlite_store import TelemetrySQLiteStore
from gym_gui.services.telemetry import TelemetryService


class TestBlackjackReplay:
    """Test Blackjack episode recording and replay."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_telemetry.db"
            yield db_path

    @pytest.fixture
    def telemetry_service(self, temp_db_path):
        """Create telemetry service with temporary database."""
        store = TelemetrySQLiteStore(temp_db_path)  # Pass Path object directly
        service = TelemetryService(store)
        yield service
        store.close()

    @pytest.fixture
    def replay_loader(self, telemetry_service):
        """Create replay loader with telemetry service."""
        return EpisodeReplayLoader(telemetry_service)

    def test_blackjack_episode_records_with_control_mode(self, telemetry_service):
        """Test that Blackjack episodes are recorded with control_mode metadata."""
        adapter = BlackjackAdapter(game_config=DEFAULT_BLACKJACK_CONFIG)
        adapter.load()
        
        # Simulate human-controlled episode
        episode_id = "blackjack-test-001"
        episode_metadata = {
            "game_id": GameId.BLACKJACK.value,
            "control_mode": ControlMode.HUMAN_ONLY.value,
            "seed": 42,
            "episode_index": 0,
        }
        
        telemetry_service.begin_episode(episode_id, metadata=episode_metadata)
        
        # Record initial step
        initial_step = adapter.reset(seed=42)
        step_record = telemetry_service.record_step(
            episode_id=episode_id,
            step_index=0,
            observation=initial_step.observation,
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
            render_payload=adapter.render(),
            agent_id="human",
            timestamp=datetime.utcnow(),
        )
        
        assert step_record is not None
        
        # Record a game step (Stick action)
        step_result = adapter.step(0)  # Stick
        step_record = telemetry_service.record_step(
            episode_id=episode_id,
            step_index=1,
            observation=step_result.observation,
            action=0,
            reward=step_result.reward,
            terminated=step_result.terminated,
            truncated=step_result.truncated,
            info=step_result.info,
            render_payload=adapter.render(),
            agent_id="human",
            timestamp=datetime.utcnow(),
        )
        
        assert step_record is not None
        
        # Finalize episode
        telemetry_service.finalize_episode(
            episode_id=episode_id,
            total_reward=step_result.reward,
            terminated=step_result.terminated,
            truncated=step_result.truncated,
            timestamp=datetime.utcnow(),
        )
        
        # Verify episode was recorded
        episodes = list(telemetry_service.recent_episodes())
        assert len(episodes) == 1
        assert episodes[0].episode_id == episode_id
        assert episodes[0].metadata["control_mode"] == ControlMode.HUMAN_ONLY.value

    def test_replay_loader_loads_blackjack_episode(self, telemetry_service, replay_loader):
        """Test that replay loader can load Blackjack episodes."""
        adapter = BlackjackAdapter(game_config=DEFAULT_BLACKJACK_CONFIG)
        adapter.load()
        
        episode_id = "blackjack-test-002"
        episode_metadata = {
            "game_id": GameId.BLACKJACK.value,
            "control_mode": ControlMode.HUMAN_ONLY.value,
            "seed": 123,
        }
        
        telemetry_service.begin_episode(episode_id, metadata=episode_metadata)
        
        # Record multiple steps
        initial_step = adapter.reset(seed=123)
        telemetry_service.record_step(
            episode_id=episode_id,
            step_index=0,
            observation=initial_step.observation,
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
            render_payload=adapter.render(),
            agent_id="human",
            timestamp=datetime.utcnow(),
        )
        
        # Hit action
        step1 = adapter.step(1)
        telemetry_service.record_step(
            episode_id=episode_id,
            step_index=1,
            observation=step1.observation,
            action=1,
            reward=step1.reward,
            terminated=step1.terminated,
            truncated=step1.truncated,
            info=step1.info,
            render_payload=adapter.render(),
            agent_id="human",
            timestamp=datetime.utcnow(),
        )
        
        # Stick action
        step2 = adapter.step(0)
        telemetry_service.record_step(
            episode_id=episode_id,
            step_index=2,
            observation=step2.observation,
            action=0,
            reward=step2.reward,
            terminated=step2.terminated,
            truncated=step2.truncated,
            info=step2.info,
            render_payload=adapter.render(),
            agent_id="human",
            timestamp=datetime.utcnow(),
        )
        
        telemetry_service.finalize_episode(
            episode_id=episode_id,
            total_reward=step2.reward,
            terminated=step2.terminated,
            truncated=step2.truncated,
            timestamp=datetime.utcnow(),
        )
        
        # Load episode via replay loader
        replay = replay_loader.load_episode(episode_id)
        
        assert replay is not None
        assert replay.episode_id == episode_id
        assert len(replay.steps) == 3
        assert replay.rollup.metadata["control_mode"] == ControlMode.HUMAN_ONLY.value

    def test_human_episode_filtering(self, telemetry_service):
        """Test that human episode filtering works for Blackjack."""
        adapter = BlackjackAdapter(game_config=DEFAULT_BLACKJACK_CONFIG)
        adapter.load()
        
        # Create human episode
        human_episode_id = "blackjack-human-001"
        human_metadata = {
            "game_id": GameId.BLACKJACK.value,
            "control_mode": ControlMode.HUMAN_ONLY.value,
            "seed": 42,
        }
        telemetry_service.begin_episode(human_episode_id, metadata=human_metadata)
        initial_step = adapter.reset(seed=42)
        telemetry_service.record_step(
            episode_id=human_episode_id,
            step_index=0,
            observation=initial_step.observation,
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
            render_payload=adapter.render(),
            agent_id="human",
            timestamp=datetime.utcnow(),
        )
        telemetry_service.finalize_episode(
            episode_id=human_episode_id,
            total_reward=0.0,
            terminated=False,
            truncated=False,
            timestamp=datetime.utcnow(),
        )
        
        # Create agent episode
        agent_episode_id = "blackjack-agent-001"
        agent_metadata = {
            "game_id": GameId.BLACKJACK.value,
            "control_mode": ControlMode.AGENT_ONLY.value,
            "seed": 43,
        }
        telemetry_service.begin_episode(agent_episode_id, metadata=agent_metadata)
        adapter.reset(seed=43)
        telemetry_service.record_step(
            episode_id=agent_episode_id,
            step_index=0,
            observation=initial_step.observation,
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
            render_payload=adapter.render(),
            agent_id="agent",
            timestamp=datetime.utcnow(),
        )
        telemetry_service.finalize_episode(
            episode_id=agent_episode_id,
            total_reward=0.0,
            terminated=False,
            truncated=False,
            timestamp=datetime.utcnow(),
        )
        
        # Get all episodes
        all_episodes = list(telemetry_service.recent_episodes())
        assert len(all_episodes) == 2
        
        # Import the filtering function from render_tabs
        from gym_gui.ui.widgets.render_tabs import _ReplayTab
        
        # Filter for human episodes only
        human_episodes = [ep for ep in all_episodes if _ReplayTab._is_human_episode(ep)]
        
        assert len(human_episodes) == 1
        assert human_episodes[0].episode_id == human_episode_id

    def test_render_payload_includes_game_id(self):
        """Test that render payload includes game_id for episode reconstruction."""
        adapter = BlackjackAdapter(game_config=DEFAULT_BLACKJACK_CONFIG)
        adapter.load()
        adapter.reset(seed=42)
        
        payload = adapter.render()
        
        assert "game_id" in payload
        assert payload["game_id"] == GameId.BLACKJACK.value

    def test_episode_rollup_game_id_extraction(self, telemetry_service):
        """Test that episode rollup correctly extracts game_id from metadata."""
        adapter = BlackjackAdapter(game_config=DEFAULT_BLACKJACK_CONFIG)
        adapter.load()
        
        episode_id = "blackjack-test-003"
        episode_metadata = {
            "game_id": GameId.BLACKJACK.value,
            "control_mode": ControlMode.HUMAN_ONLY.value,
        }
        
        telemetry_service.begin_episode(episode_id, metadata=episode_metadata)
        initial_step = adapter.reset(seed=42)
        telemetry_service.record_step(
            episode_id=episode_id,
            step_index=0,
            observation=initial_step.observation,
            action=None,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
            render_payload=adapter.render(),
            agent_id="human",
            timestamp=datetime.utcnow(),
        )
        telemetry_service.finalize_episode(
            episode_id=episode_id,
            total_reward=0.0,
            terminated=False,
            truncated=False,
            timestamp=datetime.utcnow(),
        )
        
        episodes = list(telemetry_service.recent_episodes())
        assert len(episodes) == 1
        episode = episodes[0]
        
        # Verify game_id is in metadata
        assert "game_id" in episode.metadata
        assert episode.metadata["game_id"] == GameId.BLACKJACK.value


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
