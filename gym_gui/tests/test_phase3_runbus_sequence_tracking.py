"""Tests for Phase 3: RunBus sequence number tracking and gap detection."""

import pytest

from gym_gui.telemetry.events import Topic, TelemetryEvent
from gym_gui.telemetry.run_bus import RunBus, reset_bus


@pytest.fixture
def bus():
    """Create a fresh RunBus for each test."""
    reset_bus()
    return RunBus(max_queue=64)


class TestSequenceTracking:
    """Test sequence number tracking in RunBus."""

    def test_sequence_tracking_initialized(self, bus):
        """Test that sequence tracking is initialized."""
        stats = bus.sequence_stats()
        assert isinstance(stats, dict)
        assert len(stats) == 0

    def test_sequence_tracking_per_run_agent(self, bus):
        """Test that sequence numbers are tracked per (run_id, agent_id)."""
        bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        
        # Publish events for run1/agent1
        evt1 = TelemetryEvent(
            topic=Topic.STEP_APPENDED,
            run_id="run1",
            agent_id="agent1",
            seq_id=0,
            ts_iso="2025-01-01T00:00:00",
            payload={"step_index": 0},
        )
        bus.publish(evt1)
        
        # Publish events for run1/agent2
        evt2 = TelemetryEvent(
            topic=Topic.STEP_APPENDED,
            run_id="run1",
            agent_id="agent2",
            seq_id=0,
            ts_iso="2025-01-01T00:00:01",
            payload={"step_index": 0},
        )
        bus.publish(evt2)
        
        stats = bus.sequence_stats()
        assert "run1:agent1" in stats
        assert "run1:agent2" in stats
        assert stats["run1:agent1"] == 0
        assert stats["run1:agent2"] == 0

    def test_sequence_tracking_increments(self, bus):
        """Test that sequence numbers increment correctly."""
        bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        
        for seq_id in range(5):
            evt = TelemetryEvent(
                topic=Topic.STEP_APPENDED,
                run_id="run1",
                agent_id="agent1",
                seq_id=seq_id,
                ts_iso="2025-01-01T00:00:00",
                payload={"step_index": seq_id},
            )
            bus.publish(evt)
        
        stats = bus.sequence_stats()
        assert stats["run1:agent1"] == 4

    def test_gap_detection_logs_warning(self, bus, caplog):
        """Test that gaps in sequence numbers are detected and logged."""
        bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        
        # Publish seq 0
        evt0 = TelemetryEvent(
            topic=Topic.STEP_APPENDED,
            run_id="run1",
            agent_id="agent1",
            seq_id=0,
            ts_iso="2025-01-01T00:00:00",
            payload={"step_index": 0},
        )
        bus.publish(evt0)
        
        # Publish seq 3 (gap of 2)
        evt3 = TelemetryEvent(
            topic=Topic.STEP_APPENDED,
            run_id="run1",
            agent_id="agent1",
            seq_id=3,
            ts_iso="2025-01-01T00:00:03",
            payload={"step_index": 3},
        )
        bus.publish(evt3)
        
        # Check that gap was logged
        assert "Sequence gap detected" in caplog.text
        assert "gap" in caplog.text.lower()

    def test_no_gap_for_first_event(self, bus, caplog):
        """Test that first event doesn't trigger gap warning."""
        bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        
        evt = TelemetryEvent(
            topic=Topic.STEP_APPENDED,
            run_id="run1",
            agent_id="agent1",
            seq_id=0,
            ts_iso="2025-01-01T00:00:00",
            payload={"step_index": 0},
        )
        bus.publish(evt)
        
        # Should not have gap warning
        assert "Sequence gap detected" not in caplog.text

    def test_gap_detection_multiple_runs(self, bus, caplog):
        """Test gap detection with multiple runs."""
        bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")

        # Run 1: seq 0, 1, 2
        for seq_id in range(3):
            evt = TelemetryEvent(
                topic=Topic.STEP_APPENDED,
                run_id="run1",
                agent_id="agent1",
                seq_id=seq_id,
                ts_iso="2025-01-01T00:00:00",
                payload={"step_index": seq_id},
            )
            bus.publish(evt)

        # Run 2: seq 0, 1, 5 (gap in run2)
        for seq_id in [0, 1, 5]:
            evt = TelemetryEvent(
                topic=Topic.STEP_APPENDED,
                run_id="run2",
                agent_id="agent1",
                seq_id=seq_id,
                ts_iso="2025-01-01T00:00:00",
                payload={"step_index": seq_id},
            )
            bus.publish(evt)

        # Check that gap was detected (should have 2 gap warnings: one for run2)
        assert "Sequence gap detected" in caplog.text
        # Verify sequence stats show both runs
        stats = bus.sequence_stats()
        assert "run1:agent1" in stats
        assert "run2:agent1" in stats

    def test_sequence_stats_multiple_agents(self, bus):
        """Test sequence stats with multiple agents."""
        bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        
        # Agent 1: seq 0-4
        for seq_id in range(5):
            evt = TelemetryEvent(
                topic=Topic.STEP_APPENDED,
                run_id="run1",
                agent_id="agent1",
                seq_id=seq_id,
                ts_iso="2025-01-01T00:00:00",
                payload={"step_index": seq_id},
            )
            bus.publish(evt)
        
        # Agent 2: seq 0-9
        for seq_id in range(10):
            evt = TelemetryEvent(
                topic=Topic.STEP_APPENDED,
                run_id="run1",
                agent_id="agent2",
                seq_id=seq_id,
                ts_iso="2025-01-01T00:00:00",
                payload={"step_index": seq_id},
            )
            bus.publish(evt)
        
        stats = bus.sequence_stats()
        assert stats["run1:agent1"] == 4
        assert stats["run1:agent2"] == 9

    def test_sequence_tracking_with_default_agent(self, bus):
        """Test sequence tracking when agent_id is None."""
        bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        
        evt = TelemetryEvent(
            topic=Topic.STEP_APPENDED,
            run_id="run1",
            agent_id=None,
            seq_id=0,
            ts_iso="2025-01-01T00:00:00",
            payload={"step_index": 0},
        )
        bus.publish(evt)
        
        stats = bus.sequence_stats()
        assert "run1:default" in stats
        assert stats["run1:default"] == 0

    def test_large_gap_detection(self, bus, caplog):
        """Test detection of large gaps."""
        bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")

        # Publish seq 0
        evt0 = TelemetryEvent(
            topic=Topic.STEP_APPENDED,
            run_id="run1",
            agent_id="agent1",
            seq_id=0,
            ts_iso="2025-01-01T00:00:00",
            payload={"step_index": 0},
        )
        bus.publish(evt0)

        # Publish seq 100 (gap of 99)
        evt100 = TelemetryEvent(
            topic=Topic.STEP_APPENDED,
            run_id="run1",
            agent_id="agent1",
            seq_id=100,
            ts_iso="2025-01-01T00:01:40",
            payload={"step_index": 100},
        )
        bus.publish(evt100)

        # Check that large gap was logged
        assert "Sequence gap detected" in caplog.text
        # Verify the gap was tracked correctly
        stats = bus.sequence_stats()
        assert stats["run1:agent1"] == 100

