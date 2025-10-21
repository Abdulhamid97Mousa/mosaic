"""Phase 3 Tests: Bus + Writer Split Implementation

Tests for:
1. RunBus pub/sub with multiple subscribers
2. Writer thread (db_sink.py) batching and persistence
3. UI subscribers (live_telemetry.py) immediate rendering
4. Bounded queues and overflow handling
"""

import asyncio
import pytest
from datetime import datetime, timezone

from gym_gui.telemetry.events import Topic, TelemetryEvent
from gym_gui.telemetry.run_bus import RunBus, reset_bus, get_bus
from gym_gui.telemetry.db_sink import TelemetryDBSink
from gym_gui.telemetry.sqlite_store import TelemetrySQLiteStore


@pytest.fixture
def bus():
    """Create a fresh RunBus for each test."""
    reset_bus()
    return RunBus(max_queue=64)


@pytest.fixture
def test_event():
    """Create a test TelemetryEvent."""
    return TelemetryEvent(
        topic=Topic.STEP_APPENDED,
        run_id="test-run-1",
        agent_id="agent-1",
        seq_id=1,
        ts_iso=datetime.now(timezone.utc).isoformat(),
        payload={
            "episode_id": "ep-1",
            "step_index": 0,
            "action": 0,
            "observation": [0, 0],
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {},
        },
    )


class TestRunBusPubSub:
    """Test RunBus pub/sub functionality."""

    def test_subscribe_returns_queue(self, bus):
        """Test that subscribe returns an asyncio.Queue."""
        q = bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        assert isinstance(q, asyncio.Queue)

    def test_publish_to_single_subscriber(self, bus, test_event):
        """Test publishing to a single subscriber."""
        q = bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        bus.publish(test_event)
        
        # Event should be in queue
        assert q.qsize() == 1
        evt = q.get_nowait()
        assert evt.seq_id == 1
        assert evt.run_id == "test-run-1"

    def test_publish_to_multiple_subscribers(self, bus, test_event):
        """Test publishing to multiple subscribers independently."""
        q1 = bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        q2 = bus.subscribe(Topic.STEP_APPENDED, "subscriber-2")
        
        bus.publish(test_event)
        
        # Both subscribers should receive the event
        assert q1.qsize() == 1
        assert q2.qsize() == 1
        
        evt1 = q1.get_nowait()
        evt2 = q2.get_nowait()
        assert evt1.seq_id == evt2.seq_id == 1

    def test_topic_filtering(self, bus):
        """Test that subscribers only receive events for their topic."""
        q_step = bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        q_episode = bus.subscribe(Topic.EPISODE_FINALIZED, "subscriber-1")
        
        step_event = TelemetryEvent(
            topic=Topic.STEP_APPENDED,
            run_id="test-run-1",
            agent_id="agent-1",
            seq_id=1,
            ts_iso=datetime.now(timezone.utc).isoformat(),
            payload={"step_index": 0},
        )
        
        bus.publish(step_event)
        
        # Only step queue should have the event
        assert q_step.qsize() == 1
        assert q_episode.qsize() == 0

    def test_unsubscribe(self, bus, test_event):
        """Test unsubscribing from a topic."""
        q = bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        bus.unsubscribe(Topic.STEP_APPENDED, "subscriber-1")
        
        bus.publish(test_event)
        
        # Queue should be empty after unsubscribe
        assert q.qsize() == 0

    def test_overflow_drops_oldest_event(self, bus):
        """Test that overflow drops oldest event when queue is full."""
        small_bus = RunBus(max_queue=2)
        q = small_bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        
        # Publish 3 events to a queue with max_queue=2
        for i in range(3):
            evt = TelemetryEvent(
                topic=Topic.STEP_APPENDED,
                run_id="test-run-1",
                agent_id="agent-1",
                seq_id=i,
                ts_iso=datetime.now(timezone.utc).isoformat(),
                payload={"step_index": i},
            )
            small_bus.publish(evt)
        
        # Queue should have 2 events (oldest dropped)
        assert q.qsize() == 2
        
        # First event should be seq_id=1 (seq_id=0 was dropped)
        evt1 = q.get_nowait()
        assert evt1.seq_id == 1
        
        evt2 = q.get_nowait()
        assert evt2.seq_id == 2

    def test_overflow_stats(self, bus):
        """Test overflow statistics tracking."""
        small_bus = RunBus(max_queue=1)
        q = small_bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        
        # Publish 3 events to a queue with max_queue=1
        for i in range(3):
            evt = TelemetryEvent(
                topic=Topic.STEP_APPENDED,
                run_id="test-run-1",
                agent_id="agent-1",
                seq_id=i,
                ts_iso=datetime.now(timezone.utc).isoformat(),
                payload={"step_index": i},
            )
            small_bus.publish(evt)
        
        stats = small_bus.overflow_stats()
        assert "STEP_APPENDED:subscriber-1" in stats
        assert stats["STEP_APPENDED:subscriber-1"] == 2  # 2 events dropped

    def test_queue_sizes(self, bus, test_event):
        """Test queue size reporting."""
        q1 = bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        q2 = bus.subscribe(Topic.EPISODE_FINALIZED, "subscriber-2")
        
        bus.publish(test_event)
        
        sizes = bus.queue_sizes()
        assert sizes["STEP_APPENDED:subscriber-1"] == 1
        assert sizes["EPISODE_FINALIZED:subscriber-2"] == 0

    def test_non_blocking_publish(self, bus):
        """Test that publish is non-blocking even with full queue."""
        small_bus = RunBus(max_queue=1)
        q = small_bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        
        # Fill the queue
        evt1 = TelemetryEvent(
            topic=Topic.STEP_APPENDED,
            run_id="test-run-1",
            agent_id="agent-1",
            seq_id=1,
            ts_iso=datetime.now(timezone.utc).isoformat(),
            payload={"step_index": 0},
        )
        small_bus.publish(evt1)
        
        # Publish should not block even though queue is full
        evt2 = TelemetryEvent(
            topic=Topic.STEP_APPENDED,
            run_id="test-run-1",
            agent_id="agent-1",
            seq_id=2,
            ts_iso=datetime.now(timezone.utc).isoformat(),
            payload={"step_index": 1},
        )
        small_bus.publish(evt2)  # Should not raise or block
        
        # Queue should have the newer event (oldest dropped)
        assert q.qsize() == 1
        evt = q.get_nowait()
        assert evt.seq_id == 2


class TestSequenceNumberTracking:
    """Test sequence number tracking through the bus."""

    def test_seq_id_preserved_through_publish(self, bus):
        """Test that seq_id is preserved when publishing."""
        q = bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        
        for seq_id in range(1, 6):
            evt = TelemetryEvent(
                topic=Topic.STEP_APPENDED,
                run_id="test-run-1",
                agent_id="agent-1",
                seq_id=seq_id,
                ts_iso=datetime.now(timezone.utc).isoformat(),
                payload={"step_index": seq_id - 1},
            )
            bus.publish(evt)
        
        # Verify all seq_ids are preserved
        for expected_seq_id in range(1, 6):
            evt = q.get_nowait()
            assert evt.seq_id == expected_seq_id

    def test_seq_id_logged_in_event(self, bus, test_event):
        """Test that seq_id is accessible in published events."""
        q = bus.subscribe(Topic.STEP_APPENDED, "subscriber-1")
        bus.publish(test_event)
        
        evt = q.get_nowait()
        assert hasattr(evt, "seq_id")
        assert evt.seq_id == test_event.seq_id


class TestQueueSizeConfiguration:
    """Test per-subscriber queue size configuration."""

    def test_subscribe_with_size_creates_queue_with_specified_size(self, bus):
        """Test that subscribe_with_size creates a queue with the specified size."""
        q = bus.subscribe_with_size(Topic.STEP_APPENDED, "subscriber-1", 32)
        assert q.maxsize == 32

    def test_different_subscribers_can_have_different_queue_sizes(self, bus):
        """Test that different subscribers can have different queue sizes."""
        q1 = bus.subscribe_with_size(Topic.STEP_APPENDED, "ui-subscriber", 64)
        q2 = bus.subscribe_with_size(Topic.STEP_APPENDED, "writer-subscriber", 512)

        assert q1.maxsize == 64
        assert q2.maxsize == 512

    def test_writer_queue_larger_than_ui_queue(self, bus):
        """Test that writer queue is larger than UI queue."""
        ui_q = bus.subscribe_with_size(Topic.STEP_APPENDED, "ui", 64)
        writer_q = bus.subscribe_with_size(Topic.STEP_APPENDED, "writer", 512)

        assert writer_q.maxsize > ui_q.maxsize
        assert writer_q.maxsize == 512
        assert ui_q.maxsize == 64


class TestGlobalSingleton:
    """Test global RunBus singleton."""

    def test_get_bus_returns_singleton(self):
        """Test that get_bus returns the same instance."""
        reset_bus()
        bus1 = get_bus()
        bus2 = get_bus()
        assert bus1 is bus2

    def test_reset_bus_creates_new_instance(self):
        """Test that reset_bus creates a new instance."""
        bus1 = get_bus()
        reset_bus()
        bus2 = get_bus()
        assert bus1 is not bus2

