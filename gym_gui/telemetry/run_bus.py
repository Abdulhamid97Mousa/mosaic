"""In-process Pub/Sub bus for telemetry events (Ray-inspired dual-path architecture).

This module implements a lightweight, non-blocking event distribution system that
decouples live UI updates from durable database persistence. Multiple subscribers
can consume events independently without blocking producers.

Key design principles:
- Non-blocking publish: producers never wait on slow consumers
- Per-subscriber queues: each consumer has independent buffer
- Drop policy: oldest events dropped if queue fills (prevents deadlock)
- Overflow tracking: metrics for monitoring queue health
"""

import logging
import queue
from collections import defaultdict
from typing import Dict, Optional, Tuple

from .events import TelemetryEvent, Topic

_LOGGER = logging.getLogger(__name__)

SubscriberId = str


class RunBus:
    """In-process Pub/Sub bus for telemetry events.

    Implements Ray's dual-path architecture locally:
    - Live path: fast, ephemeral pub/sub for UI (this bus)
    - Durable path: asynchronous persistence to SQLite (separate task)

    Multiple subscribers can consume events independently without blocking
    producers or each other.

    Per-run sequence tracking:
    - Tracks last_seq_id per (run_id, agent_id) for gap detection
    - Detects missing sequence numbers and logs warnings
    - Supports reconnection with from_seq parameter (future)

    Note: Uses thread-safe queue.Queue for all subscribers to support both
    async (UI) and thread-based (db_sink) consumers.
    """

    def __init__(self, max_queue: int = 2048) -> None:
        """Initialize the RunBus.

        Args:
            max_queue: Default maximum events per subscriber queue before dropping oldest.
                      Can be overridden per subscriber via subscribe_with_size().
        """
        self._queues: Dict[Tuple[Topic, SubscriberId], queue.Queue] = {}
        self._max_queue = max_queue
        self._queue_sizes: Dict[Tuple[Topic, SubscriberId], int] = {}  # Track per-subscriber sizes
        self._overflow: Dict[Tuple[Topic, SubscriberId], int] = defaultdict(int)
        self._seq_counter = 0

        # Per-run sequence tracking for gap detection
        self._last_seq: Dict[Tuple[str, str], int] = {}  # (run_id, agent_id) -> last_seq_id

        _LOGGER.info(f"RunBus initialized with default max_queue={max_queue}")
    
    def subscribe(self, topic: Topic, subscriber_id: SubscriberId) -> queue.Queue:
        """Subscribe to a topic with default queue size.

        Args:
            topic: Event topic to subscribe to
            subscriber_id: Unique identifier for this subscriber

        Returns:
            Thread-safe queue.Queue that will receive events for this topic
        """
        return self.subscribe_with_size(topic, subscriber_id, self._max_queue)

    def subscribe_with_size(
        self, topic: Topic, subscriber_id: SubscriberId, max_queue: int
    ) -> queue.Queue:
        """Subscribe to a topic with a specific queue size.

        Args:
            topic: Event topic to subscribe to
            subscriber_id: Unique identifier for this subscriber
            max_queue: Maximum queue size for this subscriber

        Returns:
            Thread-safe queue.Queue that will receive events for this topic
        """
        q: queue.Queue = queue.Queue(maxsize=max_queue)
        key = (topic, subscriber_id)
        self._queues[key] = q
        self._queue_sizes[key] = max_queue
        _LOGGER.debug(
            f"Subscriber {subscriber_id} subscribed to {topic.name}",
            extra={"max_queue": max_queue},
        )
        return q
    
    def unsubscribe(self, topic: Topic, subscriber_id: SubscriberId) -> None:
        """Unsubscribe from a topic.
        
        Args:
            topic: Event topic to unsubscribe from
            subscriber_id: Subscriber identifier
        """
        self._queues.pop((topic, subscriber_id), None)
        _LOGGER.debug(f"Subscriber {subscriber_id} unsubscribed from {topic.name}")
    
    def publish(self, evt: TelemetryEvent) -> None:
        """Publish an event to all subscribers (non-blocking).

        If a subscriber's queue is full, the oldest event is dropped to prevent
        blocking the producer. Overflow is tracked for monitoring.

        Tracks sequence numbers per (run_id, agent_id) for gap detection.

        Args:
            evt: TelemetryEvent to publish
        """
        # Track sequence numbers for gap detection
        run_agent_key = (evt.run_id, evt.agent_id or "default")
        if run_agent_key in self._last_seq:
            last_seq = self._last_seq[run_agent_key]
            gap = evt.seq_id - last_seq - 1
            if gap > 0:
                _LOGGER.warning(
                    f"Sequence gap detected in RunBus",
                    extra={
                        "run_id": evt.run_id,
                        "agent_id": evt.agent_id,
                        "gap": gap,
                        "last_seq": last_seq,
                        "seq_id": evt.seq_id,
                    },
                )
        self._last_seq[run_agent_key] = evt.seq_id

        # Fan-out to all subscribers for this topic without blocking
        for (topic, subscriber_id), q in list(self._queues.items()):
            if topic is not evt.topic:
                continue

            # If queue is full, drop oldest event
            if q.full():
                try:
                    q.get_nowait()
                    self._overflow[(topic, subscriber_id)] += 1
                except Exception as e:
                    # Catch queue.Empty (from thread-safe queue.Queue)
                    if type(e).__name__ != 'Empty':
                        raise

            # Try to put event in queue
            try:
                q.put_nowait(evt)
            except Exception as e:
                # Catch queue.Full (from thread-safe queue.Queue)
                if type(e).__name__ == 'Full':
                    # Extremely rare after one-drop; count and move on
                    self._overflow[(topic, subscriber_id)] += 1
                else:
                    raise
    
    def overflow_stats(self) -> Dict[str, int]:
        """Get overflow statistics for all subscribers.
        
        Returns:
            Dict mapping "TOPIC:subscriber_id" to overflow count
        """
        return {
            f"{t.name}:{sid}": n
            for (t, sid), n in self._overflow.items()
            if n > 0
        }
    
    def reset_overflow_stats(self) -> None:
        """Reset all overflow counters."""
        self._overflow.clear()
        _LOGGER.debug("Overflow stats reset")
    
    def queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes for all subscribers.

        Returns:
            Dict mapping "TOPIC:subscriber_id" to queue size
        """
        return {
            f"{t.name}:{sid}": q.qsize()
            for (t, sid), q in self._queues.items()
        }

    def sequence_stats(self) -> Dict[str, int]:
        """Get last sequence number for each (run_id, agent_id) pair.

        Returns:
            Dict mapping "run_id:agent_id" to last_seq_id
        """
        return {
            f"{run_id}:{agent_id}": seq_id
            for (run_id, agent_id), seq_id in self._last_seq.items()
        }


# Global singleton instance
_bus_instance: Optional[RunBus] = None


def get_bus() -> RunBus:
    """Get or create the global RunBus singleton.
    
    Returns:
        The global RunBus instance
    """
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = RunBus()
    return _bus_instance


def reset_bus() -> None:
    """Reset the global RunBus (for testing)."""
    global _bus_instance
    _bus_instance = None

