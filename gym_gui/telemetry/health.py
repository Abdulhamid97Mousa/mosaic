"""Health monitoring and observability for telemetry pipeline.

This module provides:
- Heartbeat events to detect stalled runs
- Structured logging for state changes
- Overflow statistics tracking
- Queue health monitoring
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from gym_gui.telemetry.events import Topic, TelemetryEvent
from gym_gui.telemetry.run_bus import RunBus
from gym_gui.telemetry.constants import HEALTH_MONITOR_HEARTBEAT_INTERVAL_S

_LOGGER = logging.getLogger(__name__)


@dataclass
class HealthStats:
    """Statistics about telemetry pipeline health."""

    total_events: int = 0
    dropped_events: int = 0
    queue_overflow_count: int = 0
    last_heartbeat: Optional[str] = None
    stalled_runs: list[str] = field(default_factory=list)

    @property
    def drop_rate(self) -> float:
        """Calculate event drop rate as percentage."""
        if self.total_events == 0:
            return 0.0
        return (self.dropped_events / self.total_events) * 100


class HealthMonitor:
    """Monitors telemetry pipeline health and emits heartbeat events."""

    def __init__(
        self,
        bus: RunBus,
        *,
        heartbeat_interval: float = HEALTH_MONITOR_HEARTBEAT_INTERVAL_S,
    ) -> None:
        """Initialize health monitor.
        
        Args:
            bus: RunBus instance to emit heartbeat events
            heartbeat_interval: Seconds between heartbeat events
        """
        self._bus = bus
        self._heartbeat_interval = heartbeat_interval
        self._stats = HealthStats()
        self._active_runs: dict[str, datetime] = {}
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start the health monitor."""
        if self._task is not None:
            _LOGGER.warning("Health monitor already started")
            return

        # Create background task for heartbeat loop
        # The task is stored in self._task and will run until stop() is called
        try:
            loop = asyncio.get_running_loop()
            _LOGGER.debug("Health monitor using running event loop")
        except RuntimeError:
            # Event loop not running yet, try to get the current event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    _LOGGER.error("Event loop is closed, cannot start health monitor")
                    return
                _LOGGER.debug("Health monitor using current event loop")
            except RuntimeError as e:
                _LOGGER.error(
                    "No event loop available, cannot start health monitor",
                    extra={"error": str(e)},
                )
                return

        try:
            self._task = loop.create_task(self._heartbeat_loop())
            _LOGGER.info("Health monitor started")
        except RuntimeError as e:
            # Event loop might not be running yet, defer task creation
            if "no running event loop" in str(e):
                _LOGGER.debug(
                    "Event loop not running yet, deferring health monitor start",
                    extra={"error": str(e)},
                )
                # Schedule deferred start using call_soon_threadsafe if available
                try:
                    loop.call_soon_threadsafe(self._deferred_start, loop)
                except Exception as defer_error:
                    _LOGGER.warning(
                        "Failed to defer health monitor start",
                        extra={"error": str(defer_error)},
                    )
            else:
                _LOGGER.exception(
                    "Failed to create heartbeat task",
                    extra={"error": str(e)},
                )
                self._task = None
        except Exception as e:
            _LOGGER.exception(
                "Failed to create heartbeat task",
                extra={"error": str(e)},
            )
            self._task = None

    def _deferred_start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Deferred start of health monitor when event loop is ready."""
        try:
            if self._task is None:
                self._task = loop.create_task(self._heartbeat_loop())
                _LOGGER.info("Health monitor started (deferred)")
        except Exception as e:
            _LOGGER.exception(
                "Failed to create heartbeat task in deferred start",
                extra={"error": str(e)},
            )

    async def stop(self) -> None:
        """Stop the health monitor."""
        if self._task is None:
            return

        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        _LOGGER.info("Health monitor stopped")

    def record_run_started(self, run_id: str) -> None:
        """Record that a run has started."""
        self._active_runs[run_id] = datetime.now(timezone.utc)
        _LOGGER.info("Run started", extra={"run_id": run_id})

    def record_run_finished(self, run_id: str, outcome: str) -> None:
        """Record that a run has finished."""
        if run_id in self._active_runs:
            del self._active_runs[run_id]
        _LOGGER.info("Run finished", extra={"run_id": run_id, "outcome": outcome})

    def record_event(self, event: TelemetryEvent) -> None:
        """Record that an event was processed."""
        self._stats.total_events += 1
        if event.run_id in self._active_runs:
            self._active_runs[event.run_id] = datetime.now(timezone.utc)

    def record_dropped_event(self) -> None:
        """Record that an event was dropped."""
        self._stats.dropped_events += 1

    def record_queue_overflow(self) -> None:
        """Record that a queue overflowed."""
        self._stats.queue_overflow_count += 1

    def get_stats(self) -> HealthStats:
        """Get current health statistics."""
        return HealthStats(
            total_events=self._stats.total_events,
            dropped_events=self._stats.dropped_events,
            queue_overflow_count=self._stats.queue_overflow_count,
            last_heartbeat=self._stats.last_heartbeat,
            stalled_runs=self._detect_stalled_runs(),
        )

    def _detect_stalled_runs(self) -> list[str]:
        """Detect runs that haven't received events recently."""
        now = datetime.now(timezone.utc)
        stalled = []
        for run_id, last_event in self._active_runs.items():
            elapsed = (now - last_event).total_seconds()
            if elapsed > self._heartbeat_interval * 3:  # 3x heartbeat interval
                stalled.append(run_id)
        return stalled

    async def _heartbeat_loop(self) -> None:
        """Emit periodic heartbeat events."""
        try:
            while True:
                try:
                    await asyncio.sleep(self._heartbeat_interval)
                except RuntimeError as e:
                    # Event loop not running yet - this is a fatal error
                    # We cannot await in a non-running loop, so we must exit
                    if "no running event loop" in str(e):
                        _LOGGER.error(
                            "Event loop not running in heartbeat task - exiting heartbeat loop",
                            extra={"error": str(e)},
                        )
                        return
                    raise

                # Emit heartbeat for each active run
                now = datetime.now(timezone.utc)
                for run_id in list(self._active_runs.keys()):
                    try:
                        event = TelemetryEvent(
                            topic=Topic.RUN_HEARTBEAT,
                            run_id=run_id,
                            agent_id="system",
                            seq_id=0,
                            ts_iso=now.isoformat(),
                            payload={
                                "timestamp": now.isoformat(),
                                "active_runs": len(self._active_runs),
                                "total_events": self._stats.total_events,
                                "dropped_events": self._stats.dropped_events,
                                "queue_overflow_count": self._stats.queue_overflow_count,
                            },
                        )
                        self._bus.publish(event)
                    except Exception as e:
                        _LOGGER.warning(
                            "Failed to emit heartbeat",
                            extra={"run_id": run_id, "error": str(e)},
                        )

                # Log health summary
                self._stats.last_heartbeat = now.isoformat()
                stalled = self._detect_stalled_runs()
                if stalled:
                    _LOGGER.warning(
                        "Stalled runs detected",
                        extra={
                            "stalled_runs": stalled,
                            "drop_rate": f"{self.get_stats().drop_rate:.2f}%",
                        },
                    )
        except asyncio.CancelledError:
            _LOGGER.debug("Heartbeat loop cancelled")
        except Exception as e:
            _LOGGER.exception("Heartbeat loop failed", extra={"error": str(e)})


__all__ = ["HealthStats", "HealthMonitor"]
