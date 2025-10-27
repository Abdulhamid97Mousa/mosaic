"""Replay controller that subscribes to RUN_COMPLETED events and populates replay tabs.

This module implements the replay population logic:
- Subscribes to RUN_COMPLETED events from RunBus
- Loads episodes from SQLite after training finishes
- Populates replay tabs with historical data
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Dict, Optional

from gym_gui.telemetry.events import Topic, TelemetryEvent
from gym_gui.telemetry.run_bus import RunBus, get_bus
from gym_gui.telemetry.sqlite_store import TelemetrySQLiteStore

_LOGGER = logging.getLogger(__name__)


class ReplayController:
    """Subscribes to RUN_COMPLETED events and populates replay tabs.
    
    This controller loads episodes from SQLite after training finishes
    and populates replay tabs with historical data for review.
    """

    def __init__(
        self,
        store: TelemetrySQLiteStore,
        *,
        on_run_completed: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the replay controller.
        
        Args:
            store: TelemetrySQLiteStore instance for loading episodes
            on_run_completed: Callback when run completes (run_id, payload)
        """
        self._store = store
        self._on_run_completed = on_run_completed
        
        self._bus: Optional[RunBus] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        _LOGGER.info("ReplayController initialized")

    def start(self) -> None:
        """Start the replay controller subscription thread."""
        if self._thread is not None:
            _LOGGER.warning("Replay controller already started")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="replay-controller",
            daemon=True,
        )
        self._thread.start()
        _LOGGER.info("ReplayController started")

    def stop(self) -> None:
        """Stop the replay controller subscription thread."""
        if self._thread is None:
            return
        
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            _LOGGER.warning("Replay controller thread did not stop cleanly")
        else:
            _LOGGER.info("ReplayController stopped")
        self._thread = None

    def _run(self) -> None:
        """Main loop for the replay controller subscription thread."""
        try:
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            # Run the async subscription loop
            self._loop.run_until_complete(self._subscribe_and_listen())
        except Exception as e:
            _LOGGER.exception("Fatal error in replay controller", extra={"error": str(e)})
        finally:
            if self._loop is not None:
                self._loop.close()
            _LOGGER.info("Replay controller loop exited")

    async def _subscribe_and_listen(self) -> None:
        """Subscribe to RUN_COMPLETED topic and listen for events."""
        try:
            self._bus = get_bus()
            
            # Subscribe to RUN_COMPLETED topic
            queue = self._bus.subscribe(
                Topic.RUN_COMPLETED,
                "replay-controller",
            )
            
            _LOGGER.debug("Subscribed to RUN_COMPLETED topic")
            
            # Listen for events until stopped
            while not self._stop_event.is_set():
                try:
                    # Check for run completed events (non-blocking)
                    try:
                        evt = queue.get_nowait()
                        if isinstance(evt, TelemetryEvent):
                            self._handle_run_completed_event(evt)
                    except asyncio.QueueEmpty:
                        pass
                    
                    # Small sleep to avoid busy-waiting
                    await asyncio.sleep(0.1)
                except Exception as e:
                    _LOGGER.exception("Error in listen loop", extra={"error": str(e)})
        except Exception as e:
            _LOGGER.exception("Fatal error in subscribe_and_listen", extra={"error": str(e)})

    def _handle_run_completed_event(self, evt: TelemetryEvent) -> None:
        """Handle a RUN_COMPLETED event."""
        run_id = evt.run_id
        payload = evt.payload
        outcome = payload.get("outcome", "unknown")
        
        try:
            _LOGGER.info(
                "Run completed event received",
                extra={"run_id": run_id, "outcome": outcome},
            )
            
            # Load episodes from SQLite for this run
            episodes = self._store.episodes_for_run(run_id)
            _LOGGER.debug(
                "Loaded episodes from SQLite",
                extra={"run_id": run_id, "episode_count": len(episodes)},
            )
            
            # Invoke callback with run_id and payload
            if self._on_run_completed is not None:
                self._on_run_completed(run_id, payload)
            
            _LOGGER.debug(
                "Processed run completed event",
                extra={"run_id": run_id, "outcome": outcome},
            )
        except Exception as e:
            _LOGGER.exception(
                "Error handling run completed event",
                extra={"run_id": run_id, "error": str(e)},
            )


__all__ = ["ReplayController"]

