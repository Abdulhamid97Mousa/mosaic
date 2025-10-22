"""Controller for managing live telemetry streaming from trainer runs.

This controller subscribes to RunBus for independent event delivery:
- Subscribes to STEP_APPENDED and EPISODE_FINALIZED topics
- Uses UI queue size (64 events) for responsive rendering
- Processes events in background thread
- Emits Qt signals for main thread rendering
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import TYPE_CHECKING, Optional, Dict

from qtpy import QtCore

from gym_gui.telemetry.run_bus import get_bus
from gym_gui.telemetry.events import Topic, TelemetryEvent
from gym_gui.telemetry.credit_manager import get_credit_manager

if TYPE_CHECKING:
    from gym_gui.services.trainer import TrainerClient
    from gym_gui.services.trainer.streams import TelemetryAsyncHub
    from gym_gui.ui.widgets.live_telemetry_tab import LiveTelemetryTab


class LiveTelemetryController(QtCore.QObject):
    """Coordinates telemetry hub lifecycle and manages dynamic per-agent tab creation."""

    # Signals
    run_tab_requested = QtCore.Signal(str, str, str)  # type: ignore[attr-defined] # (run_id, agent_id, tab_title)
    telemetry_stats_updated = QtCore.Signal(str, dict)  # type: ignore[attr-defined] # (run_id, stats)
    run_completed = QtCore.Signal(str)  # type: ignore[attr-defined] # (run_id)

    def __init__(
        self,
        hub: "TelemetryAsyncHub",
        client: "TrainerClient",
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._hub = hub
        self._client = client
        self._logger = logging.getLogger("gym_gui.controllers.live_telemetry")
        self._active_runs: set[str] = set()
        self._tabs: Dict[tuple[str, str], "LiveTelemetryTab"] = {}  # (run_id, agent_id) -> tab

        # Buffer for steps that arrive before tab is registered (race condition mitigation)
        self._step_buffer: Dict[tuple[str, str], list] = {}  # (run_id, agent_id) -> [steps]
        self._episode_buffer: Dict[tuple[str, str], list] = {}  # (run_id, agent_id) -> [episodes]

        # Store rendering throttle value per run (from TELEMETRY_SAMPLING_INTERVAL env var)
        self._render_throttle_per_run: Dict[str, int] = {}  # run_id -> throttle_interval

        # Store buffer sizes per run (from training config)
        self._step_buffer_size_per_run: Dict[str, int] = {}  # run_id -> step_buffer_size
        self._episode_buffer_size_per_run: Dict[str, int] = {}  # run_id -> episode_buffer_size

        # Store game_id per run (for passing to tabs)
        self._game_id_per_run: Dict[str, str] = {}  # run_id -> game_id

        # RunBus subscription for independent event delivery
        self._bus = get_bus()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._step_queue: Optional[queue.Queue] = None
        self._episode_queue: Optional[queue.Queue] = None

        # Still wire bridge signals for overflow and run_completed (not in RunBus yet)
        self._hub.bridge.queue_overflow.connect(self._on_queue_overflow)
        self._hub.bridge.run_completed.connect(self._on_run_completed_from_bridge)

        self._logger.info("LiveTelemetryController initialized with RunBus subscription")

    def start(self) -> None:
        """Start the background thread for RunBus subscription."""
        if self._thread is not None:
            self._logger.warning("Controller already started")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="live-telemetry-runbus",
            daemon=True,
        )
        self._thread.start()
        self._logger.info("LiveTelemetryController background thread started")

    def stop(self) -> None:
        """Stop the background thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self._logger.warning("Controller thread did not stop cleanly")
        else:
            self._logger.info("LiveTelemetryController background thread stopped")
        self._thread = None

    def subscribe_to_run(self, run_id: str) -> None:
        """Start streaming telemetry for a run.

        Pre-initialize credits to prevent chicken-and-egg deadlock.
        Credits are initialized BEFORE first event arrives, ensuring the
        credit system is ready when the first step/episode is received.
        """
        if run_id in self._active_runs:
            self._logger.debug("Already subscribed to run", extra={"run_id": run_id})
            return

        self._logger.debug(
            "Subscribing controller to run",
            extra={"run_id": run_id},
        )
        self._active_runs.add(run_id)

        # Pre-initialize credits for default agent
        # This ensures credits exist BEFORE first event arrives
        # We use "default" as a placeholder; actual agent_id will be extracted from first event
        credit_mgr = get_credit_manager()
        credit_mgr.initialize_stream(run_id, "default")
        self._logger.debug(
            "Pre-initialized credits for default agent",
            extra={"run_id": run_id, "agent_id": "default"},
        )

        self._hub.subscribe_run(run_id, self._client)
        self._logger.debug("Successfully subscribed to run: %s (hub notified)", run_id)

    def unsubscribe_from_run(self, run_id: str) -> None:
        """Stop streaming telemetry for a run."""
        if run_id not in self._active_runs:
            return
        self._active_runs.discard(run_id)
        # Clean up all tabs for this run
        keys_to_remove = [k for k in self._tabs if k[0] == run_id]
        for key in keys_to_remove:
            del self._tabs[key]
        self._hub.unsubscribe_run(run_id)
        self._logger.debug("Unsubscribed controller from run", extra={"run_id": run_id})

    def shutdown(self) -> None:
        """Clean up all subscriptions and stop the hub."""
        for run_id in list(self._active_runs):
            self.unsubscribe_from_run(run_id)
        self._tabs.clear()
        self._hub.stop()

    def set_render_throttle_for_run(self, run_id: str, throttle_interval: int) -> None:
        """Set the rendering throttle interval for a run.

        Args:
            run_id: The training run ID
            throttle_interval: Render every Nth step (1=every step, 2=every 2nd step, etc.)
        """
        self._render_throttle_per_run[run_id] = max(1, throttle_interval)
        self._logger.debug(
            "Set render throttle for run",
            extra={"run_id": run_id, "throttle_interval": throttle_interval},
        )

    def set_buffer_sizes_for_run(self, run_id: str, step_buffer_size: int = 100, episode_buffer_size: int = 100) -> None:
        """Set the buffer sizes for a run.

        Args:
            run_id: The training run ID
            step_buffer_size: Number of steps to keep in UI display buffer (default: 100)
            episode_buffer_size: Number of episodes to keep in UI display buffer (default: 100)
        """
        self._step_buffer_size_per_run[run_id] = max(10, step_buffer_size)
        self._episode_buffer_size_per_run[run_id] = max(10, episode_buffer_size)
        self._logger.debug(
            "Set buffer sizes for run",
            extra={
                "run_id": run_id,
                "step_buffer_size": step_buffer_size,
                "episode_buffer_size": episode_buffer_size,
            },
        )

    def get_buffer_sizes_for_run(self, run_id: str) -> tuple[int, int]:
        """Get the buffer sizes for a run.

        Args:
            run_id: The training run ID

        Returns:
            Tuple of (step_buffer_size, episode_buffer_size)
        """
        step_size = self._step_buffer_size_per_run.get(run_id, 100)
        episode_size = self._episode_buffer_size_per_run.get(run_id, 100)
        return step_size, episode_size

    def set_game_id_for_run(self, run_id: str, game_id: str) -> None:
        """Store the game_id for a run (for passing to dynamic tabs).

        Args:
            run_id: The training run ID
            game_id: The game environment ID (e.g., "FrozenLake-v1")
        """
        self._game_id_per_run[run_id] = game_id
        self._logger.debug(
            "Set game_id for run",
            extra={"run_id": run_id, "game_id": game_id},
        )

    def get_game_id_for_run(self, run_id: str) -> str | None:
        """Get the game_id for a run.

        Args:
            run_id: The training run ID

        Returns:
            The game_id string, or None if not set
        """
        return self._game_id_per_run.get(run_id)

    def get_render_throttle_for_run(self, run_id: str) -> int:
        """Get the rendering throttle interval for a run.

        Args:
            run_id: The training run ID

        Returns:
            The throttle interval (render every Nth step), defaults to 1 if not set
        """
        return self._render_throttle_per_run.get(run_id, 1)

    def register_tab(self, run_id: str, agent_id: str, tab: "LiveTelemetryTab") -> None:
        """Register a newly created tab widget for routing telemetry.

        When tab is registered, we grant initial credits to ensure the
        credit system is ready for incoming events.
        """
        key = (run_id, agent_id)
        self._tabs[key] = tab

        # Grant initial credits when tab is registered
        # This ensures credits are available for events routed to this tab
        credit_mgr = get_credit_manager()
        credit_mgr.grant_credits(run_id, agent_id, 200)
        self._logger.debug(
            "Granted initial credits to tab",
            extra={"run_id": run_id, "agent_id": agent_id, "amount": 200},
        )

        # Apply rendering throttle if set for this run
        if run_id in self._render_throttle_per_run:
            throttle = self._render_throttle_per_run[run_id]
            tab.set_render_throttle_interval(throttle)
            self._logger.debug(
                "Applied render throttle to tab",
                extra={"run_id": run_id, "agent_id": agent_id, "throttle": throttle},
            )

        self._logger.debug(
            "Tab registered and ready for telemetry",
            extra={"run_id": run_id, "agent_id": agent_id, "tab_type": type(tab).__name__},
        )

        # Flush any buffered steps that arrived before tab was registered
        if key in self._step_buffer:
            buffered_steps = self._step_buffer.pop(key)
            self._logger.info(
                "Flushing buffered steps to newly registered tab",
                extra={"run_id": run_id, "agent_id": agent_id, "buffered_count": len(buffered_steps)},
            )
            for payload in buffered_steps:
                tab.add_step(payload)

        # Flush any buffered episodes that arrived before tab was registered
        if key in self._episode_buffer:
            buffered_episodes = self._episode_buffer.pop(key)
            self._logger.info(
                "Flushing buffered episodes to newly registered tab",
                extra={"run_id": run_id, "agent_id": agent_id, "buffered_count": len(buffered_episodes)},
            )
            for payload in buffered_episodes:
                tab.add_episode(payload)

    @QtCore.Slot(str, str, str)  # type: ignore[misc]
    def _emit_tab_requested(self, run_id: str, agent_id: str, tab_title: str) -> None:
        """Helper method to emit run_tab_requested signal from main thread.

        This is called via QMetaObject.invokeMethod from the background thread,
        ensuring the signal is emitted on the main thread.
        """
        self.run_tab_requested.emit(run_id, agent_id, tab_title)

    # Note: Old signal handlers (_on_step_received, _on_episode_received) removed
    # Events now come from RunBus subscription in background thread (_process_step_queue, _process_episode_queue)

    def _on_queue_overflow(self, run_id: str, stream_type: str, dropped: int) -> None:
        stats = {
            "stream_type": stream_type,
            "dropped_total": dropped,
        }
        self.telemetry_stats_updated.emit(run_id, stats)

        # Mark overflow on all tabs for this run
        for (r_id, a_id), tab in self._tabs.items():
            if r_id == run_id and tab is not None:
                if hasattr(tab, 'mark_overflow'):
                    tab.mark_overflow(stream_type, dropped)

        self._logger.warning(
            "Telemetry queue overflow",
            extra={"run_id": run_id, "stream_type": stream_type, "dropped": dropped},
        )

    def _on_run_completed_from_bridge(self, run_id: str) -> None:
        """Handle run completion signal from bridge."""
        self._logger.info("Run completed signal received from bridge", extra={"run_id": run_id})
        self.run_completed.emit(run_id)

    def _run(self) -> None:
        """Main loop for background thread."""
        try:
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Subscribe to RunBus with UI queue size (64)
            self._step_queue = self._bus.subscribe_with_size(
                Topic.STEP_APPENDED, "live-ui", 64
            )
            self._episode_queue = self._bus.subscribe_with_size(
                Topic.EPISODE_FINALIZED, "live-ui", 64
            )

            self._logger.info(
                "Subscribed to RunBus topics",
                extra={"queue_size": 64},
            )

            # Run async loop
            self._loop.run_until_complete(self._process_events())
        except Exception as e:
            self._logger.exception("Fatal error in controller", extra={"error": str(e)})
        finally:
            if self._loop is not None:
                self._loop.close()
            self._logger.info("Controller loop exited")

    async def _process_events(self) -> None:
        """Process events from RunBus queues."""
        while not self._stop_event.is_set():
            try:
                # Process step events (non-blocking)
                if self._step_queue is not None:
                    await self._process_step_queue()

                # Process episode events (non-blocking)
                if self._episode_queue is not None:
                    await self._process_episode_queue()

                # Small sleep to avoid busy-waiting
                await asyncio.sleep(0.01)
            except Exception as e:
                self._logger.exception("Error processing events", extra={"error": str(e)})

    async def _process_step_queue(self) -> None:
        """Process all available step events from RunBus."""
        assert self._step_queue is not None

        while True:
            try:
                evt = self._step_queue.get_nowait()
                if not isinstance(evt, TelemetryEvent):
                    continue

                self._logger.debug(
                    "Step event received from RunBus",
                    extra={
                        "run_id": evt.run_id,
                        "agent_id": evt.agent_id,
                        "seq_id": evt.seq_id,
                    },
                )

                # Route to tab or buffer
                agent_id = evt.agent_id or "default"
                key = (evt.run_id, agent_id)

                # Initialize credits for actual agent_id on first event
                # This ensures credits are allocated for the specific agent
                credit_mgr = get_credit_manager()
                was_initialized = credit_mgr.initialize_stream(evt.run_id, agent_id)
                if was_initialized:
                    self._logger.debug(
                        "Initialized credits for actual agent_id",
                        extra={"run_id": evt.run_id, "agent_id": agent_id},
                    )

                # Route to tab if registered
                tab = self._tabs.get(key)
                self._logger.debug(
                    "Looking for tab",
                    extra={
                        "run_id": evt.run_id,
                        "agent_id": agent_id,
                        "key": key,
                        "tabs_available": list(self._tabs.keys()),
                        "tab_found": tab is not None,
                    },
                )
                if tab is not None:
                    try:
                        tab.add_step(evt.payload)
                    except Exception as e:
                        self._logger.warning(
                            "Error adding step to tab (tab may be destroyed)",
                            extra={"run_id": evt.run_id, "agent_id": agent_id, "error": str(e)},
                        )
                        # Remove the tab from tracking if it's destroyed
                        self._tabs.pop(key, None)
                else:
                    # Emit signal to request tab creation on first step
                    if key not in self._step_buffer:
                        self._logger.debug(
                            "First step received; requesting tab creation",
                            extra={"run_id": evt.run_id, "agent_id": agent_id},
                        )
                        # Format tab title: use agent_id if it's not numeric, otherwise use "Agent-{id}"
                        if agent_id and not agent_id.isdigit():
                            tab_title = f"Live – {agent_id}"
                        else:
                            tab_title = f"Live – Agent-{agent_id}" if agent_id else "Live – Agent"
                        # CRITICAL FIX: Use QMetaObject.invokeMethod to emit signal from main thread
                        # This is thread-safe and works across event loops
                        try:
                            QtCore.QMetaObject.invokeMethod(
                                self,
                                "_emit_tab_requested",
                                QtCore.Qt.ConnectionType.QueuedConnection,
                                QtCore.Q_ARG(str, evt.run_id),
                                QtCore.Q_ARG(str, agent_id),
                                QtCore.Q_ARG(str, tab_title),
                            )
                            self._logger.debug(
                                "Scheduled signal emission via QMetaObject.invokeMethod",
                                extra={"run_id": evt.run_id, "agent_id": agent_id},
                            )
                        except Exception as e:
                            self._logger.error(
                                "Failed to schedule signal emission",
                                extra={"error": str(e), "run_id": evt.run_id, "agent_id": agent_id},
                            )

                    # Buffer until tab is registered
                    if key not in self._step_buffer:
                        self._step_buffer[key] = []

                    # DEBUG: Log payload keys
                    payload_keys = list(evt.payload.keys()) if isinstance(evt.payload, dict) else "not_dict"
                    has_render = "render_payload_json" in evt.payload if isinstance(evt.payload, dict) else False
                    self._logger.debug(
                        f"[BUFFER] Buffering step: keys={payload_keys}, has_render_payload_json={has_render}",
                        extra={"run_id": evt.run_id, "agent_id": agent_id}
                    )

                    self._step_buffer[key].append(evt.payload)

            except Exception as e:
                # Catch queue.Empty (from thread-safe queue.Queue)
                if type(e).__name__ == 'Empty':
                    break
                raise

    async def _process_episode_queue(self) -> None:
        """Process all available episode events from RunBus."""
        assert self._episode_queue is not None

        while True:
            try:
                evt = self._episode_queue.get_nowait()
                if not isinstance(evt, TelemetryEvent):
                    continue

                self._logger.debug(
                    "Episode event received from RunBus",
                    extra={
                        "run_id": evt.run_id,
                        "agent_id": evt.agent_id,
                        "seq_id": evt.seq_id,
                    },
                )

                # Route to tab or buffer
                agent_id = evt.agent_id or "default"
                key = (evt.run_id, agent_id)

                # Route to tab if registered
                tab = self._tabs.get(key)
                if tab is not None:
                    try:
                        tab.add_episode(evt.payload)
                    except Exception as e:
                        self._logger.warning(
                            "Error adding episode to tab (tab may be destroyed)",
                            extra={"run_id": evt.run_id, "agent_id": agent_id, "error": str(e)},
                        )
                        # Remove the tab from tracking if it's destroyed
                        self._tabs.pop(key, None)
                else:
                    # Buffer until tab is registered
                    if key not in self._episode_buffer:
                        self._episode_buffer[key] = []
                    self._episode_buffer[key].append(evt.payload)

            except Exception as e:
                # Catch queue.Empty (from thread-safe queue.Queue)
                if type(e).__name__ == 'Empty':
                    break
                raise


__all__ = ["LiveTelemetryController"]
