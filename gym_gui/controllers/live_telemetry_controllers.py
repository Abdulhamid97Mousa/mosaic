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
from collections import deque
from typing import TYPE_CHECKING, Optional, Dict

from qtpy import QtCore

from gym_gui.logging_config.log_constants import (
    LOG_BUFFER_DROP,
    LOG_CREDIT_RESUMED,
    LOG_CREDIT_STARVED,
    LOG_LIVE_CONTROLLER_ALREADY_RUNNING,
    LOG_LIVE_CONTROLLER_INITIALIZED,
    LOG_LIVE_CONTROLLER_RUN_ALREADY_SUBSCRIBED,
    LOG_LIVE_CONTROLLER_RUN_SUBSCRIBED,
    LOG_LIVE_CONTROLLER_RUN_UNSUBSCRIBED,
    LOG_LIVE_CONTROLLER_THREAD_STARTED,
    LOG_LIVE_CONTROLLER_THREAD_STOPPED,
    LOG_LIVE_CONTROLLER_THREAD_STOP_TIMEOUT,
    LOG_TELEMETRY_CONTROLLER_THREAD_ERROR,
    LOG_TELEMETRY_SUBSCRIBE_ERROR,
    LOG_LIVE_CONTROLLER_LOOP_EXITED,
    LOG_LIVE_CONTROLLER_BUFFER_STEPS_FLUSHED,
    LOG_LIVE_CONTROLLER_BUFFER_EPISODES_FLUSHED,
    LOG_LIVE_CONTROLLER_QUEUE_OVERFLOW,
    LOG_LIVE_CONTROLLER_RUN_COMPLETED,
    LOG_LIVE_CONTROLLER_RUNBUS_SUBSCRIBED,
    LOG_LIVE_CONTROLLER_TAB_ADD_FAILED,
    LOG_LIVE_CONTROLLER_SIGNAL_EMIT_FAILED,
)
from gym_gui.logging_config.helpers import LogConstantMixin

from gym_gui.telemetry.run_bus import get_bus
from gym_gui.telemetry.events import Topic, TelemetryEvent
from gym_gui.telemetry.credit_manager import get_credit_manager
from gym_gui.telemetry.constants import STEP_BUFFER_SIZE, EPISODE_BUFFER_SIZE

if TYPE_CHECKING:
    from gym_gui.services.trainer import TrainerClient
    from gym_gui.services.trainer.streams import TelemetryAsyncHub
    from gym_gui.ui.widgets.live_telemetry_tab import LiveTelemetryTab


_LOGGER = logging.getLogger(__name__)


class LiveTelemetryController(QtCore.QObject, LogConstantMixin):
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
        self._logger = _LOGGER
        self._hub = hub
        self._client = client
        self._active_runs: set[str] = set()
        self._tabs: Dict[tuple[str, str], "LiveTelemetryTab"] = {}  # (run_id, agent_id) -> tab

        # Buffer for steps that arrive before tab is registered (race condition mitigation)
        # Use bounded deques to prevent memory leaks if tabs never open
        self._step_buffer: Dict[tuple[str, str], deque] = {}  # (run_id, agent_id) -> deque of steps
        self._episode_buffer: Dict[tuple[str, str], deque] = {}  # (run_id, agent_id) -> deque of episodes

        # Store rendering throttle value per run (from TELEMETRY_SAMPLING_INTERVAL env var)
        self._render_throttle_per_run: Dict[str, int] = {}  # run_id -> throttle_interval

        # Store render delay and enable flag per run (from train form metadata)
        self._render_delay_per_run: Dict[str, int] = {}  # run_id -> render delay (ms)
        self._render_enabled_per_run: Dict[str, bool] = {}  # run_id -> live render enabled flag

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
        self._control_queue: Optional[queue.Queue] = None

        # Still wire bridge signals for overflow and run_completed (not in RunBus yet)
        self._hub.bridge.queue_overflow.connect(self._on_queue_overflow)
        self._hub.bridge.run_completed.connect(self._on_run_completed_from_bridge)

        self.log_constant(
            LOG_LIVE_CONTROLLER_INITIALIZED,
            extra={"bridge": type(self._hub.bridge).__name__},
        )

    def start(self) -> None:
        """Start the background thread for RunBus subscription."""
        if self._thread is not None:
            self.log_constant(LOG_LIVE_CONTROLLER_ALREADY_RUNNING)
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="live-telemetry-runbus",
            daemon=True,
        )
        self._thread.start()
        self.log_constant(LOG_LIVE_CONTROLLER_THREAD_STARTED)

    def stop(self) -> None:
        """Stop the background thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            self.log_constant(LOG_LIVE_CONTROLLER_THREAD_STOP_TIMEOUT)
        else:
            self.log_constant(LOG_LIVE_CONTROLLER_THREAD_STOPPED)
        self._thread = None

    def subscribe_to_run(self, run_id: str) -> None:
        """Start streaming telemetry for a run.

        Pre-initialize credits to prevent chicken-and-egg deadlock.
        Credits are initialized BEFORE first event arrives, ensuring the
        credit system is ready when the first step/episode is received.
        """
        if run_id in self._active_runs:
            self.log_constant(
                LOG_LIVE_CONTROLLER_RUN_ALREADY_SUBSCRIBED,
                extra={"run_id": run_id},
            )
            return

        self.log_constant(
            LOG_LIVE_CONTROLLER_RUN_SUBSCRIBED,
            extra={"run_id": run_id, "stage": "request"},
        )
        self._active_runs.add(run_id)

        # Pre-initialize credits for default agent
        # This ensures credits exist BEFORE first event arrives
        # We use "default" as a placeholder; actual agent_id will be extracted from first event
        credit_mgr = get_credit_manager()
        credit_mgr.initialize_stream(run_id, "default")
        _LOGGER.debug(
            "Pre-initialized credits for default agent",
            extra={"run_id": run_id, "agent_id": "default"},
        )

        self._hub.subscribe_run(run_id, self._client)
        self.log_constant(
            LOG_LIVE_CONTROLLER_RUN_SUBSCRIBED,
            extra={"run_id": run_id, "stage": "bound"},
        )

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
        self._render_throttle_per_run.pop(run_id, None)
        self._render_delay_per_run.pop(run_id, None)
        self._render_enabled_per_run.pop(run_id, None)
        self._step_buffer_size_per_run.pop(run_id, None)
        self._episode_buffer_size_per_run.pop(run_id, None)
        self._game_id_per_run.pop(run_id, None)
        self.log_constant(
            LOG_LIVE_CONTROLLER_RUN_UNSUBSCRIBED,
            extra={"run_id": run_id},
        )

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
        _LOGGER.debug(
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
        _LOGGER.debug(
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

    def set_render_delay_for_run(self, run_id: str, delay_ms: int) -> None:
        self._render_delay_per_run[run_id] = max(0, delay_ms)

    def get_render_delay_for_run(self, run_id: str) -> int:
        return self._render_delay_per_run.get(run_id, 100)

    def set_live_render_enabled_for_run(self, run_id: str, enabled: bool) -> None:
        self._render_enabled_per_run[run_id] = enabled

    def is_live_render_enabled(self, run_id: str) -> bool:
        return self._render_enabled_per_run.get(run_id, True)

    def set_game_id_for_run(self, run_id: str, game_id: str) -> None:
        """Store the game_id for a run (for passing to dynamic tabs).

        Args:
            run_id: The training run ID
            game_id: The game environment ID (e.g., "FrozenLake-v1")
        """
        self._game_id_per_run[run_id] = game_id
        _LOGGER.debug(
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
        _LOGGER.debug(
            "Granted initial credits to tab",
            extra={"run_id": run_id, "agent_id": agent_id, "amount": 200},
        )

        # Apply rendering throttle if set for this run
        if run_id in self._render_throttle_per_run:
            throttle = self._render_throttle_per_run[run_id]
            tab.set_render_throttle_interval(throttle)
            _LOGGER.debug(
                "Applied render throttle to tab",
                extra={"run_id": run_id, "agent_id": agent_id, "throttle": throttle},
            )

        _LOGGER.debug(
            "Tab registered and ready for telemetry",
            extra={"run_id": run_id, "agent_id": agent_id, "tab_type": type(tab).__name__},
        )

        # Flush any buffered steps that arrived before tab was registered
        if key in self._step_buffer:
            buffered_steps = self._step_buffer.pop(key)
            self.log_constant(
                LOG_LIVE_CONTROLLER_BUFFER_STEPS_FLUSHED,
                extra={
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "buffered_count": len(buffered_steps),
                },
            )
            for payload in buffered_steps:
                tab.add_step(payload)
                credit_mgr.grant_credits(run_id, agent_id, 1)

        # Flush any buffered episodes that arrived before tab was registered
        if key in self._episode_buffer:
            buffered_episodes = self._episode_buffer.pop(key)
            self.log_constant(
                LOG_LIVE_CONTROLLER_BUFFER_EPISODES_FLUSHED,
                extra={
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "buffered_count": len(buffered_episodes),
                },
            )
            for payload in buffered_episodes:
                tab.add_episode(payload)
                credit_mgr.grant_credits(run_id, agent_id, 1)

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

        self.log_constant(
            LOG_LIVE_CONTROLLER_QUEUE_OVERFLOW,
            extra={"run_id": run_id, "stream_type": stream_type, "dropped": dropped},
        )

    def _on_run_completed_from_bridge(self, run_id: str) -> None:
        """Handle run completion signal from bridge."""
        self.log_constant(
            LOG_LIVE_CONTROLLER_RUN_COMPLETED,
            extra={"run_id": run_id},
        )
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
            self._control_queue = self._bus.subscribe_with_size(
                Topic.CONTROL, "live-ui-control", 32
            )

            self.log_constant(
                LOG_LIVE_CONTROLLER_RUNBUS_SUBSCRIBED,
                extra={"step_queue_size": 64, "episode_queue_size": 64, "control_queue_size": 32},
            )

            # Run async loop
            self._loop.run_until_complete(self._process_events())
        except Exception as e:
            self.log_constant(
                LOG_TELEMETRY_CONTROLLER_THREAD_ERROR,
                extra={"error": str(e)},
                exc_info=e,
            )
        finally:
            if self._loop is not None:
                self._loop.close()
            self.log_constant(LOG_LIVE_CONTROLLER_LOOP_EXITED)

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

                if self._control_queue is not None:
                    await self._process_control_queue()

                # Small sleep to avoid busy-waiting
                await asyncio.sleep(0.01)
            except Exception as e:
                self.log_constant(
                    LOG_TELEMETRY_CONTROLLER_THREAD_ERROR,
                    extra={"error": str(e), "stage": "process_events"},
                    exc_info=e,
                )

    async def _process_step_queue(self) -> None:
        """Process all available step events from RunBus."""
        assert self._step_queue is not None

        credit_mgr = get_credit_manager()

        while True:
            try:
                evt = self._step_queue.get_nowait()
                if not isinstance(evt, TelemetryEvent):
                    continue

                _LOGGER.debug(
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
                    _LOGGER.debug(
                        "Initialized credits for actual agent_id",
                        extra={"run_id": evt.run_id, "agent_id": agent_id},
                    )

                # Route to tab if registered
                tab = self._tabs.get(key)
                _LOGGER.debug(
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
                        credit_mgr.grant_credits(evt.run_id, agent_id, 1)
                    except Exception as e:
                        self.log_constant(
                            LOG_LIVE_CONTROLLER_TAB_ADD_FAILED,
                            extra={
                                "run_id": evt.run_id,
                                "agent_id": agent_id,
                                "error": str(e),
                                "payload_keys": list(evt.payload.keys()) if isinstance(evt.payload, dict) else "-",
                            },
                            exc_info=e,
                        )
                        # Remove the tab from tracking if it's destroyed
                        self._tabs.pop(key, None)
                else:
                    # Emit signal to request tab creation on first step
                    if key not in self._step_buffer:
                        _LOGGER.debug(
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
                            _LOGGER.debug(
                                "Scheduled signal emission via QMetaObject.invokeMethod",
                                extra={"run_id": evt.run_id, "agent_id": agent_id},
                            )
                        except Exception as e:
                            self.log_constant(
                                LOG_LIVE_CONTROLLER_SIGNAL_EMIT_FAILED,
                                extra={
                                    "run_id": evt.run_id,
                                    "agent_id": agent_id,
                                },
                                exc_info=e,
                            )

                    # Buffer until tab is registered
                    if key not in self._step_buffer:
                        self._step_buffer[key] = deque(maxlen=STEP_BUFFER_SIZE)

                    payload_keys = list(evt.payload.keys()) if isinstance(evt.payload, dict) else "not_dict"
                    has_render = (
                        isinstance(evt.payload, dict)
                        and "render_payload_json" in evt.payload
                    )
                    self._step_buffer[key].append(evt.payload)
                    level = (
                        LOG_BUFFER_DROP.level
                        if isinstance(LOG_BUFFER_DROP.level, int)
                        else getattr(logging, LOG_BUFFER_DROP.level)
                    )
                    _LOGGER.log(
                        level,
                        "%s %s",
                        LOG_BUFFER_DROP.code,
                        LOG_BUFFER_DROP.message,
                        extra={
                            "run_id": evt.run_id,
                            "agent_id": agent_id,
                            "log_code": LOG_BUFFER_DROP.code,
                            "buffer_size": len(self._step_buffer[key]),
                            "payload_keys": payload_keys,
                            "has_render_payload_json": has_render,
                            "component": LOG_BUFFER_DROP.component,
                            "subcomponent": LOG_BUFFER_DROP.subcomponent,
                            "tags": ",".join(LOG_BUFFER_DROP.tags),
                        },
                    )

            except Exception as e:
                # Catch queue.Empty (from thread-safe queue.Queue)
                if type(e).__name__ == 'Empty':
                    break
                raise

    async def _process_episode_queue(self) -> None:
        """Process all available episode events from RunBus."""
        assert self._episode_queue is not None

        credit_mgr = get_credit_manager()

        while True:
            try:
                evt = self._episode_queue.get_nowait()
                if not isinstance(evt, TelemetryEvent):
                    continue

                _LOGGER.debug(
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
                        credit_mgr.grant_credits(evt.run_id, agent_id, 1)
                    except Exception as e:
                        self.log_constant(
                            LOG_LIVE_CONTROLLER_TAB_ADD_FAILED,
                            extra={
                                "run_id": evt.run_id,
                                "agent_id": agent_id,
                                "error": str(e),
                                "payload_keys": list(evt.payload.keys()) if isinstance(evt.payload, dict) else "-",
                                "entry_type": "episode",
                            },
                            exc_info=e,
                        )
                        # Remove the tab from tracking if it's destroyed
                        self._tabs.pop(key, None)
                else:
                    # Buffer until tab is registered (bounded deque to prevent memory leaks)
                    if key not in self._episode_buffer:
                        self._episode_buffer[key] = deque(maxlen=EPISODE_BUFFER_SIZE)
                    self._episode_buffer[key].append(evt.payload)

            except queue.Empty:
                break

    async def _process_control_queue(self) -> None:
        """Process control-plane events (credit starvation/resume)."""
        assert self._control_queue is not None

        while True:
            try:
                evt = self._control_queue.get_nowait()
            except queue.Empty:
                break

            if not isinstance(evt, TelemetryEvent):
                continue

            agent_id = evt.agent_id or "default"
            state = evt.payload.get("state") if isinstance(evt.payload, dict) else None
            stream_type = evt.payload.get("stream_type") if isinstance(evt.payload, dict) else "unknown"

            if state == "STARVED":
                level = (
                    LOG_CREDIT_STARVED.level
                    if isinstance(LOG_CREDIT_STARVED.level, int)
                    else getattr(logging, LOG_CREDIT_STARVED.level)
                )
                _LOGGER.log(
                    level,
                    "%s %s",
                    LOG_CREDIT_STARVED.code,
                    LOG_CREDIT_STARVED.message,
                    extra={
                        "run_id": evt.run_id,
                        "agent_id": agent_id,
                        "stream_type": stream_type,
                        "log_code": LOG_CREDIT_STARVED.code,
                        "component": LOG_CREDIT_STARVED.component,
                        "subcomponent": LOG_CREDIT_STARVED.subcomponent,
                        "tags": ",".join(LOG_CREDIT_STARVED.tags),
                    },
                )
            elif state == "RESUMED":
                level = (
                    LOG_CREDIT_RESUMED.level
                    if isinstance(LOG_CREDIT_RESUMED.level, int)
                    else getattr(logging, LOG_CREDIT_RESUMED.level)
                )
                _LOGGER.log(
                    level,
                    "%s %s",
                    LOG_CREDIT_RESUMED.code,
                    LOG_CREDIT_RESUMED.message,
                    extra={
                        "run_id": evt.run_id,
                        "agent_id": agent_id,
                        "stream_type": stream_type,
                        "log_code": LOG_CREDIT_RESUMED.code,
                        "component": LOG_CREDIT_RESUMED.component,
                        "subcomponent": LOG_CREDIT_RESUMED.subcomponent,
                        "tags": ",".join(LOG_CREDIT_RESUMED.tags),
                    },
                )
            else:
                _LOGGER.debug(
                    "Received CONTROL event",
                    extra={"run_id": evt.run_id, "agent_id": agent_id, "payload": evt.payload},
                )


__all__ = ["LiveTelemetryController"]
