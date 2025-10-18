"""Controller for managing live telemetry streaming from trainer runs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Dict

from qtpy import QtCore

if TYPE_CHECKING:
    from gym_gui.services.trainer import TrainerClient
    from gym_gui.services.trainer.streams import TelemetryAsyncHub
    from gym_gui.ui.widgets.live_telemetry_tab import LiveTelemetryTab


class LiveTelemetryController(QtCore.QObject):
    """Coordinates telemetry hub lifecycle and manages dynamic per-agent tab creation."""

    # Signals
    run_tab_requested = QtCore.Signal(str, str, str)  # type: ignore[attr-defined] # (run_id, agent_id, tab_title)
    telemetry_stats_updated = QtCore.Signal(str, dict)  # type: ignore[attr-defined] # (run_id, stats)
    step_received = QtCore.Signal(object)  # type: ignore[attr-defined] # (step_data)
    episode_received = QtCore.Signal(object)  # type: ignore[attr-defined] # (episode_data)
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

        # Wire bridge signals
        self._hub.bridge.step_received.connect(self._on_step_received)
        self._hub.bridge.episode_received.connect(self._on_episode_received)
        self._hub.bridge.queue_overflow.connect(self._on_queue_overflow)
        self._hub.bridge.run_completed.connect(self._on_run_completed_from_bridge)

    def subscribe_to_run(self, run_id: str) -> None:
        """Start streaming telemetry for a run."""
        if run_id in self._active_runs:
            self._logger.debug("Already subscribed to run", extra={"run_id": run_id})
            return
        self._logger.debug("Subscribing controller to run: %s", run_id)
        self._active_runs.add(run_id)
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

    def register_tab(self, run_id: str, agent_id: str, tab: "LiveTelemetryTab") -> None:
        """Register a newly created tab widget for routing telemetry."""
        key = (run_id, agent_id)
        self._tabs[key] = tab
        self._logger.debug("Registered tab", extra={"run_id": run_id, "agent_id": agent_id})

    # Signal handlers
    def _on_step_received(self, step: object) -> None:
        from gym_gui.services.trainer.streams import TelemetryStep

        if not isinstance(step, TelemetryStep):
            self._logger.debug(f"Received non-TelemetryStep object: {type(step)}")
            return

        run_id = step.run_id
        payload = getattr(step, "payload", None)
        # Extract agent_id from dict payloads (preferred) or attribute fallback
        agent_id = (
            (payload.get("agent_id") if isinstance(payload, dict) else getattr(payload, "agent_id", None))
            or "default"
        )

        # Extra debug snapshot for troubleshooting
        payload_type = type(payload).__name__
        if isinstance(payload, dict):
            payload_keys = list(payload.keys())[:12]
            episode_idx = payload.get("episode_index", payload.get("episode"))
            step_idx = payload.get("step_index", payload.get("step"))
        else:
            payload_keys = []
            episode_idx = getattr(payload, "episode_index", getattr(payload, "episode", None))
            step_idx = getattr(payload, "step_index", getattr(payload, "step", None))

        self._logger.debug(
            "_on_step_received: telemetry step",
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "payload_type": payload_type,
                "payload_keys": payload_keys,
                "episode_index": episode_idx,
                "step_index": step_idx,
            },
        )

        key = (run_id, agent_id)

        # Request dynamic tab creation on first step for this (run, agent) pair
        if key not in self._tabs:
            self._tabs[key] = None  # type: ignore[assignment]
            tab_title = f"Agent-{agent_id}-Online"
            self.run_tab_requested.emit(run_id, agent_id, tab_title)
            self._logger.debug(
                "Requesting tab creation",
                extra={"run_id": run_id, "agent_id": agent_id, "title": tab_title},
            )

        # Emit step signal for main window to handle
        self.step_received.emit(step)
        self._logger.debug("Emitted step_received signal", extra={"run_id": run_id, "agent_id": agent_id})

        # Route step to registered tab if available
        tab = self._tabs.get(key)
        if tab is not None:
            tab.add_step(payload)

    def _on_episode_received(self, episode: object) -> None:
        from gym_gui.services.trainer.streams import TelemetryEpisode

        if not isinstance(episode, TelemetryEpisode):
            return

        run_id = episode.run_id
        # Extract agent_id from payload if available
        # payload is a dict, not an object
        agent_id = None
        if isinstance(episode.payload, dict):
            agent_id = episode.payload.get("agent_id")
        else:
            agent_id = getattr(episode.payload, "agent_id", None)
        agent_id = agent_id or "unknown"
        key = (run_id, agent_id)

        # Emit episode signal for main window to handle
        self.episode_received.emit(episode)

        # Route episode to registered tab
        tab = self._tabs.get(key)
        if tab is not None:
            tab.add_episode(episode.payload)

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


__all__ = ["LiveTelemetryController"]
