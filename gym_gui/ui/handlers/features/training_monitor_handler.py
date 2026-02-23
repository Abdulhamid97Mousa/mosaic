"""Training monitor handler for MainWindow.

Extracts training run monitoring, polling, and watch subscription logic
from MainWindow. Manages background threads for run discovery.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from qtpy import QtCore

if TYPE_CHECKING:
    from gym_gui.services.trainer import TrainerClientRunner, RunStatus
    from gym_gui.services.trainer.client_runner import TrainerWatchStopped
    from gym_gui.controllers.live_telemetry_controllers import LiveTelemetryController
    from gym_gui.ui.panels.analytics_tabs import AnalyticsTabManager
    from gym_gui.ui.widgets.render_tabs import RenderTabs

_LOGGER = logging.getLogger(__name__)


class TrainingMonitorHandler(QObject):
    """Handler for monitoring training runs.

    Manages:
    - Polling daemon for new training runs
    - Watch subscription for run status updates
    - Auto-subscription to run telemetry
    - Metadata backfilling from disk
    - Training finished/completed signals
    """

    # Signal to request auto-subscribe on main thread
    auto_subscribe_requested = pyqtSignal(str)

    # Statuses to watch for
    WATCHED_RUN_STATUSES: List["RunStatus"] = []  # Populated in __init__

    def __init__(
        self,
        parent: QObject,
        live_controller: "LiveTelemetryController",
        analytics_tabs: "AnalyticsTabManager",
        render_tabs: "RenderTabs",
        run_metadata: Dict[tuple, Dict[str, Any]],
        trainer_dir: Path,
        log_callback: Optional[Callable[..., None]] = None,
        status_callback: Optional[Callable[[str, int], None]] = None,
        title_callback: Optional[Callable[[str], None]] = None,
        fastlane_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """Initialize the handler.

        Args:
            parent: Parent QObject.
            live_controller: Live telemetry controller.
            analytics_tabs: Analytics tab manager.
            render_tabs: Render tabs widget.
            run_metadata: Shared run metadata dictionary.
            trainer_dir: Path to trainer directory (var/trainer).
            log_callback: Optional callback for structured logging.
            status_callback: Callback for status bar messages (message, duration_ms).
            title_callback: Callback to update render group title.
            fastlane_callback: Callback to maybe open FastLane tab (run_id, agent_id).
        """
        super().__init__(parent)

        # Import RunStatus here to avoid circular imports
        from gym_gui.services.trainer import RunStatus
        self.WATCHED_RUN_STATUSES = [
            RunStatus.INIT,
            RunStatus.HANDSHAKE,
            RunStatus.READY,
            RunStatus.EXECUTING,
            RunStatus.PAUSED,
            RunStatus.FAULTED,
            RunStatus.TERMINATED,
        ]

        self._live_controller = live_controller
        self._analytics_tabs = analytics_tabs
        self._render_tabs = render_tabs
        self._run_metadata = run_metadata
        self._trainer_dir = trainer_dir
        self._log = log_callback or (lambda *args, **kwargs: None)
        self._status = status_callback or (lambda msg, dur: None)
        self._set_title = title_callback or (lambda t: None)
        self._maybe_open_fastlane = fastlane_callback or (lambda r, a: None)

        # State tracking
        self._known_runs: Set[str] = set()
        self._trainer_daemon_ready = False
        self._trainer_poll_failures = 0
        self._trainer_poll_quiet_logged = False

        # Watch thread management
        self._run_watch_subscription: Optional[Any] = None
        self._run_watch_thread: Optional[threading.Thread] = None
        self._run_watch_stop = threading.Event()

        # Connect internal signal
        self.auto_subscribe_requested.connect(self._auto_subscribe_run_main_thread)

    @property
    def known_runs(self) -> Set[str]:
        """Return the set of known run IDs."""
        return self._known_runs

    def poll_for_new_runs(self) -> None:
        """Poll daemon for new training runs and auto-subscribe to their telemetry."""
        from gym_gui.services.service_locator import get_service_locator
        from gym_gui.services.trainer import TrainerClientRunner, RunStatus

        try:
            locator = get_service_locator()
            runner = locator.resolve(TrainerClientRunner)
            if runner is None:
                self._log(message="TrainerClientRunner not available, skipping poll")
                return

            # List runs that should have tabs (active or recently completed)
            if self._trainer_daemon_ready:
                self._log(message="Polling daemon for active training runs...")

            future = runner.list_runs(
                statuses=self.WATCHED_RUN_STATUSES,
                deadline=3.0,
            )

            def on_done(fut):
                try:
                    response = fut.result(timeout=1.0)
                except Exception as exc:
                    self._trainer_poll_failures += 1
                    if self._trainer_daemon_ready:
                        self._log(message=f"Run poll failed: {exc}")
                    else:
                        if not self._trainer_poll_quiet_logged:
                            self._log(
                                message="Trainer daemon not yet reachable; suppressing poll failures until it responds"
                            )
                            self._trainer_poll_quiet_logged = True
                    return

                self._trainer_daemon_ready = True
                if self._trainer_poll_failures and self._trainer_poll_quiet_logged:
                    self._log(message="Trainer daemon responded; resuming poll logging")
                self._trainer_poll_failures = 0
                self._trainer_poll_quiet_logged = False
                self._log(message=f"Received {len(response.runs)} active runs from daemon")

                for record in response.runs:
                    run_id = str(record.run_id)
                    if run_id not in self._known_runs:
                        # Convert protobuf status integer to human-readable name
                        status_value = record.status
                        status_name = status_value
                        if isinstance(status_value, int):
                            status_name = RunStatus.from_proto(status_value).value
                        self._log(
                            message=f"Discovered new run: {run_id} (status={status_name}, proto={status_value})"
                        )
                        self._log(message=f"Calling auto-subscribe directly for run: {run_id}")
                        self._auto_subscribe_run(run_id)
                    else:
                        self._log(message=f"Run {run_id[:12]} already known, skipping")

            future.add_done_callback(on_done)

        except Exception as e:
            self._log(message="Failed to initiate run poll", exc_info=e)

    def _auto_subscribe_run(self, run_id: str) -> None:
        """Ensure auto-subscribe logic executes on the GUI thread."""
        current_thread = QtCore.QThread.currentThread()
        widget_thread = self.thread()
        if current_thread != widget_thread:
            self._log(
                message="Queueing auto-subscribe on GUI thread",
                extra={
                    "run_id": run_id,
                    "current_thread": repr(current_thread),
                    "widget_thread": repr(widget_thread),
                },
            )
            self.auto_subscribe_requested.emit(run_id)
            return

        self._auto_subscribe_run_main_thread(run_id)

    @pyqtSlot(str)
    def _auto_subscribe_run_main_thread(self, run_id: str) -> None:
        """Auto-subscribe to a newly discovered run (always called on main thread)."""
        self._log(message="Auto-subscribing to new run", extra={"run_id": run_id})
        self._known_runs.add(run_id)
        self.backfill_run_metadata_from_disk(run_id)
        try:
            self._live_controller.subscribe_to_run(run_id)
            self._set_title(f"Live Training - {run_id[:12]}...")
            self._status(f"Detected new training run: {run_id[:12]}...", 5000)
            self._log(
                message="Subscribed to run - waiting for telemetry steps to create agent tabs",
                extra={"run_id": run_id},
            )
        except Exception as e:
            self._log(
                message=f"Failed to subscribe to run {run_id}",
                exc_info=e,
            )

    def backfill_run_metadata_from_disk(self, run_id: str) -> None:
        """Load run metadata for previously scheduled runs discovered outside the submission flow."""
        config_path = self._trainer_dir / "configs" / f"config-{run_id}.json"
        if not config_path.exists():
            self._log(
                message="No disk config found for run_id, skipping backfill",
                extra={"run_id": run_id, "path": str(config_path)},
            )
            return

        try:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            self._log(
                message="Failed to load disk config",
                extra={"run_id": run_id, "path": str(config_path)},
                exc_info=e,
            )
            return

        # Extract the nested metadata from config file structure
        # Config structure: {"arguments": ..., "metadata": {"ui": ..., "worker": ...}, ...}
        metadata = config_payload.get("metadata", {}) if isinstance(config_payload, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}

        worker_meta = metadata.get("worker", {}) if isinstance(metadata, dict) else {}
        config_block = worker_meta.get("config", {}) if isinstance(worker_meta, dict) else {}
        agent_id = config_block.get("agent_id", "default") if isinstance(config_block, dict) else "default"
        if not isinstance(agent_id, str) or not agent_id.strip():
            agent_id = "default"

        key = (run_id, agent_id)

        # Build TensorBoard artifacts payload
        tensorboard_dir = self._trainer_dir / "runs" / run_id / "tensorboard"
        artifacts_payload: Optional[Dict[str, Any]] = None
        if tensorboard_dir.is_dir():
            artifacts_payload = {
                "tensorboard": {"log_dir": str(tensorboard_dir)},
            }

        if key in self._run_metadata:
            existing_payload = self._run_metadata[key]
            if artifacts_payload and "artifacts" not in existing_payload:
                existing_payload["artifacts"] = artifacts_payload
                self._log(
                    message="Backfilled tensorboard artifacts into existing metadata",
                    extra={"run_id": run_id, "agent_id": agent_id},
                )
            elif artifacts_payload:
                current_artifacts = existing_payload.get("artifacts")
                if not isinstance(current_artifacts, dict) and isinstance(artifacts_payload, dict):
                    existing_payload["artifacts"] = artifacts_payload
                    self._log(
                        message="Backfilled tensorboard artifacts into existing metadata",
                        extra={"run_id": run_id, "agent_id": agent_id},
                    )
            meta_payload = self._run_metadata.get(key)
            self._analytics_tabs.ensure_tensorboard_tab(run_id, agent_id, meta_payload)
            self._analytics_tabs.ensure_wandb_tab(run_id, agent_id, meta_payload)
            return

        self._run_metadata[key] = metadata
        self._log(
            message="Backfilled run metadata from trainer config",
            extra={"run_id": run_id, "agent_id": agent_id, "path": str(config_path)},
        )
        self._analytics_tabs.ensure_tensorboard_tab(run_id, agent_id, metadata)
        self._analytics_tabs.ensure_wandb_tab(run_id, agent_id, metadata)
        self._maybe_open_fastlane(run_id, agent_id)

    def start_run_watch(self) -> None:
        """Start watching for run status updates from the trainer daemon."""
        from gym_gui.services.service_locator import get_service_locator
        from gym_gui.services.trainer import TrainerClientRunner, RunStatus
        from gym_gui.services.trainer.client_runner import TrainerWatchStopped

        locator = get_service_locator()
        runner: Optional[TrainerClientRunner] = locator.resolve(TrainerClientRunner)
        if runner is None:
            self._log(message="TrainerClientRunner not available; skipping run watch subscription")
            return

        subscription = runner.watch_runs(statuses=self.WATCHED_RUN_STATUSES, since_seq=0)
        self._run_watch_subscription = subscription

        def _watch_loop() -> None:
            self._log(
                message=f"Run watch thread started (statuses={','.join([status.name for status in self.WATCHED_RUN_STATUSES])})"
            )
            while not self._run_watch_stop.is_set():
                try:
                    record = subscription.get(timeout=1.0)
                except TrainerWatchStopped:
                    self._log(message="Run watch subscription closed by daemon")
                    break
                except TimeoutError:
                    continue
                except Exception as exc:
                    self._log(message=f"Run watch error: {exc}")
                    continue

                run_id = getattr(record, "run_id", "")
                status_value = getattr(record, "status", None)
                # Convert protobuf status integer to human-readable name
                status_name = status_value
                if isinstance(status_value, int):
                    status_name = RunStatus.from_proto(status_value).value
                self._log(
                    message=f"Run watch update: run_id={run_id} status={status_name} (proto={status_value})"
                )
                QtCore.QTimer.singleShot(0, lambda rid=run_id: self._auto_subscribe_run(rid))

            subscription.close()
            self._log(message="Run watch thread exiting")

        self._run_watch_thread = threading.Thread(
            target=_watch_loop,
            name="trainer-run-watch",
            daemon=True,
        )
        self._run_watch_thread.start()

    def shutdown_run_watch(self) -> None:
        """Shutdown the run watch thread and subscription."""
        self._run_watch_stop.set()
        if self._run_watch_subscription is not None:
            try:
                self._run_watch_subscription.close()
            except Exception as exc:
                self._log(
                    message="Failed to close run watch subscription during shutdown",
                    extra={
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    exc_info=exc,
                )
            self._run_watch_subscription = None
        if self._run_watch_thread is not None:
            self._run_watch_thread.join(timeout=2.0)
            self._run_watch_thread = None

    def on_run_completed(self, run_id: str) -> None:
        """Handle run completion - unsubscribe from telemetry.

        Note: Live-Agent tabs remain open so user can review the training.
        Replay tabs are created by on_training_finished() signal.
        """
        self._log(message="Run completed signal received", extra={"run_id": run_id})

        # Unsubscribe from telemetry (stops new events from arriving)
        if self._live_controller:
            try:
                self._live_controller.unsubscribe_from_run(run_id)
                self._log(message="Unsubscribed from telemetry", extra={"run_id": run_id})
            except Exception as e:
                self._log(
                    message="Failed to unsubscribe from telemetry",
                    exc_info=e,
                    extra={"run_id": run_id},
                )

        self._log(
            message="Run completed - Live-Agent tabs remain open for review",
            extra={"run_id": run_id},
        )
