from __future__ import annotations

"""Helpers to bootstrap shared services for the application."""

import logging
import os

from gym_gui.config.paths import VAR_TELEMETRY_DIR, ensure_var_directories, VAR_ROOT
from gym_gui.rendering import RendererRegistry, create_default_renderer_registry
from gym_gui.services.action_mapping import ContinuousActionMapper, create_default_action_mapper
from gym_gui.services.actor import ActorService, BDIQAgent, HumanKeyboardActor, LLMMultiStepAgent
from gym_gui.services.frame_storage import FrameStorageService
from gym_gui.services.service_locator import ServiceLocator, get_service_locator
from gym_gui.services.trainer import TrainerClient, TrainerClientConfig, TrainerClientRunner
from gym_gui.services.trainer.launcher import TrainerDaemonHandle, ensure_trainer_daemon_running
from gym_gui.services.trainer.streams import TelemetryAsyncHub
from gym_gui.services.storage import StorageRecorderService
from gym_gui.services.telemetry import TelemetryService
from gym_gui.telemetry import TelemetrySQLiteStore
from gym_gui.telemetry.db_sink import TelemetryDBSink
from gym_gui.telemetry.run_bus import get_bus
from gym_gui.telemetry.health import HealthMonitor
from gym_gui.telemetry.constants import (
    TELEMETRY_HUB_MAX_QUEUE,
    TELEMETRY_HUB_BUFFER_SIZE,
    DB_SINK_BATCH_SIZE,
    DB_SINK_CHECKPOINT_INTERVAL,
    DB_SINK_WRITER_QUEUE_SIZE,
    HEALTH_MONITOR_HEARTBEAT_INTERVAL_S,
)
from gym_gui.controllers.live_telemetry_controllers import LiveTelemetryController


def bootstrap_default_services() -> ServiceLocator:
    """Register the default service set into the shared locator."""

    locator = get_service_locator()

    storage = StorageRecorderService()
    telemetry = TelemetryService()
    telemetry.attach_storage(storage)
    ensure_var_directories()
    telemetry_db = VAR_TELEMETRY_DIR / "telemetry.sqlite"
    telemetry_store = TelemetrySQLiteStore(telemetry_db)
    telemetry.attach_store(telemetry_store)
    if os.getenv("GYM_GUI_RESET_TELEMETRY") == "1":
        telemetry_store.delete_all_episodes(wait=True)

    legacy_trainer_db = VAR_ROOT / "trainer" / "trainer.sqlite"
    if legacy_trainer_db.exists():
        logging.getLogger(__name__).warning(
            "Detected legacy trainer database at %s – new runs write to %s. Consider migrating or deleting the old file to avoid confusion.",
            legacy_trainer_db,
            VAR_TELEMETRY_DIR.parent / "trainer" / "trainer.sqlite",
        )

    actors = ActorService()
    actors.register_actor(
        HumanKeyboardActor(),
        display_name="Human (Keyboard)",
        description="Forward keyboard input captured by the UI.",
        policy_label="Manual control",
        backend_label="Qt keyboard input",
        activate=True,
    )
    actors.register_actor(
        BDIQAgent(),
        display_name="BDI-Q Agent",
        description="Belief-Desire-Intention agent with Q-learning hooks.",
        policy_label="BDI planner + Q-learning",
        backend_label="In-process Python actor",
    )
    actors.register_actor(
        LLMMultiStepAgent(),
        display_name="LLM Multi-Step Agent",
        description="Delegates decisions to an integrated language model pipeline.",
        policy_label="LLM planning with tool calls",
        backend_label="External language model service",
    )

    action_mapper: ContinuousActionMapper = create_default_action_mapper()
    renderer_registry: RendererRegistry = create_default_renderer_registry()
    frame_storage: FrameStorageService = FrameStorageService()

    locator.register(StorageRecorderService, storage)
    locator.register(TelemetryService, telemetry)
    locator.register(TelemetrySQLiteStore, telemetry_store)
    locator.register(ActorService, actors)
    locator.register(ContinuousActionMapper, action_mapper)
    locator.register(FrameStorageService, frame_storage)

    client_config = TrainerClientConfig()

    daemon_handle = ensure_trainer_daemon_running(target=client_config.target)

    trainer_client = TrainerClient(client_config)
    trainer_runner = TrainerClientRunner(trainer_client)
    
    # Initialize telemetry hub for live streaming
    # Increased buffer_size from 256 to 512 to handle high-frequency training (episode 658+)
    # max_queue=2048 handles gRPC stream buffering, buffer_size=512 handles UI processing lag
    telemetry_hub = TelemetryAsyncHub(
        max_queue=TELEMETRY_HUB_MAX_QUEUE,
        buffer_size=TELEMETRY_HUB_BUFFER_SIZE,
    )
    # Hub will auto-start on first subscribe_run call

    # Create live telemetry controller and start RunBus subscription
    # UI queue size: 64 events (responsive rendering)
    live_controller = LiveTelemetryController(telemetry_hub, trainer_client)
    live_controller.start()

    # Initialize and start database sink for durable persistence
    # Writer queue is larger (512) to handle backlog, UI queue is smaller (64)
    bus = get_bus()
    db_sink = TelemetryDBSink(
        telemetry_store,
        bus,
        batch_size=DB_SINK_BATCH_SIZE,
        checkpoint_interval=DB_SINK_CHECKPOINT_INTERVAL,
        writer_queue_size=DB_SINK_WRITER_QUEUE_SIZE,
    )
    db_sink.start()

    # Initialize and start health monitor for observability
    # Emits RUN_HEARTBEAT events every 5 seconds
    health_monitor = HealthMonitor(
        bus,
        heartbeat_interval=HEALTH_MONITOR_HEARTBEAT_INTERVAL_S,
    )
    health_monitor.start()

    locator.register(RendererRegistry, renderer_registry)
    locator.register(TrainerClient, trainer_client)
    locator.register(TrainerClientRunner, trainer_runner)
    locator.register(TelemetryAsyncHub, telemetry_hub)
    locator.register(LiveTelemetryController, live_controller)
    locator.register(TelemetryDBSink, db_sink)
    locator.register(HealthMonitor, health_monitor)
    locator.register(TrainerDaemonHandle, daemon_handle)

    # Also register under string keys for convenience in legacy code.
    locator.register("storage", storage)
    locator.register("telemetry", telemetry)
    locator.register("telemetry_store", telemetry_store)
    locator.register("actors", actors)
    locator.register("action_mapper", action_mapper)
    locator.register("renderer_registry", renderer_registry)
    locator.register("trainer_client", trainer_client)
    locator.register("trainer_client_runner", trainer_runner)
    locator.register("telemetry_hub", telemetry_hub)
    locator.register("live_telemetry_controller", live_controller)
    locator.register("trainer_daemon_handle", daemon_handle)

    return locator


__all__ = ["bootstrap_default_services"]
