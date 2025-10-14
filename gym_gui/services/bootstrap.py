from __future__ import annotations

"""Helpers to bootstrap shared services for the application."""

import os

from gym_gui.config.paths import VAR_TELEMETRY_DIR, ensure_var_directories
from gym_gui.rendering import RendererRegistry, create_default_renderer_registry
from gym_gui.services.action_mapping import ContinuousActionMapper, create_default_action_mapper
from gym_gui.services.actor import ActorService, BDIQAgent, HumanKeyboardActor, LLMMultiStepAgent
from gym_gui.services.service_locator import ServiceLocator, get_service_locator
from gym_gui.services.trainer import (
    TrainerClient,
    TrainerClientConfig,
    TrainerClientRunner,
)
from gym_gui.services.storage import StorageRecorderService
from gym_gui.services.telemetry import TelemetryService
from gym_gui.telemetry import TelemetrySQLiteStore


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

    actors = ActorService()
    actors.register_actor(
        HumanKeyboardActor(),
        display_name="Human (Keyboard)",
        description="Forward keyboard input captured by the UI.",
        activate=True,
    )
    actors.register_actor(
        BDIQAgent(),
        display_name="BDI-Q Agent",
        description="Belief-Desire-Intention agent with Q-learning hooks.",
    )
    actors.register_actor(
        LLMMultiStepAgent(),
        display_name="LLM Multi-Step Agent",
        description="Delegates decisions to an integrated language model pipeline.",
    )

    action_mapper: ContinuousActionMapper = create_default_action_mapper()
    renderer_registry: RendererRegistry = create_default_renderer_registry()

    locator.register(StorageRecorderService, storage)
    locator.register(TelemetryService, telemetry)
    locator.register(TelemetrySQLiteStore, telemetry_store)
    locator.register(ActorService, actors)
    locator.register(ContinuousActionMapper, action_mapper)
    trainer_client = TrainerClient(TrainerClientConfig())
    trainer_runner = TrainerClientRunner(trainer_client)

    locator.register(RendererRegistry, renderer_registry)
    locator.register(TrainerClient, trainer_client)
    locator.register(TrainerClientRunner, trainer_runner)

    # Also register under string keys for convenience in legacy code.
    locator.register("storage", storage)
    locator.register("telemetry", telemetry)
    locator.register("telemetry_store", telemetry_store)
    locator.register("actors", actors)
    locator.register("action_mapper", action_mapper)
    locator.register("renderer_registry", renderer_registry)
    locator.register("trainer_client", trainer_client)
    locator.register("trainer_client_runner", trainer_runner)

    return locator


__all__ = ["bootstrap_default_services"]
