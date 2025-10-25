"""Shared structured log constants for the Gym GUI project.

This module centralises the error and event identifiers used across the
application so that emitters can attach consistent metadata (component,
subcomponent, tags) alongside a stable log code and human readable message.

Each constant is represented by :class:`LogConstant`, which aligns with the
expectations of the helper utilities defined throughout the codebase.

## Usage

Log constants are designed to be used with the `_log_constant()` helper pattern:

    from gym_gui.logging_config.log_constants import LOG_SESSION_STEP_ERROR, _log_constant

    try:
        adapter.step()
    except Exception as exc:
        _log_constant(LOG_SESSION_STEP_ERROR, exc_info=exc, extra={"run_id": run_id})

## Lookup & Filtering

The module provides helpers for discovering constants at runtime:

    from gym_gui.logging_config.log_constants import get_constant_by_code, list_known_components

    # Retrieve a constant by code
    const = get_constant_by_code("LOG401")  # → LOG_SESSION_ADAPTER_LOAD_ERROR

    # List all known components
    components = list_known_components()  # → ["Controller", "Adapter", "Service", ...]

    # Build dynamic component/severity filters for the GUI
    component_snapshot = get_component_snapshot()  # → {"Controller": {"Session", "LiveTelemetry", ...}, ...}

## Validation

To validate that all log constants conform to logging standards:

    from gym_gui.logging_config.log_constants import validate_log_constants

    errors = validate_log_constants()
    if errors:
        raise ValueError(f"Invalid log constants: {errors}")

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Set
import logging


@dataclass(frozen=True, slots=True)
class LogConstant:
    """Container describing a structured log record template."""

    code: str
    level: int | str
    message: str
    component: str
    subcomponent: str
    tags: tuple[str, ...] = ()


def _tags(*values: str) -> tuple[str, ...]:
    return tuple(v for v in values if v)


def _constant(
    code: str,
    level: int | str,
    message: str,
    *,
    component: str,
    subcomponent: str,
    tags: Iterable[str] = (),
) -> LogConstant:
    return LogConstant(
        code=code,
        level=level,
        message=message,
        component=component,
        subcomponent=subcomponent,
        tags=tuple(tags),
    )


# ---------------------------------------------------------------------------
# Controller constants (LOG401–LOG429)
# ---------------------------------------------------------------------------
LOG_SESSION_ADAPTER_LOAD_ERROR = _constant(
    "LOG401",
    "ERROR",
    "Session failed to load adapter",
    component="Controller",
    subcomponent="Session",
    tags=_tags("session", "adapter", "error"),
)

LOG_SESSION_STEP_ERROR = _constant(
    "LOG402",
    "ERROR",
    "Adapter step raised an exception",
    component="Controller",
    subcomponent="Session",
    tags=_tags("session", "adapter", "step"),
)

LOG_SESSION_EPISODE_ERROR = _constant(
    "LOG403",
    "ERROR",
    "Failed to finalise episode state",
    component="Controller",
    subcomponent="Session",
    tags=_tags("session", "episode"),
)

LOG_SESSION_TIMER_PRECISION_WARNING = _constant(
    "LOG404",
    "WARNING",
    "Session timer precision degraded",
    component="Controller",
    subcomponent="Session",
    tags=_tags("session", "timer"),
)

LOG_INPUT_CONTROLLER_ERROR = _constant(
    "LOG405",
    "ERROR",
    "Input controller encountered an error",
    component="Controller",
    subcomponent="Input",
    tags=_tags("input", "controller"),
)

LOG_LIVE_CONTROLLER_INITIALIZED = _constant(
    "LOG408",
    "INFO",
    "Live telemetry controller initialised",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "lifecycle"),
)

LOG_LIVE_CONTROLLER_THREAD_STARTED = _constant(
    "LOG409",
    "INFO",
    "Live telemetry worker thread started",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "thread"),
)

LOG_LIVE_CONTROLLER_THREAD_STOPPED = _constant(
    "LOG410",
    "INFO",
    "Live telemetry worker thread stopped",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "thread"),
)

LOG_LIVE_CONTROLLER_THREAD_STOP_TIMEOUT = _constant(
    "LOG411",
    "WARNING",
    "Timed out waiting for live telemetry thread to stop",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "thread", "timeout"),
)

LOG_LIVE_CONTROLLER_ALREADY_RUNNING = _constant(
    "LOG412",
    "WARNING",
    "Live telemetry controller already running",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "lifecycle"),
)

LOG_LIVE_CONTROLLER_RUN_SUBSCRIBED = _constant(
    "LOG413",
    "INFO",
    "Subscribed to run telemetry stream",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "subscription"),
)

LOG_LIVE_CONTROLLER_RUN_UNSUBSCRIBED = _constant(
    "LOG414",
    "INFO",
    "Unsubscribed from run telemetry stream",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "subscription"),
)

LOG_LIVE_CONTROLLER_RUN_ALREADY_SUBSCRIBED = _constant(
    "LOG415",
    "WARNING",
    "Run telemetry stream already subscribed",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "subscription"),
)

LOG_LIVE_CONTROLLER_RUNBUS_SUBSCRIBED = _constant(
    "LOG416",
    "INFO",
    "RunBus subscription established",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "runbus"),
)

LOG_LIVE_CONTROLLER_RUN_COMPLETED = _constant(
    "LOG417",
    "INFO",
    "Run telemetry stream completed",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "lifecycle"),
)

LOG_LIVE_CONTROLLER_QUEUE_OVERFLOW = _constant(
    "LOG418",
    "WARNING",
    "Live telemetry queue overflowed",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "queue", "overflow"),
)

LOG_LIVE_CONTROLLER_BUFFER_STEPS_FLUSHED = _constant(
    "LOG419",
    "INFO",
    "Flushed buffered step events to UI",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "buffer", "step"),
)

LOG_LIVE_CONTROLLER_BUFFER_EPISODES_FLUSHED = _constant(
    "LOG420",
    "INFO",
    "Flushed buffered episode events to UI",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "buffer", "episode"),
)

LOG_BUFFER_DROP = _constant(
    "LOG421",
    "WARNING",
    "Dropped buffered telemetry event due to limit",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "buffer", "drop"),
)

LOG_TELEMETRY_CONTROLLER_THREAD_ERROR = _constant(
    "LOG422",
    "ERROR",
    "Telemetry controller thread crashed",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "thread", "error"),
)

LOG_LIVE_CONTROLLER_SIGNAL_EMIT_FAILED = _constant(
    "LOG423",
    "ERROR",
    "Failed to emit telemetry update signal",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "qt", "signal"),
)

LOG_LIVE_CONTROLLER_TAB_ADD_FAILED = _constant(
    "LOG424",
    "ERROR",
    "Failed to add telemetry tab to UI",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "ui"),
)

LOG_CREDIT_STARVED = _constant(
    "LOG425",
    "INFO",
    "Telemetry credits exhausted for stream",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "credits", "backpressure"),
)

LOG_CREDIT_RESUMED = _constant(
    "LOG426",
    "INFO",
    "Telemetry credits replenished for stream",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "credits", "backpressure"),
)

LOG_LIVE_CONTROLLER_LOOP_EXITED = _constant(
    "LOG427",
    "INFO",
    "Telemetry controller loop exited",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "lifecycle"),
)

LOG_TELEMETRY_SUBSCRIBE_ERROR = _constant(
    "LOG428",
    "ERROR",
    "Failed to subscribe to telemetry stream",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "subscription", "error"),
)


# ---------------------------------------------------------------------------
# Adapter constants (LOG501–LOG513)
# ---------------------------------------------------------------------------
LOG_ADAPTER_INIT_ERROR = _constant(
    "LOG501",
    "ERROR",
    "Adapter initialisation failed",
    component="Adapter",
    subcomponent="Lifecycle",
    tags=_tags("adapter", "initialise"),
)

LOG_ADAPTER_STEP_ERROR = _constant(
    "LOG502",
    "ERROR",
    "Adapter step execution failed",
    component="Adapter",
    subcomponent="Step",
    tags=_tags("adapter", "step"),
)

LOG_ADAPTER_PAYLOAD_ERROR = _constant(
    "LOG503",
    "ERROR",
    "Adapter produced invalid payload",
    component="Adapter",
    subcomponent="Payload",
    tags=_tags("adapter", "payload"),
)

LOG_ADAPTER_STATE_INVALID = _constant(
    "LOG504",
    "ERROR",
    "Adapter emitted invalid state snapshot",
    component="Adapter",
    subcomponent="State",
    tags=_tags("adapter", "state"),
)

LOG_ADAPTER_RENDER_ERROR = _constant(
    "LOG505",
    "ERROR",
    "Adapter render call failed",
    component="Adapter",
    subcomponent="Render",
    tags=_tags("adapter", "render"),
)

LOG_ADAPTER_ENV_CREATED = _constant(
    "LOG510",
    "INFO",
    "Adapter environment created",
    component="Adapter",
    subcomponent="Lifecycle",
    tags=_tags("adapter", "environment"),
)

LOG_ADAPTER_ENV_RESET = _constant(
    "LOG511",
    "INFO",
    "Adapter environment reset",
    component="Adapter",
    subcomponent="Lifecycle",
    tags=_tags("adapter", "environment"),
)

LOG_ADAPTER_STEP_SUMMARY = _constant(
    "LOG512",
    "DEBUG",
    "Adapter step summary",
    component="Adapter",
    subcomponent="Step",
    tags=_tags("adapter", "step"),
)

LOG_ADAPTER_ENV_CLOSED = _constant(
    "LOG513",
    "INFO",
    "Adapter environment closed",
    component="Adapter",
    subcomponent="Lifecycle",
    tags=_tags("adapter", "environment"),
)


# ---------------------------------------------------------------------------
# Service and telemetry constants (LOG601–LOG650)
# ---------------------------------------------------------------------------
LOG_SERVICE_TELEMETRY_STEP_REJECTED = _constant(
    "LOG601",
    "WARNING",
    "Telemetry step rejected by validation",
    component="Service",
    subcomponent="Telemetry",
    tags=_tags("telemetry", "validation"),
)

LOG_SERVICE_TELEMETRY_ASYNC_ERROR = _constant(
    "LOG602",
    "ERROR",
    "Telemetry asynchronous persistence failed",
    component="Service",
    subcomponent="Telemetry",
    tags=_tags("telemetry", "async", "error"),
)

LOG_SERVICE_DB_SINK_INITIALIZED = _constant(
    "LOG610",
    "INFO",
    "Telemetry DB sink initialised",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "lifecycle"),
)

LOG_SERVICE_DB_SINK_STARTED = _constant(
    "LOG611",
    "INFO",
    "Telemetry DB sink worker started",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "worker"),
)

LOG_SERVICE_DB_SINK_ALREADY_RUNNING = _constant(
    "LOG612",
    "WARNING",
    "Telemetry DB sink already running",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "lifecycle"),
)

LOG_SERVICE_DB_SINK_STOPPED = _constant(
    "LOG613",
    "INFO",
    "Telemetry DB sink worker stopped",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "worker"),
)

LOG_SERVICE_DB_SINK_STOP_TIMEOUT = _constant(
    "LOG614",
    "WARNING",
    "Timed out waiting for telemetry DB sink to stop",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "timeout"),
)

LOG_SERVICE_DB_SINK_LOOP_EXITED = _constant(
    "LOG615",
    "INFO",
    "Telemetry DB sink loop exited",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "lifecycle"),
)

LOG_SERVICE_DB_SINK_FATAL = _constant(
    "LOG616",
    "ERROR",
    "Telemetry DB sink encountered fatal error",
    component="Service",
    subcomponent="DBSink",
    tags=_tags("telemetry", "db", "error"),
)

LOG_SERVICE_SQLITE_DEBUG = _constant(
    "LOG617",
    "DEBUG",
    "Telemetry SQLite debug event",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "debug"),
)

LOG_SERVICE_SQLITE_INFO = _constant(
    "LOG618",
    "INFO",
    "Telemetry SQLite info event",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite"),
)

LOG_SERVICE_SQLITE_WARNING = _constant(
    "LOG619",
    "WARNING",
    "Telemetry SQLite warning",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "warning"),
)

LOG_SERVICE_SQLITE_WORKER_STARTED = _constant(
    "LOG620",
    "INFO",
    "Telemetry SQLite worker started",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "worker"),
)

LOG_SERVICE_SQLITE_WORKER_STOPPED = _constant(
    "LOG621",
    "INFO",
    "Telemetry SQLite worker stopped",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "worker"),
)

LOG_SERVICE_SQLITE_WRITE_ERROR = _constant(
    "LOG622",
    "ERROR",
    "Telemetry SQLite write failed",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "error"),
)

LOG_SERVICE_SQLITE_DESERIALIZATION_FAILED = _constant(
    "LOG623",
    "ERROR",
    "Failed to deserialize telemetry record from queue",
    component="Service",
    subcomponent="SQLite",
    tags=_tags("telemetry", "sqlite", "error"),
)

# Telemetry Bridge (TelemetryBridge class) constants
LOG_SERVICE_TELEMETRY_BRIDGE_STEP_QUEUED = _constant(
    "LOG624",
    "DEBUG",
    "Telemetry step event queued for delivery",
    component="Service",
    subcomponent="TelemetryBridge",
    tags=_tags("telemetry", "bridge", "step"),
)

LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_QUEUED = _constant(
    "LOG625",
    "DEBUG",
    "Telemetry episode event queued for delivery",
    component="Service",
    subcomponent="TelemetryBridge",
    tags=_tags("telemetry", "bridge", "episode"),
)

LOG_SERVICE_TELEMETRY_BRIDGE_STEP_DELIVERED = _constant(
    "LOG626",
    "DEBUG",
    "Telemetry step event delivered from bridge",
    component="Service",
    subcomponent="TelemetryBridge",
    tags=_tags("telemetry", "bridge", "step"),
)

LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_DELIVERED = _constant(
    "LOG627",
    "DEBUG",
    "Telemetry episode event delivered from bridge",
    component="Service",
    subcomponent="TelemetryBridge",
    tags=_tags("telemetry", "bridge", "episode"),
)

LOG_SERVICE_TELEMETRY_BRIDGE_OVERFLOW = _constant(
    "LOG628",
    "WARNING",
    "Telemetry bridge queue overflow - events dropped",
    component="Service",
    subcomponent="TelemetryBridge",
    tags=_tags("telemetry", "bridge", "overflow"),
)

LOG_SERVICE_TELEMETRY_BRIDGE_RUN_COMPLETED = _constant(
    "LOG629",
    "INFO",
    "Training run completed - signaling via bridge",
    component="Service",
    subcomponent="TelemetryBridge",
    tags=_tags("telemetry", "bridge", "completion"),
)

# Telemetry Async Hub (TelemetryAsyncHub class) constants
LOG_SERVICE_TELEMETRY_HUB_STARTED = _constant(
    "LOG640",
    "INFO",
    "Telemetry async hub started",
    component="Service",
    subcomponent="TelemetryAsyncHub",
    tags=_tags("telemetry", "hub", "lifecycle"),
)

LOG_SERVICE_TELEMETRY_HUB_SUBSCRIBED = _constant(
    "LOG641",
    "INFO",
    "Subscribed to telemetry stream for run",
    component="Service",
    subcomponent="TelemetryAsyncHub",
    tags=_tags("telemetry", "hub", "subscription"),
)

LOG_SERVICE_TELEMETRY_HUB_TRACE = _constant(
    "LOG642",
    "DEBUG",
    "Telemetry hub operation trace",
    component="Service",
    subcomponent="TelemetryAsyncHub",
    tags=_tags("telemetry", "hub", "trace"),
)

LOG_SERVICE_TELEMETRY_HUB_ERROR = _constant(
    "LOG643",
    "ERROR",
    "Telemetry hub error",
    component="Service",
    subcomponent="TelemetryAsyncHub",
    tags=_tags("telemetry", "hub", "error"),
)

LOG_RENDER_REGULATOR_NOT_STARTED = _constant(
    "LOG630",
    "WARNING",
    "Rendering regulator not started",
    component="Service",
    subcomponent="Rendering",
    tags=_tags("telemetry", "render", "regulator"),
)

LOG_RENDER_DROPPED_FRAME = _constant(
    "LOG631",
    "INFO",
    "Dropped render frame due to throttling",
    component="Service",
    subcomponent="Rendering",
    tags=_tags("telemetry", "render", "throttle"),
)

LOG_DAEMON_START = _constant(
    "LOG650",
    "INFO",
    "Trainer daemon starting",
    component="Service",
    subcomponent="TrainerDaemon",
    tags=_tags("daemon", "lifecycle"),
)

LOG_TRAINER_CLIENT_CONNECTING = _constant(
    "LOG651",
    "INFO",
    "Connecting to trainer daemon",
    component="Service",
    subcomponent="TrainerClient",
    tags=_tags("trainer", "client", "connection", "lifecycle"),
)

LOG_TRAINER_CLIENT_CONNECTED = _constant(
    "LOG652",
    "INFO",
    "Trainer daemon connection established",
    component="Service",
    subcomponent="TrainerClient",
    tags=_tags("trainer", "client", "connection", "lifecycle"),
)

LOG_TRAINER_CLIENT_CONNECTION_TIMEOUT = _constant(
    "LOG653",
    "ERROR",
    "Trainer daemon connection timeout",
    component="Service",
    subcomponent="TrainerClient",
    tags=_tags("trainer", "client", "connection", "timeout", "error"),
)

LOG_TRAINER_CLIENT_LOOP_NONFATAL = _constant(
    "LOG654",
    "DEBUG",
    "Trainer client loop non-fatal condition ignored",
    component="Service",
    subcomponent="TrainerClient",
    tags=_tags("trainer", "client", "loop", "debug"),
)

LOG_TRAINER_CLIENT_LOOP_ERROR = _constant(
    "LOG655",
    "ERROR",
    "Trainer client loop error",
    component="Service",
    subcomponent="TrainerClient",
    tags=_tags("trainer", "client", "loop", "error"),
)

LOG_SERVICE_FRAME_INFO = _constant(
    "LOG660",
    "INFO",
    "Frame storage event",
    component="Service",
    subcomponent="FrameStorage",
    tags=_tags("service", "frame", "storage"),
)

LOG_SERVICE_FRAME_WARNING = _constant(
    "LOG661",
    "WARNING",
    "Frame storage warning",
    component="Service",
    subcomponent="FrameStorage",
    tags=_tags("service", "frame", "storage", "warning"),
)

LOG_SERVICE_FRAME_ERROR = _constant(
    "LOG662",
    "ERROR",
    "Frame storage error",
    component="Service",
    subcomponent="FrameStorage",
    tags=_tags("service", "frame", "storage", "error"),
)

LOG_SERVICE_FRAME_DEBUG = _constant(
    "LOG663",
    "DEBUG",
    "Frame storage debug event",
    component="Service",
    subcomponent="FrameStorage",
    tags=_tags("service", "frame", "storage", "debug"),
)

LOG_SERVICE_VALIDATION_DEBUG = _constant(
    "LOG664",
    "DEBUG",
    "Validation service debug",
    component="Service",
    subcomponent="Validation",
    tags=_tags("service", "validation", "debug"),
)

LOG_SERVICE_VALIDATION_WARNING = _constant(
    "LOG665",
    "WARNING",
    "Validation service warning",
    component="Service",
    subcomponent="Validation",
    tags=_tags("service", "validation", "warning"),
)

LOG_SERVICE_VALIDATION_ERROR = _constant(
    "LOG666",
    "ERROR",
    "Validation service error",
    component="Service",
    subcomponent="Validation",
    tags=_tags("service", "validation", "error"),
)

LOG_SERVICE_ACTOR_SEED_ERROR = _constant(
    "LOG667",
    "ERROR",
    "Actor seeding failed",
    component="Service",
    subcomponent="Actor",
    tags=_tags("service", "actor", "seeding", "error"),
)

# ---------------------------------------------------------------------------
# Runtime/application constants (LOG680–LOG683)
# ---------------------------------------------------------------------------
LOG_RUNTIME_APP_DEBUG = _constant(
    "LOG680",
    "DEBUG",
    "Application runtime debug",
    component="Runtime",
    subcomponent="App",
    tags=_tags("runtime", "app", "debug"),
)

LOG_RUNTIME_APP_INFO = _constant(
    "LOG681",
    "INFO",
    "Application runtime event",
    component="Runtime",
    subcomponent="App",
    tags=_tags("runtime", "app"),
)

LOG_RUNTIME_APP_WARNING = _constant(
    "LOG682",
    "WARNING",
    "Application runtime warning",
    component="Runtime",
    subcomponent="App",
    tags=_tags("runtime", "app", "warning"),
)

LOG_RUNTIME_APP_ERROR = _constant(
    "LOG683",
    "ERROR",
    "Application runtime error",
    component="Runtime",
    subcomponent="App",
    tags=_tags("runtime", "app", "error"),
)


# ---------------------------------------------------------------------------
# UI constants (LOG701–LOG739)
# ---------------------------------------------------------------------------
LOG_UI_MAINWINDOW_TRACE = _constant(
    "LOG701",
    "DEBUG",
    "UI main window trace",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "mainwindow", "trace"),
)

LOG_UI_MAINWINDOW_INFO = _constant(
    "LOG702",
    "INFO",
    "UI main window event",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "mainwindow"),
)

LOG_UI_MAINWINDOW_WARNING = _constant(
    "LOG703",
    "WARNING",
    "UI main window warning",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "mainwindow", "warning"),
)

LOG_UI_MAINWINDOW_ERROR = _constant(
    "LOG704",
    "ERROR",
    "UI main window error",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "mainwindow", "error"),
)

LOG_UI_LIVE_TAB_TRACE = _constant(
    "LOG710",
    "DEBUG",
    "Live telemetry tab trace",
    component="UI",
    subcomponent="LiveTelemetryTab",
    tags=_tags("ui", "telemetry", "trace"),
)

LOG_UI_LIVE_TAB_INFO = _constant(
    "LOG711",
    "INFO",
    "Live telemetry tab event",
    component="UI",
    subcomponent="LiveTelemetryTab",
    tags=_tags("ui", "telemetry"),
)

LOG_UI_LIVE_TAB_WARNING = _constant(
    "LOG712",
    "WARNING",
    "Live telemetry tab warning",
    component="UI",
    subcomponent="LiveTelemetryTab",
    tags=_tags("ui", "telemetry", "warning"),
)

LOG_UI_LIVE_TAB_ERROR = _constant(
    "LOG713",
    "ERROR",
    "Live telemetry tab error",
    component="UI",
    subcomponent="LiveTelemetryTab",
    tags=_tags("ui", "telemetry", "error"),
)

LOG_UI_RENDER_TABS_TRACE = _constant(
    "LOG720",
    "DEBUG",
    "Render tabs trace",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "trace"),
)

LOG_UI_RENDER_TABS_INFO = _constant(
    "LOG721",
    "INFO",
    "Render tabs event",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render"),
)

LOG_UI_RENDER_TABS_WARNING = _constant(
    "LOG722",
    "WARNING",
    "Render tabs warning",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "warning"),
)

LOG_UI_RENDER_TABS_ERROR = _constant(
    "LOG723",
    "ERROR",
    "Render tabs error",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "error"),
)

LOG_UI_TRAIN_FORM_TRACE = _constant(
    "LOG730",
    "DEBUG",
    "Train form trace",
    component="UI",
    subcomponent="TrainForm",
    tags=_tags("ui", "train_form", "trace"),
)

LOG_UI_TRAIN_FORM_INFO = _constant(
    "LOG731",
    "INFO",
    "Train form event",
    component="UI",
    subcomponent="TrainForm",
    tags=_tags("ui", "train_form"),
)

LOG_UI_TRAIN_FORM_WARNING = _constant(
    "LOG732",
    "WARNING",
    "Train form warning",
    component="UI",
    subcomponent="TrainForm",
    tags=_tags("ui", "train_form", "warning"),
)

LOG_UI_TRAIN_FORM_ERROR = _constant(
    "LOG733",
    "ERROR",
    "Train form error",
    component="UI",
    subcomponent="TrainForm",
    tags=_tags("ui", "train_form", "error"),
)

LOG_UI_WORKER_TABS_TRACE = _constant(
    "LOG740",
    "DEBUG",
    "Worker tabs trace",
    component="UI",
    subcomponent="WorkerTabs",
    tags=_tags("ui", "worker", "trace"),
)

LOG_UI_WORKER_TABS_INFO = _constant(
    "LOG741",
    "INFO",
    "Worker tabs event",
    component="UI",
    subcomponent="WorkerTabs",
    tags=_tags("ui", "worker"),
)

LOG_UI_WORKER_TABS_WARNING = _constant(
    "LOG742",
    "WARNING",
    "Worker tabs warning",
    component="UI",
    subcomponent="WorkerTabs",
    tags=_tags("ui", "worker", "warning"),
)

LOG_UI_WORKER_TABS_ERROR = _constant(
    "LOG743",
    "ERROR",
    "Worker tabs error",
    component="UI",
    subcomponent="WorkerTabs",
    tags=_tags("ui", "worker", "error"),
)


# ---------------------------------------------------------------------------
# Utility constants (LOG801–LOG809)
# ---------------------------------------------------------------------------
LOG_UTIL_QT_RESEED_SKIPPED = _constant(
    "LOG801",
    "DEBUG",
    "Qt random reseed skipped",
    component="Utility",
    subcomponent="Seeding",
    tags=_tags("utility", "seeding", "qt"),
)

LOG_UTIL_QT_STATE_CAPTURE_FAILED = _constant(
    "LOG802",
    "ERROR",
    "Failed to capture Qt random generator state",
    component="Utility",
    subcomponent="Seeding",
    tags=_tags("utility", "seeding", "qt", "error"),
)

LOG_UTIL_SEED_CALLBACK_FAILED = _constant(
    "LOG803",
    "ERROR",
    "Seed callback failed",
    component="Utility",
    subcomponent="Seeding",
    tags=_tags("utility", "seeding", "callback", "error"),
)


# ---------------------------------------------------------------------------
# Worker constants (LOG901–LOG915)
# ---------------------------------------------------------------------------
LOG_WORKER_RUNTIME_EVENT = _constant(
    "LOG901",
    "INFO",
    "Worker runtime event",
    component="Worker",
    subcomponent="Runtime",
    tags=_tags("worker", "runtime"),
)

LOG_WORKER_RUNTIME_WARNING = _constant(
    "LOG902",
    "WARNING",
    "Worker runtime warning",
    component="Worker",
    subcomponent="Runtime",
    tags=_tags("worker", "runtime", "warning"),
)

LOG_WORKER_RUNTIME_ERROR = _constant(
    "LOG903",
    "ERROR",
    "Worker runtime error",
    component="Worker",
    subcomponent="Runtime",
    tags=_tags("worker", "runtime", "error"),
)

LOG_WORKER_RUNTIME_DEBUG = _constant(
    "LOG904",
    "DEBUG",
    "Worker runtime debug",
    component="Worker",
    subcomponent="Runtime",
    tags=_tags("worker", "runtime", "debug"),
)

LOG_WORKER_CONFIG_EVENT = _constant(
    "LOG905",
    "INFO",
    "Worker configuration event",
    component="Worker",
    subcomponent="Config",
    tags=_tags("worker", "config"),
)

LOG_WORKER_CONFIG_WARNING = _constant(
    "LOG906",
    "WARNING",
    "Worker configuration warning",
    component="Worker",
    subcomponent="Config",
    tags=_tags("worker", "config", "warning"),
)

LOG_WORKER_POLICY_EVENT = _constant(
    "LOG907",
    "INFO",
    "Worker policy event",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy"),
)

LOG_WORKER_POLICY_WARNING = _constant(
    "LOG908",
    "WARNING",
    "Worker policy warning",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy", "warning"),
)

LOG_WORKER_POLICY_ERROR = _constant(
    "LOG909",
    "ERROR",
    "Worker policy error",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy", "error"),
)

LOG_WORKER_BDI_EVENT = _constant(
    "LOG910",
    "INFO",
    "Worker BDI event",
    component="Worker",
    subcomponent="BDI",
    tags=_tags("worker", "bdi"),
)

LOG_WORKER_BDI_WARNING = _constant(
    "LOG911",
    "WARNING",
    "Worker BDI warning",
    component="Worker",
    subcomponent="BDI",
    tags=_tags("worker", "bdi", "warning"),
)

LOG_WORKER_BDI_ERROR = _constant(
    "LOG912",
    "ERROR",
    "Worker BDI error",
    component="Worker",
    subcomponent="BDI",
    tags=_tags("worker", "bdi", "error"),
)

LOG_WORKER_BDI_DEBUG = _constant(
    "LOG913",
    "DEBUG",
    "Worker BDI debug event",
    component="Worker",
    subcomponent="BDI",
    tags=_tags("worker", "bdi", "debug"),
)


# =========================================================================
# Helper Functions for Runtime Discovery & Validation
# =========================================================================


def get_constant_by_code(code: str) -> LogConstant | None:
    """Retrieve a log constant by its code identifier.

    Args:
        code: The constant code (e.g., "LOG401", "LOG502").

    Returns:
        The LogConstant instance if found, else None.
    """
    for const in ALL_LOG_CONSTANTS:
        if const.code == code:
            return const
    return None


def list_known_components() -> list[str]:
    """Return a sorted list of all known component names.

    Returns:
        Sorted list of unique component strings (e.g., ["Adapter", "Controller", "Service"]).
    """
    return sorted(set(const.component for const in ALL_LOG_CONSTANTS))


def get_component_snapshot() -> Dict[str, Set[str]]:
    """Build a snapshot of component → subcomponents mapping.

    This is useful for GUI filters to display hierarchical component/subcomponent
    choices without hard-coding them.

    Returns:
        A dict mapping each component name to the set of its known subcomponents.

    Example:
        >>> snapshot = get_component_snapshot()
        >>> snapshot["Controller"]
        {'Input', 'LiveTelemetry', 'Session'}
    """
    snapshot: Dict[str, Set[str]] = {}
    for const in ALL_LOG_CONSTANTS:
        if const.component not in snapshot:
            snapshot[const.component] = set()
        snapshot[const.component].add(const.subcomponent)
    return snapshot


def validate_log_constants() -> list[str]:
    """Validate that all log constants conform to logging standards.

    Checks that:
    - Every constant has a valid logging level (matches logging._nameToLevel).
    - Every constant has non-empty code, message, component, subcomponent fields.
    - No duplicate codes exist.

    Returns:
        A list of error messages. Empty list if all constants are valid.
    """
    errors: list[str] = []
    seen_codes: set[str] = set()

    for const in ALL_LOG_CONSTANTS:
        # Check non-empty fields
        if not const.code:
            errors.append(f"LogConstant has empty code: {const}")
        if not const.message:
            errors.append(f"LogConstant {const.code} has empty message")
        if not const.component:
            errors.append(f"LogConstant {const.code} has empty component")
        if not const.subcomponent:
            errors.append(f"LogConstant {const.code} has empty subcomponent")

        # Check level validity
        if isinstance(const.level, str):
            if const.level not in logging._nameToLevel:
                errors.append(
                    f"LogConstant {const.code} has invalid level: {const.level}. "
                    f"Must be one of {sorted(logging._nameToLevel.keys())}"
                )
        elif not isinstance(const.level, int):
            errors.append(
                f"LogConstant {const.code} has invalid level type: {type(const.level)}. "
                f"Must be str or int."
            )

        # Check for duplicate codes
        if const.code in seen_codes:
            errors.append(f"Duplicate constant code: {const.code}")
        seen_codes.add(const.code)

    return errors


# Aggregate tuple for quick iteration during tests or tooling.
ALL_LOG_CONSTANTS: Tuple[LogConstant, ...] = (
    LOG_SESSION_ADAPTER_LOAD_ERROR,
    LOG_SESSION_STEP_ERROR,
    LOG_SESSION_EPISODE_ERROR,
    LOG_SESSION_TIMER_PRECISION_WARNING,
    LOG_INPUT_CONTROLLER_ERROR,
    LOG_LIVE_CONTROLLER_INITIALIZED,
    LOG_LIVE_CONTROLLER_THREAD_STARTED,
    LOG_LIVE_CONTROLLER_THREAD_STOPPED,
    LOG_LIVE_CONTROLLER_THREAD_STOP_TIMEOUT,
    LOG_LIVE_CONTROLLER_ALREADY_RUNNING,
    LOG_LIVE_CONTROLLER_RUN_SUBSCRIBED,
    LOG_LIVE_CONTROLLER_RUN_UNSUBSCRIBED,
    LOG_LIVE_CONTROLLER_RUN_ALREADY_SUBSCRIBED,
    LOG_LIVE_CONTROLLER_RUNBUS_SUBSCRIBED,
    LOG_LIVE_CONTROLLER_RUN_COMPLETED,
    LOG_LIVE_CONTROLLER_QUEUE_OVERFLOW,
    LOG_LIVE_CONTROLLER_BUFFER_STEPS_FLUSHED,
    LOG_LIVE_CONTROLLER_BUFFER_EPISODES_FLUSHED,
    LOG_BUFFER_DROP,
    LOG_TELEMETRY_CONTROLLER_THREAD_ERROR,
    LOG_LIVE_CONTROLLER_SIGNAL_EMIT_FAILED,
    LOG_LIVE_CONTROLLER_TAB_ADD_FAILED,
    LOG_CREDIT_STARVED,
    LOG_CREDIT_RESUMED,
    LOG_LIVE_CONTROLLER_LOOP_EXITED,
    LOG_TELEMETRY_SUBSCRIBE_ERROR,
    LOG_ADAPTER_INIT_ERROR,
    LOG_ADAPTER_STEP_ERROR,
    LOG_ADAPTER_PAYLOAD_ERROR,
    LOG_ADAPTER_STATE_INVALID,
    LOG_ADAPTER_RENDER_ERROR,
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
    LOG_ADAPTER_ENV_CLOSED,
    LOG_SERVICE_TELEMETRY_STEP_REJECTED,
    LOG_SERVICE_TELEMETRY_ASYNC_ERROR,
    LOG_SERVICE_DB_SINK_INITIALIZED,
    LOG_SERVICE_DB_SINK_STARTED,
    LOG_SERVICE_DB_SINK_ALREADY_RUNNING,
    LOG_SERVICE_DB_SINK_STOPPED,
    LOG_SERVICE_DB_SINK_STOP_TIMEOUT,
    LOG_SERVICE_DB_SINK_LOOP_EXITED,
    LOG_SERVICE_DB_SINK_FATAL,
    LOG_SERVICE_SQLITE_DEBUG,
    LOG_SERVICE_SQLITE_INFO,
    LOG_SERVICE_SQLITE_WARNING,
    LOG_SERVICE_SQLITE_WORKER_STARTED,
    LOG_SERVICE_SQLITE_WORKER_STOPPED,
    LOG_SERVICE_SQLITE_WRITE_ERROR,
    LOG_SERVICE_SQLITE_DESERIALIZATION_FAILED,
    LOG_SERVICE_TELEMETRY_BRIDGE_STEP_QUEUED,
    LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_QUEUED,
    LOG_SERVICE_TELEMETRY_BRIDGE_STEP_DELIVERED,
    LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_DELIVERED,
    LOG_SERVICE_TELEMETRY_BRIDGE_OVERFLOW,
    LOG_SERVICE_TELEMETRY_BRIDGE_RUN_COMPLETED,
    LOG_SERVICE_TELEMETRY_HUB_STARTED,
    LOG_SERVICE_TELEMETRY_HUB_SUBSCRIBED,
    LOG_SERVICE_TELEMETRY_HUB_TRACE,
    LOG_SERVICE_TELEMETRY_HUB_ERROR,
    LOG_RENDER_REGULATOR_NOT_STARTED,
    LOG_RENDER_DROPPED_FRAME,
    LOG_DAEMON_START,
    LOG_TRAINER_CLIENT_CONNECTING,
    LOG_TRAINER_CLIENT_CONNECTED,
    LOG_TRAINER_CLIENT_CONNECTION_TIMEOUT,
    LOG_TRAINER_CLIENT_LOOP_NONFATAL,
    LOG_TRAINER_CLIENT_LOOP_ERROR,
    LOG_SERVICE_FRAME_INFO,
    LOG_SERVICE_FRAME_WARNING,
    LOG_SERVICE_FRAME_ERROR,
    LOG_SERVICE_FRAME_DEBUG,
    LOG_SERVICE_VALIDATION_DEBUG,
    LOG_SERVICE_VALIDATION_WARNING,
    LOG_SERVICE_VALIDATION_ERROR,
    LOG_SERVICE_ACTOR_SEED_ERROR,
    LOG_RUNTIME_APP_DEBUG,
    LOG_RUNTIME_APP_INFO,
    LOG_RUNTIME_APP_WARNING,
    LOG_RUNTIME_APP_ERROR,
    LOG_UI_MAINWINDOW_TRACE,
    LOG_UI_MAINWINDOW_INFO,
    LOG_UI_MAINWINDOW_WARNING,
    LOG_UI_MAINWINDOW_ERROR,
    LOG_UI_LIVE_TAB_TRACE,
    LOG_UI_LIVE_TAB_INFO,
    LOG_UI_LIVE_TAB_WARNING,
    LOG_UI_LIVE_TAB_ERROR,
    LOG_UI_RENDER_TABS_TRACE,
    LOG_UI_RENDER_TABS_INFO,
    LOG_UI_RENDER_TABS_WARNING,
    LOG_UI_RENDER_TABS_ERROR,
    LOG_UI_TRAIN_FORM_TRACE,
    LOG_UI_TRAIN_FORM_INFO,
    LOG_UI_TRAIN_FORM_WARNING,
    LOG_UI_TRAIN_FORM_ERROR,
    LOG_UI_WORKER_TABS_TRACE,
    LOG_UI_WORKER_TABS_INFO,
    LOG_UI_WORKER_TABS_WARNING,
    LOG_UI_WORKER_TABS_ERROR,
    LOG_UTIL_QT_RESEED_SKIPPED,
    LOG_UTIL_QT_STATE_CAPTURE_FAILED,
    LOG_UTIL_SEED_CALLBACK_FAILED,
    LOG_WORKER_RUNTIME_EVENT,
    LOG_WORKER_RUNTIME_DEBUG,
    LOG_WORKER_RUNTIME_WARNING,
    LOG_WORKER_RUNTIME_ERROR,
    LOG_WORKER_CONFIG_EVENT,
    LOG_WORKER_CONFIG_WARNING,
    LOG_WORKER_POLICY_EVENT,
    LOG_WORKER_POLICY_WARNING,
    LOG_WORKER_POLICY_ERROR,
    LOG_WORKER_BDI_EVENT,
    LOG_WORKER_BDI_WARNING,
    LOG_WORKER_BDI_ERROR,
    LOG_WORKER_BDI_DEBUG,
)


__all__ = (
    "LogConstant",
    "ALL_LOG_CONSTANTS",
    # Helper functions for runtime discovery
    "get_constant_by_code",
    "list_known_components",
    "get_component_snapshot",
    "validate_log_constants",
    # Constants
    "LOG_SESSION_ADAPTER_LOAD_ERROR",
    "LOG_SESSION_STEP_ERROR",
    "LOG_SESSION_EPISODE_ERROR",
    "LOG_SESSION_TIMER_PRECISION_WARNING",
    "LOG_INPUT_CONTROLLER_ERROR",
    "LOG_LIVE_CONTROLLER_INITIALIZED",
    "LOG_LIVE_CONTROLLER_THREAD_STARTED",
    "LOG_LIVE_CONTROLLER_THREAD_STOPPED",
    "LOG_LIVE_CONTROLLER_THREAD_STOP_TIMEOUT",
    "LOG_LIVE_CONTROLLER_ALREADY_RUNNING",
    "LOG_LIVE_CONTROLLER_RUN_SUBSCRIBED",
    "LOG_LIVE_CONTROLLER_RUN_UNSUBSCRIBED",
    "LOG_LIVE_CONTROLLER_RUN_ALREADY_SUBSCRIBED",
    "LOG_LIVE_CONTROLLER_RUNBUS_SUBSCRIBED",
    "LOG_LIVE_CONTROLLER_RUN_COMPLETED",
    "LOG_LIVE_CONTROLLER_QUEUE_OVERFLOW",
    "LOG_LIVE_CONTROLLER_BUFFER_STEPS_FLUSHED",
    "LOG_LIVE_CONTROLLER_BUFFER_EPISODES_FLUSHED",
    "LOG_BUFFER_DROP",
    "LOG_TELEMETRY_CONTROLLER_THREAD_ERROR",
    "LOG_LIVE_CONTROLLER_SIGNAL_EMIT_FAILED",
    "LOG_LIVE_CONTROLLER_TAB_ADD_FAILED",
    "LOG_CREDIT_STARVED",
    "LOG_CREDIT_RESUMED",
    "LOG_LIVE_CONTROLLER_LOOP_EXITED",
    "LOG_TELEMETRY_SUBSCRIBE_ERROR",
    "LOG_ADAPTER_INIT_ERROR",
    "LOG_ADAPTER_STEP_ERROR",
    "LOG_ADAPTER_PAYLOAD_ERROR",
    "LOG_ADAPTER_STATE_INVALID",
    "LOG_ADAPTER_RENDER_ERROR",
    "LOG_ADAPTER_ENV_CREATED",
    "LOG_ADAPTER_ENV_RESET",
    "LOG_ADAPTER_STEP_SUMMARY",
    "LOG_ADAPTER_ENV_CLOSED",
    "LOG_SERVICE_TELEMETRY_STEP_REJECTED",
    "LOG_SERVICE_TELEMETRY_ASYNC_ERROR",
    "LOG_SERVICE_DB_SINK_INITIALIZED",
    "LOG_SERVICE_DB_SINK_STARTED",
    "LOG_SERVICE_DB_SINK_ALREADY_RUNNING",
    "LOG_SERVICE_DB_SINK_STOPPED",
    "LOG_SERVICE_DB_SINK_STOP_TIMEOUT",
    "LOG_SERVICE_DB_SINK_LOOP_EXITED",
    "LOG_SERVICE_DB_SINK_FATAL",
    "LOG_SERVICE_SQLITE_DEBUG",
    "LOG_SERVICE_SQLITE_INFO",
    "LOG_SERVICE_SQLITE_WARNING",
    "LOG_SERVICE_SQLITE_WORKER_STARTED",
    "LOG_SERVICE_SQLITE_WORKER_STOPPED",
    "LOG_SERVICE_SQLITE_WRITE_ERROR",
    "LOG_SERVICE_SQLITE_DESERIALIZATION_FAILED",
    "LOG_SERVICE_TELEMETRY_BRIDGE_STEP_QUEUED",
    "LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_QUEUED",
    "LOG_SERVICE_TELEMETRY_BRIDGE_STEP_DELIVERED",
    "LOG_SERVICE_TELEMETRY_BRIDGE_EPISODE_DELIVERED",
    "LOG_SERVICE_TELEMETRY_BRIDGE_OVERFLOW",
    "LOG_SERVICE_TELEMETRY_BRIDGE_RUN_COMPLETED",
    "LOG_SERVICE_TELEMETRY_HUB_STARTED",
    "LOG_SERVICE_TELEMETRY_HUB_SUBSCRIBED",
    "LOG_SERVICE_TELEMETRY_HUB_TRACE",
    "LOG_SERVICE_TELEMETRY_HUB_ERROR",
    "LOG_RENDER_REGULATOR_NOT_STARTED",
    "LOG_RENDER_DROPPED_FRAME",
    "LOG_DAEMON_START",
    "LOG_TRAINER_CLIENT_CONNECTING",
    "LOG_TRAINER_CLIENT_CONNECTED",
    "LOG_TRAINER_CLIENT_CONNECTION_TIMEOUT",
    "LOG_TRAINER_CLIENT_LOOP_NONFATAL",
    "LOG_TRAINER_CLIENT_LOOP_ERROR",
    "LOG_SERVICE_FRAME_INFO",
    "LOG_SERVICE_FRAME_WARNING",
    "LOG_SERVICE_FRAME_ERROR",
    "LOG_SERVICE_FRAME_DEBUG",
    "LOG_SERVICE_VALIDATION_DEBUG",
    "LOG_SERVICE_VALIDATION_WARNING",
    "LOG_SERVICE_VALIDATION_ERROR",
    "LOG_SERVICE_ACTOR_SEED_ERROR",
    "LOG_RUNTIME_APP_DEBUG",
    "LOG_RUNTIME_APP_INFO",
    "LOG_RUNTIME_APP_WARNING",
    "LOG_RUNTIME_APP_ERROR",
    "LOG_UI_MAINWINDOW_TRACE",
    "LOG_UI_MAINWINDOW_INFO",
    "LOG_UI_MAINWINDOW_WARNING",
    "LOG_UI_MAINWINDOW_ERROR",
    "LOG_UI_LIVE_TAB_TRACE",
    "LOG_UI_LIVE_TAB_INFO",
    "LOG_UI_LIVE_TAB_WARNING",
    "LOG_UI_LIVE_TAB_ERROR",
    "LOG_UI_RENDER_TABS_TRACE",
    "LOG_UI_RENDER_TABS_INFO",
    "LOG_UI_RENDER_TABS_WARNING",
    "LOG_UI_RENDER_TABS_ERROR",
    "LOG_UI_TRAIN_FORM_TRACE",
    "LOG_UI_TRAIN_FORM_INFO",
    "LOG_UI_TRAIN_FORM_WARNING",
    "LOG_UI_TRAIN_FORM_ERROR",
    "LOG_UI_WORKER_TABS_TRACE",
    "LOG_UI_WORKER_TABS_INFO",
    "LOG_UI_WORKER_TABS_WARNING",
    "LOG_UI_WORKER_TABS_ERROR",
    "LOG_UTIL_QT_RESEED_SKIPPED",
    "LOG_UTIL_QT_STATE_CAPTURE_FAILED",
    "LOG_UTIL_SEED_CALLBACK_FAILED",
    "LOG_WORKER_RUNTIME_EVENT",
    "LOG_WORKER_RUNTIME_DEBUG",
    "LOG_WORKER_RUNTIME_WARNING",
    "LOG_WORKER_RUNTIME_ERROR",
    "LOG_WORKER_CONFIG_EVENT",
    "LOG_WORKER_CONFIG_WARNING",
    "LOG_WORKER_POLICY_EVENT",
    "LOG_WORKER_POLICY_WARNING",
    "LOG_WORKER_POLICY_ERROR",
    "LOG_WORKER_BDI_EVENT",
    "LOG_WORKER_BDI_WARNING",
    "LOG_WORKER_BDI_ERROR",
    "LOG_WORKER_BDI_DEBUG",
)
