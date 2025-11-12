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

LOG_LIVE_CONTROLLER_EVENT_FOR_DELETED_RUN = _constant(
    "LOG429",
    "WARNING",
    "Received telemetry for run marked deleted",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "deleted", "diagnostics"),
)

LOG_LIVE_CONTROLLER_TAB_REQUESTED = _constant(
    "LOG430",
    "INFO",
    "Requested creation of live telemetry tab",
    component="Controller",
    subcomponent="LiveTelemetry",
    tags=_tags("telemetry", "tab", "request"),
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

LOG_ADAPTER_MAP_GENERATION = _constant(
    "LOG514",
    "INFO",
    "Adapter map generation details",
    component="Adapter",
    subcomponent="MapGeneration",
    tags=_tags("adapter", "map", "generation"),
)

LOG_ADAPTER_HOLE_PLACEMENT = _constant(
    "LOG515",
    "DEBUG",
    "Hole placement configuration",
    component="Adapter",
    subcomponent="MapGeneration",
    tags=_tags("adapter", "map", "holes"),
)

LOG_ADAPTER_GOAL_OVERRIDE = _constant(
    "LOG516",
    "INFO",
    "Goal position override applied",
    component="Adapter",
    subcomponent="MapGeneration",
    tags=_tags("adapter", "map", "goal"),
)

LOG_ADAPTER_RENDER_PAYLOAD = _constant(
    "LOG517",
    "DEBUG",
    "Adapter render payload generated",
    component="Adapter",
    subcomponent="Render",
    tags=_tags("adapter", "render", "payload"),
)


LOG_ENV_MINIGRID_BOOT = _constant(
    "LOG518",
    "INFO",
    "MiniGrid adapter bootstrapped",
    component="Adapter",
    subcomponent="MiniGrid",
    tags=_tags("minigrid", "environment", "boot"),
)

LOG_ENV_MINIGRID_STEP = _constant(
    "LOG519",
    "DEBUG",
    "MiniGrid step checkpoint",
    component="Adapter",
    subcomponent="MiniGrid",
    tags=_tags("minigrid", "step"),
)

LOG_ENV_MINIGRID_ERROR = _constant(
    "LOG520",
    "ERROR",
    "MiniGrid adapter failure",
    component="Adapter",
    subcomponent="MiniGrid",
    tags=_tags("minigrid", "error"),
)

LOG_ENV_MINIGRID_RENDER_WARNING = _constant(
    "LOG521",
    "WARNING",
    "MiniGrid render payload unavailable",
    component="Adapter",
    subcomponent="MiniGrid",
    tags=_tags("minigrid", "render", "warning"),
)


# ALE adapter specific constants
LOG_ADAPTER_ALE_NAMESPACE_IMPORT_FAILED = _constant(
    "LOG522",
    "WARNING",
    "ALE optional namespace import failed; environments may not be auto-registered",
    component="Adapter",
    subcomponent="ALE",
    tags=_tags("ale", "import", "namespace", "warning"),
)

LOG_ADAPTER_ALE_METADATA_PROBE_FAILED = _constant(
    "LOG523",
    "DEBUG",
    "ALE metadata probe failed; continuing without optional metrics",
    component="Adapter",
    subcomponent="ALE",
    tags=_tags("ale", "metadata", "probe"),
)


# ---------------------------------------------------------------------------
# Service and telemetry constants (LOG601–LOG650)
# ---------------------------------------------------------------------------
LOG_SERVICE_SUPERVISOR_EVENT = _constant(
    "LOG680A",
    "INFO",
    "Supervisor event",
    component="Service",
    subcomponent="Supervisor",
    tags=_tags("supervisor", "event"),
)

LOG_SERVICE_SUPERVISOR_WARNING = _constant(
    "LOG680B",
    "WARNING",
    "Supervisor warning",
    component="Service",
    subcomponent="Supervisor",
    tags=_tags("supervisor", "warning"),
)

LOG_SERVICE_SUPERVISOR_ERROR = _constant(
    "LOG680C",
    "ERROR",
    "Supervisor error",
    component="Service",
    subcomponent="Supervisor",
    tags=_tags("supervisor", "error"),
)

LOG_SERVICE_SUPERVISOR_CONTROL_APPLIED = _constant(
    "LOG680D",
    "INFO",
    "Supervisor applied control update",
    component="Service",
    subcomponent="Supervisor",
    tags=_tags("supervisor", "control", "applied"),
)

LOG_SERVICE_SUPERVISOR_CONTROL_REJECTED = _constant(
    "LOG680E",
    "WARNING",
    "Supervisor control update rejected",
    component="Service",
    subcomponent="Supervisor",
    tags=_tags("supervisor", "control", "rejected"),
)

LOG_SERVICE_SUPERVISOR_ROLLBACK = _constant(
    "LOG680F",
    "INFO",
    "Supervisor rollback executed",
    component="Service",
    subcomponent="Supervisor",
    tags=_tags("supervisor", "rollback"),
)

LOG_SERVICE_SUPERVISOR_SAFETY_STATE = _constant(
    "LOG680G",
    "INFO",
    "Supervisor safety state changed",
    component="Service",
    subcomponent="Supervisor",
    tags=_tags("supervisor", "safety", "state"),
)
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

LOG_TRAINER_CLIENT_SHUTDOWN_WARNING = _constant(
    "LOG656",
    "WARNING",
    "Trainer client shutdown cleanup failed",
    component="Service",
    subcomponent="TrainerClient",
    tags=_tags("trainer", "client", "shutdown", "cleanup"),
)

LOG_TRAINER_WORKER_IMPORT_ERROR = _constant(
    "LOG657",
    "ERROR",
    "CleanRL worker module not importable; PYTHONPATH missing vendored packages (MuJoCo/telemetry workers blocked)",
    component="Service",
    subcomponent="TrainerDispatcher",
    tags=_tags("trainer", "worker", "pythonpath", "dependency"),
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

# Session (JSONL recorder / normalization) constants
LOG_SERVICE_SESSION_NUMPY_SCALAR_COERCE_FAILED = _constant(
    "LOG672",
    "DEBUG",
    "Session JSON normalization: numpy scalar coerce failed",
    component="Service",
    subcomponent="Session",
    tags=_tags("session", "json", "numpy", "coerce"),
)

LOG_SERVICE_SESSION_NDARRAY_SUMMARY_FAILED = _constant(
    "LOG673",
    "WARNING",
    "Session JSON normalization: ndarray summary failed; falling back",
    component="Service",
    subcomponent="Session",
    tags=_tags("session", "json", "ndarray", "summary"),
)

LOG_SERVICE_SESSION_LAZYFRAMES_SUMMARY_FAILED = _constant(
    "LOG674",
    "WARNING",
    "Session JSON normalization: LazyFrames summary failed; falling back",
    component="Service",
    subcomponent="Session",
    tags=_tags("session", "json", "lazyframes", "summary"),
)

LOG_SERVICE_SESSION_TOLIST_COERCE_FAILED = _constant(
    "LOG675",
    "DEBUG",
    "Session JSON normalization: tolist() coerce failed",
    component="Service",
    subcomponent="Session",
    tags=_tags("session", "json", "tolist", "coerce"),
)

LOG_SERVICE_SESSION_ITERABLE_COERCE_FAILED = _constant(
    "LOG676",
    "DEBUG",
    "Session JSON normalization: iterable coerce failed",
    component="Service",
    subcomponent="Session",
    tags=_tags("session", "json", "iterable", "coerce"),
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

LOG_SCHEMA_MISMATCH = _constant(
    "LOG668",
    "WARNING",
    "Telemetry payload failed schema validation",
    component="Service",
    subcomponent="Telemetry",
    tags=_tags("telemetry", "schema", "validation"),
)

LOG_VECTOR_AUTORESET_MODE = _constant(
    "LOG669",
    "WARNING",
    "Unsupported vector autoreset mode encountered",
    component="Service",
    subcomponent="Telemetry",
    tags=_tags("telemetry", "vector", "schema"),
)

LOG_SPACE_DESCRIPTOR_MISSING = _constant(
    "LOG670",
    "WARNING",
    "Space descriptor missing from telemetry payload",
    component="Service",
    subcomponent="Telemetry",
    tags=_tags("telemetry", "schema", "space"),
)

LOG_NORMALIZATION_STATS_DROPPED = _constant(
    "LOG671",
    "INFO",
    "Normalization statistics absent for schema-enabled payload",
    component="Service",
    subcomponent="Telemetry",
    tags=_tags("telemetry", "schema", "normalization"),
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

LOG_UI_MAINWINDOW_INVALID_CONFIG = _constant(
    "LOG705",
    "ERROR",
    "UI main window invalid training configuration",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "mainwindow", "config", "invalid"),
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

LOG_UI_RENDER_TABS_TENSORBOARD_STATUS = _constant(
    "LOG724",
    "INFO",
    "TensorBoard log directory status",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "tensorboard", "status"),
)

LOG_UI_RENDER_TABS_TENSORBOARD_WAITING = _constant(
    "LOG725",
    "DEBUG",
    "TensorBoard directory not yet available",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "tensorboard", "waiting"),
)

LOG_UI_RENDER_TABS_WANDB_STATUS = _constant(
    "LOG726",
    "INFO",
    "Weights & Biases run status",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "wandb", "status"),
)

LOG_UI_RENDER_TABS_WANDB_WARNING = _constant(
    "LOG727",
    "WARNING",
    "Weights & Biases run warning",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "wandb", "warning"),
)

LOG_UI_RENDER_TABS_WANDB_ERROR = _constant(
    "LOG728",
    "ERROR",
    "Weights & Biases run error",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "wandb", "error"),
)

LOG_UI_RENDER_TABS_WANDB_PROXY_APPLIED = _constant(
    "LOG729",
    "INFO",
    "Weights & Biases VPN proxy applied",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "wandb", "proxy", "vpn"),
)

LOG_UI_RENDER_TABS_WANDB_PROXY_SKIPPED = _constant(
    "LOG72A",
    "DEBUG",
    "Weights & Biases VPN proxy not applied",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "wandb", "proxy", "vpn"),
)

LOG_UI_RENDER_TABS_DELETE_REQUESTED = _constant(
    "LOG736",
    "INFO",
    "User requested run deletion from tab",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "tab_closure", "delete"),
)

LOG_UI_RENDER_TABS_EVENT_FOR_DELETED_RUN = _constant(
    "LOG737",
    "WARNING",
    "Live tab recreated for run already marked deleted",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "deleted", "diagnostics"),
)

LOG_UI_RENDER_TABS_TAB_ADDED = _constant(
    "LOG738",
    "INFO",
    "Render tabs added dynamic tab",
    component="UI",
    subcomponent="RenderTabs",
    tags=_tags("ui", "render", "tab"),
)

# ---------------------------------------------------------------------------
# Tab Closure Dialog constants (LOG724–LOG729)
# ---------------------------------------------------------------------------
LOG_UI_TAB_CLOSURE_DIALOG_OPENED = _constant(
    "LOG724",
    "INFO",
    "Tab closure dialog opened for run",
    component="UI",
    subcomponent="TabClosureDialog",
    tags=_tags("ui", "tab_closure", "dialog", "opened"),
)

LOG_UI_TAB_CLOSURE_CHOICE_SELECTED = _constant(
    "LOG725",
    "INFO",
    "Tab closure choice selected by user",
    component="UI",
    subcomponent="TabClosureDialog",
    tags=_tags("ui", "tab_closure", "choice"),
)

LOG_UI_TAB_CLOSURE_CHOICE_CANCELLED = _constant(
    "LOG726",
    "INFO",
    "Tab closure dialog cancelled",
    component="UI",
    subcomponent="TabClosureDialog",
    tags=_tags("ui", "tab_closure", "cancelled"),
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

LOG_UI_TRAIN_FORM_UI_PATH = _constant(
    "LOG734",
    "INFO",
    "Train form UI-only path configured",
    component="UI",
    subcomponent="TrainForm",
    tags=_tags("ui", "train_form", "ui_only_path"),
)

LOG_UI_TRAIN_FORM_TELEMETRY_PATH = _constant(
    "LOG735",
    "INFO",
    "Train form telemetry durable path configured",
    component="UI",
    subcomponent="TrainForm",
    tags=_tags("ui", "train_form", "telemetry_path"),
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

LOG_UI_PRESENTER_SIGNAL_CONNECTION_WARNING = _constant(
    "LOG744",
    "WARNING",
    "UI presenter signal connection failed",
    component="UI",
    subcomponent="Presenter",
    tags=_tags("ui", "presenter", "signal", "warning"),
)

LOG_UI_MAIN_WINDOW_SHUTDOWN_WARNING = _constant(
    "LOG745",
    "WARNING",
    "Main window shutdown cleanup failed",
    component="UI",
    subcomponent="MainWindow",
    tags=_tags("ui", "main_window", "shutdown", "cleanup"),
)

LOG_UI_TENSORBOARD_KILL_WARNING = _constant(
    "LOG746",
    "WARNING",
    "TensorBoard process kill failed",
    component="UI",
    subcomponent="TensorBoardTab",
    tags=_tags("ui", "tensorboard", "process", "kill", "warning"),
)

LOG_ADAPTER_RENDERING_WARNING = _constant(
    "LOG747",
    "WARNING",
    "Environment adapter rendering failed",
    component="Adapter",
    subcomponent="Rendering",
    tags=_tags("adapter", "rendering", "warning"),
)

LOG_TRAINER_LAUNCHER_LOG_FLUSH_WARNING = _constant(
    "LOG748",
    "WARNING",
    "Trainer launcher log file flush failed",
    component="Service",
    subcomponent="TrainerLauncher",
    tags=_tags("trainer", "launcher", "log", "flush", "warning"),
)

LOG_WORKER_CONFIG_READ_WARNING = _constant(
    "LOG749",
    "WARNING",
    "Worker config file read failed",
    component="Worker",
    subcomponent="Presenter",
    tags=_tags("worker", "presenter", "config", "read", "warning"),
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

# JSON serialization helpers
LOG_UTIL_JSON_NUMPY_SCALAR_COERCE_FAILED = _constant(
    "LOG804",
    "DEBUG",
    "Safe JSON: numpy scalar coerce failed; using fallback",
    component="Utility",
    subcomponent="Serialization",
    tags=_tags("utility", "json", "numpy", "coerce"),
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

LOG_WORKER_RUNTIME_JSON_SANITIZED = _constant(
    "LOG916",
    "INFO",
    "Worker sanitized telemetry payload for JSON compatibility",
    component="Worker",
    subcomponent="Runtime",
    tags=_tags("worker", "runtime", "telemetry", "json"),
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

LOG_WORKER_CONFIG_UI_PATH = _constant(
    "LOG907",
    "INFO",
    "Worker UI-only path settings applied",
    component="Worker",
    subcomponent="Config",
    tags=_tags("worker", "config", "ui_only_path"),
)

LOG_WORKER_CONFIG_DURABLE_PATH = _constant(
    "LOG908",
    "INFO",
    "Worker telemetry durable path settings applied",
    component="Worker",
    subcomponent="Config",
    tags=_tags("worker", "config", "telemetry_path"),
)

LOG_WORKER_POLICY_EVENT = _constant(
    "LOG909",
    "INFO",
    "Worker policy event",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy"),
)

LOG_WORKER_POLICY_WARNING = _constant(
    "LOG910",
    "WARNING",
    "Worker policy warning",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy", "warning"),
)

LOG_WORKER_POLICY_ERROR = _constant(
    "LOG911",
    "ERROR",
    "Worker policy error",
    component="Worker",
    subcomponent="Policy",
    tags=_tags("worker", "policy", "error"),
)

LOG_WORKER_BDI_EVENT = _constant(
    "LOG912",
    "INFO",
    "Worker BDI event",
    component="Worker",
    subcomponent="BDI",
    tags=_tags("worker", "bdi"),
)

LOG_WORKER_BDI_WARNING = _constant(
    "LOG913",
    "WARNING",
    "Worker BDI warning",
    component="Worker",
    subcomponent="BDI",
    tags=_tags("worker", "bdi", "warning"),
)

LOG_WORKER_BDI_ERROR = _constant(
    "LOG914",
    "ERROR",
    "Worker BDI error",
    component="Worker",
    subcomponent="BDI",
    tags=_tags("worker", "bdi", "error"),
)

LOG_WORKER_BDI_DEBUG = _constant(
    "LOG915",
    "DEBUG",
    "Worker BDI debug event",
    component="Worker",
    subcomponent="BDI",
    tags=_tags("worker", "bdi", "debug"),
)


# ---------------------------------------------------------------------------
# Episode Counter constants (LOG921–LOG930)
# ---------------------------------------------------------------------------
LOG_COUNTER_INITIALIZED = _constant(
    "LOG921",
    "INFO",
    "Episode counter initialized",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "initialization", "info"),
)

LOG_COUNTER_RESUME_SUCCESS = _constant(
    "LOG922",
    "INFO",
    "Episode counter resumed from database",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "resume", "persistence", "info"),
)

LOG_COUNTER_RESUME_FAILURE = _constant(
    "LOG923",
    "ERROR",
    "Failed to resume episode counter from database",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "resume", "persistence", "error"),
)

LOG_COUNTER_MAX_REACHED = _constant(
    "LOG924",
    "ERROR",
    "Maximum episodes per run limit reached",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "bounds", "limit", "error"),
)

LOG_COUNTER_INVALID_STATE = _constant(
    "LOG925",
    "ERROR",
    "Episode counter in invalid state",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "state", "error"),
)

LOG_COUNTER_CONCURRENCY_ERROR = _constant(
    "LOG926",
    "ERROR",
    "Episode counter concurrency error",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "concurrency", "threading", "error"),
)

LOG_COUNTER_PERSISTENCE_ERROR = _constant(
    "LOG927",
    "ERROR",
    "Episode counter persistence error",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "persistence", "database", "error"),
)

LOG_COUNTER_VALIDATION_ERROR = _constant(
    "LOG928",
    "ERROR",
    "Episode counter validation error",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "validation", "bounds", "error"),
)

LOG_COUNTER_NEXT_EPISODE = _constant(
    "LOG929",
    "DEBUG",
    "Episode counter allocated next episode index",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "allocation", "debug"),
)

LOG_COUNTER_RESET = _constant(
    "LOG930",
    "INFO",
    "Episode counter reset for new run",
    component="Core",
    subcomponent="EpisodeCounter",
    tags=_tags("counter", "reset", "info"),
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
    LOG_LIVE_CONTROLLER_EVENT_FOR_DELETED_RUN,
    LOG_LIVE_CONTROLLER_TAB_REQUESTED,
    LOG_ADAPTER_INIT_ERROR,
    LOG_ADAPTER_STEP_ERROR,
    LOG_ADAPTER_PAYLOAD_ERROR,
    LOG_ADAPTER_STATE_INVALID,
    LOG_ADAPTER_RENDER_ERROR,
    LOG_ADAPTER_ENV_CREATED,
    LOG_ADAPTER_ENV_RESET,
    LOG_ADAPTER_STEP_SUMMARY,
    LOG_ADAPTER_ENV_CLOSED,
    LOG_ADAPTER_MAP_GENERATION,
    LOG_ADAPTER_HOLE_PLACEMENT,
    LOG_ADAPTER_GOAL_OVERRIDE,
    LOG_ADAPTER_RENDER_PAYLOAD,
    LOG_ENV_MINIGRID_BOOT,
    LOG_ENV_MINIGRID_STEP,
    LOG_ENV_MINIGRID_ERROR,
    LOG_ENV_MINIGRID_RENDER_WARNING,
    LOG_ADAPTER_ALE_NAMESPACE_IMPORT_FAILED,
    LOG_ADAPTER_ALE_METADATA_PROBE_FAILED,
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
    LOG_SERVICE_SUPERVISOR_EVENT,
    LOG_SERVICE_SUPERVISOR_WARNING,
    LOG_SERVICE_SUPERVISOR_ERROR,
    LOG_SERVICE_SUPERVISOR_CONTROL_APPLIED,
    LOG_SERVICE_SUPERVISOR_CONTROL_REJECTED,
    LOG_SERVICE_SUPERVISOR_ROLLBACK,
    LOG_SERVICE_SUPERVISOR_SAFETY_STATE,
    LOG_RENDER_REGULATOR_NOT_STARTED,
    LOG_RENDER_DROPPED_FRAME,
    LOG_DAEMON_START,
    LOG_TRAINER_CLIENT_CONNECTING,
    LOG_TRAINER_CLIENT_CONNECTED,
    LOG_TRAINER_CLIENT_CONNECTION_TIMEOUT,
    LOG_TRAINER_CLIENT_LOOP_NONFATAL,
    LOG_TRAINER_CLIENT_LOOP_ERROR,
    LOG_TRAINER_CLIENT_SHUTDOWN_WARNING,
    LOG_SERVICE_FRAME_INFO,
    LOG_SERVICE_FRAME_WARNING,
    LOG_SERVICE_FRAME_ERROR,
    LOG_SERVICE_FRAME_DEBUG,
    LOG_SERVICE_SESSION_NUMPY_SCALAR_COERCE_FAILED,
    LOG_SERVICE_SESSION_NDARRAY_SUMMARY_FAILED,
    LOG_SERVICE_SESSION_LAZYFRAMES_SUMMARY_FAILED,
    LOG_SERVICE_SESSION_TOLIST_COERCE_FAILED,
    LOG_SERVICE_SESSION_ITERABLE_COERCE_FAILED,
    LOG_SERVICE_VALIDATION_DEBUG,
    LOG_SERVICE_VALIDATION_WARNING,
    LOG_SERVICE_VALIDATION_ERROR,
    LOG_SCHEMA_MISMATCH,
    LOG_VECTOR_AUTORESET_MODE,
    LOG_SPACE_DESCRIPTOR_MISSING,
    LOG_NORMALIZATION_STATS_DROPPED,
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
    LOG_UI_RENDER_TABS_TENSORBOARD_STATUS,
    LOG_UI_RENDER_TABS_TENSORBOARD_WAITING,
    LOG_UI_RENDER_TABS_WANDB_STATUS,
    LOG_UI_RENDER_TABS_WANDB_WARNING,
    LOG_UI_RENDER_TABS_WANDB_ERROR,
    LOG_UI_RENDER_TABS_DELETE_REQUESTED,
    LOG_UI_RENDER_TABS_EVENT_FOR_DELETED_RUN,
    LOG_UI_RENDER_TABS_TAB_ADDED,
    LOG_UI_TAB_CLOSURE_DIALOG_OPENED,
    LOG_UI_TAB_CLOSURE_CHOICE_SELECTED,
    LOG_UI_TAB_CLOSURE_CHOICE_CANCELLED,
    LOG_UI_TRAIN_FORM_TRACE,
    LOG_UI_TRAIN_FORM_INFO,
    LOG_UI_TRAIN_FORM_WARNING,
    LOG_UI_TRAIN_FORM_ERROR,
    LOG_UI_TRAIN_FORM_UI_PATH,
    LOG_UI_TRAIN_FORM_TELEMETRY_PATH,
    LOG_UI_WORKER_TABS_TRACE,
    LOG_UI_WORKER_TABS_INFO,
    LOG_UI_WORKER_TABS_WARNING,
    LOG_UI_WORKER_TABS_ERROR,
    LOG_UI_PRESENTER_SIGNAL_CONNECTION_WARNING,
    LOG_UI_MAIN_WINDOW_SHUTDOWN_WARNING,
    LOG_UI_TENSORBOARD_KILL_WARNING,
    LOG_ADAPTER_RENDERING_WARNING,
    LOG_TRAINER_LAUNCHER_LOG_FLUSH_WARNING,
    LOG_WORKER_CONFIG_READ_WARNING,
    LOG_UTIL_QT_RESEED_SKIPPED,
    LOG_UTIL_QT_STATE_CAPTURE_FAILED,
    LOG_UTIL_SEED_CALLBACK_FAILED,
    LOG_UTIL_JSON_NUMPY_SCALAR_COERCE_FAILED,
    LOG_WORKER_RUNTIME_EVENT,
    LOG_WORKER_RUNTIME_WARNING,
    LOG_WORKER_RUNTIME_ERROR,
    LOG_WORKER_RUNTIME_DEBUG,
    LOG_WORKER_RUNTIME_JSON_SANITIZED,
    LOG_WORKER_CONFIG_EVENT,
    LOG_WORKER_CONFIG_WARNING,
    LOG_WORKER_CONFIG_UI_PATH,
    LOG_WORKER_CONFIG_DURABLE_PATH,
    LOG_WORKER_POLICY_EVENT,
    LOG_WORKER_POLICY_WARNING,
    LOG_WORKER_POLICY_ERROR,
    LOG_WORKER_BDI_EVENT,
    LOG_WORKER_BDI_WARNING,
    LOG_WORKER_BDI_ERROR,
    LOG_WORKER_BDI_DEBUG,
    LOG_COUNTER_INITIALIZED,
    LOG_COUNTER_RESUME_SUCCESS,
    LOG_COUNTER_RESUME_FAILURE,
    LOG_COUNTER_MAX_REACHED,
    LOG_COUNTER_INVALID_STATE,
    LOG_COUNTER_CONCURRENCY_ERROR,
    LOG_COUNTER_PERSISTENCE_ERROR,
    LOG_COUNTER_VALIDATION_ERROR,
    LOG_COUNTER_NEXT_EPISODE,
    LOG_COUNTER_RESET,
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
    "LOG_LIVE_CONTROLLER_EVENT_FOR_DELETED_RUN",
    "LOG_LIVE_CONTROLLER_TAB_REQUESTED",
    "LOG_ADAPTER_INIT_ERROR",
    "LOG_ADAPTER_STEP_ERROR",
    "LOG_ADAPTER_PAYLOAD_ERROR",
    "LOG_ADAPTER_STATE_INVALID",
    "LOG_ADAPTER_RENDER_ERROR",
    "LOG_ADAPTER_ENV_CREATED",
    "LOG_ADAPTER_ENV_RESET",
    "LOG_ADAPTER_STEP_SUMMARY",
    "LOG_ADAPTER_ENV_CLOSED",
    "LOG_ADAPTER_MAP_GENERATION",
    "LOG_ADAPTER_HOLE_PLACEMENT",
    "LOG_ADAPTER_GOAL_OVERRIDE",
    "LOG_ADAPTER_RENDER_PAYLOAD",
    "LOG_ENV_MINIGRID_BOOT",
    "LOG_ENV_MINIGRID_STEP",
    "LOG_ENV_MINIGRID_ERROR",
    "LOG_ENV_MINIGRID_RENDER_WARNING",
    "LOG_ADAPTER_ALE_NAMESPACE_IMPORT_FAILED",
    "LOG_ADAPTER_ALE_METADATA_PROBE_FAILED",
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
    "LOG_SERVICE_SUPERVISOR_EVENT",
    "LOG_SERVICE_SUPERVISOR_WARNING",
    "LOG_SERVICE_SUPERVISOR_ERROR",
    "LOG_SERVICE_SUPERVISOR_CONTROL_APPLIED",
    "LOG_SERVICE_SUPERVISOR_CONTROL_REJECTED",
    "LOG_SERVICE_SUPERVISOR_ROLLBACK",
    "LOG_SERVICE_SUPERVISOR_SAFETY_STATE",
    "LOG_RENDER_REGULATOR_NOT_STARTED",
    "LOG_RENDER_DROPPED_FRAME",
    "LOG_DAEMON_START",
    "LOG_TRAINER_CLIENT_CONNECTING",
    "LOG_TRAINER_CLIENT_CONNECTED",
    "LOG_TRAINER_CLIENT_CONNECTION_TIMEOUT",
    "LOG_TRAINER_CLIENT_LOOP_NONFATAL",
    "LOG_TRAINER_CLIENT_LOOP_ERROR",
    "LOG_TRAINER_CLIENT_SHUTDOWN_WARNING",
    "LOG_SERVICE_FRAME_INFO",
    "LOG_SERVICE_FRAME_WARNING",
    "LOG_SERVICE_FRAME_ERROR",
    "LOG_SERVICE_FRAME_DEBUG",
    "LOG_SERVICE_SESSION_NUMPY_SCALAR_COERCE_FAILED",
    "LOG_SERVICE_SESSION_NDARRAY_SUMMARY_FAILED",
    "LOG_SERVICE_SESSION_LAZYFRAMES_SUMMARY_FAILED",
    "LOG_SERVICE_SESSION_TOLIST_COERCE_FAILED",
    "LOG_SERVICE_SESSION_ITERABLE_COERCE_FAILED",
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
    "LOG_UI_MAINWINDOW_INVALID_CONFIG",
    "LOG_UI_LIVE_TAB_TRACE",
    "LOG_UI_LIVE_TAB_INFO",
    "LOG_UI_LIVE_TAB_WARNING",
    "LOG_UI_LIVE_TAB_ERROR",
    "LOG_UI_RENDER_TABS_TRACE",
    "LOG_UI_RENDER_TABS_INFO",
    "LOG_UI_RENDER_TABS_WARNING",
    "LOG_UI_RENDER_TABS_ERROR",
    "LOG_UI_RENDER_TABS_TENSORBOARD_STATUS",
    "LOG_UI_RENDER_TABS_TENSORBOARD_WAITING",
    "LOG_UI_RENDER_TABS_WANDB_STATUS",
    "LOG_UI_RENDER_TABS_WANDB_WARNING",
    "LOG_UI_RENDER_TABS_WANDB_ERROR",
    "LOG_UI_RENDER_TABS_WANDB_PROXY_APPLIED",
    "LOG_UI_RENDER_TABS_WANDB_PROXY_SKIPPED",
    "LOG_UI_RENDER_TABS_DELETE_REQUESTED",
    "LOG_UI_RENDER_TABS_EVENT_FOR_DELETED_RUN",
    "LOG_UI_RENDER_TABS_TAB_ADDED",
    "LOG_UI_TAB_CLOSURE_DIALOG_OPENED",
    "LOG_UI_TAB_CLOSURE_CHOICE_SELECTED",
    "LOG_UI_TAB_CLOSURE_CHOICE_CANCELLED",
    "LOG_UI_TRAIN_FORM_TRACE",
    "LOG_UI_TRAIN_FORM_INFO",
    "LOG_UI_TRAIN_FORM_WARNING",
    "LOG_UI_TRAIN_FORM_ERROR",
    "LOG_UI_TRAIN_FORM_UI_PATH",
    "LOG_UI_TRAIN_FORM_TELEMETRY_PATH",
    "LOG_UI_WORKER_TABS_TRACE",
    "LOG_UI_WORKER_TABS_INFO",
    "LOG_UI_WORKER_TABS_WARNING",
    "LOG_UI_WORKER_TABS_ERROR",
    "LOG_UI_PRESENTER_SIGNAL_CONNECTION_WARNING",
    "LOG_UI_MAIN_WINDOW_SHUTDOWN_WARNING",
    "LOG_UI_TENSORBOARD_KILL_WARNING",
    "LOG_ADAPTER_RENDERING_WARNING",
    "LOG_TRAINER_LAUNCHER_LOG_FLUSH_WARNING",
    "LOG_WORKER_CONFIG_READ_WARNING",
    "LOG_UTIL_QT_RESEED_SKIPPED",
    "LOG_UTIL_QT_STATE_CAPTURE_FAILED",
    "LOG_UTIL_SEED_CALLBACK_FAILED",
    "LOG_UTIL_JSON_NUMPY_SCALAR_COERCE_FAILED",
    "LOG_WORKER_RUNTIME_EVENT",
    "LOG_WORKER_RUNTIME_WARNING",
    "LOG_WORKER_RUNTIME_ERROR",
    "LOG_WORKER_RUNTIME_DEBUG",
    "LOG_WORKER_RUNTIME_JSON_SANITIZED",
    "LOG_WORKER_CONFIG_EVENT",
    "LOG_WORKER_CONFIG_WARNING",
    "LOG_WORKER_CONFIG_UI_PATH",
    "LOG_WORKER_CONFIG_DURABLE_PATH",
    "LOG_WORKER_POLICY_EVENT",
    "LOG_WORKER_POLICY_WARNING",
    "LOG_WORKER_POLICY_ERROR",
    "LOG_WORKER_BDI_EVENT",
    "LOG_WORKER_BDI_WARNING",
    "LOG_WORKER_BDI_ERROR",
    "LOG_WORKER_BDI_DEBUG",
    "LOG_COUNTER_INITIALIZED",
    "LOG_COUNTER_RESUME_SUCCESS",
    "LOG_COUNTER_RESUME_FAILURE",
    "LOG_COUNTER_MAX_REACHED",
    "LOG_COUNTER_INVALID_STATE",
    "LOG_COUNTER_CONCURRENCY_ERROR",
    "LOG_COUNTER_PERSISTENCE_ERROR",
    "LOG_COUNTER_VALIDATION_ERROR",
    "LOG_COUNTER_NEXT_EPISODE",
    "LOG_COUNTER_RESET",
)
